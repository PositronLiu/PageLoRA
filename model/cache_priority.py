"""Shared cache-priority scoring for prefill and offload decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass


DEFAULT_ACCESS_DECAY_HALF_LIFE = 512.0
MAX_PREFIX_VALUE_TOKENS = 8192

_KIND_RESIDENCY_WEIGHT = {
    "adapter": 0.75,
    "kvcache_adapter": 1.0,
}


@dataclass(frozen=True)
class CachePrioritySignals:
    """Signals used to compute one cache residency priority.

    Higher priority means the item is more valuable to keep resident or to
    materialize through prefill.  Offload callers evict the lowest priority
    resident items first.
    """

    touch_count: int = 0
    last_access_count: int = 0
    current_access_count: int = 0
    target_len: int = 0
    num_bytes: int = 0
    kind: str = "kvcache_adapter"


def cache_priority(
    signals: CachePrioritySignals,
    *,
    access_decay_half_life: float = DEFAULT_ACCESS_DECAY_HALF_LIFE,
) -> float:
    """Return the shared cache priority for prefill and retention."""

    recency = _recency_signal(
        last_access_count=signals.last_access_count,
        current_access_count=signals.current_access_count,
        half_life=access_decay_half_life,
    )
    touches = max(0, int(signals.touch_count))
    if signals.current_access_count > 0 and signals.last_access_count > 0:
        touch_value = touches * recency
    else:
        touch_value = float(touches)

    activity_value = 1.0 + recency + touch_value
    prefix_value = 1.0 + min(max(0, int(signals.target_len)), MAX_PREFIX_VALUE_TOKENS) / 1024.0
    kind_value = _KIND_RESIDENCY_WEIGHT.get(signals.kind, 1.0)
    memory_cost = max(1.0, float(max(0, int(signals.num_bytes))))
    return activity_value * prefix_value * kind_value / memory_cost


def prefill_overlap_priority(
    *,
    hot_priority: float,
    common_len: int,
    target_len: int,
    current_match_len: int,
    missing_tokens: int,
) -> float:
    """Return a prefill priority from shared hot-prefix priority and overlap."""

    target_len = max(0, int(target_len))
    current_match_len = min(max(0, int(current_match_len)), target_len)
    common_len = min(max(0, int(common_len)), target_len)
    if target_len <= 0 or common_len <= current_match_len:
        return 0.0

    overlap_coverage = (common_len - current_match_len) / max(1, target_len - current_match_len)
    miss_cost = min(max(0, int(missing_tokens)), MAX_PREFIX_VALUE_TOKENS) / 2048.0
    return max(0.0, float(hot_priority)) * overlap_coverage + miss_cost


def _recency_signal(
    *,
    last_access_count: int,
    current_access_count: int,
    half_life: float,
) -> float:
    last_access_count = max(0, int(last_access_count))
    current_access_count = max(0, int(current_access_count))
    if current_access_count > 0:
        if last_access_count <= 0:
            return 0.0
        age = max(0, current_access_count - last_access_count)
        half_life = max(1.0, float(half_life))
        return 0.5 ** (age / half_life)
    if last_access_count <= 0:
        return 0.0
    return 1.0 + math.log1p(last_access_count)
