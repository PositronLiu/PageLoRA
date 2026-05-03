from __future__ import annotations

import re
import time

from multilora.state import MultiLoRAState, TypeStats

_NON_WORD_RE = re.compile(r"[^a-z0-9_]+")


def normalize_tag(tag: str) -> str:
    """Normalize one judge-produced tag into a stable type identifier."""

    normalized = tag.strip().casefold().replace("-", "_").replace(" ", "_")
    normalized = _NON_WORD_RE.sub("_", normalized)
    normalized = normalized.strip("_")
    return normalized or "general"


class TypeRegistry:
    """In-memory helper that mutates the persistent Level 3 type state.

    The registry only updates per-type statistics and status transitions.
    It does not train adapters or routers by itself.
    """

    def __init__(self, state: MultiLoRAState) -> None:
        self.state = state

    def known_tags(self) -> list[str]:
        """Return tags sorted by current sample volume."""

        ranked = sorted(
            self.state.types.values(),
            key=lambda item: (-item.sample_count, item.type_name),
        )
        return [item.type_name for item in ranked]

    def update_type(
        self,
        *,
        tag: str,
        reward: float,
        tag_confidence: float,
    ) -> TypeStats:
        """Merge one new judged example into the type registry."""

        type_name = normalize_tag(tag)
        stats = self.state.types.get(type_name)
        if stats is None:
            stats = TypeStats(type_name=type_name, adapter_name=type_name)
            self.state.types[type_name] = stats

        stats.sample_count += 1
        if reward > 0:
            stats.positive_count += 1
        elif reward < 0:
            stats.negative_count += 1
        else:
            stats.neutral_count += 1

        count = float(stats.sample_count)
        stats.avg_reward += (reward - stats.avg_reward) / count
        stats.avg_tag_confidence += (tag_confidence - stats.avg_tag_confidence) / count
        stats.last_updated_at = time.time()

        thresholds = self.state.thresholds
        if (
            stats.status == "collecting"
            and stats.sample_count >= thresholds.min_samples_to_activate_type
            and stats.positive_count >= thresholds.min_positive_samples_to_activate_type
            and stats.avg_tag_confidence >= thresholds.min_tag_confidence
        ):
            stats.status = "active"
        return stats

    def top_prompt_tags(self, limit: int | None = None) -> list[str]:
        """Return the current tag shortlist shown to judge prompts."""

        cap = limit or self.state.thresholds.max_prompt_tags
        return self.known_tags()[:cap]

    def active_type_names(self) -> list[str]:
        """Return currently active tags in a stable order."""

        active = [
            stats
            for stats in self.state.types.values()
            if stats.status == "active"
        ]
        active.sort(key=lambda item: (-item.sample_count, item.type_name))
        return [item.type_name for item in active]

    def labels_for_router(self) -> list[str]:
        """Return the label space exposed to the classifier router."""

        return self.active_type_names()
