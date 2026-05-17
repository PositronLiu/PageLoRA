"""Priority index for radix-cache background work."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from model.kvcache.radix_types import EntryKey


@dataclass
class RadixPriorityEntry:
    """Backend-neutral priority metadata for one cache key on one radix node."""

    key: EntryKey
    cache_key: str
    radix_node_id: int
    priority: float
    target_len: int = 0
    current_match_len: int = 0
    prefix_hash: str = ""
    adapter_name: str = ""
    token_ids: tuple[int, ...] = ()
    touch_count: int = 0
    last_access_count: int = 0
    last_touched_at: float = 0.0
    last_prefill_at: float | None = None
    last_offload_at: float | None = None
    inflight: bool = False
    protected: bool = False
    active: bool = True
    version: int = 0


class RadixPriorityIndex:
    """Indexed double-ended priority queue for radix-node cache work.

    The index is structural: callers compute scores and the scheduler decides
    when to use entries. This class only maintains node/key mappings and
    versioned min/max heaps.
    """

    def __init__(self) -> None:
        self.entries: dict[EntryKey, RadixPriorityEntry] = {}
        self.entries_by_node: dict[int, set[EntryKey]] = {}
        self._max_heap: list[tuple[float, int, EntryKey]] = []
        self._min_heap: list[tuple[float, int, EntryKey]] = []
        self._access_clock = 0

    def __len__(self) -> int:
        return len(self.entries)

    def clear(self) -> None:
        self.entries.clear()
        self.entries_by_node.clear()
        self._max_heap.clear()
        self._min_heap.clear()
        self._access_clock = 0

    def get(self, key: EntryKey) -> RadixPriorityEntry | None:
        return self.entries.get(key)

    def upsert(
        self,
        *,
        cache_key: str,
        radix_node_id: int,
        priority: float,
        target_len: int | None = None,
        current_match_len: int | None = None,
        prefix_hash: str | None = None,
        adapter_name: str | None = None,
        token_ids: list[int] | tuple[int, ...] | None = None,
        now: float | None = None,
        inflight: bool | None = None,
        protected: bool | None = None,
        active: bool | None = None,
        touch: bool = True,
    ) -> RadixPriorityEntry:
        key = (cache_key, radix_node_id)
        entry = self.entries.get(key)
        if entry is None:
            entry = RadixPriorityEntry(
                key=key,
                cache_key=cache_key,
                radix_node_id=radix_node_id,
                priority=priority,
            )
            self.entries[key] = entry
            self.entries_by_node.setdefault(radix_node_id, set()).add(key)
            # Push to heaps only on creation; updates happen in-place.
            entry.version += 1
            heapq.heappush(self._max_heap, (-entry.priority, entry.version, entry.key))
            heapq.heappush(self._min_heap, (entry.priority, entry.version, entry.key))

        entry.priority = priority
        if target_len is not None:
            entry.target_len = target_len
        if current_match_len is not None:
            entry.current_match_len = current_match_len
        if prefix_hash is not None:
            entry.prefix_hash = prefix_hash
        if adapter_name is not None:
            entry.adapter_name = adapter_name
        if token_ids is not None:
            entry.token_ids = tuple(token_ids)
        if inflight is not None:
            entry.inflight = inflight
        if protected is not None:
            entry.protected = protected
        if active is not None:
            entry.active = active
        if touch:
            self._touch_entry(entry, now=now)

        return entry

    def remove(self, key: EntryKey) -> None:
        entry = self.entries.pop(key, None)
        if entry is None:
            return
        keys = self.entries_by_node.get(entry.radix_node_id)
        if keys is not None:
            keys.discard(key)
            if not keys:
                self.entries_by_node.pop(entry.radix_node_id, None)

    def remove_cache_key(self, cache_key: str) -> None:
        for key in list(self.entries):
            if key[0] == cache_key:
                self.remove(key)

    def remove_node(self, radix_node_id: int) -> None:
        for key in list(self.entries_by_node.get(radix_node_id, ())):
            self.remove(key)

    def touch_node(self, radix_node_id: int, *, now: float | None = None) -> list[RadixPriorityEntry]:
        touched: list[RadixPriorityEntry] = []
        for key in list(self.entries_by_node.get(radix_node_id, ())):
            entry = self.entries.get(key)
            if entry is None:
                continue
            self._touch_entry(entry, now=now)
            touched.append(entry)
        return touched

    def on_coverage_updated(
        self,
        *,
        cache_key: str,
        node_id: int,
        covered_len: int,
    ) -> None:
        entry = self.entries.get((cache_key, node_id))
        if entry is None:
            return
        if covered_len > entry.current_match_len:
            entry.current_match_len = covered_len
        if entry.target_len > 0 and entry.current_match_len >= entry.target_len:
            entry.active = False

    def mark_inflight(self, key: EntryKey, inflight: bool) -> None:
        entry = self.entries.get(key)
        if entry is None:
            return
        entry.inflight = inflight

    def mark_protected(self, key: EntryKey, protected: bool) -> None:
        entry = self.entries.get(key)
        if entry is None:
            return
        entry.protected = protected

    def peek_max(
        self,
        *,
        skip_inflight: bool = True,
        skip_protected: bool = False,
    ) -> RadixPriorityEntry | None:
        return self._peek(self._max_heap, skip_inflight=skip_inflight, skip_protected=skip_protected)

    def peek_min(
        self,
        *,
        skip_inflight: bool = True,
        skip_protected: bool = True,
    ) -> RadixPriorityEntry | None:
        return self._peek(self._min_heap, skip_inflight=skip_inflight, skip_protected=skip_protected)

    def pop_max(
        self,
        *,
        skip_inflight: bool = True,
        skip_protected: bool = False,
    ) -> RadixPriorityEntry | None:
        entry = self.peek_max(skip_inflight=skip_inflight, skip_protected=skip_protected)
        if entry is not None:
            self.remove(entry.key)
        return entry

    def pop_min(
        self,
        *,
        skip_inflight: bool = True,
        skip_protected: bool = True,
    ) -> RadixPriorityEntry | None:
        entry = self.peek_min(skip_inflight=skip_inflight, skip_protected=skip_protected)
        if entry is not None:
            self.remove(entry.key)
        return entry

    def _touch_entry(self, entry: RadixPriorityEntry, *, now: float | None) -> None:
        self._access_clock += 1
        entry.touch_count += 1
        entry.last_access_count = self._access_clock
        if now is not None:
            entry.last_touched_at = now

    def _peek(
        self,
        heap: list[tuple[float, int, EntryKey]],
        *,
        skip_inflight: bool,
        skip_protected: bool,
    ) -> RadixPriorityEntry | None:
        while heap:
            _, version, key = heap[0]
            entry = self.entries.get(key)
            if (
                entry is None
                or entry.version != version
                or not entry.active
                or (skip_inflight and entry.inflight)
                or (skip_protected and entry.protected)
            ):
                heapq.heappop(heap)
                continue
            return entry
        return None
