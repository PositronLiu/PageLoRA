"""Compressed radix metadata for token-prefix KV cache reuse.

This module intentionally stores only prefix metadata and block references.
Large KV tensors live in ``KVCacheManager``'s block store.
"""

from __future__ import annotations

from dataclasses import dataclass

from model.kvcache.radix_priority import RadixPriorityEntry, RadixPriorityIndex
from model.kvcache.radix_storage import TokenArena, slice_block_spans, span_total_length, split_block_spans
from model.kvcache.radix_types import (
    BR_BASE_KEY,
    BR_RESIDUAL_KEY,
    FULL_KEY,
    CacheMatch,
    EntryKey,
    KVBlockSpan,
    RadixNode,
    TokenSpan,
)


@dataclass
class RadixHotPrefix:
    """One hot radix prefix reconstructed from cache-access metadata."""

    cache_key: str
    radix_node_id: int
    token_ids: tuple[int, ...]
    target_len: int
    score: float
    touch_count: int
    last_access_count: int


@dataclass
class _HotPrefixStats:
    cache_key: str
    radix_node_id: int
    target_len: int = 0
    touch_count: int = 0
    last_access_count: int = 0


class RadixPrefixCache:
    """Compressed radix tree mapping token prefixes to KV block spans."""

    def __init__(self) -> None:
        self._arena = TokenArena()
        root_edge = TokenSpan(0, 0, 0)
        self._nodes: dict[int, RadixNode] = {
            0: RadixNode(node_id=0, parent_id=None, edge=root_edge, depth=0)
        }
        self._next_node_id = 1
        self._clock = 0
        self.priority_index = RadixPriorityIndex()
        self._hot_prefix_stats: dict[EntryKey, _HotPrefixStats] = {}

    def clear(self) -> None:
        self.__init__()

    def node_count(self) -> int:
        return len(self._nodes)

    def remove_key(self, key: str) -> None:
        for node in self._nodes.values():
            node.spans_by_key.pop(key, None)
        self.priority_index.remove_cache_key(key)
        self._remove_hot_cache_key(key)

    def rename_key(self, old_key: str, new_key: str) -> None:
        """Rename one cache-key namespace across radix coverage and metadata."""

        if old_key == new_key:
            return
        for node in self._nodes.values():
            node.spans_by_key.pop(new_key, None)
        self.priority_index.remove_cache_key(new_key)
        self._remove_hot_cache_key(new_key)

        for node in self._nodes.values():
            spans = node.spans_by_key.pop(old_key, None)
            if spans:
                node.spans_by_key[new_key] = spans
        self.priority_index.rename_cache_key(old_key, new_key)
        self._rename_hot_cache_key(old_key, new_key)

    def remove_residual(self, adapter: str) -> None:
        self.remove_key(BR_RESIDUAL_KEY(adapter))

    def lookup(
        self,
        token_ids: list[int],
        adapter: str | None = None,
        keys: tuple[str, ...] | None = None,
    ) -> CacheMatch:
        """Return the longest token-prefix match and resident block spans."""

        if keys is None:
            if adapter is None:
                keys = (BR_BASE_KEY,)
                residual_key = None
            else:
                residual_key = BR_RESIDUAL_KEY(adapter)
                keys = (BR_BASE_KEY, residual_key)
        else:
            keys = tuple(dict.fromkeys(keys))
            residual_key = BR_RESIDUAL_KEY(adapter) if adapter is not None else None

        node = self._nodes[0]
        pos = 0
        key_match_len = {key: 0 for key in keys}
        key_spans = {key: [] for key in keys}
        key_contiguous = {key: True for key in keys}

        while pos < len(token_ids):
            child_id = node.children.get(token_ids[pos])
            if child_id is None:
                break
            child = self._nodes[child_id]
            common = self._common_prefix(child.edge, token_ids, pos)
            if common == 0:
                break
            edge_start = pos
            edge_end = pos + child.edge.length

            for key in keys:
                if not key_contiguous[key]:
                    continue
                spans = child.spans_by_key.get(key, [])
                coverage_len = span_total_length(spans)
                if coverage_len > child.edge.length:
                    raise ValueError("cache coverage exceeds radix edge length")
                covered = min(coverage_len, common)
                if covered:
                    key_spans[key].extend(slice_block_spans(spans, 0, covered))
                    key_match_len[key] += covered
                    if common == child.edge.length and covered == child.edge.length:
                        self._touch_hot_prefix(key, child.node_id, edge_end)
                if covered < common:
                    key_contiguous[key] = False

            pos += common
            if common < child.edge.length:
                self._clock += 1
                child.last_access = self._clock
                return CacheMatch(
                    node_id=child.node_id,
                    matched_len=pos,
                    edge_offset=common,
                    complete_node=False,
                    key_match_len=key_match_len,
                    key_spans=key_spans,
                    residual_key=residual_key,
                )
            node = child

        self._clock += 1
        node.last_access = self._clock
        return CacheMatch(
            node_id=node.node_id,
            matched_len=pos,
            edge_offset=node.edge.length,
            complete_node=True,
            key_match_len=key_match_len,
            key_spans=key_spans,
            residual_key=residual_key,
        )

    def ensure_token_path(self, token_ids: list[int]) -> int:
        node_id = self._ensure_path(token_ids)
        if token_ids:
            node_id = self._ensure_boundary(token_ids, len(token_ids))
        return node_id

    def prefix_tokens_for_node(self, node_id: int, length: int | None = None) -> tuple[int, ...]:
        """Return token ids from root to one node boundary."""

        node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(f"unknown radix node: {node_id}")
        edges: list[TokenSpan] = []
        while node.parent_id is not None:
            edges.append(node.edge)
            node = self._nodes[node.parent_id]
        tokens: list[int] = []
        for edge in reversed(edges):
            for offset in range(edge.length):
                tokens.append(self._arena.token_at(edge, offset))
        if length is not None:
            length = min(max(0, length), len(tokens))
            tokens = tokens[:length]
        return tuple(tokens)

    def tokens_for_node(self, node_id: int) -> tuple[int, ...]:
        """Compatibility alias for callers that want full node-boundary tokens."""

        return self.prefix_tokens_for_node(node_id)

    def top_hot_prefixes(self, cache_key: str, limit: int) -> list[RadixHotPrefix]:
        """Return the hottest complete radix prefixes for one cache key."""

        limit = max(0, int(limit))
        if limit <= 0:
            return []
        prefixes: list[RadixHotPrefix] = []
        for (key, node_id), stats in self._hot_prefix_stats.items():
            if key != cache_key or node_id not in self._nodes or stats.target_len <= 0:
                continue
            try:
                token_ids = self.prefix_tokens_for_node(node_id, stats.target_len)
            except KeyError:
                continue
            if not token_ids:
                continue
            recency_bonus = stats.last_access_count * 1e-6
            score = stats.touch_count * 10.0 + min(stats.target_len, 8192) / 1024.0 + recency_bonus
            prefixes.append(
                RadixHotPrefix(
                    cache_key=cache_key,
                    radix_node_id=node_id,
                    token_ids=token_ids,
                    target_len=stats.target_len,
                    score=score,
                    touch_count=stats.touch_count,
                    last_access_count=stats.last_access_count,
                )
            )
        prefixes.sort(key=lambda item: (item.score, item.target_len, item.radix_node_id), reverse=True)
        return prefixes[:limit]

    def insert(
        self,
        token_ids: list[int],
        *,
        key: str,
        start: int,
        spans: list[KVBlockSpan],
    ) -> None:
        self.insert_coverage(token_ids, key=key, start=start, spans=spans)

    def insert_coverage(
        self,
        token_ids: list[int],
        *,
        key: str,
        start: int,
        spans: list[KVBlockSpan],
    ) -> None:
        """Attach contiguous KV coverage to existing token-path edges.

        The token tree may split for real token-prefix structure, but this
        method deliberately does not split at ``start`` or at the inserted
        coverage end. Partial adapter coverage is represented by shorter spans
        on the existing edge.
        """

        span_len = span_total_length(spans)
        if not token_ids or span_len == 0:
            return
        if start < 0 or start > len(token_ids):
            raise ValueError("insert start out of range")
        if start + span_len > len(token_ids):
            raise ValueError("span length exceeds token path")

        match = self.lookup(token_ids, keys=(key,))
        if match.key_match_len.get(key, 0) < start:
            raise ValueError("non-contiguous cache insert for radix path")

        self.ensure_token_path(token_ids)

        node = self._nodes[0]
        pos = 0
        span_offset = 0
        end = start + span_len
        while pos < len(token_ids) and pos < end:
            child_id = node.children[token_ids[pos]]
            child = self._nodes[child_id]
            edge_start = pos
            edge_end = pos + child.edge.length

            overlap_start = max(edge_start, start)
            overlap_end = min(edge_end, end)
            if overlap_start < overlap_end:
                edge_offset = overlap_start - edge_start
                take_len = overlap_end - overlap_start
                edge_spans = child.spans_by_key.setdefault(key, [])
                existing_len = span_total_length(edge_spans)
                if edge_offset + take_len <= existing_len:
                    # Already fully covered; skip idempotently.
                    span_offset += take_len
                    pos = edge_end
                    node = child
                    continue
                if edge_offset != existing_len:
                    raise ValueError("non-contiguous cache insert for radix edge")
                edge_spans.extend(slice_block_spans(spans, span_offset, take_len))
                if span_total_length(edge_spans) > child.edge.length:
                    raise ValueError("cache coverage exceeds radix edge length")
                self.priority_index.on_coverage_updated(
                    cache_key=key,
                    node_id=child.node_id,
                    covered_len=edge_start + span_total_length(edge_spans),
                )
                span_offset += take_len

            pos = edge_end
            node = child

        if span_offset != span_len:
            raise ValueError("inserted span did not cover the requested range")

    def insert_base(
        self,
        token_ids: list[int],
        start: int,
        spans: list[KVBlockSpan],
    ) -> None:
        self.insert_coverage(token_ids, key=BR_BASE_KEY, start=start, spans=spans)

    def insert_residual(
        self,
        token_ids: list[int],
        adapter: str,
        start: int,
        spans: list[KVBlockSpan],
    ) -> None:
        self.insert_coverage(
            token_ids,
            key=BR_RESIDUAL_KEY(adapter),
            start=start,
            spans=spans,
        )

    def _ensure_path(self, token_ids: list[int]) -> int:
        node = self._nodes[0]
        pos = 0
        while pos < len(token_ids):
            token = token_ids[pos]
            child_id = node.children.get(token)
            if child_id is None:
                span = self._arena.append(token_ids[pos:])
                child = self._new_node(node.node_id, span, len(token_ids))
                node.children[token] = child.node_id
                return child.node_id

            child = self._nodes[child_id]
            common = self._common_prefix(child.edge, token_ids, pos)
            if common < child.edge.length:
                child = self._split_child(node, child, common)
            pos += common
            node = child
        return node.node_id

    def _ensure_boundary(self, token_ids: list[int], depth: int) -> int:
        if depth < 0 or depth > len(token_ids):
            raise ValueError("boundary depth out of range")
        if depth == 0:
            return 0

        node = self._nodes[0]
        pos = 0
        while pos < depth:
            child_id = node.children[token_ids[pos]]
            child = self._nodes[child_id]
            edge_end = pos + child.edge.length
            if depth < edge_end:
                child = self._split_child(node, child, depth - pos)
                return child.node_id
            pos = edge_end
            node = child
        return node.node_id

    def _new_node(self, parent_id: int, edge: TokenSpan, depth: int) -> RadixNode:
        node = RadixNode(
            node_id=self._next_node_id,
            parent_id=parent_id,
            edge=edge,
            depth=depth,
        )
        self._nodes[node.node_id] = node
        self._next_node_id += 1
        return node

    def _touch_hot_prefix(self, cache_key: str, node_id: int, target_len: int) -> None:
        key = (cache_key, node_id)
        stats = self._hot_prefix_stats.get(key)
        if stats is None:
            stats = _HotPrefixStats(cache_key=cache_key, radix_node_id=node_id)
            self._hot_prefix_stats[key] = stats
        stats.target_len = max(stats.target_len, int(target_len))
        stats.touch_count += 1
        self._clock += 1
        stats.last_access_count = self._clock

    def _remove_hot_cache_key(self, cache_key: str) -> None:
        for key in list(self._hot_prefix_stats):
            if key[0] == cache_key:
                self._hot_prefix_stats.pop(key, None)

    def _rename_hot_cache_key(self, old_key: str, new_key: str) -> None:
        if old_key == new_key:
            return
        for key in list(self._hot_prefix_stats):
            if key[0] != old_key:
                continue
            stats = self._hot_prefix_stats.pop(key)
            renamed = (new_key, key[1])
            stats.cache_key = new_key
            self._hot_prefix_stats[renamed] = stats

    def _split_child(
        self,
        parent: RadixNode,
        child: RadixNode,
        offset: int,
    ) -> RadixNode:
        if offset <= 0 or offset >= child.edge.length:
            raise ValueError("split offset must be inside the child edge")

        prefix_edge = self._arena.subspan(child.edge, 0, offset)
        suffix_edge = self._arena.subspan(child.edge, offset, child.edge.length - offset)

        internal = self._new_node(parent.node_id, prefix_edge, parent.depth + offset)
        new_child_spans: dict[str, list[KVBlockSpan]] = {}
        for key, spans in child.spans_by_key.items():
            coverage_len = span_total_length(spans)
            if coverage_len > child.edge.length:
                raise ValueError("cache coverage exceeds radix edge length")
            prefix_len = min(offset, coverage_len)
            suffix_len = max(0, coverage_len - offset)
            if prefix_len:
                internal.spans_by_key[key] = slice_block_spans(spans, 0, prefix_len)
            if suffix_len:
                new_child_spans[key] = slice_block_spans(spans, offset, suffix_len)
        child.spans_by_key = new_child_spans

        child.edge = suffix_edge
        child.parent_id = internal.node_id

        old_key = self._arena.token_at(prefix_edge, 0)
        suffix_key = self._arena.token_at(suffix_edge, 0)
        parent.children[old_key] = internal.node_id
        internal.children[suffix_key] = child.node_id
        return internal

    def _common_prefix(self, edge: TokenSpan, token_ids: list[int], pos: int) -> int:
        limit = min(edge.length, len(token_ids) - pos)
        common = 0
        while common < limit:
            if self._arena.token_at(edge, common) != token_ids[pos + common]:
                break
            common += 1
        return common
