"""Shared radix cache data types and cache-key helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

BR_BASE_KEY = "br:base"


def BR_RESIDUAL_KEY(adapter: str) -> str:
    return f"br:residual:{adapter}"


def FULL_KEY(adapter: str) -> str:
    return f"full:{adapter}"


EntryKey = tuple[str, int]


@dataclass(frozen=True)
class TokenSpan:
    """Slice into a token arena."""

    arena_id: int
    start: int
    length: int


@dataclass(frozen=True)
class KVBlockSpan:
    """Slice into a KV block."""

    block_id: int
    start: int
    length: int


@dataclass
class RadixNode:
    """One compressed radix-tree node.

    The node's ``edge`` is the token span from its parent to this node.
    KV spans also describe only that incoming edge segment.
    """

    node_id: int
    parent_id: int | None
    edge: TokenSpan
    depth: int
    children: dict[int, int] = field(default_factory=dict)
    spans_by_key: dict[str, list[KVBlockSpan]] = field(default_factory=dict)
    ref_count: int = 0
    last_access: int = 0
    pinned: bool = False


@dataclass
class CacheMatch:
    """Longest-prefix lookup result."""

    node_id: int
    matched_len: int
    edge_offset: int
    complete_node: bool
    key_match_len: dict[str, int]
    key_spans: dict[str, list[KVBlockSpan]]
    residual_key: str | None = None

    @property
    def token_matched_len(self) -> int:
        return self.matched_len

    @property
    def base_match_len(self) -> int:
        return self.key_match_len.get(BR_BASE_KEY, 0)

    @property
    def residual_match_len(self) -> int:
        if self.residual_key is None:
            return 0
        return self.key_match_len.get(self.residual_key, 0)

    @property
    def base_spans(self) -> list[KVBlockSpan]:
        return self.key_spans.get(BR_BASE_KEY, [])

    @property
    def residual_spans(self) -> list[KVBlockSpan]:
        if self.residual_key is None:
            return []
        return self.key_spans.get(self.residual_key, [])
