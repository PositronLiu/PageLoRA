"""Token arena and KV span helpers for radix cache metadata."""

from __future__ import annotations

from array import array

from model.kvcache.radix_types import KVBlockSpan, TokenSpan


class TokenArena:
    """Append-only compact token storage."""

    def __init__(self) -> None:
        self._arenas: list[array[int]] = [array("I")]

    def clear(self) -> None:
        self._arenas = [array("I")]

    def append(self, token_ids: list[int]) -> TokenSpan:
        if not token_ids:
            return TokenSpan(0, 0, 0)
        arena = self._arenas[0]
        start = len(arena)
        arena.extend(token_ids)
        return TokenSpan(0, start, len(token_ids))

    def subspan(self, span: TokenSpan, offset: int, length: int) -> TokenSpan:
        if offset < 0 or length < 0 or offset + length > span.length:
            raise ValueError("token subspan out of range")
        return TokenSpan(span.arena_id, span.start + offset, length)

    def token_at(self, span: TokenSpan, offset: int) -> int:
        if offset < 0 or offset >= span.length:
            raise IndexError("token span offset out of range")
        return self._arenas[span.arena_id][span.start + offset]


def span_total_length(spans: list[KVBlockSpan]) -> int:
    return sum(span.length for span in spans)


def slice_block_spans(
    spans: list[KVBlockSpan],
    start: int,
    length: int | None = None,
) -> list[KVBlockSpan]:
    """Return block spans covering ``[start, start + length)``."""

    if start < 0:
        raise ValueError("span slice start must be non-negative")
    total = span_total_length(spans)
    if length is None:
        length = total - start
    if length < 0 or start + length > total:
        raise ValueError("block span slice out of range")
    if length == 0:
        return []

    result: list[KVBlockSpan] = []
    remaining_skip = start
    remaining_len = length
    for span in spans:
        if remaining_skip >= span.length:
            remaining_skip -= span.length
            continue
        local_start = span.start + remaining_skip
        take = min(span.length - remaining_skip, remaining_len)
        result.append(KVBlockSpan(span.block_id, local_start, take))
        remaining_len -= take
        remaining_skip = 0
        if remaining_len == 0:
            break
    return result


def split_block_spans(
    spans: list[KVBlockSpan],
    offset: int,
) -> tuple[list[KVBlockSpan], list[KVBlockSpan]]:
    return slice_block_spans(spans, 0, offset), slice_block_spans(spans, offset)
