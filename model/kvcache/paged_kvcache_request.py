"""Request-local state for FlashInfer paged KV cache generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from model.kvcache.radix_types import KVBlockSpan


@dataclass
class FlashInferPagedKVRequest:
    """Per-generation paged full-KV state for one adapter and token prefix."""

    request_id: int
    token_ids: list[int]
    adapter_name: str
    match_len: int
    prefill_from: int
    materialized_from: int
    materialized_to: int
    prefix_spans: list[KVBlockSpan]
    new_spans: list[KVBlockSpan] = field(default_factory=list)
    current_kv_len: int = 0
    layer_chunk_counts: dict[int, int] = field(default_factory=dict)
    page_span_chunks: list[list[KVBlockSpan]] = field(default_factory=list)
