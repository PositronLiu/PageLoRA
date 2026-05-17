"""FlashInfer page-table construction for single-request paged KV."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from model.kvcache.radix_types import KVBlockSpan


@dataclass(frozen=True)
class FlashInferPageTable:
    """Single-request page-table tensors for FlashInfer paged KV wrappers."""

    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor


def build_flashinfer_page_table(
    spans: list[KVBlockSpan],
    *,
    page_size: int,
    device: torch.device,
) -> FlashInferPageTable:
    """Build a single-row FlashInfer page table from aligned page spans.

    The current page-table builder intentionally rejects sub-page starts and
    non-full middle pages because FlashInfer wrappers require page-aligned
    metadata for efficient paged attention.
    """

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    coalesced_spans = _coalesce_block_spans(spans)
    indices: list[int] = []
    last_page_len = 0
    for idx, span in enumerate(coalesced_spans):
        # Only the final page may be partially filled; all previous pages must
        # represent full physical KV pages.
        if span.start != 0:
            raise ValueError("FlashInfer page spans must start at page offset 0")
        if span.length <= 0 or span.length > page_size:
            raise ValueError("FlashInfer page span length is out of range")
        if idx < len(coalesced_spans) - 1 and span.length != page_size:
            raise ValueError("non-last FlashInfer page spans must be full pages")
        indices.append(span.block_id)
        last_page_len = span.length

    return FlashInferPageTable(
        paged_kv_indptr=torch.tensor([0, len(indices)], device=device, dtype=torch.int32),
        paged_kv_indices=torch.tensor(indices, device=device, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([last_page_len], device=device, dtype=torch.int32),
    )


def _coalesce_block_spans(spans: list[KVBlockSpan]) -> list[KVBlockSpan]:
    """Merge adjacent slices from the same physical page."""

    merged: list[KVBlockSpan] = []
    for span in spans:
        if span.length <= 0:
            continue
        if merged:
            prev = merged[-1]
            if prev.block_id == span.block_id and prev.start + prev.length == span.start:
                merged[-1] = KVBlockSpan(prev.block_id, prev.start, prev.length + span.length)
                continue
        merged.append(span)
    return merged
