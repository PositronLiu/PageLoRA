"""Manual and FlashInfer append writers for dense paged KV cache tensors."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch

from model.kvcache.radix_types import KVBlockSpan

PerfAdd = Callable[[str, float], None]


def write_layer_manual(
    spans: list[KVBlockSpan],
    *,
    paged_kv_cache: torch.Tensor | None,
    layer_idx: int,
    k: torch.Tensor,
    v: torch.Tensor,
    touch_page: Callable[[KVBlockSpan, torch.device], None],
    perf_add: PerfAdd,
) -> None:
    """Write K/V by assigning tensor slices into the dense paged cache."""

    pos = 0
    for span in spans:
        touch_page(span, k.device)

        if paged_kv_cache is not None:
            piece_t0 = time.perf_counter()
            k_piece = k[0, :, pos : pos + span.length, :].transpose(0, 1)
            v_piece = v[0, :, pos : pos + span.length, :].transpose(0, 1)
            perf_add("store_piece_slice_s", time.perf_counter() - piece_t0)

            dense_t0 = time.perf_counter()
            paged_kv_cache[
                layer_idx,
                span.block_id,
                0,
                span.start : span.start + span.length,
                :,
                :,
            ] = k_piece
            paged_kv_cache[
                layer_idx,
                span.block_id,
                1,
                span.start : span.start + span.length,
                :,
                :,
            ] = v_piece
            perf_add("store_dense_write_s", time.perf_counter() - dense_t0)

        pos += span.length


def write_layer_flashinfer(
    spans: list[KVBlockSpan],
    *,
    paged_kv_cache: torch.Tensor | None,
    page_size: int,
    layer_idx: int,
    k: torch.Tensor,
    v: torch.Tensor,
    touch_page: Callable[[KVBlockSpan, torch.device], None],
    perf_add: PerfAdd,
) -> None:
    """Write K/V through FlashInfer's paged cache append kernel."""

    if paged_kv_cache is None:
        raise RuntimeError("flashinfer KV append requires dense paged cache")
    if k.size(0) != 1 or v.size(0) != 1:
        raise RuntimeError("flashinfer KV append currently supports batch size 1")
    validate_flashinfer_append_spans(spans, page_size=page_size)

    for span in spans:
        touch_page(span, k.device)

    try:
        from flashinfer.page import append_paged_kv_cache
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("flashinfer.page.append_paged_kv_cache is unavailable") from exc

    piece_t0 = time.perf_counter()
    append_key = k[0].transpose(0, 1).contiguous()
    append_value = v[0].transpose(0, 1).contiguous()
    device = append_key.device
    start_pos = spans[0].start
    total_len = append_key.size(0)
    batch_indices = torch.zeros(total_len, device=device, dtype=torch.int32)
    positions = torch.arange(start_pos, start_pos + total_len, device=device, dtype=torch.int32)
    kv_indices = torch.tensor([span.block_id for span in spans], device=device, dtype=torch.int32)
    kv_indptr = torch.tensor([0, len(spans)], device=device, dtype=torch.int32)
    last_page_len = spans[-1].start + spans[-1].length
    kv_last_page_len = torch.tensor([last_page_len], device=device, dtype=torch.int32)
    perf_add("store_piece_slice_s", time.perf_counter() - piece_t0)

    append_t0 = time.perf_counter()
    append_paged_kv_cache(
        append_key,
        append_value,
        batch_indices,
        positions,
        paged_kv_cache[layer_idx],
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        kv_layout="NHD",
    )
    perf_add("store_flashinfer_append_s", time.perf_counter() - append_t0)
    perf_add("store_flashinfer_append_calls", 1.0)


def validate_flashinfer_append_spans(spans: list[KVBlockSpan], *, page_size: int) -> None:
    if not spans:
        return
    if spans[0].start < 0 or spans[0].start >= page_size:
        raise ValueError("first FlashInfer append span has invalid start")
    for idx, span in enumerate(spans):
        if span.length <= 0:
            raise ValueError("FlashInfer append spans must be non-empty")
        if idx == 0:
            if span.start + span.length > page_size:
                raise ValueError("first FlashInfer append span exceeds page size")
        elif span.start != 0:
            raise ValueError("non-first FlashInfer append spans must start at page offset 0")
        if idx < len(spans) - 1 and span.start + span.length != page_size:
            raise ValueError("non-last FlashInfer append spans must fill their pages")
