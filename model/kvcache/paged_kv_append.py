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

    if k.size(0) != 1 or v.size(0) != 1:
        raise RuntimeError("manual paged KV append currently supports batch size 1")
    piece_t0 = time.perf_counter()
    k_nhd = k[0].transpose(0, 1).contiguous()
    v_nhd = v[0].transpose(0, 1).contiguous()
    perf_add("store_piece_slice_s", time.perf_counter() - piece_t0)
    write_layer_manual_nhd(
        spans,
        paged_kv_cache=paged_kv_cache,
        page_size=_infer_page_size(paged_kv_cache, spans),
        layer_idx=layer_idx,
        k_nhd=k_nhd,
        v_nhd=v_nhd,
        touch_page=touch_page,
        perf_add=perf_add,
    )


def write_layer_manual_nhd(
    spans: list[KVBlockSpan],
    *,
    paged_kv_cache: torch.Tensor | None,
    page_size: int,
    layer_idx: int,
    k_nhd: torch.Tensor,
    v_nhd: torch.Tensor,
    touch_page: Callable[[KVBlockSpan, torch.device], None],
    perf_add: PerfAdd,
) -> None:
    """Write NHD K/V by assigning tensor slices into the dense paged cache."""

    _validate_nhd_inputs(spans, k_nhd, v_nhd)
    for span in spans:
        touch_page(span, k_nhd.device)

    if paged_kv_cache is None:
        return

    pos = 0
    idx = 0
    while idx < len(spans):
        run_len = _full_consecutive_page_run(spans, idx, page_size)
        if run_len > 1:
            token_len = run_len * page_size
            start_block = spans[idx].block_id
            dense_t0 = time.perf_counter()
            k_pages = k_nhd[pos : pos + token_len].reshape(
                run_len,
                page_size,
                k_nhd.size(1),
                k_nhd.size(2),
            )
            v_pages = v_nhd[pos : pos + token_len].reshape(
                run_len,
                page_size,
                v_nhd.size(1),
                v_nhd.size(2),
            )
            paged_kv_cache[layer_idx, start_block : start_block + run_len, 0, :, :, :] = k_pages
            paged_kv_cache[layer_idx, start_block : start_block + run_len, 1, :, :, :] = v_pages
            perf_add("store_dense_write_s", time.perf_counter() - dense_t0)
            perf_add("store_dense_write_batch_calls", 1.0)
            pos += token_len
            idx += run_len
            continue

        span = spans[idx]
        piece_t0 = time.perf_counter()
        k_piece = k_nhd[pos : pos + span.length]
        v_piece = v_nhd[pos : pos + span.length]
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
        idx += 1


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
    perf_add("store_piece_slice_s", time.perf_counter() - piece_t0)
    write_layer_flashinfer_nhd(
        spans,
        paged_kv_cache=paged_kv_cache,
        page_size=page_size,
        layer_idx=layer_idx,
        k_nhd=append_key,
        v_nhd=append_value,
        touch_page=lambda _span, _device: None,
        perf_add=perf_add,
    )


def write_layer_flashinfer_nhd(
    spans: list[KVBlockSpan],
    *,
    paged_kv_cache: torch.Tensor | None,
    page_size: int,
    layer_idx: int,
    k_nhd: torch.Tensor,
    v_nhd: torch.Tensor,
    touch_page: Callable[[KVBlockSpan, torch.device], None],
    perf_add: PerfAdd,
) -> None:
    """Write NHD K/V through FlashInfer's paged cache append kernel."""

    if paged_kv_cache is None:
        raise RuntimeError("flashinfer KV append requires dense paged cache")
    validate_flashinfer_append_spans(spans, page_size=page_size)
    _validate_nhd_inputs(spans, k_nhd, v_nhd)

    for span in spans:
        touch_page(span, k_nhd.device)

    try:
        from flashinfer.page import append_paged_kv_cache
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("flashinfer.page.append_paged_kv_cache is unavailable") from exc

    piece_t0 = time.perf_counter()
    append_key = k_nhd.contiguous()
    append_value = v_nhd.contiguous()
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


def _validate_nhd_inputs(spans: list[KVBlockSpan], k_nhd: torch.Tensor, v_nhd: torch.Tensor) -> None:
    if k_nhd.ndim != 3 or v_nhd.ndim != 3:
        raise ValueError("NHD K/V tensors must have shape [tokens, kv_heads, head_dim]")
    if k_nhd.shape != v_nhd.shape:
        raise ValueError("NHD K and V tensors must have the same shape")
    total_len = sum(span.length for span in spans)
    if total_len != k_nhd.size(0):
        raise ValueError("reserved span length does not match NHD KV chunk length")


def _full_consecutive_page_run(spans: list[KVBlockSpan], start_idx: int, page_size: int) -> int:
    if page_size <= 0:
        return 0
    first = spans[start_idx]
    if first.start != 0 or first.length != page_size:
        return 0
    run_len = 1
    expected_block_id = first.block_id + 1
    for span in spans[start_idx + 1 :]:
        if span.block_id != expected_block_id or span.start != 0 or span.length != page_size:
            break
        run_len += 1
        expected_block_id += 1
    return run_len


def _infer_page_size(paged_kv_cache: torch.Tensor | None, spans: list[KVBlockSpan]) -> int:
    if paged_kv_cache is not None:
        return int(paged_kv_cache.size(3))
    return max((span.start + span.length for span in spans), default=0)
