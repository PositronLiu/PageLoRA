"""Unit tests for FlashInfer-style paged KV cache manager."""

from __future__ import annotations

import pytest
import torch

from model.kvcache.paged_kvcache_manager import (
    FlashInferPagedKVCacheManager,
    build_flashinfer_page_table,
)
from model.kvcache.flashinfer_page_table import FlashInferPageTableBuffer
from model.kvcache.radix_cache import KVBlockSpan


def _kv(length: int, value: float) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.full((1, 2, length, 4), value)
    v = torch.full((1, 2, length, 4), value + 100)
    return k, v


def _cache_k_tokens(cache: torch.Tensor, page_id: int, length: int) -> torch.Tensor:
    return cache[page_id, 0, :length, :, :].transpose(0, 1).unsqueeze(0)


def _cache_v_tokens(cache: torch.Tensor, page_id: int, length: int) -> torch.Tensor:
    return cache[page_id, 1, :length, :, :].transpose(0, 1).unsqueeze(0)


def test_paged_kv_cache_reuses_same_adapter_prefix():
    mgr = FlashInferPagedKVCacheManager(num_layers=1, page_size=2, max_pages=8)
    req = mgr.begin_request([1, 2, 3], "general")
    assert req.match_len == 0
    k, v = _kv(3, 1)
    mgr.append_full(0, k, v, request=req)
    mgr.commit_request(req, [1, 2, 3])

    assert mgr.allocated_pages == 2

    hit = mgr.begin_request([1, 2, 3, 4], "general")
    assert hit.match_len == 3
    assert hit.prefill_from == 2
    layer_cache = mgr.layer_cache(0)
    assert layer_cache is not None
    span = hit.prefix_spans[0]
    assert span.length == 2
    assert torch.equal(_cache_k_tokens(layer_cache, span.block_id, span.length), k[:, :, :2, :])
    assert torch.equal(_cache_v_tokens(layer_cache, span.block_id, span.length), v[:, :, :2, :])
    mgr.abort_request(hit)


def test_paged_kv_cache_is_adapter_specific():
    mgr = FlashInferPagedKVCacheManager(num_layers=1, page_size=2)
    req = mgr.begin_request([1, 2, 3], "general")
    k, v = _kv(3, 2)
    mgr.append_full(0, k, v, request=req)
    mgr.commit_request(req, [1, 2, 3])

    other = mgr.begin_request([1, 2, 3], "coding")
    assert other.match_len == 0
    mgr.abort_request(other)


def test_paged_kv_dense_cache_grows_when_max_pages_unbounded():
    mgr = FlashInferPagedKVCacheManager(num_layers=1, page_size=2, max_pages=0)
    req = mgr.begin_request([1, 2, 3], "general")
    k, v = _kv(3, 6)
    mgr.append_full(0, k, v, request=req)

    assert mgr.paged_kv_cache is not None
    assert mgr.paged_kv_cache.shape[1] >= 2
    mgr.abort_request(req)


def test_paged_kv_dense_cache_preserves_token_layout():
    mgr = FlashInferPagedKVCacheManager(num_layers=1, page_size=2, max_pages=4)
    req = mgr.begin_request([1, 2, 3], "general")
    k = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).view(1, 2, 3, 4)
    v = k + 1000
    mgr.append_full(0, k, v, request=req)

    layer_cache = mgr.layer_cache(0)
    assert layer_cache is not None
    assert torch.equal(_cache_k_tokens(layer_cache, 0, 2), k[:, :, :2, :])
    assert torch.equal(_cache_v_tokens(layer_cache, 0, 2), v[:, :, :2, :])
    assert torch.equal(_cache_k_tokens(layer_cache, 1, 1), k[:, :, 2:, :])
    assert torch.equal(_cache_v_tokens(layer_cache, 1, 1), v[:, :, 2:, :])
    mgr.abort_request(req)


def test_paged_kv_append_full_nhd_preserves_token_layout():
    mgr = FlashInferPagedKVCacheManager(num_layers=1, page_size=2, max_pages=4)
    req = mgr.begin_request([1, 2, 3], "general")
    k_bhsd = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).view(1, 2, 3, 4)
    v_bhsd = k_bhsd + 1000
    mgr.append_full_nhd(
        0,
        k_bhsd[0].transpose(0, 1).contiguous(),
        v_bhsd[0].transpose(0, 1).contiguous(),
        request=req,
    )

    layer_cache = mgr.layer_cache(0)
    assert layer_cache is not None
    assert torch.equal(_cache_k_tokens(layer_cache, 0, 2), k_bhsd[:, :, :2, :])
    assert torch.equal(_cache_v_tokens(layer_cache, 0, 2), v_bhsd[:, :, :2, :])
    assert torch.equal(_cache_k_tokens(layer_cache, 1, 1), k_bhsd[:, :, 2:, :])
    assert torch.equal(_cache_v_tokens(layer_cache, 1, 1), v_bhsd[:, :, 2:, :])
    snapshot = mgr.perf_snapshot()
    assert snapshot["store_write_layer_nhd_calls"] == 1
    mgr.abort_request(req)


def test_paged_kv_append_full_nhd_batches_consecutive_pages():
    mgr = FlashInferPagedKVCacheManager(num_layers=1, page_size=2, max_pages=4)
    req = mgr.begin_request([1, 2, 3, 4], "general")
    k_bhsd = torch.arange(1 * 2 * 4 * 4, dtype=torch.float32).view(1, 2, 4, 4)
    v_bhsd = k_bhsd + 1000
    mgr.append_full_nhd(
        0,
        k_bhsd[0].transpose(0, 1).contiguous(),
        v_bhsd[0].transpose(0, 1).contiguous(),
        request=req,
    )

    layer_cache = mgr.layer_cache(0)
    assert layer_cache is not None
    assert torch.equal(_cache_k_tokens(layer_cache, 0, 2), k_bhsd[:, :, :2, :])
    assert torch.equal(_cache_k_tokens(layer_cache, 1, 2), k_bhsd[:, :, 2:, :])
    assert mgr.perf_snapshot()["store_dense_write_batch_calls"] == 1
    mgr.abort_request(req)


def test_paged_kv_flashinfer_append_matches_manual_layout():
    if not torch.cuda.is_available():
        pytest.skip("flashinfer append test requires CUDA")

    try:
        import flashinfer.page  # noqa: F401
    except Exception:
        pytest.skip("flashinfer page append is unavailable")

    device = torch.device("cuda")
    mgr = FlashInferPagedKVCacheManager(
        num_layers=1,
        page_size=2,
        max_pages=4,
        kv_append_mode="flashinfer",
        device=device,
        dtype=torch.float16,
    )
    req = mgr.begin_request([1, 2, 3], "general")
    k = torch.arange(1 * 2 * 3 * 128, device=device, dtype=torch.float16).view(1, 2, 3, 128)
    v = k + 1000
    try:
        mgr.append_full(0, k, v, request=req)
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        message = str(exc)
        if "Failed to load dynamic shared library" in message or "GLIBCXX_" in message:
            pytest.skip(f"flashinfer page append is unavailable: {message}")
        raise

    layer_cache = mgr.layer_cache(0)
    assert layer_cache is not None
    assert torch.equal(_cache_k_tokens(layer_cache, 0, 2), k[:, :, :2, :])
    assert torch.equal(_cache_v_tokens(layer_cache, 0, 2), v[:, :, :2, :])
    assert torch.equal(_cache_k_tokens(layer_cache, 1, 1), k[:, :, 2:, :])
    assert torch.equal(_cache_v_tokens(layer_cache, 1, 1), v[:, :, 2:, :])
    mgr.abort_request(req)


def test_paged_kv_full_prompt_hit_replays_last_token():
    mgr = FlashInferPagedKVCacheManager(num_layers=1, page_size=2)
    req = mgr.begin_request([7, 8, 9], "general")
    k, v = _kv(3, 3)
    mgr.append_full(0, k, v, request=req)
    mgr.commit_request(req, [7, 8, 9])

    hit = mgr.begin_request([7, 8, 9], "general")
    assert hit.match_len == 3
    assert hit.prefill_from == 2
    layer_cache = mgr.layer_cache(0)
    assert layer_cache is not None
    assert hit.prefix_spans[0].length == 2
    mgr.abort_request(hit)


def test_paged_kv_get_full_is_disabled():
    mgr = FlashInferPagedKVCacheManager(num_layers=1, page_size=2)
    with pytest.raises(RuntimeError, match="no longer materializes"):
        mgr.get_full(0)


def test_paged_kv_invalidate_adapter_frees_pages():
    mgr = FlashInferPagedKVCacheManager(num_layers=1, page_size=2)
    req = mgr.begin_request([1, 2, 3], "general")
    k, v = _kv(3, 4)
    mgr.append_full(0, k, v, request=req)
    mgr.commit_request(req, [1, 2, 3])
    assert mgr.allocated_pages == 2

    mgr.invalidate_adapter("general")
    assert mgr.allocated_pages == 0
    miss = mgr.begin_request([1, 2, 3], "general")
    assert miss.match_len == 0
    mgr.abort_request(miss)


def test_paged_kv_replace_adapter_cache_moves_shadow_pages():
    mgr = FlashInferPagedKVCacheManager(num_layers=1, page_size=2, max_pages=8)
    shadow = "__publish_shadow__:general:1"
    token_ids = [4, 5, 6]

    req = mgr.begin_request(token_ids, shadow)
    k, v = _kv(3, 7)
    mgr.append_full(0, k, v, request=req)
    mgr.commit_request(req, token_ids)

    assert mgr.allocated_pages == 2
    assert mgr.lookup_match_len(token_ids, shadow) == 3
    assert mgr.top_hot_prefixes(shadow, limit=1)[0].token_ids == tuple(token_ids)

    mgr.replace_adapter_cache(shadow, "general")

    assert mgr.allocated_pages == 2
    assert mgr.lookup_match_len(token_ids, shadow) == 0
    assert mgr.lookup_match_len(token_ids, "general") == 3
    assert mgr.top_hot_prefixes(shadow, limit=1) == []
    assert mgr.top_hot_prefixes("general", limit=1)[0].token_ids == tuple(token_ids)

    mgr.invalidate_adapter(shadow)
    assert mgr.allocated_pages == 2
    mgr.invalidate_adapter("general")
    assert mgr.allocated_pages == 0


def test_paged_kv_oom_rolls_back_partial_pages():
    mgr = FlashInferPagedKVCacheManager(num_layers=1, page_size=2, max_pages=1)
    req = mgr.begin_request([1, 2, 3], "general")
    k, v = _kv(3, 5)

    with pytest.raises(RuntimeError):
        mgr.append_full(0, k, v, request=req)

    assert mgr.allocated_pages == 0


def test_flashinfer_page_table_accepts_aligned_spans():
    table = build_flashinfer_page_table(
        [
            KVBlockSpan(block_id=11, start=0, length=2),
            KVBlockSpan(block_id=12, start=0, length=1),
        ],
        page_size=2,
        device=torch.device("cpu"),
    )

    assert table.paged_kv_indptr.tolist() == [0, 2]
    assert table.paged_kv_indices.tolist() == [11, 12]
    assert table.paged_kv_last_page_len.tolist() == [1]


def test_flashinfer_page_table_rejects_unaligned_spans():
    with pytest.raises(ValueError):
        build_flashinfer_page_table(
            [KVBlockSpan(block_id=1, start=1, length=1)],
            page_size=2,
            device=torch.device("cpu"),
        )

    with pytest.raises(ValueError):
        build_flashinfer_page_table(
            [
                KVBlockSpan(block_id=1, start=0, length=1),
                KVBlockSpan(block_id=2, start=0, length=1),
            ],
            page_size=2,
            device=torch.device("cpu"),
        )


def test_flashinfer_page_table_buffer_reuses_tensors():
    buffer = FlashInferPageTableBuffer(page_size=2, device=torch.device("cpu"), initial_capacity=1)
    first = buffer.update_from_spans([KVBlockSpan(block_id=11, start=0, length=1)])
    first_indptr = first.paged_kv_indptr
    first_last = first.paged_kv_last_page_len

    second = buffer.update_from_spans(
        [
            KVBlockSpan(block_id=11, start=0, length=2),
            KVBlockSpan(block_id=12, start=0, length=1),
        ]
    )

    assert second.paged_kv_indptr is first_indptr
    assert second.paged_kv_last_page_len is first_last
    assert second.paged_kv_indptr.tolist() == [0, 2]
    assert second.paged_kv_indices.tolist() == [11, 12]
    assert second.paged_kv_last_page_len.tolist() == [1]
