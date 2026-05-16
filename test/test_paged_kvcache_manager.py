"""Unit tests for FlashInfer-style paged KV cache manager."""

from __future__ import annotations

import pytest
import torch

from model.paged_kvcache_manager import (
    FlashInferPagedKVCacheManager,
    build_flashinfer_page_table,
)
from model.radix_cache import KVBlockSpan


def _kv(length: int, value: float) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.full((1, 2, length, 4), value)
    v = torch.full((1, 2, length, 4), value + 100)
    return k, v


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
    hk, hv = mgr.get_full(0, request=hit)
    assert hk is not None and hv is not None
    assert hk.shape[2] == 2
    assert torch.equal(hk[:, :, :2, :], k[:, :, :2, :])
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


def test_paged_kv_full_prompt_hit_replays_last_token():
    mgr = FlashInferPagedKVCacheManager(num_layers=1, page_size=2)
    req = mgr.begin_request([7, 8, 9], "general")
    k, v = _kv(3, 3)
    mgr.append_full(0, k, v, request=req)
    mgr.commit_request(req, [7, 8, 9])

    hit = mgr.begin_request([7, 8, 9], "general")
    assert hit.match_len == 3
    assert hit.prefill_from == 2
    hk, _ = mgr.get_full(0, request=hit)
    assert hk is not None
    assert hk.shape[2] == 2
    mgr.abort_request(hit)


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
