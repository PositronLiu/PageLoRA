from __future__ import annotations

from types import SimpleNamespace

import torch

from model.flashinfer.cuda_graph import (
    DecodeGraphEntry,
    DecodeGraphKey,
    FlashInferBatchDecodeCudaGraphRunner,
    FlashInferDecodeCudaGraphRunner,
    can_use_flashinfer_cuda_graph_batch_decode,
    can_use_flashinfer_cuda_graph_decode,
    flashinfer_cuda_graph_decode_enabled,
)
from model.flashinfer.layer_runner import _append_decode_kv_static
from model.kvcache.paged_kvcache_manager import FlashInferPagedKVCacheManager


def test_cuda_graph_decode_enablement_requires_flag_and_gpu_greedy(monkeypatch):
    manager = SimpleNamespace(
        config=SimpleNamespace(flashinfer_cuda_graph_decode=False),
        device=torch.device("cuda"),
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("MINI_OPENCLAW_FLASHINFER_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("MINI_OPENCLAW_FLASHINFER_CUDA_GRAPH_DECODE", raising=False)

    assert not flashinfer_cuda_graph_decode_enabled(manager.config)
    assert not can_use_flashinfer_cuda_graph_decode(manager, use_gpu_greedy=True, max_new_tokens=1)

    manager.config.flashinfer_cuda_graph_decode = True
    assert flashinfer_cuda_graph_decode_enabled(manager.config)
    assert can_use_flashinfer_cuda_graph_decode(manager, use_gpu_greedy=True, max_new_tokens=1)
    assert not can_use_flashinfer_cuda_graph_decode(manager, use_gpu_greedy=False, max_new_tokens=1)
    assert can_use_flashinfer_cuda_graph_batch_decode(manager, max_new_tokens=1)


def test_static_decode_append_uses_tensor_indices():
    layer_cache = torch.zeros(3, 2, 4, 2, 3)
    page_id = torch.tensor([1], dtype=torch.long)
    page_offset = torch.tensor([2], dtype=torch.long)
    kv_key = torch.tensor([0], dtype=torch.long)
    kv_value = torch.tensor([1], dtype=torch.long)
    k_nhd = torch.full((1, 2, 3), 3.0)
    v_nhd = torch.full((1, 2, 3), 5.0)

    _append_decode_kv_static(
        layer_cache,
        page_id=page_id,
        page_offset=page_offset,
        kv_key_index=kv_key,
        kv_value_index=kv_value,
        k_nhd=k_nhd,
        v_nhd=v_nhd,
    )

    assert torch.equal(layer_cache[1, 0, 2], k_nhd[0])
    assert torch.equal(layer_cache[1, 1, 2], v_nhd[0])
    assert layer_cache.sum().item() == k_nhd.sum().item() + v_nhd.sum().item()


def test_static_decode_append_accepts_batched_tensor_indices():
    layer_cache = torch.zeros(4, 2, 4, 2, 3)
    page_id = torch.tensor([1, 3], dtype=torch.long)
    page_offset = torch.tensor([2, 0], dtype=torch.long)
    kv_key = torch.zeros(2, dtype=torch.long)
    kv_value = torch.ones(2, dtype=torch.long)
    k_nhd = torch.stack(
        [
            torch.full((2, 3), 3.0),
            torch.full((2, 3), 7.0),
        ]
    )
    v_nhd = torch.stack(
        [
            torch.full((2, 3), 5.0),
            torch.full((2, 3), 11.0),
        ]
    )

    _append_decode_kv_static(
        layer_cache,
        page_id=page_id,
        page_offset=page_offset,
        kv_key_index=kv_key,
        kv_value_index=kv_value,
        k_nhd=k_nhd,
        v_nhd=v_nhd,
    )

    assert torch.equal(layer_cache[1, 0, 2], k_nhd[0])
    assert torch.equal(layer_cache[1, 1, 2], v_nhd[0])
    assert torch.equal(layer_cache[3, 0, 0], k_nhd[1])
    assert torch.equal(layer_cache[3, 1, 0], v_nhd[1])
    assert layer_cache.sum().item() == k_nhd.sum().item() + v_nhd.sum().item()


def test_manager_graph_chunk_metadata_advances_all_layers():
    manager = FlashInferPagedKVCacheManager(
        2,
        page_size=4,
        max_pages=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    request = manager.begin_request([1, 2], "general")
    spans = manager.reserve_chunk(request, 1)

    manager.mark_graph_decode_chunk_written(
        request,
        chunk_index=0,
        num_layers=2,
        device=torch.device("cpu"),
    )

    assert request.page_span_chunks[0] == spans
    assert request.layer_chunk_counts == {0: 1, 1: 1}


def test_page_table_buffer_signature_tracks_reusable_decode_buffers():
    manager = FlashInferPagedKVCacheManager(
        1,
        page_size=4,
        max_pages=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert manager.page_table_buffer_signature() == (0, 0, 0)
    table = manager.build_page_table(
        [SimpleNamespace(block_id=2, start=0, length=3)],
        device=torch.device("cpu"),
        reuse_buffer=True,
    )

    signature = manager.page_table_buffer_signature()
    assert signature == (
        table.paged_kv_indptr.data_ptr(),
        table.paged_kv_indices.data_ptr(),
        table.paged_kv_last_page_len.data_ptr(),
    )


def test_decode_graph_span_validation_requires_single_token_chunk():
    span = FlashInferDecodeCudaGraphRunner._single_decode_span

    assert span([]) is None
    assert span([SimpleNamespace(length=2, block_id=0, start=0)]) is None
    valid = SimpleNamespace(length=1, block_id=0, start=0)
    assert span([valid]) is valid


def test_batch_decode_graph_span_validation_requires_single_token_chunks():
    span = FlashInferBatchDecodeCudaGraphRunner._batch_decode_spans

    assert span([]) == []
    assert span([[SimpleNamespace(length=1, block_id=0, start=0)], []]) is None
    assert span([[SimpleNamespace(length=2, block_id=0, start=0)]]) is None
    valid = [
        [SimpleNamespace(length=1, block_id=0, start=0)],
        [SimpleNamespace(length=1, block_id=1, start=3)],
    ]
    assert span(valid) == [valid[0][0], valid[1][0]]


def test_decode_graph_cache_evicts_oldest_bucket(monkeypatch):
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_CUDA_GRAPH_MAX_ENTRIES", "2")
    runner = object.__new__(FlashInferDecodeCudaGraphRunner)
    runner._graphs = {}

    class FakeGraph:
        def replay(self):
            pass

    def make_key(page_table_pages: int) -> DecodeGraphKey:
        return DecodeGraphKey(
            model_id=1,
            backend_id=2,
            cache_data_ptr=3,
            cache_shape=(1, 4),
            cache_dtype=torch.float32,
            page_indptr_data_ptr=5,
            page_indices_data_ptr=6,
            page_last_len_data_ptr=7,
            token_shape=(1, 1),
            position_shape=(1, 1),
            page_table_pages=page_table_pages,
            active_adapter="general",
            active_role="serve",
            runtime_version=0,
        )

    for pages in (1, 2, 3):
        runner._store_graph(
            make_key(pages),
            DecodeGraphEntry(graph=FakeGraph(), logits=torch.zeros(1)),
        )

    assert runner.graph_count == 2
    assert make_key(1) not in runner._graphs
    assert make_key(2) in runner._graphs
    assert make_key(3) in runner._graphs
