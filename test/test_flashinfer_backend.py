from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import torch


class _FakeWrapper:
    def __init__(self, workspace_buffer, kv_layout="NHD") -> None:
        self.workspace_buffer = workspace_buffer
        self.kv_layout = kv_layout
        self.plan_kwargs = None

    def plan(self, **kwargs) -> None:
        self.plan_kwargs = kwargs

    def run(self, q, paged_kv_cache_layer):
        return q + paged_kv_cache_layer.sum() * 0


def _backend_class(monkeypatch):
    fake_flashinfer = SimpleNamespace(
        BatchDecodeWithPagedKVCacheWrapper=_FakeWrapper,
        BatchPrefillWithPagedKVCacheWrapper=_FakeWrapper,
    )
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    module = importlib.import_module("model.flashinfer_attention_backend")
    module = importlib.reload(module)
    return module.FlashInferAttentionBackend


def test_flashinfer_decode_wrapper_is_initialized_eagerly(monkeypatch):
    backend_cls = _backend_class(monkeypatch)
    backend = backend_cls(
        device=torch.device("cpu"),
        workspace_mb=1,
        num_qo_heads=2,
        num_kv_heads=2,
        head_dim=4,
        page_size=2,
    )

    assert backend._decode_wrapper is not None
    assert backend._prefill_wrapper is not None
    backend.plan_decode(
        paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        paged_kv_indices=torch.tensor([0], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
        q_data_type=torch.float32,
        kv_data_type=torch.float32,
    )
    out = backend.run_decode(
        torch.ones(1, 2, 1, 4),
        torch.zeros(1, 2, 2, 2, 4),
    )

    assert out.shape == (1, 2, 1, 4)
    assert backend._decode_wrapper.plan_kwargs["pos_encoding_mode"] == "NONE"


def test_flashinfer_prefill_wrapper_is_initialized_eagerly(monkeypatch):
    backend_cls = _backend_class(monkeypatch)
    backend = backend_cls(
        device=torch.device("cpu"),
        workspace_mb=1,
        num_qo_heads=2,
        num_kv_heads=2,
        head_dim=4,
        page_size=2,
    )

    backend.plan_prefill(
        qo_indptr=torch.tensor([0, 2], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        paged_kv_indices=torch.tensor([0], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([2], dtype=torch.int32),
        q_data_type=torch.float32,
        kv_data_type=torch.float32,
    )
    out = backend.run_prefill(
        torch.ones(2, 2, 4),
        torch.zeros(1, 2, 2, 2, 4),
    )

    assert out.shape == (2, 2, 4)
    assert backend._prefill_wrapper.plan_kwargs["pos_encoding_mode"] == "NONE"


def test_flashinfer_prefill_wrapper_accepts_bhsd_queries(monkeypatch):
    backend_cls = _backend_class(monkeypatch)
    backend = backend_cls(
        device=torch.device("cpu"),
        workspace_mb=1,
        num_qo_heads=2,
        num_kv_heads=2,
        head_dim=4,
        page_size=2,
    )

    out = backend.run_prefill(
        torch.ones(1, 2, 3, 4),
        torch.zeros(1, 2, 2, 2, 4),
    )

    assert out.shape == (1, 2, 3, 4)
