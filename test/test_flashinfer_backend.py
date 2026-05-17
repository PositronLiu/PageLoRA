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


class _StrictDecodeWrapper(_FakeWrapper):
    def plan(
        self,
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type="float16",
        kv_data_type=None,
    ) -> None:
        self.plan_kwargs = {
            "indptr": indptr,
            "indices": indices,
            "last_page_len": last_page_len,
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "page_size": page_size,
            "pos_encoding_mode": pos_encoding_mode,
            "q_data_type": q_data_type,
            "kv_data_type": kv_data_type,
        }


class _StrictPrefillWrapper(_FakeWrapper):
    def plan(
        self,
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        page_size,
        head_dim_vo=None,
        custom_mask=None,
        packed_custom_mask=None,
        causal=False,
        pos_encoding_mode="NONE",
        q_data_type="float16",
        kv_data_type=None,
    ) -> None:
        self.plan_kwargs = {
            "qo_indptr": qo_indptr,
            "paged_kv_indptr": paged_kv_indptr,
            "paged_kv_indices": paged_kv_indices,
            "paged_kv_last_page_len": paged_kv_last_page_len,
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim_qk": head_dim_qk,
            "page_size": page_size,
            "head_dim_vo": head_dim_vo,
            "custom_mask": custom_mask,
            "packed_custom_mask": packed_custom_mask,
            "causal": causal,
            "pos_encoding_mode": pos_encoding_mode,
            "q_data_type": q_data_type,
            "kv_data_type": kv_data_type,
        }


def _backend_class_for(monkeypatch, *, decode_wrapper, prefill_wrapper):
    fake_flashinfer = SimpleNamespace(
        BatchDecodeWithPagedKVCacheWrapper=decode_wrapper,
        BatchPrefillWithPagedKVCacheWrapper=prefill_wrapper,
    )
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    module = importlib.import_module("model.flashinfer_attention_backend")
    module = importlib.reload(module)
    return module.FlashInferAttentionBackend


def _backend_class(monkeypatch):
    return _backend_class_for(
        monkeypatch,
        decode_wrapper=_FakeWrapper,
        prefill_wrapper=_FakeWrapper,
    )


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


def test_flashinfer_plan_preserves_bfloat16_dtype_with_strict_signatures(monkeypatch):
    backend_cls = _backend_class_for(
        monkeypatch,
        decode_wrapper=_StrictDecodeWrapper,
        prefill_wrapper=_StrictPrefillWrapper,
    )
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
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    backend.plan_decode(
        paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        paged_kv_indices=torch.tensor([0], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    assert backend._prefill_wrapper.plan_kwargs["head_dim_qk"] == 4
    assert backend._prefill_wrapper.plan_kwargs["q_data_type"] is torch.bfloat16
    assert backend._prefill_wrapper.plan_kwargs["kv_data_type"] is torch.bfloat16
    assert backend._decode_wrapper.plan_kwargs["head_dim"] == 4
    assert backend._decode_wrapper.plan_kwargs["q_data_type"] is torch.bfloat16
    assert backend._decode_wrapper.plan_kwargs["kv_data_type"] is torch.bfloat16
