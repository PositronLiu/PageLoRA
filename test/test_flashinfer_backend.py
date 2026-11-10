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
        sm_scale=None,
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
            "sm_scale": sm_scale,
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
        sm_scale=None,
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
            "sm_scale": sm_scale,
        }


def _backend_class_for(monkeypatch, *, decode_wrapper, prefill_wrapper):
    fake_flashinfer = SimpleNamespace(
        BatchDecodeWithPagedKVCacheWrapper=decode_wrapper,
        BatchPrefillWithPagedKVCacheWrapper=prefill_wrapper,
    )
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    module = importlib.import_module("model.flashinfer.attention_backend")
    module = importlib.reload(module)
    return module.FlashInferAttentionBackend


def _backend_class(monkeypatch):
    return _backend_class_for(
        monkeypatch,
        decode_wrapper=_FakeWrapper,
        prefill_wrapper=_FakeWrapper,
    )


def test_qwen_group_seven_selects_tensor_core_decode(monkeypatch):
    class TensorCoreWrapper(_FakeWrapper):
        def __init__(self, workspace_buffer, kv_layout="NHD", use_tensor_cores=False):
            super().__init__(workspace_buffer, kv_layout)
            assert use_tensor_cores, 'group_size=7 cannot use the CUDA-core decode kernel'

    backend_cls = _backend_class_for(monkeypatch, decode_wrapper=TensorCoreWrapper,
                                     prefill_wrapper=_FakeWrapper)
    backend = backend_cls(device=torch.device('cpu'), workspace_mb=1,
                          num_qo_heads=28, num_kv_heads=4, head_dim=128, page_size=32)
    assert backend.decode_use_tensor_cores



def test_decode_plan_buffers_survive_replanning_and_resize(monkeypatch):
    class GraphWrapper(_FakeWrapper):
        def __init__(self, workspace_buffer, kv_layout='NHD', **kwargs):
            super().__init__(workspace_buffer, kv_layout)
            self.options = kwargs
            self._plan_info = [1, 2]

    backend_cls = _backend_class_for(monkeypatch, decode_wrapper=GraphWrapper,
                                     prefill_wrapper=_FakeWrapper)
    backend = backend_cls(device=torch.device('cpu'), workspace_mb=1,
                          num_qo_heads=28, num_kv_heads=4, head_dim=128, page_size=32)
    # Exercise allocation/lifetime policy on CPU without a CUDA runtime.
    backend.device = SimpleNamespace(type='cuda')
    original_empty = torch.empty
    monkeypatch.setattr(torch, 'empty', lambda *a, **kw: original_empty(*a, **dict(kw, device='cpu')))
    indptr = torch.tensor([0, 1], dtype=torch.int32)
    indices = torch.tensor([0], dtype=torch.int32)
    last = torch.tensor([1], dtype=torch.int32)
    backend._ensure_graph_safe_decode_buffers(indptr, indices, last)
    wrapper = backend._decode_wrapper
    signature = backend.decode_graph_signature()
    assert wrapper.options['use_cuda_graph']
    assert wrapper.options['backend'] == 'fa2'
    backend._ensure_graph_safe_decode_buffers(indptr, indices, last)
    assert backend._decode_wrapper is wrapper
    wrapper._plan_info = [1, 3]
    assert signature != backend.decode_graph_signature()
    backend._ensure_graph_safe_decode_buffers(indptr, torch.zeros(17, dtype=torch.int32), last)
    assert backend._decode_wrapper is not wrapper


def test_query_rescaling_prevents_finite_mask_from_winning(monkeypatch):
    class FiniteMaskWrapper(_FakeWrapper):
        def run(self, q, kv):
            k, v = kv
            scores = q[:, 0].float() @ k[:, 0].float().T
            scores.masked_fill_(torch.ones_like(scores, dtype=torch.bool).triu(1), -50000.)
            return ((scores * self.plan_kwargs['sm_scale']).softmax(-1) @ v[:, 0].float()).unsqueeze(1)

    backend_cls = _backend_class_for(monkeypatch, decode_wrapper=_FakeWrapper,
                                     prefill_wrapper=FiniteMaskWrapper)
    backend = backend_cls(device=torch.device('cpu'), workspace_mb=1,
                          num_qo_heads=1, num_kv_heads=1, head_dim=1, page_size=2)
    backend.plan_prefill(qo_indptr=torch.tensor([0, 2]), paged_kv_indptr=torch.tensor([0, 1]),
        paged_kv_indices=torch.tensor([0]), paged_kv_last_page_len=torch.tensor([2]))
    q = torch.full((2, 1, 1), -1000.)
    kv = (torch.full((2, 1, 1), 100.), torch.tensor([1., 9.]).reshape(2, 1, 1))
    result = backend.run_prefill(q, kv)
    assert torch.allclose(result[:, 0, 0], torch.tensor([1., 5.]))
    # Without rescaling, the masked second position beats the real first score.
    backend._prefill_wrapper.plan_kwargs['sm_scale'] = 1.
    assert backend._prefill_wrapper.run(q, kv)[0, 0, 0] == 9.


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


def test_plan_decode_reuses_page_table_buffer():
    from model.flashinfer.generation import _plan_decode

    calls = []
    table = SimpleNamespace(
        paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        paged_kv_indices=torch.tensor([7], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
    )

    class FakeCacheManager:
        def build_page_table(self, spans, *, device=None, reuse_buffer=False):
            calls.append((spans, device, reuse_buffer))
            return table

    class FakeBackend:
        def __init__(self) -> None:
            self.kwargs = None

        def plan_decode(self, **kwargs) -> None:
            self.kwargs = kwargs

    backend = FakeBackend()
    request = SimpleNamespace(prefix_spans=["prefix"], new_spans=["new"])

    output = _plan_decode(
        backend,
        FakeCacheManager(),
        request,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert output is table
    assert calls == [(["prefix", "new"], torch.device("cpu"), True)]
    assert backend.kwargs["paged_kv_indices"] is table.paged_kv_indices
    assert backend.kwargs["q_data_type"] is torch.float32


def test_async_host_plan_retains_sources_until_upload_event(monkeypatch):
    from collections import deque

    Backend = _backend_class(monkeypatch)
    backend = Backend.__new__(Backend)
    backend.device = torch.device('cuda')
    backend.num_qo_heads, backend.num_kv_heads = 28, 4
    backend.head_dim, backend.page_size, backend.query_rescale = 128, 32, 16.
    backend._decode_wrapper = _FakeWrapper(None)
    backend._decode_host_uploads = deque()
    backend._ensure_graph_safe_decode_buffers = lambda *args: None

    class Event:
        def __init__(self):
            self.done = False
            self.recorded = False
        def record(self, stream):
            self.recorded = True
        def query(self):
            return self.done
        def synchronize(self):
            self.done = True

    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: 'stream')
    def source():
        return SimpleNamespace(device=torch.device('cpu'), is_pinned=lambda: True)
    tensors = (source(), source(), source())
    def plan(items):
        backend.plan_decode(paged_kv_indptr=items[0], paged_kv_indices=items[1],
                            paged_kv_last_page_len=items[2], q_data_type=torch.float16)
    plan(tensors)
    event, retained = backend._decode_host_uploads[0]
    assert event.recorded and retained == tensors
    assert backend._decode_wrapper.plan_kwargs['non_blocking'] is True
    assert backend._decode_wrapper.plan_kwargs['sm_scale'] == 16 / (128 ** .5)
    event.done = True
    plan((source(), source(), source()))
    assert all(e is not event for e, _ in backend._decode_host_uploads)
    oldest = backend._decode_host_uploads[0][0]
    for _ in range(9):
        plan((source(), source(), source()))
    assert len(backend._decode_host_uploads) <= 8
    assert oldest.done  # Backpressure waits rather than recycling pending pinned memory.
