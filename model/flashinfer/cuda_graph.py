"""CUDA graph runners for FlashInfer single-token decode paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from core.utils import LOG, WARNING
from model.flashinfer.layer_runner import (
    run_flashinfer_decode_layer_static_append,
    run_flashinfer_decode_layer_static_append_batch,
)
from model.flashinfer.norm import apply_rmsnorm
from model.flashinfer.packed_ops import project_linear
from model.kvcache.paged_kvcache_manager import FlashInferPagedKVCacheManager
from model.kvcache.radix_types import KVBlockSpan

if TYPE_CHECKING:
    from model.flashinfer.attention_backend import FlashInferAttentionBackend
    from model.model_manager import ModelManager


def flashinfer_cuda_graph_decode_enabled(config: object) -> bool:
    """Return whether FlashInfer decode CUDA graph replay is requested."""

    return bool(getattr(config, "flashinfer_cuda_graph_decode", False)) or _env_enabled(
        "MINI_OPENCLAW_FLASHINFER_CUDA_GRAPH_DECODE",
        default=_env_enabled("MINI_OPENCLAW_FLASHINFER_CUDA_GRAPH", default=False),
    )


def flashinfer_cuda_graph_warmup_steps() -> int:
    """Return warmup iterations before graph capture."""

    raw = os.getenv("MINI_OPENCLAW_FLASHINFER_CUDA_GRAPH_WARMUP", "1")
    try:
        return max(0, int(raw))
    except ValueError:
        return 1


def can_use_flashinfer_cuda_graph_decode(
    manager: "ModelManager",
    *,
    use_gpu_greedy: bool,
    max_new_tokens: int,
) -> bool:
    """Check the supported graph decode envelope."""

    if not flashinfer_cuda_graph_decode_enabled(manager.config):
        return False
    if not use_gpu_greedy:
        return False
    if max_new_tokens <= 0:
        return False
    return manager.device.type == "cuda" and torch.cuda.is_available()


def can_use_flashinfer_cuda_graph_batch_decode(
    manager: "ModelManager",
    *,
    max_new_tokens: int,
) -> bool:
    """Check whether batched FlashInfer graph decode may be attempted."""

    if not flashinfer_cuda_graph_decode_enabled(manager.config):
        return False
    if max_new_tokens <= 0:
        return False
    return manager.device.type == "cuda" and torch.cuda.is_available()


@dataclass(frozen=True)
class DecodeGraphKey:
    """Static properties that make a captured decode graph reusable."""

    model_id: int
    backend_id: int
    cache_data_ptr: int
    cache_shape: tuple[int, ...]
    cache_dtype: torch.dtype
    page_indptr_data_ptr: int
    page_indices_data_ptr: int
    page_last_len_data_ptr: int
    token_shape: tuple[int, ...]
    position_shape: tuple[int, ...]
    page_table_pages: int
    active_adapter: str | None
    active_role: str | None
    runtime_version: int


@dataclass
class DecodeGraphEntry:
    """Captured graph and its graph-owned output tensor."""

    graph: torch.cuda.CUDAGraph
    logits: torch.Tensor


@dataclass
class BatchDecodeGraphEntry:
    """Captured batched graph plus graph-owned input/output tensors."""

    graph: torch.cuda.CUDAGraph
    logits: torch.Tensor
    token_buf: torch.Tensor
    position_buf: torch.Tensor
    page_id_buf: torch.Tensor
    page_offset_buf: torch.Tensor
    kv_key_index: torch.Tensor
    kv_value_index: torch.Tensor


class FlashInferDecodeCudaGraphRunner:
    """Capture and replay the FlashInfer single-request decode step."""

    def __init__(
        self,
        manager: "ModelManager",
        backend: "FlashInferAttentionBackend",
        cache_mgr: FlashInferPagedKVCacheManager,
    ) -> None:
        self.manager = manager
        self.backend = backend
        self.cache_mgr = cache_mgr
        self.device = manager.device
        self.token_buf = torch.empty((1, 1), device=self.device, dtype=torch.long)
        self.position_buf = torch.empty((1, 1), device=self.device, dtype=torch.long)
        self.page_id_buf = torch.empty((1,), device=self.device, dtype=torch.long)
        self.page_offset_buf = torch.empty((1,), device=self.device, dtype=torch.long)
        self.kv_key_index = torch.zeros((1,), device=self.device, dtype=torch.long)
        self.kv_value_index = torch.ones((1,), device=self.device, dtype=torch.long)
        self._graphs: dict[DecodeGraphKey, DecodeGraphEntry] = {}
        self._disabled_reason: str | None = None
        self._capture_count = 0

    @property
    def disabled_reason(self) -> str | None:
        """Return why graph replay was permanently disabled, if any."""

        return self._disabled_reason

    def is_compatible(
        self,
        backend: "FlashInferAttentionBackend",
        cache_mgr: FlashInferPagedKVCacheManager,
    ) -> bool:
        """Return whether this runner can be reused for the given state."""

        return self.backend is backend and self.cache_mgr is cache_mgr

    @property
    def graph_count(self) -> int:
        """Return the number of cached decode graph buckets."""

        return len(self._graphs)

    def clear(self) -> None:
        """Drop captured graph buckets."""

        self._graphs.clear()

    def run(
        self,
        *,
        token_tensor: torch.Tensor,
        position: int,
        chunk_spans: list[KVBlockSpan],
        page_table_pages: int,
    ) -> torch.Tensor | None:
        """Replay a captured decode graph, capturing first if needed."""

        if self._disabled_reason is not None:
            return None
        span = self._single_decode_span(chunk_spans)
        if span is None:
            self._disabled_reason = "decode graph requires one-token single-page chunks"
            return None

        self._ensure_dense_cache_capacity(span)
        cache = self.cache_mgr.paged_kv_cache
        if cache is None:
            return None

        self.token_buf.copy_(token_tensor)
        self.position_buf.fill_(int(position))
        self.page_id_buf.fill_(int(span.block_id))
        self.page_offset_buf.fill_(int(span.start))

        key = self._make_key(cache, page_table_pages=page_table_pages)
        entry = self._graphs.get(key)
        if entry is None:
            entry = self._capture(key)
            if entry is None:
                return None

        entry.graph.replay()
        return entry.logits

    def _capture(self, key: DecodeGraphKey) -> DecodeGraphEntry | None:
        try:
            entry = self._capture_graph()
        except Exception as exc:  # pragma: no cover - depends on CUDA/FlashInfer runtime
            self._disabled_reason = str(exc)
            self._graphs.clear()
            WARNING(f"FlashInfer CUDA graph decode disabled after capture failure: {exc}")
            return None
        self._capture_count += 1
        self._store_graph(key, entry)
        LOG(
            "FlashInfer CUDA graph decode captured "
            f"count={self._capture_count} pages={key.page_table_pages} "
            f"cache_pages={key.cache_shape[1] if len(key.cache_shape) > 1 else 0} "
            f"cached={len(self._graphs)}"
        )
        return entry

    def _capture_graph(self) -> DecodeGraphEntry:
        warmup_stream = torch.cuda.Stream(device=self.device)
        current_stream = torch.cuda.current_stream(self.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(flashinfer_cuda_graph_warmup_steps()):
                self._decode_static()
        current_stream.wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = self._decode_static()
        return DecodeGraphEntry(graph=graph, logits=logits)

    def _store_graph(self, key: DecodeGraphKey, entry: DecodeGraphEntry) -> None:
        max_entries = _env_int("MINI_OPENCLAW_FLASHINFER_CUDA_GRAPH_MAX_ENTRIES", default=32)
        if max_entries <= 0:
            self._graphs.clear()
            return
        while len(self._graphs) >= max_entries:
            self._graphs.pop(next(iter(self._graphs)))
        self._graphs[key] = entry

    def _decode_static(self) -> torch.Tensor:
        model = self.manager.model
        hidden = model.get_input_embeddings()(self.token_buf)
        for layer_idx, layer in enumerate(self.manager.lora_runtime.layers):
            hidden = run_flashinfer_decode_layer_static_append(
                layer,
                self.manager.lora_runtime.layer_plans[layer_idx],
                layer_idx,
                hidden,
                self.cache_mgr,
                self.position_buf,
                backend=self.backend,
                page_id=self.page_id_buf,
                page_offset=self.page_offset_buf,
                kv_key_index=self.kv_key_index,
                kv_value_index=self.kv_value_index,
            )
        normed = apply_rmsnorm(model.model.norm, hidden[:, -1:, :])
        return project_linear(model.lm_head, normed)

    def _ensure_dense_cache_capacity(self, span: KVBlockSpan) -> None:
        layer = self.manager.lora_runtime.layers[0]
        attn_config = layer.self_attn.config
        self.cache_mgr.ensure_dense_cache_shape(
            layer_idx=0,
            required_pages=int(span.block_id) + 1,
            num_kv_heads=int(getattr(attn_config, "num_key_value_heads", attn_config.num_attention_heads)),
            head_dim=int(layer.self_attn.head_dim),
            dtype=self.manager.model.get_input_embeddings().weight.dtype,
            device=self.device,
        )

    def _make_key(self, cache: torch.Tensor, *, page_table_pages: int) -> DecodeGraphKey:
        runtime = self.manager.lora_runtime
        page_indptr_ptr, page_indices_ptr, page_last_len_ptr = self.cache_mgr.page_table_buffer_signature()
        return DecodeGraphKey(
            model_id=id(self.manager.model),
            backend_id=id(self.backend),
            cache_data_ptr=int(cache.data_ptr()),
            cache_shape=tuple(int(dim) for dim in cache.shape),
            cache_dtype=cache.dtype,
            page_indptr_data_ptr=page_indptr_ptr,
            page_indices_data_ptr=page_indices_ptr,
            page_last_len_data_ptr=page_last_len_ptr,
            token_shape=tuple(int(dim) for dim in self.token_buf.shape),
            position_shape=tuple(int(dim) for dim in self.position_buf.shape),
            page_table_pages=int(page_table_pages),
            active_adapter=getattr(runtime, "active_logical_adapter_name", None),
            active_role=getattr(runtime, "active_role", None),
            runtime_version=int(getattr(runtime, "flashinfer_graph_version", 0)),
        )

    @staticmethod
    def _single_decode_span(spans: list[KVBlockSpan]) -> KVBlockSpan | None:
        if len(spans) != 1:
            return None
        span = spans[0]
        if span.length != 1:
            return None
        return span


class FlashInferBatchDecodeCudaGraphRunner:
    """Capture and replay FlashInfer batched single-token decode steps."""

    def __init__(
        self,
        manager: "ModelManager",
        backend: "FlashInferAttentionBackend",
        cache_mgr: FlashInferPagedKVCacheManager,
    ) -> None:
        self.manager = manager
        self.backend = backend
        self.cache_mgr = cache_mgr
        self.device = manager.device
        self._graphs: dict[DecodeGraphKey, BatchDecodeGraphEntry] = {}
        self._disabled_reason: str | None = None
        self._capture_count = 0

    @property
    def disabled_reason(self) -> str | None:
        """Return why graph replay was permanently disabled, if any."""

        return self._disabled_reason

    def is_compatible(
        self,
        backend: "FlashInferAttentionBackend",
        cache_mgr: FlashInferPagedKVCacheManager,
    ) -> bool:
        """Return whether this runner can be reused for the given state."""

        return self.backend is backend and self.cache_mgr is cache_mgr

    @property
    def graph_count(self) -> int:
        """Return the number of cached decode graph buckets."""

        return len(self._graphs)

    def clear(self) -> None:
        """Drop captured graph buckets."""

        self._graphs.clear()

    def run(
        self,
        *,
        token_ids: list[int],
        positions: list[int],
        chunk_spans_batch: list[list[KVBlockSpan]],
        page_table_pages: int,
    ) -> torch.Tensor | None:
        """Replay a captured batched decode graph, capturing first if needed."""

        if self._disabled_reason is not None:
            return None
        if not token_ids:
            return None
        if len(token_ids) != len(positions) or len(token_ids) != len(chunk_spans_batch):
            self._disabled_reason = "batch decode graph metadata sizes differ"
            return None

        spans = self._batch_decode_spans(chunk_spans_batch)
        if spans is None:
            self._disabled_reason = "batch decode graph requires one-token single-page chunks"
            return None

        self._ensure_dense_cache_capacity(spans)
        cache = self.cache_mgr.paged_kv_cache
        if cache is None:
            return None

        batch_size = len(token_ids)
        key = self._make_key(cache, batch_size=batch_size, page_table_pages=page_table_pages)
        entry = self._graphs.get(key)
        if entry is None:
            entry = self._make_entry(batch_size)
            self._copy_inputs(
                entry,
                token_ids=token_ids,
                positions=positions,
                spans=spans,
            )
            entry = self._capture(key, entry)
            if entry is None:
                return None
        else:
            self._copy_inputs(
                entry,
                token_ids=token_ids,
                positions=positions,
                spans=spans,
            )

        entry.graph.replay()
        return entry.logits

    def _copy_inputs(
        self,
        entry: BatchDecodeGraphEntry,
        *,
        token_ids: list[int],
        positions: list[int],
        spans: list[KVBlockSpan],
    ) -> None:
        entry.token_buf[:, 0].copy_(torch.tensor(token_ids, device=self.device, dtype=torch.long))
        entry.position_buf[:, 0].copy_(torch.tensor(positions, device=self.device, dtype=torch.long))
        entry.page_id_buf.copy_(
            torch.tensor([span.block_id for span in spans], device=self.device, dtype=torch.long)
        )
        entry.page_offset_buf.copy_(
            torch.tensor([span.start for span in spans], device=self.device, dtype=torch.long)
        )

    def _capture(self, key: DecodeGraphKey, entry: BatchDecodeGraphEntry) -> BatchDecodeGraphEntry | None:
        try:
            entry = self._capture_graph(entry)
        except Exception as exc:  # pragma: no cover - depends on CUDA/FlashInfer runtime
            self._disabled_reason = str(exc)
            self._graphs.clear()
            WARNING(f"FlashInfer batch CUDA graph decode disabled after capture failure: {exc}")
            return None
        self._capture_count += 1
        self._store_graph(key, entry)
        LOG(
            "FlashInfer batch CUDA graph decode captured "
            f"count={self._capture_count} batch={entry.token_buf.size(0)} pages={key.page_table_pages} "
            f"cache_pages={key.cache_shape[1] if len(key.cache_shape) > 1 else 0} "
            f"cached={len(self._graphs)}"
        )
        return entry

    def _capture_graph(self, entry: BatchDecodeGraphEntry) -> BatchDecodeGraphEntry:
        warmup_stream = torch.cuda.Stream(device=self.device)
        current_stream = torch.cuda.current_stream(self.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(flashinfer_cuda_graph_warmup_steps()):
                self._decode_static(entry)
        current_stream.wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = self._decode_static(entry)
        entry.graph = graph
        entry.logits = logits
        return entry

    def _make_entry(self, batch_size: int) -> BatchDecodeGraphEntry:
        return BatchDecodeGraphEntry(
            graph=torch.cuda.CUDAGraph(),
            logits=torch.empty((batch_size, 1, 1), device=self.device),
            token_buf=torch.empty((batch_size, 1), device=self.device, dtype=torch.long),
            position_buf=torch.empty((batch_size, 1), device=self.device, dtype=torch.long),
            page_id_buf=torch.empty((batch_size,), device=self.device, dtype=torch.long),
            page_offset_buf=torch.empty((batch_size,), device=self.device, dtype=torch.long),
            kv_key_index=torch.zeros((batch_size,), device=self.device, dtype=torch.long),
            kv_value_index=torch.ones((batch_size,), device=self.device, dtype=torch.long),
        )

    def _store_graph(self, key: DecodeGraphKey, entry: BatchDecodeGraphEntry) -> None:
        max_entries = _env_int("MINI_OPENCLAW_FLASHINFER_CUDA_GRAPH_MAX_ENTRIES", default=32)
        if max_entries <= 0:
            self._graphs.clear()
            return
        while len(self._graphs) >= max_entries:
            self._graphs.pop(next(iter(self._graphs)))
        self._graphs[key] = entry

    def _decode_static(self, entry: BatchDecodeGraphEntry) -> torch.Tensor:
        model = self.manager.model
        hidden = model.get_input_embeddings()(entry.token_buf)
        for layer_idx, layer in enumerate(self.manager.lora_runtime.layers):
            hidden = run_flashinfer_decode_layer_static_append_batch(
                layer,
                self.manager.lora_runtime.layer_plans[layer_idx],
                layer_idx,
                hidden,
                self.cache_mgr,
                entry.position_buf,
                backend=self.backend,
                page_id=entry.page_id_buf,
                page_offset=entry.page_offset_buf,
                kv_key_index=entry.kv_key_index,
                kv_value_index=entry.kv_value_index,
            )
        normed = apply_rmsnorm(model.model.norm, hidden[:, -1:, :])
        return project_linear(model.lm_head, normed)

    def _ensure_dense_cache_capacity(self, spans: list[KVBlockSpan]) -> None:
        layer = self.manager.lora_runtime.layers[0]
        attn_config = layer.self_attn.config
        self.cache_mgr.ensure_dense_cache_shape(
            layer_idx=0,
            required_pages=max(int(span.block_id) for span in spans) + 1,
            num_kv_heads=int(getattr(attn_config, "num_key_value_heads", attn_config.num_attention_heads)),
            head_dim=int(layer.self_attn.head_dim),
            dtype=self.manager.model.get_input_embeddings().weight.dtype,
            device=self.device,
        )

    def _make_key(
        self,
        cache: torch.Tensor,
        *,
        batch_size: int,
        page_table_pages: int,
    ) -> DecodeGraphKey:
        runtime = self.manager.lora_runtime
        page_indptr_ptr, page_indices_ptr, page_last_len_ptr = self.cache_mgr.batch_page_table_buffer_signature()
        return DecodeGraphKey(
            model_id=id(self.manager.model),
            backend_id=id(self.backend),
            cache_data_ptr=int(cache.data_ptr()),
            cache_shape=tuple(int(dim) for dim in cache.shape),
            cache_dtype=cache.dtype,
            page_indptr_data_ptr=page_indptr_ptr,
            page_indices_data_ptr=page_indices_ptr,
            page_last_len_data_ptr=page_last_len_ptr,
            token_shape=(int(batch_size), 1),
            position_shape=(int(batch_size), 1),
            page_table_pages=int(page_table_pages),
            active_adapter=getattr(runtime, "active_logical_adapter_name", None),
            active_role=getattr(runtime, "active_role", None),
            runtime_version=int(getattr(runtime, "flashinfer_graph_version", 0)),
        )

    @staticmethod
    def _batch_decode_spans(spans_batch: list[list[KVBlockSpan]]) -> list[KVBlockSpan] | None:
        spans: list[KVBlockSpan] = []
        for chunk_spans in spans_batch:
            if len(chunk_spans) != 1:
                return None
            span = chunk_spans[0]
            if span.length != 1:
                return None
            spans.append(span)
        return spans


def _env_enabled(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default
