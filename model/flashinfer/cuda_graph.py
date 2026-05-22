"""CUDA graph runner for the FlashInfer single-token decode path."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from core.utils import LOG, WARNING
from model.flashinfer.layer_runner import run_flashinfer_decode_layer_static_append
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


@dataclass(frozen=True)
class DecodeGraphKey:
    """Static properties that make a captured decode graph reusable."""

    model_id: int
    cache_data_ptr: int
    cache_shape: tuple[int, ...]
    cache_dtype: torch.dtype
    token_shape: tuple[int, ...]
    position_shape: tuple[int, ...]
    page_table_pages: int
    active_adapter: str | None
    active_role: str | None


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
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_key: DecodeGraphKey | None = None
        self._static_logits: torch.Tensor | None = None
        self._disabled_reason: str | None = None
        self._capture_count = 0

    @property
    def disabled_reason(self) -> str | None:
        """Return why graph replay was permanently disabled, if any."""

        return self._disabled_reason

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
        if self._graph is None or self._graph_key != key:
            if not self._capture(key):
                return None

        if self._graph is None or self._static_logits is None:
            return None
        self._graph.replay()
        return self._static_logits

    def _capture(self, key: DecodeGraphKey) -> bool:
        try:
            self._capture_graph(key)
        except Exception as exc:  # pragma: no cover - depends on CUDA/FlashInfer runtime
            self._disabled_reason = str(exc)
            self._graph = None
            self._graph_key = None
            self._static_logits = None
            WARNING(f"FlashInfer CUDA graph decode disabled after capture failure: {exc}")
            return False
        self._capture_count += 1
        LOG(
            "FlashInfer CUDA graph decode captured "
            f"count={self._capture_count} pages={key.page_table_pages} "
            f"cache_pages={key.cache_shape[1] if len(key.cache_shape) > 1 else 0}"
        )
        return True

    def _capture_graph(self, key: DecodeGraphKey) -> None:
        warmup_stream = torch.cuda.Stream(device=self.device)
        current_stream = torch.cuda.current_stream(self.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(flashinfer_cuda_graph_warmup_steps()):
                self._decode_static()
        current_stream.wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._static_logits = self._decode_static()
        self._graph = graph
        self._graph_key = key

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
        return DecodeGraphKey(
            model_id=id(self.manager.model),
            cache_data_ptr=int(cache.data_ptr()),
            cache_shape=tuple(int(dim) for dim in cache.shape),
            cache_dtype=cache.dtype,
            token_shape=tuple(int(dim) for dim in self.token_buf.shape),
            position_shape=tuple(int(dim) for dim in self.position_buf.shape),
            page_table_pages=int(page_table_pages),
            active_adapter=getattr(runtime, "active_logical_adapter_name", None),
            active_role=getattr(runtime, "active_role", None),
        )

    @staticmethod
    def _single_decode_span(spans: list[KVBlockSpan]) -> KVBlockSpan | None:
        if len(spans) != 1:
            return None
        span = spans[0]
        if span.length != 1:
            return None
        return span


def _env_enabled(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
