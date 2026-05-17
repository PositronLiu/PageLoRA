"""Single-layer FlashInfer paged-KV execution helpers."""

from __future__ import annotations

import time
from typing import Literal

import torch

from model.flashinfer.attention_backend import FlashInferAttentionBackend
from model.flashinfer.perf import FlashInferPerfTrace
from model.flashinfer.projection import apply_flashinfer_rope_pos_ids, project_qkv_with_optional_norm
from model.kvcache.paged_kvcache_manager import FlashInferPagedKVCacheManager
from model.kvcache.paged_kvcache_request import FlashInferPagedKVRequest


def _run_flashinfer_attention(
    layer: torch.nn.Module,
    plan,
    layer_idx: int,
    hidden: torch.Tensor,
    cache_mgr: FlashInferPagedKVCacheManager,
    position_ids: torch.Tensor,
    request_cache: FlashInferPagedKVRequest,
    *,
    mode: Literal["prefill", "decode"],
    backend: FlashInferAttentionBackend,
    perf: FlashInferPerfTrace | None = None,
) -> torch.Tensor:
    """Run one decoder layer while writing rotated K/V into paged cache."""

    layer_t0 = perf.tic() if perf is not None else time.perf_counter()
    bsz, seqlen, _ = hidden.shape
    qkv_t0 = perf.tic() if perf is not None else time.perf_counter()
    normed = layer.input_layernorm(hidden)
    projected = project_qkv_with_optional_norm(
        layer,
        plan,
        normed,
        use_flashinfer_norm=getattr(backend, "use_flashinfer_qk_norm", False),
    )
    if perf is not None:
        perf.add_layer(mode, layer_idx, "qkv_s", qkv_t0)

    rope_t0 = perf.tic() if perf is not None else time.perf_counter()
    Q_attn, K_store = apply_flashinfer_rope_pos_ids(
        layer,
        projected.q_bshd,
        projected.k_bshd,
        position_ids,
    )
    V = projected.v_bhsd
    if perf is not None:
        perf.add_layer(mode, layer_idx, "rope_s", rope_t0)

    append_t0 = perf.tic() if perf is not None else time.perf_counter()
    cache_mgr.append_full(layer_idx, K_store, V, request=request_cache)
    if perf is not None:
        perf.add_layer(mode, layer_idx, "kv_append_s", append_t0)

    cache_t0 = perf.tic() if perf is not None else time.perf_counter()
    layer_cache = cache_mgr.layer_cache(layer_idx)
    if layer_cache is None:
        raise RuntimeError("FlashInfer paged KV requires a dense paged_kv_cache")
    if perf is not None:
        perf.add_layer(mode, layer_idx, "layer_cache_s", cache_t0)

    attn_t0 = perf.tic() if perf is not None else time.perf_counter()
    if mode == "decode":
        attn_output = backend.run_decode(Q_attn, layer_cache)
    else:
        attn_output = backend.run_prefill(Q_attn, layer_cache)
    if perf is not None:
        perf.add_layer(mode, layer_idx, "attn_run_s", attn_t0)

    post_t0 = perf.tic() if perf is not None else time.perf_counter()
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    o_module = plan.modules.get("self_attn.o_proj")
    attn_output = o_module(attn_output) if o_module is not None else layer.self_attn.o_proj(attn_output)

    hidden = hidden + attn_output
    mlp_output = layer.mlp(layer.post_attention_layernorm(hidden))
    result = hidden + mlp_output
    if perf is not None:
        perf.add_layer(mode, layer_idx, "post_s", post_t0)
        perf.add_layer(mode, layer_idx, "total_s", layer_t0)
        perf.count_layer(mode, layer_idx)
    return result
