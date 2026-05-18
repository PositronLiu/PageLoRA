"""Single-layer FlashInfer paged-KV execution helpers."""

from __future__ import annotations

import time
from typing import Literal

import torch

from model.flashinfer.attention_backend import FlashInferAttentionBackend
from model.flashinfer.perf import FlashInferPerfTrace
from model.flashinfer.projection import (
    apply_flashinfer_rope_pos_ids,
    apply_flashinfer_rope_pos_ids_for_append,
    project_qkv_with_optional_norm,
)
from model.flashinfer.packed_ops import (
    apply_gate_up_activation,
    project_down,
    project_gate_up_packed,
)
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
    input_norm_t0 = perf.tic() if perf is not None else time.perf_counter()
    normed = layer.input_layernorm(hidden)
    if perf is not None:
        perf.add_layer(mode, layer_idx, "input_norm_s", input_norm_t0)
    qkv_project_t0 = perf.tic() if perf is not None else time.perf_counter()
    projected = project_qkv_with_optional_norm(
        layer,
        plan,
        normed,
        use_flashinfer_norm=getattr(backend, "use_flashinfer_qk_norm", False),
    )
    if perf is not None:
        perf.add_layer(mode, layer_idx, "qkv_project_s", qkv_project_t0)
        perf.add_layer(mode, layer_idx, "qkv_s", qkv_t0)

    rope_t0 = perf.tic() if perf is not None else time.perf_counter()
    if bsz == 1:
        Q_attn, K_append = apply_flashinfer_rope_pos_ids_for_append(
            layer,
            projected.q_bshd,
            projected.k_bshd,
            position_ids,
        )
        V_append = projected.v_bshd[0].contiguous()
    else:
        Q_attn, K_store = apply_flashinfer_rope_pos_ids(
            layer,
            projected.q_bshd,
            projected.k_bshd,
            position_ids,
        )
        K_append = K_store[0].transpose(0, 1).contiguous()
        V_append = projected.v_bhsd[0].transpose(0, 1).contiguous()
    if perf is not None:
        perf.add_layer(mode, layer_idx, "rope_s", rope_t0)

    append_t0 = perf.tic() if perf is not None else time.perf_counter()
    cache_mgr.append_full_nhd(layer_idx, K_append, V_append, request=request_cache)
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
    o_layout_t0 = perf.tic() if perf is not None else time.perf_counter()
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    if perf is not None:
        perf.add_layer(mode, layer_idx, "o_layout_s", o_layout_t0)

    o_proj_t0 = perf.tic() if perf is not None else time.perf_counter()
    o_module = plan.modules.get("self_attn.o_proj")
    attn_output = o_module(attn_output) if o_module is not None else layer.self_attn.o_proj(attn_output)
    if perf is not None:
        perf.add_layer(mode, layer_idx, "o_proj_s", o_proj_t0)

    residual_t0 = perf.tic() if perf is not None else time.perf_counter()
    hidden = hidden + attn_output
    if perf is not None:
        perf.add_layer(mode, layer_idx, "post_residual_s", residual_t0)

    post_norm_t0 = perf.tic() if perf is not None else time.perf_counter()
    mlp_input = layer.post_attention_layernorm(hidden)
    if perf is not None:
        perf.add_layer(mode, layer_idx, "post_norm_s", post_norm_t0)

    mlp_output = _run_mlp(layer.mlp, mlp_input, mode=mode, layer_idx=layer_idx, perf=perf)
    final_residual_t0 = perf.tic() if perf is not None else time.perf_counter()
    result = hidden + mlp_output
    if perf is not None:
        perf.add_layer(mode, layer_idx, "post_final_residual_s", final_residual_t0)
        perf.add_layer(mode, layer_idx, "post_s", post_t0)
        perf.add_layer(mode, layer_idx, "total_s", layer_t0)
        perf.count_layer(mode, layer_idx)
    return result


def _run_mlp(
    mlp: torch.nn.Module,
    hidden: torch.Tensor,
    *,
    mode: Literal["prefill", "decode"],
    layer_idx: int,
    perf: FlashInferPerfTrace | None,
) -> torch.Tensor:
    gate_up_t0 = perf.tic() if perf is not None else time.perf_counter()
    gate_up = project_gate_up_packed(mlp, hidden)
    if gate_up is None:
        mlp_output = mlp(hidden)
        if perf is not None:
            perf.add_layer(mode, layer_idx, "mlp_s", gate_up_t0)
        return mlp_output
    if perf is not None:
        perf.add_layer(mode, layer_idx, "mlp_gate_up_s", gate_up_t0)

    act_t0 = perf.tic() if perf is not None else time.perf_counter()
    activated = apply_gate_up_activation(gate_up, mlp.act_fn)
    if perf is not None:
        perf.add_layer(mode, layer_idx, "mlp_act_s", act_t0)

    down_t0 = perf.tic() if perf is not None else time.perf_counter()
    output = project_down(mlp, activated)
    if perf is not None:
        perf.add_layer(mode, layer_idx, "mlp_down_s", down_t0)
    return output
