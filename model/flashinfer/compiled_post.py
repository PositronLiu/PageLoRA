"""torch.compile helpers for post-attention FlashInfer decode compute."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from core.utils import WARNING
from model.flashinfer.compile_config import (
    can_compile_decode_tensor,
    compile_module,
    mark_compile_step_begin,
    torch_compile_clone_outputs,
    torch_compile_post_enabled,
)
from model.flashinfer.static_params import StaticPostParams, resolve_post_params


@dataclass(frozen=True)
class CompiledPostAttention:
    module: torch.nn.Module
    cache_key: tuple[object, ...]


class StaticPostAttentionCompute(torch.nn.Module):
    """Tensor-only post-attention block for single-token decode."""

    def __init__(self, params: StaticPostParams) -> None:
        super().__init__()
        self.o_weight = params.o_proj.weight
        self.o_bias = params.o_proj.bias
        self.post_norm_weight = params.post_norm.weight
        self.post_norm_eps = float(params.post_norm.eps)
        self.gate_up_weight = params.mlp.gate_up.weight
        self.gate_up_bias = params.mlp.gate_up.bias
        self.down_weight = params.mlp.down.weight
        self.down_bias = params.mlp.down.bias

    def forward(self, hidden: torch.Tensor, attn_output: torch.Tensor) -> torch.Tensor:
        attn_output = attn_output.transpose(1, 2).contiguous().view(hidden.size(0), hidden.size(1), -1)
        attn_output = F.linear(attn_output, self.o_weight, self.o_bias)
        hidden = hidden + attn_output
        mlp_input = _rmsnorm_inline(hidden, self.post_norm_weight, self.post_norm_eps)
        gate_up = F.linear(mlp_input, self.gate_up_weight, self.gate_up_bias)
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        mlp_output = F.linear(activated, self.down_weight, self.down_bias)
        return hidden + mlp_output


def run_compiled_post_attention(
    plan,
    layer: torch.nn.Module,
    hidden: torch.Tensor,
    attn_output: torch.Tensor,
) -> torch.Tensor | None:
    """Run compiled post-attention block when enabled and available."""

    compiled = _get_compiled_post_attention(plan, layer, hidden)
    if compiled is None:
        return None
    try:
        mark_compile_step_begin()
        output = compiled.module(hidden, attn_output)
        return output.clone() if torch_compile_clone_outputs() else output
    except Exception as exc:  # pragma: no cover - depends on torch.compile backend
        setattr(plan, "_flashinfer_compile_post_disabled", True)
        WARNING(f"FlashInfer compiled post-attention disabled after runtime failure: {exc}")
        return None


def _get_compiled_post_attention(plan, layer: torch.nn.Module, hidden: torch.Tensor) -> CompiledPostAttention | None:
    if not torch_compile_post_enabled() or not can_compile_decode_tensor(hidden):
        return None
    if getattr(plan, "_flashinfer_compile_post_disabled", False):
        return None

    params = resolve_post_params(layer, plan)
    if params is None:
        return None
    cache_key = ("compiled_post", tuple(hidden.shape), hidden.dtype, hidden.device, params.cache_key)
    cached = getattr(plan, "_flashinfer_compiled_post", None)
    if isinstance(cached, CompiledPostAttention) and cached.cache_key == cache_key:
        return cached

    module = StaticPostAttentionCompute(params).eval()
    try:
        compiled_module = compile_module(module)
    except Exception as exc:  # pragma: no cover - depends on optional compiler backend
        setattr(plan, "_flashinfer_compile_post_disabled", True)
        WARNING(f"FlashInfer compiled post-attention disabled during compile: {exc}")
        return None
    compiled = CompiledPostAttention(module=compiled_module, cache_key=cache_key)
    setattr(plan, "_flashinfer_compiled_post", compiled)
    return compiled


def _rmsnorm_inline(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = hidden.dtype
    hidden_f = hidden.float()
    variance = hidden_f.pow(2).mean(dim=-1, keepdim=True)
    output = hidden_f * torch.rsqrt(variance + eps)
    return (output * weight).to(dtype)
