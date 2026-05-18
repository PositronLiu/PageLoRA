"""Packed projection helpers for the FlashInfer decode path."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import flashinfer
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PackedLinear:
    """Cached packed linear parameters."""

    weight: torch.Tensor
    bias: torch.Tensor | None
    out_features: tuple[int, ...]
    cache_key: tuple[object, ...]


def packed_qkv_enabled() -> bool:
    return _env_enabled("MINI_OPENCLAW_FLASHINFER_PACKED_QKV", default=True)


def packed_mlp_enabled() -> bool:
    return _env_enabled("MINI_OPENCLAW_FLASHINFER_PACKED_MLP", default=True)


def fused_activation_enabled() -> bool:
    return _env_enabled("MINI_OPENCLAW_FLASHINFER_FUSED_ACT", default=True)


def project_qkv_packed(
    layer: torch.nn.Module,
    plan,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Project Q/K/V with one packed base linear plus optional LoRA deltas.

    Returns ``None`` when the modules cannot be packed safely, allowing callers
    to fall back to the original three-projection path.
    """

    if not packed_qkv_enabled():
        return None
    attn = layer.self_attn
    q_module = plan.modules.get("self_attn.q_proj") or attn.q_proj
    k_module = plan.modules.get("self_attn.k_proj") or attn.k_proj
    v_module = plan.modules.get("self_attn.v_proj") or attn.v_proj
    packed = _cached_packed_linear(
        owner=plan,
        attr="_flashinfer_packed_qkv",
        modules=(q_module, k_module, v_module),
    )
    if packed is None:
        return None

    q_dim, k_dim, v_dim = packed.out_features
    qkv = F.linear(hidden, packed.weight, packed.bias)
    q, k, v = torch.split(qkv, [q_dim, k_dim, v_dim], dim=-1)
    q_delta = _lora_delta(q_module, hidden)
    if q_delta is not None:
        q = q + q_delta
    k_delta = _lora_delta(k_module, hidden)
    if k_delta is not None:
        k = k + k_delta
    v_delta = _lora_delta(v_module, hidden)
    if v_delta is not None:
        v = v + v_delta
    return q, k, v


def project_gate_up_packed(
    mlp: torch.nn.Module,
    hidden: torch.Tensor,
) -> torch.Tensor | None:
    """Project MLP gate/up with one packed base linear plus LoRA deltas."""

    if not packed_mlp_enabled():
        return None
    gate_module = getattr(mlp, "gate_proj", None)
    up_module = getattr(mlp, "up_proj", None)
    if gate_module is None or up_module is None:
        return None
    packed = _cached_packed_linear(
        owner=mlp,
        attr="_flashinfer_packed_gate_up",
        modules=(gate_module, up_module),
    )
    if packed is None:
        return None

    gate_dim, up_dim = packed.out_features
    gate_up = F.linear(hidden, packed.weight, packed.bias)
    gate_delta = _lora_delta(gate_module, hidden)
    up_delta = _lora_delta(up_module, hidden)
    if gate_delta is None and up_delta is None:
        return gate_up

    gate, up = torch.split(gate_up, [gate_dim, up_dim], dim=-1)
    if gate_delta is not None:
        gate = gate + gate_delta
    if up_delta is not None:
        up = up + up_delta
    return torch.cat([gate, up], dim=-1)


def apply_gate_up_activation(
    gate_up: torch.Tensor,
    act_fn,
) -> torch.Tensor:
    """Apply SwiGLU activation, using FlashInfer's fused op when possible."""

    if fused_activation_enabled() and _is_silu_activation(act_fn) and _can_use_flashinfer_silu(gate_up):
        return flashinfer.silu_and_mul(gate_up)
    gate, up = gate_up.chunk(2, dim=-1)
    return act_fn(gate) * up


def project_down(mlp: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Run the MLP down projection while preserving optional LoRA wrappers."""

    down_module = getattr(mlp, "down_proj", None)
    if down_module is None:
        raise RuntimeError("MLP module does not expose down_proj")
    if _can_functional_linear(down_module):
        return F.linear(hidden, down_module.weight, down_module.bias)
    return down_module(hidden)


def _cached_packed_linear(
    *,
    owner: object,
    attr: str,
    modules: tuple[torch.nn.Module, ...],
) -> PackedLinear | None:
    if not all(_can_functional_linear(module) for module in modules):
        return None
    cache_key = _packed_cache_key(modules)
    cached = getattr(owner, attr, None)
    if isinstance(cached, PackedLinear) and cached.cache_key == cache_key:
        return cached

    weights = [_linear_weight(module) for module in modules]
    biases = [_linear_bias(module) for module in modules]
    weight = torch.cat(weights, dim=0).contiguous()
    bias: torch.Tensor | None
    if any(bias_tensor is not None for bias_tensor in biases):
        bias_pieces = [
            bias_tensor
            if bias_tensor is not None
            else torch.zeros(
                _linear_weight(module).size(0),
                device=weight.device,
                dtype=weight.dtype,
            )
            for module, bias_tensor in zip(modules, biases)
        ]
        bias = torch.cat(bias_pieces, dim=0).contiguous()
    else:
        bias = None
    packed = PackedLinear(
        weight=weight,
        bias=bias,
        out_features=tuple(weight_tensor.size(0) for weight_tensor in weights),
        cache_key=cache_key,
    )
    setattr(owner, attr, packed)
    return packed


def _can_functional_linear(module: torch.nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    return isinstance(weight, torch.Tensor) and weight.ndim == 2


def _linear_weight(module: torch.nn.Module) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError("module does not expose a 2D linear weight")
    return weight


def _linear_bias(module: torch.nn.Module) -> torch.Tensor | None:
    bias = getattr(module, "bias", None)
    if bias is None:
        return None
    if not isinstance(bias, torch.Tensor) or bias.ndim != 1:
        raise TypeError("module exposes an invalid linear bias")
    return bias


def _packed_cache_key(modules: Iterable[torch.nn.Module]) -> tuple[object, ...]:
    pieces: list[object] = []
    for module in modules:
        weight = _linear_weight(module)
        bias = _linear_bias(module)
        pieces.extend(
            [
                id(module),
                weight.data_ptr(),
                tuple(weight.shape),
                weight.dtype,
                weight.device,
            ]
        )
        if bias is None:
            pieces.append(None)
        else:
            pieces.extend([bias.data_ptr(), tuple(bias.shape), bias.dtype, bias.device])
    return tuple(pieces)


def _lora_delta(module: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor | None:
    active_role = getattr(module, "active_role", "base")
    if active_role == "base":
        return None
    if active_role == "serve":
        a = getattr(module, "serve_a", None)
        b = getattr(module, "serve_b", None)
    else:
        a = getattr(module, "train_a", None)
        b = getattr(module, "train_b", None)
    if a is None or b is None:
        return None

    dropout = getattr(module, "dropout", None)
    dropped = dropout(hidden) if callable(dropout) else hidden
    down = F.linear(dropped, a, None)
    scale = float(getattr(module, "scale", 1.0))
    return F.linear(down, b, None) * scale


def _is_silu_activation(act_fn) -> bool:
    name = act_fn.__class__.__name__.lower()
    if name in {"silu", "siluactivation", "swishactivation"}:
        return True
    fn_name = getattr(act_fn, "__name__", "").lower()
    return fn_name in {"silu", "swish"}


def _can_use_flashinfer_silu(gate_up: torch.Tensor) -> bool:
    if gate_up.device.type != "cuda":
        return False
    if not hasattr(flashinfer, "silu_and_mul"):
        return False
    return gate_up.shape[-1] * gate_up.dtype.itemsize % 16 == 0


def _env_enabled(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
