"""Static tensor parameter resolution for compiled FlashInfer decode blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from model.flashinfer.effective_weights import EffectivePackedLinear, effective_linear
from model.flashinfer.packed_ops import get_base_gate_up_packed_linear, packed_mlp_enabled


@dataclass(frozen=True)
class StaticLinearParams:
    weight: torch.Tensor
    bias: torch.Tensor | None
    cache_key: tuple[object, ...]


@dataclass(frozen=True)
class StaticPackedLinearParams:
    weight: torch.Tensor
    bias: torch.Tensor | None
    out_features: tuple[int, ...]
    cache_key: tuple[object, ...]


@dataclass(frozen=True)
class StaticMLPParams:
    gate_up: StaticPackedLinearParams
    down: StaticLinearParams
    cache_key: tuple[object, ...]


@dataclass(frozen=True)
class StaticNormParams:
    weight: torch.Tensor
    eps: float
    cache_key: tuple[object, ...]


@dataclass(frozen=True)
class StaticPostParams:
    o_proj: StaticLinearParams
    post_norm: StaticNormParams
    mlp: StaticMLPParams
    cache_key: tuple[object, ...]


def resolve_linear_params(module: torch.nn.Module) -> StaticLinearParams | None:
    """Resolve a linear module into static tensors, preserving LoRA train fallback."""

    active_role = getattr(module, "active_role", "base")
    if active_role == "serve":
        effective = effective_linear(module)
        if effective is None:
            return None
        return StaticLinearParams(
            weight=effective.weight,
            bias=effective.bias,
            cache_key=("effective_linear", effective.cache_key),
        )
    if active_role != "base":
        return None
    if not _can_functional_linear(module):
        return None
    weight = _linear_weight(module)
    bias = _linear_bias(module)
    return StaticLinearParams(
        weight=weight,
        bias=bias,
        cache_key=("linear", _linear_cache_key(module)),
    )


def resolve_mlp_params(mlp: torch.nn.Module) -> StaticMLPParams | None:
    """Resolve MLP gate/up/down tensors for a static compiled decode path."""

    if not packed_mlp_enabled():
        return None
    if not _is_silu_activation(getattr(mlp, "act_fn", None)):
        return None
    gate_module = getattr(mlp, "gate_proj", None)
    up_module = getattr(mlp, "up_proj", None)
    down_module = getattr(mlp, "down_proj", None)
    if gate_module is None or up_module is None or down_module is None:
        return None

    packed = get_base_gate_up_packed_linear(mlp)
    if packed is None:
        return None
    if (_has_active_lora(gate_module) or _has_active_lora(up_module)) and not isinstance(packed, EffectivePackedLinear):
        return None
    down = resolve_linear_params(down_module)
    if down is None:
        return None

    gate_up = StaticPackedLinearParams(
        weight=packed.weight,
        bias=packed.bias,
        out_features=packed.out_features,
        cache_key=(packed.__class__.__name__, packed.cache_key),
    )
    cache_key = ("mlp", gate_up.cache_key, down.cache_key)
    return StaticMLPParams(gate_up=gate_up, down=down, cache_key=cache_key)


def resolve_post_params(layer: torch.nn.Module, plan) -> StaticPostParams | None:
    """Resolve post-attention tensors for static compiled decode."""

    o_module = plan.modules.get("self_attn.o_proj") or layer.self_attn.o_proj
    o_proj = resolve_linear_params(o_module)
    post_norm = resolve_norm_params(layer.post_attention_layernorm)
    mlp = resolve_mlp_params(layer.mlp)
    if o_proj is None or post_norm is None or mlp is None:
        return None
    return StaticPostParams(
        o_proj=o_proj,
        post_norm=post_norm,
        mlp=mlp,
        cache_key=("post", o_proj.cache_key, post_norm.cache_key, mlp.cache_key),
    )


def resolve_norm_params(norm: torch.nn.Module) -> StaticNormParams | None:
    if not _is_rmsnorm_like(norm):
        return None
    weight = getattr(norm, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 1:
        return None
    return StaticNormParams(
        weight=weight,
        eps=_norm_eps(norm),
        cache_key=("rmsnorm", _tensor_cache_key(weight), _norm_eps(norm), norm.__class__.__name__),
    )


def _has_active_lora(module: torch.nn.Module) -> bool:
    return getattr(module, "active_role", "base") != "base"


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


def _linear_cache_key(module: torch.nn.Module) -> tuple[object, ...]:
    weight = _linear_weight(module)
    bias = _linear_bias(module)
    pieces: list[object] = [id(module), _tensor_cache_key(weight)]
    if bias is None:
        pieces.append(None)
    else:
        pieces.append(_tensor_cache_key(bias))
    return tuple(pieces)


def _tensor_cache_key(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
        int(getattr(tensor, "_version", 0)),
    )


def _is_silu_activation(act_fn) -> bool:
    if act_fn is None:
        return False
    name = act_fn.__class__.__name__.lower()
    if name in {"silu", "siluactivation", "swishactivation"}:
        return True
    fn_name = getattr(act_fn, "__name__", "").lower()
    return fn_name in {"silu", "swish"}


def _is_rmsnorm_like(norm: torch.nn.Module) -> bool:
    return "rmsnorm" in norm.__class__.__name__.lower()


def _norm_eps(norm: torch.nn.Module) -> float:
    return float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
