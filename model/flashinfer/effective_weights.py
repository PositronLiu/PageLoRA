"""Serve-time effective linear weights for FlashInfer inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class EffectiveLinear:
    """A cached linear weight with the active serve LoRA delta merged in."""

    weight: torch.Tensor
    bias: torch.Tensor | None
    cache_key: tuple[object, ...]


@dataclass(frozen=True)
class EffectivePackedLinear:
    """Cached packed effective linear parameters."""

    weight: torch.Tensor
    bias: torch.Tensor | None
    out_features: tuple[int, ...]
    cache_key: tuple[object, ...]


def effective_weights_enabled() -> bool:
    memory_mode = os.getenv("MINI_OPENCLAW_FLASHINFER_MEMORY_MODE", "throughput").lower()
    storage = os.getenv("MINI_OPENCLAW_FLASHINFER_PACKED_WEIGHT_STORAGE", "").lower()
    low_vram_default = memory_mode in {"balanced", "low_vram"} or storage in {
        "shared",
        "share",
        "alias",
        "shared_storage",
    }
    return _env_enabled(
        "MINI_OPENCLAW_FLASHINFER_EFFECTIVE_WEIGHTS",
        default=not low_vram_default,
    )


def effective_linear(module: torch.nn.Module) -> EffectiveLinear | None:
    """Return a serve-mode effective linear weight for a LoRA-wrapped module.

    This intentionally only handles frozen serve adapters. Training adapters
    need the original LoRA path so gradients can flow into A/B parameters.
    """

    if not effective_weights_enabled() or not _can_merge_serve_lora(module):
        return None
    cache_key = _effective_linear_cache_key(module)
    cached = getattr(module, "_flashinfer_effective_linear", None)
    if isinstance(cached, EffectiveLinear) and cached.cache_key == cache_key:
        return cached

    weight = _linear_weight(module)
    bias = _linear_bias(module)
    scale = float(getattr(module, "scale", 1.0))
    with torch.no_grad():
        delta = _lora_weight_delta(module)
        merged = (weight.detach() + delta * scale).contiguous()
    effective = EffectiveLinear(weight=merged, bias=bias, cache_key=cache_key)
    setattr(module, "_flashinfer_effective_linear", effective)
    return effective


def effective_packed_linear(
    *,
    owner: object,
    attr: str,
    modules: tuple[torch.nn.Module, ...],
) -> EffectivePackedLinear | None:
    """Return packed effective weights for a group of serve LoRA linears."""

    if not effective_weights_enabled() or not all(_can_merge_serve_lora(module) for module in modules):
        return None
    if not _can_pack_linears(modules):
        return None
    cache_key: tuple[object, ...] = tuple(_effective_linear_cache_key(module) for module in modules)
    cached = getattr(owner, attr, None)
    if isinstance(cached, EffectivePackedLinear) and cached.cache_key == cache_key:
        return cached

    first_weight = _linear_weight(modules[0])
    out_features = tuple(_linear_weight(module).size(0) for module in modules)
    weight = torch.empty(
        (sum(out_features), first_weight.size(1)),
        device=first_weight.device,
        dtype=first_weight.dtype,
    )
    biases = [_linear_bias(module) for module in modules]
    packed_bias = (
        torch.empty(sum(out_features), device=first_weight.device, dtype=first_weight.dtype)
        if any(bias is not None for bias in biases)
        else None
    )
    with torch.no_grad():
        offset = 0
        for module, bias, output_dim in zip(modules, biases, out_features):
            target = weight[offset : offset + output_dim]
            target.copy_(_linear_weight(module).detach())
            target.add_(_lora_weight_delta(module), alpha=float(getattr(module, "scale", 1.0)))
            if packed_bias is not None:
                if bias is None:
                    packed_bias[offset : offset + output_dim].zero_()
                else:
                    packed_bias[offset : offset + output_dim].copy_(
                        bias.detach().to(device=first_weight.device, dtype=first_weight.dtype)
                    )
            offset += output_dim
    packed = EffectivePackedLinear(
        weight=weight,
        bias=packed_bias,
        out_features=out_features,
        cache_key=cache_key,
    )
    setattr(owner, attr, packed)
    return packed


def clear_flashinfer_effective_weight_cache(*roots: object) -> None:
    """Drop cached effective weights below the given objects."""

    seen: set[int] = set()
    for root in roots:
        _clear_effective_attrs(root, seen)


def _can_merge_serve_lora(module: torch.nn.Module) -> bool:
    if getattr(module, "active_role", "base") != "serve":
        return False
    if not _can_functional_linear(module):
        return False
    a = getattr(module, "serve_a", None)
    b = getattr(module, "serve_b", None)
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return False
    weight = _linear_weight(module)
    if a.ndim != 2 or b.ndim != 2:
        return False
    if a.device != weight.device or b.device != weight.device:
        return False
    if a.size(1) != weight.size(1) or b.size(0) != weight.size(0) or b.size(1) != a.size(0):
        return False
    dropout = getattr(module, "dropout", None)
    if module.training and not isinstance(dropout, nn.Identity):
        return False
    return True


def _effective_linear_cache_key(module: torch.nn.Module) -> tuple[object, ...]:
    weight = _linear_weight(module)
    bias = _linear_bias(module)
    a = getattr(module, "serve_a")
    b = getattr(module, "serve_b")
    pieces: list[object] = [
        id(module),
        "serve",
        weight.data_ptr(),
        tuple(weight.shape),
        weight.dtype,
        weight.device,
        _tensor_version(weight),
        a.data_ptr(),
        tuple(a.shape),
        a.dtype,
        a.device,
        _tensor_version(a),
        b.data_ptr(),
        tuple(b.shape),
        b.dtype,
        b.device,
        _tensor_version(b),
        float(getattr(module, "scale", 1.0)),
    ]
    if bias is None:
        pieces.append(None)
    else:
        pieces.extend([bias.data_ptr(), tuple(bias.shape), bias.dtype, bias.device, _tensor_version(bias)])
    return tuple(pieces)


def _can_pack_linears(modules: tuple[torch.nn.Module, ...]) -> bool:
    if not modules:
        return False
    first = _linear_weight(modules[0])
    for module in modules[1:]:
        weight = _linear_weight(module)
        if weight.size(1) != first.size(1) or weight.device != first.device or weight.dtype != first.dtype:
            return False
    return True


def _lora_weight_delta(module: torch.nn.Module) -> torch.Tensor:
    weight = _linear_weight(module)
    a = getattr(module, "serve_a")
    b = getattr(module, "serve_b")
    return torch.matmul(
        b.detach().to(device=weight.device, dtype=weight.dtype),
        a.detach().to(device=weight.device, dtype=weight.dtype),
    )


def _clear_effective_attrs(obj: object, seen: set[int]) -> None:
    if obj is None:
        return
    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if hasattr(obj, "__dict__"):
        for name in list(vars(obj).keys()):
            if name.startswith("_flashinfer_effective_"):
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass

    if isinstance(obj, nn.Module):
        for child in obj.children():
            _clear_effective_attrs(child, seen)
        return
    if isinstance(obj, dict):
        values: Iterable[object] = obj.values()
    elif isinstance(obj, (list, tuple, set)):
        values = obj
    else:
        values = ()
    for value in values:
        _clear_effective_attrs(value, seen)


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


def _tensor_version(tensor: torch.Tensor) -> int:
    return int(getattr(tensor, "_version", 0))


def _env_enabled(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
