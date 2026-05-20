"""Packed projection helpers for the FlashInfer decode path."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import flashinfer
import torch
import torch.nn.functional as F

from model.flashinfer.effective_weights import EffectivePackedLinear, effective_linear, effective_packed_linear


@dataclass(frozen=True)
class PackedLinear:
    """Cached packed linear parameters."""

    weight: torch.Tensor
    bias: torch.Tensor | None
    out_features: tuple[int, ...]
    cache_key: tuple[object, ...]


@dataclass(frozen=True)
class PackedLoRADelta:
    """Cached packed LoRA delta parameters for grouped projections."""

    a: torch.Tensor
    b: torch.Tensor
    out_features: tuple[int, ...]
    cache_key: tuple[object, ...]


def packed_qkv_enabled() -> bool:
    return _env_enabled("MINI_OPENCLAW_FLASHINFER_PACKED_QKV", default=True)


def packed_mlp_enabled() -> bool:
    return _env_enabled("MINI_OPENCLAW_FLASHINFER_PACKED_MLP", default=True)


def packed_lora_delta_enabled() -> bool:
    return _env_enabled("MINI_OPENCLAW_FLASHINFER_PACKED_LORA_DELTA", default=True)


def fused_activation_enabled() -> bool:
    return _env_enabled("MINI_OPENCLAW_FLASHINFER_FUSED_ACT", default=True)


def shared_packed_base_weight_storage_enabled() -> bool:
    """Return whether packed base weights should replace original storages."""

    storage = os.getenv("MINI_OPENCLAW_FLASHINFER_PACKED_WEIGHT_STORAGE")
    if storage is not None:
        return storage.lower() in {"shared", "share", "alias", "shared_storage"}
    return _flashinfer_memory_mode() in {"balanced", "low_vram"}


def prepare_shared_packed_base_weights(
    layers: Sequence[torch.nn.Module],
    layer_plans: Sequence[object],
) -> dict[str, int]:
    """Materialize packed base weights early when shared storage mode is enabled."""

    stats = {"qkv": 0, "gate_up": 0}
    if not shared_packed_base_weight_storage_enabled():
        return stats

    for layer, plan in zip(layers, layer_plans):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and packed_qkv_enabled():
            modules = (
                getattr(plan, "modules", {}).get("self_attn.q_proj") or getattr(attn, "q_proj", None),
                getattr(plan, "modules", {}).get("self_attn.k_proj") or getattr(attn, "k_proj", None),
                getattr(plan, "modules", {}).get("self_attn.v_proj") or getattr(attn, "v_proj", None),
            )
            if all(module is not None for module in modules):
                packed = _cached_packed_linear(
                    owner=plan,
                    attr="_flashinfer_packed_qkv",
                    modules=modules,  # type: ignore[arg-type]
                )
                if packed is not None:
                    stats["qkv"] += 1

        mlp = getattr(layer, "mlp", None)
        if mlp is not None and packed_mlp_enabled():
            packed = get_base_gate_up_packed_linear(mlp)
            if packed is not None and isinstance(packed, PackedLinear):
                stats["gate_up"] += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return stats


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
    effective = effective_packed_linear(
        owner=plan,
        attr="_flashinfer_effective_qkv",
        modules=(q_module, k_module, v_module),
    )
    if effective is not None:
        q_dim, k_dim, v_dim = effective.out_features
        qkv = F.linear(hidden, effective.weight, effective.bias)
        return torch.split(qkv, [q_dim, k_dim, v_dim], dim=-1)

    packed = _cached_packed_linear(
        owner=plan,
        attr="_flashinfer_packed_qkv",
        modules=(q_module, k_module, v_module),
    )
    if packed is None:
        return None

    q_dim, k_dim, v_dim = packed.out_features
    qkv = F.linear(hidden, packed.weight, packed.bias)
    packed_delta = _packed_lora_delta(
        owner=plan,
        attr="_flashinfer_packed_qkv_lora_delta",
        modules=(q_module, k_module, v_module),
        hidden=hidden,
    )
    if packed_delta is not None:
        qkv = qkv + packed_delta
    q, k, v = torch.split(qkv, [q_dim, k_dim, v_dim], dim=-1)
    if packed_delta is not None:
        return q, k, v

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

    packed = get_base_gate_up_packed_linear(mlp)
    if packed is None:
        return None

    gate_dim, up_dim = packed.out_features
    gate_up = F.linear(hidden, packed.weight, packed.bias)
    if isinstance(packed, EffectivePackedLinear):
        return gate_up
    gate_module = getattr(mlp, "gate_proj", None)
    up_module = getattr(mlp, "up_proj", None)
    packed_delta = _packed_lora_delta(
        owner=mlp,
        attr="_flashinfer_packed_gate_up_lora_delta",
        modules=(gate_module, up_module),
        hidden=hidden,
    )
    if packed_delta is not None:
        return gate_up + packed_delta

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


def get_base_gate_up_packed_linear(mlp: torch.nn.Module) -> PackedLinear | EffectivePackedLinear | None:
    """Return packed gate/up parameters used by the eager and compiled paths."""

    if not packed_mlp_enabled():
        return None
    gate_module = getattr(mlp, "gate_proj", None)
    up_module = getattr(mlp, "up_proj", None)
    if gate_module is None or up_module is None:
        return None
    effective = effective_packed_linear(
        owner=mlp,
        attr="_flashinfer_effective_gate_up",
        modules=(gate_module, up_module),
    )
    if effective is not None:
        return effective

    return _cached_packed_linear(
        owner=mlp,
        attr="_flashinfer_packed_gate_up",
        modules=(gate_module, up_module),
    )


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
    return project_linear(down_module, hidden)


def project_linear(module: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Run one linear module with serve effective weights when available."""

    effective = effective_linear(module)
    if effective is not None:
        return F.linear(hidden, effective.weight, effective.bias)
    if _has_active_lora(module):
        return module(hidden)
    if _can_functional_linear(module):
        return F.linear(hidden, module.weight, module.bias)
    return module(hidden)


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

    share_storage = shared_packed_base_weight_storage_enabled() and _can_share_packed_linear_storage(modules)
    weights = [_linear_weight(module) for module in modules]
    biases = [_linear_bias(module) for module in modules]
    weight = torch.cat(weights, dim=0).contiguous()
    out_features = tuple(weight_tensor.size(0) for weight_tensor in weights)
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
    if share_storage:
        _share_packed_linear_storage(modules, weight, bias, out_features)
        cache_key = _packed_cache_key(modules)
    packed = PackedLinear(
        weight=weight,
        bias=bias,
        out_features=out_features,
        cache_key=cache_key,
    )
    setattr(owner, attr, packed)
    return packed


def _packed_lora_delta(
    *,
    owner: object,
    attr: str,
    modules: tuple[torch.nn.Module, ...],
    hidden: torch.Tensor,
) -> torch.Tensor | None:
    packed = _cached_packed_lora_delta(owner=owner, attr=attr, modules=modules)
    if packed is None:
        return None
    down = F.linear(hidden, packed.a, None)
    return F.linear(down, packed.b, None)


def _cached_packed_lora_delta(
    *,
    owner: object,
    attr: str,
    modules: tuple[torch.nn.Module, ...],
) -> PackedLoRADelta | None:
    if not packed_lora_delta_enabled():
        return None
    infos = [_lora_pack_info(module) for module in modules]
    if any(info is None for info in infos):
        return None
    lora_infos = [info for info in infos if info is not None]
    if not lora_infos:
        return None

    first_a, _first_b, _first_scale, _first_key = lora_infos[0]
    in_features = first_a.size(1)
    device = first_a.device
    dtype = first_a.dtype
    for a, b, _scale, _key in lora_infos:
        if a.ndim != 2 or b.ndim != 2:
            return None
        if a.size(1) != in_features or b.size(1) != a.size(0):
            return None
        if a.device != device or b.device != device or a.dtype != dtype or b.dtype != dtype:
            return None

    cache_key = ("packed_lora_delta", tuple(key for _a, _b, _scale, key in lora_infos))
    cached = getattr(owner, attr, None)
    if isinstance(cached, PackedLoRADelta) and cached.cache_key == cache_key:
        return cached

    ranks = tuple(a.size(0) for a, _b, _scale, _key in lora_infos)
    out_features = tuple(b.size(0) for _a, b, _scale, _key in lora_infos)
    packed_a = torch.cat([a.detach() for a, _b, _scale, _key in lora_infos], dim=0).contiguous()
    packed_b = torch.zeros(
        (sum(out_features), sum(ranks)),
        device=device,
        dtype=dtype,
    )
    out_offset = 0
    rank_offset = 0
    for (_a, b, scale, _key), out_dim, rank in zip(lora_infos, out_features, ranks):
        packed_b[out_offset : out_offset + out_dim, rank_offset : rank_offset + rank].copy_(
            b.detach() * scale
        )
        out_offset += out_dim
        rank_offset += rank

    packed = PackedLoRADelta(
        a=packed_a,
        b=packed_b.contiguous(),
        out_features=out_features,
        cache_key=cache_key,
    )
    setattr(owner, attr, packed)
    return packed


def _can_functional_linear(module: torch.nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    return isinstance(weight, torch.Tensor) and weight.ndim == 2


def _has_active_lora(module: torch.nn.Module) -> bool:
    return getattr(module, "active_role", "base") != "base"


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


def _can_share_packed_linear_storage(modules: Iterable[torch.nn.Module]) -> bool:
    for module in modules:
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 2:
            return False
        if weight.requires_grad:
            return False
        bias = getattr(module, "bias", None)
        if bias is not None:
            if not isinstance(bias, torch.nn.Parameter) or bias.ndim != 1:
                return False
            if bias.requires_grad:
                return False
    return True


def _share_packed_linear_storage(
    modules: tuple[torch.nn.Module, ...],
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: tuple[int, ...],
) -> None:
    weight_offset = 0
    for module, output_dim in zip(modules, out_features):
        old_weight = _linear_weight(module)
        shared_weight = weight[weight_offset : weight_offset + output_dim].view_as(old_weight)
        setattr(module, "weight", torch.nn.Parameter(shared_weight, requires_grad=False))
        weight_offset += output_dim

    biases = [_linear_bias(module) for module in modules]
    if bias is None or not all(bias_tensor is not None for bias_tensor in biases):
        return
    bias_offset = 0
    for module, output_dim in zip(modules, out_features):
        old_bias = _linear_bias(module)
        if old_bias is None:
            return
        shared_bias = bias[bias_offset : bias_offset + output_dim].view_as(old_bias)
        setattr(module, "bias", torch.nn.Parameter(shared_bias, requires_grad=False))
        bias_offset += output_dim


def _lora_pack_info(
    module: torch.nn.Module | None,
) -> tuple[torch.Tensor, torch.Tensor, float, tuple[object, ...]] | None:
    if module is None:
        return None
    active_role = getattr(module, "active_role", "base")
    if active_role != "serve":
        return None
    if not _lora_dropout_is_identity(module):
        return None
    a = getattr(module, "serve_a", None)
    b = getattr(module, "serve_b", None)
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return None
    if a.ndim != 2 or b.ndim != 2:
        return None
    scale = float(getattr(module, "scale", 1.0))
    cache_key = (
        id(module),
        active_role,
        _tensor_cache_key(a),
        _tensor_cache_key(b),
        scale,
        _dropout_cache_key(getattr(module, "dropout", None)),
    )
    return a, b, scale, cache_key


def _lora_dropout_is_identity(module: torch.nn.Module) -> bool:
    dropout = getattr(module, "dropout", None)
    if dropout is None:
        return True
    if isinstance(dropout, torch.nn.Identity):
        return True
    if isinstance(dropout, torch.nn.Dropout):
        return float(dropout.p) == 0.0 or not dropout.training
    return False


def _dropout_cache_key(dropout: object) -> tuple[object, ...] | None:
    if dropout is None:
        return None
    p = float(getattr(dropout, "p", 0.0))
    training = bool(getattr(dropout, "training", False))
    return (dropout.__class__.__name__, p, training)


def _tensor_cache_key(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
        int(getattr(tensor, "_version", 0)),
    )


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


def _flashinfer_memory_mode() -> str:
    return os.getenv("MINI_OPENCLAW_FLASHINFER_MEMORY_MODE", "throughput").lower()
