"""torch.compile helpers for the FlashInfer decode MLP path."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from core.utils import WARNING
from model.flashinfer.compile_config import (
    can_compile_decode_tensor,
    compile_module,
    torch_compile_mlp_enabled,
)
from model.flashinfer.static_params import StaticMLPParams, resolve_mlp_params


@dataclass(frozen=True)
class CompiledMLP:
    module: torch.nn.Module
    cache_key: tuple[object, ...]


class StaticMLPCompute(torch.nn.Module):
    """Tensor-only MLP compute for single-token decode."""

    def __init__(self, params: StaticMLPParams) -> None:
        super().__init__()
        self.gate_up_weight = params.gate_up.weight
        self.gate_up_bias = params.gate_up.bias
        self.down_weight = params.down.weight
        self.down_bias = params.down.bias

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate_up = F.linear(hidden, self.gate_up_weight, self.gate_up_bias)
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        return F.linear(activated, self.down_weight, self.down_bias)


def run_compiled_mlp(plan, mlp: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor | None:
    """Run compiled MLP when enabled and available, else return None."""

    compiled = _get_compiled_mlp(plan, mlp, hidden)
    if compiled is None:
        return None
    try:
        return compiled.module(hidden)
    except Exception as exc:  # pragma: no cover - depends on torch.compile backend
        setattr(plan, "_flashinfer_compile_mlp_disabled", True)
        WARNING(f"FlashInfer compiled MLP disabled after runtime failure: {exc}")
        return None


def _get_compiled_mlp(plan, mlp: torch.nn.Module, hidden: torch.Tensor) -> CompiledMLP | None:
    if not torch_compile_mlp_enabled() or not can_compile_decode_tensor(hidden):
        return None
    if getattr(plan, "_flashinfer_compile_mlp_disabled", False):
        return None

    params = resolve_mlp_params(mlp)
    if params is None:
        return None
    cache_key = ("compiled_mlp", tuple(hidden.shape), hidden.dtype, hidden.device, params.cache_key)
    cached = getattr(plan, "_flashinfer_compiled_mlp", None)
    if isinstance(cached, CompiledMLP) and cached.cache_key == cache_key:
        return cached

    module = StaticMLPCompute(params).eval()
    try:
        compiled_module = compile_module(module)
    except Exception as exc:  # pragma: no cover - depends on optional compiler backend
        setattr(plan, "_flashinfer_compile_mlp_disabled", True)
        WARNING(f"FlashInfer compiled MLP disabled during compile: {exc}")
        return None
    compiled = CompiledMLP(module=compiled_module, cache_key=cache_key)
    setattr(plan, "_flashinfer_compiled_mlp", compiled)
    return compiled
