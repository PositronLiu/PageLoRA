"""Normalization helpers for the FlashInfer generation path."""

from __future__ import annotations

import os

import flashinfer
import torch


def flashinfer_rmsnorm_enabled() -> bool:
    return _env_enabled("MINI_OPENCLAW_FLASHINFER_RMSNORM", default=True)


def apply_rmsnorm(norm: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Apply a model RMSNorm, using FlashInfer when the module is compatible."""

    if flashinfer_rmsnorm_enabled() and _can_use_flashinfer_rmsnorm(hidden, norm):
        flat = hidden.reshape(-1, hidden.size(-1))
        if not flat.is_contiguous():
            flat = flat.contiguous()
        out = _flashinfer_rmsnorm(flat, norm)
        return out.view_as(hidden)
    return norm(hidden)


def _flashinfer_rmsnorm(flat: torch.Tensor, norm: torch.nn.Module) -> torch.Tensor:
    weight = getattr(norm, "weight")
    eps = _norm_eps(norm)
    if _is_gemma_rmsnorm(norm) and hasattr(flashinfer.norm, "gemma_rmsnorm"):
        return flashinfer.norm.gemma_rmsnorm(flat, weight, eps=eps)
    return flashinfer.norm.rmsnorm(flat, weight, eps=eps)


def _can_use_flashinfer_rmsnorm(hidden: torch.Tensor, norm: torch.nn.Module) -> bool:
    if not _is_rmsnorm_like(norm):
        return False
    weight = getattr(norm, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return False
    if hidden.ndim < 2 or weight.ndim != 1 or weight.numel() != hidden.size(-1):
        return False
    if hidden.device.type != "cuda" or weight.device.type != "cuda":
        return False
    flashinfer_norm = getattr(flashinfer, "norm", None)
    if flashinfer_norm is None or not hasattr(flashinfer_norm, "rmsnorm"):
        return False
    if _is_gemma_rmsnorm(norm) and not hasattr(flashinfer_norm, "gemma_rmsnorm"):
        return False
    return True


def _is_gemma_rmsnorm(norm: torch.nn.Module) -> bool:
    return "gemma" in norm.__class__.__name__.lower()


def _is_rmsnorm_like(norm: torch.nn.Module) -> bool:
    return "rmsnorm" in norm.__class__.__name__.lower()


def _norm_eps(norm: torch.nn.Module) -> float:
    return float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))


def _env_enabled(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
