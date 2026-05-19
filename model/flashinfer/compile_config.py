"""Configuration and cache helpers for FlashInfer torch.compile paths."""

from __future__ import annotations

import os
from typing import Iterable

import torch


def torch_compile_enabled() -> bool:
    return _env_enabled("MINI_OPENCLAW_FLASHINFER_TORCH_COMPILE", default=False)


def torch_compile_mlp_enabled() -> bool:
    return _env_enabled(
        "MINI_OPENCLAW_FLASHINFER_COMPILE_MLP",
        default=torch_compile_enabled(),
    )


def torch_compile_post_enabled() -> bool:
    return _env_enabled(
        "MINI_OPENCLAW_FLASHINFER_COMPILE_POST",
        default=torch_compile_enabled(),
    )


def torch_compile_mode() -> str:
    return os.getenv("MINI_OPENCLAW_FLASHINFER_COMPILE_MODE", "reduce-overhead")


def torch_compile_fullgraph() -> bool:
    return _env_enabled("MINI_OPENCLAW_FLASHINFER_COMPILE_FULLGRAPH", default=False)


def torch_compile_cudagraphs() -> bool:
    return _env_enabled("MINI_OPENCLAW_FLASHINFER_COMPILE_CUDAGRAPHS", default=False)


def torch_compile_clone_outputs() -> bool:
    return _env_enabled(
        "MINI_OPENCLAW_FLASHINFER_COMPILE_CLONE_OUTPUTS",
        default=torch_compile_cudagraphs(),
    )


def can_compile_decode_tensor(hidden: torch.Tensor) -> bool:
    return hidden.device.type == "cuda" and hidden.ndim == 3 and hidden.size(0) == 1 and hidden.size(1) == 1


def mark_compile_step_begin() -> None:
    if not torch_compile_cudagraphs():
        return
    compiler = getattr(torch, "compiler", None)
    marker = getattr(compiler, "cudagraph_mark_step_begin", None) if compiler is not None else None
    if callable(marker):
        marker()


def compile_module(module: torch.nn.Module) -> torch.nn.Module:
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable")
    kwargs: dict[str, object] = {"fullgraph": torch_compile_fullgraph()}
    if torch_compile_cudagraphs():
        kwargs["mode"] = torch_compile_mode()
    else:
        kwargs["options"] = {"triton.cudagraphs": False}
    return torch.compile(module, **kwargs)


def clear_flashinfer_compile_cache(*roots: object) -> None:
    """Drop cached compiled modules and compile failure markers below roots."""

    seen: set[int] = set()
    for root in roots:
        _clear_compile_attrs(root, seen)


def _clear_compile_attrs(obj: object, seen: set[int]) -> None:
    if obj is None:
        return
    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if hasattr(obj, "__dict__"):
        for name in list(vars(obj).keys()):
            if name.startswith("_flashinfer_compil"):
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass

    if isinstance(obj, torch.nn.Module):
        for child in obj.children():
            _clear_compile_attrs(child, seen)
        return
    if isinstance(obj, dict):
        values: Iterable[object] = obj.values()
    elif isinstance(obj, (list, tuple, set)):
        values = obj
    else:
        values = ()
    for value in values:
        _clear_compile_attrs(value, seen)


def _env_enabled(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
