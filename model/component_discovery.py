"""Shared model structure discovery helpers.

The current main runtime only needs a *minimal* discovery path for manual LoRA
patch installation. Full-component discovery is kept only for legacy /
experimental dispatcher code paths and is no longer part of the supported main
execution flow.
"""

from __future__ import annotations

import inspect
from typing import Optional, Set

import torch.nn as nn
from core.utils import WARNING

# Candidate attribute paths for common decoder-only architectures.
LAYER_ATTR_PATHS = [
    ("model", "layers"),              # LLaMA, Qwen, Mistral, Gemma
    ("transformer", "h"),             # GPT-2, GPT-Neo
    ("model", "decoder", "layers"),   # T5 decoder variants
    ("decoder", "layers"),
    ("layers",),
]

EMBED_ATTR_PATHS = [
    ("model", "embed_tokens"),        # LLaMA, Qwen, Mistral
    ("transformer", "wte"),           # GPT-2
    ("model", "decoder", "embed_tokens"),
    ("embed_tokens",),
]

FINAL_NORM_PATHS = [
    ("model", "norm"),                # LLaMA, Qwen, Mistral
    ("transformer", "ln_f"),          # GPT-2
    ("model", "decoder", "final_layer_norm"),
    ("norm",),
]

HEAD_ATTR_PATHS = [
    ("lm_head",),
    ("transformer", "lm_head"),
]

ROTARY_ATTR_PATHS = [
    ("model", "rotary_emb"),
    ("rotary_emb",),
]


def _get_nested_attr(obj: object, path: tuple[str, ...]) -> Optional[object]:
    """Traverse a nested attribute path and return ``None`` on the first miss.

    A safe method to get obj.attr.attr... without risking an AttributeError. Returns None if any attribute does not exists. 
    """

    current = obj
    for attr in path:
        if not hasattr(current, attr):
            return None
        current = getattr(current, attr)
    return current


def _try_paths(obj: object, paths: list[tuple[str, ...]]) -> Optional[object]:
    """Return the first successfully resolved attribute path.
    
    Iterates through candidate attribute paths and returns the first one that resolves to a non-None value. If none of the paths resolve, returns None.
    """

    for path in paths:
        result = _get_nested_attr(obj, path)
        if result is not None:
            return result
    return None


def _resolve_peft_model(model: nn.Module) -> nn.Module:
    """Unwrap PEFT layers and return the underlying HF task model."""

    if hasattr(model, "base_model"):
        model = model.base_model
    if hasattr(model, "model"):
        inner = model.model
        if type(inner).__name__.endswith(("ForCausalLM", "ForSequenceClassification")):
            return inner
        if hasattr(inner, "model"):
            return inner.model
    return model


def discover_lora_patch_targets(model: nn.Module) -> dict[str, object]:
    """Discover only the structure needed for manual LoRA patching.

    The manual LoRA runtime only needs:

    - the unwrapped HF task model
    - its config
    - the decoder layer list

    It does *not* need embedding / final_norm / lm_head / rotary discovery.
    """

    hf_model = _resolve_peft_model(model)
    config = hf_model.config if hasattr(hf_model, "config") else None

    layers = _try_paths(hf_model, LAYER_ATTR_PATHS)
    if layers is None or not hasattr(layers, "__len__") or len(layers) == 0:
        raise AttributeError(
            f"Cannot find decoder layers in {type(hf_model).__name__}. "
            f"Tried: {LAYER_ATTR_PATHS}"
        )
    return {
        "layers": list(layers),
        "config": config,
        "hf_task_model": hf_model,
    }


def discover_generation_components(model: nn.Module) -> dict[str, object]:
    """Discover the full generation component set for legacy dispatcher paths.

    WARNING:
    This path is no longer part of the supported main runtime. It is kept only
    for old experimental code such as model offload dispatchers.
    """

    WARNING(
        "discover_generation_components() is a legacy experimental path. "
        "The current supported runtime no longer depends on full model "
        "component discovery."
    )

    hf_model = _resolve_peft_model(model)
    config = hf_model.config if hasattr(hf_model, "config") else None

    layers = _try_paths(hf_model, LAYER_ATTR_PATHS)
    if layers is None or not hasattr(layers, "__len__") or len(layers) == 0:
        raise AttributeError(
            f"Cannot find decoder layers in {type(hf_model).__name__}. "
            f"Tried: {LAYER_ATTR_PATHS}"
        )
    layers = list(layers)

    embedding = _try_paths(hf_model, EMBED_ATTR_PATHS)
    if embedding is None:
        # Fallback for architectures that do not match the common attribute
        # paths but still expose a single token embedding module.
        for _, module in hf_model.named_modules():
            if isinstance(module, nn.Embedding):
                embedding = module
                break
    if embedding is None:
        raise AttributeError("Cannot find embedding layer")

    final_norm = _try_paths(hf_model, FINAL_NORM_PATHS)

    lm_head = _try_paths(hf_model, HEAD_ATTR_PATHS)
    if lm_head is None:
        raise AttributeError("Cannot find lm_head")

    rotary_emb = _try_paths(hf_model, ROTARY_ATTR_PATHS)

    return {
        "layers": layers,
        "embedding": embedding,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "rotary_emb": rotary_emb,
        "config": config,
        "hf_task_model": hf_model,
    }


def discover_components(model: nn.Module) -> dict[str, object]:
    """Backward-compatible alias for full component discovery."""

    return discover_generation_components(model)


def introspect_layer_forward(layer: nn.Module) -> Set[str]:
    """Return the set of kwargs accepted by a decoder layer forward."""

    try:
        sig = inspect.signature(layer.forward)
        return set(sig.parameters.keys())
    except (ValueError, TypeError):
        return set()


def preserve_attn_implementation(layer: nn.Module, model_config: object) -> None:
    """Restore attention implementation flags after device transfers.

    Some HF modules may lose their internal attention implementation marker
    after ``.to(device)``. Re-applying it keeps dispatcher execution aligned
    with the model's original configuration.
    """

    attn_impl = getattr(model_config, "_attn_implementation", None)
    if attn_impl is None:
        return

    for _, module in layer.named_modules():
        if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
            module.config._attn_implementation = attn_impl
        if hasattr(module, "_attn_implementation"):
            module._attn_implementation = attn_impl


__all__ = [
    "EMBED_ATTR_PATHS",
    "FINAL_NORM_PATHS",
    "HEAD_ATTR_PATHS",
    "LAYER_ATTR_PATHS",
    "ROTARY_ATTR_PATHS",
    "discover_components",
    "discover_generation_components",
    "discover_lora_patch_targets",
    "introspect_layer_forward",
    "preserve_attn_implementation",
]
