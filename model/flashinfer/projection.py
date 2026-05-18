"""Attention projection helpers for custom generation paths."""

from __future__ import annotations

import os
from dataclasses import dataclass

import flashinfer
import torch

from core.utils import WARNING
from model.flashinfer.packed_ops import project_qkv_packed


@dataclass(frozen=True)
class ProjectedQKV:
    """Q/K/V after projection and optional per-head Q/K normalization."""

    q_bshd: torch.Tensor
    k_bshd: torch.Tensor
    v_bshd: torch.Tensor
    v_bhsd: torch.Tensor


@dataclass(frozen=True)
class AttentionProjectionOutput:
    """Q/K/V tensors ready for paged attention and cache writes."""

    q_attn_bhsd: torch.Tensor
    k_store_bhsd: torch.Tensor
    v_store_bhsd: torch.Tensor


def project_qkv_with_optional_norm(
    layer: torch.nn.Module,
    plan,
    normed_hidden: torch.Tensor,
    *,
    use_flashinfer_norm: bool | None = None,
) -> ProjectedQKV:
    """Project Q/K/V and apply optional model-specific Q/K head norms.

    Qwen3 applies ``q_norm`` and ``k_norm`` after reshaping projections to
    ``[batch, seq, heads, head_dim]``. Other model families often omit these
    attributes, so this helper discovers them at runtime instead of branching
    on model names.
    """

    attn = layer.self_attn
    if use_flashinfer_norm is None:
        use_flashinfer_norm = os.getenv("MINI_OPENCLAW_FLASHINFER_QK_NORM", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    config = attn.config
    num_qo_heads = int(config.num_attention_heads)
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_qo_heads))
    head_dim = int(attn.head_dim)

    bsz, seqlen, _ = normed_hidden.shape
    packed_qkv = project_qkv_packed(layer, plan, normed_hidden)
    if packed_qkv is None:
        q_module = plan.modules.get("self_attn.q_proj") or attn.q_proj
        k_module = plan.modules.get("self_attn.k_proj") or attn.k_proj
        v_module = plan.modules.get("self_attn.v_proj") or attn.v_proj

        q_bshd = _project_with_optional_head_norm(
            projection=q_module,
            norm=getattr(attn, "q_norm", None),
            hidden=normed_hidden,
            num_heads=num_qo_heads,
            head_dim=head_dim,
            use_flashinfer_norm=use_flashinfer_norm,
        )
        k_bshd = _project_with_optional_head_norm(
            projection=k_module,
            norm=getattr(attn, "k_norm", None),
            hidden=normed_hidden,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            use_flashinfer_norm=use_flashinfer_norm,
        )
        v_bshd = v_module(normed_hidden).view(bsz, seqlen, num_kv_heads, head_dim)
    else:
        q_flat, k_flat, v_flat = packed_qkv
        q_bshd = q_flat.view(bsz, seqlen, num_qo_heads, head_dim)
        k_bshd = k_flat.view(bsz, seqlen, num_kv_heads, head_dim)
        q_norm = getattr(attn, "q_norm", None)
        if q_norm is not None:
            q_bshd = _apply_optional_head_norm(
                q_bshd,
                q_norm,
                use_flashinfer_norm=use_flashinfer_norm,
            )
        k_norm = getattr(attn, "k_norm", None)
        if k_norm is not None:
            k_bshd = _apply_optional_head_norm(
                k_bshd,
                k_norm,
                use_flashinfer_norm=use_flashinfer_norm,
            )
        v_bshd = v_flat.view(bsz, seqlen, num_kv_heads, head_dim)
    return ProjectedQKV(
        q_bshd=q_bshd,
        k_bshd=k_bshd,
        v_bshd=v_bshd,
        v_bhsd=v_bshd.transpose(1, 2).contiguous(),
    )


def apply_flashinfer_rope_pos_ids(
    layer: torch.nn.Module,
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply FlashInfer RoPE and return Q/K in ``[B, H, S, D]`` layout."""

    q_nhd, k_nhd, bsz, seqlen, num_qo_heads, num_kv_heads, head_dim = _apply_flashinfer_rope_pos_ids_nhd(
        layer,
        q_bshd,
        k_bshd,
        position_ids,
    )
    q_bhsd = q_nhd.view(bsz, seqlen, num_qo_heads, head_dim).transpose(1, 2).contiguous()
    k_bhsd = k_nhd.view(bsz, seqlen, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    return q_bhsd, k_bhsd


def apply_flashinfer_rope_pos_ids_for_append(
    layer: torch.nn.Module,
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply FlashInfer RoPE and return Q for attention plus K in append layout.

    The paged KV store wants K as ``[tokens, kv_heads, head_dim]``. Returning
    that tensor directly avoids producing ``[B, H, S, D]`` only to transpose it
    back during cache append.
    """

    q_nhd, k_nhd, bsz, seqlen, num_qo_heads, _num_kv_heads, head_dim = _apply_flashinfer_rope_pos_ids_nhd(
        layer,
        q_bshd,
        k_bshd,
        position_ids,
    )
    q_bhsd = q_nhd.view(bsz, seqlen, num_qo_heads, head_dim).transpose(1, 2).contiguous()
    return q_bhsd, k_nhd


def _apply_flashinfer_rope_pos_ids_nhd(
    layer: torch.nn.Module,
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int, int]:
    """Apply FlashInfer RoPE and keep the flat NHD tensors."""

    bsz, seqlen, num_qo_heads, head_dim = q_bshd.shape
    k_bsz, k_seqlen, num_kv_heads, k_head_dim = k_bshd.shape
    if (k_bsz, k_seqlen, k_head_dim) != (bsz, seqlen, head_dim):
        raise ValueError("Q/K shapes are incompatible for RoPE")
    if position_ids.numel() != bsz * seqlen:
        raise ValueError("position_ids must contain one position per query token")

    config = layer.self_attn.config
    rope_theta = _rope_theta(config)
    rope_scale = _rope_scale(config)
    rotary_dim = int(getattr(config, "partial_rotary_factor", 1.0) * head_dim)
    if rotary_dim <= 0 or rotary_dim > head_dim:
        raise RuntimeError(f"unsupported rotary_dim={rotary_dim} for head_dim={head_dim}")

    q_nhd = q_bshd.reshape(bsz * seqlen, num_qo_heads, head_dim).contiguous()
    k_nhd = k_bshd.reshape(bsz * seqlen, num_kv_heads, head_dim).contiguous()
    pos_ids = position_ids.reshape(-1).to(device=q_nhd.device, dtype=torch.int64)

    q_nhd, k_nhd = flashinfer.apply_rope_pos_ids(
        q_nhd,
        k_nhd,
        pos_ids,
        rotary_dim=rotary_dim,
        interleave=False,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
    )
    return q_nhd, k_nhd, bsz, seqlen, num_qo_heads, num_kv_heads, head_dim


def project_qkv_with_flashinfer_rope(
    layer: torch.nn.Module,
    plan,
    normed_hidden: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    use_flashinfer_norm: bool | None = None,
) -> AttentionProjectionOutput:
    """Project Q/K/V, apply optional Q/K norms, then apply FlashInfer RoPE."""

    projected = project_qkv_with_optional_norm(
        layer,
        plan,
        normed_hidden,
        use_flashinfer_norm=use_flashinfer_norm,
    )
    q_attn, k_store = apply_flashinfer_rope_pos_ids(
        layer,
        projected.q_bshd,
        projected.k_bshd,
        position_ids,
    )
    return AttentionProjectionOutput(
        q_attn_bhsd=q_attn,
        k_store_bhsd=k_store,
        v_store_bhsd=projected.v_bhsd,
    )


def _project_with_optional_head_norm(
    *,
    projection: torch.nn.Module,
    norm: torch.nn.Module | None,
    hidden: torch.Tensor,
    num_heads: int,
    head_dim: int,
    use_flashinfer_norm: bool,
) -> torch.Tensor:
    bsz, seqlen, _ = hidden.shape
    states = projection(hidden).view(bsz, seqlen, num_heads, head_dim)
    if norm is not None:
        states = _apply_optional_head_norm(
            states,
            norm,
            use_flashinfer_norm=use_flashinfer_norm,
        )
    return states


def _apply_optional_head_norm(
    states: torch.Tensor,
    norm: torch.nn.Module,
    *,
    use_flashinfer_norm: bool,
) -> torch.Tensor:
    if use_flashinfer_norm and _can_use_flashinfer_rmsnorm(states, norm):
        flat = states.reshape(-1, states.size(-1)).contiguous()
        out = flashinfer.norm.rmsnorm(flat, norm.weight, eps=_norm_eps(norm))
        return out.view_as(states)
    if not use_flashinfer_norm:
        WARNING("unable to use flashinfer rmsnorm")
    return norm(states)


def _can_use_flashinfer_rmsnorm(states: torch.Tensor, norm: torch.nn.Module) -> bool:
    weight = getattr(norm, "weight", None)
    if weight is None:
        return False
    if weight.ndim != 1 or weight.numel() != states.size(-1):
        return False
    if states.device.type != "cuda" or weight.device.type != "cuda":
        return False
    if not hasattr(getattr(flashinfer, "norm", None), "rmsnorm"):
        return False
    return True


def _norm_eps(norm: torch.nn.Module) -> float:
    return float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))


def _rope_theta(config: object) -> float:
    return float(getattr(config, "rope_theta", 10000.0))


def _rope_scale(config: object) -> float:
    rope_scaling = getattr(config, "rope_scaling", None)
    if not rope_scaling:
        return 1.0
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
        if rope_type in (None, "default"):
            return 1.0
        if rope_type == "linear":
            return float(rope_scaling.get("factor", 1.0))
        if rope_type == "llama3":
            raise RuntimeError("llama3 RoPE scaling requires flashinfer.apply_llama31_rope_pos_ids")
    raise RuntimeError(f"unsupported RoPE scaling config: {rope_scaling!r}")
