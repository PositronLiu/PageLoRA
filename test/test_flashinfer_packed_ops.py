from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.flashinfer import packed_ops
from model.flashinfer.projection import project_qkv_with_optional_norm
from multilora.linear import LoRALinear


def _linear(in_features: int, out_features: int, *, bias: bool = False) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=bias)
    with torch.no_grad():
        layer.weight.copy_(
            torch.arange(out_features * in_features, dtype=torch.float32).view(out_features, in_features)
            / 100.0
        )
        if bias:
            layer.bias.copy_(torch.arange(out_features, dtype=torch.float32) / 10.0)
    return layer


def test_packed_qkv_matches_three_linear_projections(monkeypatch):
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_PACKED_QKV", "1")
    attn = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=2, num_key_value_heads=1),
        head_dim=4,
        q_proj=_linear(6, 8, bias=True),
        k_proj=_linear(6, 4, bias=True),
        v_proj=_linear(6, 4, bias=True),
    )
    layer = SimpleNamespace(self_attn=attn)
    plan = SimpleNamespace(modules={})
    hidden = torch.randn(2, 3, 6)

    projected = project_qkv_with_optional_norm(layer, plan, hidden)

    assert torch.allclose(projected.q_bshd.reshape(2, 3, 8), attn.q_proj(hidden))
    assert torch.allclose(projected.k_bshd.reshape(2, 3, 4), attn.k_proj(hidden))
    assert torch.allclose(projected.v_bshd.reshape(2, 3, 4), attn.v_proj(hidden))
    assert getattr(plan, "_flashinfer_packed_qkv").weight.shape == (16, 6)


def test_packed_qkv_preserves_lora_delta(monkeypatch):
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_PACKED_QKV", "1")
    q_proj = LoRALinear(_linear(6, 8, bias=False), rank=2, alpha=4.0, dropout=0.0)
    k_proj = LoRALinear(_linear(6, 4, bias=False), rank=2, alpha=4.0, dropout=0.0)
    v_proj = LoRALinear(_linear(6, 4, bias=False), rank=2, alpha=4.0, dropout=0.0)
    for module in (q_proj, k_proj, v_proj):
        module.bind_serve(
            nn.Parameter(torch.full((2, 6), 0.1), requires_grad=False),
            nn.Parameter(torch.full((module.out_features, 2), 0.2), requires_grad=False),
        )
        module.set_active_serve()
    attn = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=2, num_key_value_heads=1),
        head_dim=4,
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
    )
    layer = SimpleNamespace(self_attn=attn)
    plan = SimpleNamespace(modules={
        "self_attn.q_proj": q_proj,
        "self_attn.k_proj": k_proj,
        "self_attn.v_proj": v_proj,
    })
    hidden = torch.randn(1, 2, 6)

    projected = project_qkv_with_optional_norm(layer, plan, hidden)

    assert torch.allclose(projected.q_bshd.reshape(1, 2, 8), q_proj(hidden))
    assert torch.allclose(projected.k_bshd.reshape(1, 2, 4), k_proj(hidden))
    assert torch.allclose(projected.v_bshd.reshape(1, 2, 4), v_proj(hidden))


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = _linear(6, 5, bias=True)
        self.up_proj = _linear(6, 5, bias=True)
        self.down_proj = _linear(5, 6, bias=True)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def test_packed_gate_up_mlp_matches_reference(monkeypatch):
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_PACKED_MLP", "1")
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_FUSED_ACT", "0")
    mlp = _TinyMLP()
    hidden = torch.randn(2, 3, 6)

    gate_up = packed_ops.project_gate_up_packed(mlp, hidden)
    activated = packed_ops.apply_gate_up_activation(gate_up, mlp.act_fn)
    output = packed_ops.project_down(mlp, activated)

    assert torch.allclose(output, mlp(hidden))
    assert getattr(mlp, "_flashinfer_packed_gate_up").weight.shape == (10, 6)


def test_apply_gate_up_activation_fallback_matches_silu(monkeypatch):
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_FUSED_ACT", "0")
    gate_up = torch.randn(4, 10)
    output = packed_ops.apply_gate_up_activation(gate_up, nn.SiLU())
    gate, up = gate_up.chunk(2, dim=-1)

    assert torch.allclose(output, F.silu(gate) * up)
