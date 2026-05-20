from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.flashinfer import packed_ops
from model.flashinfer.effective_weights import (
    clear_flashinfer_effective_weight_cache,
    effective_linear,
    effective_weights_enabled,
)
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


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)


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

    _assert_close(projected.q_bshd.reshape(2, 3, 8), attn.q_proj(hidden))
    _assert_close(projected.k_bshd.reshape(2, 3, 4), attn.k_proj(hidden))
    _assert_close(projected.v_bshd.reshape(2, 3, 4), attn.v_proj(hidden))
    assert getattr(plan, "_flashinfer_packed_qkv").weight.shape == (16, 6)


def test_packed_qkv_preserves_lora_delta(monkeypatch):
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_PACKED_QKV", "1")
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_EFFECTIVE_WEIGHTS", "0")
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_PACKED_LORA_DELTA", "1")
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

    _assert_close(projected.q_bshd.reshape(1, 2, 8), q_proj(hidden))
    _assert_close(projected.k_bshd.reshape(1, 2, 4), k_proj(hidden))
    _assert_close(projected.v_bshd.reshape(1, 2, 4), v_proj(hidden))
    packed_lora = getattr(plan, "_flashinfer_packed_qkv_lora_delta")
    assert packed_lora.a.shape == (6, 6)
    assert packed_lora.b.shape == (16, 6)


def test_packed_qkv_lora_delta_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_PACKED_QKV", "1")
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_EFFECTIVE_WEIGHTS", "0")
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_PACKED_LORA_DELTA", "0")
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

    _assert_close(projected.q_bshd.reshape(1, 2, 8), q_proj(hidden))
    _assert_close(projected.k_bshd.reshape(1, 2, 4), k_proj(hidden))
    _assert_close(projected.v_bshd.reshape(1, 2, 4), v_proj(hidden))
    assert not hasattr(plan, "_flashinfer_packed_qkv_lora_delta")


def test_effective_qkv_matches_lora_forward(monkeypatch):
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_PACKED_QKV", "1")
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_EFFECTIVE_WEIGHTS", "1")
    q_proj = LoRALinear(_linear(6, 8, bias=False), rank=2, alpha=4.0, dropout=0.0)
    k_proj = LoRALinear(_linear(6, 4, bias=False), rank=2, alpha=4.0, dropout=0.0)
    v_proj = LoRALinear(_linear(6, 4, bias=False), rank=2, alpha=4.0, dropout=0.0)
    for index, module in enumerate((q_proj, k_proj, v_proj), start=1):
        module.bind_serve(
            nn.Parameter(torch.full((2, 6), 0.03 * index), requires_grad=False),
            nn.Parameter(torch.full((module.out_features, 2), 0.04 * index), requires_grad=False),
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

    _assert_close(projected.q_bshd.reshape(1, 2, 8), q_proj(hidden))
    _assert_close(projected.k_bshd.reshape(1, 2, 4), k_proj(hidden))
    _assert_close(projected.v_bshd.reshape(1, 2, 4), v_proj(hidden))
    assert getattr(plan, "_flashinfer_effective_qkv").weight.shape == (16, 6)
    assert not hasattr(q_proj, "_flashinfer_effective_linear")
    assert not hasattr(k_proj, "_flashinfer_effective_linear")
    assert not hasattr(v_proj, "_flashinfer_effective_linear")


def test_effective_linear_matches_lora_forward_and_can_clear(monkeypatch):
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_EFFECTIVE_WEIGHTS", "1")
    module = LoRALinear(_linear(6, 4, bias=True), rank=2, alpha=4.0, dropout=0.0)
    module.bind_serve(
        nn.Parameter(torch.full((2, 6), 0.1), requires_grad=False),
        nn.Parameter(torch.full((4, 2), 0.2), requires_grad=False),
    )
    module.set_active_serve()
    hidden = torch.randn(2, 3, 6)

    effective = effective_linear(module)
    output = packed_ops.project_linear(module, hidden)

    assert effective is not None
    _assert_close(output, module(hidden))
    assert hasattr(module, "_flashinfer_effective_linear")
    clear_flashinfer_effective_weight_cache(module)
    assert not hasattr(module, "_flashinfer_effective_linear")


def test_project_linear_falls_back_to_lora_when_effective_disabled(monkeypatch):
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_EFFECTIVE_WEIGHTS", "0")
    module = LoRALinear(_linear(6, 4, bias=True), rank=2, alpha=4.0, dropout=0.0)
    module.bind_serve(
        nn.Parameter(torch.full((2, 6), 0.1), requires_grad=False),
        nn.Parameter(torch.full((4, 2), 0.2), requires_grad=False),
    )
    module.set_active_serve()
    hidden = torch.randn(2, 3, 6)

    output = packed_ops.project_linear(module, hidden)

    _assert_close(output, module(hidden))


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

    _assert_close(output, mlp(hidden))
    assert getattr(mlp, "_flashinfer_packed_gate_up").weight.shape == (10, 6)


def test_balanced_memory_mode_disables_effective_weights_by_default(monkeypatch):
    monkeypatch.delenv("MINI_OPENCLAW_FLASHINFER_EFFECTIVE_WEIGHTS", raising=False)
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_MEMORY_MODE", "balanced")

    assert not effective_weights_enabled()


def test_shared_packed_gate_up_storage_matches_reference(monkeypatch):
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_PACKED_MLP", "1")
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_FUSED_ACT", "0")
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_PACKED_WEIGHT_STORAGE", "shared")
    mlp = _TinyMLP()
    for param in mlp.parameters():
        param.requires_grad = False
    hidden = torch.randn(2, 3, 6)

    gate_up = packed_ops.project_gate_up_packed(mlp, hidden)
    activated = packed_ops.apply_gate_up_activation(gate_up, mlp.act_fn)
    output = packed_ops.project_down(mlp, activated)

    _assert_close(output, mlp(hidden))
    packed = getattr(mlp, "_flashinfer_packed_gate_up")
    packed_storage = packed.weight.untyped_storage().data_ptr()
    assert mlp.gate_proj.weight.untyped_storage().data_ptr() == packed_storage
    assert mlp.up_proj.weight.untyped_storage().data_ptr() == packed_storage
    assert packed_ops.project_gate_up_packed(mlp, hidden) is not None
    assert getattr(mlp, "_flashinfer_packed_gate_up") is packed


class _TinyLoRAMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = LoRALinear(_linear(6, 5, bias=True), rank=2, alpha=4.0, dropout=0.0)
        self.up_proj = LoRALinear(_linear(6, 5, bias=True), rank=2, alpha=4.0, dropout=0.0)
        self.down_proj = _linear(5, 6, bias=True)
        self.act_fn = nn.SiLU()
        for index, module in enumerate((self.gate_proj, self.up_proj), start=1):
            module.bind_serve(
                nn.Parameter(torch.full((2, 6), 0.03 * index), requires_grad=False),
                nn.Parameter(torch.full((5, 2), 0.04 * index), requires_grad=False),
            )
            module.set_active_serve()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def test_packed_gate_up_preserves_lora_delta(monkeypatch):
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_PACKED_MLP", "1")
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_EFFECTIVE_WEIGHTS", "0")
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_PACKED_LORA_DELTA", "1")
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_FUSED_ACT", "0")
    mlp = _TinyLoRAMLP()
    hidden = torch.randn(2, 3, 6)

    gate_up = packed_ops.project_gate_up_packed(mlp, hidden)
    activated = packed_ops.apply_gate_up_activation(gate_up, mlp.act_fn)
    output = packed_ops.project_down(mlp, activated)

    _assert_close(output, mlp(hidden))
    packed_lora = getattr(mlp, "_flashinfer_packed_gate_up_lora_delta")
    assert packed_lora.a.shape == (4, 6)
    assert packed_lora.b.shape == (10, 4)


def test_apply_gate_up_activation_fallback_matches_silu(monkeypatch):
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_FUSED_ACT", "0")
    gate_up = torch.randn(4, 10)
    output = packed_ops.apply_gate_up_activation(gate_up, nn.SiLU())
    gate, up = gate_up.chunk(2, dim=-1)

    _assert_close(output, F.silu(gate) * up)
