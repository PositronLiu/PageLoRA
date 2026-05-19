from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.flashinfer import compiled_mlp, compiled_post, packed_ops
from model.flashinfer.compile_config import clear_flashinfer_compile_cache
from model.flashinfer.norm import apply_rmsnorm


def _linear(in_features: int, out_features: int, *, bias: bool = True) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=bias)
    with torch.no_grad():
        layer.weight.copy_(
            torch.arange(out_features * in_features, dtype=torch.float32).view(out_features, in_features)
            / 100.0
        )
        if bias:
            layer.bias.copy_(torch.arange(out_features, dtype=torch.float32) / 50.0)
    return layer


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = _linear(6, 8)
        self.up_proj = _linear(6, 8)
        self.down_proj = _linear(8, 6)
        self.act_fn = nn.SiLU()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class TinyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(1, hidden_size + 1, dtype=torch.float32) / hidden_size)
        self.variance_epsilon = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden_f = hidden.float()
        variance = hidden_f.pow(2).mean(dim=-1, keepdim=True)
        return (hidden_f * torch.rsqrt(variance + self.variance_epsilon) * self.weight).to(hidden.dtype)


class TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = SimpleNamespace(o_proj=_linear(6, 6))
        self.post_attention_layernorm = TinyRMSNorm(6)
        self.mlp = TinyMLP()


def _expected_mlp(mlp: TinyMLP, hidden: torch.Tensor) -> torch.Tensor:
    gate_up = packed_ops.project_gate_up_packed(mlp, hidden)
    assert gate_up is not None
    activated = packed_ops.apply_gate_up_activation(gate_up, mlp.act_fn)
    return packed_ops.project_down(mlp, activated)


def test_compiled_mlp_matches_eager_and_reuses_cache(monkeypatch):
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_COMPILE_MLP", "1")
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_PACKED_MLP", "1")
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_FUSED_ACT", "0")
    monkeypatch.setattr(compiled_mlp, "can_compile_decode_tensor", lambda hidden: True)
    compile_calls = []

    def fake_compile(module: nn.Module) -> nn.Module:
        compile_calls.append(module)
        return module

    monkeypatch.setattr(compiled_mlp, "compile_module", fake_compile)
    plan = SimpleNamespace()
    mlp = TinyMLP()
    hidden = torch.randn(1, 1, 6)

    output = compiled_mlp.run_compiled_mlp(plan, mlp, hidden)
    cached_output = compiled_mlp.run_compiled_mlp(plan, mlp, hidden)

    assert output is not None
    assert cached_output is not None
    assert len(compile_calls) == 1
    assert torch.allclose(output, _expected_mlp(mlp, hidden), rtol=1e-5, atol=1e-5)
    assert torch.allclose(cached_output, output, rtol=1e-5, atol=1e-5)


def test_compiled_mlp_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("MINI_OPENCLAW_FLASHINFER_TORCH_COMPILE", raising=False)
    monkeypatch.delenv("MINI_OPENCLAW_FLASHINFER_COMPILE_MLP", raising=False)
    monkeypatch.setattr(compiled_mlp, "can_compile_decode_tensor", lambda hidden: True)

    output = compiled_mlp.run_compiled_mlp(SimpleNamespace(), TinyMLP(), torch.randn(1, 1, 6))

    assert output is None


def test_compiled_post_attention_matches_eager_and_reuses_cache(monkeypatch):
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_COMPILE_POST", "1")
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_PACKED_MLP", "1")
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_FUSED_ACT", "0")
    monkeypatch.setattr(compiled_post, "can_compile_decode_tensor", lambda hidden: True)
    compile_calls = []

    def fake_compile(module: nn.Module) -> nn.Module:
        compile_calls.append(module)
        return module

    monkeypatch.setattr(compiled_post, "compile_module", fake_compile)
    plan = SimpleNamespace(modules={})
    layer = TinyLayer()
    hidden = torch.randn(1, 1, 6)
    attn_output = torch.randn(1, 2, 1, 3)

    output = compiled_post.run_compiled_post_attention(plan, layer, hidden, attn_output)
    cached_output = compiled_post.run_compiled_post_attention(plan, layer, hidden, attn_output)

    eager_attn = attn_output.transpose(1, 2).contiguous().view(1, 1, 6)
    eager_attn = packed_ops.project_linear(layer.self_attn.o_proj, eager_attn)
    eager_hidden = hidden + eager_attn
    eager_mlp_input = apply_rmsnorm(layer.post_attention_layernorm, eager_hidden)
    expected = eager_hidden + _expected_mlp(layer.mlp, eager_mlp_input)

    assert output is not None
    assert cached_output is not None
    assert len(compile_calls) == 1
    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-5)
    assert torch.allclose(cached_output, output, rtol=1e-5, atol=1e-5)


def test_clear_compile_cache_walks_plans_and_layers():
    layer = nn.Sequential(nn.Linear(2, 2))
    layer._flashinfer_compiled_mlp = object()
    layer[0]._flashinfer_compile_mlp_disabled = True
    plan = SimpleNamespace(
        _flashinfer_compiled_post=object(),
        modules={"linear": layer[0]},
    )

    clear_flashinfer_compile_cache([plan], [layer])

    assert not hasattr(plan, "_flashinfer_compiled_post")
    assert not hasattr(layer, "_flashinfer_compiled_mlp")
    assert not hasattr(layer[0], "_flashinfer_compile_mlp_disabled")
