from __future__ import annotations

import torch

from model.flashinfer import norm as norm_ops


class _RMSNormLike(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(1, hidden_size + 1, dtype=torch.float32))
        self.variance_epsilon = eps
        self.called = False

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.called = True
        var = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        return (hidden.float() * torch.rsqrt(var + self.variance_epsilon) * self.weight).to(hidden.dtype)


def test_apply_rmsnorm_uses_flashinfer_when_compatible(monkeypatch):
    calls = {}

    def fake_rmsnorm(flat, weight, eps=1e-6):
        calls["shape"] = tuple(flat.shape)
        calls["weight"] = weight.detach().clone()
        calls["eps"] = eps
        return flat + 3.0

    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_RMSNORM", "1")
    monkeypatch.setattr(norm_ops, "_can_use_flashinfer_rmsnorm", lambda hidden, norm: True)
    monkeypatch.setattr(norm_ops.flashinfer.norm, "rmsnorm", fake_rmsnorm)
    norm = _RMSNormLike(4, eps=2e-5)
    hidden = torch.zeros(2, 3, 4)

    output = norm_ops.apply_rmsnorm(norm, hidden)

    assert not norm.called
    assert calls["shape"] == (6, 4)
    assert calls["eps"] == 2e-5
    assert torch.equal(calls["weight"], norm.weight.detach())
    assert torch.allclose(output, torch.full_like(hidden, 3.0))


def test_apply_rmsnorm_falls_back_when_disabled(monkeypatch):
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_RMSNORM", "0")
    monkeypatch.setattr(norm_ops, "_can_use_flashinfer_rmsnorm", lambda hidden, norm: True)
    norm = _RMSNormLike(4, eps=1e-5)
    hidden = torch.randn(2, 3, 4)

    output = norm_ops.apply_rmsnorm(norm, hidden)

    assert norm.called
    assert torch.allclose(output, norm.forward(hidden))


def test_apply_rmsnorm_does_not_replace_layernorm(monkeypatch):
    monkeypatch.setenv("MINI_OPENCLAW_FLASHINFER_RMSNORM", "1")
    layernorm = torch.nn.LayerNorm(4)
    hidden = torch.randn(2, 3, 4)

    output = norm_ops.apply_rmsnorm(layernorm, hidden)

    assert torch.allclose(output, layernorm(hidden))
