"""Layer-local LoRA storage and forward helpers.

This module intentionally stays narrow:

- it owns per-linear LoRA parameter storage
- it exposes efficient forward-time pointer switching
- it does not own global adapter lifecycle semantics

Global concerns such as active logical adapter, active role, and train-workspace
ownership belong to ``multilora.runtime.LoRARuntime``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_SANITIZE_TABLE = str.maketrans("./:-", "____")

def sanitize_adapter_key(name: str) -> str:
    return name.translate(_SANITIZE_TABLE)


@dataclass
class LoRAModuleWeights:
    """Metadata and parameter handles for one LoRA-instrumented linear layer."""

    a: nn.Parameter
    b: nn.Parameter
    alpha: float
    rank: int

    @property
    def scale(self) -> float:
        return self.alpha / self.rank


class ManualLoRALinear(nn.Module):
    """Linear layer with multiple serve adapters and one optional train workspace.

    ``serve`` adapters are long-lived and live in ``lora_serve_params``.
    ``train`` is an on-demand workspace that only exists while one logical
    adapter is being trained or resumed from an interrupted checkpoint.

    Forward stays efficient because the active LoRA pointers are cached in
    ``_active_a`` / ``_active_b`` and do not require per-token dictionary
    lookups.
    """

    def __init__(
        self,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Long-lived serve registry. Keys are sanitized logical adapter names.
        self.lora_serve_params = nn.ParameterDict()

        # Optional train workspace. These parameters are materialized lazily from
        # serve weights and released after commit or explicit discard.
        self.train_a: nn.Parameter | None = None
        self.train_b: nn.Parameter | None = None

        # Cached pointers used directly by forward.
        self._active_a: nn.Parameter | None = None
        self._active_b: nn.Parameter | None = None

    def _serve_param_keys(self, adapter_name: str) -> tuple[str, str]:
        key = sanitize_adapter_key(adapter_name)
        return f"{key}__a", f"{key}__b"

    def has_logical_adapter(self, adapter_name: str) -> bool:
        """Return whether this layer already owns the given serve adapter."""

        key_a, key_b = self._serve_param_keys(adapter_name)
        return key_a in self.lora_serve_params and key_b in self.lora_serve_params

    def register_logical_adapter(self, adapter_name: str, init_from: str | None = None) -> None:
        """Create one serve adapter for this layer.

        When ``init_from`` is provided, the new adapter copies serve weights
        from the source adapter. Otherwise it uses standard LoRA init so the
        initial delta is zero.
        """

        if self.has_logical_adapter(adapter_name):
            return

        key_a, key_b = self._serve_param_keys(adapter_name)
        if init_from is not None and self.has_logical_adapter(init_from):
            src = self.get_serve_weights(init_from)
            new_a = nn.Parameter(src.a.detach().clone(), requires_grad=False)
            new_b = nn.Parameter(src.b.detach().clone(), requires_grad=False)
        else:
            new_a = nn.Parameter(torch.empty(self.rank, self.in_features), requires_grad=False)
            new_b = nn.Parameter(torch.empty(self.out_features, self.rank), requires_grad=False)
            nn.init.kaiming_uniform_(new_a, a=math.sqrt(5))
            nn.init.zeros_(new_b)
        self.lora_serve_params[key_a] = new_a
        self.lora_serve_params[key_b] = new_b

    def get_serve_weights(self, adapter_name: str) -> LoRAModuleWeights:
        """Return the serve weights for one logical adapter."""

        key_a, key_b = self._serve_param_keys(adapter_name)
        return LoRAModuleWeights(
            a=self.lora_serve_params[key_a],
            b=self.lora_serve_params[key_b],
            alpha=self.alpha,
            rank=self.rank,
        )

    def has_train_workspace(self) -> bool:
        """Return whether this layer currently owns materialized train params."""

        return self.train_a is not None and self.train_b is not None

    def materialize_train_from_serve(self, adapter_name: str) -> None:
        """Create or refresh the train workspace directly from serve weights.

        The preferred path is GPU -> GPU copy because serve adapters are already
        expected to live on the target training device. This avoids the slower
        CPU staging path that would otherwise introduce extra H -> D traffic.
        """

        serve = self.get_serve_weights(adapter_name)
        with torch.no_grad():
            self.train_a = nn.Parameter(serve.a.detach().clone(), requires_grad=True)
            self.train_b = nn.Parameter(serve.b.detach().clone(), requires_grad=True)

    def release_train(self) -> None:
        """Drop the train workspace entirely when it is no longer needed."""

        old_train_a = self.train_a
        old_train_b = self.train_b
        self.train_a = None
        self.train_b = None
        if self._active_a is old_train_a or self._active_b is old_train_b:
            self._active_a = None
            self._active_b = None

    def commit_train_to_serve(self, adapter_name: str) -> None:
        """Publish the current train workspace back into its serve adapter."""

        if self.train_a is None or self.train_b is None:
            raise RuntimeError("train workspace is not materialized")
        serve = self.get_serve_weights(adapter_name)
        with torch.no_grad():
            serve.a.data.copy_(self.train_a.data.to(serve.a.device))
            serve.b.data.copy_(self.train_b.data.to(serve.b.device))

    def set_active_base(self) -> None:
        """Switch this layer into pure-base mode."""

        self._active_a = None
        self._active_b = None

    def set_active_serve(self, adapter_name: str) -> None:
        """Switch this layer into one serve adapter."""

        serve = self.get_serve_weights(adapter_name)
        self._active_a = serve.a
        self._active_b = serve.b

    def set_active_train(self) -> None:
        """Switch this layer into train-workspace mode."""

        if self.train_a is None or self.train_b is None:
            raise RuntimeError("train mode requested without materialized train workspace")
        self._active_a = self.train_a
        self._active_b = self.train_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply base projection and, when active, the cached LoRA delta."""

        base = F.linear(x, self.weight, self.bias)
        if self._active_a is None or self._active_b is None:
            return base
        dropped = self.dropout(x)
        lora_down = F.linear(dropped, self._active_a, None)
        lora_up = F.linear(lora_down, self._active_b, None)
        return base + lora_up * self.scale
