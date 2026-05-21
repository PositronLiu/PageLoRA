"""Layer-local LoRA execution helpers.

`LoRALinear` intentionally stays narrow:

- it owns the base linear weight/bias references
- it owns the currently bound serve slot for this layer
- it owns the optional train workspace for this layer
- it applies LoRA during forward

It does not manage:

- multiple logical adapters
- global active adapter selection
- checkpoint lifecycle
- adapter offload policy

Those belong to ``multilora.runtime.LoRARuntime``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class LoRALinear(nn.Module):
    """Single-layer LoRA execution unit.

    The layer only knows about:

    - one currently bound serve adapter
    - one optional train workspace
    - one currently active role

    Multi-adapter management is handled entirely by ``LoRARuntime``.
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

        # Bound serve slot for the currently active logical adapter.
        self.serve_a: nn.Parameter | None = None
        self.serve_b: nn.Parameter | None = None

        # Optional train workspace for the current global train owner.
        self.train_a: nn.Parameter | None = None
        self.train_b: nn.Parameter | None = None

        # Minimal execution state. This is cheaper to reason about than
        # carrying extra active-parameter caches in the layer itself.
        self.active_role: str = "base"

    def bind_serve(self, a: nn.Parameter, b: nn.Parameter) -> None:
        """Bind this layer to one logical adapter's serve parameters."""

        self.serve_a = a
        self.serve_b = b

    def has_train_workspace(self) -> bool:
        """Return whether this layer currently owns materialized train params."""

        return self.train_a is not None and self.train_b is not None

    def materialize_train(self, a: nn.Parameter, b: nn.Parameter) -> None:
        """Create or refresh the train workspace from source serve weights.

        The preferred path is GPU -> GPU copy because serve adapters are
        already expected to live on the target training device.
        """

        with torch.no_grad():
            self.train_a = nn.Parameter(a.detach().clone(), requires_grad=True)
            self.train_b = nn.Parameter(b.detach().clone(), requires_grad=True)

    def release_train(self) -> None:
        """Drop the train workspace entirely when it is no longer needed."""

        self.train_a = None
        self.train_b = None
        if self.active_role == "train":
            self.active_role = "base"

    def commit_train_to_serve(self, target_a: nn.Parameter, target_b: nn.Parameter) -> None:
        """Publish the current train workspace back into one serve target."""

        if self.train_a is None or self.train_b is None:
            raise RuntimeError("train workspace is not materialized")
        with torch.no_grad():
            target_a.data.copy_(self.train_a.data.to(target_a.device))
            target_b.data.copy_(self.train_b.data.to(target_b.device))

    def is_lora_merged(self) -> bool:
        """Return whether the bound serve LoRA is folded into the base weight."""

        return bool(getattr(self, "_merged_lora_adapter", None))

    def merge_serve_lora(self, adapter_name: str) -> bool:
        """Fold the active serve LoRA into ``weight`` for inference."""

        if self.active_role != "serve" or self.serve_a is None or self.serve_b is None:
            return False
        if not self._dropout_mergeable():
            return False
        merged_adapter = getattr(self, "_merged_lora_adapter", None)
        if merged_adapter == adapter_name:
            return False
        if merged_adapter is not None:
            self.unmerge_serve_lora()

        delta = self._serve_lora_weight_delta()
        with torch.no_grad():
            self.weight.data.add_(delta)
        self._merged_lora_adapter = adapter_name
        return True

    def unmerge_serve_lora(self) -> bool:
        """Remove a previously folded serve LoRA from ``weight``."""

        if not self.is_lora_merged():
            return False
        if self.serve_a is None or self.serve_b is None:
            raise RuntimeError("cannot unmerge serve LoRA without bound serve parameters")

        delta = self._serve_lora_weight_delta()
        with torch.no_grad():
            self.weight.data.sub_(delta)
        try:
            delattr(self, "_merged_lora_adapter")
        except AttributeError:
            pass
        return True

    def _serve_lora_weight_delta(self) -> torch.Tensor:
        if self.serve_a is None or self.serve_b is None:
            raise RuntimeError("serve LoRA parameters are not bound")
        return torch.matmul(
            self.serve_b.detach().to(device=self.weight.device, dtype=self.weight.dtype),
            self.serve_a.detach().to(device=self.weight.device, dtype=self.weight.dtype),
        ).mul(float(self.scale))

    def _dropout_mergeable(self) -> bool:
        if isinstance(self.dropout, nn.Identity):
            return True
        if isinstance(self.dropout, nn.Dropout):
            return float(self.dropout.p) == 0.0 or not self.dropout.training
        return False

    def set_active_base(self) -> None:
        """Switch this layer into pure-base mode."""

        self.active_role = "base"

    def set_active_serve(self) -> None:
        """Switch this layer into serve mode."""

        if self.serve_a is None or self.serve_b is None:
            raise RuntimeError("serve mode requested without a bound serve adapter")
        self.active_role = "serve"

    def set_active_train(self) -> None:
        """Switch this layer into train-workspace mode."""

        if self.train_a is None or self.train_b is None:
            raise RuntimeError("train mode requested without materialized train workspace")
        self.active_role = "train"

    def forward(self, x: torch.Tensor, *, return_lora_down: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Apply base projection and, when active, the current LoRA delta.

        When ``return_lora_down=True``, returns ``(base, lora_down)`` where
        ``lora_down`` is the low-rank intermediate activation ``xA``. This
        allows callers to store the decomposed KV cache and reconstruct the
        full output later via ``base + (lora_down @ B) * scale``.
        """

        base = F.linear(x, self.weight, self.bias)
        if self.active_role == "base" or self.is_lora_merged():
            if return_lora_down:
                return base, None
            return base

        if self.active_role == "serve":
            if self.serve_a is None or self.serve_b is None:
                raise RuntimeError("serve mode active without bound serve parameters")
            a = self.serve_a
            b = self.serve_b
        else:
            if self.train_a is None or self.train_b is None:
                raise RuntimeError("train mode active without train workspace")
            a = self.train_a
            b = self.train_b

        dropped = self.dropout(x)
        lora_down = F.linear(dropped, a, None)
        if return_lora_down:
            return base, lora_down
        lora_up = F.linear(lora_down, b, None)
        return base + lora_up * self.scale


def init_lora_pair(
    *,
    rank: int,
    in_features: int,
    out_features: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[nn.Parameter, nn.Parameter]:
    """Create one freshly initialized LoRA A/B parameter pair."""

    a = nn.Parameter(
        torch.empty(rank, in_features, device=device, dtype=dtype),
        requires_grad=False,
    )
    b = nn.Parameter(
        torch.empty(out_features, rank, device=device, dtype=dtype),
        requires_grad=False,
    )
    nn.init.kaiming_uniform_(a, a=math.sqrt(5))
    nn.init.zeros_(b)
    return a, b
