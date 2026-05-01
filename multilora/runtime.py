"""Manual LoRA runtime built on top of HF base model structure.

This module keeps HuggingFace as the source of model structure and checkpoint
loading, but removes PEFT from the runtime path. Target linear modules are
wrapped by a lightweight LoRA module that:

- keeps base weights in the original linear storage
- stores serve/train/reference LoRA weights explicitly
- switches adapters by changing a small runtime flag

This makes slot/offload logic much easier to reason about because adapter
state is explicit and no PEFT wrapper hierarchy needs to be traversed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.component_discovery import discover_components


class AdapterNames(Enum):
    SERVE = "serve"
    TRAIN = "train"
    REFERENCE = "ref"


def normalize_adapter_name(name: str | None) -> str:
    """Normalize public adapter aliases into the internal canonical names."""

    if name in (None, "", AdapterNames.SERVE.value, "policy"):
        return AdapterNames.SERVE.value
    if name == AdapterNames.TRAIN.value:
        return AdapterNames.TRAIN.value
    if name == AdapterNames.REFERENCE.value:
        return AdapterNames.REFERENCE.value
    raise ValueError(f"Unknown adapter name: {name}")


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
    """Linear layer with explicit serve/train/reference LoRA weights.

    The base linear weight/bias remain standard parameters so existing HF
    decoder block code continues to work. LoRA is applied with two additional
    large tensor ops, which keeps execution efficient while avoiding PEFT
    wrapper complexity.
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
        self.active_adapter_name = AdapterNames.SERVE.value

        self.lora_serve_a = nn.Parameter(torch.empty(rank, self.in_features), requires_grad=False)
        self.lora_serve_b = nn.Parameter(torch.empty(self.out_features, rank), requires_grad=False)
        self.lora_train_a = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_train_b = nn.Parameter(torch.empty(self.out_features, rank))
        self.lora_ref_a = nn.Parameter(torch.empty(rank, self.in_features), requires_grad=False)
        self.lora_ref_b = nn.Parameter(torch.empty(self.out_features, rank), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize LoRA weights so the initial delta is zero."""

        nn.init.kaiming_uniform_(self.lora_serve_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_serve_b)
        with torch.no_grad():
            self.lora_train_a.copy_(self.lora_serve_a)
            self.lora_train_b.copy_(self.lora_serve_b)
            self.lora_ref_a.copy_(self.lora_serve_a)
            self.lora_ref_b.copy_(self.lora_serve_b)

    def set_active_adapter(self, name: str) -> None:
        """Switch which adapter is applied during forward."""

        self.active_adapter_name = normalize_adapter_name(name)

    def get_adapter_weights(self, name: str | None = None) -> LoRAModuleWeights:
        """Return the selected adapter parameter handles."""

        adapter_name = normalize_adapter_name(name or self.active_adapter_name)
        if adapter_name == AdapterNames.REFERENCE.value:
            return LoRAModuleWeights(
                a=self.lora_ref_a,
                b=self.lora_ref_b,
                alpha=self.alpha,
                rank=self.rank,
            )
        if adapter_name == AdapterNames.TRAIN.value:
            return LoRAModuleWeights(
                a=self.lora_train_a,
                b=self.lora_train_b,
                alpha=self.alpha,
                rank=self.rank,
            )
        return LoRAModuleWeights(
            a=self.lora_serve_a,
            b=self.lora_serve_b,
            alpha=self.alpha,
            rank=self.rank,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply base linear projection plus the active LoRA delta."""

        base = F.linear(x, self.weight, self.bias)
        adapter = self.get_adapter_weights()
        dropped = self.dropout(x)
        lora_down = F.linear(dropped, adapter.a, None)
        lora_up = F.linear(lora_down, adapter.b, None)
        return base + lora_up * adapter.scale


@dataclass
class LayerExecutionPlan:
    """Execution metadata for one decoder block."""

    modules: dict[str, ManualLoRALinear]


class LoRARuntime:
    """Own manual LoRA storage, adapter switching, and checkpoint I/O."""

    def __init__(
        self,
        model: nn.Module,
        *,
        target_modules: tuple[str, ...],
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        self.model = model
        self.target_modules = target_modules
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.active_adapter_name = AdapterNames.SERVE.value

        comps = discover_components(model)
        self.layers: list[nn.Module] = comps["layers"]
        self.layer_plans: list[LayerExecutionPlan] = []
        self._manual_modules: list[tuple[int, str, ManualLoRALinear]] = []
        self._install_wrappers()
        self._freeze_non_train_parameters()

    def _install_wrappers(self) -> None:
        """Replace target linear modules in each decoder layer with manual LoRA wrappers."""

        for layer_idx, layer in enumerate(self.layers):
            wrapped_modules: dict[str, ManualLoRALinear] = {}
            for module_name, module in list(layer.named_modules()):
                if not module_name:
                    continue
                leaf_name = module_name.rsplit(".", 1)[-1]
                if leaf_name not in self.target_modules:
                    continue
                if not isinstance(module, nn.Linear):
                    continue

                wrapper = ManualLoRALinear(
                    module,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                )
                self._set_nested_module(layer, module_name, wrapper)
                wrapped_modules[module_name] = wrapper
                self._manual_modules.append((layer_idx, module_name, wrapper))
            self.layer_plans.append(LayerExecutionPlan(modules=wrapped_modules))

    def _freeze_non_train_parameters(self) -> None:
        """Freeze every non-train parameter in the model tree.

        Without PEFT, the base model parameters would otherwise remain
        trainable by default, which would waste gradient memory and break the
        intended "LoRA-only update" training regime. Only the dedicated
        training adapter should receive gradients; the serve/reference copies
        act as published snapshots.
        """

        for param in self.model.parameters():
            param.requires_grad = False
        for _, _, module in self._manual_modules:
            module.lora_train_a.requires_grad = True
            module.lora_train_b.requires_grad = True

    @staticmethod
    def _set_nested_module(root: nn.Module, path: str, module: nn.Module) -> None:
        """Replace a nested child module given its dot-separated path."""

        parent = root
        parts = path.split(".")
        for attr in parts[:-1]:
            parent = getattr(parent, attr)
        setattr(parent, parts[-1], module)

    def set_active_adapter(self, name: str) -> None:
        """Switch the active adapter for all wrapped modules."""

        name = normalize_adapter_name(name)
        self.active_adapter_name = name
        for _, _, module in self._manual_modules:
            module.set_active_adapter(name)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """Yield only training adapter parameters."""

        for _, _, module in self._manual_modules:
            yield module.lora_train_a
            yield module.lora_train_b

    def move_all_lora_to_gpu(self, device: torch.device) -> int:
        """Move all LoRA adapter tensors to GPU."""

        moved = 0
        with torch.no_grad():
            for _, _, module in self._manual_modules:
                for param in (
                    module.lora_serve_a,
                    module.lora_serve_b,
                    module.lora_train_a,
                    module.lora_train_b,
                    module.lora_ref_a,
                    module.lora_ref_b,
                ):
                    if param.data.device != device:
                        param.data = param.data.to(device)
                        moved += 1
        return moved

    def export_adapter_state(self, name: str) -> dict[str, torch.Tensor]:
        """Export one adapter tensor set in a compact layer/module keyed format."""

        name = normalize_adapter_name(name)
        state: dict[str, torch.Tensor] = {}
        for layer_idx, module_name, module in self._manual_modules:
            prefix = f"layers.{layer_idx}.{module_name}"
            adapter = module.get_adapter_weights(name)
            state[f"{prefix}.lora_{name}_a"] = adapter.a.detach().cpu()
            state[f"{prefix}.lora_{name}_b"] = adapter.b.detach().cpu()
        return state

    def load_adapter_state(self, name: str, state: dict[str, torch.Tensor]) -> None:
        """Load one adapter tensor set from a compact layer/module keyed format."""

        name = normalize_adapter_name(name)
        manual_map = {
            (layer_idx, module_name): module
            for layer_idx, module_name, module in self._manual_modules
        }
        with torch.no_grad():
            for key, tensor in state.items():
                parts = key.split(".")
                if len(parts) < 4 or parts[0] != "layers":
                    continue
                layer_idx = int(parts[1])
                module_name = ".".join(parts[2:-1])
                leaf_name = parts[-1]
                module = manual_map.get((layer_idx, module_name))
                if module is None:
                    continue
                expected_prefix = f"lora_{name}_"
                if not leaf_name.startswith(expected_prefix):
                    continue
                target = getattr(module, leaf_name, None)
                if isinstance(target, nn.Parameter):
                    target.data.copy_(tensor.to(target.device))

    def copy_adapter_(self, src: str, dst: str) -> None:
        """Copy one adapter tensor set into another in-place."""

        src = normalize_adapter_name(src)
        dst = normalize_adapter_name(dst)
        if src == dst:
            return
        with torch.no_grad():
            for _, _, module in self._manual_modules:
                src_weights = module.get_adapter_weights(src)
                dst_weights = module.get_adapter_weights(dst)
                dst_weights.a.data.copy_(src_weights.a.data)
                dst_weights.b.data.copy_(src_weights.b.data)

    def commit_train_to_serve(self) -> None:
        """Publish the latest completed training adapter to the serve adapter."""

        self.copy_adapter_(AdapterNames.TRAIN.value, AdapterNames.SERVE.value)

    def export_serve_state(self) -> dict[str, torch.Tensor]:
        """Export the serve adapter state."""

        return self.export_adapter_state(AdapterNames.SERVE.value)

    def export_train_state(self) -> dict[str, torch.Tensor]:
        """Export the training adapter state."""

        return self.export_adapter_state(AdapterNames.TRAIN.value)

    def load_serve_state(self, state: dict[str, torch.Tensor]) -> None:
        """Load the serve adapter state."""

        self.load_adapter_state(AdapterNames.SERVE.value, state)

    def load_train_state(self, state: dict[str, torch.Tensor]) -> None:
        """Load the training adapter state."""

        self.load_adapter_state(AdapterNames.TRAIN.value, state)

    def export_policy_state(self) -> dict[str, torch.Tensor]:
        """Backward-compatible alias for the training adapter state."""

        return self.export_train_state()

    def load_policy_state(self, state: dict[str, torch.Tensor]) -> None:
        """Backward-compatible alias for loading the training adapter state."""

        self.load_train_state(state)

    def get_layer_adapter(
        self,
        layer_idx: int,
        module_name: str,
        adapter_name: str | None = None,
    ) -> LoRAModuleWeights | None:
        """Return one module's LoRA weights for explicit dispatcher use."""

        try:
            module = self.layer_plans[layer_idx].modules[module_name]
        except (IndexError, KeyError):
            return None
        return module.get_adapter_weights(adapter_name)
