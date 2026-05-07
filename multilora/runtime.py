"""Global Multi-LoRA runtime lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import torch
import torch.nn as nn

from model.component_discovery import discover_lora_patch_targets
from multilora.linear import LoRAModuleWeights, LoRALinear, init_lora_pair


class AdapterRole(Enum):
    """Runtime role applied during forward."""

    BASE = "base"
    SERVE = "serve"
    TRAIN = "train"


def normalize_adapter_role(name: str | None) -> str:
    """Normalize public adapter-role aliases into canonical runtime names."""

    if name in (None, "", AdapterRole.SERVE.value, "policy"):
        return AdapterRole.SERVE.value
    if name in (AdapterRole.TRAIN.value,):
        return AdapterRole.TRAIN.value
    if name in (AdapterRole.BASE.value, "ref", "reference"):
        return AdapterRole.BASE.value
    raise ValueError(f"Unknown adapter role: {name}")


@dataclass
class LayerExecutionPlan:
    """Execution metadata for one decoder block."""

    modules: dict[str, LoRALinear]


class LoRARuntime:
    """Manual Multi-LoRA storage, switching, and checkpoint I/O."""

    def __init__(
        self,
        model: nn.Module,
        *,
        target_modules: tuple[str, ...],
        rank: int,
        alpha: float,
        dropout: float,
        logical_adapters: tuple[str, ...] = ("general",),
        default_logical_adapter: str = "general",
        adapter_offload: bool = False,
        adapter_offload_lru_size: int = 1,
    ) -> None:
        self.model = model
        self.target_modules = target_modules
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.logical_adapter_names: list[str] = []
        self.default_logical_adapter = default_logical_adapter
        self.active_logical_adapter_name = default_logical_adapter
        self.active_role = AdapterRole.SERVE.value
        self.current_train_owner: str | None = None

        self._adapter_offload = adapter_offload
        self._adapter_offload_lru_size = adapter_offload_lru_size
        self._device: torch.device | None = None
        self._gpu_resident_adapters: list[str] = []
        self._kvcache_manager: object | None = None

        targets = discover_lora_patch_targets(model)
        self.layers: list[nn.Module] = targets["layers"]
        self.layer_plans: list[LayerExecutionPlan] = []
        self._manual_modules: list[tuple[int, str, LoRALinear]] = []
        self._serve_registry: dict[str, dict[tuple[int, str], LoRAModuleWeights]] = {}
        self._install_wrappers()
        for index, adapter_name in enumerate(logical_adapters):
            init_from = None if index == 0 else self.logical_adapter_names[0]
            self.register_logical_adapter(adapter_name, init_from=init_from)
        self._freeze_base_and_serve_parameters()
        self.set_active(default_logical_adapter, AdapterRole.SERVE.value)

    def _install_wrappers(self) -> None:
        """Replace target linear modules in each decoder layer with manual LoRA wrappers."""

        for layer_idx, layer in enumerate(self.layers):
            wrapped_modules: dict[str, LoRALinear] = {}
            for module_name, module in list(layer.named_modules()):
                if not module_name:
                    continue
                leaf_name = module_name.rsplit(".", 1)[-1]
                if leaf_name not in self.target_modules or not isinstance(module, nn.Linear):
                    continue

                wrapper = LoRALinear(
                    module,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                )
                self._set_nested_module(layer, module_name, wrapper)
                wrapped_modules[module_name] = wrapper
                self._manual_modules.append((layer_idx, module_name, wrapper))
            self.layer_plans.append(LayerExecutionPlan(modules=wrapped_modules))

    def _freeze_base_and_serve_parameters(self) -> None:
        """Freeze base and serve parameters; train workspaces are created later."""

        for param in self.model.parameters():
            param.requires_grad = False
        for adapter_layers in self._serve_registry.values():
            for weights in adapter_layers.values():
                weights.a.requires_grad = False
                weights.b.requires_grad = False

    @staticmethod
    def _set_nested_module(root: nn.Module, path: str, module: nn.Module) -> None:
        """Replace a nested child module given its dot-separated path."""

        parent = root
        parts = path.split(".")
        for attr in parts[:-1]:
            parent = getattr(parent, attr)
        setattr(parent, parts[-1], module)

    def _broadcast_active(self) -> None:
        """Broadcast the current global active state into per-layer execution state."""

        if self.active_role == AdapterRole.BASE.value:
            for _, _, module in self._manual_modules:
                module.set_active_base()
            return

        if self.active_role == AdapterRole.SERVE.value:
            adapter_name = self.active_logical_adapter_name

            if self._adapter_offload and self._device is not None:
                self.ensure_adapter_on_gpu(adapter_name, self._device)

            adapter_layers = self._serve_registry[adapter_name]
            for layer_idx, module_name, module in self._manual_modules:
                weights = adapter_layers[(layer_idx, module_name)]
                module.bind_serve(weights.a, weights.b)
                module.set_active_serve()

            if self._adapter_offload:
                self._maybe_offload_inactive()
            return

        for _, _, module in self._manual_modules:
            module.set_active_train()

    # ── adapter offload ──────────────────────────────────────────────

    def ensure_adapter_on_gpu(self, adapter_name: str, device: torch.device) -> None:
        """Move one adapter's serve weights to *device* and track as GPU-resident."""

        if adapter_name not in self._serve_registry:
            return
        with torch.no_grad():
            for weights in self._serve_registry[adapter_name].values():
                if weights.a.data.device != device:
                    weights.a.data = weights.a.data.to(device)
                if weights.b.data.device != device:
                    weights.b.data = weights.b.data.to(device)
        if adapter_name in self._gpu_resident_adapters:
            self._gpu_resident_adapters.remove(adapter_name)
        self._gpu_resident_adapters.append(adapter_name)

        if self._kvcache_manager is not None:
            self._kvcache_manager.ensure_residual_on_device(adapter_name, device)

    def offload_adapter_to_cpu(self, adapter_name: str) -> None:
        """Move one adapter's serve weights to CPU and remove from GPU set."""

        if adapter_name not in self._serve_registry:
            return
        with torch.no_grad():
            for weights in self._serve_registry[adapter_name].values():
                if weights.a.data.device.type != "cpu":
                    weights.a.data = weights.a.data.cpu()
                if weights.b.data.device.type != "cpu":
                    weights.b.data = weights.b.data.cpu()
        if adapter_name in self._gpu_resident_adapters:
            self._gpu_resident_adapters.remove(adapter_name)

        if self._kvcache_manager is not None:
            self._kvcache_manager.offload_residual(adapter_name)

    def _maybe_offload_inactive(self) -> None:
        """Evict LRU adapters beyond the active + LRU budget."""

        if not self._adapter_offload or self._device is None:
            return
        max_resident = 1 + self._adapter_offload_lru_size
        inactive = [a for a in self._gpu_resident_adapters
                     if a != self.active_logical_adapter_name]
        while len(self._gpu_resident_adapters) > max_resident and inactive:
            adapter = inactive.pop(0)  # oldest inactive
            self._gpu_resident_adapters.remove(adapter)
            self.offload_adapter_to_cpu(adapter)

    def register_logical_adapter(self, adapter_name: str, init_from: str | None = None) -> None:
        """Register one logical adapter across every wrapped layer."""

        if adapter_name in self.logical_adapter_names:
            return

        adapter_layers: dict[tuple[int, str], LoRAModuleWeights] = {}
        for layer_idx, module_name, module in self._manual_modules:
            if init_from is not None and init_from in self._serve_registry:
                src = self._serve_registry[init_from][(layer_idx, module_name)]
                a = nn.Parameter(src.a.detach().clone(), requires_grad=False)
                b = nn.Parameter(src.b.detach().clone(), requires_grad=False)
            else:
                a, b = init_lora_pair(
                    rank=module.rank,
                    in_features=module.in_features,
                    out_features=module.out_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
            adapter_layers[(layer_idx, module_name)] = LoRAModuleWeights(
                a=a,
                b=b,
                alpha=module.alpha,
                rank=module.rank,
            )

        self._serve_registry[adapter_name] = adapter_layers
        self.logical_adapter_names.append(adapter_name)

        if self._adapter_offload and self._device is not None and adapter_name != self.active_logical_adapter_name:
            self.offload_adapter_to_cpu(adapter_name)

    def list_logical_adapters(self) -> list[str]:
        """Return all registered logical adapter names."""

        return list(self.logical_adapter_names)

    def set_kvcache_manager(self, mgr: object) -> None:
        """Bind the B+R ``KVCacheManager`` so adapter offload can
        co-manage residual caches alongside adapter weights."""

        self._kvcache_manager = mgr

    def set_active(self, adapter_name: str | None, role: str) -> None:
        """Switch the active logical adapter and runtime role."""

        role = normalize_adapter_role(role)
        if role == AdapterRole.SERVE.value:
            adapter_name = adapter_name or self.default_logical_adapter
            if adapter_name not in self.logical_adapter_names:
                raise KeyError(f"Unknown logical adapter: {adapter_name}")
            self.active_logical_adapter_name = adapter_name
        elif role == AdapterRole.TRAIN.value:
            if self.current_train_owner is None:
                raise RuntimeError("train role requested without an active train workspace")
            adapter_name = adapter_name or self.current_train_owner
            if adapter_name != self.current_train_owner:
                raise RuntimeError(
                    f"train workspace belongs to {self.current_train_owner}, got request for {adapter_name}"
                )
            self.active_logical_adapter_name = adapter_name
        elif adapter_name is not None:
            self.active_logical_adapter_name = adapter_name

        self.active_role = role
        self._broadcast_active()

    def set_active_adapter(self, name: str) -> None:
        """Backward-compatible alias.

        Older call sites may pass a role alias such as ``policy`` or ``ref``.
        Newer call sites should use ``set_active(adapter_name, role)`` directly.
        """

        try:
            role = normalize_adapter_role(name)
        except ValueError:
            self.set_active(name, AdapterRole.SERVE.value)
            return
        if role == AdapterRole.BASE.value:
            self.set_active(None, role)
        else:
            self.set_active(self.default_logical_adapter, role)

    def has_active_train(self) -> bool:
        """Return whether a train workspace is currently active."""

        return self.current_train_owner is not None

    def active_train_owner(self) -> str | None:
        """Return the logical adapter currently owning the train workspace."""

        return self.current_train_owner

    def prepare_train_workspace(self, adapter_name: str, device: torch.device | None = None) -> bool:
        """Create or reuse the global train workspace for one logical adapter.

        Returns ``True`` when the workspace was freshly materialized and callers
        should rebuild train-time state such as the optimizer parameter list.
        """

        if self.current_train_owner is not None and self.current_train_owner != adapter_name:
            raise RuntimeError(
                f"train workspace already belongs to {self.current_train_owner}; cannot switch to {adapter_name}"
            )

        already_materialized = all(module.has_train_workspace() for _, _, module in self._manual_modules)
        if self.current_train_owner == adapter_name and already_materialized:
            return False

        if self._adapter_offload and device is not None:
            self.ensure_adapter_on_gpu(adapter_name, device)

        adapter_layers = self._serve_registry[adapter_name]
        for layer_idx, module_name, module in self._manual_modules:
            weights = adapter_layers[(layer_idx, module_name)]
            module.materialize_train(weights.a, weights.b)
        self.current_train_owner = adapter_name
        return True

    def clear_train_workspace(self) -> None:
        """Release train ownership after a successful publish or explicit discard."""

        for _, _, module in self._manual_modules:
            module.release_train()
        self.current_train_owner = None
        if self.active_role == AdapterRole.TRAIN.value:
            self.set_active(self.default_logical_adapter, AdapterRole.SERVE.value)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """Yield the currently materialized train workspace parameters."""

        for _, _, module in self._manual_modules:
            if module.train_a is not None:
                yield module.train_a
            if module.train_b is not None:
                yield module.train_b

    def move_all_lora_to_gpu(self, device: torch.device) -> int:
        """Move serve adapters to GPU and move train only when materialized.

        When ``adapter_offload`` is enabled, only the active adapter (and
        adapters already in the GPU-resident LRU pool) are moved.  Otherwise
        all registered adapters are moved (original behaviour).
        """
        self._device = device
        moved = 0
        with torch.no_grad():
            if self._adapter_offload:
                adapters_to_move = set(self._gpu_resident_adapters)
                adapters_to_move.add(self.active_logical_adapter_name)
                for adapter_name in adapters_to_move:
                    if adapter_name not in self._serve_registry:
                        continue
                    for weights in self._serve_registry[adapter_name].values():
                        if weights.a.data.device != device:
                            weights.a.data = weights.a.data.to(device)
                            moved += 1
                        if weights.b.data.device != device:
                            weights.b.data = weights.b.data.to(device)
                            moved += 1
            else:
                for adapter_layers in self._serve_registry.values():
                    for weights in adapter_layers.values():
                        if weights.a.data.device != device:
                            weights.a.data = weights.a.data.to(device)
                            moved += 1
                        if weights.b.data.device != device:
                            weights.b.data = weights.b.data.to(device)
                            moved += 1
            for _, _, module in self._manual_modules:
                if module.train_a is not None and module.train_a.data.device != device:
                    module.train_a.data = module.train_a.data.to(device)
                    moved += 1
                if module.train_b is not None and module.train_b.data.device != device:
                    module.train_b.data = module.train_b.data.to(device)
                    moved += 1

        if self._adapter_offload:
            self._gpu_resident_adapters.clear()
            for adapter_name in self.logical_adapter_names:
                if adapter_name not in self._serve_registry:
                    continue
                first_weights = next(iter(self._serve_registry[adapter_name].values()))
                if first_weights.a.data.device == device:
                    self._gpu_resident_adapters.append(adapter_name)
            # Offload any adapters that exceed the resident budget.
            # During init all adapters are created on the base model's
            # device; this call forces inactive extras to CPU.
            self._maybe_offload_inactive()
        return moved

    def export_serve_state(self, adapter_name: str) -> dict[str, torch.Tensor]:
        """Export one logical adapter's serve state."""

        state: dict[str, torch.Tensor] = {}
        adapter_layers = self._serve_registry[adapter_name]
        for (layer_idx, module_name), weights in adapter_layers.items():
            prefix = f"layers.{layer_idx}.{module_name}"
            state[f"{prefix}.lora_serve_a"] = weights.a.detach().cpu()
            state[f"{prefix}.lora_serve_b"] = weights.b.detach().cpu()
        return state

    def load_serve_state(self, adapter_name: str, state: dict[str, torch.Tensor]) -> None:
        """Load one logical adapter's serve state."""

        if adapter_name not in self.logical_adapter_names:
            self.register_logical_adapter(adapter_name, init_from=self.default_logical_adapter)
        adapter_layers = self._serve_registry[adapter_name]
        with torch.no_grad():
            for key, tensor in state.items():
                parts = key.split(".")
                if len(parts) < 4 or parts[0] != "layers":
                    continue
                layer_idx = int(parts[1])
                module_name = ".".join(parts[2:-1])
                leaf_name = parts[-1]
                weights = adapter_layers.get((layer_idx, module_name))
                if weights is None:
                    continue
                if leaf_name == "lora_serve_a":
                    weights.a.data.copy_(tensor.to(weights.a.device))
                elif leaf_name == "lora_serve_b":
                    weights.b.data.copy_(tensor.to(weights.b.device))
        if self.active_role == AdapterRole.SERVE.value and self.active_logical_adapter_name == adapter_name:
            self._broadcast_active()

    def export_train_state(self) -> dict[str, torch.Tensor] | None:
        """Export the current train workspace state if it exists."""

        if self.current_train_owner is None:
            return None
        state: dict[str, torch.Tensor] = {}
        for layer_idx, module_name, module in self._manual_modules:
            if module.train_a is None or module.train_b is None:
                return None
            prefix = f"layers.{layer_idx}.{module_name}"
            state[f"{prefix}.lora_train_a"] = module.train_a.detach().cpu()
            state[f"{prefix}.lora_train_b"] = module.train_b.detach().cpu()
        return state

    def load_train_state(
        self,
        adapter_name: str,
        state: dict[str, torch.Tensor],
        device: torch.device | None = None,
    ) -> None:
        """Restore the train workspace for one logical adapter."""

        _ = device
        self.prepare_train_workspace(adapter_name)
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
                if module is None or module.train_a is None or module.train_b is None:
                    continue
                if leaf_name == "lora_train_a":
                    module.train_a.data.copy_(tensor.to(module.train_a.device))
                elif leaf_name == "lora_train_b":
                    module.train_b.data.copy_(tensor.to(module.train_b.device))
        self.current_train_owner = adapter_name

    def commit_train_to_serve(self, adapter_name: str | None = None, *, clear_train: bool = True) -> None:
        """Publish the active train workspace into its serve adapter."""

        owner = adapter_name or self.current_train_owner
        if owner is None or self.current_train_owner != owner:
            raise RuntimeError("no matching active train workspace to commit")
        adapter_layers = self._serve_registry[owner]
        for layer_idx, module_name, module in self._manual_modules:
            weights = adapter_layers[(layer_idx, module_name)]
            module.commit_train_to_serve(weights.a, weights.b)
        if clear_train:
            self.clear_train_workspace()
        elif self.active_role == AdapterRole.SERVE.value and self.active_logical_adapter_name == owner:
            self._broadcast_active()

    def export_policy_state(self) -> dict[str, torch.Tensor] | None:
        """Backward-compatible alias for the current train state."""

        return self.export_train_state()

    def load_policy_state(self, state: dict[str, torch.Tensor]) -> None:
        """Backward-compatible alias using the default logical adapter."""

        self.load_train_state(self.default_logical_adapter, state)

    def get_layer_adapter(
        self,
        layer_idx: int,
        module_name: str,
        adapter_name: str | None = None,
        role: str = AdapterRole.SERVE.value,
    ) -> LoRAModuleWeights | None:
        """Return one layer/module's active LoRA weights for auxiliary code."""

        role = normalize_adapter_role(role)
        if role == AdapterRole.BASE.value:
            return None
        if role == AdapterRole.TRAIN.value:
            try:
                module = self.layer_plans[layer_idx].modules[module_name]
            except (IndexError, KeyError):
                return None
            if module.train_a is None or module.train_b is None:
                return None
            return LoRAModuleWeights(module.train_a, module.train_b, module.alpha, module.rank)
        if adapter_name is None:
            raise ValueError("serve weights require a logical adapter name")
        return self._serve_registry[adapter_name][(layer_idx, module_name)]
