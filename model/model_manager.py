from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.utils import LOG, get_device, get_torch_dtype
from model.memory_aware import MemoryAwareCoordinator
from multilora.runtime import AdapterRole, LoRARuntime, normalize_adapter_role


def _select_single_adapter_cache_mode(config: object) -> str:
    """Resolve cache mode when MultiLoRA routing/evolution is disabled."""

    requested_mode = getattr(config, "kvcache_mode", "hf")
    if getattr(config, "enable_flashinfer_paged_kv", False) or requested_mode == "flashinfer":
        return "flashinfer"
    if requested_mode == "full" or not getattr(config, "enable_br_cache", True):
        return "full"
    return "hf"


@dataclass
class ModelConfig:
    """Configuration for model loading.

    Args:
        model_name: Model name or path.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout.
        target_modules: List of module names for LoRA.
        cpu_offload: Experimental fallback path that keeps base model on CPU.
            The default runtime path keeps the model resident on GPU.
        kvcache_decompose: Enable B+R decomposed KV cache for multi-LoRA
            sharing.  When enabled, a custom generation loop replaces
            ``model.generate()``.
    Returns:
        None.
    """

    model_name: str
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    cpu_offload: bool = False
    enable_multilora: bool = False
    enable_br_cache: bool = True
    enable_flashinfer_paged_kv: bool = False
    flashinfer_page_size: int = 32
    flashinfer_max_pages: int = 0
    flashinfer_workspace_mb: int = 128
    flashinfer_kv_append_mode: str = "manual"
    flashinfer_qk_norm: bool = False
    kvcache_mode: str = "hf"
    kvcache_decompose: bool = False
    logical_adapters: tuple[str, ...] = ("general",)
    default_logical_adapter: str = "general"
    adapter_offload: bool = False
    adapter_offload_lru_size: int = 1


class ModelManager:
    """Manage base model, LoRA adapters, and tokenizer.

    Args:
        config: ModelConfig.
        device: Torch device.
    Returns:
        None.
    """

    def __init__(self, config: ModelConfig, device: torch.device | None = None):
        self.config = config
        self.device = device or get_device()
        self.cpu_offload = config.cpu_offload
        self.default_logical_adapter = config.default_logical_adapter
        self.active_logical_adapter_name = config.default_logical_adapter
        if config.kvcache_mode == "hf":
            if config.enable_multilora:
                if config.enable_br_cache:
                    config.kvcache_mode = "br"
                elif config.enable_flashinfer_paged_kv:
                    config.kvcache_mode = "flashinfer"
                else:
                    config.kvcache_mode = "full"
                config.kvcache_decompose = config.kvcache_mode == "br"
            elif config.kvcache_decompose:
                config.kvcache_mode = "br"
                config.enable_br_cache = True
            else:
                config.kvcache_mode = _select_single_adapter_cache_mode(config)
                config.enable_br_cache = False
                config.enable_flashinfer_paged_kv = config.kvcache_mode == "flashinfer"
                config.kvcache_decompose = False
        dtype = get_torch_dtype()
        LOG(
            f"ModelManager init model={config.model_name} device={self.device} dtype={dtype} "
            f"cpu_offload={self.cpu_offload} enable_multilora={config.enable_multilora} "
            f"kvcache_mode={config.kvcache_mode} kvcache_decompose={config.kvcache_decompose}"
        )

        # Tokenizer drives prompt formatting and id conversion.
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        LOG("Tokenizer loaded")

        if self.cpu_offload:
            # Experimental fallback path kept for future offload experiments.
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            LOG("Base model loaded on CPU (cpu_offload=True)")
        else:
            # Default runtime path: keep the base model resident on GPU.
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                dtype=dtype,
                trust_remote_code=True,
            )
            self.model.to(self.device)
            LOG("Base model loaded")

        # Training defaults keep memory stable. Inference sessions temporarily
        # override these flags through ``inference_session``.
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()

        self._memory_coordinator: MemoryAwareCoordinator | None = None
        if config.adapter_offload:
            self._memory_coordinator = MemoryAwareCoordinator(
                target_fraction=0.85,
                device=self.device,
            )
            LOG("MemoryAwareCoordinator initialized for adapter+KV cache offload")

        self.lora_runtime = LoRARuntime(
            self.model,
            target_modules=config.target_modules,
            rank=config.lora_r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            logical_adapters=config.logical_adapters,
            default_logical_adapter=config.default_logical_adapter,
            adapter_offload=config.adapter_offload,
            adapter_offload_lru_size=config.adapter_offload_lru_size,
            memory_coordinator=self._memory_coordinator,
        )
        self.lora_runtime.set_active(self.default_logical_adapter, AdapterRole.SERVE.value)
        self._enable_gradient_checkpointing_inputs()

        lora_moved = self.lora_runtime.move_all_lora_to_gpu(self.device)
        LOG(
            f"Manual LoRA ready active={self.lora_runtime.active_logical_adapter_name} "
            f"lora_tensors_gpu={lora_moved}"
            f"{' (adapter offload)' if config.adapter_offload else ''}"
        )

        if self._memory_coordinator is not None:
            self._memory_coordinator.register_lora_runtime(self.lora_runtime)
            self._memory_coordinator.set_active_adapter(config.default_logical_adapter)

        # Managed radix KV cache manager (lazy-init if enabled).
        self._kvcache_manager = None
        cache_mode = config.kvcache_mode
        if cache_mode == "hf" and config.kvcache_decompose:
            cache_mode = "br"
            config.kvcache_mode = cache_mode
        if cache_mode in {"br", "full", "flashinfer"}:
            num_layers = len(self.lora_runtime.layers)
            if cache_mode == "br":
                from model.brcache.manager import BRKVCacheManager

                self._kvcache_manager = BRKVCacheManager(
                    num_layers=num_layers,
                    memory_coordinator=self._memory_coordinator,
                )
            elif cache_mode == "full":
                from model.fullcache.manager import FullKVCacheManager

                self._kvcache_manager = FullKVCacheManager(
                    num_layers=num_layers,
                    memory_coordinator=self._memory_coordinator,
                )
            else:
                from model.kvcache.paged_kvcache_manager import FlashInferPagedKVCacheManager

                self._kvcache_manager = FlashInferPagedKVCacheManager(
                    num_layers=num_layers,
                    page_size=config.flashinfer_page_size,
                    max_pages=config.flashinfer_max_pages,
                    workspace_mb=config.flashinfer_workspace_mb,
                    kv_append_mode=config.flashinfer_kv_append_mode,
                    device=self.device,
                    dtype=dtype,
                    model_config=self.model.config,
                    memory_coordinator=self._memory_coordinator,
                )
            self.lora_runtime.set_kvcache_manager(self._kvcache_manager)
            LOG(f"{type(self._kvcache_manager).__name__} initialized layers={num_layers}")
            if self._memory_coordinator is not None:
                self._memory_coordinator.register_kvcache_manager(self._kvcache_manager)
            if cache_mode == "flashinfer":
                from model.flashinfer.packed_ops import (
                    prepare_shared_packed_base_weights,
                    shared_packed_base_weight_storage_enabled,
                )

                if shared_packed_base_weight_storage_enabled():
                    stats = prepare_shared_packed_base_weights(
                        self.lora_runtime.layers,
                        self.lora_runtime.layer_plans,
                    )
                    LOG(
                        "FlashInfer shared packed base weights ready "
                        f"qkv={stats['qkv']} gate_up={stats['gate_up']}"
                    )

    def _enable_gradient_checkpointing_inputs(self) -> None:
        """Ensure checkpointed forwards still see grad-requiring inputs.

        With LoRA training the base model is mostly frozen, so enabling gradient
        checkpointing without forcing the input embeddings to require grad can
        produce the warning "None of the inputs have requires_grad=True" and
        break adapter training.
        """

        enable_fn = getattr(self.model, "enable_input_require_grads", None)
        if callable(enable_fn):
            enable_fn()
            LOG("Enabled input require grads for checkpointed training")
            return

        embeddings = self.model.get_input_embeddings()

        def _require_grads_hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)
            return output

        embeddings.register_forward_hook(_require_grads_hook)
        LOG("Registered embedding hook to require grads for checkpointed training")

    def list_logical_adapters(self) -> list[str]:
        """Return all registered logical adapters."""

        return self.lora_runtime.list_logical_adapters()

    def set_active_logical_adapter(self, adapter_name: str) -> None:
        """Update the default logical adapter used by foreground generation."""

        if adapter_name not in self.lora_runtime.list_logical_adapters():
            raise KeyError(f"Unknown logical adapter: {adapter_name}")
        self.active_logical_adapter_name = adapter_name

    def get_active_logical_adapter(self) -> str:
        """Return the default logical adapter currently selected."""

        return self.active_logical_adapter_name

    @property
    def kvcache_manager(self):
        """Return the managed radix KV cache manager, or ``None``."""
        return self._kvcache_manager

    def set_serve(self, adapter_name: str | None = None) -> None:
        """Activate the serve adapter used by foreground generation."""

        adapter_name = adapter_name or self.active_logical_adapter_name
        self.lora_runtime.set_active(adapter_name, AdapterRole.SERVE.value)
        self.active_logical_adapter_name = adapter_name
        LOG(f"Switched active adapter to serve adapter={adapter_name}")

    def set_train(self, adapter_name: str | None = None) -> None:
        """Activate the training adapter used by background updates."""

        adapter_name = adapter_name or self.lora_runtime.active_train_owner() or self.active_logical_adapter_name
        self.lora_runtime.set_active(adapter_name, AdapterRole.TRAIN.value)
        self.active_logical_adapter_name = adapter_name
        LOG(f"Switched active adapter to train adapter={adapter_name}")

    def set_base(self) -> None:
        """Activate pure-base forward mode with no LoRA delta."""

        self.lora_runtime.set_active(None, AdapterRole.BASE.value)
        LOG("Switched active adapter to base")

    def prepare_train_adapter(self, adapter_name: str | None = None) -> bool:
        """Materialize or reuse the global train workspace for one adapter.

        Returns ``True`` when the train workspace was freshly materialized and
        train-time state such as the optimizer should be rebuilt.
        """

        adapter_name = adapter_name or self.active_logical_adapter_name
        changed = self.lora_runtime.prepare_train_workspace(adapter_name, device=self.device)
        self.active_logical_adapter_name = adapter_name
        LOG(f"Prepared train workspace adapter={adapter_name}")
        return changed

    def clear_train_adapter(self) -> None:
        """Release the train workspace after it has been committed."""

        self.lora_runtime.clear_train_workspace()
        LOG("Cleared train workspace")

    def has_active_train(self) -> bool:
        """Return whether a train workspace is active."""

        return self.lora_runtime.has_active_train()

    def active_train_adapter(self) -> str | None:
        """Return the adapter name that currently owns the train workspace."""

        return self.lora_runtime.active_train_owner()

    def export_serve_state(self, adapter_name: str | None = None) -> dict[str, torch.Tensor]:
        """Export the serve adapter state for durable checkpointing."""

        return self.lora_runtime.export_serve_state(adapter_name or self.active_logical_adapter_name)

    def export_train_state(self) -> dict[str, torch.Tensor] | None:
        """Export the train adapter state for training resume."""

        return self.lora_runtime.export_train_state()

    def load_serve_state(self, adapter_name: str, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a previously exported serve adapter state."""

        self.lora_runtime.load_serve_state(adapter_name, state_dict)
        if self._kvcache_manager is not None:
            self._kvcache_manager.invalidate_adapter(adapter_name)

    def load_train_state(self, adapter_name: str, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a previously exported train adapter state."""

        self.lora_runtime.load_train_state(adapter_name, state_dict, device=self.device)

    def commit_train_to_serve(self, adapter_name: str | None = None) -> None:
        """Publish the completed training adapter to foreground serving."""

        adapter_name = adapter_name or self.lora_runtime.active_train_owner()
        self.lora_runtime.commit_train_to_serve(adapter_name)
        if adapter_name is not None:
            self.active_logical_adapter_name = adapter_name
            if self._kvcache_manager is not None:
                self._kvcache_manager.invalidate_adapter(adapter_name)
        LOG(f"Committed train adapter into serve adapter={adapter_name}")

    def stage_train_adapter_for_publish(self, adapter_name: str, publish_id: str) -> str:
        """Stage the current train workspace as an internal shadow serve adapter."""

        shadow_name = f"__publish_shadow__:{adapter_name}:{publish_id}"
        self.lora_runtime.stage_shadow_from_train(adapter_name, shadow_name)
        LOG(f"Staged train adapter for publish adapter={adapter_name} shadow={shadow_name}")
        return shadow_name

    def publish_staged_adapter(self, adapter_name: str, shadow_name: str) -> None:
        """Publish staged train weights and adopt staged KV cache, if available."""

        self.lora_runtime.commit_train_to_serve(adapter_name, clear_train=True)
        if self._kvcache_manager is not None:
            replace_cache = getattr(self._kvcache_manager, "replace_adapter_cache", None)
            if callable(replace_cache):
                replace_cache(shadow_name, adapter_name)
            else:
                invalidate = getattr(self._kvcache_manager, "invalidate_adapter", None)
                if callable(invalidate):
                    invalidate(adapter_name)
                    invalidate(shadow_name)
        self.lora_runtime.drop_shadow_adapter(shadow_name)
        self.active_logical_adapter_name = adapter_name
        LOG(f"Published staged adapter adapter={adapter_name} shadow={shadow_name}")

    @contextmanager
    def use_adapter(
        self,
        adapter_name: str | None = None,
        role: str = AdapterRole.SERVE.value,
    ) -> Iterator[None]:
        """Context manager to switch adapters temporarily.

        Args:
            adapter_name: Logical adapter name for serve/train modes.
            role: Runtime role: base, serve, or train.
        Returns:
            None.
        """

        current_name = self.lora_runtime.active_logical_adapter_name
        current_role = self.lora_runtime.active_role
        resolved_role = normalize_adapter_role(role)
        if resolved_role == AdapterRole.BASE.value:
            self.lora_runtime.set_active(None, resolved_role)
        else:
            resolved_name = adapter_name or self.active_logical_adapter_name
            self.lora_runtime.set_active(resolved_name, resolved_role)
        try:
            yield
        finally:
            if current_role == AdapterRole.BASE.value:
                self.lora_runtime.set_active(None, current_role)
            else:
                self.lora_runtime.set_active(current_name, current_role)

    @contextmanager
    def inference_session(self, enable_cache: bool = True) -> Iterator[None]:
        """Temporarily switch the model into inference-friendly mode.

        Args:
            enable_cache: Whether to enable KV cache during the session.
        Returns:
            None.
        """

        was_training = self.model.training
        had_checkpointing = bool(getattr(self.model, "is_gradient_checkpointing", False))
        previous_use_cache = getattr(self.model.config, "use_cache", None)

        self.model.eval()
        if had_checkpointing:
            self.model.gradient_checkpointing_disable()
        if previous_use_cache is not None:
            self.model.config.use_cache = enable_cache
        try:
            yield
        finally:
            if previous_use_cache is not None:
                self.model.config.use_cache = previous_use_cache
            if had_checkpointing:
                self.model.gradient_checkpointing_enable()
            if was_training:
                self.model.train()
            else:
                self.model.eval()

    @contextmanager
    def no_grad(self) -> Iterator[None]:
        """Context manager for no-grad inference.

        Args:
            None.
        Returns:
            None.
        """

        with torch.no_grad():
            yield

    def trainable_parameters(self):
        """Return trainable parameters.

        Args:
            None.
        Returns:
            Iterable of parameters.
        """

        return self.lora_runtime.trainable_parameters()
