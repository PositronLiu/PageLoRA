from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.utils import LOG, get_device, get_torch_dtype
from multilora.runtime import AdapterRole, LoRARuntime, normalize_adapter_role

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
    infer_cpu_offload: bool = False
    enable_multilora: bool = False
    enable_br_cache: bool = True
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
        self._inference_dispatcher = None
        self.default_logical_adapter = config.default_logical_adapter
        self.active_logical_adapter_name = config.default_logical_adapter
        if config.kvcache_mode == "hf" and config.enable_multilora:
            config.kvcache_mode = "br" if config.enable_br_cache else "full"
            config.kvcache_decompose = config.kvcache_mode == "br"
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
        )
        self.lora_runtime.set_active(self.default_logical_adapter, AdapterRole.SERVE.value)
        self._enable_gradient_checkpointing_inputs()

        lora_moved = self.lora_runtime.move_all_lora_to_gpu(self.device)
        LOG(
            f"Manual LoRA ready active={self.lora_runtime.active_logical_adapter_name} "
            f"lora_tensors_gpu={lora_moved}"
            f"{' (adapter offload)' if config.adapter_offload else ''}"
        )

        # Managed radix KV cache manager (lazy-init if enabled).
        self._kvcache_manager = None
        cache_mode = config.kvcache_mode
        if cache_mode == "hf" and config.kvcache_decompose:
            cache_mode = "br"
            config.kvcache_mode = cache_mode
        if cache_mode in {"br", "full"}:
            num_layers = len(self.lora_runtime.layers)
            if cache_mode == "br":
                from model.br_kvcache_manager import BRKVCacheManager

                self._kvcache_manager = BRKVCacheManager(num_layers=num_layers)
            else:
                from model.full_kvcache_manager import FullKVCacheManager

                self._kvcache_manager = FullKVCacheManager(num_layers=num_layers)
            self.lora_runtime.set_kvcache_manager(self._kvcache_manager)
            LOG(f"{type(self._kvcache_manager).__name__} initialized layers={num_layers}")

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

    def set_reference(self) -> None:
        """Backward-compatible alias that now maps reference to the base model."""

        self.set_base()

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

    def export_policy_state(self) -> dict[str, torch.Tensor] | None:
        """Backward-compatible alias for exporting the training adapter."""

        return self.export_train_state()

    def load_policy_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Backward-compatible alias for loading the training adapter."""

        self.load_train_state(self.active_logical_adapter_name, state_dict)

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

    def set_policy(self) -> None:
        """Backward-compatible alias that points to the serve adapter."""

        self.set_serve()

    def get_inference_dispatcher(self):
        """Return a cached inference dispatcher for CPU-offloaded generation.

        Construction is lazy because normal generation paths do not need this
        object, and building it repeatedly would add avoidable overhead to
        repeated foreground requests.
        """

        if self._inference_dispatcher is None:
            from model.inference_dispatcher import InferenceDispatcher

            self._inference_dispatcher = InferenceDispatcher(self.model, self.device, self.lora_runtime)
        return self._inference_dispatcher
