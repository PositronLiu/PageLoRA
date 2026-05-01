from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.utils import LOG, get_device, get_torch_dtype
from model.lora_runtime import AdapterNames, LoRARuntime

@dataclass
class ModelConfig:
    """Configuration for model loading.

    Args:
        model_name: Model name or path.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout.
        target_modules: List of module names for LoRA.
        cpu_offload: If True, keep base model on CPU and only move LoRA params to GPU.
    Returns:
        None.
    """

    model_name: str
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    cpu_offload: bool = True
    infer_cpu_offload: bool = False


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
        dtype = get_torch_dtype()
        LOG(f"ModelManager init model={config.model_name} device={self.device} dtype={dtype} cpu_offload={self.cpu_offload}")

        # Tokenizer drives prompt formatting and id conversion.
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        LOG("Tokenizer loaded")

        if self.cpu_offload:
            # CPU offload mode: load base model to CPU first, then only move LoRA params to GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float32,  # Load in FP32 on CPU to save memory
                trust_remote_code=True,
            )
            # Base model stays on CPU - no .to(device) here
            LOG("Base model loaded on CPU (cpu_offload=True)")
        else:
            # Normal mode: load directly to GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                dtype=dtype,
                trust_remote_code=True,
            )
            self.model.to(self.device)
            LOG("Base model loaded")

        # Training defaults keep memory stable.
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()

        self.lora_runtime = LoRARuntime(
            self.model,
            target_modules=config.target_modules,
            rank=config.lora_r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        self.lora_runtime.set_active_adapter(AdapterNames.POLICY.value)
        self._enable_gradient_checkpointing_inputs()

        if self.cpu_offload:
            lora_moved = self.lora_runtime.move_all_lora_to_gpu(self.device)
            LOG(
                f"Manual LoRA ready (CPU offload) active={self.lora_runtime.active_adapter_name} "
                f"lora_tensors_gpu={lora_moved}"
            )
        else:
            lora_moved = self.lora_runtime.move_all_lora_to_gpu(self.device)
            LOG(
                f"Manual LoRA ready active={self.lora_runtime.active_adapter_name} "
                f"lora_tensors_gpu={lora_moved}"
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

    def set_policy(self) -> None:
        """Activate the policy adapter.

        Args:
            None.
        Returns:
            None.
        """
        self.lora_runtime.set_active_adapter(AdapterNames.POLICY.value)
        LOG("Switched active adapter to policy")

    def set_reference(self) -> None:
        """Activate the reference adapter.

        Args:
            None.
        Returns:
            None.
        """
        self.lora_runtime.set_active_adapter(AdapterNames.REFERENCE.value)
        LOG("Switched active adapter to ref")

    def export_policy_state(self) -> dict[str, torch.Tensor]:
        """Export the trainable policy adapter state for checkpointing."""

        return self.lora_runtime.export_policy_state()

    def load_policy_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a previously exported policy adapter state."""

        self.lora_runtime.load_policy_state(state_dict)

    @contextmanager
    def use_adapter(self, name: str) -> Iterator[None]:
        """Context manager to switch adapters temporarily.

        Args:
            name: Adapter name.
        Returns:
            None.
        """

        current = self.lora_runtime.active_adapter_name
        self.lora_runtime.set_active_adapter(name)
        try:
            yield
        finally:
            self.lora_runtime.set_active_adapter(current)

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

    def get_inference_dispatcher(self):
        """Return a cached inference dispatcher for CPU-offloaded generation.

        Construction is lazy because normal generation paths do not need this
        object, and building it repeatedly would add avoidable overhead to
        repeated foreground requests.
        """

        if self._inference_dispatcher is None:
            from model.inference_dispatcher import InferenceDispatcher

            self._inference_dispatcher = InferenceDispatcher(self.model, self.device)
        return self._inference_dispatcher
