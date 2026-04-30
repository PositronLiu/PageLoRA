from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.utils import LOG, get_device, get_torch_dtype

class AdapterNames(Enum):
    POLICY = "policy"  # for training and inference
    REFERENCE = "ref"  # for reference

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

        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(config.target_modules),
        )
        # Create a trainable policy adapter and a frozen reference adapter.
        self.model = get_peft_model(self.model, lora_cfg, adapter_name=AdapterNames.POLICY.value)
        self.model.add_adapter(AdapterNames.REFERENCE.value, lora_cfg)
        self._freeze_adapter(AdapterNames.REFERENCE.value)
        self.model.set_adapter(AdapterNames.POLICY.value)
        self._enable_gradient_checkpointing_inputs()

        if self.cpu_offload:
            # Move only LoRA params to GPU, keep base model on CPU
            self._move_lora_to_gpu()
            LOG(
                f"Adapters ready (CPU offload) active={getattr(self.model, 'active_adapters', ['policy'])} "
                f"trainable={sum(1 for p in self.model.parameters() if p.requires_grad)}"
            )
        else:
            LOG(
                f"Adapters ready active={getattr(self.model, 'active_adapters', ['policy'])} "
                f"trainable={sum(1 for p in self.model.parameters() if p.requires_grad)}"
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

    def _move_lora_to_gpu(self) -> None:
        """Move only LoRA parameters to GPU, keep base model on CPU.

        This is the core of CPU offload: the base model stays on CPU,
        only LoRA adapter params (lora_A, lora_B) are moved to GPU.
        """
        lora_moved = 0
        for name, param in self.model.named_parameters():
            if "lora_" in name.lower():
                with torch.no_grad():
                    param.data = param.data.to(self.device)
                lora_moved += 1
        LOG(f"Moved {lora_moved} LoRA params to GPU (base model stays on CPU)")

    def _freeze_adapter(self, adapter_name: str) -> None:
        """Freeze parameters of a specific LoRA adapter.

        Args:
            adapter_name: Adapter name.
        Returns:
            None.
        """

        self.model.set_adapter(adapter_name)
        frozen = 0
        for name, param in self.model.named_parameters():
            if "lora" in name and adapter_name in name:
                param.requires_grad = False
                frozen += 1
        LOG(f"Adapter frozen name={adapter_name} params={frozen}")

    def set_policy(self) -> None:
        """Activate the policy adapter.

        Args:
            None.
        Returns:
            None.
        """

        self.model.set_adapter(AdapterNames.POLICY.value)
        LOG("Switched active adapter to policy")

    def set_reference(self) -> None:
        """Activate the reference adapter.

        Args:
            None.
        Returns:
            None.
        """

        self.model.set_adapter(AdapterNames.REFERENCE.value)
        LOG("Switched active adapter to ref")

    def export_policy_state(self) -> dict[str, torch.Tensor]:
        """Export the trainable policy adapter state for checkpointing."""

        policy_state: dict[str, torch.Tensor] = {}
        for name, tensor in self.model.state_dict().items():
            if AdapterNames.POLICY.value not in name:
                continue
            policy_state[name] = tensor.detach().cpu()
        return policy_state

    def load_policy_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a previously exported policy adapter state."""

        self.model.load_state_dict(state_dict, strict=False)

    @contextmanager
    def use_adapter(self, name: str) -> Iterator[None]:
        """Context manager to switch adapters temporarily.

        Args:
            name: Adapter name.
        Returns:
            None.
        """

        current = getattr(self.model, "active_adapter", None)
        if current is None:
            adapters = getattr(self.model, "active_adapters", None)
            current = adapters[0] if adapters else AdapterNames.POLICY.value
        self.model.set_adapter(name)
        try:
            yield
        finally:
            self.model.set_adapter(current)

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

        return (p for p in self.model.parameters() if p.requires_grad)
