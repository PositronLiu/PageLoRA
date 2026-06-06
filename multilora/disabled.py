"""No-op facade used when MultiLoRA orchestration is disabled."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from train.trainer import Trainer
    from buffer.replay_buffer import ReplayBuffer


class DisabledMultiLoRAManager:
    """Keep AgentRuntime wiring cheap in single-adapter mode."""

    def route(self, text: str) -> None:
        return None

    def attach_training_runtime(self, trainer: "Trainer", buffer: "ReplayBuffer") -> None:
        return None

    def flush_pending_samples(self) -> None:
        return None
