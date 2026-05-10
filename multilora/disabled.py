"""No-op facade used when MultiLoRA orchestration is disabled."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.types import TrainSample
    from runtime.judge_worker import JudgeTask
    from train.trainer import Trainer
    from buffer.replay_buffer import ReplayBuffer


class DisabledMultiLoRAManager:
    """Keep AgentRuntime wiring cheap in single-adapter mode."""

    def route(self, text: str) -> None:
        return None

    def known_tags_for_prompt(self) -> list[str]:
        return []

    def on_judge_finalized(
        self,
        task: "JudgeTask",
        adapter_samples: list["TrainSample"],
        dropped_neutral: int,
    ) -> list["TrainSample"]:
        return adapter_samples

    def attach_training_runtime(self, trainer: "Trainer", buffer: "ReplayBuffer") -> None:
        return None

    def flush_pending_samples(self) -> None:
        return None
