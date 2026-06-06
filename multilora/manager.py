"""Compatibility facade for the removed router/evolution MultiLoRA path.

The runtime still supports multiple logical LoRA adapters for multi-agent
serving/training. Judge-driven task tagging, automatic adapter promotion, and
classifier/router training are intentionally disabled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from buffer.replay_buffer import ReplayBuffer
    from train.trainer import Trainer


class MultiLoRAManager:
    """No-op compatibility layer for legacy MultiLoRA orchestration callers."""

    def __init__(
        self,
        *,
        model_manager: object,
        scheduler: object,
        state_path: str | object | None = None,
        router_samples_path: str | object | None = None,
        router_artifact_path: str | object | None = None,
        rule_router: object | None = None,
        auto_evolve: bool = False,
        **_: Any,
    ) -> None:
        self.model_manager = model_manager
        self.scheduler = scheduler
        self.state_path = state_path
        self.router_samples_path = router_samples_path
        self.router_artifact_path = router_artifact_path
        self.rule_router = rule_router
        self.auto_evolve = False

    def route(self, text: str) -> None:
        """Automatic adapter routing is disabled."""

        return None

    def attach_training_runtime(self, trainer: "Trainer", buffer: "ReplayBuffer") -> None:
        return None

    def flush_pending_samples(self) -> None:
        return None
