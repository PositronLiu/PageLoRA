from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from core.types import TrainSample, TurnRecord
from core.utils import LOG
from multilora.classifier_router import (
    ClassifierRouter,
    load_classifier_router,
    save_classifier_router,
    train_classifier_router,
)
from multilora.router import BaseRouter, RouterDecision
from multilora.state import MultiLoRAState, MultiLoRAStateStore, TypeStats
from multilora.type_registry import TypeRegistry, normalize_tag
from runtime.task import CPUTask

if TYPE_CHECKING:
    from buffer.replay_buffer import ReplayBuffer
    from model.model_manager import ModelManager
    from runtime.scheduler import Scheduler
    from train.trainer import Trainer
    from runtime.judge_worker import JudgeTask


def _last_user_text(turn: TurnRecord) -> str:
    """Extract the most recent user message before the assistant response."""

    for message in reversed(turn.messages):
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                return content
    return ""


def _best_tag_from_votes(votes: list[Any]) -> tuple[str | None, float]:
    """Compute the most frequent non-empty vote tag and its vote ratio."""

    counts: dict[str, int] = {}
    total = 0
    for vote in votes:
        if not isinstance(vote, dict):
            continue
        raw_tag = vote.get("tag")
        if not isinstance(raw_tag, str):
            continue
        normalized = normalize_tag(raw_tag)
        if not normalized:
            continue
        counts[normalized] = counts.get(normalized, 0) + 1
        total += 1
    if not counts or total <= 0:
        return None, 0.0
    best_tag, best_count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return best_tag, best_count / float(total)


class MultiLoRAManager:
    """Single-user Level 3 orchestration for tags, router state, and adapters.

    This manager intentionally keeps the runtime simple:

    - one persistent state file
    - one append-only router sample file
    - at most one active train target at a time
    - one lightweight CPU-side classifier router
    """

    def __init__(
        self,
        *,
        model_manager: ModelManager,
        scheduler: Scheduler,
        state_path: str | Path,
        router_samples_path: str | Path,
        router_artifact_path: str | Path,
        rule_router: BaseRouter | None = None,
        auto_evolve: bool = True,
        min_samples_to_activate_type: int = 24,
        min_positive_samples_to_activate_type: int = 8,
        min_router_train_samples: int = 32,
        min_router_confidence: float = 0.6,
        min_tag_confidence: float = 0.5,
        max_prompt_tags: int = 24,
    ) -> None:
        self.model_manager = model_manager
        self.scheduler = scheduler
        self.rule_router = rule_router
        self.auto_evolve = auto_evolve
        self.state_store = MultiLoRAStateStore(
            state_path,
            general_adapter=model_manager.default_logical_adapter,
        )
        self.state: MultiLoRAState = self.state_store.load()
        self.type_registry = TypeRegistry(self.state)
        self.router_samples_path = Path(router_samples_path)
        self.router_artifact_path = Path(router_artifact_path)
        self.classifier_router: ClassifierRouter | None = None
        self._router_train_inflight = False
        self._known_router_samples = self._load_router_samples()
        self._trainer: Trainer | None = None
        self._buffer: ReplayBuffer | None = None
        self._lock = threading.Lock()

        # CLI thresholds override the cold-start defaults while still persisting
        # back into the state file for future restarts.
        self.state.thresholds.min_samples_to_activate_type = min_samples_to_activate_type
        self.state.thresholds.min_positive_samples_to_activate_type = min_positive_samples_to_activate_type
        self.state.thresholds.min_router_train_samples = min_router_train_samples
        self.state.thresholds.min_router_confidence = min_router_confidence
        self.state.thresholds.min_tag_confidence = min_tag_confidence
        self.state.thresholds.max_prompt_tags = max_prompt_tags

        self._restore_registered_adapters()
        self._load_classifier_if_available()

    def attach_training_runtime(self, trainer: Trainer, buffer: ReplayBuffer) -> None:
        """Attach the current single training runtime to Level 3 orchestration."""

        self._trainer = trainer
        self._buffer = buffer
        active = self.state.active_training_type
        if active:
            self._activate_training_target_locked(active)

    def route(self, text: str) -> RouterDecision:
        """Route one foreground request using the current Level 3 state."""

        with self._lock:
            if self.classifier_router is not None and self.state.router.mode == "classifier":
                return self.classifier_router.route(text)
            if self.state.router.mode == "rule" and self.rule_router is not None:
                return self.rule_router.route(text)
            return RouterDecision(
                adapter_name=self.state.general_adapter,
                confidence=0.0,
                rule_name="bootstrap_general",
                matched_terms=(),
                fallback_used=True,
            )

    def known_tags_for_prompt(self) -> list[str]:
        """Return a short, stable tag shortlist for judge prompting."""

        with self._lock:
            return self.type_registry.top_prompt_tags()

    def on_judge_finalized(
        self,
        task: JudgeTask,
        adapter_samples: list[TrainSample],
        dropped_neutral: int,
    ) -> list[TrainSample]:
        """Consume one judged turn and return the samples that should enter replay."""

        best_tag, tag_confidence = _best_tag_from_votes(task.votes)
        if best_tag is None:
            return adapter_samples

        reward = float(task.signal.binary_reward or 0.0) if task.signal is not None else 0.0
        turn = task.turn
        turn.judge_tag = best_tag
        turn.judge_tag_confidence = tag_confidence

        with self._lock:
            stats = self.type_registry.update_type(
                tag=best_tag,
                reward=reward,
                tag_confidence=tag_confidence,
            )
            self._append_router_sample_locked(
                turn=turn,
                tag=best_tag,
                reward=reward,
                tag_confidence=tag_confidence,
                dropped_neutral=dropped_neutral,
            )
            self._maybe_activate_type_locked(stats)
            replay_samples = self._select_replay_samples_locked(
                stats=stats,
                turn=turn,
                adapter_samples=adapter_samples,
            )
            self._maybe_schedule_router_train_locked()
            self.state_store.save(self.state)
            return replay_samples

    def _restore_registered_adapters(self) -> None:
        """Ensure state-declared adapters are available after restart."""

        for stats in self.state.types.values():
            if stats.adapter_name is None:
                continue
            self.model_manager.lora_runtime.register_logical_adapter(
                stats.adapter_name,
                init_from=self.state.general_adapter,
            )

    def _load_classifier_if_available(self) -> None:
        """Load the persisted classifier artifact when the state points to it."""

        artifact_path = self.state.router.artifact_path or str(self.router_artifact_path)
        if self.state.router.mode != "classifier":
            return
        if not Path(artifact_path).exists():
            return
        self.classifier_router = load_classifier_router(
            artifact_path,
            confidence_threshold=self.state.thresholds.min_router_confidence,
        )
        self.state.router.artifact_path = artifact_path

    def _load_router_samples(self) -> list[dict[str, Any]]:
        """Load the append-only router dataset once at startup."""

        if not self.router_samples_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.router_samples_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _append_router_sample_locked(
        self,
        *,
        turn: TurnRecord,
        tag: str,
        reward: float,
        tag_confidence: float,
        dropped_neutral: int,
    ) -> None:
        """Append one router-training example to the local JSONL dataset."""

        record = {
            "created_at": time.time(),
            "text": _last_user_text(turn),
            "label": tag,
            "reward": reward,
            "tag_confidence": tag_confidence,
            "logical_adapter_name": turn.logical_adapter_name,
            "router_rule": turn.router_rule,
            "router_confidence": turn.router_confidence,
            "router_fallback_used": turn.router_fallback_used,
            "dropped_neutral_for_adapter": dropped_neutral > 0,
        }
        self._known_router_samples.append(record)
        self.router_samples_path.parent.mkdir(parents=True, exist_ok=True)
        with self.router_samples_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _maybe_activate_type_locked(self, stats: TypeStats) -> None:
        """Create and select a new train target once a type becomes active."""

        if not self.auto_evolve:
            return
        if stats.status != "active":
            return
        adapter_name = stats.adapter_name or stats.type_name
        stats.adapter_name = adapter_name
        self.model_manager.lora_runtime.register_logical_adapter(
            adapter_name,
            init_from=self.state.general_adapter,
        )
        if self.state.active_training_type is None:
            self._activate_training_target_locked(stats.type_name)

    def _activate_training_target_locked(self, type_name: str) -> None:
        """Switch the single background trainer to one chosen active type."""

        stats = self.state.types.get(type_name)
        if stats is None:
            return
        self.state.active_training_type = type_name
        if self._buffer is not None:
            self._buffer.clear()
        if self._trainer is not None:
            self._trainer.set_logical_adapter_name(stats.adapter_name or type_name)
        LOG(f"MultiLoRA activated training target type={type_name}")

    def _select_replay_samples_locked(
        self,
        *,
        stats: TypeStats,
        turn: TurnRecord,
        adapter_samples: list[TrainSample],
    ) -> list[TrainSample]:
        """Choose which samples should enter the single replay buffer.

        Before any specialized adapter becomes active, samples keep training the
        general adapter. Once one active train target exists, only samples for
        that target continue into replay, and they are rewritten to that
        adapter name.
        """

        if not adapter_samples:
            return []
        active_type = self.state.active_training_type
        if active_type is None:
            return adapter_samples
        if active_type != stats.type_name:
            return []
        adapter_name = stats.adapter_name or stats.type_name
        routed_samples: list[TrainSample] = []
        for sample in adapter_samples:
            routed_sample = TrainSample(
                session_id=sample.session_id,
                turn_id=sample.turn_id,
                sample_type=sample.sample_type,
                prompt_ids=list(sample.prompt_ids),
                response_ids=list(sample.response_ids),
                old_logprobs=list(sample.old_logprobs),
                reward=float(sample.reward),
                teacher_logprobs=None if sample.teacher_logprobs is None else list(sample.teacher_logprobs),
                ref_logprobs=None if sample.ref_logprobs is None else list(sample.ref_logprobs),
                token_mask=None if sample.token_mask is None else list(sample.token_mask),
                meta=dict(sample.meta),
            )
            routed_sample.meta["logical_adapter_name"] = adapter_name
            routed_sample.meta["judge_tag"] = stats.type_name
            routed_sample.meta["judge_tag_confidence"] = turn.judge_tag_confidence
            routed_samples.append(routed_sample)
        return routed_samples

    def _router_training_examples_locked(self) -> list[tuple[str, str]]:
        """Build the labeled dataset currently eligible for classifier fitting."""

        labels = set(self.type_registry.labels_for_router())
        examples: list[tuple[str, str]] = []
        for record in self._known_router_samples:
            label = record.get("label")
            text = record.get("text")
            if not isinstance(label, str) or not isinstance(text, str):
                continue
            if label not in labels:
                continue
            if not text.strip():
                continue
            examples.append((text, label))
        return examples

    def _maybe_schedule_router_train_locked(self) -> None:
        """Submit one lightweight CPU classifier fit when enough data exists."""

        if self._router_train_inflight:
            return
        examples = self._router_training_examples_locked()
        if len(examples) < self.state.thresholds.min_router_train_samples:
            return

        general_adapter = self.state.general_adapter
        artifact_path = str(self.router_artifact_path)
        confidence_threshold = self.state.thresholds.min_router_confidence
        self._router_train_inflight = True

        def action() -> dict[str, Any] | None:
            router = train_classifier_router(
                general_adapter=general_adapter,
                labeled_texts=examples,
                confidence_threshold=confidence_threshold,
            )
            if router is None:
                return None
            save_classifier_router(router, artifact_path)
            return {
                "trained_samples": len(examples),
                "artifact_path": artifact_path,
                "router": router,
            }

        def on_done() -> None:
            self._router_train_inflight = False
            result = task.result
            if not isinstance(result, dict):
                return
            router = result.get("router")
            if not isinstance(router, ClassifierRouter):
                return
            with self._lock:
                self.classifier_router = router
                self.state.router.mode = "classifier"
                self.state.router.version += 1
                self.state.router.artifact_path = str(result["artifact_path"])
                self.state.router.last_trained_at = time.time()
                self.state.router.trained_samples = int(result["trained_samples"])
                self.state.router.label_space = self.type_registry.labels_for_router()
                self.state_store.save(self.state)
            LOG(
                f"MultiLoRA router trained samples={result['trained_samples']} "
                f"labels={len(router.labels)}"
            )

        task = CPUTask(
            owner="multilora-router-train",
            action=action,
            callback=on_done,
        )
        self.scheduler.submit_task_cpu(task)
