from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from core.types import TrainSample, TurnRecord
from core.utils import LOG
from multilora.classifier_router import (
    ClassifierRouter,
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


@dataclass
class RouterSample:
    """Lightweight pending sample used only for classifier-router training."""

    text: str
    label: str
    reward: float
    tag_confidence: float
    created_at: float


def _copy_train_sample(sample: TrainSample) -> TrainSample:
    """Detach one TrainSample before storing or rewriting it."""

    return TrainSample(
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
        self.router_pending_dir = self.router_samples_path.parent / "router_pending"
        self.adapter_pending_dir = self.router_samples_path.parent / "adapter_pending"
        self.classifier_router: ClassifierRouter | None = None
        self._router_train_inflight = False
        self._pending_router_by_label = self._load_router_pending()
        self._pending_adapter_by_label = self._load_adapter_pending()
        self._dirty_router_labels: set[str] = set()
        self._dirty_adapter_labels: set[str] = set()
        self._pending_flush_interval_s = 30.0
        self._last_pending_flush_at = time.time()
        self._trainer: Trainer | None = None
        self._buffer: ReplayBuffer | None = None
        self._lock = threading.RLock()

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
            for sample in self._drain_pending_adapter_samples_locked(active):
                self._buffer.push(sample)

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
            if stats.type_name == self.state.general_adapter:
                self.state_store.save(self.state)
                return adapter_samples
            self._stage_router_sample_locked(
                turn=turn,
                label=stats.type_name,
                reward=reward,
                tag_confidence=tag_confidence,
            )
            self._stage_adapter_samples_locked(stats.type_name, adapter_samples)
            self._maybe_activate_type_locked(stats)
            replay_samples = self._select_replay_samples_locked(
                stats=stats,
                turn=turn,
            )
            self._maybe_schedule_router_train_locked()
            self._maybe_flush_pending_samples_locked()
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
        self.classifier_router = ClassifierRouter.load(
            artifact_path,
            confidence_threshold=self.state.thresholds.min_router_confidence,
        )
        self.state.router.artifact_path = artifact_path

    def _label_path(self, root: Path, label: str) -> Path:
        """Return the pending JSONL path for one normalized label."""

        return root / f"{normalize_tag(label)}.jsonl"

    def _load_router_pending(self) -> dict[str, list[RouterSample]]:
        """Load unconsumed router samples grouped by label."""

        pending: dict[str, list[RouterSample]] = defaultdict(list)
        if not self.router_pending_dir.exists():
            return pending
        for path in self.router_pending_dir.glob("*.jsonl"):
            label = normalize_tag(path.stem)
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    pending[label].append(
                        RouterSample(
                            text=str(payload.get("text", "")),
                            label=normalize_tag(str(payload.get("label", label))),
                            reward=float(payload.get("reward", 0.0)),
                            tag_confidence=float(payload.get("tag_confidence", 0.0)),
                            created_at=float(payload.get("created_at", 0.0)),
                        )
                    )
        return pending

    def _load_adapter_pending(self) -> dict[str, list[TrainSample]]:
        """Load unconsumed adapter samples grouped by label."""

        pending: dict[str, list[TrainSample]] = defaultdict(list)
        if not self.adapter_pending_dir.exists():
            return pending
        for path in self.adapter_pending_dir.glob("*.jsonl"):
            label = normalize_tag(path.stem)
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    pending[label].append(TrainSample(**json.loads(line)))
        return pending

    def _stage_router_sample_locked(
        self,
        *,
        turn: TurnRecord,
        label: str,
        reward: float,
        tag_confidence: float,
    ) -> None:
        """Stage one router example in memory without synchronous disk IO."""

        label = normalize_tag(label)
        sample = RouterSample(
            created_at=time.time(),
            text=_last_user_text(turn),
            label=label,
            reward=reward,
            tag_confidence=tag_confidence,
        )
        if sample.text.strip():
            self._pending_router_by_label[label].append(sample)
            self._dirty_router_labels.add(label)

    def _stage_adapter_samples_locked(self, label: str, samples: list[TrainSample]) -> None:
        """Stage full adapter samples until their label is trainable."""

        label = normalize_tag(label)
        if not samples:
            return
        self._pending_adapter_by_label[label].extend(_copy_train_sample(sample) for sample in samples)
        self._dirty_adapter_labels.add(label)

    def _write_router_pending_label_locked(self, label: str) -> None:
        """Persist one label's unconsumed router samples."""

        path = self._label_path(self.router_pending_dir, label)
        samples = self._pending_router_by_label.get(label, [])
        path.parent.mkdir(parents=True, exist_ok=True)
        if not samples:
            path.unlink(missing_ok=True)
            return
        with path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")

    def _write_adapter_pending_label_locked(self, label: str) -> None:
        """Persist one label's unconsumed adapter samples."""

        path = self._label_path(self.adapter_pending_dir, label)
        samples = self._pending_adapter_by_label.get(label, [])
        path.parent.mkdir(parents=True, exist_ok=True)
        if not samples:
            path.unlink(missing_ok=True)
            return
        with path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")

    def flush_pending_samples(self) -> None:
        """Persist dirty pending sample buckets."""

        with self._lock:
            router_labels = tuple(self._dirty_router_labels)
            adapter_labels = tuple(self._dirty_adapter_labels)
            for label in router_labels:
                self._write_router_pending_label_locked(label)
            for label in adapter_labels:
                self._write_adapter_pending_label_locked(label)
            self._dirty_router_labels.difference_update(router_labels)
            self._dirty_adapter_labels.difference_update(adapter_labels)
            self._last_pending_flush_at = time.time()

    def _maybe_flush_pending_samples_locked(self) -> None:
        """Flush pending samples at a coarse interval to avoid per-sample IO."""

        if time.time() - self._last_pending_flush_at < self._pending_flush_interval_s:
            return
        router_labels = tuple(self._dirty_router_labels)
        adapter_labels = tuple(self._dirty_adapter_labels)
        for label in router_labels:
            self._write_router_pending_label_locked(label)
        for label in adapter_labels:
            self._write_adapter_pending_label_locked(label)
        self._dirty_router_labels.difference_update(router_labels)
        self._dirty_adapter_labels.difference_update(adapter_labels)
        self._last_pending_flush_at = time.time()

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
    ) -> list[TrainSample]:
        """Choose which samples should enter the single replay buffer.

        Collecting types remain pending-only. Only the active training target
        drains staged samples into replay.
        """

        active_type = self.state.active_training_type
        if active_type is None or active_type != stats.type_name:
            return []
        return self._drain_pending_adapter_samples_locked(
            stats.type_name,
            tag_confidence=turn.judge_tag_confidence,
        )

    def _drain_pending_adapter_samples_locked(
        self,
        type_name: str,
        *,
        tag_confidence: float | None = None,
    ) -> list[TrainSample]:
        """Move one label's pending adapter samples into replay-ready form."""

        stats = self.state.types.get(type_name)
        if stats is None:
            return []
        adapter_name = stats.adapter_name or stats.type_name
        pending = self._pending_adapter_by_label.pop(stats.type_name, [])
        if not pending:
            return []
        self._dirty_adapter_labels.add(stats.type_name)
        routed_samples: list[TrainSample] = []
        for sample in pending:
            routed_sample = _copy_train_sample(sample)
            routed_sample.meta["logical_adapter_name"] = adapter_name
            routed_sample.meta["judge_tag"] = stats.type_name
            if tag_confidence is not None:
                routed_sample.meta["judge_tag_confidence"] = tag_confidence
            routed_samples.append(routed_sample)
        return routed_samples

    def _pop_router_training_examples_locked(self) -> list[RouterSample]:
        """Pop currently trainable pending router samples."""

        labels = set(self.type_registry.labels_for_router())
        examples: list[RouterSample] = []
        for label in sorted(labels):
            samples = self._pending_router_by_label.pop(label, [])
            if not samples:
                continue
            examples.extend(samples)
            self._dirty_router_labels.add(label)
        return examples

    def _restore_router_training_examples_locked(self, examples: list[RouterSample]) -> None:
        """Put router examples back when background training fails."""

        for sample in examples:
            label = normalize_tag(sample.label)
            self._pending_router_by_label[label].append(sample)
            self._dirty_router_labels.add(label)

    def _maybe_schedule_router_train_locked(self) -> None:
        """Submit one incremental CPU classifier update when enough data exists."""

        if self._router_train_inflight:
            return
        examples = self._pop_router_training_examples_locked()
        if len(examples) < self.state.thresholds.min_router_train_samples:
            self._restore_router_training_examples_locked(examples)
            return

        artifact_path = str(self.router_artifact_path)
        if self.classifier_router is None:
            router_snapshot = ClassifierRouter.empty(
                general_adapter=self.state.general_adapter,
                confidence_threshold=self.state.thresholds.min_router_confidence,
            )
        else:
            router_snapshot = self.classifier_router.clone()
        self._router_train_inflight = True

        def action() -> dict[str, Any] | None:
            router = router_snapshot
            trained = router.train_batch([(sample.text, sample.label) for sample in examples])
            if trained <= 0:
                return None
            router.save(artifact_path)
            return {
                "trained_delta": trained,
                "trained_samples": router.trained_samples,
                "artifact_path": artifact_path,
                "router": router,
            }

        def on_done() -> None:
            self._router_train_inflight = False
            result = task.result
            if not isinstance(result, dict):
                with self._lock:
                    self._restore_router_training_examples_locked(examples)
                return
            router = result.get("router")
            if not isinstance(router, ClassifierRouter):
                with self._lock:
                    self._restore_router_training_examples_locked(examples)
                return
            with self._lock:
                self.classifier_router = router
                self.state.router.mode = "classifier"
                self.state.router.version += 1
                self.state.router.artifact_path = str(result["artifact_path"])
                self.state.router.last_trained_at = time.time()
                self.state.router.trained_samples = int(result["trained_samples"])
                self.state.router.label_space = self.type_registry.labels_for_router()
                consumed_labels = {sample.label for sample in examples}
                for label in consumed_labels:
                    self._write_router_pending_label_locked(label)
                    self._dirty_router_labels.discard(label)
                self.state_store.save(self.state)
            LOG(
                f"MultiLoRA router trained delta={result['trained_delta']} "
                f"total={result['trained_samples']} "
                f"labels={len(router.labels)}"
            )

        task = CPUTask(
            owner="multilora-router-train",
            action=action,
            callback=on_done,
        )
        self.scheduler.submit_task_cpu(task)
