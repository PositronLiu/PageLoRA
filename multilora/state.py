from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TypeStats:
    """Persistent aggregate state for one discovered task type."""

    type_name: str
    status: str = "collecting"
    sample_count: int = 0
    positive_count: int = 0
    neutral_count: int = 0
    negative_count: int = 0
    avg_reward: float = 0.0
    avg_tag_confidence: float = 0.0
    adapter_name: str | None = None
    adapter_version: int = 0
    last_updated_at: float = 0.0


@dataclass
class RouterModeState:
    """Persistent router state and artifact metadata."""

    mode: str = "empty"
    version: int = 0
    artifact_path: str | None = None
    last_trained_at: float | None = None
    trained_samples: int = 0
    label_space: list[str] = field(default_factory=list)


@dataclass
class ThresholdConfig:
    """Level 3 thresholds kept in the persistent state file."""

    min_samples_to_activate_type: int = 24
    min_positive_samples_to_activate_type: int = 8
    min_router_train_samples: int = 32
    min_router_confidence: float = 0.6
    min_tag_confidence: float = 0.5
    max_prompt_tags: int = 24


@dataclass
class MultiLoRAState:
    """Top-level persistent state for Level 3 MultiLoRA evolution."""

    version: int
    updated_at: float
    general_adapter: str
    active_training_type: str | None
    router: RouterModeState
    thresholds: ThresholdConfig
    types: dict[str, TypeStats] = field(default_factory=dict)


def build_default_state(general_adapter: str) -> MultiLoRAState:
    """Build the minimal cold-start state used on first boot."""

    return MultiLoRAState(
        version=1,
        updated_at=time.time(),
        general_adapter=general_adapter,
        active_training_type=None,
        router=RouterModeState(mode="empty"),
        thresholds=ThresholdConfig(),
        types={},
    )


def _type_from_dict(raw: dict[str, Any]) -> TypeStats:
    return TypeStats(**raw)


def _router_from_dict(raw: dict[str, Any]) -> RouterModeState:
    return RouterModeState(**raw)


def _thresholds_from_dict(raw: dict[str, Any]) -> ThresholdConfig:
    return ThresholdConfig(**raw)


def state_from_dict(raw: dict[str, Any], *, general_adapter: str) -> MultiLoRAState:
    """Decode one JSON dictionary into the runtime dataclass form."""

    router = _router_from_dict(raw.get("router", {}))
    thresholds = _thresholds_from_dict(raw.get("thresholds", {}))
    type_map = {
        type_name: _type_from_dict(type_payload)
        for type_name, type_payload in raw.get("types", {}).items()
    }
    return MultiLoRAState(
        version=int(raw.get("version", 1)),
        updated_at=float(raw.get("updated_at", time.time())),
        general_adapter=str(raw.get("general_adapter", general_adapter)),
        active_training_type=raw.get("active_training_type"),
        router=router,
        thresholds=thresholds,
        types=type_map,
    )


def state_to_dict(state: MultiLoRAState) -> dict[str, Any]:
    """Convert the persistent state into a JSON-serializable dictionary."""

    return {
        "version": state.version,
        "updated_at": state.updated_at,
        "general_adapter": state.general_adapter,
        "active_training_type": state.active_training_type,
        "router": asdict(state.router),
        "thresholds": asdict(state.thresholds),
        "types": {type_name: asdict(type_stats) for type_name, type_stats in state.types.items()},
    }


class MultiLoRAStateStore:
    """Tiny JSON state store used by the single-user local runtime.

    The store intentionally does not implement any multi-process locking or
    remote coordination. A single local runtime owns the file.
    """

    def __init__(self, path: str | Path, *, general_adapter: str) -> None:
        self.path = Path(path)
        self.general_adapter = general_adapter

    def load(self) -> MultiLoRAState:
        """Load the state file, or return a cold-start default state."""

        if not self.path.exists():
            return build_default_state(self.general_adapter)
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return state_from_dict(payload, general_adapter=self.general_adapter)

    def save(self, state: MultiLoRAState) -> None:
        """Atomically write the current state to disk."""

        state.updated_at = time.time()
        payload = state_to_dict(state)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.path.name + ".",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
