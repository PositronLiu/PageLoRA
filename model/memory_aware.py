"""Memory-aware coordinator for KV cache + adapter offload.

Monitors GPU memory usage and coordinates eviction decisions between adapter
weights and KV cache blocks, prioritizing large cold items while protecting
the currently active adapter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import torch


class MemoryPressure(Enum):
    """GPU memory pressure levels derived from allocated / budget ratio."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OffloadCandidate:
    """One offloadable unit with eviction priority scoring.

    Priority is computed as ``bytes * kind_weight / (1 + recency)`` so that
    large, cold items are evicted first.  Adapter weights get a higher
    *kind_weight* than KV cache because they are cheaper to reload (a single
    ``.to(device)`` call vs. re-computing prefix KV).
    """

    kind: str
    name: str
    bytes: int
    last_access: int = 0
    touch_count: int = 0

    _KIND_WEIGHT: dict[str, float] = field(
        default_factory=lambda: {"adapter": 1.5, "kvcache_adapter": 1.0},
        init=False,
        repr=False,
    )

    @property
    def priority(self) -> float:
        w = self._KIND_WEIGHT.get(self.kind, 1.0)
        return self.bytes * w / ((1.0 + self.last_access) * (1.0 + self.touch_count))


@dataclass(frozen=True)
class TrainMemoryEstimate:
    """Predicted incremental GPU memory needed by one train step."""

    sample_count: int
    max_seq_len: int
    activation_bytes: int
    fixed_workspace_bytes: int

    @property
    def extra_bytes(self) -> int:
        return self.activation_bytes + self.fixed_workspace_bytes


class MemoryAwareCoordinator:
    """Coordinates adapter and KV cache offload based on GPU memory pressure."""

    def __init__(
        self,
        *,
        target_fraction: float = 0.85,
        device: torch.device | None = None,
        train_hidden_size: int = 4096,
        train_dtype_bytes: int = 2,
        train_activation_multiplier: float = 8.0,
        train_fixed_workspace_bytes: int = 512 * 1024 * 1024,
        train_safety_margin_bytes: int = 512 * 1024 * 1024,
        free_memory_fn: object | None = None,
    ) -> None:
        self.target_fraction = max(0.1, min(1.0, float(target_fraction)))
        self._device = device
        self.train_hidden_size = max(1, int(train_hidden_size))
        self.train_dtype_bytes = max(1, int(train_dtype_bytes))
        self.train_activation_multiplier = max(0.0, float(train_activation_multiplier))
        self.train_fixed_workspace_bytes = max(0, int(train_fixed_workspace_bytes))
        self.train_safety_margin_bytes = max(0, int(train_safety_margin_bytes))
        self._free_memory_fn = free_memory_fn

        self.adapter_weight_bytes: dict[str, int] = {}
        self.kvcache_bytes: dict[str, int] = {}
        self._adapter_last_access: dict[str, int] = {}
        self._kvcache_last_access: dict[str, int] = {}
        self._adapter_touch_count: dict[str, int] = {}
        self._kvcache_touch_count: dict[str, int] = {}
        self._access_clock = 0
        self._active_adapter = ""

        self._lora_runtime: object | None = None
        self._kvcache_manager: object | None = None

    # ── properties ─────────────────────────────────────────────────

    @property
    def active_adapter(self) -> str:
        return self._active_adapter

    @property
    def access_clock(self) -> int:
        return self._access_clock

    # ── registration ───────────────────────────────────────────────

    def register_lora_runtime(self, runtime: object) -> None:
        """Bind to a LoRA runtime and compute adapter weight byte sizes."""
        self._lora_runtime = runtime
        self._compute_adapter_sizes(runtime)

    def register_kvcache_manager(self, mgr: object) -> None:
        """Bind to a KV cache manager for offload delegation."""
        self._kvcache_manager = mgr

    def register_adapter_bytes(self, adapter_name: str, num_bytes: int) -> None:
        self.adapter_weight_bytes[adapter_name] = int(num_bytes)

    def register_kvcache_bytes(self, adapter_name: str, num_bytes: int) -> None:
        self.kvcache_bytes[adapter_name] = self.kvcache_bytes.get(adapter_name, 0) + int(num_bytes)

    def set_kvcache_bytes(self, adapter_name: str, num_bytes: int) -> None:
        """Set the current GPU-resident KV byte estimate for one adapter."""

        num_bytes = int(num_bytes)
        if num_bytes <= 0:
            self.kvcache_bytes.pop(adapter_name, None)
            return
        self.kvcache_bytes[adapter_name] = num_bytes

    def set_active_adapter(self, adapter_name: str) -> None:
        self._active_adapter = adapter_name

    # ── memory pressure ────────────────────────────────────────────

    def check_pressure(self) -> MemoryPressure:
        """Return current GPU memory pressure level."""
        if self._device is not None and self._device.type == "cpu":
            return MemoryPressure.LOW

        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            used = max(allocated, reserved)
            total = torch.cuda.get_device_properties(0).total_memory
        except (RuntimeError, AssertionError):
            return MemoryPressure.LOW

        budget = int(total * self.target_fraction)
        if budget <= 0:
            return MemoryPressure.CRITICAL

        ratio = used / budget
        if ratio < 0.70:
            return MemoryPressure.LOW
        if ratio < 0.90:
            return MemoryPressure.MODERATE
        if ratio < 1.0:
            return MemoryPressure.HIGH
        return MemoryPressure.CRITICAL

    # ── access tracking ────────────────────────────────────────────

    def notify_adapter_access(self, adapter_name: str) -> None:
        self._access_clock += 1
        self._adapter_last_access[adapter_name] = self._access_clock
        self._adapter_touch_count[adapter_name] = self._adapter_touch_count.get(adapter_name, 0) + 1

    def notify_kvcache_access(self, adapter_name: str) -> None:
        self._access_clock += 1
        self._kvcache_last_access[adapter_name] = self._access_clock
        self._kvcache_touch_count[adapter_name] = self._kvcache_touch_count.get(adapter_name, 0) + 1

    # ── size estimation ────────────────────────────────────────────

    def estimate_adapter_bytes(self, adapter_name: str) -> int:
        return self.adapter_weight_bytes.get(adapter_name, 0)

    def estimate_kvcache_bytes(self, adapter_name: str) -> int:
        return self.kvcache_bytes.get(adapter_name, 0)

    def estimate_train_memory(self, *, sample_count: int, max_seq_len: int) -> TrainMemoryEstimate:
        """Predict incremental train-step memory from batch shape.

        The model is intentionally simple and conservative: LoRA training with
        checkpointing is dominated by activation/workspace growth proportional
        to batch tokens, plus a fixed temporary workspace for logits/loss and
        allocator fragmentation.
        """

        sample_count = max(0, int(sample_count))
        max_seq_len = max(0, int(max_seq_len))
        token_count = sample_count * max_seq_len
        activation_bytes = int(
            token_count
            * self.train_hidden_size
            * self.train_dtype_bytes
            * self.train_activation_multiplier
        )
        return TrainMemoryEstimate(
            sample_count=sample_count,
            max_seq_len=max_seq_len,
            activation_bytes=activation_bytes,
            fixed_workspace_bytes=self.train_fixed_workspace_bytes,
        )

    # ── candidate building ─────────────────────────────────────────

    def get_offload_candidates(self) -> list[OffloadCandidate]:
        """Return offload candidates sorted by eviction priority (highest first).

        The active adapter (weights + KV cache) is excluded from the candidate
        list so it is never evicted.
        """
        candidates: list[OffloadCandidate] = []

        for name, num_bytes in self.adapter_weight_bytes.items():
            if name == self._active_adapter:
                continue
            if num_bytes <= 0:
                continue
            candidates.append(
                OffloadCandidate(
                    kind="adapter",
                    name=name,
                    bytes=num_bytes,
                    last_access=self._adapter_last_access.get(name, 0),
                    touch_count=self._adapter_touch_count.get(name, 0),
                )
            )

        for name, num_bytes in self.kvcache_bytes.items():
            if name == self._active_adapter:
                continue
            if num_bytes <= 0:
                continue
            candidates.append(
                OffloadCandidate(
                    kind="kvcache_adapter",
                    name=name,
                    bytes=num_bytes,
                    last_access=self._kvcache_last_access.get(name, 0),
                    touch_count=self._kvcache_touch_count.get(name, 0),
                )
            )

        candidates.sort(key=lambda c: c.priority, reverse=True)
        return candidates

    # ── offload ────────────────────────────────────────────────────

    def prepare_for_train(self, *, sample_count: int, max_seq_len: int) -> int:
        """Free cold GPU-resident state if the next train step is predicted to need it."""

        estimate = self.estimate_train_memory(sample_count=sample_count, max_seq_len=max_seq_len)
        required = estimate.extra_bytes + self.train_safety_margin_bytes
        free = self._free_gpu_bytes()
        shortfall = required - free
        if shortfall <= 0:
            return 0
        return self.offload_until_bytes(shortfall)

    def offload_until_bytes(self, target_bytes: int) -> int:
        """Offload candidates by priority until at least ``target_bytes`` is freed."""

        target_bytes = max(0, int(target_bytes))
        if target_bytes <= 0:
            return 0

        total_freed = 0
        for candidate in self.get_offload_candidates():
            if total_freed >= target_bytes:
                break
            total_freed += self._offload_one(candidate)
        return total_freed

    def maybe_offload(
        self,
        target_pressure: MemoryPressure = MemoryPressure.MODERATE,
    ) -> int:
        """Offload candidates until memory pressure drops to *target_pressure*.

        Returns the total bytes freed.
        """
        current = self.check_pressure()
        if _pressure_rank(current) <= _pressure_rank(target_pressure):
            return 0

        candidates = self.get_offload_candidates()
        if not candidates:
            return 0

        target_bytes = self._bytes_to_free(target_pressure)
        if target_bytes <= 0:
            return 0

        total_freed = 0
        for candidate in candidates:
            if total_freed >= target_bytes:
                break
            total_freed += self._offload_one(candidate)

        return total_freed

    def _offload_one(self, candidate: OffloadCandidate) -> int:
        """Offload a single candidate via the appropriate delegate."""
        if candidate.kind == "adapter":
            if self._lora_runtime is not None:
                offload_fn = getattr(self._lora_runtime, "offload_adapter_to_cpu", None)
                if callable(offload_fn):
                    offload_fn(candidate.name)
            self.adapter_weight_bytes.pop(candidate.name, None)
            return candidate.bytes

        if candidate.kind == "kvcache_adapter":
            if self._kvcache_manager is not None:
                offload_fn = getattr(self._kvcache_manager, "release_adapter_cache_for_train", None)
                if not callable(offload_fn):
                    offload_fn = getattr(self._kvcache_manager, "offload_adapter_cache", None)
                if callable(offload_fn):
                    offload_fn(candidate.name)
            self.kvcache_bytes.pop(candidate.name, None)
            return candidate.bytes

        return 0

    def _free_gpu_bytes(self) -> int:
        if callable(self._free_memory_fn):
            return max(0, int(self._free_memory_fn()))
        if self._device is not None and self._device.type == "cpu":
            return 2**63 - 1
        if not torch.cuda.is_available():
            return 2**63 - 1
        try:
            free, _total = torch.cuda.mem_get_info()
            return int(free)
        except (RuntimeError, AssertionError):
            return 2**63 - 1

    def _bytes_to_free(self, target: MemoryPressure) -> int:
        """Estimate bytes to free to reach *target* pressure level."""
        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            used = max(allocated, reserved)
            total = torch.cuda.get_device_properties(0).total_memory
            budget = int(total * self.target_fraction)
            if budget <= 0:
                return used
            # Target is ~75% of budget for MODERATE, ~60% for LOW
            if target == MemoryPressure.LOW:
                target_used = int(budget * 0.60)
            elif target == MemoryPressure.MODERATE:
                target_used = int(budget * 0.75)
            elif target == MemoryPressure.HIGH:
                target_used = int(budget * 0.95)
            else:
                return max(0, used - budget + 1)
            return max(0, used - target_used)
        except (RuntimeError, AssertionError):
            return 0

    # ── adapter size computation ───────────────────────────────────

    def _compute_adapter_sizes(self, runtime: object) -> None:
        """Walk the LoRA runtime's serve registry and record per-adapter byte sizes."""
        logical_names: list[str] = getattr(runtime, "logical_adapter_names", [])
        serve_registry: dict = getattr(runtime, "_serve_registry", {})
        for adapter_name in logical_names:
            layers = serve_registry.get(adapter_name, {})
            total_bytes = 0
            for weights in layers.values():
                a = getattr(weights, "a", None)
                b = getattr(weights, "b", None)
                if a is not None:
                    total_bytes += a.numel() * a.element_size()
                if b is not None:
                    total_bytes += b.numel() * b.element_size()
            if total_bytes > 0:
                self.adapter_weight_bytes[adapter_name] = total_bytes

    # ── serialization ──────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "adapter_weight_bytes": dict(self.adapter_weight_bytes),
            "kvcache_bytes": dict(self.kvcache_bytes),
            "adapter_last_access": dict(self._adapter_last_access),
            "kvcache_last_access": dict(self._kvcache_last_access),
            "adapter_touch_count": dict(self._adapter_touch_count),
            "kvcache_touch_count": dict(self._kvcache_touch_count),
            "access_clock": self._access_clock,
            "active_adapter": self._active_adapter,
        }

    def load_state_dict(self, state: dict) -> None:
        self.adapter_weight_bytes = state.get("adapter_weight_bytes", {})
        self.kvcache_bytes = state.get("kvcache_bytes", {})
        self._adapter_last_access = state.get("adapter_last_access", {})
        self._kvcache_last_access = state.get("kvcache_last_access", {})
        self._adapter_touch_count = state.get("adapter_touch_count", {})
        self._kvcache_touch_count = state.get("kvcache_touch_count", {})
        self._access_clock = state.get("access_clock", 0)
        self._active_adapter = state.get("active_adapter", "")


def _pressure_rank(p: MemoryPressure) -> int:
    return list(MemoryPressure).index(p)
