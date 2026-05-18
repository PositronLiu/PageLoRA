"""Performance tracing helpers for FlashInfer generation."""

from __future__ import annotations

import os
import time
from collections import defaultdict

import torch

from core.utils import LOG


class FlashInferPerfTrace:
    """Accumulate low-frequency performance counters for one generation."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.enabled = os.getenv("MINI_OPENCLAW_FLASHINFER_PERF_LOG", "1").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.sync_cuda = os.getenv("MINI_OPENCLAW_FLASHINFER_PERF_SYNC", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.decode_log_every = max(0, int(os.getenv("MINI_OPENCLAW_FLASHINFER_PERF_LOG_EVERY", "128")))
        self.sections: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.layer_sections: dict[str, dict[int, dict[str, float]]] = {
            "prefill": defaultdict(lambda: defaultdict(float)),
            "decode": defaultdict(lambda: defaultdict(float)),
        }
        self.layer_counts: dict[str, dict[int, int]] = {
            "prefill": defaultdict(int),
            "decode": defaultdict(int),
        }

    def tic(self) -> float:
        if self.enabled and self.sync_cuda and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def add(self, name: str, started: float, *, count: int = 1) -> None:
        if not self.enabled:
            return
        if self.sync_cuda and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.sections[name] += time.perf_counter() - started
        self.counts[name] += count

    def add_value(self, name: str, value: float, *, count: int = 1) -> None:
        if not self.enabled:
            return
        self.sections[name] += float(value)
        self.counts[name] += count

    def add_layer(self, mode: str, layer_idx: int, name: str, started: float) -> None:
        if not self.enabled:
            return
        if self.sync_cuda and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.layer_sections[mode][layer_idx][name] += time.perf_counter() - started

    def count_layer(self, mode: str, layer_idx: int) -> None:
        if self.enabled:
            self.layer_counts[mode][layer_idx] += 1

    def _section(self, name: str) -> float:
        return self.sections.get(name, 0.0)

    def log_decode_progress(
        self,
        *,
        generated_tokens: int,
        cached_tokens: int,
        current_pos: int,
        started: float,
        cache_snapshot: dict[str, float],
    ) -> None:
        if not self.enabled or not self.decode_log_every:
            return
        if generated_tokens % self.decode_log_every != 0:
            return
        elapsed = time.perf_counter() - started
        tps = generated_tokens / elapsed if elapsed > 0 else 0.0
        LOG(
            "[flashinfer_perf] decode_progress "
            f"generated={generated_tokens} cached={cached_tokens} current_pos={current_pos} "
            f"elapsed_s={elapsed:.4f} tokens_per_s={tps:.4f} "
            f"decode_plan_s={self._section('decode_plan_s'):.4f} "
            f"decode_layers_s={self._section('decode_layers_s'):.4f} "
            f"decode_lm_head_s={self._section('decode_lm_head_s'):.4f} "
            f"manager_page_table_s={cache_snapshot.get('manager_build_page_table_s', 0.0):.4f} "
            f"store_dense_write_s={cache_snapshot.get('store_dense_write_s', 0.0):.4f} "
            f"allocated_pages={cache_snapshot.get('manager_allocated_pages', 0.0):.0f} "
            f"dense_capacity_pages={cache_snapshot.get('store_dense_capacity_pages', 0.0):.0f}"
        )

    def log_summary(
        self,
        *,
        prompt_tokens: int,
        prefill_tokens: int,
        generated_tokens: int,
        cached_generated_tokens: int,
        cache_snapshot: dict[str, float],
        request_span_count: int,
        request_page_chunks: int,
    ) -> None:
        if not self.enabled:
            return
        if self.sync_cuda and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        total_s = self._section("generation_total_s")
        decode_s = self._section("decode_total_s")
        decode_tps = generated_tokens / decode_s if decode_s > 0 else 0.0
        LOG(
            "[flashinfer_perf] summary "
            f"prompt_tokens={prompt_tokens} prefill_tokens={prefill_tokens} "
            f"generated={generated_tokens} cached_generated={cached_generated_tokens} "
            f"total_s={total_s:.4f} decode_s={decode_s:.4f} decode_tokens_per_s={decode_tps:.4f} "
            f"perf_sync_cuda={int(self.sync_cuda)}"
        )
        section_names = [
            "backend_create_s",
            "begin_request_s",
            "prefill_total_s",
            "prefill_input_s",
            "prefill_reserve_s",
            "prefill_plan_s",
            "prefill_layers_s",
            "prefill_lm_head_s",
            "decode_total_s",
            "decode_sample_s",
            "decode_embed_s",
            "decode_reserve_s",
            "decode_plan_s",
            "decode_layers_s",
            "decode_lm_head_s",
            "commit_s",
            "decode_text_s",
        ]
        LOG(
            "[flashinfer_perf] sections "
            + " ".join(f"{name}={self._section(name):.6f}" for name in section_names)
        )
        cache_names = [
            "manager_begin_request_s",
            "manager_radix_lookup_s",
            "manager_usable_prefix_s",
            "manager_prefix_span_slice_s",
            "manager_reserve_chunk_s",
            "manager_build_page_table_s",
            "manager_append_full_s",
            "manager_commit_request_s",
            "manager_radix_insert_coverage_s",
            "store_reserve_spans_s",
            "store_write_layer_s",
            "store_piece_slice_s",
            "store_page_metadata_s",
            "store_dense_write_s",
            "store_flashinfer_append_s",
            "store_dense_realloc_s",
        ]
        LOG(
            "[flashinfer_perf] kvcache "
            + " ".join(f"{name}={cache_snapshot.get(name, 0.0):.6f}" for name in cache_names)
            + " "
            + " ".join(
                f"{name}={cache_snapshot.get(name, 0.0):.0f}"
                for name in [
                    "manager_build_page_table_calls",
                    "manager_append_full_calls",
                    "store_write_layer_calls",
                    "store_flashinfer_append_calls",
                    "store_dense_realloc_count",
                    "manager_allocated_pages",
                    "store_dense_capacity_pages",
                ]
            )
            + f" request_spans={request_span_count} request_page_chunks={request_page_chunks}"
        )
        for mode in ("prefill", "decode"):
            self._log_layer_summary(mode)

    def _log_layer_summary(self, mode: str) -> None:
        layers = self.layer_sections[mode]
        if not layers:
            return
        totals = {
            layer_idx: sections.get("total_s", 0.0)
            for layer_idx, sections in layers.items()
        }
        top_layers = sorted(totals.items(), key=lambda item: item[1], reverse=True)[:5]
        aggregate: dict[str, float] = defaultdict(float)
        for sections in layers.values():
            for name, value in sections.items():
                aggregate[name] += value
        LOG(
            f"[flashinfer_perf] {mode}_layers "
            + " ".join(f"{name}={aggregate.get(name, 0.0):.6f}" for name in [
                "total_s",
                "qkv_s",
                "input_norm_s",
                "qkv_project_s",
                "rope_s",
                "kv_append_s",
                "layer_cache_s",
                "attn_run_s",
                "post_s",
                "o_layout_s",
                "o_proj_s",
                "post_residual_s",
                "post_norm_s",
                "mlp_gate_up_s",
                "mlp_act_s",
                "mlp_down_s",
                "mlp_s",
                "post_final_residual_s",
            ])
            + " top_total="
            + ",".join(
                f"{layer_idx}:{total_s:.6f}/{self.layer_counts[mode].get(layer_idx, 0)}"
                for layer_idx, total_s in top_layers
            )
        )
