from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import pytest

from test.inference import bench_flashinfer_generate, bench_hf_generate, bench_prefix_caching, bench_vllm_generate


def _tiny_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark-path", default="both")
    return parser


def test_split_hf_entrypoint_forces_hf_path(monkeypatch):
    captured = {}
    monkeypatch.setattr(bench_hf_generate, "build_parser", _tiny_parser)
    monkeypatch.setattr(bench_hf_generate, "run_benchmark", lambda args: captured.update(vars(args)))
    monkeypatch.setattr(sys, "argv", ["bench", "--model", "dummy", "--benchmark-path", "all"])

    bench_hf_generate.main()

    assert captured["benchmark_path"] == "hf"


def test_split_flashinfer_entrypoint_forces_flashinfer_path(monkeypatch):
    captured = {}
    monkeypatch.setattr(bench_flashinfer_generate, "build_parser", _tiny_parser)
    monkeypatch.setattr(bench_flashinfer_generate, "run_benchmark", lambda args: captured.update(vars(args)))
    monkeypatch.setattr(sys, "argv", ["bench", "--model", "dummy", "--benchmark-path", "all"])

    bench_flashinfer_generate.main()

    assert captured["benchmark_path"] == "flashinfer"


def test_split_vllm_entrypoint_forces_vllm_path(monkeypatch):
    captured = {}
    monkeypatch.setattr(bench_vllm_generate, "build_parser", _tiny_parser)
    monkeypatch.setattr(bench_vllm_generate, "run_benchmark", lambda args: captured.update(vars(args)))
    monkeypatch.setattr(sys, "argv", ["bench", "--model", "dummy", "--benchmark-path", "all"])

    bench_vllm_generate.main()

    assert captured["benchmark_path"] == "vllm"


def test_prefix_parser_accepts_single_and_branch_modes():
    parser = bench_prefix_caching.build_prefix_parser()

    single_args = parser.parse_args(["--model", "dummy", "--prefix-bench-mode", "single"])
    branch_args = parser.parse_args(
        [
            "--model",
            "dummy",
            "--prefix-backend",
            "both",
            "--prefix-bench-mode",
            "branch",
            "--prefix-reset-cache-between-runs",
            "false",
        ]
    )

    assert single_args.prefix_bench_mode == "single"
    assert branch_args.prefix_backend == "both"
    assert branch_args.prefix_bench_mode == "branch"
    assert branch_args.prefix_reset_cache_between_runs is False


def test_vllm_prefix_reset_requires_supported_successful_reset():
    reset = bench_prefix_caching._reset_vllm_prefix_cache
    reset(object(), False)
    with pytest.raises(RuntimeError, match="does not expose"):
        reset(object(), True)
    with pytest.raises(RuntimeError, match="reset failed"):
        reset(SimpleNamespace(reset_prefix_cache=lambda: False), True)
    reset(SimpleNamespace(reset_prefix_cache=lambda: True), True)
    reset(SimpleNamespace(reset_prefix_cache=lambda: None), True)


@pytest.mark.parametrize("cold", [True, False])
def test_vllm_prefix_sequences_reset_without_probe_or_mutating_args(monkeypatch, cold):
    bench = bench_prefix_caching
    args = bench.build_prefix_parser().parse_args(
        ["--model", "dummy", "--warmup-runs", "1", "--measure-runs", "2",
         "--prefix-reset-cache-between-runs", str(cold).lower()]
    )
    events = []
    llm = SimpleNamespace(get_tokenizer=lambda: object(), reset_prefix_cache=lambda: events.append("reset"))
    monkeypatch.setattr(bench, "_set_seed", lambda *a, **kw: None)
    monkeypatch.setattr(bench, "_make_vllm_llm", lambda a: llm)
    monkeypatch.setattr(bench, "_make_vllm_lora_request", lambda a: None)
    monkeypatch.setattr(bench, "_release_cuda_memory", lambda: None)
    monkeypatch.setattr(bench, "_build_prefix_prompt_ids", lambda *a, **kw: ([], [[1, 2], [1, 3]]))

    def generate(engine, ids, run_args, device, **kwargs):
        assert run_args.throughput_timing == "total"
        events.append(tuple(ids))
        return SimpleNamespace(total_s=0.5, generated_tokens=4)

    monkeypatch.setattr(bench, "_run_vllm_once", generate)
    stats = bench._run_vllm_prefix_mode(args, None, mode="branch")
    expected_sequence = (["reset"] if cold else []) + [(1, 2), (1, 3)]
    assert events == expected_sequence * 3
    assert args.throughput_timing == "decode"
    assert args.vllm_enable_prefix_caching is False
    assert stats.latencies_s == [0.5] * 4
