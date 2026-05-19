from __future__ import annotations

import argparse
import sys

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
