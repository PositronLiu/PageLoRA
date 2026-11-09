"""Reproducible, sequential single-GPU course experiments (standard library only)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def build_jobs(args):
    common = ["--model", args.model, "--device", "cuda:0", "--adapter-role", "base",
              "--ignore-eos", "--sampling-mode", "greedy_fast", "--compute-logprobs", "false",
              "--repetition-penalty", "1.0", "--seed", "42",
              "--warmup-runs", str(args.warmup_runs), "--measure-runs", str(args.measure_runs)]
    jobs = []
    for length in args.prompt_lengths:
        tokens = ["--target-prompt-tokens", str(length), "--max-new-tokens", str(args.output_tokens)]
        if args.suite == "prefix":
            for backend in ("flashinfer", "vllm"):
                for reset in ("true", "false"):
                    extra = ["--prefix-backend", backend, "--prefix-bench-mode", "branch",
                             "--shared-prefix-tokens", str(length), "--branch-suffix-tokens", "64",
                             "--prefix-requests", "4", "--prefix-reset-cache-between-runs", reset,
                             "--throughput-timing", "total"]
                    if backend == "flashinfer":
                        extra += ["--flashinfer-cuda-graph-decode"]
                    else:
                        extra += ["--vllm-tensor-parallel-size", "1", "--vllm-dtype", "bfloat16",
                                  "--vllm-gpu-memory-utilization", "0.8",
                                  "--vllm-max-model-len", str(args.max_model_len)]
                    jobs.append((f"prefix_{backend}_{length}_reset_{reset}",
                                 [sys.executable, "-m", "test.inference.bench_prefix_caching"] + common + tokens + extra))
        else:
            for backend, graph in (("hf", False), ("vllm", False), ("flashinfer", False), ("flashinfer", True)):
                extra = ["--throughput-timing", "total"]
                if backend == "flashinfer":
                    extra += ["--flashinfer-cache-policy", "cold"]
                    if graph:
                        extra += ["--flashinfer-cuda-graph-decode"]
                if backend == "vllm":
                    extra += ["--vllm-tensor-parallel-size", "1", "--vllm-dtype", "bfloat16",
                              "--vllm-gpu-memory-utilization", "0.8",
                              "--vllm-max-model-len", str(args.max_model_len)]
                suffix = f"_graph_{int(graph)}" if backend == "flashinfer" else ""
                jobs.append((f"{backend}_{length}{suffix}",
                             [sys.executable, "-m", f"test.inference.bench_{backend}_generate"] + common + tokens + extra))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", choices=("core", "prefix"), default="core")
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[2048])
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measure-runs", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu", default="0", help="Physical GPU exposed as logical cuda:0.")
    parser.add_argument("--output-dir", default="course_results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if "," in args.gpu or not args.gpu.strip():
        parser.error("--gpu must select exactly one GPU index or UUID")
    if min(args.prompt_lengths + [args.output_tokens, args.measure_runs, args.max_model_len]) <= 0 or args.warmup_runs < 0:
        parser.error("lengths and measured runs must be positive; warmup runs must be non-negative")
    # Prompt builders add chat-template and synthetic-block overhead.
    if max(args.prompt_lengths) + args.output_tokens + 512 > args.max_model_len:
        parser.error("increase --max-model-len to allow prompt/template overhead and generated tokens")
    jobs = build_jobs(args)
    for name, command in jobs:
        print(name, shlex.join(command), flush=True)
    if args.dry_run:
        return
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = Path(args.output_dir).resolve() / f"{args.suite}_{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1",
               MINI_OPENCLAW_FLASHINFER_PERF_LOG="1", MINI_OPENCLAW_FLASHINFER_PERF_SYNC="0",
               MINI_OPENCLAW_FLASHINFER_TORCH_COMPILE="0",
               MINI_OPENCLAW_FLASHINFER_COMPILE_MLP="0", MINI_OPENCLAW_FLASHINFER_COMPILE_POST="0")

    def record(command):
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, errors="replace")
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

    metadata = {"utc": stamp, "platform": platform.platform(), "python": sys.version,
                "args": vars(args), "environment": {k: v for k, v in env.items() if k.startswith(("CUDA_", "MINI_OPENCLAW_"))},
                "git": record(["git", "rev-parse", "HEAD"]),
                "diff": record(["git", "diff", "--stat"]),
                "packages": record([sys.executable, "-m", "pip", "freeze"])}
    metadata["torch_preflight"] = record([
        sys.executable, "-c",
        "import torch; print(torch.__version__, torch.version.cuda); "
        "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
        "print(torch.cuda.get_device_name(0)); "
        "assert torch.cuda.is_bf16_supported(), "
        "'This matrix requires BF16 to match the explicit vLLM dtype'",
    ])
    try:
        metadata["gpu"] = record(["nvidia-smi"])
    except FileNotFoundError:
        raise SystemExit("nvidia-smi not found; run this script on the course CUDA server")
    (destination / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if metadata["torch_preflight"]["returncode"]:
        raise SystemExit(f"CUDA/BF16 preflight failed; inspect {destination / 'metadata.json'}")
    outcomes = []
    for name, command in jobs:
        log = destination / f"{name}.log"
        print(f"Running {name}; log: {log}", flush=True)
        with log.open("w", encoding="utf-8") as output:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, text=True)
        outcomes.append({"name": name, "command": command, "returncode": result.returncode, "log": str(log)})
        (destination / "runs.json").write_text(json.dumps(outcomes, ensure_ascii=False, indent=2), encoding="utf-8")
        if result.returncode:
            raise SystemExit(f"Experiment failed ({result.returncode}); inspect {log}")
    print(f"Completed {len(jobs)} jobs: {destination}")


if __name__ == "__main__":
    main()
