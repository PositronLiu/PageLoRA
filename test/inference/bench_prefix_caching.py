"""Benchmark prefix-cache reuse for repeated inference sequences.

The ``single`` mode repeats one full prompt, which exercises a single radix
path without branching. The ``branch`` mode uses one shared prefix with
different suffixes, which exercises shared-prefix reuse with radix branching.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch

from test.inference.common import (
    _invalidate_adapter_cache,
    _make_flashinfer_manager,
    _make_vllm_llm,
    _make_vllm_lora_request,
    _parse_bool,
    _percentile,
    _release_cuda_memory,
    _run_generate_once,
    _run_vllm_once,
    _set_seed,
    build_parser,
)


@dataclass
class PrefixBenchStats:
    """Aggregated request timings for one prefix-cache benchmark sequence."""

    name: str
    load_time_s: float
    requests_per_sequence: int
    prompt_tokens: list[int]
    latencies_s: list[float]
    generated_tokens: list[int]

    @property
    def mean_latency_s(self) -> float:
        return statistics.mean(self.latencies_s)

    @property
    def mean_tokens_per_s(self) -> float:
        per_request = [
            tokens / latency for tokens, latency in zip(self.generated_tokens, self.latencies_s) if latency > 0
        ]
        return statistics.mean(per_request) if per_request else 0.0


def _encode_prompt_ids(tokenizer, text: str, *, enable_thinking: bool) -> list[int]:
    """Encode one chat prompt and return token ids."""

    from data.tokenizer_utils import encode_chat

    _, prompt_ids = encode_chat(
        tokenizer,
        [{"role": "user", "content": text}],
        enable_thinking=enable_thinking,
    )
    return prompt_ids


def _format_shared_prefix(blocks: list[str]) -> str:
    """Build the shared context portion for prefix-cache benchmarks."""

    context = "\n\n".join(blocks)
    if not context:
        return "Shared prefix cache benchmark context."
    return (
        "You are given a shared synthetic context for a prefix-cache benchmark. "
        "Keep all marker numbers available for the final instruction.\n\n"
        f"{context}"
    )


def _build_shared_prefix(args: argparse.Namespace, tokenizer) -> str:
    """Grow a shared prompt prefix until it reaches the requested token count."""

    target_tokens = int(args.shared_prefix_tokens)
    if target_tokens <= 0:
        target_tokens = int(args.target_prompt_tokens)
    if target_tokens <= 0:
        return args.prompt.strip()

    blocks: list[str] = []
    unit = args.long_context_unit.strip()
    shared = _format_shared_prefix(blocks)
    prompt_ids = _encode_prompt_ids(tokenizer, shared, enable_thinking=args.enable_thinking)
    while len(prompt_ids) < target_tokens:
        block_index = len(blocks) + 1
        blocks.append(
            f"Shared block {block_index:04d}: {unit} "
            f"SharedMarker={block_index:04d}. This block is identical across all branch requests."
        )
        shared = _format_shared_prefix(blocks)
        prompt_ids = _encode_prompt_ids(tokenizer, shared, enable_thinking=args.enable_thinking)
    return shared


def _build_suffix(args: argparse.Namespace, tokenizer, *, request_index: int, mode: str) -> str:
    """Build the request-specific final suffix."""

    if mode == "single":
        return (
            "\n\nFinal instruction:\n"
            f"{args.prompt.strip()}\n"
            "Use the shared context and answer deterministically."
        )

    lines = [
        "\n\nBranch-specific instruction:",
        f"Branch request {request_index:04d} asks for the shared highest marker and this branch id.",
        f"UniqueBranchMarker={request_index:04d}.",
        args.prompt.strip(),
    ]
    target_tokens = max(0, int(args.branch_suffix_tokens))
    while target_tokens > 0 and len(_encode_prompt_ids(tokenizer, "\n".join(lines), enable_thinking=False)) < target_tokens:
        repeat_index = len(lines)
        lines.append(
            f"Branch suffix filler {request_index:04d}-{repeat_index:04d}: "
            "this text differs after the common prefix."
        )
    return "\n".join(lines)


def _build_prefix_prompt_ids(
    args: argparse.Namespace,
    tokenizer,
    *,
    mode: str,
) -> tuple[list[str], list[list[int]]]:
    """Build tokenized prompts for one prefix-cache benchmark mode."""

    request_count = max(1, int(args.prefix_requests))
    shared = _build_shared_prefix(args, tokenizer)
    prompts: list[str] = []
    prompt_ids: list[list[int]] = []
    for index in range(1, request_count + 1):
        suffix_index = 1 if mode == "single" else index
        text = shared + _build_suffix(args, tokenizer, request_index=suffix_index, mode=mode)
        prompts.append(text)
        prompt_ids.append(_encode_prompt_ids(tokenizer, text, enable_thinking=args.enable_thinking))
    return prompts, prompt_ids


def _print_prefix_request(
    *,
    name: str,
    phase: str,
    sequence_index: int,
    request_index: int,
    prompt_tokens: int,
    elapsed: float,
    generated_tokens: int,
) -> None:
    """Print one prefix-cache request result."""

    tokens_per_s = generated_tokens / elapsed if elapsed > 0 else 0.0
    print(
        f"[{name}] {phase}#{sequence_index} request#{request_index} "
        f"prompt_tokens={prompt_tokens} "
        f"latency_s={elapsed:.4f} "
        f"generated_tokens={generated_tokens} "
        f"tokens_per_s={tokens_per_s:.4f}",
        flush=True,
    )


def _print_prefix_summary(stats: PrefixBenchStats) -> None:
    """Print aggregate prefix-cache timing numbers."""

    first_request_latencies = [
        latency for index, latency in enumerate(stats.latencies_s) if index % stats.requests_per_sequence == 0
    ]
    reuse_request_latencies = [
        latency for index, latency in enumerate(stats.latencies_s) if index % stats.requests_per_sequence != 0
    ]
    print(f"\n[{stats.name}]")
    print(f"load_time_s={stats.load_time_s:.4f}")
    print(f"requests_per_sequence={stats.requests_per_sequence}")
    print(f"prompt_tokens_min={min(stats.prompt_tokens)}")
    print(f"prompt_tokens_max={max(stats.prompt_tokens)}")
    print(f"mean_request_latency_s={stats.mean_latency_s:.4f}")
    print(f"p50_request_latency_s={_percentile(stats.latencies_s, 0.50):.4f}")
    print(f"p95_request_latency_s={_percentile(stats.latencies_s, 0.95):.4f}")
    print(f"mean_first_request_latency_s={statistics.mean(first_request_latencies):.4f}")
    if reuse_request_latencies:
        print(f"mean_reuse_request_latency_s={statistics.mean(reuse_request_latencies):.4f}")
    print(f"mean_generated_tokens={statistics.mean(stats.generated_tokens):.2f}")
    print(f"mean_tokens_per_s={stats.mean_tokens_per_s:.4f}")


def _run_flashinfer_prefix_mode(args: argparse.Namespace, device: torch.device, *, mode: str) -> PrefixBenchStats:
    """Benchmark project FlashInfer prefix-cache reuse for one mode."""

    _set_seed(args.seed)
    started = time.perf_counter()
    manager = _make_flashinfer_manager(args, device)
    load_time_s = time.perf_counter() - started
    name = f"flashinfer_prefix_{mode}"
    try:
        _, prompt_ids = _build_prefix_prompt_ids(args, manager.tokenizer, mode=mode)
        print(
            f"\n[{name}] prompt_tokens={[len(ids) for ids in prompt_ids]} "
            f"reset_cache_between_runs={args.prefix_reset_cache_between_runs}",
            flush=True,
        )
        for sequence_index in range(1, max(0, args.warmup_runs) + 1):
            if args.prefix_reset_cache_between_runs:
                _invalidate_adapter_cache(manager, "general")
            for request_index, ids in enumerate(prompt_ids, start=1):
                elapsed, token_count = _run_generate_once(manager, ids, args)
                _print_prefix_request(
                    name=name,
                    phase="warmup",
                    sequence_index=sequence_index,
                    request_index=request_index,
                    prompt_tokens=len(ids),
                    elapsed=elapsed,
                    generated_tokens=token_count,
                )

        latencies: list[float] = []
        generated_tokens: list[int] = []
        prompt_token_counts: list[int] = []
        for sequence_index in range(1, max(1, args.measure_runs) + 1):
            if args.prefix_reset_cache_between_runs:
                _invalidate_adapter_cache(manager, "general")
            for request_index, ids in enumerate(prompt_ids, start=1):
                elapsed, token_count = _run_generate_once(manager, ids, args)
                latencies.append(elapsed)
                generated_tokens.append(token_count)
                prompt_token_counts.append(len(ids))
                _print_prefix_request(
                    name=name,
                    phase="measure",
                    sequence_index=sequence_index,
                    request_index=request_index,
                    prompt_tokens=len(ids),
                    elapsed=elapsed,
                    generated_tokens=token_count,
                )
        return PrefixBenchStats(
            name=name,
            load_time_s=load_time_s,
            requests_per_sequence=len(prompt_ids),
            prompt_tokens=prompt_token_counts,
            latencies_s=latencies,
            generated_tokens=generated_tokens,
        )
    finally:
        manager = None
        _release_cuda_memory()


def _run_vllm_prefix_mode(args: argparse.Namespace, device: torch.device, *, mode: str) -> PrefixBenchStats:
    """Benchmark vLLM prefix-cache reuse for one mode."""

    _set_seed(args.seed, seed_cuda=False)
    args.vllm_enable_prefix_caching = True
    started = time.perf_counter()
    llm = _make_vllm_llm(args)
    load_time_s = time.perf_counter() - started
    name = f"vllm_prefix_{mode}"
    try:
        tokenizer = llm.get_tokenizer()
        _, prompt_ids = _build_prefix_prompt_ids(args, tokenizer, mode=mode)
        lora_request = _make_vllm_lora_request(args)
        print(f"\n[{name}] prompt_tokens={[len(ids) for ids in prompt_ids]}", flush=True)
        for sequence_index in range(1, max(0, args.warmup_runs) + 1):
            for request_index, ids in enumerate(prompt_ids, start=1):
                elapsed, token_count = _run_vllm_once(llm, ids, args, device, lora_request=lora_request)
                _print_prefix_request(
                    name=name,
                    phase="warmup",
                    sequence_index=sequence_index,
                    request_index=request_index,
                    prompt_tokens=len(ids),
                    elapsed=elapsed,
                    generated_tokens=token_count,
                )

        latencies: list[float] = []
        generated_tokens: list[int] = []
        prompt_token_counts: list[int] = []
        for sequence_index in range(1, max(1, args.measure_runs) + 1):
            for request_index, ids in enumerate(prompt_ids, start=1):
                elapsed, token_count = _run_vllm_once(llm, ids, args, device, lora_request=lora_request)
                latencies.append(elapsed)
                generated_tokens.append(token_count)
                prompt_token_counts.append(len(ids))
                _print_prefix_request(
                    name=name,
                    phase="measure",
                    sequence_index=sequence_index,
                    request_index=request_index,
                    prompt_tokens=len(ids),
                    elapsed=elapsed,
                    generated_tokens=token_count,
                )
        return PrefixBenchStats(
            name=name,
            load_time_s=load_time_s,
            requests_per_sequence=len(prompt_ids),
            prompt_tokens=prompt_token_counts,
            latencies_s=latencies,
            generated_tokens=generated_tokens,
        )
    finally:
        llm = None
        _release_cuda_memory()


def build_prefix_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for prefix-cache benchmarks."""

    parser = build_parser()
    parser.description = "Benchmark prefix-cache reuse for FlashInfer radix cache or vLLM prefix caching."
    parser.set_defaults(benchmark_path="flashinfer")
    parser.add_argument(
        "--prefix-backend",
        choices=("flashinfer", "vllm", "both"),
        default="flashinfer",
        help="Which prefix-cache backend to benchmark.",
    )
    parser.add_argument(
        "--prefix-bench-mode",
        choices=("single", "branch", "both"),
        default="both",
        help="single repeats one full prompt; branch reuses a shared prefix with distinct suffixes.",
    )
    parser.add_argument(
        "--prefix-requests",
        type=int,
        default=4,
        help="Number of consecutive requests in each measured prefix-cache sequence.",
    )
    parser.add_argument(
        "--shared-prefix-tokens",
        type=int,
        default=0,
        help="Approximate shared-prefix token target. 0 reuses --target-prompt-tokens.",
    )
    parser.add_argument(
        "--branch-suffix-tokens",
        type=int,
        default=0,
        help="Approximate minimum branch suffix token count. 0 uses a short unique suffix.",
    )
    parser.add_argument(
        "--prefix-reset-cache-between-runs",
        type=_parse_bool,
        default=True,
        metavar="true|false",
        help="For FlashInfer, clear radix/paged cache before each warmup or measured sequence.",
    )
    return parser


def main() -> None:
    """Run the selected prefix-cache benchmark modes."""

    from core.utils import get_device

    args = build_prefix_parser().parse_args()
    device = torch.device(args.device) if args.device else get_device()
    modes = ("single", "branch") if args.prefix_bench_mode == "both" else (args.prefix_bench_mode,)
    backends = ("flashinfer", "vllm") if args.prefix_backend == "both" else (args.prefix_backend,)

    print("[prefix-cache-generate]")
    print(f"model={args.model}")
    print(f"device={device}")
    print(f"backend={args.prefix_backend}")
    print(f"mode={args.prefix_bench_mode}")
    print(f"requests={args.prefix_requests}")
    print(f"shared_prefix_tokens={args.shared_prefix_tokens or args.target_prompt_tokens}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print(f"sampling_mode={args.sampling_mode}")
    print(f"compute_logprobs={args.compute_logprobs}")
    print(f"ignore_eos={args.ignore_eos}")

    stats: list[PrefixBenchStats] = []
    for backend in backends:
        for mode in modes:
            if backend == "flashinfer":
                stats.append(_run_flashinfer_prefix_mode(args, device, mode=mode))
            else:
                stats.append(_run_vllm_prefix_mode(args, device, mode=mode))

    for item in stats:
        _print_prefix_summary(item)


if __name__ == "__main__":
    main()
