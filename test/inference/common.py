"""Shared implementation for HF, FlashInfer, and vLLM inference benchmarks.

Usage:

```bash
python -m test.inference.bench_flashinfer_generate \
  --model /data1/models/Qwen3-8B-forLoRA/Qwen3-8B \
  --max-new-tokens 128 \
  --target-prompt-tokens 4096 \
  --warmup-runs 1 \
  --measure-runs 3
```

The entry points intentionally require ``--model`` and load the real model.
They are benchmark harnesses, not default CI unit tests.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_bool(value: str) -> bool:
    """Parse true/false CLI values."""

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


@dataclass
class TimedGenerateResult:
    """Timing for one generation request split into prefill and decode."""

    total_s: float
    prefill_s: float
    decode_s: float
    generated_tokens: int

    @property
    def decode_tokens_per_s(self) -> float:
        return self.generated_tokens / self.decode_s if self.decode_s > 0 else 0.0

    @property
    def total_tokens_per_s(self) -> float:
        return self.generated_tokens / self.total_s if self.total_s > 0 else 0.0


@dataclass
class GenerateBenchStats:
    """Aggregated latency and throughput numbers for one generation path."""

    name: str
    load_time_s: float
    prompt_tokens: int
    latencies_s: list[float]
    prefill_latencies_s: list[float]
    decode_latencies_s: list[float]
    generated_tokens: list[int]
    tokens_per_s: list[float]
    throughput_timing: str

    @property
    def mean_latency_s(self) -> float:
        return statistics.mean(self.latencies_s)

    @property
    def mean_tokens_per_s(self) -> float:
        return statistics.mean(self.tokens_per_s)

    @property
    def mean_prefill_s(self) -> float:
        return statistics.mean(self.prefill_latencies_s)

    @property
    def mean_decode_s(self) -> float:
        return statistics.mean(self.decode_latencies_s)

    @property
    def mean_generated_tokens(self) -> float:
        return statistics.mean(self.generated_tokens)


def _sync_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA before/after timing so latency is meaningful."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: list[float], q: float) -> float:
    """Return a simple percentile for a short timing sample list."""

    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _set_seed(seed: int, *, seed_cuda: bool = True) -> None:
    """Reset Python and Torch RNGs before constructing each manager."""

    random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _release_cuda_memory() -> None:
    """Best-effort cleanup between the two real-model loads."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def _build_prompt(args: argparse.Namespace) -> str:
    """Build the legacy benchmark prompt, optionally repeated."""

    repeat = max(1, int(args.prompt_repeat))
    return "\n".join(args.prompt for _ in range(repeat))


def _build_long_context_prompt(
    args: argparse.Namespace,
    tokenizer,
) -> tuple[str, list[int]]:
    """Build and encode a synthetic long-context chat prompt."""

    from data.tokenizer_utils import encode_chat

    target_tokens = max(0, int(args.target_prompt_tokens))
    if target_tokens <= 0:
        prompt = _build_prompt(args)
        _, prompt_ids = encode_chat(
            tokenizer,
            [{"role": "user", "content": prompt}],
            enable_thinking=args.enable_thinking,
        )
        return prompt, prompt_ids

    blocks: list[str] = []
    unit = args.long_context_unit.strip()
    final_instruction = args.prompt.strip()
    prompt = _format_long_context_prompt(blocks, final_instruction)
    _, prompt_ids = encode_chat(
        tokenizer,
        [{"role": "user", "content": prompt}],
        enable_thinking=args.enable_thinking,
    )

    while len(prompt_ids) < target_tokens:
        block_index = len(blocks) + 1
        blocks.append(
            f"Context block {block_index:04d}: {unit} "
            f"Marker={block_index:04d}. The correct benchmark behavior is to keep "
            "all earlier context available while answering the final instruction."
        )
        prompt = _format_long_context_prompt(blocks, final_instruction)
        _, prompt_ids = encode_chat(
            tokenizer,
            [{"role": "user", "content": prompt}],
            enable_thinking=args.enable_thinking,
        )

    return prompt, prompt_ids


def _format_long_context_prompt(blocks: list[str], final_instruction: str) -> str:
    """Format synthetic context blocks plus the final generation task."""

    context = "\n\n".join(blocks)
    if context:
        return (
            "You are given a long synthetic context for a KV-cache benchmark. "
            "Use the context only as reference material.\n\n"
            f"{context}\n\n"
            "Final instruction:\n"
            f"{final_instruction}"
        )
    return final_instruction


def _make_hf_manager(args: argparse.Namespace, device: torch.device) -> ModelManager:
    """Create the project HF-generate path manager."""

    from model.model_manager import ModelConfig, ModelManager

    return ModelManager(
        ModelConfig(
            model_name=args.model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            cpu_offload=False,
            enable_multilora=False,
            logical_adapters=("general",),
            default_logical_adapter="general",
            kvcache_mode="hf",
        ),
        device=device,
    )


def _make_flashinfer_manager(args: argparse.Namespace, device: torch.device) -> ModelManager:
    """Create the FlashInfer paged-KV manager."""

    from model.model_manager import ModelConfig, ModelManager

    return ModelManager(
        ModelConfig(
            model_name=args.model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            cpu_offload=False,
            enable_multilora=True,
            enable_br_cache=False,
            enable_flashinfer_paged_kv=True,
            flashinfer_page_size=args.flashinfer_page_size,
            flashinfer_max_pages=args.flashinfer_max_pages,
            flashinfer_workspace_mb=args.flashinfer_workspace_mb,
            flashinfer_kv_append_mode=args.flashinfer_kv_append_mode,
            flashinfer_qk_norm=args.flashinfer_qk_norm,
            kvcache_mode="flashinfer",
            kvcache_decompose=False,
            logical_adapters=("general",),
            default_logical_adapter="general",
        ),
        device=device,
    )


def _make_vllm_llm(args: argparse.Namespace):
    """Create a direct vLLM engine for benchmark comparison."""

    if args.vllm_enable_lora and not args.vllm_lora_path:
        raise ValueError("--vllm-enable-lora requires --vllm-lora-path")
    if torch.cuda.is_initialized():
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM

    kwargs: dict[str, object] = {
        "model": args.model,
        "tokenizer": args.model,
        "trust_remote_code": args.vllm_trust_remote_code,
        "tensor_parallel_size": args.vllm_tensor_parallel_size,
        "dtype": args.vllm_dtype,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "enforce_eager": args.vllm_enforce_eager,
        "enable_prefix_caching": args.vllm_enable_prefix_caching,
        "seed": args.seed,
    }
    if args.vllm_enable_lora or args.vllm_lora_path:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = args.vllm_max_lora_rank or args.lora_r
    return LLM(**kwargs)


def _make_vllm_lora_request(args: argparse.Namespace):
    """Create vLLM LoRA request when the benchmark enables a LoRA adapter."""

    if not args.vllm_enable_lora and not args.vllm_lora_path:
        return None
    if not args.vllm_lora_path:
        raise ValueError("--vllm-enable-lora requires --vllm-lora-path")

    from vllm.lora.request import LoRARequest

    return LoRARequest(
        lora_name=args.vllm_lora_name,
        lora_int_id=args.vllm_lora_id,
        lora_path=args.vllm_lora_path,
    )


def _invalidate_adapter_cache(manager: ModelManager, adapter_name: str) -> None:
    """Clear managed KV cache for cold FlashInfer measurements."""

    cache_mgr = manager.kvcache_manager
    invalidate = getattr(cache_mgr, "invalidate_adapter", None)
    if callable(invalidate):
        invalidate(adapter_name)


def _run_generate_once(
    manager: ModelManager,
    prompt_ids: list[int],
    args: argparse.Namespace,
) -> TimedGenerateResult:
    """Run one deterministic generation and return elapsed time plus tokens."""

    from model.generation import generate_with_logprobs

    _sync_if_cuda(manager.device)
    started = time.perf_counter()
    output = generate_with_logprobs(
        manager,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        adapter_name="general",
        ignore_eos=args.ignore_eos,
        sampling_mode=args.sampling_mode,
        compute_logprobs=args.compute_logprobs,
        adapter_role=args.adapter_role,
    )
    _sync_if_cuda(manager.device)
    elapsed = time.perf_counter() - started
    output_timings = getattr(output, "timings", {}) or {}
    prefill_s = float(output_timings.get("prefill_s", 0.0))
    decode_s = float(output_timings.get("decode_s", 0.0))
    if decode_s <= 0.0:
        decode_s = max(0.0, elapsed - prefill_s)
    if prefill_s <= 0.0:
        prefill_s = max(0.0, elapsed - decode_s)
    return TimedGenerateResult(
        total_s=elapsed,
        prefill_s=prefill_s,
        decode_s=decode_s,
        generated_tokens=len(output.response_ids),
    )


def _run_hf_generate_once(
    manager: ModelManager,
    prompt_ids: list[int],
    args: argparse.Namespace,
) -> TimedGenerateResult:
    """Run HF model forward with KV cache and explicit prefill/decode timing."""

    from model.generation import _sample_next_tokens
    from multilora.runtime import AdapterRole

    if args.sampling_mode == "flashinfer_topk_topp":
        raise ValueError("--sampling-mode flashinfer_topk_topp cannot be used with hf paths")

    model = manager.model
    tokenizer = manager.tokenizer
    device = manager.device
    forward_parameters = inspect.signature(model.forward).parameters
    supports_cache_position = "cache_position" in forward_parameters
    input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    context_adapter = None if args.adapter_role == AdapterRole.BASE.value else "general"
    token_histories: list[list[int]] | None = None
    if not (
        args.sampling_mode == "greedy_fast"
        and not args.compute_logprobs
        and args.repetition_penalty == 1.0
    ):
        token_histories = [list(prompt_ids)]

    _sync_if_cuda(device)
    total_t0 = time.perf_counter()
    with manager.use_adapter(adapter_name=context_adapter, role=args.adapter_role):
        with manager.inference_session(enable_cache=True):
            with torch.no_grad():
                prefill_t0 = time.perf_counter()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                _sync_if_cuda(device)
                prefill_s = time.perf_counter() - prefill_t0
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                generated_ids: list[int] = []
                logprobs: list[float] = []
                decode_t0 = time.perf_counter()
                for step in range(args.max_new_tokens):
                    if (
                        args.sampling_mode == "greedy_fast"
                        and not args.compute_logprobs
                        and args.repetition_penalty == 1.0
                    ):
                        next_token_tensor = torch.argmax(logits, dim=-1)
                        next_logprobs: list[float] = []
                    else:
                        if token_histories is None:
                            raise RuntimeError("token histories must be initialized outside fast greedy mode")
                        next_tokens, next_logprobs = _sample_next_tokens(
                            logits,
                            token_histories=token_histories,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            repetition_penalty=args.repetition_penalty,
                            sampling_mode=args.sampling_mode,
                            compute_logprobs=args.compute_logprobs,
                        )
                        next_token_tensor = torch.tensor(next_tokens, device=device, dtype=torch.long)

                    token_id = int(next_token_tensor[0].item())
                    generated_ids.append(token_id)
                    if args.compute_logprobs and next_logprobs:
                        logprobs.append(next_logprobs[0])
                    if token_histories is not None:
                        token_histories[0].append(token_id)
                    if not args.ignore_eos and token_id == tokenizer.eos_token_id:
                        break
                    if step + 1 >= args.max_new_tokens:
                        break

                    token_input = next_token_tensor.view(1, 1)
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
                        dim=1,
                    )
                    model_inputs = {
                        "input_ids": token_input,
                        "past_key_values": past_key_values,
                        "attention_mask": attention_mask,
                        "use_cache": True,
                    }
                    if supports_cache_position:
                        model_inputs["cache_position"] = torch.tensor(
                            [attention_mask.size(1) - 1],
                            device=device,
                            dtype=torch.long,
                        )
                    outputs = model(**model_inputs, return_dict=True)
                    logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values
                _sync_if_cuda(device)
                decode_s = time.perf_counter() - decode_t0
    total_s = time.perf_counter() - total_t0
    return TimedGenerateResult(
        total_s=total_s,
        prefill_s=prefill_s,
        decode_s=decode_s,
        generated_tokens=len(generated_ids),
    )


def _vllm_sampling_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Translate benchmark sampling args into vLLM SamplingParams kwargs."""

    greedy_fast = args.sampling_mode == "greedy_fast"
    return {
        "temperature": 0.0 if greedy_fast else args.temperature,
        "top_p": 1.0 if greedy_fast else args.top_p,
        "top_k": 0 if greedy_fast else args.top_k,
        "repetition_penalty": 1.0 if greedy_fast else args.repetition_penalty,
        "ignore_eos": args.ignore_eos,
        "max_tokens": args.max_new_tokens,
        "logprobs": 1 if args.compute_logprobs else None,
        "seed": args.seed,
        "detokenize": False,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
    }


def _run_vllm_once(
    llm,
    prompt_ids: list[int],
    args: argparse.Namespace,
    device: torch.device,
    *,
    lora_request=None,
) -> TimedGenerateResult:
    """Run one vLLM generation and return elapsed time plus generated tokens."""

    if getattr(args, "throughput_timing", "decode") == "total":
        elapsed, generated, outputs = _run_vllm_generate_call(
            llm,
            prompt_ids,
            args,
            device,
            lora_request=lora_request,
            max_tokens=args.max_new_tokens,
        )
        metrics_prefill_s, metrics_decode_s = _extract_vllm_split_timing(outputs, elapsed)
        return TimedGenerateResult(
            total_s=elapsed,
            prefill_s=metrics_prefill_s,
            decode_s=metrics_decode_s,
            generated_tokens=generated,
        )

    probe_elapsed, _, _ = _run_vllm_generate_call(
        llm,
        prompt_ids,
        args,
        device,
        lora_request=lora_request,
        max_tokens=1,
    )
    elapsed, generated, outputs = _run_vllm_generate_call(
        llm,
        prompt_ids,
        args,
        device,
        lora_request=lora_request,
        max_tokens=args.max_new_tokens,
    )
    metrics_prefill_s, metrics_decode_s = _extract_vllm_split_timing(outputs, elapsed)
    if metrics_prefill_s > 0.0 and metrics_decode_s > 0.0:
        prefill_s = metrics_prefill_s
        decode_s = metrics_decode_s
    else:
        prefill_s = probe_elapsed
        decode_s = max(0.0, elapsed - prefill_s)
    return TimedGenerateResult(
        total_s=elapsed,
        prefill_s=prefill_s,
        decode_s=decode_s,
        generated_tokens=generated,
    )


def _run_vllm_generate_call(
    llm,
    prompt_ids: list[int],
    args: argparse.Namespace,
    device: torch.device,
    *,
    lora_request=None,
    max_tokens: int,
):
    """Run one raw vLLM generate call."""

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sampling_kwargs = _vllm_sampling_kwargs(args)
    sampling_kwargs["max_tokens"] = max_tokens
    sampling_params = SamplingParams(**sampling_kwargs)
    prompt = TokensPrompt(prompt_token_ids=prompt_ids)
    _sync_if_cuda(device)
    started = time.perf_counter()
    outputs = llm.generate(
        [prompt],
        sampling_params=sampling_params,
        use_tqdm=False,
        lora_request=lora_request,
    )
    _sync_if_cuda(device)
    elapsed = time.perf_counter() - started
    generated = 0
    if outputs and outputs[0].outputs:
        generated = len(outputs[0].outputs[0].token_ids)
    return elapsed, generated, outputs


def _extract_vllm_split_timing(outputs, elapsed: float) -> tuple[float, float]:
    """Extract TTFT/decode split from vLLM RequestOutput metrics when available."""

    total_s = max(0.0, elapsed)
    if not outputs:
        return 0.0, total_s
    metrics = getattr(outputs[0], "metrics", None)
    if metrics is None:
        return 0.0, total_s
    first_token_time = getattr(metrics, "first_token_time", None)
    finished_time = getattr(metrics, "finished_time", None)
    arrival_time = getattr(metrics, "arrival_time", None)
    if (
        isinstance(first_token_time, (int, float))
        and isinstance(arrival_time, (int, float))
        and isinstance(finished_time, (int, float))
        and arrival_time <= first_token_time <= finished_time
    ):
        return (
            max(0.0, float(first_token_time) - float(arrival_time)),
            max(0.0, float(finished_time) - float(first_token_time)),
        )
    return 0.0, total_s


def _result_tokens_per_s(result: TimedGenerateResult, args: argparse.Namespace) -> float:
    """Return throughput under the selected timing denominator."""

    if getattr(args, "throughput_timing", "decode") == "total":
        return result.total_tokens_per_s
    return result.decode_tokens_per_s


def _print_run_result(
    *,
    name: str,
    phase: str,
    run_index: int,
    result: TimedGenerateResult,
    generated_tokens: int,
    args: argparse.Namespace,
) -> None:
    """Print one generation run result immediately for long benchmark jobs."""

    tokens_per_s = _result_tokens_per_s(result, args)
    print(
        f"[{name}] {phase}#{run_index} "
        f"total_s={result.total_s:.4f} "
        f"prefill_s={result.prefill_s:.4f} "
        f"decode_s={result.decode_s:.4f} "
        f"generated_tokens={generated_tokens} "
        f"tokens_per_s={tokens_per_s:.4f} "
        f"throughput_timing={args.throughput_timing}",
        flush=True,
    )


def _benchmark_manager(
    *,
    name: str,
    load_time_s: float,
    manager: ModelManager,
    args: argparse.Namespace,
) -> GenerateBenchStats:
    """Benchmark one manager with the configured prompt and run counts."""

    prompt, prompt_ids = _build_long_context_prompt(args, manager.tokenizer)

    is_flashinfer = name == "flashinfer_paged_kv"
    cold_flashinfer = is_flashinfer and args.flashinfer_cache_policy == "cold"
    hot_flashinfer = is_flashinfer and args.flashinfer_cache_policy == "hot"
    runner = _run_hf_generate_once if name == "hf_generate" else _run_generate_once

    print(f"\n[{name}] prompt_tokens={len(prompt_ids)} prompt_chars={len(prompt)}", flush=True)
    for run_index in range(1, max(0, args.warmup_runs) + 1):
        if cold_flashinfer:
            _invalidate_adapter_cache(manager, "general")
        result = runner(manager, prompt_ids, args)
        _print_run_result(
            name=name,
            phase="warmup",
            run_index=run_index,
            result=result,
            generated_tokens=result.generated_tokens,
            args=args,
        )

    if hot_flashinfer and args.warmup_runs <= 0:
        result = _run_generate_once(manager, prompt_ids, args)
        _print_run_result(
            name=name,
            phase="cache_prime",
            run_index=1,
            result=result,
            generated_tokens=result.generated_tokens,
            args=args,
        )

    latencies: list[float] = []
    prefill_latencies: list[float] = []
    decode_latencies: list[float] = []
    generated_tokens: list[int] = []
    tokens_per_s: list[float] = []
    for run_index in range(1, max(1, args.measure_runs) + 1):
        if cold_flashinfer:
            _invalidate_adapter_cache(manager, "general")
        result = runner(manager, prompt_ids, args)
        latencies.append(result.total_s)
        prefill_latencies.append(result.prefill_s)
        decode_latencies.append(result.decode_s)
        generated_tokens.append(result.generated_tokens)
        tokens_per_s.append(_result_tokens_per_s(result, args))
        _print_run_result(
            name=name,
            phase="measure",
            run_index=run_index,
            result=result,
            generated_tokens=result.generated_tokens,
            args=args,
        )

    return GenerateBenchStats(
        name=name,
        load_time_s=load_time_s,
        prompt_tokens=len(prompt_ids),
        latencies_s=latencies,
        prefill_latencies_s=prefill_latencies,
        decode_latencies_s=decode_latencies,
        generated_tokens=generated_tokens,
        tokens_per_s=tokens_per_s,
        throughput_timing=args.throughput_timing,
    )


def _benchmark_vllm(
    *,
    load_time_s: float,
    llm,
    args: argparse.Namespace,
    device: torch.device,
) -> GenerateBenchStats:
    """Benchmark direct vLLM generation with the configured prompt."""

    tokenizer = llm.get_tokenizer()
    prompt, prompt_ids = _build_long_context_prompt(args, tokenizer)
    name = "vllm_generate"
    lora_request = _make_vllm_lora_request(args)

    print(f"\n[{name}] prompt_tokens={len(prompt_ids)} prompt_chars={len(prompt)}", flush=True)
    for run_index in range(1, max(0, args.warmup_runs) + 1):
        result = _run_vllm_once(llm, prompt_ids, args, device, lora_request=lora_request)
        _print_run_result(
            name=name,
            phase="warmup",
            run_index=run_index,
            result=result,
            generated_tokens=result.generated_tokens,
            args=args,
        )

    latencies: list[float] = []
    prefill_latencies: list[float] = []
    decode_latencies: list[float] = []
    generated_tokens: list[int] = []
    tokens_per_s: list[float] = []
    for run_index in range(1, max(1, args.measure_runs) + 1):
        result = _run_vllm_once(llm, prompt_ids, args, device, lora_request=lora_request)
        latencies.append(result.total_s)
        prefill_latencies.append(result.prefill_s)
        decode_latencies.append(result.decode_s)
        generated_tokens.append(result.generated_tokens)
        tokens_per_s.append(_result_tokens_per_s(result, args))
        _print_run_result(
            name=name,
            phase="measure",
            run_index=run_index,
            result=result,
            generated_tokens=result.generated_tokens,
            args=args,
        )

    return GenerateBenchStats(
        name=name,
        load_time_s=load_time_s,
        prompt_tokens=len(prompt_ids),
        latencies_s=latencies,
        prefill_latencies_s=prefill_latencies,
        decode_latencies_s=decode_latencies,
        generated_tokens=generated_tokens,
        tokens_per_s=tokens_per_s,
        throughput_timing=args.throughput_timing,
    )


def _print_path_stats(stats: GenerateBenchStats) -> None:
    """Print one path's benchmark summary."""

    print(f"\n[{stats.name}]")
    print(f"load_time_s={stats.load_time_s:.4f}")
    print(f"prompt_tokens={stats.prompt_tokens}")
    print(f"mean_total_s={stats.mean_latency_s:.4f}")
    print(f"p50_total_s={_percentile(stats.latencies_s, 0.50):.4f}")
    print(f"p95_total_s={_percentile(stats.latencies_s, 0.95):.4f}")
    print(f"mean_prefill_s={stats.mean_prefill_s:.4f}")
    print(f"p50_prefill_s={_percentile(stats.prefill_latencies_s, 0.50):.4f}")
    print(f"p95_prefill_s={_percentile(stats.prefill_latencies_s, 0.95):.4f}")
    print(f"mean_decode_s={stats.mean_decode_s:.4f}")
    print(f"p50_decode_s={_percentile(stats.decode_latencies_s, 0.50):.4f}")
    print(f"p95_decode_s={_percentile(stats.decode_latencies_s, 0.95):.4f}")
    print(f"mean_generated_tokens={stats.mean_generated_tokens:.2f}")
    print(f"mean_tokens_per_s={stats.mean_tokens_per_s:.4f}")
    print(f"throughput_timing={stats.throughput_timing}")


def _print_pair_comparison(label: str, numerator: GenerateBenchStats, denominator: GenerateBenchStats) -> None:
    """Print latency and throughput ratios for two benchmark paths."""

    latency_ratio = numerator.mean_latency_s / denominator.mean_latency_s
    tps_ratio = numerator.mean_tokens_per_s / denominator.mean_tokens_per_s
    print(f"{label}_latency_ratio={latency_ratio:.4f}")
    print(f"{label}_tokens_per_s_ratio={tps_ratio:.4f}")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the real-model generation benchmark."""

    parser = argparse.ArgumentParser(description="Benchmark HF, FlashInfer paged-KV, and vLLM generate.")
    parser.add_argument("--model", required=True, help="HuggingFace model name or local model path.")
    parser.add_argument("--device", default=None, help="Torch device, for example cuda, cuda:0, or cpu.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup generations per path.")
    parser.add_argument("--measure-runs", type=int, default=3, help="Measured generations per path.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generated token upper bound.")
    parser.add_argument(
        "--prompt",
        default=(
            "Summarize the recurring theme across the context blocks in one concise paragraph, "
            "then state the highest marker number you saw."
        ),
        help="Final instruction appended after the synthetic long context.",
    )
    parser.add_argument(
        "--target-prompt-tokens",
        type=int,
        default=4096,
        help="Approximate minimum prompt token count for long-context benchmarking. Use 0 for legacy prompt-repeat mode.",
    )
    parser.add_argument(
        "--long-context-unit",
        default=(
            "This benchmark paragraph discusses local online reinforcement learning, "
            "paged KV cache reuse, adapter routing, scheduler timing, and long-context "
            "generation. It is intentionally repetitive but includes a unique marker."
        ),
        help="Synthetic context block repeated until target-prompt-tokens is reached.",
    )
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=1,
        help="Legacy repeat count used only when --target-prompt-tokens is 0.",
    )
    parser.add_argument("--enable-thinking", action="store_true", help="Enable tokenizer chat-template thinking mode.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling value.")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling value.")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="Repetition penalty.")
    parser.add_argument("--ignore-eos", action="store_true", help="Force generation to run until max-new-tokens or max-time.")
    parser.add_argument(
        "--sampling-mode",
        choices=("project_default", "greedy_fast", "flashinfer_topk_topp"),
        default="project_default",
        help=(
            "project_default preserves the original project sampling path; "
            "greedy_fast uses argmax without repetition penalty; "
            "flashinfer_topk_topp uses FlashInfer sampling in custom paths only."
        ),
    )
    parser.add_argument(
        "--compute-logprobs",
        type=_parse_bool,
        default=True,
        metavar="true|false",
        help="Whether to compute selected-token logprobs during benchmark generation.",
    )
    parser.add_argument(
        "--throughput-timing",
        choices=("decode", "total"),
        default="decode",
        help="decode uses generated_tokens/decode_s; total uses generated_tokens/total_s.",
    )
    parser.add_argument(
        "--benchmark-path",
        choices=("hf", "flashinfer", "vllm", "both", "all"),
        default="both",
        help="Select generation path. both means hf+flashinfer; all means hf+flashinfer+vllm.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed reset before constructing each manager.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")
    parser.add_argument(
        "--adapter-role",
        choices=("serve", "base"),
        default="serve",
        help="Project runtime adapter role. base skips project LoRA deltas for HF/FlashInfer paths.",
    )
    parser.add_argument("--flashinfer-page-size", type=int, default=32, help="FlashInfer KV page size.")
    parser.add_argument("--flashinfer-max-pages", type=int, default=0, help="0 means grow dense page cache on demand.")
    parser.add_argument("--flashinfer-workspace-mb", type=int, default=128, help="FlashInfer workspace size in MiB.")
    parser.add_argument(
        "--flashinfer-kv-append-mode",
        choices=("manual", "flashinfer"),
        default="manual",
        help="How FlashInfer paged KV writes are performed.",
    )
    parser.add_argument(
        "--flashinfer-qk-norm",
        action="store_true",
        help="Use flashinfer.norm.rmsnorm for q_norm/k_norm when compatible.",
    )
    parser.add_argument(
        "--flashinfer-cache-policy",
        choices=("cold", "hot"),
        default="cold",
        help="cold clears managed KV cache before each measured FlashInfer run; hot measures prefix reuse.",
    )
    parser.add_argument("--vllm-dtype", default="auto", help="vLLM dtype argument, for example auto, bfloat16, or float16.")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization argument.",
    )
    parser.add_argument("--vllm-enforce-eager", action="store_true", help="Run vLLM with enforce_eager=True.")
    parser.add_argument(
        "--vllm-trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to vLLM.",
    )
    parser.add_argument(
        "--vllm-enable-prefix-caching",
        action="store_true",
        help="Enable vLLM prefix caching. Leave off for colder repeated-prompt comparisons.",
    )
    parser.add_argument(
        "--vllm-enable-lora",
        action="store_true",
        help="Enable vLLM LoRA serving and require --vllm-lora-path for generation.",
    )
    parser.add_argument("--vllm-lora-path", default="", help="Path to a vLLM-compatible LoRA adapter.")
    parser.add_argument("--vllm-lora-name", default="general", help="vLLM LoRA adapter name.")
    parser.add_argument("--vllm-lora-id", type=int, default=1, help="Positive integer id for the vLLM LoRA adapter.")
    parser.add_argument(
        "--vllm-max-lora-rank",
        type=int,
        default=0,
        help="vLLM max_lora_rank. 0 means reuse --lora-r.",
    )
    return parser


def run_benchmark(
    args: argparse.Namespace,
) -> tuple[GenerateBenchStats | None, GenerateBenchStats | None, GenerateBenchStats | None]:
    """Run the real-model benchmark from already parsed arguments."""

    from core.utils import get_device

    device = torch.device(args.device) if args.device else get_device()

    print("[flashinfer-vs-hf-generate]")
    print(f"model={args.model}")
    print(f"device={device}")
    print(f"warmup_runs={args.warmup_runs}")
    print(f"measure_runs={args.measure_runs}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print(f"target_prompt_tokens={args.target_prompt_tokens}")
    print(f"ignore_eos={args.ignore_eos}")
    print(f"sampling_mode={args.sampling_mode}")
    print(f"compute_logprobs={args.compute_logprobs}")
    print(f"throughput_timing={args.throughput_timing}")
    print(f"benchmark_path={args.benchmark_path}")
    print(f"adapter_role={args.adapter_role}")
    print(f"flashinfer_cache_policy={args.flashinfer_cache_policy}")
    print(f"flashinfer_kv_append_mode={args.flashinfer_kv_append_mode}")
    print(f"flashinfer_qk_norm={args.flashinfer_qk_norm}")
    print(f"vllm_enable_prefix_caching={args.vllm_enable_prefix_caching}")
    print(f"vllm_enforce_eager={args.vllm_enforce_eager}")
    print(f"vllm_enable_lora={args.vllm_enable_lora or bool(args.vllm_lora_path)}")
    print(f"vllm_lora_path={args.vllm_lora_path}")
    print(f"vllm_lora_name={args.vllm_lora_name}")
    print(f"vllm_lora_id={args.vllm_lora_id}")
    print(f"vllm_max_lora_rank={args.vllm_max_lora_rank or args.lora_r}")

    hf_stats: GenerateBenchStats | None = None
    flashinfer_stats: GenerateBenchStats | None = None
    vllm_stats: GenerateBenchStats | None = None

    if args.benchmark_path in {"hf", "both", "all"}:
        if args.sampling_mode == "flashinfer_topk_topp":
            raise ValueError("--sampling-mode flashinfer_topk_topp cannot be used with hf paths")
        _set_seed(args.seed)
        started = time.perf_counter()
        hf_manager = _make_hf_manager(args, device)
        hf_load_time = time.perf_counter() - started
        try:
            hf_stats = _benchmark_manager(
                name="hf_generate",
                load_time_s=hf_load_time,
                manager=hf_manager,
                args=args,
            )
        finally:
            hf_manager = None
            _release_cuda_memory()

    if args.benchmark_path in {"flashinfer", "both", "all"}:
        _set_seed(args.seed)
        started = time.perf_counter()
        flashinfer_manager = _make_flashinfer_manager(args, device)
        flashinfer_load_time = time.perf_counter() - started
        try:
            flashinfer_stats = _benchmark_manager(
                name="flashinfer_paged_kv",
                load_time_s=flashinfer_load_time,
                manager=flashinfer_manager,
                args=args,
            )
        finally:
            flashinfer_manager = None
            _release_cuda_memory()

    if args.benchmark_path in {"vllm", "all"}:
        _set_seed(args.seed, seed_cuda=False)
        started = time.perf_counter()
        vllm_llm = _make_vllm_llm(args)
        vllm_load_time = time.perf_counter() - started
        try:
            vllm_stats = _benchmark_vllm(
                load_time_s=vllm_load_time,
                llm=vllm_llm,
                args=args,
                device=device,
            )
        finally:
            vllm_llm = None
            _release_cuda_memory()

    if hf_stats is not None:
        _print_path_stats(hf_stats)
    if flashinfer_stats is not None:
        _print_path_stats(flashinfer_stats)
    if vllm_stats is not None:
        _print_path_stats(vllm_stats)

    if sum(stats is not None for stats in (hf_stats, flashinfer_stats, vllm_stats)) >= 2:
        print("\n[comparison]")
        if hf_stats is not None and flashinfer_stats is not None:
            _print_pair_comparison("flashinfer_over_hf", flashinfer_stats, hf_stats)
        if hf_stats is not None and vllm_stats is not None:
            _print_pair_comparison("vllm_over_hf", vllm_stats, hf_stats)
        if flashinfer_stats is not None and vllm_stats is not None:
            _print_pair_comparison("flashinfer_over_vllm", flashinfer_stats, vllm_stats)

    return hf_stats, flashinfer_stats, vllm_stats


def main() -> None:
    """Run the real-model benchmark and print a compact comparison."""

    run_benchmark(build_parser().parse_args())


if __name__ == "__main__":
    main()
