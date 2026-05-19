"""Benchmark only direct vLLM generate.

Example:

```bash
python -m test.inference.bench_vllm_generate \
  --model /data1/models/Qwen3-8B-forLoRA/Qwen3-8B \
  --target-prompt-tokens 4096 \
  --max-new-tokens 512 \
  --sampling-mode greedy_fast \
  --compute-logprobs false \
  --ignore-eos \
  --vllm-enable-lora \
  --vllm-lora-path .artifacts/vllm_zero_lora_qwen3_r16
```
"""

from __future__ import annotations

from test.inference.common import build_parser, run_benchmark


def main() -> None:
    """Run the vLLM-only benchmark."""

    parser = build_parser()
    parser.description = "Benchmark only direct vLLM generate."
    args = parser.parse_args()
    args.benchmark_path = "vllm"
    run_benchmark(args)


if __name__ == "__main__":
    main()
