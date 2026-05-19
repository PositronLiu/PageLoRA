"""Benchmark only the project FlashInfer paged-KV generate path.

Example:

```bash
python -m test.inference.bench_flashinfer_generate \
  --model /data1/models/Qwen3-8B-forLoRA/Qwen3-8B \
  --target-prompt-tokens 4096 \
  --max-new-tokens 512 \
  --sampling-mode greedy_fast \
  --compute-logprobs false \
  --ignore-eos
```
"""

from __future__ import annotations

from test.inference.common import build_parser, run_benchmark


def main() -> None:
    """Run the FlashInfer-only benchmark."""

    parser = build_parser()
    parser.description = "Benchmark only the project FlashInfer paged-KV generate path."
    args = parser.parse_args()
    args.benchmark_path = "flashinfer"
    run_benchmark(args)


if __name__ == "__main__":
    main()
