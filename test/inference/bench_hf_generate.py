"""Benchmark only the project HF generate path.

Example:

```bash
python -m test.inference.bench_hf_generate \
  --model /data1/models/Qwen3-8B-forLoRA/Qwen3-8B \
  --target-prompt-tokens 4096 \
  --max-new-tokens 512
```
"""

from __future__ import annotations

from test.inference.common import build_parser, run_benchmark


def main() -> None:
    """Run the HF-only benchmark."""

    parser = build_parser()
    parser.description = "Benchmark only the project HF generate path."
    args = parser.parse_args()
    args.benchmark_path = "hf"
    run_benchmark(args)


if __name__ == "__main__":
    main()
