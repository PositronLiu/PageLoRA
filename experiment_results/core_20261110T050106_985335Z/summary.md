# Benchmark summary

Throughput = total output tokens / total measured request time. p95 uses linear interpolation.
Device memory is sampled across loading, warmup and measurement; it is not an exact process peak.
Failed/incomplete rows must not be used to claim speedups. Timings inherit log rounding.

| Job | Status | Mean s | p95 s | Output tok/s | Device sampled MiB | Graph speedup |
|---|---|---:|---:|---:|---:|---:|
| hf_512 | ok | 1.0292 | 1.0342 | 31.0931 | 3923.0000 | - |
