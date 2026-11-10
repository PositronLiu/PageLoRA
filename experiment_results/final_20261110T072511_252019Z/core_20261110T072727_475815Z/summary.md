# Benchmark summary

Throughput = total output tokens / total measured request time. p95 uses linear interpolation.
Device memory is sampled across loading, warmup and measurement; it is not an exact process peak.
Failed/incomplete rows must not be used to claim speedups. Timings inherit log rounding.

| Job | Status | Mean s | p95 s | Output tok/s | Device sampled MiB | Graph speedup |
|---|---|---:|---:|---:|---:|---:|
| hf_512 | ok | 1.9715 | 1.9737 | 64.9262 | 15713.0000 | - |
| vllm_512 | ok | 0.7580 | 0.7599 | 168.8721 | 66787.0000 | - |
| flashinfer_512_graph_0 | ok | 1.8072 | 1.8106 | 70.8274 | 24149.0000 | - |
| flashinfer_512_graph_1 | ok | 1.0030 | 1.0658 | 127.6146 | 25221.0000 | 1.8018 |
| hf_2048 | ok | 1.8769 | 1.8790 | 68.1979 | 16533.0000 | - |
| vllm_2048 | ok | 0.7855 | 0.7871 | 162.9515 | 66787.0000 | - |
| flashinfer_2048_graph_0 | ok | 1.8737 | 1.9480 | 68.3159 | 24633.0000 | - |
| flashinfer_2048_graph_1 | ok | 1.0386 | 1.0970 | 123.2440 | 25605.0000 | 1.8040 |
