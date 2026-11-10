# Benchmark summary

Throughput = total output tokens / total measured request time. p95 uses linear interpolation.
Device memory is sampled across loading, warmup and measurement; it is not an exact process peak.
Failed/incomplete rows must not be used to claim speedups. Timings inherit log rounding.

| Job | Status | Mean s | p95 s | Output tok/s | Device sampled MiB | Graph speedup |
|---|---|---:|---:|---:|---:|---:|
| prefix_flashinfer_512_reset_true | ok | 0.9829 | 0.9973 | 130.2309 | 25405.0000 | - |
| prefix_flashinfer_512_reset_false | ok | 0.9968 | 1.0007 | 128.4087 | 25341.0000 | - |
| prefix_vllm_512_reset_true | ok | 0.7487 | 0.7571 | 170.9584 | 66787.0000 | - |
| prefix_vllm_512_reset_false | ok | 0.7409 | 0.7415 | 172.7669 | 66787.0000 | - |
| prefix_flashinfer_2048_reset_true | ok | 0.9954 | 1.0304 | 128.5870 | 25603.0000 | - |
| prefix_flashinfer_2048_reset_false | ok | 0.9803 | 0.9838 | 130.5689 | 25451.0000 | - |
| prefix_vllm_2048_reset_true | ok | 0.7631 | 0.7910 | 167.7407 | 66787.0000 | - |
| prefix_vllm_2048_reset_false | ok | 0.7524 | 0.7538 | 170.1234 | 66787.0000 | - |
