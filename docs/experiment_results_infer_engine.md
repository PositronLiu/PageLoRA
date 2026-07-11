# Infer Engine Experiment Results Ledger

Date: 2026-05-29

Purpose: keep one local source of truth for experiments that are already worth reusing, so future runs only target new code paths or missing ablations.

Main hardware used in these runs:

- GPU: NVIDIA GeForce RTX 3090 24GB
- Main models:
  - `/data1/models/Qwen3-8B`
  - `/data1/models/Qwen3-8B-forLoRA/Qwen3-8B`
  - `/data1/models/Llama-2-7b`
  - `/data1/models/xLAM-7b-r`
  - `/data1/models/deepseek-coder-6.7b-instruct`

Metric convention:

- `tok/s` below means decode throughput unless noted otherwise.
- Most reliable rows use `warmup_runs >= 1` and `measure_runs >= 3`.
- `.local` logs are ignored by git; this file is a local experiment index, not a paper-ready artifact by itself.

## Agent 7B Engine Bench, 2026-05-29

Sources:

- Single request logs: `.local/experiments/results/bench_agent7b_single_graph_20260529/raw_logs/`
- Prefix reuse logs: `.local/experiments/results/bench_agent7b_prefix_graph_20260529/raw_logs/`

Settings:

- Models: `/data1/models/xLAM-7b-r`, `/data1/models/deepseek-coder-6.7b-instruct`
- Single request: `target_prompt_tokens=2048`, `max_new_tokens=512`, `warmup_runs=1`, `measure_runs=3`
- Prefix reuse: `shared_prefix_tokens=2048`, `branch_suffix_tokens=64`, `prefix_requests=4`, `prefix_bench_mode=both`, `max_new_tokens=128`, `warmup_runs=1`, `measure_runs=3`
- Generation: `sampling_mode=greedy_fast`, `ignore_eos=True`, `compute_logprobs=False`, `repetition_penalty=1.0`, `throughput_timing=decode`
- FlashInfer: `--flashinfer-cuda-graph-decode`, `MINI_OPENCLAW_FLASHINFER_CUDA_GRAPH_MAX_ENTRIES=64`, packed QKV/MLP/LoRA delta enabled
- FlashInfer base uses `adapter_role=base`; LoRA merge uses `MINI_OPENCLAW_FLASHINFER_SINGLE_LORA_MERGE=1`; no-merge uses true LoRA A/B delta with `MINI_OPENCLAW_FLASHINFER_SINGLE_LORA_MERGE=0`
- vLLM: default compile/CUDA graph path, `--vllm-dtype bfloat16`, `--vllm-max-model-len 4096`, `--vllm-gpu-memory-utilization 0.8`; LoRA rows use rank-16 zero LoRA adapters to exercise vLLM LoRA serving
- Prefix FlashInfer required `PATH=$HOME/miniconda3/envs/pytorch/bin:$PATH` for `ninja` and `LD_LIBRARY_PATH=$HOME/miniconda3/envs/pytorch/lib:$LD_LIBRARY_PATH` so FlashInfer JIT prefill modules load the conda `libstdc++`.

### Single Request

| Model | Engine | Mode | Prompt | Total s | Prefill s | Decode s | tok/s | Load s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| xLAM-7b-r | FlashInfer | base | 2114 | 11.0310 | 0.3812 | 10.6154 | 48.2318 | 57.3673 |
| xLAM-7b-r | FlashInfer | LoRA merge | 2114 | 10.3858 | 0.3851 | 9.9695 | 51.3567 | 5.5027 |
| xLAM-7b-r | FlashInfer | LoRA no-merge | 2114 | 10.6918 | 0.3970 | 10.2638 | 49.8843 | 5.1170 |
| xLAM-7b-r | vLLM | base | 2114 | 10.2148 | 0.4832 | 9.7316 | 52.6123 | 46.5862 |
| xLAM-7b-r | vLLM | LoRA serving | 2114 | 11.1198 | 0.5198 | 10.6000 | 48.3018 | 54.4902 |
| DeepSeek-Coder-6.7B-Instruct | FlashInfer | base | 2098 | 10.9707 | 0.3573 | 10.5787 | 48.4001 | 17.3119 |
| DeepSeek-Coder-6.7B-Instruct | FlashInfer | LoRA merge | 2098 | 10.2582 | 0.3588 | 9.8684 | 51.8829 | 6.2392 |
| DeepSeek-Coder-6.7B-Instruct | FlashInfer | LoRA no-merge | 2098 | 10.5654 | 0.3723 | 10.1620 | 50.3840 | 5.0951 |
| DeepSeek-Coder-6.7B-Instruct | vLLM | base | 2098 | 10.8355 | 0.4408 | 10.3947 | 49.2559 | 44.6880 |
| DeepSeek-Coder-6.7B-Instruct | vLLM | LoRA serving | 2098 | 11.7268 | 0.4749 | 11.2518 | 45.5039 | 56.8055 |

### Prefix Reuse

| Model | Engine | Mode | Prefix mode | Prompt min-max | Mean req s | First req s | Reuse req s | tok/s | Load s |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| xLAM-7b-r | FlashInfer | base | single | 2147 | 2.8144 | 3.1659 | 2.6972 | 45.6996 | 5.1648 |
| xLAM-7b-r | FlashInfer | base | branch | 2168 | 2.8292 | 3.1831 | 2.7113 | 45.4610 | 2.8125 |
| xLAM-7b-r | FlashInfer | LoRA merge | single | 2147 | 2.6178 | 2.9708 | 2.5001 | 49.1699 | 5.7286 |
| xLAM-7b-r | FlashInfer | LoRA merge | branch | 2168 | 2.6297 | 2.9825 | 2.5121 | 48.9439 | 2.8352 |
| xLAM-7b-r | FlashInfer | LoRA no-merge | single | 2147 | 2.6974 | 3.0554 | 2.5781 | 47.7099 | 5.3038 |
| xLAM-7b-r | FlashInfer | LoRA no-merge | branch | 2168 | 2.7061 | 3.0646 | 2.5866 | 47.5558 | 2.8216 |
| xLAM-7b-r | vLLM | base | single | 2147 | 2.4457 | 2.4459 | 2.4456 | 52.3374 | 29.6473 |
| xLAM-7b-r | vLLM | base | branch | 2168 | 2.4461 | 2.4458 | 2.4463 | 52.3272 | 25.3304 |
| xLAM-7b-r | vLLM | LoRA serving | single | 2147 | 2.6636 | 2.6627 | 2.6640 | 48.0545 | 56.5157 |
| xLAM-7b-r | vLLM | LoRA serving | branch | 2168 | 2.6683 | 2.6666 | 2.6688 | 47.9709 | 36.6326 |
| DeepSeek-Coder-6.7B-Instruct | FlashInfer | base | single | 2151 | 2.7825 | 3.1154 | 2.6715 | 46.2063 | 5.0214 |
| DeepSeek-Coder-6.7B-Instruct | FlashInfer | base | branch | 2175 | 2.7847 | 3.1169 | 2.6739 | 46.1687 | 2.7640 |
| DeepSeek-Coder-6.7B-Instruct | FlashInfer | LoRA merge | single | 2151 | 2.5783 | 2.9069 | 2.4687 | 49.8952 | 5.3154 |
| DeepSeek-Coder-6.7B-Instruct | FlashInfer | LoRA merge | branch | 2175 | 2.5896 | 2.9147 | 2.4812 | 49.6691 | 2.9050 |
| DeepSeek-Coder-6.7B-Instruct | FlashInfer | LoRA no-merge | single | 2151 | 2.6567 | 2.9931 | 2.5445 | 48.4194 | 5.0481 |
| DeepSeek-Coder-6.7B-Instruct | FlashInfer | LoRA no-merge | branch | 2175 | 2.6621 | 2.9981 | 2.5501 | 48.3197 | 2.8273 |
| DeepSeek-Coder-6.7B-Instruct | vLLM | base | single | 2151 | 2.5898 | 2.5888 | 2.5902 | 49.4244 | 40.2514 |
| DeepSeek-Coder-6.7B-Instruct | vLLM | base | branch | 2175 | 2.5959 | 2.5957 | 2.5960 | 49.3081 | 32.1972 |
| DeepSeek-Coder-6.7B-Instruct | vLLM | LoRA serving | single | 2151 | 2.8052 | 2.8042 | 2.8055 | 45.6299 | 54.0934 |
| DeepSeek-Coder-6.7B-Instruct | vLLM | LoRA serving | branch | 2175 | 2.8124 | 2.8121 | 2.8125 | 45.5133 | 38.2647 |

Reusable conclusions:

- Both accepted agent-capable 5B-7B models run under the current FlashInfer path with CUDA graph enabled for base, LoRA merge, and true LoRA no-merge.
- Single-request long decode favors FlashInfer LoRA merge for both models: 51.36 tok/s on xLAM and 51.88 tok/s on DeepSeek. vLLM base remains strongest for xLAM base at 52.61 tok/s, while FlashInfer and vLLM base are close on DeepSeek.
- vLLM LoRA serving is slower than vLLM base by about 8.2% on xLAM and 7.6% on DeepSeek in the 512-token single-request run.
- Prefix reuse shows the same ranking pattern at 128 decode tokens: FlashInfer LoRA merge is faster than FlashInfer no-merge and vLLM LoRA; vLLM base is still fastest on xLAM prefix, while DeepSeek vLLM base and FlashInfer LoRA merge are close.
- FlashInfer prefix reuse rows show request 2-4 are faster than request 1 because the shared prefix is reused. vLLM rows are hot-cache steady-state and show nearly identical first/reuse latency in this benchmark harness.
- vLLM load/init remains much higher because each prefix mode initializes a vLLM engine and performs compile/profiling/CUDA graph capture. FlashInfer branch-mode load is low because FlashInfer JIT artifacts were already cached by the preceding single-mode run.

## Primary Results To Reuse

These are the current best numbers for LoRA, CUDA graph, and vLLM comparison.

Source: `.local/experiments/results/bench_lora_graph_matrix_20260523/summary.md`

Settings:

- `target_prompt_tokens=2048`, `max_new_tokens=128`
- `warmup_runs=1`, `measure_runs=3`
- `sampling_mode=greedy_fast`, `ignore_eos=True`
- FlashInfer no-merge uses true LoRA A/B delta: `MINI_OPENCLAW_FLASHINFER_EFFECTIVE_WEIGHTS=0`, `MINI_OPENCLAW_FLASHINFER_MEMORY_MODE=balanced`
- vLLM uses LoRA serving path with zero LoRA adapters and default CUDA graph enabled

| Model | Engine / config | LoRA mode | CUDA graph | Prompt tokens | Mean total s | Mean prefill s | Mean decode s | Decode tok/s | Load/init s |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | FlashInfer | merge | on | 2100 | 3.1618 | 0.3870 | 2.7421 | 46.6840 | 6.6161 |
| Qwen3-8B | FlashInfer | merge | off | 2100 | 3.7097 | 0.3899 | 3.3062 | 38.7156 | 6.2137 |
| Qwen3-8B | FlashInfer | no-merge | on | 2100 | 3.2633 | 0.4004 | 2.8305 | 45.2289 | 5.8111 |
| Qwen3-8B | FlashInfer | no-merge | off | 2100 | 4.7006 | 0.4034 | 4.2844 | 29.8762 | 5.8159 |
| Qwen3-8B | vLLM | no-merge LoRA serving | on | 2100 | 3.3395 | 0.5039 | 2.8356 | 45.1408 | 40.6242 |
| Llama-2-7B | FlashInfer | merge | on | 2119 | 2.9126 | 0.3587 | 2.5244 | 50.7065 | 8.2471 |
| Llama-2-7B | FlashInfer | merge | off | 2119 | 2.9442 | 0.3600 | 2.5735 | 49.7368 | 6.8938 |
| Llama-2-7B | FlashInfer | no-merge | on | 2119 | 2.9954 | 0.3723 | 2.5944 | 49.3432 | 6.4589 |
| Llama-2-7B | FlashInfer | no-merge | off | 2119 | 3.8351 | 0.3718 | 3.4534 | 37.0656 | 6.3649 |
| Llama-2-7B | vLLM | no-merge LoRA serving | on | 2119 | 3.2256 | 0.4754 | 2.7502 | 46.5418 | 46.3767 |

Reusable conclusions:

- FlashInfer CUDA graph supports both single LoRA merge and true no-merge LoRA delta paths.
- Qwen3-8B:
  - merge graph is +20.6% over merge no-graph.
  - no-merge graph is +51.4% over no-merge no-graph.
  - merge graph is +3.4% over vLLM LoRA.
  - no-merge graph is roughly tied with vLLM LoRA: +0.2%.
- Llama-2-7B:
  - merge graph is only +1.9% over merge no-graph.
  - no-merge graph is +33.1% over no-merge no-graph.
  - merge graph is +8.9% over vLLM LoRA.
  - no-merge graph is +6.0% over vLLM LoRA.
- vLLM load/init is much higher in these runs, about 40-46s, because it performs compile/profiling/KV allocation/broad CUDA graph capture.
- FlashInfer decode timing still includes page table and FlashInfer planning. CUDA graph helps model execution, but page-table/plan reuse is still the next stability bottleneck.

## Multi-Request Prefix Reuse With LoRA And CUDA Graph

Source: `.local/experiments/results/bench_prefix_lora_graph_20260523/summary.md`

Settings:

- `prefix_bench_mode=branch`, `prefix_requests=4`
- `shared_prefix_tokens=2048`, `branch_suffix_tokens=64`
- `max_new_tokens=128`, `warmup_runs=1`, `measure_runs=3`
- FlashInfer uses `prefix_reset_cache_between_runs=true`: request 1 is cold in each measured sequence, requests 2-4 reuse the shared prefix.
- vLLM keeps prefix cache across warmup and measured calls in the current benchmark, so its rows are hot-cache steady-state LoRA serving numbers.
- vLLM logs confirm `enable_prefix_caching=True`, `enforce_eager=False`, CUDA graph capture enabled, and `cudagraph_specialize_lora=True`.

| Model | Engine / config | LoRA mode | CUDA graph | Prompt tokens | Mean req s | First req s | Reuse req s | tok/s | Load/init s |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | FlashInfer | merge | on | 2103 | 2.8103 | 3.1458 | 2.6984 | 45.7486 | 6.3224 |
| Qwen3-8B | FlashInfer | merge | off | 2103 | 3.3463 | 3.6886 | 3.2322 | 38.3769 | 5.9366 |
| Qwen3-8B | FlashInfer | no-merge | on | 2103 | 2.9129 | 3.2765 | 2.7918 | 44.1543 | 5.7809 |
| Qwen3-8B | FlashInfer | no-merge | off | 2103 | 4.2395 | 4.5795 | 4.1261 | 30.2545 | 5.9926 |
| Qwen3-8B | vLLM | LoRA serving | on | 2103 | 2.8862 | 2.8851 | 2.8865 | 44.3496 | 41.5594 |
| Llama-2-7B | FlashInfer | merge | on | 2160 | 2.5912 | 2.9096 | 2.4850 | 49.6296 | 6.5095 |
| Llama-2-7B | FlashInfer | merge | off | 2160 | 2.7231 | 3.0475 | 2.6150 | 47.2130 | 6.2741 |
| Llama-2-7B | FlashInfer | no-merge | on | 2160 | 2.6662 | 2.9949 | 2.5566 | 48.2344 | 6.0053 |
| Llama-2-7B | FlashInfer | no-merge | off | 2160 | 3.3993 | 3.7284 | 3.2896 | 37.7665 | 6.8366 |
| Llama-2-7B | vLLM | LoRA serving | on | 2160 | 2.8057 | 2.8047 | 2.8061 | 45.6209 | 38.8070 |

Reusable conclusions:

- Qwen3-8B: FlashInfer merge graph is +19.2% over merge no-graph and +3.2% over vLLM LoRA. No-merge graph is +45.9% over no-merge no-graph and roughly tied with vLLM LoRA at -0.4%.
- Llama-2-7B: FlashInfer merge graph is +5.1% over merge no-graph and +8.8% over vLLM LoRA. No-merge graph is +27.7% over no-merge no-graph and +5.7% over vLLM LoRA.
- FlashInfer graph helps both merge and true no-merge LoRA paths, but page-table/FlashInfer plan management still appears in graph-row latency.
- vLLM has very stable hot-cache request latency but much higher load/init time due to compile/profiling/KV allocation/CUDA graph capture.

## Qwen3-8B 2048-Token Engine Comparison

Source: `experiments/inference_3090_2048_compare_20260520_162006`

Settings:

- `/data1/models/Qwen3-8B`
- `target_prompt_tokens=2048`, actual prompt tokens 2100
- `max_new_tokens=512`
- `warmup_runs=3`, `measure_runs=5`
- No FlashInfer CUDA graph in this run

| Engine / config | LoRA mode | Prompt tokens | Mean total s | Mean prefill s | Mean decode s | Decode tok/s | Load/init s |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM base | none | 2100 | 10.9835 | 0.4764 | 10.5071 | 48.7288 | 41.7563 |
| vLLM LoRA | vLLM LoRA | 2100 | 12.0410 | 0.5146 | 11.5264 | 44.4198 | 68.5188 |
| FlashInfer base | none | 2100 | 13.3748 | 0.4032 | 12.9518 | 39.5314 | 5.4178 |
| FlashInfer LoRA effective on | merged/effective weights | 2100 | 13.6070 | 0.4030 | 13.1903 | 38.8164 | 5.1953 |
| FlashInfer LoRA effective off | A/B delta path | 2100 | 16.1313 | 0.4117 | 15.7058 | 32.5994 | 5.4476 |
| HF base | none | 2100 | 16.6803 | 0.5699 | 16.0949 | 31.8117 | 5.1675 |
| HF LoRA | HF LoRA | 2100 | 21.9231 | 0.5818 | 21.3326 | 24.0012 | 5.1745 |

Reusable conclusions:

- vLLM base is fastest before our CUDA graph work: 48.73 tok/s.
- FlashInfer effective-weight LoRA is +19.1% over FlashInfer A/B delta path.
- FlashInfer base and effective LoRA load in about 5.2-5.4s, much faster than vLLM initialization.
- This run is still useful for long 512-token decode stability, but the LoRA/CUDA graph matrix above supersedes it for current graph-path claims.

## Qwen3-8B 4096-Token Long Context Comparison

Sources:

- `experiments/inference_split_20260519_110531/comparison.md`
- `experiments/inference_compare_20260519_152535/comparison.md`

Settings:

- `/data1/models/Qwen3-8B-forLoRA/Qwen3-8B`
- `target_prompt_tokens=4096`, actual prompt tokens 4140
- `max_new_tokens=512`
- `warmup_runs=3`, `measure_runs=5`
- No FlashInfer CUDA graph

Best split-timing table:

| Engine / config | Prompt tokens | Mean total s | Mean prefill s | Mean decode s | Decode tok/s | Load/init s |
|---|---:|---:|---:|---:|---:|---:|
| vLLM base | 4140 | 13.3089 | 0.6043 | 12.7046 | 40.3004 | 24.3271 |
| vLLM LoRA | 4140 | 14.3753 | 0.6604 | 13.7149 | 37.3318 | 35.2810 |
| FlashInfer LoRA cache on | 4140 | 13.7387 | 0.5598 | 13.1606 | 38.9044 | 10.8244 |
| FlashInfer LoRA cache off | 4140 | 16.3951 | 0.5831 | 15.7980 | 32.4225 | 10.4629 |
| HF base | 4140 | 16.8817 | 0.7497 | 16.1192 | 31.7634 | 7.8485 |
| HF LoRA | 4140 | 17.8492 | 0.7812 | 17.0609 | 30.0103 | 10.9041 |

Reusable conclusions:

- FlashInfer LoRA effective/cache-on reaches 38.90 tok/s, about 96.5% of vLLM base and above vLLM LoRA.
- Effective-weight cache-on is about +20.0% over cache-off in this long-context setup.
- This is the best historical evidence for the value of FlashInfer effective weight caching before CUDA graph support.

## Llama-2-7B Base And LoRA Comparison

Sources:

- `.local/experiments/results/llama2_7b_flashinfer_base_20260522_161254.log`
- `.local/experiments/results/llama2_7b_flashinfer_lora_merge_20260522_162624.log`
- `.local/experiments/results/llama2_7b_flashinfer_lora_delta_20260522_162741.log`
- `.local/experiments/results/llama2_7b_hf_base_20260522_161602.log`
- `.local/experiments/results/llama2_7b_vllm_base_20260522_161411.log`
- `.local/experiments/results/llama2_7b_vllm_lora_20260522_162408.log`

Settings:

- `/data1/models/Llama-2-7b`
- actual prompt tokens 2119
- `max_new_tokens=512`
- `warmup_runs=1`, `measure_runs=3`
- No FlashInfer CUDA graph

| Engine / config | LoRA mode | Prompt tokens | Mean total s | Mean prefill s | Mean decode s | Decode tok/s | Load/init s |
|---|---|---:|---:|---:|---:|---:|---:|
| FlashInfer base | none | 2119 | 11.4782 | 0.3549 | 11.1071 | 46.1266 | 6.4651 |
| FlashInfer LoRA merge | merged/effective weights | 2119 | 10.6108 | 0.3589 | 10.2401 | 49.9997 | 7.8085 |
| FlashInfer LoRA delta | A/B delta path | 2119 | 13.7566 | 0.3722 | 13.3729 | 38.2865 | 6.8325 |
| HF base | none | 2119 | 13.8247 | 0.5027 | 13.3098 | 38.4681 | 6.2306 |
| vLLM base | none | 2119 | 10.8372 | 0.4442 | 10.3930 | 49.2641 | 47.7554 |
| vLLM LoRA | vLLM LoRA | 2119 | 11.7044 | 0.4776 | 11.2268 | 45.6052 | 59.9369 |

Reusable conclusions:

- Llama-2-7B FlashInfer LoRA merge is competitive with vLLM base before CUDA graph: 50.00 vs 49.26 tok/s.
- FlashInfer A/B delta path is much slower than merge/effective weights: 38.29 vs 50.00 tok/s.
- vLLM LoRA is slower than vLLM base by about 7.4%, with much higher initialization time than FlashInfer.

## Prefix Cache And Reuse Experiments

Use these as prefix/KV-cache behavior references, not as primary engine throughput numbers.

### FlashInfer Prefix Reuse

Source: `.local/experiments/results/inference_request_prefix_async_20260521_150541.log`

Settings:

- `/data1/models/Qwen3-8B`
- backend: FlashInfer
- shared prefix target: 2048 tokens
- `requests=4`, `max_new_tokens=512`

| Mode | Prompt tokens | Mean request latency s | First request latency s | Reuse request latency s | Mean tok/s |
|---|---:|---:|---:|---:|---:|
| single shared prefix | 2084 | 12.8044 | 13.1486 | 12.6897 | 39.9957 |
| branch shared prefix | 2169 | 12.9198 | 13.2700 | 12.8030 | 39.6388 |

Reusable conclusion:

- FlashInfer prefix reuse path works for repeated shared-prefix requests; reuse requests are modestly faster than the first request in this single-GPU sequential setup.

### vLLM Prefix Reference

Source: `experiments/inference_3090_2048_prefix_20260520_163751`

Status:

- vLLM prefix rows completed with status 0.
- FlashInfer rows in this directory exited status 1, so do not use those FlashInfer rows. Use the `.local/experiments/results/inference_request_prefix_async_20260521_150541.log` FlashInfer rows above instead.

| Engine / config | Prefix mode | Prompt tokens | Mean tok/s | Load/init s |
|---|---|---:|---:|---:|
| vLLM base | single | 2084 | 48.7139 | 26.8876 |
| vLLM base | branch | 2103 | 48.6161 | 29.6424 |
| vLLM LoRA | single | 2084 | 44.3798 | 53.2718 |
| vLLM LoRA | branch | 2103 | 44.3252 | 42.9559 |

## Agent Runtime Smoke

These are runtime integration smoke numbers for agent request execution. They are useful for checking queueing and end-to-end runtime behavior, but should not be mixed with raw engine throughput tables.

Sources:

- `.local/experiments/results/inference_agent_runtime_20260521_151331.log`
- `.local/experiments/results/inference_agent_runtime_qknorm_20260521_151610.log`

| Run | Request | Prompt tokens | Response tokens | Queue wait s | Execution s | Execution tok/s |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 2068 | 128 | 0.000665 | 5.5413 | 23.0993 |
| baseline | 2 | 2225 | 128 | 0.000406 | 4.2558 | 30.0764 |
| qk-norm path | 1 | 2068 | 128 | 0.000717 | 5.0441 | 25.3760 |
| qk-norm path | 2 | 2225 | 115 | 0.000412 | 3.0234 | 38.0371 |

Reusable conclusions:

- Scheduler queue wait was sub-millisecond in these smoke runs.
- The qk-norm path improved end-to-end request execution relative to the earlier baseline.

## CUDA Graph Functional Smoke

Use these only to confirm graph behavior after code changes. Do not cite them as throughput numbers because they generate only 8 tokens.

Sources:

- `.local/experiments/results/smoke_cuda_graph_cache_20260523/qwen3_8b_graph_cache_after_signature.log`
- `.local/experiments/results/smoke_cuda_graph_lora_no_merge/qwen3_8b_graph_lora_no_merge_balanced.log`

| Smoke | Prompt tokens | Generated tokens | Observation |
|---|---:|---:|---|
| Qwen3-8B graph cache after page-table signature | 128 | 8 | Warmup captured one graph bucket; measure reused cached graph with `decode_cuda_graph_s=0.001293` and no recapture. |
| Qwen3-8B no-merge LoRA graph, balanced mode | 128 | 8 | True no-merge LoRA graph path executed successfully. |

Reusable conclusions:

- Graph cache reuse works after adding the page-table buffer signature to the cache key.
- The true no-merge LoRA path can run under CUDA graph capture.

## Small-Model Sanity Experiments

These were useful for validating benchmark plumbing and warmup behavior, but are not representative of Qwen/Llama production performance.

Sources:

- `.local/experiments/results/llama_smollm2_both_warm_20260522_152624.log`
- `.local/experiments/results/llama_smollm2_vllm_eager_20260522_161724.log`
- `.local/experiments/results/llama_smollm2_vllm_spawn_20260522_154335.log`

| Model / setup | Engine | Prompt tokens | Generated tokens | Decode tok/s | Load/init s | Notes |
|---|---|---:|---:|---:|---:|---|
| SmolLM2 warm run | HF | 515 | 64 | 59.4399 | 3.0556 | Baseline after warmup. |
| SmolLM2 warm run | FlashInfer | 515 | 64 | 66.1630 | 1.0190 | FlashInfer beats HF after warmup. |
| SmolLM2 eager | vLLM eager | 515 | 64 | 63.6134 | 23.9535 | Reasonable vLLM small-model reference. |
| SmolLM2 spawn | vLLM | 515 | 64 | 197.7420 | 37.2045 | Treat as an outlier/smoke result, not a paper number. |

## Superseded Or Caution Rows

Keep these entries to avoid rerunning or accidentally citing stale numbers.

### Older CUDA Graph LoRA Merge Benchmark

Source: `.local/experiments/results/bench_cuda_graph_lora_merge_20260523/summary.md`

Status: superseded by `.local/experiments/results/bench_lora_graph_matrix_20260523/summary.md`.

Why superseded:

- It was run before the latest graph bucket/page-table buffer signature cache reuse changes.
- Capture/planning cost was counted into measured decode more heavily.
- It only covered LoRA merge, not true no-merge LoRA graph.

Historical numbers:

| Model | Engine / config | Decode tok/s |
|---|---|---:|
| Qwen3-8B | FlashInfer graph + LoRA merge | 43.0701 |
| Qwen3-8B | FlashInfer no graph + LoRA merge | 38.6731 |
| Qwen3-8B | vLLM LoRA | 45.1385 |
| Llama-2-7B | FlashInfer graph + LoRA merge | 46.7789 |
| Llama-2-7B | FlashInfer no graph + LoRA merge | 49.7487 |
| Llama-2-7B | vLLM LoRA | 46.5370 |

Use the newer matrix table for current graph claims.

### Failed Or Incomplete Prefix Runs

Source: `experiments/inference_3090_2048_prefix_20260520_163751`

- vLLM rows completed successfully.
- FlashInfer rows exited with status 1 and only partially logged request-level results.
- Do not cite those partial FlashInfer prefix rows; use `.local/experiments/results/inference_request_prefix_async_20260521_150541.log` instead.

## Quick Selection Guide

Use this to decide whether a new run is actually needed.

| Question | Existing result to use | Rerun only if |
|---|---|---|
| FlashInfer CUDA graph vs no graph with LoRA? | Primary LoRA CUDA graph matrix, 2026-05-23 | CUDA graph cache/page-table/plan code changes. |
| Does graph support LoRA no-merge? | Primary matrix plus no-merge smoke | LoRA runtime or graph key invalidation changes. |
| FlashInfer vs vLLM LoRA on Qwen/Llama? | Primary matrix | vLLM version, GPU util, model path, or LoRA adapter changes. |
| Effective weight cache value on long Qwen context? | 4096-token long context comparison | Effective weight cache implementation changes. |
| Llama base/LoRA behavior before graph? | Llama-2-7B comparison, 2026-05-22 | Llama model, LoRA rank/modules, or FlashInfer kernel path changes. |
| Prefix reuse behavior? | FlashInfer prefix async and vLLM prefix reference | Prefix cache manager or radix metadata changes. |
| Agent runtime end-to-end smoke? | Agent runtime smoke rows | Scheduler, agent runtime, or qk-norm path changes. |

## Log Index

Primary:

- `.local/experiments/results/bench_lora_graph_matrix_20260523/summary.md`
- `.local/experiments/results/bench_lora_graph_matrix_20260523/*.log`

Historical engine comparisons:

- `experiments/inference_3090_2048_compare_20260520_162006/*.raw.log`
- `experiments/inference_split_20260519_110531/comparison.md`
- `experiments/inference_compare_20260519_152535/comparison.md`
- `.local/experiments/results/llama2_7b_*20260522*.log`

Prefix and agent:

- `.local/experiments/results/inference_request_prefix_async_20260521_150541.log`
- `.local/experiments/results/inference_request_prefix_sync_20260521_151130.log`
- `experiments/inference_3090_2048_prefix_20260520_163751/*.raw.log`
- `.local/experiments/results/inference_agent_runtime_20260521_151331.log`
- `.local/experiments/results/inference_agent_runtime_qknorm_20260521_151610.log`

Smoke:

- `.local/experiments/results/smoke_cuda_graph_cache_20260523/*.log`
- `.local/experiments/results/smoke_cuda_graph_lora_no_merge/*.log`
- `.local/experiments/results/llama_smollm2_*.log`

Result notes and engineering writeups:

- `.local/experiments/results/blog.md`
- `.local/experiments/results/cuda_graph.md`
- `.local/experiments/results/final_optimization.md`
- `.local/experiments/results/oom.md`
- `.local/experiments/results/slow.md`
- `.local/experiments/results/vllm_lora.md`
