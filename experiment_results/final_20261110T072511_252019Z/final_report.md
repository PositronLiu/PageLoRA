# PageLoRA 实验报告

正确性检查：通过

## 性能实测

| 任务 | 状态 | 平均耗时 s | p95 s | 输出 token/s |
|---|---|---:|---:|---:|
| hf_512 | ok | 1.9715 | 1.9737 | 64.9262 |
| vllm_512 | ok | 0.7580 | 0.7599 | 168.8721 |
| flashinfer_512_graph_0 | ok | 1.8072 | 1.8106 | 70.8274 |
| flashinfer_512_graph_1 | ok | 1.0030 | 1.0658 | 127.6146 |
| hf_2048 | ok | 1.8769 | 1.8790 | 68.1979 |
| vllm_2048 | ok | 0.7855 | 0.7871 | 162.9515 |
| flashinfer_2048_graph_0 | ok | 1.8737 | 1.9480 | 68.3159 |
| flashinfer_2048_graph_1 | ok | 1.0386 | 1.0970 | 123.2440 |
| prefix_flashinfer_512_reset_true | ok | 0.9829 | 0.9973 | 130.2309 |
| prefix_flashinfer_512_reset_false | ok | 0.9968 | 1.0007 | 128.4087 |
| prefix_vllm_512_reset_true | ok | 0.7487 | 0.7571 | 170.9584 |
| prefix_vllm_512_reset_false | ok | 0.7409 | 0.7415 | 172.7669 |
| prefix_flashinfer_2048_reset_true | ok | 0.9954 | 1.0304 | 128.5870 |
| prefix_flashinfer_2048_reset_false | ok | 0.9803 | 0.9838 | 130.5689 |
| prefix_vllm_2048_reset_true | ok | 0.7631 | 0.7910 | 167.7407 |
| prefix_vllm_2048_reset_false | ok | 0.7524 | 0.7538 | 170.1234 |

## 运行设备与范围

{"profile": "full", "hardware": {"name": "NVIDIA H800", "capability": [9, 0], "memory_gib": 79.178955078125, "bf16": true, "torch": "2.10.0+cu128", "cuda": "12.8", "dtype": "float16"}}

T4 档仅为 HF/vLLM 小模型预实验，不覆盖 FlashInfer/Graph/前缀复用；不同 GPU 或精度的结果不能直接作为优化加速比。

证据完整性：完整，可进入结果分析

## 结果分析待填写

- 在相同实际输入长度、输出长度下比较 HF、vLLM、FlashInfer eager 与 graph 的端到端耗时。
- 引用 core 汇总中的 graph_speedup_total；小于 1 也应如实报告。
- 比较 prefix reset=true/false，并结合首请求与后续请求耗时解释前缀复用。
- 显存为加载、预热、测量全程的整卡采样值，不是精确进程峰值。
- p95 来自有限重复样本；结果限于当前模型、GPU、长度和串行负载。
- 本实验使用 base 权重，不代表多 LoRA 并发服务性能；graph 开关对照需结合正确性 replay 证据。

## 附件

保留整个目录：源码哈希、源码快照、流水线状态、环境版本、原始日志、正确性输出、逐请求 CSV 和汇总。
性能数字由日志自动生成；解释、局限与最终汇报需根据实测补充。
