# 实验结果归档

`run_final_experiments` 在成功、失败或可捕获的中断后自动将精简证据保存到这里。每次运行使用独立时间戳目录；T4 预实验与 H800 正式实验应分别保留。

包含报告、CSV、环境信息、源码哈希、token 输出与日志。每个文件最多 2 MiB，超限日志只保留尾部；完整 logits 数组和源码 zip 不上传，数值比较指标仍保留。具体省略项见每次运行的 `archive_manifest.json`。原始完整结果仍保留在运行时指定的 `--output-dir`，诊断数值分歧时可能还需要原始 logits。

服务器运行后可直接提交本目录：

```bash
git add experiment_results
git diff --cached --stat
git commit -m "results: add T4 small-model experiment"
git push origin HEAD
```

旧结果也可归档：`python -m scripts.archive_experiments /原始结果/final_时间戳`。同名归档存在时拒绝覆盖；请保留原归档并使用新的目录名。强制终止进程或机器关机可能来不及自动归档，此时使用上述命令手动归档已有文件。
