"""Correctness-gated experiment bundle; no GPU imports in dry-run/report mode."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def source_manifest(root=ROOT):
    records = {}
    for folder in ('core', 'model', 'test', 'scripts', 'data'):
        for path in sorted((root / folder).rglob('*.py')):
            records[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('requirements.txt', 'pyproject.toml'):
        path = root / name
        if path.exists():
            records[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return records


def build_commands(args, destination):
    profile = getattr(args, 'profile', 'full')
    dtype = getattr(args, 'dtype', 'auto')
    common = ['--dtype', dtype, '--model', args.model, '--gpu', args.gpu, '--output-dir', str(destination)]
    commands = [[sys.executable, '-m', 'scripts.check_inference_correctness', *common,
                 '--max-model-len', str(args.max_model_len)]]
    if profile == 't4':
        commands[0] += ['--variants', 'hf', 'vllm']
    for suite in (('core',) if profile == 't4' else ('core', 'prefix')):
        commands.append([sys.executable, '-m', 'scripts.run_course_experiments', *common,
            '--suite', suite, '--prompt-lengths', *map(str, args.prompt_lengths),
            '--output-tokens', str(args.output_tokens), '--warmup-runs', str(args.warmup_runs),
            '--measure-runs', str(args.measure_runs), '--max-model-len', str(args.max_model_len)])
    if profile == 't4':
        commands[-1] += ['--backends', 'hf', 'vllm']
    return commands


def write_report(destination):
    destination = Path(destination)
    gates = list(destination.glob('correctness_*/report.json'))
    correct = len(gates) == 1 and json.loads(gates[0].read_text(encoding='utf-8')).get('strict_pass') is True
    lines = ['# PageLoRA 实验报告', '', f'正确性检查：{"通过" if correct else "未通过或证据不完整"}', '',
             '## 性能实测', '', '| 任务 | 状态 | 平均耗时 s | p95 s | 输出 token/s |',
             '|---|---|---:|---:|---:|']
    complete = correct
    for suite in ('core', 'prefix'):
        files = list(destination.glob(f'{suite}_*/summary.json'))
        if len(files) != 1:
            complete = False
            lines.append(f'| {suite} | 缺少唯一汇总 | - | - | - |')
            continue
        rows = json.loads(files[0].read_text(encoding='utf-8'))
        complete = complete and bool(rows) and all(r['status'] == 'ok' for r in rows)
        for row in rows:
            metrics = [f'{row[k]:.4f}' if isinstance(row.get(k), (float, int)) else '-'
                       for k in ('mean_total_s', 'p95_total_s', 'total_tokens_per_s')]
            lines.append(f"| {row['name']} | {row['status']} | " + ' | '.join(metrics) + ' |')
    state_path = destination / 'pipeline.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    lines += ['', '## 运行设备与范围', '',
              json.dumps({'profile': state.get('profile', 'full'), 'hardware': state.get('hardware')}, ensure_ascii=False),
              '', 'T4 档仅为 HF/vLLM 小模型预实验，不覆盖 FlashInfer/Graph/前缀复用；不同 GPU 或精度的结果不能直接作为优化加速比。']
    complete = complete and state.get('completed_stages') == 3 and state.get('source_unchanged') is True
    lines += ['', f'证据完整性：{"完整，可进入结果分析" if complete else "不完整，不应据此结题或宣称加速"}', '',
        '## 结果分析待填写', '',
        '- 在相同实际输入长度、输出长度下比较 HF、vLLM、FlashInfer eager 与 graph 的端到端耗时。',
        '- 引用 core 汇总中的 graph_speedup_total；小于 1 也应如实报告。',
        '- 比较 prefix reset=true/false，并结合首请求与后续请求耗时解释前缀复用。',
        '- 显存为加载、预热、测量全程的整卡采样值，不是精确进程峰值。',
        '- p95 来自有限重复样本；结果限于当前模型、GPU、长度和串行负载。',
        '- 本实验使用 base 权重，不代表多 LoRA 并发服务性能；graph 开关对照需结合正确性 replay 证据。', '',
        '## 附件', '',
        '保留整个目录：源码哈希、源码快照、流水线状态、环境版本、原始日志、正确性输出、逐请求 CSV 和汇总。',
        '性能数字由日志自动生成；解释、局限与最终汇报需根据实测补充。']
    (destination / 'final_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    parser.add_argument('--profile', choices=('full', 't4'), default='full')
    parser.add_argument('--dtype', choices=('auto', 'float16', 'bfloat16'), default='auto')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--output-dir', default='course_results')
    parser.add_argument('--prompt-lengths', nargs='+', type=int, default=[512, 2048])
    parser.add_argument('--output-tokens', type=int, default=128)
    parser.add_argument('--warmup-runs', type=int, default=2)
    parser.add_argument('--measure-runs', type=int, default=10)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--report-only', type=Path)
    args = parser.parse_args()
    if args.report_only:
        raise SystemExit(0 if write_report(args.report_only) else 1)
    if not args.model:
        parser.error('--model is required')
    if min(args.prompt_lengths + [args.output_tokens, args.measure_runs]) < 1 or args.warmup_runs < 0:
        parser.error('Invalid lengths or repetition counts')
    if not args.gpu.strip() or ',' in args.gpu or max(args.prompt_lengths) + args.output_tokens + 512 > args.max_model_len:
        parser.error('Select one GPU and sufficient --max-model-len')
    destination = Path(args.output_dir).resolve() / ('final_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    hardware = None
    if not args.dry_run:
        from scripts.hardware_support import probe_gpu
        hardware = probe_gpu(args.gpu, args.dtype)
        args.dtype = hardware['dtype']
    commands = build_commands(args, destination)
    if args.dry_run:
        for command in commands:
            print(json.dumps(command, ensure_ascii=False))
        return
    destination.mkdir(parents=True)
    manifest = source_manifest()
    (destination / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    import zipfile
    with zipfile.ZipFile(destination / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in manifest:
            archive.write(ROOT / name, name)
    state = {'profile': args.profile, 'hardware': hardware, 'commands': commands, 'completed_stages': 0, 'source_unchanged': True}
    try:
        for command in commands:
            if source_manifest() != manifest:
                state['source_unchanged'] = False
                raise RuntimeError('Source changed during experiments; start a fresh bundle')
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                raise RuntimeError(f'Stage failed with exit code {result.returncode}; inspect retained logs')
            state['completed_stages'] += 1
    except (RuntimeError, KeyboardInterrupt) as exc:
        state['error'] = str(exc) or 'Interrupted'
    finally:
        state['source_unchanged'] = source_manifest() == manifest
        (destination / 'pipeline.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
        complete = write_report(destination)
        print('Results:', destination, flush=True)
        from scripts.archive_experiments import archive
        print('Git-ready results:', archive(destination), flush=True)
    success = complete or (args.profile == 't4' and state['completed_stages'] == 2 and state['source_unchanged'] and 'error' not in state)
    raise SystemExit(0 if success else 1)


if __name__ == '__main__':
    main()
