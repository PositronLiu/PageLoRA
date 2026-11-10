"""Compare opt-in decode optimizations behind the existing correctness gate."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from scripts.archive_experiments import archive
from scripts.run_final_experiments import ROOT, source_manifest

HOST_PLAN = 'MINI_OPENCLAW_FLASHINFER_HOST_PLAN'
FUSED_NORM = 'MINI_OPENCLAW_FLASHINFER_FUSED_ADD_RMSNORM'
VARIANTS = {'baseline': ('0', '0'), 'host_plan': ('1', '0'),
            'fused_norm': ('0', '1'), 'combined': ('1', '1')}


def build_stages(args, name, destination):
    host, fused = VARIANTS[name]
    env = dict(os.environ, **{HOST_PLAN: host, FUSED_NORM: fused})
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('MKL_NUM_THREADS', '4')
    common = ['--model', args.model, '--dtype', args.dtype, '--gpu', args.gpu,
              '--max-model-len', str(args.max_model_len), '--output-dir', str(destination)]
    candidates = ['hf', 'flashinfer_eager', 'flashinfer_graph']
    if name == 'baseline':
        candidates.append('vllm')
    gate = [sys.executable, '-m', 'scripts.check_inference_correctness', *common,
            '--variants', *candidates]
    bench = [sys.executable, '-m', 'scripts.run_course_experiments', *common,
             '--suite', 'core', '--backends',
             *(['hf', 'vllm', 'flashinfer'] if name == 'baseline' else ['flashinfer']),
             '--prompt-lengths', *map(str, args.prompt_lengths),
             '--output-tokens', str(args.output_tokens),
             '--warmup-runs', str(args.warmup_runs), '--measure-runs', str(args.measure_runs)]
    return env, gate, bench


def run_variant(args, name, destination):
    destination.mkdir(parents=True)
    env, gate, bench = build_stages(args, name, destination)
    for label, command in [('correctness', gate), ('performance', bench)]:
        print(f'{name}: {label}', flush=True)
        with (destination / f'{label}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name}: {label} failed; inspect {destination / (label + ".log")}')
        if label == 'correctness':
            reports = list(destination.glob('correctness_*/report.json'))
            if len(reports) != 1 or json.loads(reports[0].read_text())['strict_pass'] is not True:
                raise RuntimeError(f'{name}: correctness evidence missing or failed')
    summaries = list(destination.glob('core_*/summary.json'))
    if len(summaries) != 1:
        raise RuntimeError(f'{name}: expected one core summary')
    rows = json.loads(summaries[0].read_text())
    expected = (4 if name == 'baseline' else 2) * len(args.prompt_lengths)
    if len(rows) != expected or any(r['status'] != 'ok' or r['samples'] != args.measure_runs for r in rows):
        raise RuntimeError(f'{name}: incomplete measurements')
    return rows


def write_comparison(destination, results):
    baseline = {r['name']: r for r in results.get('baseline', [])}
    lines = ['# Decode optimization ablation', '',
             'Same-run graph-to-graph comparison; speedup = baseline time / candidate time.',
             'All times include prefill and decode. Changes are experimental until measured.', '',
             '| Variant | Job | Actual input tokens | Mean s | Output token/s | vs baseline Graph |',
             '|---|---|---:|---:|---:|---:|']
    for variant, rows in results.items():
        for row in rows:
            if not row['name'].endswith('_graph_1'):
                continue
            ref = baseline.get(row['name'])
            if ref is None or any(ref[k] != row[k] for k in
                                  ('prompt_tokens_min', 'prompt_tokens_max', 'output_tokens', 'samples')):
                raise RuntimeError('Missing baseline or mismatched measured workload')
            speedup = ref['mean_total_s'] / row['mean_total_s']
            lines.append(f"| {variant} | {row['name']} | {row['prompt_tokens_min']} | "
                         f"{row['mean_total_s']:.5f} | {row['total_tokens_per_s']:.2f} | {speedup:.3f}x |")
    (destination / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--dtype', choices=['float16', 'bfloat16'], default='float16')
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--prompt-lengths', type=int, nargs='+', default=[512, 2048])
    parser.add_argument('--output-tokens', type=int, default=128)
    parser.add_argument('--warmup-runs', type=int, default=2)
    parser.add_argument('--measure-runs', type=int, default=10)
    parser.add_argument('--output-dir', default='course_results')
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.variants or args.variants[0] != 'baseline' or len(set(args.variants)) != len(args.variants):
        parser.error('Start with baseline and select each variant at most once')
    if min(args.prompt_lengths + [args.output_tokens, args.measure_runs, args.max_model_len]) <= 0 or args.warmup_runs < 0:
        parser.error('Lengths and measurement count must be positive; warmup must be non-negative')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    destination = Path(args.output_dir).resolve() / f'decode_optimization_{stamp}'
    if args.dry_run:
        for name in args.variants:
            env, gate, bench = build_stages(args, name, destination / name)
            print(json.dumps({'variant': name, 'environment': {k: env[k] for k in (HOST_PLAN, FUSED_NORM)},
                              'correctness': gate, 'performance': bench}, indent=2))
        return
    destination.mkdir(parents=True)
    manifest = source_manifest()
    (destination / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    with zipfile.ZipFile(destination / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as snapshot:
        for path in manifest:
            snapshot.write(ROOT / path, path)
    state = {'args': vars(args), 'completed': [], 'source_unchanged': True}
    results = {}
    try:
        for name in args.variants:
            if source_manifest() != manifest:
                raise RuntimeError('Source changed during ablation')
            results[name] = run_variant(args, name, destination / name)
            state['completed'].append(name)
        write_comparison(destination, results)
    except (RuntimeError, OSError, ValueError, KeyError, KeyboardInterrupt) as exc:
        state['error'] = str(exc) or 'Interrupted'
    finally:
        state['source_unchanged'] = source_manifest() == manifest
        (destination / 'pipeline.json').write_text(json.dumps(state, indent=2))
        print('Raw results:', destination, flush=True)
        print('Git-ready results:', archive(destination), flush=True)
    raise SystemExit(0 if 'error' not in state and state['source_unchanged'] else 1)


if __name__ == '__main__':
    main()
