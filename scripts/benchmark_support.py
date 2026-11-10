"""Standard-library benchmark artifacts; memory sampling stays outside timing workers."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class DeviceMemorySampler:
    """Whole physical GPU sampling, including load/warmup; not a process peak."""

    def __init__(self, gpu, path, interval=0.5):
        self.gpu, self.path, self.interval = gpu, Path(path), interval
        self.stop_event = threading.Event()
        self.samples = []
        self.errors = []
        self.thread = None

    def _sample(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '-i', self.gpu,
                 '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5, check=True)
            fields = result.stdout.strip().split(',')
            used, total, utilization = [float(value.strip()) for value in fields]
            self.samples.append({'unix_s': time.time(), 'used_mib': used,
                                 'total_mib': total, 'utilization_percent': utilization})
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.errors.append(str(exc))

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            self._sample()

    def __enter__(self):
        self._sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_event.set()
        self.thread.join()
        report = {'scope': 'whole_device_load_warmup_measurement', 'interval_s': self.interval,
                  'sample_count': len(self.samples), 'samples': self.samples, 'errors': self.errors,
                  'sampled_peak_used_mib': max((s['used_mib'] for s in self.samples), default=None),
                  'baseline_used_mib': self.samples[0]['used_mib'] if self.samples else None,
                  'warning': 'Includes other processes and preallocation; sampling can miss transient peaks.'}
        self.path.write_text(json.dumps(report, indent=2), encoding='utf-8')


MEASURE = re.compile(r'^\[([^]]+)\] measure#(\d+)(?: request#(\d+))? (.*)$')
PAIR = re.compile(r'([a-z_]+)=([^\s]+)')


def parse_measurements(text):
    rows = []
    prompt_tokens = None
    for line in text.splitlines():
        if 'prompt_tokens=' in line and 'prompt_chars=' in line:
            match = re.search(r'prompt_tokens=(\d+)', line)
            if match:
                prompt_tokens = int(match[1])
        match = MEASURE.match(line.strip())
        if not match:
            continue
        fields = dict(PAIR.findall(match[4]))
        elapsed = float(fields.get('total_s', fields.get('latency_s', 'nan')))
        tokens = int(fields['generated_tokens'])
        if not math.isfinite(elapsed) or elapsed <= 0 or tokens < 0:
            raise ValueError('Invalid measured latency/token count')
        rows.append({'backend': match[1], 'sequence': int(match[2]),
                     'request': int(match[3]) if match[3] else None,
                     'prompt_tokens': int(fields.get('prompt_tokens', prompt_tokens)) if fields.get('prompt_tokens', prompt_tokens) is not None else None,
                     'total_s': elapsed, 'generated_tokens': tokens})
    return rows


def summarize(directory):
    """Rebuild CSV/JSON/Markdown from retained logs, including failed jobs."""
    directory = Path(directory)
    runs = json.loads((directory / 'runs.json').read_text(encoding='utf-8'))
    summaries = []
    raw = []
    for run in runs:
        row = {'name': run['name'], 'returncode': run['returncode'], 'status': 'failed', 'error': ''}
        # Prefer portable paths so downloaded artifacts can be summarized locally.
        log = directory / Path(run['log']).name
        try:
            measurements = parse_measurements(log.read_text(encoding='utf-8', errors='replace'))
            if not measurements:
                raise ValueError('No measured requests found')
            latency = [m['total_s'] for m in measurements]
            command = run['command']
            expected = int(command[command.index('--max-new-tokens') + 1])
            repeats = int(command[command.index('--measure-runs') + 1])
            requests = int(command[command.index('--prefix-requests') + 1]) if '--prefix-requests' in command else 1
            complete = len(measurements) == repeats * requests
            expected_keys = {(sequence, request if '--prefix-requests' in command else None)
                             for sequence in range(1, repeats + 1) for request in range(1, requests + 1)}
            complete = complete and {(m['sequence'], m['request']) for m in measurements} == expected_keys
            fixed_length = all(m['generated_tokens'] == expected for m in measurements)
            row.update(status=('failed' if run['returncode'] != 0 else ('ok' if complete and fixed_length else 'incomplete')),
                       samples=len(latency), mean_total_s=statistics.mean(latency),
                       p50_total_s=percentile(latency, .5), p95_total_s=percentile(latency, .95),
                       output_tokens=sum(m['generated_tokens'] for m in measurements),
                       total_tokens_per_s=sum(m['generated_tokens'] for m in measurements) / sum(latency),
                       prompt_tokens_min=min((m['prompt_tokens'] for m in measurements if m['prompt_tokens'] is not None), default=None),
                       prompt_tokens_max=max((m['prompt_tokens'] for m in measurements if m['prompt_tokens'] is not None), default=None),
                       fixed_output_length=fixed_length, complete_samples=complete)
            for label, selected in [('first', [m['total_s'] for m in measurements if m['request'] == 1]),
                                    ('reuse', [m['total_s'] for m in measurements if m['request'] is not None and m['request'] > 1])]:
                if selected:
                    row[f'mean_{label}_request_s'] = statistics.mean(selected)
            raw.extend(dict(job=run['name'], **m) for m in measurements)
        except (OSError, ValueError, KeyError) as exc:
            row['error'] = str(exc)
        memory_path = directory / f"{run['name']}.memory.json"
        if memory_path.exists():
            memory = json.loads(memory_path.read_text(encoding='utf-8'))
            row['device_sampled_peak_mib'] = memory.get('sampled_peak_used_mib')
            row['memory_samples'] = memory.get('sample_count', 0)
        summaries.append(row)
    by_name = {r['name']: r for r in summaries}
    commands = {r['name']: r['command'] for r in runs}
    for row in summaries:
        if row['name'].startswith('flashinfer_') and row['name'].endswith('_graph_1'):
            off = by_name.get(row['name'][:-1] + '0', {})
            matched_commands = [arg for arg in commands[row['name']] if arg != '--flashinfer-cuda-graph-decode'] == commands.get(off.get('name'))
            if matched_commands and row.get('prompt_tokens_min') is not None and row['status'] == off.get('status') == 'ok' and all(
                row.get(k) == off.get(k) for k in ['prompt_tokens_min', 'prompt_tokens_max', 'output_tokens', 'samples']
            ):
                row['graph_speedup_total'] = off['mean_total_s'] / row['mean_total_s']
    (directory / 'summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    for name, records in [('summary.csv', summaries), ('measurements.csv', raw)]:
        if records:
            fields = list(dict.fromkeys(k for row in records for k in row))
            with (directory / name).open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
    lines = ['# Benchmark summary', '',
             'Throughput = total output tokens / total measured request time. p95 uses linear interpolation.',
             'Device memory is sampled across loading, warmup and measurement; it is not an exact process peak.',
             'Failed/incomplete rows must not be used to claim speedups. Timings inherit log rounding.', '',
             '| Job | Status | Mean s | p95 s | Output tok/s | Device sampled MiB | Graph speedup |',
             '|---|---|---:|---:|---:|---:|---:|']
    def fmt(value):
        return f'{value:.4f}' if isinstance(value, (int, float)) else '-'
    for r in summaries:
        lines.append(f"| {r['name']} | {r['status']} | " + ' | '.join(fmt(r.get(k)) for k in
            ['mean_total_s', 'p95_total_s', 'total_tokens_per_s', 'device_sampled_peak_mib', 'graph_speedup_total']) + ' |')
    (directory / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summaries
