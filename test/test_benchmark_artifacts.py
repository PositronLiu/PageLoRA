"""CPU-only regression coverage for report logic and correctness comparators."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts.benchmark_support import DeviceMemorySampler, parse_measurements, summarize
from scripts.check_inference_correctness import compare_sequences, compare_logits


class ArtifactTests(unittest.TestCase):
    def test_sequence_divergence_and_short_output(self):
        self.assertEqual(compare_sequences([1,2,3],[1,9,3])['first_difference'],1)
        self.assertFalse(compare_sequences([1,2],[1])['equal'])
        self.assertTrue(compare_sequences([1,2],[1,2])['equal'])

    def test_logits_reports_margin_tolerance_and_nonfinite(self):
        result=compare_logits([1.,2.,1.9],[1.01,2.01,1.91],.02,0)
        self.assertTrue(result['allclose'])
        self.assertAlmostEqual(result['reference_top1_top2_margin'],.1)
        self.assertFalse(compare_logits([1.,2.],[1.,float('nan')],1,1)['allclose'])
        self.assertFalse(compare_logits([1.],[1.,2.],1,1)['available'])

    def test_parser_excludes_warmup_and_understands_prefix(self):
        rows=parse_measurements('[hf_generate] prompt_tokens=33 prompt_chars=100\n'
            '[hf_generate] warmup#1 total_s=9 generated_tokens=32\n'
            '[hf_generate] measure#1 total_s=2 generated_tokens=32\n'
            '[vllm_prefix_branch] measure#2 request#3 prompt_tokens=42 latency_s=1 generated_tokens=16\n')
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['prompt_tokens'],33)
        self.assertEqual(rows[1]['request'],3)
        with self.assertRaises(ValueError):
            parse_measurements('[hf] measure#1 total_s=0 generated_tokens=32')

    def test_summary_weighted_throughput_speedup_and_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp); runs=[]
            for name,seconds,tokens,extra in [
                ('flashinfer_512_graph_0',[2,4],32,[]),
                ('flashinfer_512_graph_1',[1,2],32,['--flashinfer-cuda-graph-decode']),
                ('hf_512',[1],31,[])]:
                log=directory/f'{name}.log'
                log.write_text('[engine] prompt_tokens=540 prompt_chars=2000\n'+''.join(
                    f'[engine] measure#{i} total_s={s} generated_tokens={tokens}\n' for i,s in enumerate(seconds,1)))
                runs.append({'name':name,'returncode':0,'log':f'/remote/download/{log.name}',
                             'command':['python','--max-new-tokens','32','--measure-runs','2']+extra})
            (directory/'runs.json').write_text(json.dumps(runs))
            rows=summarize(directory)
            self.assertAlmostEqual(rows[0]['total_tokens_per_s'],64/6)
            self.assertEqual(rows[1]['graph_speedup_total'],2)
            self.assertEqual(rows[2]['status'],'incomplete')
            self.assertTrue((directory/'measurements.csv').exists())
            runs[1]['command']+=['--seed','99']
            (directory/'runs.json').write_text(json.dumps(runs))
            self.assertNotIn('graph_speedup_total',summarize(directory)[1])

    def test_sampler_records_errors_without_faking_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'memory.json'
            with patch('scripts.benchmark_support.subprocess.run',side_effect=FileNotFoundError('missing')):
                with DeviceMemorySampler('0',path,60):
                    pass
            report=json.loads(path.read_text())
            self.assertIsNone(report['sampled_peak_used_mib'])
            self.assertTrue(report['errors'])
            with patch('scripts.benchmark_support.subprocess.run',return_value=SimpleNamespace(stdout='1024, 24564, 50')):
                with DeviceMemorySampler('0',path,60):
                    pass
            self.assertEqual(json.loads(path.read_text())['sampled_peak_used_mib'],1024)


if __name__=='__main__':
    unittest.main()
