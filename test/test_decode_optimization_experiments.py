from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_decode_optimization_experiments as runner


class DecodeOptimizationTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(model='/models/qwen', dtype='float16', gpu='0', max_model_len=8192,
                               prompt_lengths=[512], output_tokens=128, warmup_runs=2, measure_runs=10)

    def test_variants_override_inherited_switches_and_keep_reference(self):
        with patch.dict('os.environ', {runner.HOST_PLAN: '1', runner.FUSED_NORM: '1'}):
            env, gate, bench = runner.build_stages(self.args(), 'baseline', Path('/tmp/baseline'))
        self.assertEqual((env[runner.HOST_PLAN], env[runner.FUSED_NORM]), ('0', '0'))
        self.assertIn('vllm', gate)
        self.assertIn('vllm', bench)
        env, gate, bench = runner.build_stages(self.args(), 'combined', Path('/tmp/combined'))
        self.assertEqual((env[runner.HOST_PLAN], env[runner.FUSED_NORM]), ('1', '1'))
        self.assertIn('hf', gate)
        self.assertIn('flashinfer_graph', gate)
        self.assertNotIn('vllm', bench)

    def test_failed_gate_never_starts_performance(self):
        with TemporaryDirectory() as tmp, patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=1)) as run:
            with self.assertRaisesRegex(RuntimeError, 'correctness failed'):
                runner.run_variant(self.args(), 'host_plan', Path(tmp) / 'variant')
            self.assertEqual(run.call_count, 1)

    def test_zero_exit_without_correctness_evidence_is_rejected(self):
        with TemporaryDirectory() as tmp, patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            with self.assertRaisesRegex(RuntimeError, 'evidence missing'):
                runner.run_variant(self.args(), 'host_plan', Path(tmp) / 'variant')
            self.assertEqual(run.call_count, 1)

    def test_comparison_preserves_slowdown_and_rejects_different_input(self):
        base = dict(name='flashinfer_512_graph_1', prompt_tokens_min=529, prompt_tokens_max=529,
                    output_tokens=1280, samples=10, mean_total_s=1., total_tokens_per_s=128.)
        slower = dict(base, mean_total_s=2., total_tokens_per_s=64.)
        with TemporaryDirectory() as tmp:
            runner.write_comparison(Path(tmp), {'baseline': [base], 'host_plan': [slower]})
            self.assertIn('0.500x', (Path(tmp) / 'comparison.md').read_text())
            with self.assertRaisesRegex(RuntimeError, 'mismatched'):
                runner.write_comparison(Path(tmp), {'baseline': [base], 'host_plan': [dict(slower, prompt_tokens_max=530)]})


if __name__ == '__main__':
    unittest.main()
