import argparse
import json
from pathlib import Path
import tempfile
import unittest

from scripts.check_inference_correctness import validate_coverage
from scripts.run_final_experiments import build_commands, source_manifest, write_report


class FinalExperimentTests(unittest.TestCase):
    def test_correctness_fails_closed(self):
        row = {'case': 'a', 'tokens': [1], 'first_logits': [1., 2.]}
        reports = {'hf': [row], 'flashinfer_eager': [dict(row, reuse_tokens=[1])]}
        self.assertEqual(validate_coverage(reports, list(reports), ['a'], 1), [])
        for bad in ([], [row, row], [dict(row, first_logits=None)], [dict(row, tokens=[])]):
            self.assertTrue(validate_coverage({'hf': bad}, ['hf'], ['a'], 1))
        self.assertTrue(validate_coverage(reports, ['hf', 'vllm'], ['a'], 1))

    def test_pipeline_orders_gate_before_benchmarks(self):
        args = argparse.Namespace(model='/models/test space', gpu='0', max_model_len=8192,
            prompt_lengths=[512], output_tokens=128, warmup_runs=2, measure_runs=10)
        commands = build_commands(args, Path('results'))
        self.assertIn('scripts.check_inference_correctness', commands[0])
        self.assertEqual([c[c.index('--suite') + 1] for c in commands[1:]], ['core', 'prefix'])
        self.assertTrue(all('/models/test space' in c for c in commands))

    def test_missing_or_failed_evidence_never_finalizes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertFalse(write_report(root))
            for folder in ('core_1', 'prefix_1', 'correctness_1'):
                (root / folder).mkdir()
            (root / 'correctness_1/report.json').write_text(json.dumps({'strict_pass': True}))
            for suite in ('core', 'prefix'):
                (root / f'{suite}_1/summary.json').write_text(json.dumps([{'name': suite, 'status': 'ok'}]))
            self.assertFalse(write_report(root))
            (root / 'pipeline.json').write_text(json.dumps({'completed_stages': 3, 'source_unchanged': True}))
            self.assertTrue(write_report(root))
            (root / 'prefix_1/summary.json').write_text(json.dumps([{'name': 'bad', 'status': 'incomplete'}]))
            self.assertFalse(write_report(root))

    def test_source_fingerprint_detects_uncommitted_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model').mkdir()
            path = root / 'model/example.py'
            path.write_text('before')
            before = source_manifest(root)
            path.write_text('after')
            self.assertNotEqual(before, source_manifest(root))


if __name__ == '__main__':
    unittest.main()
