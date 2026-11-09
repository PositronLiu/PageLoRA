"""CPU-only checks for the course launcher; run with unittest, no ML packages."""

import argparse
import unittest

from scripts.run_course_experiments import build_jobs


class CourseJobsTest(unittest.TestCase):
    def args(self, suite="core"):
        return argparse.Namespace(model="/models/model with spaces", suite=suite,
                                  prompt_lengths=[512, 2048], output_tokens=128,
                                  warmup_runs=2, measure_runs=5, max_model_len=8192)

    def test_core_has_matched_baselines_and_graph_ablation(self):
        jobs = build_jobs(self.args())
        self.assertEqual(len(jobs), 8)
        for _, command in jobs:
            self.assertEqual(command[command.index("--adapter-role") + 1], "base")
            self.assertEqual(command[command.index("--throughput-timing") + 1], "total")
            self.assertIn("/models/model with spaces", command)
        off, on = jobs[2][1], jobs[3][1]
        self.assertEqual(on, off + ["--flashinfer-cuda-graph-decode"])

    def test_prefix_has_both_cache_policies_for_both_backends(self):
        jobs = build_jobs(self.args("prefix"))
        self.assertEqual(len(jobs), 8)
        combinations = set()
        for _, command in jobs:
            combinations.add((command[command.index("--prefix-backend") + 1],
                              command[command.index("--prefix-reset-cache-between-runs") + 1]))
        self.assertEqual(combinations, {("flashinfer", "true"), ("flashinfer", "false"),
                                        ("vllm", "true"), ("vllm", "false")})


if __name__ == "__main__":
    unittest.main()
