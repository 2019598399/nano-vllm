import importlib.util
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "pd_compare", Path(__file__).parents[1] / "benchmarks" / "pd_compare.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BenchmarkStatsTest(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(MODULE.percentile([], 99), 0.0)
        self.assertEqual(MODULE.percentile([1], 99), 1.0)
        self.assertAlmostEqual(MODULE.percentile([1, 2, 3, 4], 50), 2.5)

    def test_request_summary_uses_only_selected_requests(self):
        records = {
            1: {"submitted": 1.0, "token_times": [1.1, 1.2, 1.4]},
            2: {"submitted": 1.0, "token_times": [10.0, 20.0]},
        }
        summary = MODULE.summarize_requests(records, [1])
        self.assertAlmostEqual(summary["mean_ttft_ms"], 100.0)
        self.assertAlmostEqual(summary["mean_itl_ms"], 150.0)
        self.assertEqual(summary["itl_samples"], 2)


if __name__ == "__main__":
    unittest.main()
