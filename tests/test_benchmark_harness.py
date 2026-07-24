import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))

from harness.metrics import aggregate_runs, bootstrap_ci, summarize_records
from harness.service import ServiceClient
from harness.workload import synthetic_trace
from serve_compare import (
    compare_results,
    load_resumable,
    run_fingerprint,
    validate_suite,
)


class WorkloadTest(unittest.TestCase):
    def test_synthetic_trace_is_deterministic(self):
        first = synthetic_trace("mixed_chat", 10, 4.0, 7, 1000)
        second = synthetic_trace("mixed_chat", 10, 4.0, 7, 1000)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual([request.request_id for request in first.requests], list(range(10)))

    def test_interference_trace_has_foreground_and_prefill_only_background(self):
        trace = synthetic_trace("prefill_interference", 8, 4.0, 7, 1000)
        foreground = [request for request in trace.requests if request.group == "foreground"]
        background = [request for request in trace.requests if request.group == "background"]
        self.assertEqual(len(foreground), 4)
        self.assertTrue(all(request.output_tokens == 256 for request in foreground))
        self.assertTrue(all(
            len(request.prompt_token_ids) == 1024 and request.output_tokens == 1
            for request in background
        ))


class MetricsTest(unittest.TestCase):
    def test_summary_and_slo_goodput(self):
        records = {
            0: {"arrival_ns": 0, "token_times_ns": [10_000_000, 12_000_000, 14_000_000], "input_tokens": 4},
            1: {"arrival_ns": 1_000_000, "token_times_ns": [11_000_000, 13_000_000, 15_000_000], "input_tokens": 4},
        }
        summary = summarize_records(records, gpu_count=2, ttft_slo_ms=20, tpot_slo_ms=3)
        self.assertEqual(summary["slo_attainment"], 1.0)
        self.assertGreater(summary["goodput_request_s"], 0)
        self.assertEqual(summary["per_gpu_goodput_request_s"], summary["goodput_request_s"] / 2)

    def test_bootstrap_and_run_aggregation(self):
        self.assertEqual(bootstrap_ci([3.0]), [3.0, 3.0])
        runs = []
        for value in (2.0, 3.0, 4.0):
            summary = {
                "itl_ms": {"p99": value}, "ttft_ms": {"p95": value},
                "tpot_ms": {"p95": value}, "e2e_ms": {"p95": value},
                "output_tok_s": value, "request_s": value,
                "per_gpu_request_s": value, "slo_attainment": 1.0,
                "per_gpu_goodput_request_s": value,
            }
            runs.append({"summary": summary})
        aggregate = aggregate_runs(runs)
        self.assertEqual(aggregate["p99_itl_ms"]["median"], 3.0)
        self.assertEqual(aggregate["p99_itl_ms"]["range"], [2.0, 4.0])


class PipelineSafetyTest(unittest.TestCase):
    def test_worker_startup_timeout_terminates_process(self):
        process = mock.Mock(pid=42, exitcode=None)
        process.is_alive.return_value = True
        connection = mock.Mock()
        connection.poll.return_value = False
        context = mock.Mock()
        context.Pipe.return_value = (connection, mock.Mock())
        context.Process.return_value = process
        with mock.patch("harness.service.mp.get_context", return_value=context):
            with self.assertRaisesRegex(TimeoutError, "local-tp1 worker pid=42"):
                ServiceClient("local-tp1", "0", {}, startup_timeout_s=0.01)
        process.terminate.assert_called_once()
        process.join.assert_called_once_with(timeout=5)

    def test_worker_ready_path_records_runtime(self):
        process = mock.Mock(pid=42, exitcode=None)
        process.is_alive.return_value = True
        connection = mock.Mock()
        connection.poll.return_value = True
        connection.recv.return_value = ("ready", {"stats": {"graph_replays": 0}, "worker_seed": 7})
        context = mock.Mock()
        context.Pipe.return_value = (connection, mock.Mock())
        context.Process.return_value = process
        with mock.patch("harness.service.mp.get_context", return_value=context):
            client = ServiceClient("local-tp1", "0", {}, startup_timeout_s=0.01)
        self.assertEqual(client.runtime["worker_seed"], 7)
        self.assertEqual(client.last_stats, {"graph_replays": 0})

    def test_suite_validation_rejects_unknown_profile(self):
        suite = {
            "smoke": {
                "profiles": ["unknown"], "requests": 1,
                "factors": [0.5], "paired": 1,
            }
        }
        with self.assertRaisesRegex(ValueError, "unknown workload profiles"):
            validate_suite(suite, "smoke")

    def test_resume_rejects_legacy_or_mismatched_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps({"benchmark_fingerprint": "old"}))
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                load_resumable(path, "new")

    def test_run_fingerprint_is_stable_and_sensitive_to_config(self):
        trace = synthetic_trace("mixed_chat", 2, 4.0, 7, 1000)
        context = {"experiment": "baseline", "profile": "mixed_chat"}
        with mock.patch("serve_compare.source_hash", return_value="source"):
            first = run_fingerprint(trace, "pd", {"seed": 1}, context)
            second = run_fingerprint(trace, "pd", {"seed": 1}, context)
            changed = run_fingerprint(trace, "pd", {"seed": 2}, context)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)


class RegressionGateTest(unittest.TestCase):
    @staticmethod
    def _result(output_tok_s, p99_itl_ms):
        runs = []
        for repeat in range(5):
            summary = {
                "itl_ms": {"p99": p99_itl_ms}, "ttft_ms": {"p95": 10.0},
                "tpot_ms": {"p95": 2.0}, "e2e_ms": {"p95": 20.0},
                "output_tok_s": output_tok_s, "request_s": 5.0,
                "per_gpu_request_s": 2.5, "slo_attainment": 1.0,
                "per_gpu_goodput_request_s": 2.5,
            }
            runs.append({"experiment": "baseline", "profile": "mixed_chat", "rate_factor": 0.75,
                         "mode": "pd", "repeat": repeat, "summary": summary})
        return {"runs": runs, "groups": {"baseline|mixed_chat|0.75|pd": aggregate_runs(runs)}}

    def test_paired_regression_gate(self):
        baseline = self._result(100.0, 10.0)
        current = self._result(90.0, 12.0)
        with contextlib.redirect_stdout(io.StringIO()):
            regressions = compare_results(current, baseline)
        self.assertEqual({item["metric"] for item in regressions}, {"output_tok_s", "p99_itl_ms"})
        self.assertTrue(all(item["paired_repeats"] == 5 for item in regressions))

    def test_single_pair_is_not_a_statistical_regression(self):
        baseline = self._result(100.0, 10.0)
        current = self._result(50.0, 20.0)
        baseline["runs"] = baseline["runs"][:1]
        current["runs"] = current["runs"][:1]
        with contextlib.redirect_stdout(io.StringIO()):
            regressions = compare_results(current, baseline)
        self.assertEqual(regressions, [])


if __name__ == "__main__":
    unittest.main()
