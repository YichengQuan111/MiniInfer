"""Small hand-computed benchmark statistics and format checks; no model stand-ins."""
import csv
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_depth_benchmark import CSV_FIELDS, STATUSES, distribution, percentile, phase_at, summarize_csv, summarize_memory


class DepthBenchmarkTests(unittest.TestCase):
    def fixture(self, directory):
        path = Path(directory) / "raw.csv"
        rows = [("first", 0, "succeeded", 80), ("warmup", 0, "failure", 10),
                ("steady", 0, "succeeded", 1), ("steady", 1, "busy", 100),
                ("steady", 2, "succeeded", 3), ("steady", 3, "closed", 5),
                ("steady", 4, "timeout", 7), ("steady", 5, "failure", 11)]
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(CSV_FIELDS)
            writer.writerows(row + ("", "", "") for row in rows)
        summary = {"startup_ms": 200, "connect_ms": 0, "first_ms": 80, "warmup": 1, "iterations": 6,
                   "elapsed_steady_ms": 200,
                   "counts": {"succeeded": 3, "busy": 1, "closed": 1, "timeout": 1, "failure": 2},
                   "steady_counts": {"succeeded": 2, "busy": 1, "closed": 1, "timeout": 1, "failure": 1}}
        return path, summary

    def test_linear_percentiles_and_empty(self):
        self.assertEqual(percentile([4, 1, 3, 2], 50), 2.5)
        self.assertAlmostEqual(percentile([4, 1, 3, 2], 95), 3.85)
        self.assertAlmostEqual(percentile([4, 1, 3, 2], 99), 3.97)
        self.assertEqual(distribution([7])["p99_ms"], 7)
        self.assertIsNone(distribution([])["p50_ms"])

    def test_success_only_and_whole_steady_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary = self.fixture(directory)
            result = summarize_csv(path, summary)
            self.assertEqual(sum(result["attempt_counts"].values()), 8)
            self.assertEqual(result["phase_counts"]["steady"], summary["steady_counts"])
            self.assertEqual(result["successful_steady_latency"]["samples"], 2)
            self.assertEqual(result["successful_steady_latency"]["p50_ms"], 2)
            self.assertAlmostEqual(result["successful_steady_latency"]["p95_ms"], 2.9)
            self.assertEqual(result["successful_throughput_per_second"], 10)
            self.assertIsNone(result["device_successful_steady"]["compute_ms"]["p50_ms"])

    def test_counts_first_and_elapsed_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary = self.fixture(directory)
            for key, value in (("first_ms", 70), ("iterations", 5), ("elapsed_steady_ms", 2), ("elapsed_steady_ms", float("nan"))):
                changed = copy.deepcopy(summary)
                changed[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    summarize_csv(path, changed)
            summary["counts"]["failure"] -= 1
            with self.assertRaises(ValueError):
                summarize_csv(path, summary)

    def test_malformed_csv_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary = self.fixture(directory)
            original = path.read_text()
            for bad in (original.replace("phase,index", "index,phase"),
                        original.replace("steady,0,succeeded", "steady,1,succeeded"),
                        original.replace("steady,0,succeeded,1", "steady,0,succeeded,nan"),
                        original.replace("steady,1,busy", "steady,1,skipped"),
                        original.replace("steady,1,busy,100,,,", "steady,1,busy,100,,,,")):
                path.write_text(bad)
                with self.assertRaises(ValueError):
                    summarize_csv(path, summary)

    def test_phase_boundaries_and_memory_scopes(self):
        boundaries = [(0, "startup"), (1, "first"), (2, "warmup"), (3, "steady")]
        self.assertEqual(phase_at(.9, boundaries), "startup")
        self.assertEqual(phase_at(3, boundaries), "steady")
        samples = [{"role": "client", "metric": "rss_kib", "phase": "startup", "value": 20},
                   {"role": "client", "metric": "rss_kib", "phase": "steady", "value": 30},
                   {"role": "server", "metric": "rss_kib", "phase": "steady", "value": 90},
                   {"role": "gpu_device_0_total", "metric": "used_mib", "phase": "startup", "value": None}]
        result = summarize_memory(samples)
        self.assertEqual(result["client:rss_kib"]["whole"]["peak"], 30)
        self.assertEqual(result["server:rss_kib"]["steady"]["peak"], 90)
        self.assertIsNone(result["gpu_device_0_total:used_mib"]["whole"]["peak"])
        self.assertIsNone(result["client:rss_kib"]["first"]["peak"])


if __name__ == "__main__":
    unittest.main()
