"""Hand-calculated concurrency statistics and command contracts, without inference."""
import copy
import csv
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_depth_concurrency import (CSV_FIELDS, STATUSES, command_for, json_exclusive,
                                   metric_ranges, percentile, summarize_csv)


class ConcurrencyBenchmarkTests(unittest.TestCase):
    def fixture(self, directory):
        path = Path(directory) / "raw.csv"
        # Sum of request times (417ms) exceeds 200ms wall: requests overlap.
        rows = [(0, 0, "succeeded", 100, "123", ""),
                (1, 1, "succeeded", 150, "123", ""),
                (2, 0, "busy", 10, "", "queue full"),
                (3, 1, "closed", 7, "", "closed"),
                (4, 0, "failure", 20, "", 'a, "failure"'),
                (5, 1, "succeeded", 130, "123", "")]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(CSV_FIELDS)
            writer.writerows(rows)
        summary = {"iterations": 6, "concurrency": 2, "elapsed_steady_ms": 200,
                   "counts": {"succeeded": 3, "busy": 1, "closed": 1, "failure": 1},
                   "p50_ms": 130, "p99_ms": 149.6, "successful_qps": 15,
                   "baseline_checksum": "123", "ownership_ok": True,
                   "counters_ok": True, "drained": True,
                   "sdk_snapshot": {"inflight": 0, "queued": 0, "executing": 0, "reserved_bytes": 0}}
        return path, summary, rows

    def test_linear_percentiles(self):
        self.assertEqual(percentile([4, 1, 3, 2], 50), 2.5)
        self.assertAlmostEqual(percentile([4, 1, 3, 2], 99), 3.97)
        self.assertEqual(percentile([7], 99), 7)
        self.assertIsNone(percentile([], 50))

    def test_overlap_uses_whole_wall_and_success_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary, _ = self.fixture(directory)
            result = summarize_csv(path, summary)
            self.assertEqual(result["successful_qps"], 15)
            self.assertEqual(result["p50_ms"], 130)
            self.assertAlmostEqual(result["p99_ms"], 149.6)
            self.assertEqual(result["caller_attempt_counts"], [3, 3])
            self.assertEqual(result["counts"], summary["counts"])

    def test_no_success_retains_rejects_and_null_percentiles(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary, rows = self.fixture(directory)
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(CSV_FIELDS)
                writer.writerows((i, row[1], "busy", row[3], "", "busy") for i, row in enumerate(rows))
            summary.update(counts={status: 6 if status == "busy" else 0 for status in STATUSES},
                           p50_ms=None, p99_ms=None, successful_qps=0)
            result = summarize_csv(path, summary)
            self.assertIsNone(result["p99_ms"])
            self.assertEqual(result["counts"]["busy"], 6)

    def test_summary_inconsistencies_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary, _ = self.fixture(directory)
            for key, value in (("iterations", 5), ("concurrency", 1), ("p50_ms", 80),
                               ("successful_qps", 30), ("elapsed_steady_ms", 140),
                               ("elapsed_steady_ms", float("nan")), ("baseline_checksum", "999"),
                               ("ownership_ok", False), ("counters_ok", False), ("drained", False)):
                changed = copy.deepcopy(summary)
                changed[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    summarize_csv(path, changed)
            changed = copy.deepcopy(summary)
            changed["sdk_snapshot"]["reserved_bytes"] = 1
            with self.assertRaises(ValueError):
                summarize_csv(path, changed)

    def test_malformed_rows_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary, _ = self.fixture(directory)
            original = path.read_text()
            for bad in (original.replace("index,caller", "caller,index"),
                        original.replace("0,0,succeeded", "1,0,succeeded"),
                        original.replace("0,0,succeeded", "0,2,succeeded"),
                        original.replace("succeeded,100", "succeeded,nan"),
                        original.replace("2,0,busy", "2,0,retry"),
                        original.replace("0,0,succeeded,100,123,", "0,0,succeeded,100,123,,")):
                path.write_text(bad)
                with self.assertRaises(ValueError):
                    summarize_csv(path, summary)

    def test_repeats_report_ranges_not_combined_percentiles(self):
        runs = [{"statistics": {"p50_ms": p, "p99_ms": p * 2,
                                "successful_qps": 100 / p, "elapsed_steady_ms": 100}, "errors": []}
                for p in (1, 10, 3)]
        runs.append({"statistics": None, "errors": ["bad run"]})
        self.assertEqual(metric_ranges(runs)["p99_ms"],
                         {"repeats": 3, "min": 2, "median": 6, "max": 20})

    def test_caller_concurrency_does_not_change_worker_configuration(self):
        args = SimpleNamespace(executable=Path("benchmark"), backend="trt", package=Path("package"),
                               image=Path("input.ppm"), iterations=300, warmup=20,
                               worker_count=4, ort_threads=4, queue_capacity=4, byte_budget=536870912)
        for concurrency in (1, 4, 8):
            command = command_for(args, concurrency, Path("new_run"))
            options = dict(zip(command[1::2], command[2::2]))
            self.assertEqual(options["--concurrency"], str(concurrency))
            self.assertEqual(options["--worker-count"], "4")
            self.assertEqual(options["--ort-threads"], "4")
            self.assertEqual(options["--queue-capacity"], "4")
            self.assertEqual(options["--byte-budget"], "536870912")
            self.assertEqual(options["--iterations"], "300")

    def test_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            json_exclusive(path, {"original": True})
            with self.assertRaises(FileExistsError):
                json_exclusive(path, {"overwrite": True})
            self.assertIn('"original"', path.read_text())


if __name__ == "__main__":
    unittest.main()
