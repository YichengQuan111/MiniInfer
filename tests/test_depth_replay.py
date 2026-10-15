"""Hand-computed replay contracts, not simulated performance evidence."""
import copy
import contextlib
import csv
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_depth_replay import CSV_FIELDS, Protocol, STATUSES, parser, summarize_csv, validate_args


class ReplayTests(unittest.TestCase):
    def fixture(self, directory):
        path = Path(directory) / "raw.csv"
        # Four steady plans: two successes, one busy, one capacity drop. The
        # second success completes at the excluded right boundary (drain).
        rows = [
            ["steady", 1, 0, 0, 0, 1, 2, 20, 21, "succeeded", 1, 1, 2, 3, 4, 5],
            ["steady", 2, 1, 1, 250, 251, 252, 253, 254, "busy", 0, 1, 0, 0, 0, 0],
            ["steady", 3, 2, 0, 500, 501, "", "", "", "not_submitted_capacity", "", "", "", "", "", ""],
            ["steady", 4, 3, 1, 750, 751, 752, 1000, 1001, "succeeded", 1, 1, 2, 3, 4, 5],
        ]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(CSV_FIELDS)
            # Completion order is intentionally different from plan order.
            writer.writerows([rows[2], rows[0], rows[1], rows[3]])
        blank = {"planned": 0, "submitted": 0, "counts": dict.fromkeys(STATUSES, 0), "duration_ms": 0,
                 "drain_ms": 0, "completed_in_window": 0, "succeeded_in_window": 0,
                 "drain_completed": 0, "outstanding_at_window_end": 0, "peak_outstanding": 0}
        steady = copy.deepcopy(blank)
        steady.update(planned=4, submitted=3, duration_ms=1000, drain_ms=10,
                      completed_in_window=2, succeeded_in_window=1, drain_completed=1,
                      outstanding_at_window_end=1, peak_outstanding=2)
        steady["counts"].update(succeeded=2, busy=1, not_submitted_capacity=1)
        summary = {"schema": "miniinfer-depth-replay-v1", "rate": 4, "max_lag_ms": 100,
                   "image_count": 2, "warmup": blank, "steady": steady,
                   "sdk_snapshot": {"accepted": 2, "succeeded": 2, "failed": 0, "delivered": 2,
                                    "rejected_busy": 1, "rejected_closed": 0, "rejected_invalid": 0,
                                    "inflight": 0, "reserved_bytes": 0}}
        return path, summary

    def test_full_denominator_success_percentile_and_window_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary = self.fixture(directory)
            result = summarize_csv(path, summary)["steady"]
            self.assertEqual(result["planned"], 4)
            self.assertEqual(result["success_rate_of_planned"], .5)
            self.assertEqual(result["window_success_rate_of_planned"], .25)
            self.assertEqual(result["successful_throughput_per_second"], 1)
            self.assertEqual(result["drain_completed"], 1)
            self.assertEqual(result["sdk_successful_latency"]["p50_ms"], (18 + 248) / 2)
            self.assertAlmostEqual(result["replay_successful_latency"]["p99_ms"], 247.7)
            self.assertEqual(result["generator_lag"]["samples"], 4)
            self.assertEqual(result["successful_cpu_wall_stages"]["backend_ms"]["p99_ms"], 4)

    def test_summary_counts_snapshots_and_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary = self.fixture(directory)
            changes = [("steady", "submitted", 4), ("steady", "duration_ms", 1),
                       ("steady", "planned", 1000001), ("steady", "succeeded_in_window", 2),
                       ("steady", "outstanding_at_window_end", 0), ("sdk_snapshot", "accepted", 3),
                       ("sdk_snapshot", "inflight", 1), ("sdk_snapshot", "failed", 1)]
            for section, field, value in changes:
                changed = copy.deepcopy(summary)
                changed[section][field] = value
                with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                    summarize_csv(path, changed)

    def test_malformed_or_inconsistent_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary = self.fixture(directory)
            original = path.read_text()
            changes = [original.replace("phase,request_id", "request_id,phase"),
                       original.replace("steady,1,0,0", "steady,1,2,0"),
                       original.replace("steady,1,0,0", "steady,2,0,0"),
                       original.replace("steady,2,1,1", "steady,2,1,0"),
                       original.replace("0,1,2,20,21,succeeded", "0,1,2,nan,21,succeeded"),
                       original.replace("0,1,2,20,21,succeeded", "0,1,22,20,21,succeeded"),
                       original.replace("succeeded,1,1,2,3,4,5", "succeeded,0,1,2,3,4,5"),
                       original.replace("succeeded,1,1,2,3,4,5", "succeeded,1,1,2,3,-4,5"),
                       original.replace("not_submitted_capacity,,,,,,", "not_submitted_late,,,,,,"),
                       original.replace("not_submitted_capacity,,,,,,", "not_submitted_capacity,0,,,,,"),
                       original + original.splitlines()[2] + "\n"]
            for bad in changes:
                path.write_text(bad)
                with self.subTest(bad=bad[-100:]), self.assertRaises(ValueError):
                    summarize_csv(path, summary)

    def test_accepted_failure_and_generator_late_keep_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary = self.fixture(directory)
            raw = path.read_text().replace("752,1000,1001,succeeded", "752,1000,1001,failure")
            raw = raw.replace("500,501,,,,not_submitted_capacity", "500,701,,,,not_submitted_late")
            path.write_text(raw)
            summary["steady"]["counts"].update(succeeded=1, failure=1, not_submitted_capacity=0, not_submitted_late=1)
            summary["sdk_snapshot"].update(succeeded=1, failed=1)
            result = summarize_csv(path, summary)["steady"]
            self.assertEqual(result["accepted_failed"], 1)
            self.assertEqual(result["success_rate_of_planned"], .25)

    def test_all_rejections_and_empty_success_distribution(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary = self.fixture(directory)
            statuses = ("busy", "closed", "invalid_input", "failure", "not_submitted_capacity", "not_submitted_late")
            rows = []
            for index, status in enumerate(statuses):
                planned = 250 * index
                if status.startswith("not_submitted"):
                    generated = planned + (101 if status.endswith("late") else 1)
                    row = ["steady", index + 1, index, index % 2, planned, generated, "", "", "", status] + [""] * 6
                else:
                    row = ["steady", index + 1, index, index % 2, planned, planned + 1,
                           planned + 2, planned + 3, planned + 4, status, 0, 0, 0, 0, 0, 0]
                rows.append(row)
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(CSV_FIELDS)
                writer.writerows(rows)
            summary["steady"].update(planned=6, submitted=4, duration_ms=1500, completed_in_window=4,
                                     succeeded_in_window=0, drain_completed=0, outstanding_at_window_end=0)
            summary["steady"]["counts"] = {status: int(status in statuses) for status in STATUSES}
            summary["sdk_snapshot"].update(accepted=0, succeeded=0, delivered=0, rejected_closed=1, rejected_invalid=1)
            result = summarize_csv(path, summary)["steady"]
            self.assertEqual(result["planned"], 6)
            self.assertEqual(result["rejected"], 4)
            self.assertEqual(result["success_rate_of_planned"], 0)
            self.assertIsNone(result["sdk_successful_latency"]["p99_ms"])

    def test_warmup_success_does_not_enter_steady_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary = self.fixture(directory)
            with path.open(newline="") as stream:
                rows = list(csv.reader(stream))
            for row in rows[1:]:
                row[1] = str(int(row[1]) + 1)
            rows.append(["warmup", 1, 0, 0, 0, 1, 2, 10, 11, "succeeded", 1, 1, 1, 1, 1, 1])
            with path.open("w", newline="") as stream:
                csv.writer(stream).writerows(rows)
            summary["warmup"].update(planned=1, submitted=1, duration_ms=250, completed_in_window=1,
                                     succeeded_in_window=1, peak_outstanding=1)
            summary["warmup"]["counts"]["succeeded"] = 1
            summary["sdk_snapshot"].update(accepted=3, succeeded=3, delivered=3)
            result = summarize_csv(path, summary)
            self.assertEqual(result["warmup"]["successful_throughput_per_second"], 4)
            self.assertEqual(result["steady"]["successful_throughput_per_second"], 1)

    def test_protocol_rejects_bad_duplicate_missing_and_nonfinite(self):
        for line in ("not json", "[]", '{"event":"ready","startup_ms":NaN}',
                     '{"event":"ready","startup_ms":true}', '{"event":"summary"}'):
            protocol = Protocol()
            protocol.consume(line, .1)
            self.assertTrue(protocol.errors)
        protocol = Protocol()
        protocol.consume('{"event":"ready","startup_ms":10}', .1)
        protocol.consume('{"event":"ready","startup_ms":10}', .2)
        self.assertTrue(protocol.errors)
        for _ in range(100):
            protocol.consume("bad", .2)
        self.assertEqual(len(protocol.errors), 32)

    def test_cli_bounds_and_image_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "replay"
            executable.touch()
            package = root / "package"
            package.mkdir()
            image = root / "portrait.ppm"
            image.touch()
            listing = root / "images.txt"
            listing.write_text("# fixed order\nportrait.ppm\nportrait.ppm\n")
            base = ["--executable", str(executable), "--package", str(package), "--backend", "ort",
                    "--images-list", str(listing), "--output", str(root / "new"), "--rate", "4"]
            args = parser().parse_args(base)
            validate_args(args)
            self.assertEqual(args.images, [image, image])
            self.assertEqual(args.repeats, 3)
            self.assertEqual(args.sdk_workers, 4)
            for flag, value in (("--sdk-workers", "257"), ("--workers", "257"), ("--max-outstanding", "100001"),
                                ("--duration", "86401"), ("--duration", "86400"),
                                ("--repeats", "101"), ("--process-timeout-seconds", "10")):
                with self.subTest(flag=flag, value=value), self.assertRaises(ValueError):
                    validate_args(parser().parse_args(base + [flag, value]))
            for flag, value in (("--sdk-workers", "0"), ("--workers", "0"), ("--rate", "nan"), ("--rate", "inf"), ("--warmup", "-1")):
                with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser().parse_args(base + [flag, value])


if __name__ == "__main__":
    unittest.main()
