"""Hand-computed replay contracts, not simulated performance evidence."""
import copy
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_depth_replay import (CSV_FIELDS, CSV_FIELDS_V2, CSV_FIELDS_V3, V3_EXTRA, Protocol, STATUSES, build_command,
                              parser, summarize_csv, validate_args, validate_configuration)


class ReplayTests(unittest.TestCase):
    def test_v4_output_checks_exclude_startup_from_steady_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary, rows = self.v2_fixture(directory)
            summary.update(schema="miniinfer-depth-replay-v4", preprocess_mode="cpu", sdk_telemetry_enabled=True,
                output_checks={"enabled": True, "startup_golden_calls": 2, "sampled": 2, "full": 2, "failures": 0})
            for row in rows:
                row.update({key: "0" if row["submit_ms"] else "" for key in V3_EXTRA})
            for phase in ("warmup", "steady"):
                summary[phase].update(h2d_bytes=0, rgb8_h2d_bytes=0, fp32_h2d_bytes=0)
            summary["sdk_snapshot"].update(accepted=4, succeeded=4, delivered=4)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS_V3)
                writer.writeheader()
                writer.writerows(rows)
            actual = summarize_csv(path, summary)["steady"]
            self.assertEqual(actual["accepted_succeeded"], 2)
            self.assertEqual(actual["successful_throughput_per_second"], 1)
            for field, value in (("enabled", 1), ("startup_golden_calls", 1), ("sampled", 1), ("full", 0), ("failures", 1)):
                changed = copy.deepcopy(summary)
                changed["output_checks"][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    summarize_csv(path, changed)
            summary["sdk_snapshot"]["accepted"] = 2
            with self.assertRaisesRegex(ValueError, "conservation"):
                summarize_csv(path, summary)

    def test_v3_separates_device_time_and_checks_transfer_conservation(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary, rows = self.v2_fixture(directory)
            summary.update(schema="miniinfer-depth-replay-v3", preprocess_mode="cuda-full-fused")
            for row in rows:
                row.update({key: "0" if row["submit_ms"] else "" for key in V3_EXTRA})
                if row["accepted"] == "1":
                    row.update(h2d_bytes="300", h2d_ms="0.1", gpu_preprocess_ms="0.2", model_ms="3", d2h_ms="0.4")
            summary["warmup"].update(h2d_bytes=0, rgb8_h2d_bytes=0, fp32_h2d_bytes=0)
            summary["steady"].update(h2d_bytes=600, rgb8_h2d_bytes=600, fp32_h2d_bytes=0)
            def write():
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS_V3)
                    writer.writeheader()
                    writer.writerows(rows)
            write()
            actual = summarize_csv(path, summary)["steady"]
            self.assertEqual(actual["successful_device_stages"]["model_ms"]["p99_ms"], 3)
            self.assertNotIn("model_ms", actual["successful_cpu_wall_stages"])
            self.assertEqual(actual["h2d_bytes"], 600)
            summary["steady"]["rgb8_h2d_bytes"] = 599
            with self.assertRaisesRegex(ValueError, "byte accounting"):
                summarize_csv(path, summary)
            summary["steady"]["rgb8_h2d_bytes"] = 600
            rejected = next(row for row in rows if row["status"] == "busy")
            rejected["model_ms"] = "1"
            write()
            with self.assertRaisesRegex(ValueError, "rejected request"):
                summarize_csv(path, summary)

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

    def v2_fixture(self, directory, asynchronous=True):
        path, summary = self.fixture(directory)
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["request_handle_id"] = row["request_id"] if asynchronous and row["accepted"] == "1" else ""
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS_V2)
            writer.writeheader()
            writer.writerows(rows)
        summary.update(schema="miniinfer-depth-replay-v2", async_sdk=asynchronous, slots=1,
                       reuse_host=True, reuse_gpu=True, consumer_delay_ms=0)
        summary["sdk_snapshot"].update(queued=0, executing=0, ready=0, abandoned=0, abandoned_completed=0)
        return path, summary, rows

    def write_v2_rows(self, path, rows):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS_V2)
            writer.writeheader()
            writer.writerows(rows)

    def test_v2_handle_ids_required_unique_and_only_for_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary, rows = self.v2_fixture(directory)
            self.assertEqual(summarize_csv(path, summary)["steady"]["accepted"], 2)
            accepted = [index for index, row in enumerate(rows) if row["accepted"] == "1"]
            denied = [index for index, row in enumerate(rows) if row["accepted"] != "1"]
            changes = [(accepted[0], ""), (accepted[0], "0"), (accepted[0], "-1"),
                       (accepted[0], str(2**64)), (accepted[1], rows[accepted[0]]["request_handle_id"]),
                       (denied[0], "999"), (denied[1], "998")]
            for index, handle in changes:
                changed = copy.deepcopy(rows)
                changed[index]["request_handle_id"] = handle
                self.write_v2_rows(path, changed)
                with self.subTest(index=index, handle=handle), self.assertRaises(ValueError):
                    summarize_csv(path, summary)
            # IDs are opaque uint64 values, not plan indices or bytearray offsets.
            rows[accepted[0]]["request_handle_id"] = str(2**64-1)
            self.write_v2_rows(path, rows)
            self.assertEqual(summarize_csv(path, summary)["steady"]["accepted"], 2)

    def test_v2_slow_consumer_uses_sdk_ready_for_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary, rows = self.v2_fixture(directory)
            first = next(row for row in rows if row["request_id"] == "1")
            first["delivered_ms"] = "1005" # Ready at 20ms; consumer delivers in drain.
            summary["steady"]["outstanding_at_window_end"] = 2
            summary["consumer_delay_ms"] = 985
            self.write_v2_rows(path, rows)
            result = summarize_csv(path, summary)["steady"]
            self.assertEqual(result["successful_throughput_per_second"], 1)
            self.assertEqual(result["sdk_successful_latency"]["p50_ms"], 133)
            self.assertEqual(result["delivery_successful_latency"]["p50_ms"], 493)
            self.assertEqual(result["sdk_unfinished_at_window_end"], 1)
            self.assertEqual(result["outstanding_at_window_end"], 2)
            self.assertEqual(result["undelivered_at_window_end"], 2)
            self.assertEqual(result["delivered_in_window"], 1)
            self.assertEqual(result["delivered_during_drain"], 2)
            summary["steady"]["outstanding_at_window_end"] = 1
            with self.assertRaises(ValueError):
                summarize_csv(path, summary)

    def test_v2_handle_id_cannot_be_reused_after_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary, rows = self.v2_fixture(directory)
            for row in rows:
                row["request_id"] = str(int(row["request_id"]) + 1)
            warmup = dict(zip(CSV_FIELDS_V2, ("warmup", "1", "0", "0", "0", "1", "2", "10", "11",
                                              "succeeded", "1", "1", "1", "1", "1", "1", "900")))
            rows.append(warmup)
            summary["warmup"].update(planned=1, submitted=1, duration_ms=250, completed_in_window=1,
                                     succeeded_in_window=1, peak_outstanding=1)
            summary["warmup"]["counts"]["succeeded"] = 1
            summary["sdk_snapshot"].update(accepted=3, succeeded=3, delivered=3)
            self.write_v2_rows(path, rows)
            self.assertEqual(summarize_csv(path, summary)["warmup"]["accepted"], 1)
            warmup["request_handle_id"] = next(row["request_handle_id"] for row in rows if row["phase"] == "steady" and row["accepted"] == "1")
            self.write_v2_rows(path, rows)
            with self.assertRaisesRegex(ValueError, "duplicate SDK"):
                summarize_csv(path, summary)

    def test_v2_final_execution_ready_and_abandonment_are_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary, _ = self.v2_fixture(directory)
            for key in ("queued", "executing", "ready", "abandoned", "abandoned_completed"):
                changed = copy.deepcopy(summary)
                changed["sdk_snapshot"][key] = 1
                with self.subTest(key=key), self.assertRaises(ValueError):
                    summarize_csv(path, changed)
                del changed["sdk_snapshot"][key]
                with self.subTest(missing=key), self.assertRaises(KeyError):
                    summarize_csv(path, changed)

    def test_v1_and_v2_sync_protocol_remain_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            path, summary = self.fixture(directory)
            original = summarize_csv(path, summary)
            path, updated, _ = self.v2_fixture(directory, asynchronous=False)
            self.assertEqual(original, summarize_csv(path, updated))
            for version in (summary, updated):
                protocol = Protocol()
                for event in ({"event": "ready", "startup_ms": 10},
                              {"event": "phase", "phase": "warmup"},
                              {"event": "phase", "phase": "steady"},
                              {"event": "phase", "phase": "drain", "source": "steady"},
                              {"event": "summary", **version}):
                    protocol.consume(json.dumps(event), .1)
                self.assertEqual(protocol.errors, [])
                self.assertEqual(protocol.summary["schema"], version["schema"])

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
            args.sequence = root / "sequence.txt"
            legacy_command = build_command(args, root / "repeat")
            for flag in ("--async-sdk", "--slots", "--reuse-host", "--reuse-gpu", "--consumer-delay-ms"):
                self.assertNotIn(flag, legacy_command)
            _, legacy_summary = self.fixture(directory)
            validate_configuration(args, legacy_summary)
            async_args = parser().parse_args(base + ["--async-sdk", "--reuse-host", "0", "--reuse-gpu", "0", "--consumer-delay-ms", "60000"])
            validate_args(async_args)
            async_args.sequence = args.sequence
            command = build_command(async_args, root / "repeat")
            self.assertEqual(command[command.index("--async-sdk") + 1], "1")
            self.assertEqual(command[command.index("--consumer-delay-ms") + 1], "60000")
            dual_args = parser().parse_args(base + ["--backend", "trt", "--async-sdk", "--slots", "2"])
            validate_args(dual_args)
            dual_args.sequence = args.sequence
            dual_command = build_command(dual_args, root / "repeat")
            self.assertEqual(dual_command[dual_command.index("--slots") + 1], "2")
            with self.assertRaises(ValueError):
                validate_configuration(async_args, legacy_summary)
            _, async_summary, _ = self.v2_fixture(directory)
            async_summary.update(reuse_host=False, reuse_gpu=False, consumer_delay_ms=60000)
            validate_configuration(async_args, async_summary)
            for key, value in (("async_sdk", False), ("slots", 2), ("reuse_host", True), ("consumer_delay_ms", 0)):
                changed = copy.deepcopy(async_summary)
                changed[key] = value
                with self.subTest(configuration=key), self.assertRaises(ValueError):
                    validate_configuration(async_args, changed)
            for flag, value in (("--slots", "2"), ("--reuse-host", "0"), ("--reuse-gpu", "0"), ("--consumer-delay-ms", "1")):
                with self.subTest(nonasync=flag), self.assertRaises(ValueError):
                    validate_args(parser().parse_args(base + [flag, value]))
            for value in ("-1", "60001"):
                with self.subTest(delay=value), self.assertRaises(ValueError):
                    validate_args(parser().parse_args(base + ["--async-sdk", "--consumer-delay-ms", value]))
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
