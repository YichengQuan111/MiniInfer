"""Synthetic arithmetic/contract tests; these are not performance evidence."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import analyze_depth_stability as stability
from run_depth_replay import CSV_FIELDS_V3, STATUSES, V3_EXTRA, summarize_csv


class StabilityTests(unittest.TestCase):
    def fixture(self, directory, duration=121, latency=10, busy=False, rss_value=None):
        root = Path(directory)
        repeat = root / "repeat_01"
        repeat.mkdir()
        count = duration
        sampled = count - int(busy)
        final = {key: 0 for key in stability.SDK_FIELDS if key not in ("elapsed_ms", "phase")}
        final.update(accepted=sampled + 2, succeeded=sampled + 2, delivered=sampled + 2,
            peak_io_device_bytes=16 * stability.MIB, peak_io_pinned_bytes=16 * stability.MIB,
            device_allocation_calls=2, pinned_allocation_calls=2, host_workspace_allocation_calls=4,
            context_required_bytes=32 * stability.MIB, engine_serialized_bytes=stability.MIB,
            gpu_slots=1, shutdown_complete=1)
        snapshot = {key: value for key, value in final.items() if key not in ("shutdown_complete", "phase_boundary")}
        snapshot.update(fatal_backend=False, rejected_busy=int(busy), rejected_closed=0, rejected_invalid=0)
        blank = {"planned": 0, "submitted": 0, "counts": dict.fromkeys(STATUSES, 0), "duration_ms": 0,
            "drain_ms": 0, "completed_in_window": 0, "succeeded_in_window": 0, "drain_completed": 0,
            "outstanding_at_window_end": 0, "peak_outstanding": 0,
            "h2d_bytes": 0, "rgb8_h2d_bytes": 0, "fp32_h2d_bytes": 0}
        steady = dict(blank, planned=count, submitted=count, duration_ms=duration * 1000,
                      completed_in_window=count, succeeded_in_window=sampled, peak_outstanding=1,
                      h2d_bytes=sampled * 3, rgb8_h2d_bytes=sampled * 3,
                      counts=dict(blank["counts"], succeeded=sampled, busy=int(busy)))
        full = sum((index // 2) % 128 == 0 for index in range(count) if not (busy and index == 1))
        summary = {"schema": "miniinfer-depth-replay-v4", "async_sdk": True, "rate": 1, "image_count": 2,
            "max_lag_ms": 100, "preprocess_mode": "cuda-full-fused", "slots": 1,
            "reuse_host": True, "reuse_gpu": True, "sdk_telemetry_enabled": True,
            "output_checks": {"enabled": True, "startup_golden_calls": 2, "sampled": sampled,
                              "full": full, "failures": 0},
            "warmup": blank, "steady": steady, "sdk_snapshot": snapshot}
        with (repeat / "raw.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS_V3)
            writer.writeheader()
            for index in range(count):
                planned = index * 1000
                is_busy = busy and index == 1
                row = dict.fromkeys(CSV_FIELDS_V3, "0")
                row.update(phase="steady", request_id=index + 1, index=index, image_index=index % 2,
                    planned_ms=planned, generated_ms=planned + .1, submit_ms=planned + .2,
                    ready_ms=planned + (.3 if is_busy else latency + .2),
                    delivered_ms=planned + (.4 if is_busy else latency + .3),
                    status="busy" if is_busy else "succeeded", accepted=0 if is_busy else 1,
                    request_handle_id="" if is_busy else index + 3,
                    h2d_bytes=0 if is_busy else 3)
                for key in V3_EXTRA:
                    if key != "h2d_bytes":
                        row[key] = 0
                writer.writerow(row)
        with (repeat / "memory.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("elapsed_ms", "phase", "role", "metric", "value"))
            writer.writerow((0, "startup", "replay", "rss_kib", 128 * 1024))
            for elapsed in range(0, duration * 1000, 50):
                value = rss_value(elapsed) if rss_value else 128 * stability.MIB
                writer.writerow((1000 + elapsed, "steady", "replay", "rss_kib", value / 1024))
                if elapsed % 1000 == 0:
                    writer.writerow((1000 + elapsed, "steady", "replay", "cpu_percent_one_core", 200))
                    # Device-total is deliberately much larger than 2GiB. It
                    # must remain observational, never become application IO.
                    writer.writerow((1000 + elapsed, "steady", "gpu_device_0_total", "used_mib", 6000))
            writer.writerow((1000 + duration * 1000, "drain", "replay", "rss_kib", 128 * 1024))
        with (repeat / "sdk.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=stability.SDK_FIELDS)
            writer.writeheader()
            startup = dict(final, elapsed_ms=0, phase="startup", accepted=0, succeeded=0, delivered=0,
                           io_device_bytes=16 * stability.MIB, io_pinned_bytes=16 * stability.MIB,
                           host_workspace_bytes=8 * stability.MIB, shutdown_complete=0)
            writer.writerow(startup)
            for second in range(duration):
                completed = second - int(busy and second >= 2)
                writer.writerow(dict(startup, elapsed_ms=1000 + second * 1000, phase="steady",
                                     phase_boundary=int(second == 0),
                                     accepted=2 + completed, succeeded=2 + completed, delivered=2 + completed))
            writer.writerow(dict(final, elapsed_ms=1010 + duration * 1000, phase="teardown"))
        run = {"returncode": 0, "errors": [], "telemetry_errors": [], "process_summary": summary,
            "statistics": summarize_csv(repeat / "raw.csv", summary), "memory_csv": "memory.csv",
            "request_csv": "raw.csv", "sdk_csv": "sdk.csv", "phase_events": [
                {"phase": "warmup", "elapsed_ms": 990}, {"phase": "steady", "elapsed_ms": 1000},
                {"phase": "drain", "source": "steady", "elapsed_ms": 1000 + duration * 1000}]}
        report = {"schema": "miniinfer-depth-replay-report-v2", "all_runs_valid": True,
            "case": {"duration": duration, "budget": 64 * stability.MIB, "queue": 4, "sdk_workers": 4,
                     "slots": 1, "reuse_host": True, "reuse_gpu": True, "async_sdk": True, "repeats": 1},
            "runs": [run]}
        path = root / "report.json"
        self.save(path, report)
        return path, report

    @staticmethod
    def save(path, report):
        path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")

    @staticmethod
    def edit_sdk(path, action):
        csv_path = path.parent / "repeat_01/sdk.csv"
        with csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        action(rows)
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=stability.SDK_FIELDS)
            writer.writeheader(); writer.writerows(rows)

    def test_hand_computed_complete_windows_and_linear_trend(self):
        samples = [(float(t), (100 if t < 60000 else 132) * stability.MIB) for t in range(0, 120001, 50)]
        result = stability.rss_windows(samples)
        self.assertEqual(result["complete_window_count"], 2)
        self.assertEqual([item["samples"] for item in result["complete_windows"]], [1200, 1200])
        self.assertEqual([item["median_bytes"] for item in result["complete_windows"]], [100 * stability.MIB, 132 * stability.MIB])
        self.assertEqual(result["last_minus_first_median_bytes"], 32 * stability.MIB)
        self.assertEqual(result["highest_minus_first_median_bytes"], 32 * stability.MIB)
        self.assertEqual(result["trend_on_window_medians"]["slope_mib_per_minute"], 32)
        self.assertEqual(result["trend_on_window_medians"]["r_squared"], 1)
        self.assertEqual(result["partial_tail"]["samples"], 1)

    def test_exact_1800_seconds_accepts_last_rss_at_1799_95(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.fixture(directory, duration=1800)
            result = stability.analyze(path, require_soak=True)
            self.assertTrue(result["passed"], result)
            run = result["runs"][0]
            self.assertEqual(run["rss"]["observed_span_ms"], 1799950)
            self.assertEqual(run["rss"]["complete_window_count"], 29)
            self.assertEqual(run["output_checks"]["startup_golden_calls"], 2)
            self.assertEqual(run["output_checks"]["sampled"], 1800)
            self.assertEqual(run["sdk"]["final"]["accepted"], 1802)

    def test_normal_short_run_does_not_pretend_to_be_soak(self):
        with tempfile.TemporaryDirectory() as directory:
            path, report = self.fixture(directory)
            report["case"]["duration"] = 1800  # Configuration is not duration evidence.
            self.save(path, report)
            self.assertTrue(stability.analyze(path)["passed"])
            result = stability.analyze(path, require_soak=True)
            self.assertFalse(result["passed"])
            self.assertFalse(result["runs"][0]["soak_evidence"]["eligible"])

    def test_highest_window_growth_fails_even_if_last_returns_to_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            value = lambda t: (180 if 60000 <= t < 120000 else 100) * stability.MIB
            path, _ = self.fixture(directory, duration=181, rss_value=value)
            result = stability.analyze(path)
            self.assertFalse(result["passed"])
            run = result["runs"][0]
            self.assertEqual(run["rss"]["last_minus_first_median_bytes"], 0)
            self.assertEqual(run["rss"]["highest_minus_first_median_bytes"], 80 * stability.MIB)
            self.assertFalse(run["checks"]["rss_window_growth"])

    def test_last_60_second_tail_growth_fails_despite_stable_complete_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            value = lambda t: (208 if t >= 1740000 else 128) * stability.MIB
            path, _ = self.fixture(directory, duration=1800, rss_value=value)
            result = stability.analyze(path, require_soak=True)
            self.assertFalse(result["passed"])
            run = result["runs"][0]
            self.assertEqual(run["rss"]["complete_window_count"], 29)
            self.assertTrue(run["checks"]["rss_window_growth"])
            tail = run["rss"]["partial_tail"]
            self.assertEqual(tail["observed_span_ms"], 59950)
            self.assertEqual(tail["median_growth_from_first_complete_window_bytes"], 80 * stability.MIB)
            self.assertTrue(tail["used_for_growth_gate"])
            self.assertFalse(run["checks"]["rss_tail_growth"])

    def test_short_preflight_keeps_growth_unassessed_and_identity_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path, report = self.fixture(directory, duration=3)
            report["identity"] = {"package": "/frozen/model", "executable_sha256": "a" * 64,
                                  "images": [{"path": "/frozen/input", "sha256": "b" * 64}]}
            self.save(path, report)
            result = stability.analyze(path)
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["case"], report["case"])
            self.assertEqual(result["identity"], report["identity"])
            run = result["runs"][0]
            self.assertFalse(run["rss"]["growth_assessed"])
            self.assertFalse(run["rss"]["partial_tail"]["used_for_growth_gate"])
            self.assertIsNone(run["checks"]["rss_window_growth"])
            self.assertIsNone(run["checks"]["rss_tail_growth"])

    def test_rss_peak_is_a_gate_but_gpu_device_total_is_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.fixture(directory, duration=2, rss_value=lambda _: 2 * stability.GIB + 1024)
            result = stability.analyze(path)
            self.assertFalse(result["passed"])
            self.assertFalse(result["runs"][0]["checks"]["rss_peak_bound"])
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.fixture(directory, duration=2)
            result = stability.analyze(path)
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["runs"][0]["observations"]["gpu_device_0_total:used_mib:steady"]["peak"], 6000)

    def test_all_three_ledger_conservation_rules_are_checked(self):
        for field in ("inflight", "accepted", "succeeded"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path, _ = self.fixture(directory, duration=2)
                self.edit_sdk(path, lambda rows: rows[1].update({field: str(int(rows[1][field]) + 1)}))
                result = stability.analyze(path)
                self.assertFalse(result["passed"])
                self.assertIn("conservation", result["runs"][0]["errors"][0])

    def test_missing_sample_count_and_nonzero_output_failure_fail(self):
        for change in ("missing", "failure", "full"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                path, report = self.fixture(directory, duration=2)
                checks = report["runs"][0]["process_summary"]["output_checks"]
                if change == "missing":
                    del checks["sampled"]
                elif change == "failure":
                    checks["failures"] = 1
                else:
                    checks["full"] = 1  # Positive, but one of two periodic full checks is missing.
                self.save(path, report)
                self.assertFalse(stability.analyze(path)["passed"])

    def test_startup_golden_calls_are_not_replay_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path, report = self.fixture(directory, duration=2)
            report["runs"][0]["process_summary"]["sdk_snapshot"]["accepted"] = 2
            self.save(path, report)
            result = stability.analyze(path)
            self.assertFalse(result["passed"])
            self.assertIn("accepted conservation", result["runs"][0]["errors"][0])

    def test_reuse_allocation_growth_and_shutdown_resource_leak_fail(self):
        for change in ("growth", "leak"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                path, report = self.fixture(directory, duration=3)
                if change == "growth":
                    self.edit_sdk(path, lambda rows: [row.update(device_allocation_calls="3") for row in rows[2:]])
                    report["runs"][0]["process_summary"]["sdk_snapshot"]["device_allocation_calls"] = 3
                else:
                    self.edit_sdk(path, lambda rows: rows[-1].update(io_device_bytes="1"))
                    report["runs"][0]["process_summary"]["sdk_snapshot"]["io_device_bytes"] = 1
                self.save(path, report)
                result = stability.analyze(path)
                self.assertFalse(result["passed"])
                if change == "growth":
                    self.assertFalse(result["runs"][0]["checks"]["fixed_reuse_allocations"])
                else:
                    self.assertIn("shutdown retained", result["runs"][0]["errors"][0])

    def test_allocation_before_first_periodic_steady_sample_fails(self):
        for field in ("device_allocation_calls", "pinned_allocation_calls", "host_workspace_allocation_calls"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path, report = self.fixture(directory, duration=3)
                baseline = report["runs"][0]["process_summary"]["sdk_snapshot"][field]

                def grow_before_first_periodic(rows):
                    # Boundary is at 1000ms, first periodic sample at 2000ms.
                    # A hypothetical allocation at 1500ms must not disappear
                    # into the first periodic sample's baseline.
                    for row in rows:
                        if row["phase"] in ("steady", "teardown") and not int(row["phase_boundary"]):
                            row[field] = str(baseline + 1)

                self.edit_sdk(path, grow_before_first_periodic)
                report["runs"][0]["process_summary"]["sdk_snapshot"][field] = baseline + 1
                self.save(path, report)
                result = stability.analyze(path)
                self.assertFalse(result["passed"], result)
                run = result["runs"][0]
                self.assertFalse(run["checks"]["fixed_reuse_allocations"])
                self.assertEqual(run["sdk"]["allocation_counters"][field]["growth"], 1)

    def test_missing_late_or_duplicate_steady_boundary_fails(self):
        for change in ("missing", "late", "duplicate"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                path, _ = self.fixture(directory, duration=3)

                def alter_boundary(rows):
                    steady = [row for row in rows if row["phase"] == "steady"]
                    if change in ("missing", "late"):
                        steady[0]["phase_boundary"] = "0"
                    if change in ("late", "duplicate"):
                        steady[1]["phase_boundary"] = "1"

                self.edit_sdk(path, alter_boundary)
                result = stability.analyze(path)
                self.assertFalse(result["passed"], result)
                self.assertIn("synchronous steady SDK phase boundary", result["runs"][0]["errors"][0])

    def test_pre_steady_allocations_remain_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            path, report = self.fixture(directory, duration=3)
            # Startup-to-boundary growth may come from permitted warmup work;
            # only allocations after the synchronous boundary are forbidden.
            def warmup_growth(rows):
                for row in rows:
                    if row["phase"] in ("steady", "teardown"):
                        row["device_allocation_calls"] = "3"

            self.edit_sdk(path, warmup_growth)
            report["runs"][0]["process_summary"]["sdk_snapshot"]["device_allocation_calls"] = 3
            self.save(path, report)
            result = stability.analyze(path)
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["runs"][0]["sdk"]["allocation_counters"]["device_allocation_calls"]["growth"], 0)

    def test_quota_bound_checked_even_when_ledger_is_conserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.fixture(directory, duration=2)
            self.edit_sdk(path, lambda rows: rows[1].update(inflight="1", ready="1", accepted="3", succeeded="3",
                                                          reserved_bytes=str(64 * stability.MIB + 1)))
            result = stability.analyze(path)
            self.assertFalse(result["passed"])
            self.assertIn("byte budget", result["runs"][0]["errors"][0])

    def test_application_io_plus_context_requirement_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path, report = self.fixture(directory, duration=2)
            self.edit_sdk(path, lambda rows: [row.update(context_required_bytes=str(2 * stability.GIB)) for row in rows])
            report["runs"][0]["process_summary"]["sdk_snapshot"]["context_required_bytes"] = 2 * stability.GIB
            self.save(path, report)
            result = stability.analyze(path)
            self.assertFalse(result["passed"])
            self.assertFalse(result["runs"][0]["checks"]["application_io_plus_context_bound"])

    def test_stress_only_relaxes_latency_and_explicit_backpressure(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.fixture(directory, duration=3, latency=70, busy=True)
            self.assertFalse(stability.analyze(path)["passed"])
            self.assertTrue(stability.analyze(path, stress=True)["passed"])
            self.edit_sdk(path, lambda rows: rows[1].update(accepted="999"))
            self.assertFalse(stability.analyze(path, stress=True)["passed"])

    def test_missing_phase_boundaries_and_rss_holes_cannot_claim_soak(self):
        with tempfile.TemporaryDirectory() as directory:
            path, report = self.fixture(directory, duration=1800)
            del report["runs"][0]["phase_events"]
            self.save(path, report)
            result = stability.analyze(path, require_soak=True)
            self.assertFalse(result["passed"])
            self.assertFalse(result["runs"][0]["soak_evidence"]["checks"]["measured_phase_boundary_available"])
            memory = path.parent / "repeat_01/memory.csv"
            with memory.open(newline="") as stream:
                reader = csv.DictReader(stream)
                columns = reader.fieldnames
                rows = [row for row in reader if not (row["metric"] == "rss_kib" and 61000 <= float(row["elapsed_ms"]) < 63000)]
            with memory.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader(); writer.writerows(rows)
            self.assertFalse(stability.analyze(path, require_soak=True)["runs"][0]["checks"]["rss_sampling_density"])

    def test_run_errors_and_missing_statistics_fail(self):
        for change in ("errors", "statistics"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                path, report = self.fixture(directory, duration=2)
                if change == "errors":
                    report["runs"][0]["errors"] = ["reader failure"]
                else:
                    del report["runs"][0]["statistics"]
                self.save(path, report)
                self.assertFalse(stability.analyze(path)["passed"])

    def test_cli_writes_exclusively(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.fixture(directory, duration=2)
            output = Path(directory) / "audit.json"
            with patch.object(sys, "argv", ["analyze_depth_stability.py", str(path), "--output", str(output)]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(stability.main(), 0)
                saved = output.read_bytes()
                self.assertEqual(stability.main(), 1)
                self.assertEqual(output.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
