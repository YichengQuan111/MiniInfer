"""Unit tests for the D14 baseline driver's validation and orchestration."""

import json
import importlib.util
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER_SPEC = importlib.util.spec_from_file_location(
    "run_runtime_baseline", PROJECT_ROOT / "tools" / "run_runtime_baseline.py"
)
baseline = importlib.util.module_from_spec(DRIVER_SPEC)
DRIVER_SPEC.loader.exec_module(baseline)


def _expected(mode="runtime", workers=2, batch=8, iterations=5, warmup=0,
              window=8, queue_capacity=64):
    return {
        "mode": mode,
        "workers": workers,
        "batch": batch,
        "iterations": iterations,
        "warmup": warmup,
        "window": window,
        "queue_capacity": queue_capacity,
    }


def _valid_result(expected, *, attempts=None, accepted=None, rejected=0,
                  queue_full_rejected=0, closed_rejected=0, checksum=2.5):
    iterations = expected["iterations"]
    if attempts is None:
        attempts = iterations
    if accepted is None:
        accepted = attempts - rejected
    successes = accepted
    result = dict(expected)
    result.update({
        "attempts": attempts,
        "accepted": accepted,
        "successes": successes,
        "failures": 0,
        "rejected": rejected,
        "queue_full_rejected": queue_full_rejected,
        "closed_rejected": closed_rejected,
        "wall_ms": 1.0,
        "req_per_sec": 1000.0,
        "samples_per_sec": 8000.0,
        "mean_ms": 1.0,
        "p50_ms": 1.0,
        "p95_ms": 1.0,
        "p99_ms": 1.0,
        "checksum": checksum,
        "rss_after_warmup_bytes": 100,
        "rss_after_timing_bytes": 110,
        "peak_rss_bytes": 120,
        "max_inflight": 1 if expected["mode"] == "direct_cpu" else 2,
    })
    for group in baseline.TIMING_GROUPS:
        for percentile in baseline.TIMING_PERCENTILES:
            value = 0.0 if expected["mode"] == "direct_cpu" and group in {
                "queue", "collection"
            } else 0.25
            result[f"{group}_{percentile}"] = value
    return result


def _arg(command, name):
    return command[command.index(name) + 1]


class RuntimeBaselineValidationTests(unittest.TestCase):
    def test_checksum_comparison_allows_different_rejection_counts(self):
        first = _valid_result(_expected(), checksum=5.0)
        second = _valid_result(_expected(), accepted=3, rejected=2,
                               queue_full_rejected=2, checksum=3.0)
        runs = [{"batch": 8, "result": first}, {"batch": 8, "result": second}]
        baseline._check_batch_consistency(runs)
        second["checksum"] = 4.0
        with self.assertRaisesRegex(ValueError, "inconsistent checksums"):
            baseline._check_batch_consistency(runs)

    def test_accepts_rejections_without_assuming_all_iterations_were_accepted(self):
        expected = _expected(iterations=5)
        result = _valid_result(
            expected,
            attempts=5,
            accepted=3,
            rejected=2,
            queue_full_rejected=2,
        )
        self.assertIs(baseline.validate_result(result, expected), result)

    def test_rejects_invalid_json_from_mocked_cli(self):
        with mock.patch.object(
            baseline.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout="{bad json", stderr=""),
        ):
            with self.assertRaisesRegex(ValueError, "valid finite JSON"):
                baseline.execute_one(["benchmark_cli"], _expected())

    def test_rejects_non_finite_json_from_mocked_cli(self):
        with mock.patch.object(
            baseline.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout='{"x": NaN}', stderr=""),
        ):
            with self.assertRaisesRegex(ValueError, "valid finite JSON"):
                baseline.execute_one(["benchmark_cli"], _expected())

    def test_rejects_fake_counter_relationships_from_mocked_cli(self):
        expected = _expected()
        false_counts = _valid_result(expected)
        false_counts["attempts"] = 4
        output = json.dumps(false_counts)
        with mock.patch.object(
            baseline.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout=output, stderr=""),
        ):
            with self.assertRaisesRegex(ValueError, "attempts must equal"):
                baseline.execute_one(["benchmark_cli"], expected)

    def test_rejects_parameter_error_before_running_any_process(self):
        with mock.patch.object(baseline.subprocess, "run") as run:
            with self.assertRaises(SystemExit) as raised:
                baseline.main([
                    "--cli", "missing-cli", "--reference", "missing-reference",
                    "--output", "unused-output.json", "--iterations", "0",
                ])
        self.assertEqual(raised.exception.code, 2)
        run.assert_not_called()

    def test_preserves_an_existing_output_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "baseline.json"
            output.write_text("keep this report", encoding="utf-8")
            with mock.patch.object(baseline.subprocess, "run") as run:
                with self.assertRaises(SystemExit) as raised:
                    baseline.main([
                        "--cli", "missing-cli", "--reference", "missing-reference",
                        "--output", str(output),
                    ])
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep this report")
            run.assert_not_called()

    def test_default_run_plan_is_36_ordered_independent_processes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cli = root / "benchmark_cli"
            cli.write_bytes(b"mock binary")
            reference = root / "reference"
            reference.mkdir()
            (reference / "inputs.f32").write_bytes(bytes(range(256)) * 64)
            for filename in ("w1.f32", "b1.f32", "w2.f32", "b2.f32"):
                (reference / filename).write_bytes(filename.encode("ascii"))
            output = root / "report.json"

            def fake_process(command, **kwargs):
                batch = int(_arg(command, "--batch"))
                mode = _arg(command, "--mode")
                workers = 0 if mode == "direct_cpu" else int(_arg(command, "--workers"))
                expected = _expected(
                    mode=mode,
                    workers=workers,
                    batch=batch,
                    iterations=int(_arg(command, "--iterations")),
                    warmup=int(_arg(command, "--warmup")),
                    window=int(_arg(command, "--window")),
                    queue_capacity=int(_arg(command, "--queue-capacity")),
                )
                input_file = pathlib.Path(_arg(command, "--input"))
                self.assertEqual(input_file.stat().st_size, batch * 64 * 4)
                result = _valid_result(expected, checksum=float(batch))
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(result, allow_nan=False),
                    stderr="",
                )

            with mock.patch.object(baseline, "collect_environment", return_value={"stub": True}), \
                 mock.patch.object(baseline.subprocess, "run", side_effect=fake_process) as run:
                self.assertEqual(baseline.main([
                    "--cli", str(cli), "--reference", str(reference),
                    "--output", str(output),
                ]), 0)

            calls = run.call_args_list
            self.assertEqual(len(calls), 36)
            observed = [
                (_arg(call.args[0], "--batch"),
                 _arg(call.args[0], "--mode"),
                 _arg(call.args[0], "--workers"))
                for call in calls[:8]
            ]
            self.assertEqual(observed, [
                ("1", "direct_cpu", "1"),
                ("1", "runtime", "1"),
                ("1", "runtime", "2"),
                ("1", "runtime", "4"),
                ("1", "direct_cpu", "1"),
                ("1", "runtime", "1"),
                ("1", "runtime", "2"),
                ("1", "runtime", "4"),
            ])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(report["runs"]), 36)
            self.assertEqual(report["parameters"], {
                "iterations": 1000,
                "warmup": 200,
                "repeats": 3,
                "window": 8,
                "queue_capacity": 64,
            })


if __name__ == "__main__":
    unittest.main()
