"""CPU-only package identity tests with synthetic depth outputs."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import subprocess

import test_depth_validation as fixtures

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from depth.reference import sha256
from depth.validate_backend import validate
from package_depth import create, verify
import verify_depth_sdk


class DepthPackageTests(unittest.TestCase):
    def fixture(self, size=518, backend="ort", precision="fp32"):
        fixture = (fixtures.BackendValidationTests if size == 518 else fixtures.BackendValidation392Tests)()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        identity = "ort-cpu-fp32" if backend == "ort" else "tensorrt-" + precision
        thresholds = json.loads(fixture.thresholds.read_text())
        thresholds["backend"] = identity
        fixture.thresholds.write_text(json.dumps(thresholds))
        fixture.args.split = "final"
        fixture.args.extra_arg = ["--backend-id", identity]
        report = validate(fixture.args)
        self.assertTrue(report["passed"], report["errors"])
        args = argparse.Namespace(model=str(fixture.model), onnx=str(fixture.model),
            output=str(fixture.root / "package"), reference=str(fixture.bundle),
            samples=str(fixture.samples), thresholds=str(fixture.thresholds),
            report=str(Path(fixture.args.output) / "report.json"), backend=backend,
            precision=precision, golden="holdout", model_size=size)
        return fixture, args

    def package(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            create(args)
            return verify(args.output)

    def test_both_sizes_bind_reference_and_keep_six_manifest_fields(self):
        for size in (518, 392):
            with self.subTest(size=size):
                fixture, args = self.fixture(size)
                metadata = self.package(args)
                self.assertEqual(metadata["model_size"], size)
                self.assertEqual(metadata["schema_version"], 2)
                self.assertEqual(len((Path(args.output) / "manifest.txt").read_text().splitlines()), 6)
                self.assertEqual(json.loads(Path(args.report).read_text())["reference_manifest_sha256"],
                                 sha256(fixture.bundle / "manifest.json"))

    def test_same_ids_with_changed_reference_manifest_are_rejected(self):
        fixture, args = self.fixture(392)
        path = fixture.bundle / "manifest.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "reference manifest changed"):
            create(args)

    def test_cross_size_report_and_missing_identity_are_rejected(self):
        for mutation in ({"model_size": 518}, {"contract": "depth-v1"},
                         {"reference_manifest_sha256": "0" * 64}, {"model_size": True}):
            with self.subTest(mutation=mutation):
                _, args = self.fixture(392)
                path = Path(args.report)
                report = json.loads(path.read_text())
                report.update(mutation)
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    create(args)

    def test_threshold_size_mismatch_is_rejected_before_cli(self):
        fixture, _ = self.fixture(392)
        fixture.args.output = str(fixture.root / "mismatch")
        threshold = json.loads(fixture.thresholds.read_text())
        threshold["model_size"] = 518
        fixture.thresholds.write_text(json.dumps(threshold))
        report = validate(fixture.args)
        self.assertFalse(report["passed"])
        self.assertEqual(report["commands"], [])

    def test_legacy_v1_package_inventory_verifies(self):
        _, args = self.fixture()
        self.package(args)
        root = Path(args.output)
        metadata = json.loads((root / "package.json").read_text())
        metadata["schema_version"] = 1
        del metadata["model_size"]
        report = json.loads((root / "validation.json").read_text())
        for field in ("model_size", "contract", "reference_manifest_sha256"):
            del report[field]
        (root / "validation.json").write_text(json.dumps(report))
        metadata["files"]["validation.json"] = {"bytes": (root / "validation.json").stat().st_size,
                                                    "sha256": sha256(root / "validation.json")}
        (root / "package.json").write_text(json.dumps(metadata))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            verify(root)
        self.assertIn("legacy_depth_v1_inventory", output.getvalue())

    def test_392_cannot_use_legacy_package_schema(self):
        _, args = self.fixture(392)
        self.package(args)
        path = Path(args.output) / "package.json"
        metadata = json.loads(path.read_text())
        metadata["schema_version"] = 1
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "schema/model_size"):
            verify(args.output)

    def test_engine_shape_identity_is_checked(self):
        fixture, args = self.fixture(392, "trt")
        engine = {"schema": "miniinfer-engine-1", "contract": "depth-v2-392", "precision": "fp32",
                  "engine_sha256": sha256(fixture.model), "onnx_sha256": sha256(fixture.model),
                  "input": "1x3x518x518-fp32", "output": "1x392x392-fp32"}
        Path(str(fixture.model) + ".manifest").write_text("".join(f"{k}={v}\n" for k, v in engine.items()))
        with self.assertRaisesRegex(ValueError, "engine metadata/hash mismatch"):
            create(args)

    def test_sdk_verifier_infers_392_and_rejects_size_override(self):
        fixture, args = self.fixture(392)
        self.package(args)
        root = Path(args.output)
        def run(command, **kwargs):
            Path(command[-1]).write_bytes((root / "golden/restored.f32").read_bytes())
            return subprocess.CompletedProcess(command, 0, "", "")
        output = fixture.root / "sdk-verify"
        argv = ["verify_depth_sdk", "--cli", str(fixture.cli), "--package", str(root), "--output", str(output)]
        with patch.object(sys, "argv", argv), patch.object(verify_depth_sdk.subprocess, "run", side_effect=run), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(verify_depth_sdk.main(), 0)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["model_size"], 392)
        self.assertEqual(report["contract"], "depth-v2-392")
        argv[-1] = str(fixture.root / "sdk-mismatch")
        with patch.object(sys, "argv", argv + ["--model-size", "518"]), \
                patch.object(verify_depth_sdk.subprocess, "run") as cli, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(verify_depth_sdk.main(), 1)
            cli.assert_not_called()


if __name__ == "__main__":
    unittest.main()
