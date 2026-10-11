"""Synthetic temporary bundles and a fake CLI; never use the real final split."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from depth.preprocess import postprocess, ppm_bytes, preprocess
from depth.reference import CONTRACT, sha256, write_array, write_json
from depth.validate_backend import main, parse_args, validate


FAKE_CLI = '''
import argparse, json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, TOOLS_PATH)
from depth.preprocess import geometry, postprocess, preprocess, read_ppm
p = argparse.ArgumentParser()
for name in ("model", "image", "tensor", "width", "height", "output", "tag"):
    p.add_argument("--" + name)
a = p.parse_args()
mode = Path(a.model).read_text()
if mode == "fail":
    print("synthetic CLI failure", file=sys.stderr)
    sys.exit(7)
if a.image:
    tensor, g = preprocess(read_ppm(Path(a.image)))
else:
    tensor = np.fromfile(a.tensor, dtype="<f4").reshape(1,3,518,518)
    g = geometry(int(a.width), int(a.height))
raw = np.fromfile(Path(a.model).parent / "seed.f32", dtype="<f4").reshape(1,518,518)
if mode == "wrong":
    raw += np.float32(0.5)
if mode == "reverse":
    raw = np.float32(3) - raw
restored = postprocess(raw, g)
if mode == "nan":
    raw[0,0,0] = np.nan
if mode == "tensor_changed" and a.tensor:
    tensor.flat[0] += 1
if mode == "restore_wrong":
    restored += np.float32(0.1)
prefix = a.output
tensor.astype("<f4").tofile(prefix + ".preprocessed.f32")
raw.astype("<f4").tofile(prefix + ".raw.f32")
restored.astype("<f4").tofile(prefix + ".depth.f32")
if mode == "truncated":
    Path(prefix + ".raw.f32").write_bytes(b"abcd")
geo = g.to_dict()
if mode == "geometry":
    geo["pad_left"] += 1
Path(prefix + ".json").write_text(json.dumps({"geometry": geo, "seconds": {"inference": 0.01}}))
'''


class BackendValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "reference"
        self.bundle.mkdir()
        shutil.copyfile(CONTRACT, self.bundle / "contract.json")
        self.raw = np.broadcast_to(np.linspace(2, 1, 518, dtype=np.float32)[None, None, :],
                                   (1, 518, 518)).copy()
        self.raw.astype("<f4").tofile(self.root / "seed.f32")
        samples, saved = [], []
        for identifier, split in (("dev", "development"), ("holdout", "final")):
            directory = self.bundle / "samples" / identifier
            directory.mkdir(parents=True)
            rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
            original = self.root / (identifier + ".ppm")
            original.write_bytes(ppm_bytes(rgb))
            tensor, transform = preprocess(rgb)
            restored = postprocess(self.raw, transform)
            files = {"preprocessed.f32": write_array(directory / "preprocessed.f32", tensor, "<f4"),
                     "raw.f32": write_array(directory / "raw.f32", self.raw, "<f4"),
                     "restored.f32": write_array(directory / "restored.f32", restored, "<f4")}
            shutil.copyfile(original, directory / "original.ppm")
            files["original.ppm"] = {"sha256": sha256(directory / "original.ppm"),
                                     "bytes": original.stat().st_size}
            pairs = [{"near": [0, 0], "far": [1, 0], "status": "valid"}]
            samples.append({"id": identifier, "path": str(original), "sha256": sha256(original),
                            "split": split, "width": 2, "height": 1, "near_far_pairs": pairs})
            metadata_path = directory / "metadata.json"
            write_json(metadata_path, {"id": identifier, "split": split,
                "source_sha256": sha256(original), "geometry": transform.to_dict(),
                "near_far_pairs": pairs, "files": files})
            saved.append({"id": identifier, "metadata": f"samples/{identifier}/metadata.json",
                          "metadata_sha256": sha256(metadata_path)})
        self.samples = self.root / "samples.json"
        write_json(self.samples, {"schema_version": 1, "samples": samples})
        shutil.copyfile(self.samples, self.bundle / "samples.json")
        write_json(self.bundle / "manifest.json", {"format_version": 1, "contract_version": "depth-v1",
            "samples_manifest_sha256": sha256(self.samples), "samples_manifest_file": "samples.json",
            "contract_sha256": sha256(self.bundle / "contract.json"), "samples": saved})
        self.thresholds = self.root / "thresholds.json"
        write_json(self.thresholds, {"schema_version": 1, "backend": "synthetic",
            "atol": 0, "rtol": 0, "max_violation_fraction": 0, "preprocess_atol": 0,
            "postprocess_atol": 0, "max_ordering_drop": 0, "max_mae": 0, "max_p99_abs": 0})
        self.cli = self.root / "fake.py"
        self.cli.write_text(FAKE_CLI.replace("TOOLS_PATH", repr(str(PROJECT_ROOT / "tools"))), encoding="utf-8")
        self.model = self.root / "model.fake"
        self.model.write_text("good")
        self.args = argparse.Namespace(cli=str(self.cli), model=str(self.model), reference=str(self.bundle),
                                       samples=str(self.samples), split="development",
                                       thresholds=str(self.thresholds), output=str(self.root / "output"), extra_arg=[])

    def run_driver(self, mode="good"):
        self.model.write_text(mode)
        return validate(self.args)

    def test_passed_layers_hashes_split_and_commands(self):
        self.args.extra_arg = ["--tag", "sample-tag"]
        report = self.run_driver()
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual([s["id"] for s in report["samples"]], ["dev"])
        self.assertEqual(len(report["commands"]), 2)
        self.assertTrue(all("sample-tag" in r["command"] for r in report["commands"]))
        self.assertTrue(all("holdout" not in " ".join(r["command"]) for r in report["commands"]))
        self.assertEqual(len(report["samples"][0]["numerical"]), 8)
        self.assertEqual(report["aggregate"]["ordering"]["candidate"]["accuracy"], 1)
        pair_result = report["samples"][0]["pair_results"][0]
        self.assertTrue(pair_result["python"]["correct"])
        self.assertTrue(pair_result["candidate"]["correct"])
        for metric in report["aggregate"]["numerical"].values():
            self.assertEqual(metric["max_abs"], 0)
            self.assertTrue(metric["passed"])
        self.assertIn(str(self.bundle / "samples/holdout/raw.f32"), report["input_sha256"])
        self.assertEqual(json.loads((self.root / "output/report.json").read_text())["passed"], True)

    def test_numeric_regression_is_nonpassing_without_runtime_error(self):
        report = self.run_driver("wrong")
        self.assertFalse(report["passed"])
        self.assertEqual(report["errors"], [])
        metric = report["samples"][0]["numerical"]["raw"]
        self.assertAlmostEqual(metric["mae"], 0.5, places=6)
        self.assertEqual(metric["violation_fraction"], 1)

    def test_ordering_gate_with_permissive_numeric_limits(self):
        limits = json.loads(self.thresholds.read_text())
        limits.update(atol=2, max_mae=2, max_p99_abs=2)
        self.thresholds.write_text(json.dumps(limits))
        report = self.run_driver("reverse")
        self.assertFalse(report["passed"])
        self.assertEqual(report["aggregate"]["ordering_drop"], 1)
        self.assertFalse(report["samples"][0]["ordering_passed"])
        self.assertTrue(all(v["passed"] for v in report["aggregate"]["numerical"].values()))

    def test_postprocess_layer_catches_restore_error(self):
        report = self.run_driver("restore_wrong")
        self.assertFalse(report["passed"])
        self.assertTrue(report["samples"][0]["numerical"]["raw"]["passed"])
        self.assertFalse(report["samples"][0]["numerical"]["postprocessing"]["passed"])

    def test_cli_failure_record_and_nonzero_main(self):
        self.model.write_text("fail")
        argv = []
        for name in ("cli", "model", "reference", "samples", "split", "thresholds", "output"):
            argv.extend(["--" + name, getattr(self.args, name)])
        self.assertEqual(main(argv), 1)
        report = json.loads((self.root / "output/report.json").read_text())
        self.assertFalse(report["passed"])
        self.assertEqual(report["commands"][0]["returncode"], 7)
        self.assertIn("synthetic CLI failure", report["errors"][0]["message"])

    def test_invalid_cli_artifacts_never_pass(self):
        for mode in ("nan", "geometry", "truncated", "tensor_changed"):
            with self.subTest(mode=mode):
                self.args.output = str(self.root / ("output_" + mode))
                report = self.run_driver(mode)
                self.assertFalse(report["passed"])
                self.assertTrue(report["errors"])

    def test_corrupt_unselected_artifact_is_rejected_before_cli(self):
        path = self.bundle / "samples/holdout/raw.f32"
        with path.open("r+b") as stream:
            stream.write(b"bad!")
        report = self.run_driver()
        self.assertFalse(report["passed"])
        self.assertEqual(report["commands"], [])
        self.assertIn("SHA256 mismatch", report["errors"][0]["message"])

    def test_changed_samples_manifest_is_rejected(self):
        self.samples.write_text(self.samples.read_text() + "\n")
        report = self.run_driver()
        self.assertFalse(report["passed"])
        self.assertEqual(report["commands"], [])

    def test_invalid_thresholds_are_recorded(self):
        value = json.loads(self.thresholds.read_text())
        value["atol"] = -1
        self.thresholds.write_text(json.dumps(value))
        report = self.run_driver()
        self.assertFalse(report["passed"])
        self.assertIn("threshold atol", report["errors"][0]["message"])

    def test_existing_output_refuses_overwrite(self):
        destination = Path(self.args.output)
        destination.mkdir()
        sentinel = destination / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            validate(self.args)
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertFalse((destination / "report.json").exists())

    def test_reserved_extra_argument_rejected(self):
        self.args.extra_arg = ["--model=other.onnx"]
        report = self.run_driver()
        self.assertFalse(report["passed"])
        self.assertEqual(report["commands"], [])

    def test_parser_extra_arg_equals_form(self):
        argv = []
        for name in ("cli", "model", "reference", "samples", "split", "thresholds", "output"):
            argv.extend(["--" + name, getattr(self.args, name)])
        argv.extend(["--extra-arg=--tag", "--extra-arg=example"])
        self.assertEqual(parse_args(argv).extra_arg, ["--tag", "example"])


if __name__ == "__main__":
    unittest.main()
