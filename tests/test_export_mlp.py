"""Contract tests for the standalone digits MLP exporter."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORTER = PROJECT_ROOT / "tools" / "export_mlp.py"
MANIFEST_NAME = "manifest.json"

ARRAY_LAYOUT = {
    "w1.f32": ("<f4", (128, 64)),
    "b1.f32": ("<f4", (128,)),
    "w2.f32": ("<f4", (10, 128)),
    "b2.f32": ("<f4", (10,)),
    "inputs.f32": ("<f4", None),
    "logits.f32": ("<f4", None),
    "labels.i32": ("<i4", None),
    "targets.i32": ("<i4", None),
    "sample_indices.i32": ("<i4", None),
}


def _element_count(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


class ExportMlpContractTests(unittest.TestCase):
    maxDiff = None

    def _run_export(self, output_dir: Path, *, samples: int, seed: int, max_iter: int):
        self.assertTrue(EXPORTER.is_file(), f"Exporter not found: {EXPORTER}")
        command = [
            sys.executable,
            str(EXPORTER),
            "--output-dir",
            str(output_dir),
            "--samples",
            str(samples),
            "--seed",
            str(seed),
            "--max-iter",
            str(max_iter),
        ]
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )

    def _failure_detail(self, result: subprocess.CompletedProcess) -> str:
        return (result.stdout + "\n" + result.stderr)[-3000:]

    def _load_manifest(self, output_dir: Path) -> dict:
        manifest_path = output_dir / MANIFEST_NAME
        self.assertTrue(manifest_path.is_file(), f"Missing {manifest_path.name}")
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        self.assertIsInstance(manifest, dict)
        return manifest

    def _read_arrays(self, output_dir: Path, manifest: dict, samples: int) -> dict:
        import numpy as np

        self.assertEqual(manifest.get("format_version"), 1)
        file_entries = manifest.get("files")
        self.assertIsInstance(file_entries, dict, "manifest.files must map filenames to metadata")
        arrays = {}

        for filename, (dtype_name, fixed_shape) in ARRAY_LAYOUT.items():
            with self.subTest(file=filename):
                self.assertIn(filename, file_entries)
                metadata = file_entries[filename]
                self.assertIsInstance(metadata, dict)
                dtype = np.dtype(dtype_name)
                if fixed_shape is None:
                    shape = {
                        "inputs.f32": (samples, 64),
                        "logits.f32": (samples, 10),
                        "labels.i32": (samples,),
                        "targets.i32": (samples,),
                        "sample_indices.i32": (samples,),
                    }[filename]
                else:
                    shape = fixed_shape

                element_count = _element_count(shape)
                expected_bytes = element_count * dtype.itemsize
                self.assertEqual(metadata.get("dtype"), dtype_name)
                self.assertEqual(metadata.get("shape"), list(shape))
                self.assertEqual(metadata.get("elements"), element_count)
                self.assertEqual(metadata.get("bytes"), expected_bytes)

                data_path = output_dir / filename
                self.assertTrue(data_path.is_file(), f"Missing {filename}")
                raw = data_path.read_bytes()
                self.assertEqual(len(raw), expected_bytes, f"Wrong byte length for {filename}")
                self.assertRegex(metadata.get("sha256", ""), r"^[0-9a-f]{64}$")
                self.assertEqual(metadata["sha256"], hashlib.sha256(raw).hexdigest())

                # Parsing explicitly as little-endian also checks that the exported
                # values have the byte order promised by the manifest contract.
                parsed = np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
                self.assertEqual(parsed.dtype.str, dtype_name)
                arrays[filename] = parsed

        return arrays

    def _validate_manifest_and_arrays(
        self, output_dir: Path, *, samples: int, seed: int, max_iter: int
    ) -> dict:
        import numpy as np
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split

        manifest = self._load_manifest(output_dir)
        arrays = self._read_arrays(output_dir, manifest, samples)

        model = manifest.get("model")
        self.assertIsInstance(model, dict)
        self.assertEqual(model.get("input_features"), 64)
        self.assertEqual(model.get("hidden_features"), 128)
        self.assertEqual(model.get("output_features"), 10)
        self.assertEqual(model.get("dtype"), "float32")
        self.assertEqual(model.get("byte_order"), "little")
        self.assertEqual(model.get("weights_layout"), "out_in")
        self.assertEqual(model.get("order"), "C")
        self.assertEqual(model.get("activation"), "relu")
        self.assertEqual(model.get("max_batch_size"), 256)

        preprocessing = manifest.get("preprocessing")
        self.assertIsInstance(preprocessing, dict)
        self.assertEqual(preprocessing.get("divisor"), 16.0)

        reference = manifest.get("reference")
        self.assertIsInstance(reference, dict)
        self.assertEqual(reference.get("sample_count"), samples)
        self.assertEqual(reference.get("selection"), "first_held_out_indices")
        self.assertEqual(reference.get("atol"), 1e-4)
        self.assertEqual(reference.get("rtol"), 1e-4)
        self.assertEqual(reference.get("argmax_tie_break"), "lowest_index")

        training = manifest.get("training")
        self.assertIsInstance(training, dict)
        self.assertEqual(training.get("seed"), seed)
        split = training.get("split")
        self.assertIsInstance(split, dict)
        self.assertEqual(split.get("test_size"), 0.2)
        self.assertEqual(split.get("random_state"), 42)
        self.assertTrue(split.get("stratify"))
        self.assertEqual(split.get("train_samples"), 1437)
        self.assertEqual(split.get("test_samples"), 360)
        parameters = training.get("parameters")
        self.assertIsInstance(parameters, dict)
        self.assertEqual(parameters.get("max_iter"), max_iter)
        self.assertEqual(parameters.get("random_state"), seed)
        self.assertEqual(parameters.get("hidden_layer_sizes"), [128])
        self.assertEqual(parameters.get("activation"), "relu")
        self.assertEqual(parameters.get("solver"), "adam")
        self.assertGreater(training.get("n_iter", 0), 0)
        self.assertLessEqual(training["n_iter"], max_iter)
        self.assertIsInstance(training.get("convergence_warnings"), list)
        self.assertIsInstance(training.get("loss"), (int, float))
        self.assertTrue(math.isfinite(float(training["loss"])))

        environment = manifest.get("environment")
        self.assertIsInstance(environment, dict)
        for label, key in (
            ("Python", "python"),
            ("NumPy", "numpy"),
            ("scikit-learn", "scikit_learn"),
            ("SciPy", "scipy"),
            ("joblib", "joblib"),
            ("threadpoolctl", "threadpoolctl"),
        ):
            version = environment.get(key)
            self.assertIsInstance(version, str, f"Missing {label} version in environment")
            self.assertTrue(version.strip(), f"Empty {label} version in environment")
        self.assertEqual(environment.get("thread_limit"), 1)

        metrics = manifest.get("metrics")
        self.assertIsInstance(metrics, dict)
        for key in ("holdout_accuracy", "reference_accuracy", "sklearn_holdout_accuracy"):
            accuracy = metrics.get(key)
            self.assertIsInstance(accuracy, (int, float), f"Missing measured {key}")
            self.assertTrue(math.isfinite(float(accuracy)))
            self.assertGreaterEqual(float(accuracy), 0.0)
            self.assertLessEqual(float(accuracy), 1.0)
        self.assertAlmostEqual(
            float(metrics["holdout_accuracy"]),
            float(metrics["sklearn_holdout_accuracy"]),
            delta=1e-12,
        )
        self.assertIs(metrics.get("reference_matches_sklearn"), True)
        self.assertIs(metrics.get("holdout_matches_sklearn"), True)

        digits = load_digits()
        indices = arrays["sample_indices.i32"].astype(np.int64)
        self.assertEqual(indices.shape, (samples,))
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < len(digits.target)))
        self.assertEqual(len(np.unique(indices)), samples, "holdout samples must be unique")
        _, holdout_indices = train_test_split(
            np.arange(len(digits.target)),
            test_size=0.2,
            random_state=split["random_state"],
            stratify=digits.target,
        )
        np.testing.assert_array_equal(indices, holdout_indices[:samples])
        expected_inputs = (digits.data[indices] / 16.0).astype("<f4")
        np.testing.assert_array_equal(arrays["inputs.f32"], expected_inputs)
        np.testing.assert_array_equal(
            arrays["targets.i32"], digits.target[indices].astype("<i4")
        )
        measured_reference_accuracy = float(
            np.mean(arrays["labels.i32"] == arrays["targets.i32"])
        )
        self.assertAlmostEqual(
            float(metrics["reference_accuracy"]), measured_reference_accuracy, delta=1e-7
        )

        return arrays

    def _assert_scalar_forward_matches(self, arrays: dict) -> None:
        import numpy as np

        inputs = arrays["inputs.f32"]
        w1, b1 = arrays["w1.f32"], arrays["b1.f32"]
        w2, b2 = arrays["w2.f32"], arrays["b2.f32"]
        actual_logits = arrays["logits.f32"]
        actual_labels = arrays["labels.i32"]
        scalar_logits = np.empty_like(actual_logits)

        # Deliberately scalar and independent of the exporter's implementation:
        # each multiply and add rounds to float32, like a C++ float accumulator.
        for sample_index in range(inputs.shape[0]):
            hidden = np.empty(128, dtype=np.float32)
            for hidden_index in range(128):
                total = np.float32(b1[hidden_index])
                for feature_index in range(64):
                    product = np.float32(
                        w1[hidden_index, feature_index] * inputs[sample_index, feature_index]
                    )
                    total = np.float32(total + product)
                hidden[hidden_index] = np.float32(max(float(total), 0.0))

            for output_index in range(10):
                total = np.float32(b2[output_index])
                for hidden_index in range(128):
                    product = np.float32(w2[output_index, hidden_index] * hidden[hidden_index])
                    total = np.float32(total + product)
                scalar_logits[sample_index, output_index] = total

        tolerance = 1e-4 + 1e-4 * np.abs(actual_logits.astype(np.float64))
        differences = np.abs(
            actual_logits.astype(np.float64) - scalar_logits.astype(np.float64)
        )
        self.assertTrue(
            np.all(differences <= tolerance),
            f"scalar forward pass exceeded tolerance; max abs error={differences.max()}",
        )
        np.testing.assert_array_equal(
            actual_labels,
            np.argmax(scalar_logits, axis=1).astype("<i4"),
            err_msg="labels must be the lowest-index argmax of the reference logits",
        )

    def test_cli_rejects_invalid_parameters(self):
        invalid_cases = (
            ("zero samples", {"samples": 0}),
            ("negative samples", {"samples": -1}),
            ("samples beyond holdout", {"samples": 361}),
            ("negative seed", {"seed": -1}),
            ("seed beyond sklearn range", {"seed": 2**32}),
            ("zero max_iter", {"max_iter": 0}),
        )
        defaults = {"samples": 8, "seed": 42, "max_iter": 5}

        with tempfile.TemporaryDirectory(prefix="export-mlp-invalid-") as temp_dir:
            root = Path(temp_dir)
            for case_name, overrides in invalid_cases:
                with self.subTest(case=case_name):
                    parameters = {**defaults, **overrides}
                    output_dir = root / re.sub(r"[^a-z0-9]+", "-", case_name.lower())
                    result = self._run_export(
                        output_dir,
                        **parameters,
                    )
                    self.assertNotEqual(
                        result.returncode,
                        0,
                        f"invalid arguments unexpectedly succeeded: {case_name}\n"
                        f"{self._failure_detail(result)}",
                    )
                    self.assertFalse(output_dir.exists(), "invalid arguments created output")

    def test_nonempty_output_directory_is_preserved(self):
        with tempfile.TemporaryDirectory(prefix="export-mlp-nonempty-") as temp_dir:
            output_dir = Path(temp_dir) / "existing-output"
            output_dir.mkdir()
            sentinel = output_dir / "sentinel.bin"
            original = b"keep this existing output intact\x00\xff"
            sentinel.write_bytes(original)

            result = self._run_export(output_dir, samples=8, seed=42, max_iter=5)
            self.assertNotEqual(result.returncode, 0, self._failure_detail(result))
            self.assertEqual(sentinel.read_bytes(), original)
            self.assertEqual({path.name for path in output_dir.iterdir()}, {sentinel.name})

    def test_artifacts_are_deterministic_and_match_scalar_reference(self):
        samples, seed, max_iter = 12, 73, 7
        with tempfile.TemporaryDirectory(prefix="export-mlp-repeat-") as temp_dir:
            root = Path(temp_dir)
            first_dir = root / "first"
            second_dir = root / "second"
            second_dir.mkdir()  # Existing empty destinations must also be accepted.

            first = self._run_export(
                first_dir, samples=samples, seed=seed, max_iter=max_iter
            )
            self.assertEqual(first.returncode, 0, self._failure_detail(first))
            second = self._run_export(
                second_dir, samples=samples, seed=seed, max_iter=max_iter
            )
            self.assertEqual(second.returncode, 0, self._failure_detail(second))

            first_arrays = self._validate_manifest_and_arrays(
                first_dir, samples=samples, seed=seed, max_iter=max_iter
            )
            self._validate_manifest_and_arrays(
                second_dir, samples=samples, seed=seed, max_iter=max_iter
            )
            for filename in ARRAY_LAYOUT:
                with self.subTest(deterministic_file=filename):
                    self.assertEqual(
                        (first_dir / filename).read_bytes(),
                        (second_dir / filename).read_bytes(),
                    )
            self.assertEqual(
                (first_dir / MANIFEST_NAME).read_bytes(),
                (second_dir / MANIFEST_NAME).read_bytes(),
            )

            self._assert_scalar_forward_matches(first_arrays)

    def test_versioned_reference_bundle(self):
        arrays = self._validate_manifest_and_arrays(
            PROJECT_ROOT / "data" / "reference", samples=64, seed=42, max_iter=500
        )
        self._assert_scalar_forward_matches(arrays)


if __name__ == "__main__":
    unittest.main()
