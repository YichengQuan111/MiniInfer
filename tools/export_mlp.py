#!/usr/bin/env python3
"""Train the fixed MiniInfer digits MLP and export its FP32 reference bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import sklearn
    from sklearn.datasets import load_digits
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from threadpoolctl import threadpool_limits
except ImportError as exc:  # pragma: no cover - depends on the user's Python environment
    raise SystemExit(
        "error: export dependencies are missing; install numpy, scikit-learn, and threadpoolctl"
    ) from exc


INPUT_FEATURES = 64
HIDDEN_FEATURES = 128
OUTPUT_FEATURES = 10
MAX_BATCH_SIZE = 256
SPLIT_TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42
THREAD_LIMIT = 1


class ExportError(Exception):
    """An expected, user-actionable export failure."""


def positive_int(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def random_seed(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0 or parsed > 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("must be between 0 and 4294967295")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export the fixed 64-128-10 digits MLP reference bundle."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="destination directory (must not exist or must be empty)",
    )
    parser.add_argument("--samples", type=positive_int, default=64)
    parser.add_argument("--seed", type=random_seed, default=42)
    parser.add_argument("--max-iter", type=positive_int, default=500)
    return parser.parse_args(argv)


def validate_output_dir(output_dir: Path) -> None:
    try:
        if output_dir.exists():
            if not output_dir.is_dir():
                raise ExportError(f"output path is not a directory: {output_dir}")
            if next(output_dir.iterdir(), None) is not None:
                raise ExportError(f"output directory is not empty; refusing to overwrite: {output_dir}")
    except OSError as exc:
        raise ExportError(f"cannot inspect output directory {output_dir}: {exc}") from exc


def array_bytes(array: np.ndarray, dtype: str) -> bytes:
    return np.ascontiguousarray(array, dtype=np.dtype(dtype)).tobytes(order="C")


def write_array(directory: Path, filename: str, array: np.ndarray, dtype: str) -> None:
    (directory / filename).write_bytes(array_bytes(array, dtype))


def read_array(directory: Path, filename: str, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    path = directory / filename
    raw = path.read_bytes()
    data_type = np.dtype(dtype)
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * data_type.itemsize
    if len(raw) != expected_bytes:
        raise ExportError(
            f"{filename} has {len(raw)} bytes; expected exactly {expected_bytes}"
        )
    return np.frombuffer(raw, dtype=data_type).reshape(shape, order="C").copy(order="C")


def fp32_forward(
    inputs: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """Run the exported network using FP32 NumPy operations only."""
    z1 = np.matmul(inputs, w1.T)
    z1 += b1
    np.maximum(z1, np.float32(0.0), out=z1)
    logits = np.matmul(z1, w2.T)
    logits += b2
    return np.ascontiguousarray(logits, dtype=np.float32)


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(predictions == targets, dtype=np.float64))


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ExportError(f"cannot determine installed version of {distribution}") from exc


def build_file_manifest(directory: Path, specs: dict[str, tuple[str, tuple[int, ...]]]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for filename in sorted(specs):
        dtype, shape = specs[filename]
        path = directory / filename
        raw = path.read_bytes()
        dtype_obj = np.dtype(dtype)
        elements = int(np.prod(shape, dtype=np.int64))
        expected_bytes = elements * dtype_obj.itemsize
        if len(raw) != expected_bytes:
            raise ExportError(
                f"{filename} has {len(raw)} bytes after writing; expected {expected_bytes}"
            )
        files[filename] = {
            "dtype": dtype,
            "shape": list(shape),
            "elements": elements,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return files


def commit_bundle(staging_dir: Path, output_dir: Path) -> None:
    """Publish the complete bundle as one directory rename on the Linux target."""
    validate_output_dir(output_dir)
    try:
        # Staging shares the destination's parent/filesystem. POSIX rename rejects
        # a non-empty destination and avoids publishing individual partial files.
        staging_dir.rename(output_dir)
    except OSError as exc:
        raise ExportError(f"cannot publish bundle to {output_dir}: {exc}") from exc


def export(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().absolute()

    # Check path constraints before loading data or doing any training.
    validate_output_dir(output_dir)

    digits = load_digits()
    labels = np.asarray(digits.target, dtype=np.int32)
    inputs = np.asarray(digits.data, dtype=np.float32) / np.float32(16.0)
    indices = np.arange(labels.size, dtype=np.int32)
    train_indices, holdout_indices = train_test_split(
        indices,
        test_size=SPLIT_TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=labels,
    )
    if args.samples > holdout_indices.size:
        raise ExportError(
            f"--samples ({args.samples}) exceeds the holdout size ({holdout_indices.size})"
        )

    reference_indices = np.asarray(holdout_indices[:args.samples], dtype=np.int32)
    train_inputs = inputs[train_indices]
    train_labels = labels[train_indices]
    holdout_inputs = inputs[holdout_indices]
    holdout_labels = labels[holdout_indices]
    reference_targets = labels[reference_indices]

    model_params = {
        "hidden_layer_sizes": (HIDDEN_FEATURES,),
        "activation": "relu",
        "solver": "adam",
        "batch_size": 64,
        "learning_rate_init": 0.001,
        "max_iter": args.max_iter,
        "tol": 1e-4,
        "n_iter_no_change": 20,
        "early_stopping": False,
        "shuffle": True,
        "random_state": args.seed,
        "alpha": 0.0001,
    }
    classifier = MLPClassifier(**model_params)
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always", ConvergenceWarning)
        with threadpool_limits(limits=THREAD_LIMIT):
            classifier.fit(train_inputs, train_labels)
    convergence_warnings = [str(record.message) for record in warning_records]

    # Create parents only after all CLI and dataset-size validation has passed.
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name or 'export'}-staging-", dir=output_dir.parent
        ) as staging_name:
            staging_dir = Path(staging_name)
            w1 = np.asarray(classifier.coefs_[0].T, dtype="<f4", order="C")
            b1 = np.asarray(classifier.intercepts_[0], dtype="<f4", order="C")
            w2 = np.asarray(classifier.coefs_[1].T, dtype="<f4", order="C")
            b2 = np.asarray(classifier.intercepts_[1], dtype="<f4", order="C")
            reference_inputs = np.asarray(inputs[reference_indices], dtype="<f4", order="C")
            targets = np.asarray(reference_targets, dtype="<i4", order="C")
            sample_indices = np.asarray(reference_indices, dtype="<i4", order="C")

            shapes: dict[str, tuple[str, tuple[int, ...]]] = {
                "w1.f32": ("<f4", (HIDDEN_FEATURES, INPUT_FEATURES)),
                "b1.f32": ("<f4", (HIDDEN_FEATURES,)),
                "w2.f32": ("<f4", (OUTPUT_FEATURES, HIDDEN_FEATURES)),
                "b2.f32": ("<f4", (OUTPUT_FEATURES,)),
                "inputs.f32": ("<f4", (args.samples, INPUT_FEATURES)),
                "logits.f32": ("<f4", (args.samples, OUTPUT_FEATURES)),
                "labels.i32": ("<i4", (args.samples,)),
                "targets.i32": ("<i4", (args.samples,)),
                "sample_indices.i32": ("<i4", (args.samples,)),
            }

            for filename, array in (
                ("w1.f32", w1),
                ("b1.f32", b1),
                ("w2.f32", w2),
                ("b2.f32", b2),
                ("inputs.f32", reference_inputs),
                ("targets.i32", targets),
                ("sample_indices.i32", sample_indices),
            ):
                write_array(staging_dir, filename, array, shapes[filename][0])

            # References are computed from the on-disk FP32 values that a C++ loader sees.
            loaded_w1 = read_array(staging_dir, "w1.f32", "<f4", shapes["w1.f32"][1])
            loaded_b1 = read_array(staging_dir, "b1.f32", "<f4", shapes["b1.f32"][1])
            loaded_w2 = read_array(staging_dir, "w2.f32", "<f4", shapes["w2.f32"][1])
            loaded_b2 = read_array(staging_dir, "b2.f32", "<f4", shapes["b2.f32"][1])
            loaded_reference_inputs = read_array(
                staging_dir, "inputs.f32", "<f4", shapes["inputs.f32"][1]
            )
            loaded_targets = read_array(
                staging_dir, "targets.i32", "<i4", shapes["targets.i32"][1]
            )
            with threadpool_limits(limits=THREAD_LIMIT):
                reference_logits = fp32_forward(
                    loaded_reference_inputs, loaded_w1, loaded_b1, loaded_w2, loaded_b2
                )
                holdout_logits = fp32_forward(
                    np.asarray(holdout_inputs, dtype=np.float32),
                    loaded_w1,
                    loaded_b1,
                    loaded_w2,
                    loaded_b2,
                )
                reference_predictions = np.argmax(reference_logits, axis=1).astype(np.int32)
                holdout_predictions = np.argmax(holdout_logits, axis=1).astype(np.int32)
                sklearn_reference_predictions = np.asarray(
                    classifier.predict(loaded_reference_inputs), dtype=np.int32
                )
                sklearn_holdout_predictions = np.asarray(
                    classifier.predict(holdout_inputs), dtype=np.int32
                )

            write_array(staging_dir, "logits.f32", reference_logits, "<f4")
            write_array(staging_dir, "labels.i32", reference_predictions, "<i4")

            # Confirm the emitted reference arrays round-trip exactly before creating the manifest.
            roundtrip_logits = read_array(
                staging_dir, "logits.f32", "<f4", shapes["logits.f32"][1]
            )
            roundtrip_predictions = read_array(
                staging_dir, "labels.i32", "<i4", shapes["labels.i32"][1]
            )
            if not np.array_equal(roundtrip_logits, reference_logits):
                raise ExportError("logits.f32 did not round-trip exactly")
            if not np.array_equal(roundtrip_predictions, np.argmax(roundtrip_logits, axis=1)):
                raise ExportError("labels.i32 does not match argmax(logits.f32)")
            if not np.array_equal(loaded_targets, reference_targets):
                raise ExportError("targets.i32 did not match the selected digits labels")

            agreement_count = int(
                np.count_nonzero(reference_predictions == sklearn_reference_predictions)
            )
            metrics = {
                "holdout_accuracy": accuracy(holdout_predictions, holdout_labels),
                "reference_accuracy": accuracy(reference_predictions, loaded_targets),
                "sklearn_holdout_accuracy": accuracy(
                    sklearn_holdout_predictions, holdout_labels
                ),
                "reference_matches_sklearn": bool(
                    np.array_equal(reference_predictions, sklearn_reference_predictions)
                ),
                "holdout_matches_sklearn": bool(
                    np.array_equal(holdout_predictions, sklearn_holdout_predictions)
                ),
                "reference_sklearn_match_count": agreement_count,
                "holdout_sklearn_match_count": int(
                    np.count_nonzero(holdout_predictions == sklearn_holdout_predictions)
                ),
            }

            files = build_file_manifest(staging_dir, shapes)
            manifest = {
                "format_version": 1,
                "model": {
                    "input_features": INPUT_FEATURES,
                    "hidden_features": HIDDEN_FEATURES,
                    "output_features": OUTPUT_FEATURES,
                    "dtype": "float32",
                    "byte_order": "little",
                    "weights_layout": "out_in",
                    "order": "C",
                    "activation": "relu",
                    "max_batch_size": MAX_BATCH_SIZE,
                },
                "preprocessing": {"divisor": 16.0},
                "reference": {
                    "sample_count": args.samples,
                    "selection": "first_held_out_indices",
                    "atol": 1e-4,
                    "rtol": 1e-4,
                    "argmax_tie_break": "lowest_index",
                },
                "files": files,
                "training": {
                    "seed": args.seed,
                    "split": {
                        "test_size": SPLIT_TEST_SIZE,
                        "random_state": SPLIT_RANDOM_STATE,
                        "stratify": True,
                        "train_samples": int(train_indices.size),
                        "test_samples": int(holdout_indices.size),
                    },
                    "parameters": json_value(classifier.get_params(deep=False)),
                    "n_iter": int(classifier.n_iter_),
                    "loss": float(classifier.loss_),
                    "convergence_warnings": convergence_warnings,
                },
                "environment": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scikit_learn": sklearn.__version__,
                    "scipy": package_version("scipy"),
                    "joblib": package_version("joblib"),
                    "threadpoolctl": package_version("threadpoolctl"),
                    "thread_limit": THREAD_LIMIT,
                },
                "metrics": metrics,
            }
            manifest_text = json.dumps(
                manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            ) + "\n"
            (staging_dir / "manifest.json").write_text(manifest_text, encoding="utf-8", newline="\n")

            commit_bundle(staging_dir, output_dir)
    except ExportError:
        raise
    except OSError as exc:
        raise ExportError(f"could not write export bundle to {output_dir}: {exc}") from exc

    return manifest


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        manifest = export(args)
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # Give a concise failure instead of a training traceback.
        print(f"error: export failed: {exc}", file=sys.stderr)
        return 1

    print(f"Exported MiniInfer MLP to {Path(args.output_dir).expanduser().absolute()}")
    print(
        "FP32 accuracy: "
        f"holdout={manifest['metrics']['holdout_accuracy']:.4f}, "
        f"reference={manifest['metrics']['reference_accuracy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
