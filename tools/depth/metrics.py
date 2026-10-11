"""Backend-independent depth statistics; no normalization or alignment.

Numerical checks use abs(candidate-reference) > atol + rtol*abs(reference).
Empty numerical tensors are rejected: error summaries have no defined value.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import numpy as np

from .preprocess import DepthError


def _real_array(value, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise DepthError(f"{name} must be a real numeric array") from exc
    if array.dtype.kind not in "iuf":
        raise DepthError(f"{name} must be a real numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float64)
    if not np.isfinite(array).all():
        count = int(np.count_nonzero(~np.isfinite(array)))
        raise DepthError(f"{name} contains {count} non-finite values (NaN or infinity)")
    return array


def _tolerance(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise DepthError(f"{name} must be a finite non-negative real scalar")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise DepthError(f"{name} must be finite and non-negative") from exc
    if not np.isfinite(result) or result < 0:
        raise DepthError(f"{name} must be finite and non-negative")
    return result


def numerical_metrics(reference, candidate, atol, rtol) -> dict:
    """Compare equal-shape finite tensors in float64 without broadcasting.

    p99_abs uses np.quantile(abs_error, 0.99, method="linear").
    The tolerance bound includes equality; relative error uses the reference.
    nonfinite_count is zero on successful calls; non-finite input raises DepthError.
    """
    atol, rtol = _tolerance(atol, "atol"), _tolerance(rtol, "rtol")
    reference = _real_array(reference, "reference")
    candidate = _real_array(candidate, "candidate")
    if reference.shape != candidate.shape:
        raise DepthError(f"shape mismatch: reference {reference.shape}, candidate {candidate.shape}")
    if reference.size == 0:
        raise DepthError("numerical tensors must not be empty")
    with np.errstate(over="ignore", invalid="ignore"):
        error = np.abs(candidate - reference)
        bound = atol + rtol * np.abs(reference)
    if not np.isfinite(error).all() or not np.isfinite(bound).all():
        raise DepthError("numerical error or tolerance bound overflowed float64")
    max_abs = float(np.max(error))
    # Scale reductions to avoid overflow when squaring large finite errors.
    scaled = error / max_abs if max_abs else error
    violations = int(np.count_nonzero(error > bound))
    count = int(error.size)
    return {
        "count": count,
        "mae": float(np.mean(scaled, dtype=np.float64) * max_abs),
        "rmse": float(np.sqrt(np.mean(scaled * scaled, dtype=np.float64)) * max_abs),
        "max_abs": max_abs,
        "p99_abs": float(np.quantile(error, 0.99, method="linear")),
        "nonfinite_count": 0,
        "violations": violations,
        "violation_fraction": violations / count,
    }


def _coordinate(value, name: str, width: int, height: int) -> tuple[int, int]:
    if (not isinstance(value, (list, tuple, np.ndarray))
            or (isinstance(value, np.ndarray) and value.ndim != 1)
            or len(value) != 2):
        raise DepthError(f"{name} must be an integer [x,y] coordinate")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in value):
        raise DepthError(f"{name} must contain two integer coordinates")
    x, y = int(value[0]), int(value[1])
    if not (0 <= x < width and 0 <= y < height):
        raise DepthError(f"{name} coordinate [{x},{y}] outside width={width}, height={height}")
    return x, y


def ordering_metrics(depth_hw, pairs) -> dict:
    """Score original-image [x,y] pairs with the bigger-is-nearer rule.

    Every pair is counted independently, including duplicate annotations.
    Identical near/far pixels produce a tie, which fails. Uncertain coordinates
    are still checked, but uncertain pairs never affect total or ties.
    """
    depth = _real_array(depth_hw, "depth_hw")
    if depth.ndim != 2 or depth.size == 0:
        raise DepthError("depth_hw must be a non-empty HW array")
    if not isinstance(pairs, (list, tuple)):
        raise DepthError("pairs must be a list or tuple of annotations")
    height, width = depth.shape
    correct = total = ignored = ties = 0
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping):
            raise DepthError(f"pair {index} must be an annotation object")
        if not all(key in pair for key in ("near", "far", "status")):
            raise DepthError(f"pair {index} requires near, far and status")
        status = pair["status"]
        if not isinstance(status, str) or status not in ("valid", "uncertain"):
            raise DepthError(f"pair {index} status must be valid or uncertain")
        nx, ny = _coordinate(pair["near"], f"pair {index} near", width, height)
        fx, fy = _coordinate(pair["far"], f"pair {index} far", width, height)
        if status == "uncertain":
            ignored += 1
            continue
        total += 1
        near, far = depth[ny, nx], depth[fy, fx]
        correct += int(near > far)
        ties += int(near == far)
    return {"correct": correct, "total": total, "ignored": ignored, "ties": ties,
            "accuracy": correct / total if total else None}
