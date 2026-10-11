"""Hand-computed metric cases, independent of model outputs or tolerances."""

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from depth.metrics import numerical_metrics, ordering_metrics
from depth.preprocess import DepthError


def pair(near, far, status="valid"):
    return {"near": near, "far": far, "status": status}


class NumericalMetricsTests(unittest.TestCase):
    def test_hand_computed_errors_and_linear_quantile(self):
        result = numerical_metrics(np.zeros((2, 2)), [[0, 1], [2, 3]], 1, 0)
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["mae"], 1.5)
        self.assertAlmostEqual(result["rmse"], np.sqrt(3.5))
        self.assertEqual(result["max_abs"], 3)
        self.assertAlmostEqual(result["p99_abs"], 2.97)
        self.assertEqual(result["nonfinite_count"], 0)
        self.assertEqual(result["violations"], 2)
        self.assertEqual(result["violation_fraction"], 0.5)

    def test_exact_tolerance_boundary_and_reference_direction(self):
        result = numerical_metrics([0, 2, -2, 2], [0.5, 3, -3, 3.125], 0.5, 0.25)
        self.assertEqual(result["violations"], 1)
        self.assertEqual(result["violation_fraction"], 0.25)
        self.assertEqual(numerical_metrics([2], [3], 0, 0.375)["violations"], 1)
        self.assertEqual(numerical_metrics([3], [2], 0, 0.375)["violations"], 0)

    def test_no_affine_alignment(self):
        result = numerical_metrics([1, 2, 3], [3, 5, 7], 0, 0)
        self.assertEqual(result["mae"], 3)
        self.assertEqual(result["violations"], 3)

    def test_float64_statistics_and_zero_error(self):
        result = numerical_metrics(np.array([0, 0], np.float32),
                                   np.array([3, 4], np.float32), 0, 0)
        self.assertEqual(result["mae"], 3.5)
        self.assertAlmostEqual(result["rmse"], np.sqrt(12.5))
        zeros = numerical_metrics([1, 1], [1, 1], 0, 0)
        for name in ("mae", "rmse", "max_abs", "p99_abs", "violation_fraction"):
            self.assertEqual(zeros[name], 0)

    def test_empty_and_shape_mismatch_rejected(self):
        for reference, candidate in [([], []), (np.empty((0, 2)), np.empty((0, 2))),
                                     ([1, 2], [[1, 2]]), ([[1], [2]], [[1, 2]])]:
            with self.subTest(reference=reference), self.assertRaises(DepthError):
                numerical_metrics(reference, candidate, 0, 0)

    def test_nonfinite_in_either_tensor_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            for reference, candidate in [([value], [0]), ([0], [value])]:
                with self.subTest(value=value), self.assertRaisesRegex(DepthError, "non-finite"):
                    numerical_metrics(reference, candidate, 0, 0)

    def test_invalid_tolerances_rejected(self):
        for bad in (-1, np.nan, np.inf, -np.inf, True, "0", [0]):
            for atol, rtol in ((bad, 0), (0, bad)):
                with self.subTest(bad=bad), self.assertRaises(DepthError):
                    numerical_metrics([0], [0], atol, rtol)

    def test_nonreal_and_overflow_rejected(self):
        for bad in ([1j], ["1"], [True]):
            with self.subTest(bad=bad), self.assertRaises(DepthError):
                numerical_metrics(bad, bad, 0, 0)
        with self.assertRaisesRegex(DepthError, "overflow"):
            numerical_metrics([-1e308], [1e308], 0, 0)


class OrderingMetricsTests(unittest.TestCase):
    def test_direction_coordinates_ties_and_uncertain(self):
        depth = [[1, 4, 4], [3, 2, 0]]
        result = ordering_metrics(depth, [pair([1, 0], [0, 1]), pair([0, 0], [1, 1]),
                                          pair([1, 0], [2, 0]),
                                          pair([2, 1], [1, 0], "uncertain")])
        self.assertEqual(result, {"correct": 1, "total": 3, "ignored": 1,
                                  "ties": 1, "accuracy": 1 / 3})

    def test_duplicate_points_and_duplicate_annotations(self):
        annotation = pair([0, 0], [1, 0])
        result = ordering_metrics([[2, 1]], [annotation, annotation, pair([0, 0], [0, 0])])
        self.assertEqual(result, {"correct": 2, "total": 3, "ignored": 0,
                                  "ties": 1, "accuracy": 2 / 3})

    def test_unannotated_and_all_uncertain_are_null(self):
        for annotations, ignored in (([], 0), ([pair([0, 0], [0, 0], "uncertain")], 1)):
            result = ordering_metrics([[1]], annotations)
            self.assertEqual(result, {"correct": 0, "total": 0, "ignored": ignored,
                                      "ties": 0, "accuracy": None})

    def test_invalid_coordinates_including_uncertain_rejected(self):
        for coordinate in ([True, 0], [0.0, 0], ["0", 0], [-1, 0], [2, 0],
                           [0, 1], [0], [0, 0, 0], None, np.array(0), np.array([[0, 0]])):
            for status in ("valid", "uncertain"):
                for near, far in ((coordinate, [0, 0]), ([0, 0], coordinate)):
                    with self.subTest(coordinate=coordinate, status=status), self.assertRaises(DepthError):
                        ordering_metrics([[1, 2]], [pair(near, far, status)])

    def test_numpy_integer_coordinates(self):
        result = ordering_metrics([[2, 1]], [pair([np.int64(0), np.int32(0)], [1, 0])])
        self.assertEqual(result["accuracy"], 1)

    def test_invalid_schema_and_depth_rejected(self):
        for annotations in (None, {}, [None], [{}], [pair([0, 0], [0, 0], "unknown")],
                            [pair([0, 0], [0, 0], np.array(["valid"]))]):
            with self.subTest(annotations=annotations), self.assertRaises(DepthError):
                ordering_metrics([[1]], annotations)
        for depth in ([], [1], [[[1]]], [[np.nan]], [[np.inf]], [[1j]]):
            with self.subTest(depth=depth), self.assertRaises(DepthError):
                ordering_metrics(depth, [])


if __name__ == "__main__":
    unittest.main()
