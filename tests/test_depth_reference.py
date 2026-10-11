"""Hand-computed depth geometry/FP32 cases; no torch or model weights needed."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from depth.preprocess import (DepthError, MEAN, STD, bilinear_half_pixel, geometry,
                              postprocess, ppm_bytes, preprocess, read_ppm,
                              rgb_from_bytes, validate_dimensions)
from depth.reference import load_samples, write_array


class DepthTransformTests(unittest.TestCase):
    def test_round_half_up_and_odd_padding(self):
        landscape = geometry(4, 3)  # 3*518/4=388.5 ->389;129 pad ->64,65
        self.assertEqual((landscape.resized_width, landscape.resized_height), (518, 389))
        self.assertEqual((landscape.pad_top, landscape.pad_bottom), (64, 65))
        portrait = geometry(3, 4)
        self.assertEqual((portrait.pad_left, portrait.pad_right), (64, 65))
        self.assertEqual((portrait.pad_top, portrait.pad_bottom), (0, 0))
        self.assertEqual(geometry(1, 4096).resized_width, 1)

    def test_dimension_limits(self):
        validate_dimensions(2048, 1024)
        validate_dimensions(4096, 1)
        for width, height in [(0, 1), (-1, 2), (1, 0), (4097, 1), (2049, 1024),
                              (1.0, 1), (True, 1)]:
            with self.subTest(width=width, height=height), self.assertRaises(DepthError):
                validate_dimensions(width, height)

    def test_half_pixel_clamp_hand_computed(self):
        source = np.array([[0, 10], [20, 30]], dtype=np.float32)
        expected = np.array([[0, 2.5, 7.5, 10], [5, 7.5, 12.5, 15],
                             [15, 17.5, 22.5, 25], [20, 22.5, 27.5, 30]], dtype=np.float32)
        np.testing.assert_array_equal(bilinear_half_pixel(source, 4, 4), expected)
        np.testing.assert_array_equal(bilinear_half_pixel(source, 1, 1), [[15]])
        np.testing.assert_array_equal(bilinear_half_pixel(source, 2, 2), source)

    def test_singleton_axes_and_rgb_channels(self):
        source = np.array([[[0, 1, 2], [4, 5, 6]]], dtype=np.float32)
        result = bilinear_half_pixel(source, 3, 4)
        expected = np.array([[0, 1, 3, 4], [0, 1, 3, 4], [0, 1, 3, 4]], dtype=np.float32)
        for channel in range(3):
            np.testing.assert_array_equal(result[:, :, channel], expected + channel)
        self.assertTrue(result.flags.c_contiguous)

    def test_preprocess_extreme_values_and_normalized_padding(self):
        rgb = np.array([[[0, 255, 128], [0, 255, 128], [0, 255, 128], [0, 255, 128]]], dtype=np.uint8)
        tensor, transform = preprocess(rgb)
        self.assertEqual(tensor.shape, (1, 3, 518, 518))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        expected = (np.array([0, 1, np.float32(128) / np.float32(255)], dtype=np.float32) - MEAN) / STD
        np.testing.assert_array_equal(tensor[0, :, transform.pad_top, 0], expected)
        np.testing.assert_array_equal(tensor[0, :, 0, 0], -MEAN / STD)

    def test_uint8_becomes_float_before_interpolation(self):
        rgb = np.zeros((518, 1036, 3), dtype=np.uint8)
        rgb[:, 1::2, :] = 1
        tensor, transform = preprocess(rgb)
        # Every x source coordinate is x*2+0.5, averaging 0 and 1 without uint8 quantization.
        expected = (np.full(3, np.float32(0.5) / np.float32(255)) - MEAN) / STD
        np.testing.assert_array_equal(tensor[0, :, transform.pad_top, 0], expected)

    def test_postprocess_crops_padding_before_restore(self):
        transform = geometry(4, 2)  # cropped plane 259x518 ->2x4
        raw = np.full((1, 518, 518), -1000, dtype=np.float32)
        plane = (np.arange(259, dtype=np.float32)[:, None] * np.float32(10)
                 + np.arange(518, dtype=np.float32)[None, :])
        raw[0, transform.pad_top:transform.pad_top + 259] = plane
        result = postprocess(raw, transform)
        # y=(.5*129.5-.5,1.5*129.5-.5)=(64.25,193.75); x step129.5.
        expected = np.array([[706.75, 836.25, 965.75, 1095.25],
                             [2001.75, 2131.25, 2260.75, 2390.25]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        self.assertTrue(result.flags.owndata)
        self.assertTrue(result.flags.c_contiguous)
        raw.fill(0)
        np.testing.assert_array_equal(result, expected)

    def test_no_per_image_normalization(self):
        raw = np.full((1, 518, 518), 42.25, dtype=np.float32)
        np.testing.assert_array_equal(postprocess(raw, geometry(3, 2)), np.full((2, 3), 42.25))

    def test_invalid_arrays_and_transform(self):
        for rgb in [np.zeros((2, 2, 3), dtype=np.float32), np.zeros((2, 2, 4), dtype=np.uint8),
                    np.zeros((0, 2, 3), dtype=np.uint8)]:
            with self.assertRaises(DepthError):
                preprocess(rgb)
        for value in [float("nan"), float("inf"), float("-inf")]:
            raw = np.zeros((1, 518, 518), dtype=np.float32)
            raw[0, 0, 0] = value  # even discarded padding must be checked
            with self.assertRaises(DepthError):
                postprocess(raw, geometry(3, 2))
            with self.assertRaises(DepthError):
                bilinear_half_pixel(np.array([[value]], dtype=np.float32), 2, 2)
        for raw in [np.zeros((518, 518), dtype=np.float32), np.zeros((1, 518, 518), dtype=np.float64)]:
            with self.assertRaises(DepthError):
                postprocess(raw, geometry(2, 2))
        with self.assertRaises(DepthError):
            postprocess(np.zeros((1, 518, 518), dtype=np.float32), replace(geometry(3, 2), pad_left=1))


class DepthArtifactTests(unittest.TestCase):
    def test_exact_rgb_byte_count(self):
        for payload in (b"", bytes(5), bytes(7)):
            with self.assertRaises(DepthError):
                rgb_from_bytes(payload, 2, 1)
        np.testing.assert_array_equal(rgb_from_bytes(bytes(range(6)), 2, 1), [[[0, 1, 2], [3, 4, 5]]])

    def test_ppm_comments_crlf_and_raster_whitespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.ppm"
            raster = bytes([10, 13, 32, 35, 0, 255])
            for header in [b"P6\n# real header comment\n2 1\n255\n", b"P6\r\n2 1\r\n255\r\n"]:
                path.write_bytes(header + raster)
                np.testing.assert_array_equal(read_ppm(path), rgb_from_bytes(raster, 2, 1))
            image = rgb_from_bytes(raster, 2, 1)
            path.write_bytes(ppm_bytes(image))
            np.testing.assert_array_equal(read_ppm(path), image)
            for payload in [b"P3\n2 1\n255\n" + raster, b"P6\n2 1\n256\n" + raster,
                            b"P6\n2 1\n255\n" + raster[:-1], b"P6\n2 1\n255\n" + raster + b"x",
                            b"P6\n4097 1\n255\n", b"P6\n"]:
                path.write_bytes(payload)
                with self.assertRaises(DepthError):
                    read_ppm(path)

    def test_fp32_file_encoding_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tensor.f32"
            meta = write_array(path, np.array([[1, -2]], dtype=np.float32), "<f4")
            self.assertEqual(path.read_bytes(), b"\x00\x00\x80\x3f\x00\x00\x00\xc0")
            self.assertEqual(meta["shape"], [1, 2])
            self.assertEqual(meta["bytes"], 8)
            with self.assertRaises(FileExistsError):
                write_array(path, np.ones((2,), dtype=np.float32), "<f4")

    def test_sample_manifest_validation_without_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "input.ppm").write_bytes(ppm_bytes(np.zeros((2, 2, 3), dtype=np.uint8)))
            manifest = directory / "samples.json"
            sample = {"id": "test", "path": "input.ppm", "split": "development", "near_far_pairs": []}
            def write(samples):
                manifest.write_text(json.dumps({"schema_version": 1, "samples": samples}), encoding="utf-8")
            write([sample])
            self.assertEqual(load_samples(manifest)[0]["entry"]["near_far_pairs"], [])
            for samples in [[], [sample, sample], [dict(sample, id="../bad")],
                            [dict(sample, sha256="wrong")], [dict(sample, split="unknown")],
                            [dict(sample, near_far_pairs=[{"status": "valid", "near": [0, 0], "far": [2, 1]}])]]:
                write(samples)
                with self.assertRaises(DepthError):
                    load_samples(manifest)

    def test_cli_refuses_existing_output_before_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            marker = output / "keep"
            marker.write_text("preserve", encoding="utf-8")
            result = subprocess.run([sys.executable, str(PROJECT_ROOT / "tools/depth/reference.py"),
                                     "reference", "--repo", "missing", "--weights", "missing",
                                     "--samples", "missing", "--output", str(output)],
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite", result.stderr)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_frozen_contract_matches_implemented_shape_and_limits(self):
        contract = json.loads((PROJECT_ROOT / "data/depth/contract.json").read_text(encoding="utf-8"))
        self.assertEqual(contract["input"]["shape"], [1, 3, 518, 518])
        self.assertEqual(contract["raw_output"]["shape"], [1, 518, 518])
        self.assertFalse(contract["postprocess"]["per_image_minmax"])
        self.assertEqual(contract["source_image"]["max_pixels"], 2097152)


if __name__ == "__main__":
    unittest.main()
