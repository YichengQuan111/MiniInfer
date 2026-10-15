"""Frozen 518/392 RGB8 geometry and FP32 transforms (NumPy only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

MODEL_SIZE = 518
MAX_SIDE = 4096
MAX_PIXELS = 2097152
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DepthError(ValueError):
    """Invalid depth input, artifact or contract."""


def validate_model_size(model_size: int) -> int:
    if type(model_size) is not int or model_size not in (518, 392):
        raise DepthError("model_size must be integer 518 or 392")
    return model_size


def contract_version(model_size: int) -> str:
    return "depth-v1" if validate_model_size(model_size) == 518 else "depth-v2-392"


def validate_dimensions(width: int, height: int) -> None:
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (width, height)):
        raise DepthError("width and height must be integers")
    if width <= 0 or height <= 0:
        raise DepthError("width and height must be positive")
    if max(width, height) > MAX_SIDE or width * height > MAX_PIXELS:
        raise DepthError(f"image exceeds max_side={MAX_SIDE} or max_pixels={MAX_PIXELS}")


@dataclass(frozen=True)
class Geometry:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def geometry(width: int, height: int, model_size: int = MODEL_SIZE) -> Geometry:
    model_size = validate_model_size(model_size)
    validate_dimensions(width, height)
    longest = max(width, height)
    rw = max(1, (width * model_size + longest // 2) // longest)
    rh = max(1, (height * model_size + longest // 2) // longest)
    left, top = (model_size - rw) // 2, (model_size - rh) // 2
    return Geometry(width, height, rw, rh, left, top,
                    model_size - rw - left, model_size - rh - top)


def geometry_model_size(transform: Geometry) -> int:
    if not isinstance(transform, Geometry) or any(type(v) is not int for v in transform.to_dict().values()):
        raise DepthError("geometry must contain eight integer fields")
    size = validate_model_size(transform.resized_width + transform.pad_left + transform.pad_right)
    if (transform.resized_height + transform.pad_top + transform.pad_bottom != size
            or transform != geometry(transform.original_width, transform.original_height, size)):
        raise DepthError("geometry does not match the frozen transform")
    return size


def _axis(source: int, target: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Every operation below is FP32, in this order; clamp before floor.
    coordinate = (np.arange(target, dtype=np.float32) + np.float32(0.5))
    coordinate = coordinate * np.float32(source / target) - np.float32(0.5)
    coordinate = np.clip(coordinate, np.float32(0), np.float32(source - 1))
    lower = np.floor(coordinate).astype(np.intp)
    return lower, np.minimum(lower + 1, source - 1), coordinate - lower.astype(np.float32)


def bilinear_half_pixel(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an HW or HWC FP32 array with clamp and horizontal-then-vertical lerp."""
    validate_dimensions(width, height)
    if not isinstance(image, np.ndarray) or image.dtype != np.float32 or image.ndim not in (2, 3):
        raise DepthError("bilinear input must be an HW or HWC float32 array")
    validate_dimensions(image.shape[1], image.shape[0])
    if image.ndim == 3 and image.shape[2] == 0:
        raise DepthError("bilinear input must have at least one channel")
    if not np.isfinite(image).all():
        raise DepthError("bilinear input contains NaN or infinity")
    y0, y1, fy = _axis(image.shape[0], height)
    x0, x1, fx = _axis(image.shape[1], width)
    fx = fx[None, :] if image.ndim == 2 else fx[None, :, None]
    fy = fy[:, None] if image.ndim == 2 else fy[:, None, None]
    upper_left, upper_right = image[y0[:, None], x0], image[y0[:, None], x1]
    lower_left, lower_right = image[y1[:, None], x0], image[y1[:, None], x1]
    upper = upper_left + (upper_right - upper_left) * fx
    lower = lower_left + (lower_right - lower_left) * fx
    result = upper + (lower - upper) * fy
    if not np.isfinite(result).all():
        raise DepthError("bilinear interpolation produced non-finite values")
    return np.ascontiguousarray(result, dtype=np.float32)


def preprocess(rgb: np.ndarray, model_size: int = MODEL_SIZE) -> tuple[np.ndarray, Geometry]:
    model_size = validate_model_size(model_size)
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise DepthError("image must be uint8 RGB in HWC layout with three channels")
    transform = geometry(rgb.shape[1], rgb.shape[0], model_size)
    pixels = rgb.astype(np.float32) / np.float32(255)
    resized = bilinear_half_pixel(pixels, transform.resized_height, transform.resized_width)
    padded = np.zeros((model_size, model_size, 3), dtype=np.float32)
    padded[transform.pad_top:transform.pad_top + transform.resized_height,
           transform.pad_left:transform.pad_left + transform.resized_width] = resized
    normalized = (padded - MEAN) / STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None]), transform


def postprocess(raw: np.ndarray, transform: Geometry) -> np.ndarray:
    model_size = geometry_model_size(transform)
    if not isinstance(raw, np.ndarray) or raw.dtype != np.float32 or raw.shape != (1, model_size, model_size):
        raise DepthError(f"raw output must be float32 [1,{model_size},{model_size}]")
    if not np.isfinite(raw).all():
        raise DepthError("raw output contains NaN or infinity")
    cropped = raw[0, transform.pad_top:transform.pad_top + transform.resized_height,
                  transform.pad_left:transform.pad_left + transform.resized_width]
    return bilinear_half_pixel(cropped, transform.original_height, transform.original_width)


def rgb_from_bytes(payload: bytes, width: int, height: int) -> np.ndarray:
    validate_dimensions(width, height)
    expected = 3 * width * height
    if len(payload) != expected:
        raise DepthError(f"RGB8 payload has {len(payload)} bytes; expected exactly {expected}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()


def read_ppm(path: Path) -> np.ndarray:
    """Read one P6 RGB8 image; comments allowed between header tokens, exact payload."""
    with path.open("rb") as stream:
        tokens = []
        for _ in range(4):
            token = bytearray()
            while True:
                char = stream.read(1)
                if not char:
                    raise DepthError("truncated PPM header")
                if char == b"#":
                    if len(stream.readline(4097)) > 4096:
                        raise DepthError("PPM comment too long")
                    continue
                if char not in b" \t\r\n\v\f":
                    token.extend(char)
                    break
            while True:
                char = stream.read(1)
                if not char:
                    raise DepthError("truncated PPM header")
                if char in b" \t\r\n\v\f":
                    break
                token.extend(char)
                if len(token) > 32:
                    raise DepthError("PPM header token too long")
            tokens.append(bytes(token))
        if tokens[0] != b"P6" or tokens[3] != b"255":
            raise DepthError("PPM must be P6 with maxval 255")
        try:
            width, height = int(tokens[1]), int(tokens[2])
        except ValueError as exc:
            raise DepthError("invalid PPM dimensions") from exc
        validate_dimensions(width, height)
        # Consume CRLF as one separator; never skip arbitrary raster whitespace.
        if char == b"\r":
            position = stream.tell()
            if stream.read(1) != b"\n":
                stream.seek(position)
        return rgb_from_bytes(stream.read(3 * width * height + 1), width, height)


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".ppm", ".pnm"):
        return read_ppm(path)
    try:
        from PIL import Image
    except ImportError as exc:
        raise DepthError("Pillow is required for images other than PPM P6") from exc
    with Image.open(path) as source:
        validate_dimensions(*source.size)
        # No automatic EXIF rotation, color-profile conversion or alpha blending.
        return np.ascontiguousarray(np.array(source.convert("RGB"), dtype=np.uint8))


def ppm_bytes(rgb: np.ndarray) -> bytes:
    preprocess_shape = rgb.shape
    if rgb.dtype != np.uint8 or len(preprocess_shape) != 3 or preprocess_shape[2] != 3:
        raise DepthError("PPM writer expects uint8 RGB HWC")
    validate_dimensions(rgb.shape[1], rgb.shape[0])
    return f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode("ascii") + rgb.tobytes(order="C")
