from __future__ import annotations

from typing import Any

import numpy as np

from .io_utils import resize_rgb


FEATURE_NAMES = [
    "contact_flag",
    "normal_force",
    "tangential_force",
    "slip_score",
    "left_finger_pressure_mean",
    "right_finger_pressure_mean",
    "pressure_sum",
    "pressure_center_x",
    "pressure_center_y",
]


def tactile_to_features(tactile: np.ndarray | None) -> np.ndarray | None:
    if tactile is None:
        return None
    x = np.asarray(tactile, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    finite = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    abs_x = np.abs(finite)
    pressure_sum = abs_x.sum(axis=1, keepdims=True)
    half = max(finite.shape[1] // 2, 1)
    left_mean = abs_x[:, :half].mean(axis=1, keepdims=True)
    right_mean = abs_x[:, half:].mean(axis=1, keepdims=True) if finite.shape[1] > 1 else left_mean.copy()
    normal_force = finite.mean(axis=1, keepdims=True)
    tangential_force = finite.std(axis=1, keepdims=True)
    delta = np.zeros_like(pressure_sum)
    delta[1:] = np.abs(pressure_sum[1:] - pressure_sum[:-1])
    contact_flag = (pressure_sum > 1e-5).astype(np.float32)
    idx = np.arange(finite.shape[1], dtype=np.float32)[None, :]
    denom = np.maximum(pressure_sum, 1e-6)
    center = (abs_x * idx).sum(axis=1, keepdims=True) / denom
    center = center / max(finite.shape[1] - 1, 1)
    return np.concatenate(
        [contact_flag, normal_force, tangential_force, delta, left_mean, right_mean, pressure_sum, center, np.zeros_like(center)],
        axis=1,
    ).astype(np.float32)


class TactileImageEncoder:
    """Deterministic lightweight encoder for tactile RGB frames.

    This is intentionally dependency-light so conversion and robot deployment use
    identical preprocessing without importing torch/OpenPI. It projects resized
    tactile RGB pixels to a fixed state vector that can be appended to pi0 state.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        *,
        image_size: tuple[int, int] = (32, 32),
        seed: int = 1701,
    ):
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = int(embedding_dim)
        self.image_size = image_size
        self.seed = int(seed)
        self._projections: dict[tuple[int, int, int], np.ndarray] = {}

    def encode_pair(self, left: Any | None = None, right: Any | None = None) -> np.ndarray:
        left_dim = self.embedding_dim // 2
        right_dim = self.embedding_dim - left_dim
        return np.concatenate(
            [
                self._encode_one(left, left_dim, salt=0),
                self._encode_one(right, right_dim, salt=1),
            ],
            axis=0,
        ).astype(np.float32)

    def _encode_one(self, image: Any | None, output_dim: int, *, salt: int) -> np.ndarray:
        if output_dim <= 0:
            return np.empty((0,), dtype=np.float32)
        if image is None:
            return np.zeros((output_dim,), dtype=np.float32)
        rgb = resize_rgb(_load_image_if_needed(image), self.image_size).astype(np.float32) / 255.0
        flat = (rgb - 0.5).reshape(-1)
        key = (flat.size, output_dim, self.seed + salt)
        if key not in self._projections:
            self._projections[key] = _projection(flat.size, output_dim, self.seed + salt)
        projection = self._projections[key]
        emb = flat @ projection
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
        return emb.astype(np.float32)


def tactile_images_to_embeddings(
    left: Any | None,
    right: Any | None,
    *,
    embedding_dim: int = 128,
    image_size: tuple[int, int] = (32, 32),
) -> np.ndarray | None:
    length = _sequence_length(left)
    if length is None:
        length = _sequence_length(right)
    if length is None:
        return None
    encoder = TactileImageEncoder(embedding_dim=embedding_dim, image_size=image_size)
    return np.stack(
        [encoder.encode_pair(_item_at(left, i), _item_at(right, i)) for i in range(length)],
        axis=0,
    ).astype(np.float32)


def _projection(input_dim: int, output_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((input_dim, output_dim)).astype(np.float32) / np.sqrt(input_dim)).astype(np.float32)


def _sequence_length(value: Any | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim < 4:
            return None
        return int(value.shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _item_at(value: Any | None, index: int) -> Any | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.ndim >= 4:
        return value[index]
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _load_image_if_needed(image: Any) -> np.ndarray:
    if isinstance(image, (str, bytes)):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Reading tactile image paths requires opencv-python") from exc
        path = image.decode("utf-8") if isinstance(image, bytes) else image
        bgr = cv2.imread(path)
        if bgr is None:
            raise RuntimeError(f"Failed to read tactile image path {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.asarray(image)
