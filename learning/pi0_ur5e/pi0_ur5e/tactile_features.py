from __future__ import annotations

import numpy as np


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
