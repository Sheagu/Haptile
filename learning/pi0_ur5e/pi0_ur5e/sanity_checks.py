from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import save_rgb, write_json


def inspect_episodes(episodes, output_dir: str | Path, *, include_tactile: bool = False, sample_images: int = 20) -> dict[str, Any]:
    output = Path(output_dir)
    debug_dir = output / "debug_vis"
    debug_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for episode in episodes:
        reports.append(inspect_episode(episode, debug_dir, include_tactile=include_tactile, sample_images=sample_images))
    summary = {
        "episode_count": len(episodes),
        "total_frames": int(sum(r["length"] for r in reports)),
        "episodes": reports,
    }
    write_json(output / "inspection_report.json", summary)
    return summary


def inspect_episode(episode, debug_dir: Path, *, include_tactile: bool, sample_images: int) -> dict[str, Any]:
    ts = np.asarray(episode.timestamps, dtype=np.float64)
    dt = np.diff(ts) if len(ts) > 1 else np.empty((0,), dtype=np.float64)
    action = np.asarray(episode.action, dtype=np.float32)
    state = np.asarray(episode.robot_state, dtype=np.float32)
    report = {
        "episode_id": episode.episode_id,
        "length": int(len(ts)),
        "timestamp": {
            "mean_hz": float(1.0 / np.mean(dt[dt > 0])) if np.any(dt > 0) else None,
            "min_dt": float(np.min(dt)) if dt.size else None,
            "max_dt": float(np.max(dt)) if dt.size else None,
            "duplicate_count": int(np.sum(dt <= 0)) if dt.size else 0,
            "drop_frame_like_count": int(np.sum(dt > 2.5 * np.median(dt[dt > 0]))) if np.any(dt > 0) else 0,
        },
        "action_range": {
            "min": np.nanmin(action, axis=0).tolist(),
            "max": np.nanmax(action, axis=0).tolist(),
            "large_xyz_count": int(np.sum(np.linalg.norm(action[:, :3], axis=1) > 0.10)) if action.shape[1] >= 3 else 0,
            "large_rot_count": int(np.sum(np.linalg.norm(action[:, 3:6], axis=1) > 0.75)) if action.shape[1] >= 6 else 0,
            "gripper_out_of_range_count": int(np.sum((action[:, -1] < -0.05) | (action[:, -1] > 1.05))) if action.shape[1] >= 7 else 0,
        },
        "alignment": action_state_alignment(action, state),
        "gripper_close_events": find_gripper_close_events(action[:, -1] if action.shape[1] else np.empty((0,))),
        "tactile": None,
    }
    if include_tactile and episode.tactile is not None:
        tactile = np.asarray(episode.tactile, dtype=np.float32)
        report["tactile"] = {
            "shape": list(tactile.shape),
            "has_nan": bool(np.isnan(tactile).any()),
            "has_inf": bool(np.isinf(tactile).any()),
            "abs_max": float(np.nanmax(np.abs(tactile))) if tactile.size else 0.0,
        }
    _save_debug_images(episode, debug_dir, sample_images)
    return report


def action_state_alignment(action: np.ndarray, state: np.ndarray) -> dict[str, Any]:
    if len(action) < 3 or state.shape[1] < 3 or action.shape[1] < 3:
        return {"available": False}
    state_delta = state[1:, :3] - state[:-1, :3]
    action_xyz = action[:-1, :3]
    corrs = []
    for i in range(3):
        a = action_xyz[:, i]
        b = state_delta[:, i]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            corrs.append(None)
        else:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return {"available": True, "xyz_corr": corrs, "mean_abs_state_delta": np.mean(np.abs(state_delta), axis=0).tolist()}


def find_gripper_close_events(gripper: np.ndarray, threshold: float = 0.5) -> list[int]:
    if len(gripper) < 2:
        return []
    was_open = gripper[:-1] < threshold
    now_closed = gripper[1:] >= threshold
    return (np.where(was_open & now_closed)[0] + 1).astype(int).tolist()


def _save_debug_images(episode, debug_dir: Path, sample_images: int) -> None:
    if episode.base_rgb is None and episode.wrist_rgb is None:
        return
    length = len(episode.timestamps)
    indices = np.linspace(0, max(length - 1, 0), num=min(sample_images, length), dtype=int)
    for idx in indices:
        try:
            base = _image_at(episode.base_rgb, idx)
            wrist = _image_at(episode.wrist_rgb, idx)
            if base is None and wrist is None:
                continue
            if base is None:
                base = np.zeros_like(wrist)
            if wrist is None:
                wrist = np.zeros_like(base)
            h = min(base.shape[0], wrist.shape[0])
            panel = np.concatenate([base[:h], wrist[:h]], axis=1)
            save_rgb(debug_dir / f"{episode.episode_id}_{idx:06d}.png", panel)
        except Exception:
            continue


def _image_at(source, idx: int):
    if source is None or isinstance(source, list):
        return None
    arr = np.asarray(source[idx])
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8)
