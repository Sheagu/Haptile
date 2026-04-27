from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io_utils import decode_h5_video, load_pickle, load_yaml, parse_timestamp
from .schema import Episode, Pi0Ur5eConfig
from .tactile_features import tactile_to_features
from .transforms import build_action


DEFAULT_FIELD_MAP = {
    "base_rgb": ["base_rgb", "base_camera_rgb", "base_camera_rgb_0", "observation.images.base_rgb"],
    "wrist_rgb": ["wrist_rgb", "wrist_camera_rgb", "camera_wrist_rgb", "observation.images.wrist_rgb"],
    "robot_state": ["robot_state", "joint_positions", "ee_pos_quat", "observation.state"],
    "ee_pose": ["ee_pos_quat", "eef_pose", "ee_pose", "tcp_pose"],
    "action": ["action", "control", "actions"],
    "gripper_state": ["gripper_state", "gripper_position"],
    "tactile": ["tactile", "touch", "force", "pressure"],
    "language_instruction": ["language_instruction", "prompt", "task"],
}


class DatasetReader:
    def __init__(self, root: str | Path, config_path: str | Path | None = None, config: dict[str, Any] | None = None):
        self.root = Path(root)
        cfg = load_yaml(config_path)
        if config:
            cfg.update(config)
        self.config = Pi0Ur5eConfig(
            action_mode=cfg.get("action_mode", "ee_delta_6d_gripper"),
            image_size=tuple(cfg.get("image_size", [224, 224])),
            camera_padding_strategy=cfg.get("camera_padding_strategy", "duplicate_base"),
            include_tactile=bool(cfg.get("include_tactile", False)),
            tactile_feature_mode=cfg.get("tactile_feature_mode", "none"),
            default_prompt=cfg.get("default_prompt", "pick up the paper cup and place it on the target"),
            hz=cfg.get("hz"),
            field_map=cfg.get("field_map", {}),
        )
        self.field_map = {**DEFAULT_FIELD_MAP, **self.config.field_map}

    def episodes(self) -> list[Episode]:
        if (self.root / "meta" / "info.json").exists() and (self.root / "data").exists():
            return self._read_lerobot_v2()
        if (self.root / "meta" / "info.json").exists() and (self.root / "episodes").exists():
            return self._read_lerobot_jsonl()
        candidates = self._episode_paths()
        episodes = [self._read_episode_path(path) for path in candidates]
        return [episode for episode in episodes if episode is not None]

    def _episode_paths(self) -> list[Path]:
        if self.root.is_file():
            return [self.root]
        h5s = sorted(self.root.glob("**/trajectory.h5"))
        if h5s:
            return h5s
        frame_dirs = []
        for path in sorted(self.root.iterdir()):
            if path.is_dir() and list(path.glob("*.pkl")):
                frame_dirs.append(path)
        if frame_dirs:
            return frame_dirs
        npzs = sorted(self.root.glob("*.npz"))
        if npzs:
            return npzs
        pkls = sorted(self.root.glob("*.pkl"))
        return [self.root] if pkls else []

    def _read_episode_path(self, path: Path) -> Episode | None:
        if path.name == "trajectory.h5":
            frames, timestamps, metadata = self._frames_from_h5(path)
        elif path.is_dir():
            frames, timestamps, metadata = self._frames_from_pkl_dir(path)
        elif path.suffix == ".npz":
            return self._episode_from_npz(path)
        else:
            return None
        if not frames:
            return None
        return self._episode_from_frames(path.stem if path.is_file() else path.name, frames, timestamps, metadata)

    def _frames_from_pkl_dir(self, path: Path) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
        files = sorted(path.glob("*.pkl"), key=_natural_key)
        frames = []
        times = []
        for idx, file in enumerate(files):
            frame = load_pickle(file)
            frame["file_path"] = str(file)
            frames.append(frame)
            times.append(parse_timestamp(frame.get("timestamp"), idx))
        return frames, np.asarray(times, dtype=np.float64), {"source_path": str(path)}

    def _frames_from_h5(self, path: Path) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("Reading trajectory.h5 requires h5py") from exc
        with h5py.File(path, "r") as f:
            frames_group = f.get("frames", {})
            numeric = {key: np.asarray(ds[:]) for key, ds in frames_group.items()}
            frame_count = int(f.attrs.get("frame_count", 0)) or (len(next(iter(numeric.values()))) if numeric else len(f.get("timestamps", [])))
            videos = {}
            for key, ds in f.get("videos", {}).items():
                videos[key] = decode_h5_video(ds, is_depth=False)
            times = np.arange(frame_count, dtype=np.float64)
            if "timestamps" in f:
                times = np.asarray([parse_timestamp(x, i) for i, x in enumerate(f["timestamps"][:])], dtype=np.float64)
            attrs = {k: _attr_to_python(v) for k, v in f.attrs.items()}
        grouped = _group_video_streams(videos)
        frames = []
        for i in range(frame_count):
            frame = {key: value[i] for key, value in numeric.items() if len(value) > i}
            for key, value in grouped.items():
                if len(value) > i:
                    frame[key] = value[i]
            frame["file_path"] = f"{path}::{i}"
            frames.append(frame)
        attrs["source_path"] = str(path)
        return frames, times, attrs

    def _episode_from_npz(self, path: Path) -> Episode:
        data = np.load(path, allow_pickle=True)
        frame_count = len(data[self._first_existing(data, "robot_state")])
        timestamps = np.asarray(data["timestamps"] if "timestamps" in data else np.arange(frame_count), dtype=np.float64)
        prompt = str(data["language_instruction"]) if "language_instruction" in data else self.config.default_prompt
        state = np.asarray(data[self._first_existing(data, "robot_state")], dtype=np.float32)
        gripper = np.asarray(data["gripper_state"], dtype=np.float32) if "gripper_state" in data else None
        action = build_action(np.asarray(data["action"], dtype=np.float32) if "action" in data else None, state, gripper, self.config.action_mode)
        tactile = np.asarray(data["tactile"], dtype=np.float32) if "tactile" in data else None
        if self.config.include_tactile and self.config.tactile_feature_mode == "low_dim":
            tactile = tactile_to_features(tactile)
        ep = Episode(path.stem, timestamps, data.get("base_rgb"), data.get("wrist_rgb"), state, action, gripper, tactile, prompt, {"source_path": str(path)})
        ep.validate()
        return ep

    def _episode_from_frames(self, episode_id: str, frames: list[dict[str, Any]], timestamps: np.ndarray, metadata: dict[str, Any]) -> Episode:
        prompt = self._first_value(frames, "language_instruction") or self.config.default_prompt
        base = self._stack_or_paths(frames, "base_rgb")
        wrist = self._stack_or_paths(frames, "wrist_rgb")
        state = self._numeric_series(frames, "ee_pose")
        if state is None:
            state = self._numeric_series(frames, "robot_state")
        if state is None:
            raise ValueError(f"{episode_id}: could not detect robot_state/ee pose fields")
        gripper = self._numeric_series(frames, "gripper_state")
        raw_action = self._numeric_series(frames, "action")
        action = build_action(raw_action, state, gripper, self.config.action_mode)
        tactile = self._numeric_series(frames, "tactile")
        if self.config.include_tactile and self.config.tactile_feature_mode == "low_dim":
            tactile = tactile_to_features(tactile)
        meta = {
            "task_name": metadata.get("task_name", ""),
            "success": metadata.get("success", None),
            "hz": metadata.get("video_fps", self.config.hz),
            "source_path": metadata.get("source_path", ""),
            "camera_info": {"base_rgb": _shape_or_type(base), "wrist_rgb": _shape_or_type(wrist)},
            "action_type": self.config.action_mode,
        }
        ep = Episode(episode_id, timestamps, base, wrist, state.astype(np.float32), action.astype(np.float32), gripper, tactile, str(prompt), meta)
        ep.validate()
        return ep

    def _read_lerobot_jsonl(self) -> list[Episode]:
        episodes = []
        for path in sorted((self.root / "episodes").glob("episode_*.jsonl")):
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            frames = []
            for row in rows:
                frames.append({
                    "base_rgb": row.get("observation", {}).get("images", {}).get("base_rgb"),
                    "wrist_rgb": row.get("observation", {}).get("images", {}).get("wrist_rgb"),
                    "robot_state": row.get("observation", {}).get("state"),
                    "action": row.get("action"),
                    "language_instruction": row.get("language_instruction"),
                })
            timestamps = np.asarray([row.get("timestamp", i) for i, row in enumerate(rows)], dtype=np.float64)
            episodes.append(self._episode_from_frames(path.stem, frames, timestamps, {"source_path": str(path)}))
        return episodes

    def _read_lerobot_v2(self) -> list[Episode]:
        os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")
        try:
            import datasets
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "Reading a real LeRobot dataset requires LeRobot. Run with the OpenPI uv environment."
            ) from exc
        datasets.config.HF_DATASETS_CACHE = Path(os.environ["HF_DATASETS_CACHE"])
        dataset = LeRobotDataset(repo_id=str(self.root), root=self.root)
        grouped: dict[int, list[dict[str, Any]]] = {}
        for idx in range(len(dataset)):
            item = dataset[idx]
            ep_idx = int(_to_numpy(item["episode_index"]).reshape(()))
            grouped.setdefault(ep_idx, []).append(item)
        episodes = []
        for ep_idx, items in sorted(grouped.items()):
            frames = []
            timestamps = []
            for item in items:
                frames.append(
                    {
                        "base_rgb": _image_to_hwc(item.get("observation.images.base_rgb")),
                        "wrist_rgb": _image_to_hwc(item.get("observation.images.wrist_rgb")),
                        "robot_state": _to_numpy(item["observation.state"]),
                        "action": _to_numpy(item["action"]),
                        "language_instruction": item.get("task", self.config.default_prompt),
                    }
                )
                timestamps.append(float(_to_numpy(item["timestamp"]).reshape(())))
            episodes.append(
                self._episode_from_frames(
                    f"episode_{ep_idx:06d}",
                    frames,
                    np.asarray(timestamps, dtype=np.float64),
                    {"source_path": str(self.root), "task_name": self.config.default_prompt},
                )
            )
        return episodes

    def _first_existing(self, mapping: Any, canonical: str) -> str:
        for key in self.field_map.get(canonical, [canonical]):
            if key in mapping:
                return key
        raise KeyError(canonical)

    def _first_value(self, frames: list[dict[str, Any]], canonical: str) -> Any:
        for frame in frames:
            for key in self.field_map.get(canonical, [canonical]):
                if key in frame:
                    return frame[key]
        return None

    def _numeric_series(self, frames: list[dict[str, Any]], canonical: str) -> np.ndarray | None:
        values = []
        found = False
        for frame in frames:
            value = _lookup(frame, self.field_map.get(canonical, [canonical]))
            if value is None:
                return None if not found else np.asarray(values, dtype=np.float32)
            found = True
            values.append(np.asarray(value, dtype=np.float32).reshape(-1))
        return np.asarray(values, dtype=np.float32) if found else None

    def _stack_or_paths(self, frames: list[dict[str, Any]], canonical: str) -> Any:
        values = []
        for frame in frames:
            value = _lookup(frame, self.field_map.get(canonical, [canonical]))
            if value is None:
                return None
            values.append(value)
        if values and isinstance(values[0], str):
            return values
        try:
            return np.asarray(values)
        except ValueError:
            return values


def _lookup(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
        current = mapping
        ok = True
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                ok = False
                break
        if ok:
            return current
    return None


def _group_video_streams(videos: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    groups: dict[str, list[tuple[int, np.ndarray]]] = {}
    result = {}
    for key, value in videos.items():
        match = re.match(r"^(.*)_(\d+)$", key)
        if match:
            groups.setdefault(match.group(1), []).append((int(match.group(2)), value))
        else:
            result[key] = value
    for key, items in groups.items():
        items.sort(key=lambda item: item[0])
        result[key] = np.stack([value for _, value in items], axis=1)
    return result


def _natural_key(path: Path):
    return [int(text) if text.isdigit() else text for text in re.split(r"(\d+)", path.name)]


def _attr_to_python(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _shape_or_type(value: Any) -> Any:
    return list(value.shape) if hasattr(value, "shape") else type(value).__name__


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _image_to_hwc(value: Any) -> np.ndarray:
    image = _to_numpy(value)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    return image.astype(np.uint8)
