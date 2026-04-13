#!/usr/bin/env python3
"""Convert legacy per-frame PKL trajectories into compact trajectory.h5 files.

The legacy format stores one PKL per frame and often duplicates tactile PNG files on disk.
This script converts each trajectory directory into a single H5 file with:

- RGB streams stored as embedded MP4 byte arrays
- Depth stored as RealSense-style 8-bit grayscale video
- Numeric arrays stored as compressed HDF5 datasets

By default, large legacy-only fields such as ``*_marker_tracking`` and ``*_raw_rgb`` are dropped
to reduce output size.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


DEPTH_DISPLAY_MIN_MM = 200
DEPTH_DISPLAY_MAX_MM = 1500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert legacy PKL frame folders into compact trajectory.h5 files."
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help="Directory containing legacy trajectory folders, e.g. shared/data/bc_data/grab_32/0402",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where converted folders should be written. Defaults to <input_root>_h5",
    )
    parser.add_argument(
        "--keep-marker-tracking",
        action="store_true",
        help="Keep *_marker_tracking arrays instead of dropping them.",
    )
    parser.add_argument(
        "--keep-raw-rgb",
        action="store_true",
        help="Keep *_raw_rgb arrays instead of dropping them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output trajectory.h5 files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first N trajectory folders for testing.",
    )
    return parser.parse_args()


def infer_timestamp_from_path(path: Path) -> dt.datetime:
    stem = path.stem
    return dt.datetime.strptime(stem, "%Y-%m-%dT%H-%M-%S-%f")


def filter_legacy_frame(
    frame: dict[str, Any],
    *,
    keep_marker_tracking: bool,
    keep_raw_rgb: bool,
) -> dict[str, Any]:
    filtered: dict[str, Any] = dict(frame)

    # For tactile streams, store the raw RGB frames but keep the compact stream names
    # so downstream consumers can continue reading tactile_left/right_rgb.
    for sensor_name in ("tactile_left", "tactile_right"):
        raw_key = f"{sensor_name}_raw_rgb"
        rgb_key = f"{sensor_name}_rgb"
        if raw_key in filtered:
            filtered[rgb_key] = filtered[raw_key]

    result: dict[str, Any] = {}
    for key, value in filtered.items():
        if not keep_marker_tracking and key.endswith("_marker_tracking"):
            continue
        if key.endswith("_raw_rgb"):
            if keep_raw_rgb and key not in {f"{name}_raw_rgb" for name in ("tactile_left", "tactile_right")}:
                result[key] = value
            continue
        result[key] = value
    return result


class CompactTrajectoryWriter:
    def __init__(self, output_path: Path, video_fps: float):
        self.output_path = output_path
        self.file = h5py.File(output_path, "w")
        self.frames_group = self.file.create_group("frames")
        self.videos_group = self.file.create_group("videos")
        self.timestamps = self.file.create_dataset(
            "timestamps",
            shape=(0,),
            maxshape=(None,),
            chunks=(256,),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        self.video_fps = float(video_fps)
        self.file.attrs["video_fps"] = self.video_fps
        self.frame_count = 0
        self.file.attrs["frame_count"] = self.frame_count

        self.datasets: dict[str, h5py.Dataset] = {}
        self.video_tempfiles: dict[str, str] = {}
        self.video_writers: dict[str, cv2.VideoWriter] = {}
        self.video_source_kinds: dict[str, str] = {}

    @staticmethod
    def _pack_depth_frame(array: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        depth_mm = array.astype(np.float32)
        valid = depth_mm > 0
        depth_clipped = np.clip(depth_mm, DEPTH_DISPLAY_MIN_MM, DEPTH_DISPLAY_MAX_MM)
        depth_scaled = (
            (depth_clipped - DEPTH_DISPLAY_MIN_MM)
            * 255.0
            / (DEPTH_DISPLAY_MAX_MM - DEPTH_DISPLAY_MIN_MM)
        )
        depth_uint8 = np.zeros(array.shape, dtype=np.uint8)
        depth_uint8[valid] = depth_scaled[valid].astype(np.uint8)
        depth_rgb = np.repeat(depth_uint8[..., None], 3, axis=-1)
        return depth_rgb, {
            "source_kind": "depth",
            "source_dtype": str(array.dtype),
            "encoding": "depth_uint8_gray",
            "depth_min_mm": DEPTH_DISPLAY_MIN_MM,
            "depth_max_mm": DEPTH_DISPLAY_MAX_MM,
        }

    @staticmethod
    def _iter_video_streams(key: str, array: np.ndarray):
        if key.endswith("_rgb") and array.dtype == np.uint8:
            if array.ndim == 3:
                yield key, array, {"source_kind": "rgb", "source_dtype": "uint8"}
                return
            if array.ndim == 4 and array.shape[-1] == 3:
                for idx, frame in enumerate(array):
                    yield (
                        f"{key}_{idx}",
                        frame,
                        {"source_kind": "rgb", "source_dtype": "uint8", "source_index": idx},
                    )
                return

        if key.endswith("_depth"):
            if array.ndim == 2:
                packed, attrs = CompactTrajectoryWriter._pack_depth_frame(array)
                yield key, packed, attrs
                return
            if array.ndim == 3:
                for idx, frame in enumerate(array):
                    packed, attrs = CompactTrajectoryWriter._pack_depth_frame(frame)
                    attrs = dict(attrs)
                    attrs["source_index"] = idx
                    yield f"{key}_{idx}", packed, attrs

    def _append_video_frame(self, key: str, array: np.ndarray, attrs: dict[str, Any]) -> None:
        writer = self.video_writers.get(key)
        if writer is None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_file.close()
            self.video_tempfiles[key] = temp_file.name
            height, width = array.shape[:2]
            writer = cv2.VideoWriter(
                temp_file.name,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.video_fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {key}")
            self.video_writers[key] = writer

            video_ds = self.videos_group.create_dataset(
                key,
                shape=(0,),
                maxshape=(None,),
                dtype=np.uint8,
            )
            video_ds.attrs["fps"] = self.video_fps
            video_ds.attrs["height"] = height
            video_ds.attrs["width"] = width
            video_ds.attrs["channels"] = array.shape[2]
            video_ds.attrs["codec"] = "mp4v"
            for attr_key, attr_value in attrs.items():
                video_ds.attrs[attr_key] = attr_value
            self.video_source_kinds[key] = str(attrs.get("source_kind", "rgb"))

        if self.video_source_kinds[key] == "depth":
            writer.write(array)
        else:
            writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

    def _create_dataset(self, key: str, value: Any) -> h5py.Dataset:
        array = np.asarray(value)
        if array.dtype.kind in {"U", "S", "O"} and array.ndim == 0:
            ds = self.frames_group.create_dataset(
                key,
                shape=(0,),
                maxshape=(None,),
                chunks=(256,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            self.datasets[key] = ds
            return ds

        ds = self.frames_group.create_dataset(
            key,
            shape=(0,) + array.shape,
            maxshape=(None,) + array.shape,
            chunks=(1,) + array.shape if array.ndim > 0 else (256,),
            dtype=array.dtype,
            compression="gzip",
            compression_opts=4,
        )
        self.datasets[key] = ds
        return ds

    def append(self, timestamp: dt.datetime, frame: dict[str, Any]) -> None:
        for key, value in frame.items():
            array = np.asarray(value)
            video_streams = list(self._iter_video_streams(key, array))
            if video_streams:
                for video_key, video_array, video_attrs in video_streams:
                    self._append_video_frame(video_key, video_array, video_attrs)
                continue

            ds = self.datasets.get(key)
            if ds is None:
                ds = self._create_dataset(key, value)

            if ds.dtype.metadata is not None and ds.dtype.metadata.get("vlen") is str:
                ds.resize(self.frame_count + 1, axis=0)
                item = value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
                ds[self.frame_count] = item
                continue

            if ds.shape[1:] != array.shape or ds.dtype != array.dtype:
                raise ValueError(
                    f"Inconsistent shape/dtype for key '{key}': "
                    f"expected shape {ds.shape[1:]}, dtype {ds.dtype}; got {array.shape}, {array.dtype}"
                )
            ds.resize(self.frame_count + 1, axis=0)
            ds[self.frame_count] = array

        self.timestamps.resize(self.frame_count + 1, axis=0)
        self.timestamps[self.frame_count] = timestamp.isoformat()
        self.frame_count += 1
        self.file.attrs["frame_count"] = self.frame_count

    def close(self) -> None:
        for key, writer in self.video_writers.items():
            writer.release()
            temp_path = self.video_tempfiles[key]
            with open(temp_path, "rb") as fh:
                video_bytes = np.frombuffer(fh.read(), dtype=np.uint8)
            ds = self.videos_group[key]
            ds.resize((video_bytes.shape[0],))
            ds[...] = video_bytes
            os.unlink(temp_path)
        self.file.flush()
        self.file.close()


def infer_video_fps(pkl_paths: list[Path]) -> float:
    if len(pkl_paths) < 2:
        return 15.0
    timestamps = [infer_timestamp_from_path(path) for path in pkl_paths]
    diffs = np.array(
        [(timestamps[i + 1] - timestamps[i]).total_seconds() for i in range(len(timestamps) - 1)],
        dtype=float,
    )
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return 15.0
    return float(np.clip(1.0 / np.median(positive), 1.0, 120.0))


def convert_trajectory(
    input_dir: Path,
    output_dir: Path,
    *,
    keep_marker_tracking: bool,
    keep_raw_rgb: bool,
    overwrite: bool,
) -> None:
    pkl_paths = [Path(p) for p in sorted(glob.glob(str(input_dir / "*.pkl")))]
    if not pkl_paths:
        print(f"Skipping {input_dir}: no PKL frames found")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "trajectory.h5"
    if output_path.exists() and not overwrite:
        print(f"Skipping {input_dir.name}: {output_path} already exists")
        return

    writer = CompactTrajectoryWriter(output_path, video_fps=infer_video_fps(pkl_paths))
    try:
        for pkl_path in pkl_paths:
            with open(pkl_path, "rb") as fh:
                frame = pickle.load(fh)
            filtered = filter_legacy_frame(
                frame,
                keep_marker_tracking=keep_marker_tracking,
                keep_raw_rgb=keep_raw_rgb,
            )
            writer.append(infer_timestamp_from_path(pkl_path), filtered)
    finally:
        writer.close()

    print(f"Converted {input_dir} -> {output_path}")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else input_root.parent / f"{input_root.name}_h5"
    )

    traj_dirs = [path for path in sorted(input_root.iterdir()) if path.is_dir()]
    if args.limit is not None:
        traj_dirs = traj_dirs[: args.limit]

    print(f"Input root:  {input_root}")
    print(f"Output root: {output_root}")
    print(f"Trajectories: {len(traj_dirs)}")

    for traj_dir in traj_dirs:
        convert_trajectory(
            traj_dir,
            output_root / traj_dir.name,
            keep_marker_tracking=args.keep_marker_tracking,
            keep_raw_rgb=args.keep_raw_rgb,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
