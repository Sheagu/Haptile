#!/usr/bin/env python3
"""Check TeleUR H5 trajectories for frame-count mismatches and corrupt videos.

python Data_analysis/check_bc_data_integrity.py shared/data/bc_data/rubiks_cube

检查这些问题：

trajectory.h5 是否能打开
root attr frame_count 是否存在、是否有效
/timestamps 长度是否和 frame_count 一致
/frames/* 每个 dataset 第一维是否和 frame_count 一致
/videos/* 是否缺少常见 camera stream
embedded mp4 是否能被 OpenCV 打开
mp4 解码出来的帧数是否和 frame_count 一致
timestamp 是否严格递增
可选检查 /frames 里的 NaN/Inf

"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import cv2
except ImportError:  # Video checks can be skipped if OpenCV is unavailable.
    cv2 = None


DEFAULT_EXPECTED_VIDEO_STREAMS = [
    "base_camera_rgb_0",
    "base_camera_rgb_1",
    "tactile_left_rgb",
    "tactile_right_rgb",
    "base_camera_depth_0",
    "base_camera_depth_1",
]


@dataclass
class DatasetInfo:
    name: str
    h5_path: str
    frame_count: int | None = None
    timestamps: int | None = None
    frame_datasets: dict[str, int] = field(default_factory=dict)
    video_frames: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check trajectory.h5 files under an input data root for frame-count mismatches, missing streams, "
            "timestamp problems, and unreadable embedded mp4 videos."
        )
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help="Input folder containing trajectory subfolders.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search trajectory.h5 files recursively instead of only direct child folders.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Check only the first N trajectories.",
    )
    parser.add_argument(
        "--expected-video-stream",
        action="append",
        default=None,
        help=(
            "Expected stream name under /videos. Can be repeated. "
            "Defaults to the usual TeleUR camera streams."
        ),
    )
    parser.add_argument(
        "--no-video-check",
        action="store_true",
        help="Do not decode embedded mp4 bytes with OpenCV.",
    )
    parser.add_argument(
        "--video-frame-tolerance",
        type=int,
        default=0,
        help="Allowed absolute difference between H5 frame_count and decoded video frames.",
    )
    parser.add_argument(
        "--check-finite",
        action="store_true",
        help="Scan numeric /frames datasets for NaN/Inf values.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write a JSON report.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Always exit with code 0 even if issues are found.",
    )
    return parser.parse_args()


def decode_h5_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def find_h5_files(data_root: Path, recursive: bool) -> list[Path]:
    pattern = "**/trajectory.h5" if recursive else "*/trajectory.h5"
    return sorted(data_root.glob(pattern))


def read_video_frame_count(video_bytes: np.ndarray, suffix: str = ".mp4") -> tuple[int | None, str | None]:
    if cv2 is None:
        return None, "OpenCV is not installed; cannot decode embedded videos"

    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(np.asarray(video_bytes, dtype=np.uint8).tobytes())
        tmp.flush()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            return None, "OpenCV could not open embedded mp4 bytes"

        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, _ = cap.read()
            if not ok:
                return frame_count, "OpenCV opened video but could not read the first frame"
            return frame_count, None
        finally:
            cap.release()


def parse_timestamps(raw_timestamps: np.ndarray) -> tuple[list[dt.datetime], list[str]]:
    timestamps: list[dt.datetime] = []
    problems: list[str] = []

    for idx, value in enumerate(raw_timestamps):
        text = decode_h5_string(value)
        try:
            timestamps.append(dt.datetime.fromisoformat(text))
        except ValueError:
            problems.append(f"timestamps[{idx}] is not ISO format: {text!r}")

    return timestamps, problems


def check_timestamps(info: DatasetInfo, timestamps_ds: h5py.Dataset, expected_frames: int | None) -> None:
    timestamp_count = int(timestamps_ds.shape[0])
    info.timestamps = timestamp_count

    if expected_frames is not None and timestamp_count != expected_frames:
        info.issues.append(
            f"/timestamps length {timestamp_count} != frame_count attr {expected_frames}"
        )

    timestamps, problems = parse_timestamps(timestamps_ds[()])
    info.issues.extend(problems[:5])
    if len(problems) > 5:
        info.issues.append(f"... {len(problems) - 5} more timestamp parse errors")

    if len(timestamps) >= 2:
        non_increasing = []
        intervals = []
        for idx, (prev, cur) in enumerate(zip(timestamps, timestamps[1:]), start=1):
            delta = (cur - prev).total_seconds()
            intervals.append(delta)
            if delta <= 0:
                non_increasing.append(idx)
        if non_increasing:
            info.issues.append(
                "timestamps are not strictly increasing at indices "
                + ", ".join(map(str, non_increasing[:10]))
            )

        if intervals:
            avg_dt = sum(intervals) / len(intervals)
            max_dt = max(intervals)
            if avg_dt > 0 and max_dt > max(1.0, avg_dt * 5.0):
                info.warnings.append(
                    f"large timestamp gap detected: max_dt={max_dt:.3f}s, avg_dt={avg_dt:.3f}s"
                )


def check_freq_file(info: DatasetInfo, dataset_dir: Path, expected_frames: int | None) -> None:
    freq_path = dataset_dir / "freq.txt"
    if not freq_path.is_file() or expected_frames is None:
        return

    entries = 0
    for line in freq_path.read_text(encoding="utf-8", errors="replace").splitlines():
        left, sep, right = line.partition(":")
        if sep and left.strip().isdigit():
            try:
                float(right.strip())
            except ValueError:
                continue
            entries += 1

    # Some collection runs write one extra frequency sample at shutdown.
    if entries and entries not in {expected_frames, expected_frames + 1}:
        info.warnings.append(f"freq.txt has {entries} per-frame entries, frame_count is {expected_frames}")


def check_frames_group(
    info: DatasetInfo,
    frames_group: h5py.Group,
    expected_frames: int | None,
    check_finite: bool,
) -> None:
    for key, ds in sorted(frames_group.items()):
        if not isinstance(ds, h5py.Dataset):
            continue
        if len(ds.shape) == 0:
            info.warnings.append(f"/frames/{key} is scalar; expected first dimension to be frames")
            continue

        length = int(ds.shape[0])
        info.frame_datasets[key] = length
        if expected_frames is not None and length != expected_frames:
            info.issues.append(f"/frames/{key} length {length} != frame_count attr {expected_frames}")

        if check_finite and np.issubdtype(ds.dtype, np.number):
            data = ds[()]
            if not np.isfinite(data).all():
                bad = int(np.size(data) - np.count_nonzero(np.isfinite(data)))
                info.issues.append(f"/frames/{key} contains {bad} NaN/Inf value(s)")


def check_videos_group(
    info: DatasetInfo,
    videos_group: h5py.Group,
    expected_frames: int | None,
    expected_video_streams: list[str],
    do_video_check: bool,
    tolerance: int,
) -> None:
    present_streams = {key for key, obj in videos_group.items() if isinstance(obj, h5py.Dataset)}
    for key in expected_video_streams:
        if key not in present_streams:
            info.issues.append(f"missing /videos/{key}")

    for key, ds in sorted(videos_group.items()):
        if not isinstance(ds, h5py.Dataset):
            continue
        if ds.dtype != np.dtype("uint8") or len(ds.shape) != 1:
            info.issues.append(f"/videos/{key} has unexpected shape/dtype: {ds.shape}, {ds.dtype}")
            continue
        if int(ds.shape[0]) == 0:
            info.issues.append(f"/videos/{key} is empty")
            continue

        fps = float(ds.attrs.get("fps", 0.0))
        width = int(ds.attrs.get("width", 0))
        height = int(ds.attrs.get("height", 0))
        if fps <= 0:
            info.warnings.append(f"/videos/{key} has invalid fps attr: {fps}")
        if width <= 0 or height <= 0:
            info.warnings.append(f"/videos/{key} has invalid size attrs: width={width}, height={height}")

        if not do_video_check:
            continue

        decoded_frames, error = read_video_frame_count(ds[()])
        if error is not None:
            info.issues.append(f"/videos/{key}: {error}")
        if decoded_frames is None:
            continue

        info.video_frames[key] = decoded_frames
        if expected_frames is not None and abs(decoded_frames - expected_frames) > tolerance:
            info.issues.append(
                f"/videos/{key} decoded frames {decoded_frames} != frame_count attr {expected_frames}"
            )


def check_one_h5(
    h5_path: Path,
    expected_video_streams: list[str],
    do_video_check: bool,
    video_frame_tolerance: int,
    check_finite: bool,
) -> DatasetInfo:
    dataset_dir = h5_path.parent
    info = DatasetInfo(name=dataset_dir.name, h5_path=str(h5_path))

    try:
        with h5py.File(h5_path, "r") as f:
            raw_frame_count = f.attrs.get("frame_count")
            if raw_frame_count is None:
                info.issues.append("missing root attr frame_count")
            else:
                info.frame_count = int(raw_frame_count)
                if info.frame_count <= 0:
                    info.issues.append(f"invalid frame_count attr: {info.frame_count}")

            video_fps = float(f.attrs.get("video_fps", 0.0))
            if not math.isfinite(video_fps) or video_fps <= 0:
                info.warnings.append(f"invalid root video_fps attr: {video_fps}")

            if "timestamps" not in f:
                info.issues.append("missing /timestamps")
            elif not isinstance(f["timestamps"], h5py.Dataset):
                info.issues.append("/timestamps is not a dataset")
            else:
                check_timestamps(info, f["timestamps"], info.frame_count)

            if "frames" not in f or not isinstance(f["frames"], h5py.Group):
                info.issues.append("missing /frames group")
            else:
                check_frames_group(info, f["frames"], info.frame_count, check_finite)

            if "videos" not in f or not isinstance(f["videos"], h5py.Group):
                info.issues.append("missing /videos group")
            else:
                check_videos_group(
                    info,
                    f["videos"],
                    info.frame_count,
                    expected_video_streams,
                    do_video_check,
                    video_frame_tolerance,
                )

        check_freq_file(info, dataset_dir, info.frame_count)
    except OSError as exc:
        info.issues.append(f"failed to open H5 file: {exc}")

    return info


def print_dataset_result(info: DatasetInfo) -> None:
    status = "OK" if info.ok else "BAD"
    details = []
    if info.frame_count is not None:
        details.append(f"frames={info.frame_count}")
    if info.timestamps is not None:
        details.append(f"timestamps={info.timestamps}")
    if info.video_frames:
        unique_video_counts = sorted(set(info.video_frames.values()))
        details.append(f"video_frames={unique_video_counts}")
    suffix = f" ({', '.join(details)})" if details else ""
    print(f"[{status}] {info.name}{suffix}")

    for issue in info.issues:
        print(f"  ISSUE: {issue}")
    for warning in info.warnings:
        print(f"  WARN:  {warning}")


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    expected_video_streams = (
        args.expected_video_stream
        if args.expected_video_stream is not None
        else DEFAULT_EXPECTED_VIDEO_STREAMS
    )
    do_video_check = not args.no_video_check
    if do_video_check and cv2 is None:
        print("Warning: OpenCV is not installed, video decode checks will be skipped.", file=sys.stderr)
        do_video_check = False

    h5_files = find_h5_files(data_root, args.recursive)
    if args.limit is not None:
        h5_files = h5_files[: args.limit]

    print(f"Data root: {data_root}")
    print(f"Trajectories: {len(h5_files)}")
    print(f"Video decode check: {'on' if do_video_check else 'off'}")
    print()

    results: list[DatasetInfo] = []
    for h5_path in h5_files:
        info = check_one_h5(
            h5_path,
            expected_video_streams=expected_video_streams,
            do_video_check=do_video_check,
            video_frame_tolerance=args.video_frame_tolerance,
            check_finite=args.check_finite,
        )
        results.append(info)
        print_dataset_result(info)

    bad = [info for info in results if info.issues]
    warned = [info for info in results if info.warnings]
    total_frames = sum(info.frame_count or 0 for info in results)

    print()
    print("Summary")
    print(f"  Checked:  {len(results)}")
    print(f"  OK:       {len(results) - len(bad)}")
    print(f"  BAD:      {len(bad)}")
    print(f"  Warnings: {len(warned)}")
    print(f"  Frames:   {total_frames}")

    if args.json is not None:
        report = {
            "data_root": str(data_root),
            "checked": len(results),
            "bad": len(bad),
            "warnings": len(warned),
            "total_frames": total_frames,
            "results": [asdict(info) for info in results],
        }
        args.json.expanduser().resolve().write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  JSON:     {args.json.expanduser().resolve()}")

    if bad and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
