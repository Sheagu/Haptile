#!/usr/bin/env python3
"""Trim TeleUR trajectory folders by detecting motion in the EEF position.
基于 /frames/ee_pos_quat 的前三维位置计算相邻帧位移，只找第一次连续移动和最后一次连续移动，裁掉头尾，中间暂停会保留

python Data_analysis/trim_bc_data_by_eef_motion.py \
  shared/data/bc_data/wipe_board \
  shared/data/bc_data/wipe_board_trimmed
  
入参可选：--dry-run，只打印裁切范围，不生成文件

裁切考虑的默认参数：
motion_threshold = 0.001  # 1mm/frame
min_motion_run = 3        # 连续 3 个移动 interval 才算移动
padding = 5               # 前后各多保留 5 帧

裁切时会同步处理：
trajectory.h5 里的 /frames/*
/timestamps
/videos/* 内嵌 mp4，重新裁成相同帧数
freq.txt，重新编号并重新计算 Average/Max/Min/Std FPS
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class TrimResult:
    name: str
    input_h5: str
    output_dir: str
    original_frames: int
    trimmed_frames: int
    start: int
    end: int
    removed_head: int
    removed_tail: int
    dry_run: bool
    issues: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trim each trajectory.h5 under an input folder using EEF position motion. "
            "Only leading and trailing still frames are removed; pauses in the middle are kept."
        )
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help="Input folder containing trajectory subfolders, e.g. shared/data/bc_data/rubiks_cube",
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Output folder for trimmed trajectory subfolders.",
    )
    parser.add_argument(
        "--eef-key",
        default="ee_pos_quat",
        help="Dataset under /frames used for EEF pose. Default: ee_pos_quat",
    )
    parser.add_argument(
        "--position-dims",
        type=int,
        default=3,
        help="Use the first N columns of --eef-key as position. Default: 3",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.001,
        help="Per-frame EEF position displacement threshold for motion. Default: 0.001 meter",
    )
    parser.add_argument(
        "--min-motion-run",
        type=int,
        default=3,
        help="Require this many consecutive moving frame intervals. Default: 3",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=5,
        help="Extra frames to keep before first motion and after last motion. Default: 5",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Trim only the first N trajectories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print proposed trim ranges without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output trajectory folders.",
    )
    parser.add_argument(
        "--copy-extra-files",
        action="store_true",
        help="Copy files other than trajectory.h5 and freq.txt into each output folder.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write a JSON trim report.",
    )
    return parser.parse_args()


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def find_h5_files(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("*/trajectory.h5"))


def find_motion_run(mask: np.ndarray, min_run: int, reverse: bool = False) -> tuple[int, int] | None:
    if min_run <= 1:
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            return None
        idx = int(indices[-1] if reverse else indices[0])
        return idx, idx + 1

    indices = range(len(mask) - min_run, -1, -1) if reverse else range(0, len(mask) - min_run + 1)
    for idx in indices:
        if bool(mask[idx : idx + min_run].all()):
            return idx, idx + min_run
    return None


def detect_trim_range(
    h5_path: Path,
    eef_key: str,
    position_dims: int,
    motion_threshold: float,
    min_motion_run: int,
    padding: int,
) -> tuple[int, int, int]:
    with h5py.File(h5_path, "r") as f:
        dataset_path = f"frames/{eef_key}"
        if dataset_path not in f:
            raise KeyError(f"Missing /{dataset_path}")

        eef = np.asarray(f[dataset_path])
        frame_count = int(f.attrs.get("frame_count", eef.shape[0]))

    if eef.ndim != 2 or eef.shape[1] < position_dims:
        raise ValueError(
            f"/frames/{eef_key} must have shape (N, >= {position_dims}); got {eef.shape}"
        )
    if frame_count != eef.shape[0]:
        raise ValueError(f"frame_count attr {frame_count} != /frames/{eef_key} length {eef.shape[0]}")
    if frame_count <= 1:
        return 0, frame_count, frame_count

    positions = eef[:, :position_dims].astype(np.float64)
    displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    moving = displacements > motion_threshold

    first_run = find_motion_run(moving, min_motion_run, reverse=False)
    last_run = find_motion_run(moving, min_motion_run, reverse=True)
    if first_run is None or last_run is None:
        return 0, frame_count, frame_count

    start = max(0, first_run[0] - padding)
    # last_run[1] is an exclusive index in the displacement array. Add one more
    # because displacement i connects frame i -> frame i + 1.
    end = min(frame_count, last_run[1] + 1 + padding)
    if end <= start:
        return 0, frame_count, frame_count
    return start, end, frame_count


def adjusted_chunks(src: h5py.Dataset, shape: tuple[int, ...]) -> tuple[int, ...] | None:
    if src.chunks is None or not shape:
        return None
    return tuple(max(1, min(chunk, dim)) for chunk, dim in zip(src.chunks, shape))


def create_dataset_like(dst_group: h5py.Group, name: str, src: h5py.Dataset, data: Any) -> h5py.Dataset:
    kwargs: dict[str, Any] = {"dtype": src.dtype}
    shape = np.shape(data)
    chunks = adjusted_chunks(src, shape)
    if chunks is not None:
        kwargs["chunks"] = chunks
    if src.compression is not None:
        kwargs["compression"] = src.compression
        kwargs["compression_opts"] = src.compression_opts
    if src.shuffle:
        kwargs["shuffle"] = src.shuffle
    if src.fletcher32:
        kwargs["fletcher32"] = src.fletcher32

    dst = dst_group.create_dataset(name, data=data, **kwargs)
    copy_attrs(src.attrs, dst.attrs)
    return dst


def read_trimmed_video_bytes(src: h5py.Dataset, start: int, end: int) -> bytes:
    if cv2 is None:
        raise RuntimeError("OpenCV is required to trim embedded mp4 videos")

    with tempfile.NamedTemporaryFile(suffix=".mp4") as src_tmp, tempfile.NamedTemporaryFile(suffix=".mp4") as dst_tmp:
        src_tmp.write(np.asarray(src, dtype=np.uint8).tobytes())
        src_tmp.flush()

        cap = cv2.VideoCapture(src_tmp.name)
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open embedded mp4 video")

        fps = float(src.attrs.get("fps", cap.get(cv2.CAP_PROP_FPS) or 15.0))
        width = int(src.attrs.get("width", cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(src.attrs.get("height", cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(
            dst_tmp.name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0 else 15.0,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("OpenCV could not create trimmed mp4 video")

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(start, end):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Video ended before requested frame {end}")
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
        finally:
            writer.release()
            cap.release()

        return Path(dst_tmp.name).read_bytes()


def copy_trimmed_group(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    start: int,
    end: int,
    original_frames: int,
) -> None:
    copy_attrs(src_group.attrs, dst_group.attrs)
    for key, item in src_group.items():
        if isinstance(item, h5py.Group):
            child = dst_group.create_group(key)
            copy_trimmed_group(item, child, start, end, original_frames)
            continue

        if isinstance(item, h5py.Dataset) and item.shape and item.shape[0] == original_frames:
            data = item[start:end]
        else:
            data = item[()]
        create_dataset_like(dst_group, key, item, data)


def copy_trimmed_videos(src_group: h5py.Group, dst_group: h5py.Group, start: int, end: int) -> None:
    copy_attrs(src_group.attrs, dst_group.attrs)
    for key, item in src_group.items():
        if isinstance(item, h5py.Group):
            child = dst_group.create_group(key)
            copy_trimmed_videos(item, child, start, end)
            continue
        if not isinstance(item, h5py.Dataset):
            continue
        data = np.frombuffer(read_trimmed_video_bytes(item, start, end), dtype=np.uint8)
        dst = dst_group.create_dataset(key, data=data, dtype=np.uint8, maxshape=(None,))
        copy_attrs(item.attrs, dst.attrs)


def write_trimmed_h5(input_h5: Path, output_h5: Path, start: int, end: int, original_frames: int) -> None:
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_h5, "r") as src, h5py.File(output_h5, "w") as dst:
        copy_attrs(src.attrs, dst.attrs)
        dst.attrs["frame_count"] = end - start

        for key, item in src.items():
            if key == "videos" and isinstance(item, h5py.Group):
                videos_dst = dst.create_group(key)
                copy_trimmed_videos(item, videos_dst, start, end)
            elif isinstance(item, h5py.Group):
                group_dst = dst.create_group(key)
                copy_trimmed_group(item, group_dst, start, end, original_frames)
            elif isinstance(item, h5py.Dataset) and item.shape and item.shape[0] == original_frames:
                create_dataset_like(dst, key, item, item[start:end])
            elif isinstance(item, h5py.Dataset):
                create_dataset_like(dst, key, item, item[()])


def parse_freq_lines(freq_path: Path) -> tuple[list[str], list[tuple[int, str]]]:
    header: list[str] = []
    entries: list[tuple[int, str]] = []
    for line in freq_path.read_text(encoding="utf-8", errors="replace").splitlines():
        left, sep, right = line.partition(":")
        if sep and left.strip().isdigit():
            entries.append((int(left.strip()), right.strip()))
        else:
            header.append(line)
    return header, entries


def write_trimmed_freq(
    input_freq: Path,
    output_freq: Path,
    start: int,
    end: int,
    original_frames: int,
) -> None:
    if not input_freq.is_file():
        return

    header, entries = parse_freq_lines(input_freq)
    if len(entries) < original_frames:
        raise ValueError(f"freq.txt has {len(entries)} frame entries, expected at least {original_frames}")

    trimmed_values = [value for _, value in entries[start:end]]
    numeric_values = np.array([float(value) for value in trimmed_values], dtype=np.float64)

    output_lines = [
        f"Average FPS: {float(numeric_values.mean()) if len(numeric_values) else 0.0}",
        f"Max FPS: {float(numeric_values.max()) if len(numeric_values) else 0.0}",
        f"Min FPS: {float(numeric_values.min()) if len(numeric_values) else 0.0}",
        f"Std FPS: {float(numeric_values.std()) if len(numeric_values) else 0.0}",
        "",
    ]
    output_lines.extend(f"{idx}: {value}" for idx, value in enumerate(trimmed_values))

    # Preserve non-standard header lines after the standard summary if present.
    extra_header = [line for line in header if not line.startswith(("Average FPS:", "Max FPS:", "Min FPS:", "Std FPS:")) and line]
    if extra_header:
        output_lines.extend(["", *extra_header])

    output_freq.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def copy_extra_files(input_dir: Path, output_dir: Path) -> None:
    for path in input_dir.iterdir():
        if path.name in {"trajectory.h5", "freq.txt"}:
            continue
        dst = output_dir / path.name
        if path.is_dir():
            shutil.copytree(path, dst, dirs_exist_ok=True)
        elif path.is_file():
            shutil.copy2(path, dst)


def trim_one_dataset(args: argparse.Namespace, h5_path: Path) -> TrimResult:
    input_dir = h5_path.parent
    output_dir = args.output_root / input_dir.name
    issues: list[str] = []

    start, end, original_frames = detect_trim_range(
        h5_path,
        eef_key=args.eef_key,
        position_dims=args.position_dims,
        motion_threshold=args.motion_threshold,
        min_motion_run=args.min_motion_run,
        padding=args.padding,
    )
    trimmed_frames = end - start

    if not args.dry_run:
        if output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"Output directory already exists: {output_dir}")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            write_trimmed_h5(h5_path, output_dir / "trajectory.h5", start, end, original_frames)
            write_trimmed_freq(input_dir / "freq.txt", output_dir / "freq.txt", start, end, original_frames)
            if args.copy_extra_files:
                copy_extra_files(input_dir, output_dir)
        except Exception as exc:
            issues.append(str(exc))

    return TrimResult(
        name=input_dir.name,
        input_h5=str(h5_path),
        output_dir=str(output_dir),
        original_frames=original_frames,
        trimmed_frames=trimmed_frames,
        start=start,
        end=end,
        removed_head=start,
        removed_tail=original_frames - end,
        dry_run=bool(args.dry_run),
        issues=issues,
    )


def print_result(result: TrimResult) -> None:
    status = "BAD" if result.issues else "OK"
    print(
        f"[{status}] {result.name}: "
        f"{result.original_frames} -> {result.trimmed_frames} frames, "
        f"keep [{result.start}:{result.end}], "
        f"remove head={result.removed_head}, tail={result.removed_tail}"
    )
    for issue in result.issues:
        print(f"  ISSUE: {issue}")


def main() -> int:
    args = parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")
    if not args.dry_run and cv2 is None:
        raise RuntimeError("OpenCV is required because embedded videos must be trimmed to keep H5 frame alignment")

    h5_files = find_h5_files(args.input_root)
    if args.limit is not None:
        h5_files = h5_files[: args.limit]

    print(f"Input root:  {args.input_root}")
    print(f"Output root: {args.output_root}")
    print(f"Datasets:    {len(h5_files)}")
    print(
        "Motion:      "
        f"key=/frames/{args.eef_key}, threshold={args.motion_threshold}, "
        f"min_run={args.min_motion_run}, padding={args.padding}"
    )
    print(f"Dry run:     {args.dry_run}")
    print()

    results: list[TrimResult] = []
    for h5_path in h5_files:
        try:
            result = trim_one_dataset(args, h5_path)
        except Exception as exc:
            input_dir = h5_path.parent
            result = TrimResult(
                name=input_dir.name,
                input_h5=str(h5_path),
                output_dir=str(args.output_root / input_dir.name),
                original_frames=0,
                trimmed_frames=0,
                start=0,
                end=0,
                removed_head=0,
                removed_tail=0,
                dry_run=bool(args.dry_run),
                issues=[str(exc)],
            )
        results.append(result)
        print_result(result)

    bad = [result for result in results if result.issues]
    print()
    print("Summary")
    print(f"  Checked:       {len(results)}")
    print(f"  Successful:    {len(results) - len(bad)}")
    print(f"  Failed:        {len(bad)}")
    print(f"  Removed head:  {sum(r.removed_head for r in results)}")
    print(f"  Removed tail:  {sum(r.removed_tail for r in results)}")

    if args.json is not None:
        args.json.expanduser().resolve().write_text(
            json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  JSON:          {args.json.expanduser().resolve()}")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
