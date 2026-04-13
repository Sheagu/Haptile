#!/usr/bin/env python3
"""Make a 4x5 grid video from the first 20 base_camera_rgb_1.mp4 files."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine the first 20 base_camera_rgb_1.mp4 videos into a 4x5 grid video."
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help="Root folder like exported_videos_batch containing per-trajectory subfolders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output mp4 path. Defaults to <input_root>/base_camera_rgb_1_grid_4x5.mp4",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of videos to include. Default: 20",
    )
    parser.add_argument(
        "--last",
        action="store_true",
        help="Use the last N videos instead of the first N videos.",
    )
    return parser.parse_args()


def fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    src_h, src_w = frame.shape[:2]
    if src_h == 0 or src_w == 0:
        return canvas

    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_root / "base_camera_rgb_1_grid_4x5.mp4"
    )

    video_paths = []
    for subdir in sorted(input_root.iterdir()):
        if not subdir.is_dir():
            continue
        candidate = subdir / "base_camera_rgb_1.mp4"
        if candidate.is_file():
            video_paths.append(candidate)

    if len(video_paths) < args.limit:
        raise ValueError(
            f"Found only {len(video_paths)} base_camera_rgb_1.mp4 files under {input_root}, "
            f"but need {args.limit}"
        )

    if args.last:
        video_paths = video_paths[-args.limit:]
    else:
        video_paths = video_paths[:args.limit]

    captures = [cv2.VideoCapture(str(path)) for path in video_paths]
    try:
        for path, cap in zip(video_paths, captures):
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {path}")

        widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in captures]
        heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in captures]
        frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures]
        fps_values = [cap.get(cv2.CAP_PROP_FPS) for cap in captures if cap.get(cv2.CAP_PROP_FPS) > 0]

        tile_w = max(widths)
        tile_h = max(heights)
        fps = fps_values[0] if fps_values else 15.0
        total_frames = min(frame_counts)

        rows = 4
        cols = 5
        out_size = (tile_w * cols, tile_h * rows)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            out_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open output video: {output_path}")

        try:
            for _ in range(total_frames):
                frames = []
                for cap in captures:
                    ok, frame = cap.read()
                    if not ok:
                        frames = []
                        break
                    frames.append(fit_frame(frame, tile_w, tile_h))
                if len(frames) != len(video_paths):
                    break

                row_frames = []
                for row_idx in range(rows):
                    start = row_idx * cols
                    row_frames.append(np.hstack(frames[start:start + cols]))
                writer.write(np.vstack(row_frames))
        finally:
            writer.release()
    finally:
        for cap in captures:
            cap.release()

    print(f"Created grid video: {output_path}")
    print(f"Used {len(video_paths)} videos, fps={fps:.2f}, frames={total_frames}")


if __name__ == "__main__":
    main()
