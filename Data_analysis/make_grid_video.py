#!/usr/bin/env python3
"""Tile multiple mp4 clips into a single grid video using OpenCV."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tile N mp4 clips into a COLS x ROWS grid video."
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help="Root directory containing per-trajectory exported_videos_batch subfolders.",
    )
    parser.add_argument(
        "--stream",
        default="base_camera_rgb_1.mp4",
        help="Video filename to pick from each trajectory folder (default: base_camera_rgb_1.mp4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("grid_output.mp4"),
        help="Output video path (default: grid_output.mp4).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=5,
        help="Number of columns in the grid (default: 5).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=4,
        help="Number of rows in the grid (default: 4).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output FPS. Defaults to the FPS of the first clip.",
    )
    parser.add_argument(
        "--cell-width",
        type=int,
        default=None,
        help="Width of each cell in pixels. Defaults to the width of the first clip.",
    )
    parser.add_argument(
        "--cell-height",
        type=int,
        default=None,
        help="Height of each cell in pixels. Defaults to the height of the first clip.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N clips found (overrides --cols/--rows count).",
    )
    return parser.parse_args()


def find_clips(data_root: Path, stream: str, limit: int | None) -> list[Path]:
    clips = sorted(data_root.rglob(stream))
    if not clips:
        raise FileNotFoundError(
            f"No '{stream}' files found under {data_root}.\n"
            "Run batch_export_h5_videos.py first to export the videos."
        )
    if limit is not None:
        clips = clips[:limit]
    return clips


def make_grid_video(
    clips: list[Path],
    output: Path,
    cols: int,
    rows: int,
    fps: float | None,
    cell_w: int | None,
    cell_h: int | None,
) -> None:
    n = cols * rows
    if len(clips) < n:
        print(f"Warning: only {len(clips)} clips found, need {n} for a {cols}x{rows} grid.")
        print("Remaining cells will be blank.")

    caps = [cv2.VideoCapture(str(p)) for p in clips[:n]]

    # Derive parameters from first clip
    ref = caps[0]
    if fps is None:
        fps = ref.get(cv2.CAP_PROP_FPS) or 10.0
    if cell_w is None:
        cell_w = int(ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    if cell_h is None:
        cell_h = int(ref.get(cv2.CAP_PROP_FRAME_HEIGHT))

    grid_w = cols * cell_w
    grid_h = rows * cell_h

    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (grid_w, grid_h))

    # last_frames holds the most recent valid frame for each clip
    last_frames: list[np.ndarray | None] = [None] * len(caps)
    # pad slots for clips fewer than grid cells
    blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

    print(f"Grid size : {cols} cols x {rows} rows  ({grid_w}x{grid_h} px)")
    print(f"FPS       : {fps}")
    print(f"Clips     : {len(clips[:n])}")
    print(f"Output    : {output}")

    frame_idx = 0
    while True:
        cells = []
        any_alive = False
        for i, cap in enumerate(caps):
            ok, frame = cap.read()
            if ok:
                frame = cv2.resize(frame, (cell_w, cell_h))
                last_frames[i] = frame
                any_alive = True
            else:
                # hold last frame instead of going black
                frame = last_frames[i] if last_frames[i] is not None else blank.copy()
            cells.append(frame)

        # Pad to full grid if fewer clips than cells
        while len(cells) < n:
            cells.append(blank.copy())

        if not any_alive:
            break

        rows_imgs = []
        for r in range(rows):
            row_cells = cells[r * cols : r * cols + cols]
            rows_imgs.append(np.concatenate(row_cells, axis=1))
        grid_frame = np.concatenate(rows_imgs, axis=0)

        writer.write(grid_frame)
        frame_idx += 1

    for cap in caps:
        cap.release()
    writer.release()
    print(f"Done. {frame_idx} frames written -> {output}")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()

    n = args.limit if args.limit is not None else args.cols * args.rows
    clips = find_clips(data_root, args.stream, limit=n)

    print(f"Found {len(clips)} clip(s) matching '{args.stream}'")
    for i, p in enumerate(clips, 1):
        print(f"  [{i:02d}] {p}")
    print()

    make_grid_video(
        clips=clips,
        output=args.output.expanduser().resolve(),
        cols=args.cols,
        rows=args.rows,
        fps=args.fps,
        cell_w=args.cell_width,
        cell_h=args.cell_height,
    )


if __name__ == "__main__":
    main()
