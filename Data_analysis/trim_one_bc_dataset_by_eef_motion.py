#!/usr/bin/env python3
"""Trim one TeleUR trajectory folder by detecting motion in the EEF position.

python Data_analysis/trim_one_bc_dataset_by_eef_motion.py \
  shared/data/bc_data/rubiks_cube/0424_173508 \
  shared/data/bc_data/rubiks_cube_trimmed/0424_173508 \
  --dry-run


"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from trim_bc_data_by_eef_motion import (
    TrimResult,
    copy_extra_files,
    detect_trim_range,
    write_trimmed_freq,
    write_trimmed_h5,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trim a single trajectory folder using EEF position motion. "
            "Only leading and trailing still frames are removed; pauses in the middle are kept."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input trajectory folder containing trajectory.h5 and optionally freq.txt.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output trajectory folder for the trimmed trajectory.h5 and freq.txt.",
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
        "--dry-run",
        action="store_true",
        help="Print the proposed trim range without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing output folder.",
    )
    parser.add_argument(
        "--copy-extra-files",
        action="store_true",
        help="Copy files other than trajectory.h5 and freq.txt into the output folder.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write a JSON trim report.",
    )
    return parser.parse_args()


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
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    input_h5 = input_dir / "trajectory.h5"
    issues: list[str] = []

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_h5.is_file():
        raise FileNotFoundError(f"trajectory.h5 not found: {input_h5}")

    start, end, original_frames = detect_trim_range(
        input_h5,
        eef_key=args.eef_key,
        position_dims=args.position_dims,
        motion_threshold=args.motion_threshold,
        min_motion_run=args.min_motion_run,
        padding=args.padding,
    )
    trimmed_frames = end - start

    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}")
    print(
        "Motion:     "
        f"key=/frames/{args.eef_key}, threshold={args.motion_threshold}, "
        f"min_run={args.min_motion_run}, padding={args.padding}"
    )
    print(f"Dry run:    {args.dry_run}")
    print()

    if not args.dry_run:
        if output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"Output directory already exists: {output_dir}")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            write_trimmed_h5(input_h5, output_dir / "trajectory.h5", start, end, original_frames)
            write_trimmed_freq(input_dir / "freq.txt", output_dir / "freq.txt", start, end, original_frames)
            if args.copy_extra_files:
                copy_extra_files(input_dir, output_dir)
        except Exception as exc:
            issues.append(str(exc))

    result = TrimResult(
        name=input_dir.name,
        input_h5=str(input_h5),
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
    print_result(result)

    if args.json is not None:
        args.json.expanduser().resolve().write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"JSON: {args.json.expanduser().resolve()}")

    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
