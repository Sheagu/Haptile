#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export/copy a trained pi0 policy artifact.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dst = args.output_dir / args.checkpoint.name
    if args.checkpoint.is_dir():
        shutil.copytree(args.checkpoint, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(args.checkpoint, dst)
    print(f"Exported policy artifact to {dst}")


if __name__ == "__main__":
    main()
