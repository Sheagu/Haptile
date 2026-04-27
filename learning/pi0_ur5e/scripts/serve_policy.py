#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch OpenPI's official policy server for pi0_ur5e_cup.")
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--config-name", default="pi0_ur5e_cup")
    parser.add_argument("--checkpoint-dir", required=True, type=str)
    parser.add_argument("--dataset-root", default=None, type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-prompt", default="pick up the paper cup and place it on the target")
    parser.add_argument("--record", default="false")
    parser.add_argument("--dry-run", default="false")
    args = parser.parse_args()

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("install_openpi_config.py")),
            "--openpi-root",
            str(args.openpi_root),
        ],
        check=True,
    )
    env = os.environ.copy()
    if args.dataset_root is not None:
        env["PI0_UR5E_LEROBOT_REPO_ID"] = str(args.dataset_root.resolve())

    command = [
        "uv",
        "run",
        "scripts/serve_policy.py",
        "--port",
        str(args.port),
        "--default-prompt",
        args.default_prompt,
        "--policy.config",
        args.config_name,
        "--policy.dir",
        args.checkpoint_dir,
    ]
    if str(args.record).lower() == "true":
        command.append("--record")
    print("OpenPI serve command:")
    print(" ".join(command))
    if str(args.dry_run).lower() == "true":
        return
    subprocess.run(command, cwd=args.openpi_root, env=env, check=True)


if __name__ == "__main__":
    main()
