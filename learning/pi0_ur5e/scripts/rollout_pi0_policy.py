#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi0_ur5e.io_utils import write_json
from pi0_ur5e.policy_client import DryRunPolicyClient, PicklePolicyClient, safe_action
from pi0_ur5e.schema import Pi0Ur5eConfig


def main():
    parser = argparse.ArgumentParser(description="Dry-run or execute pi0 policy rollout with explicit real-robot opt-in.")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--allow-real-robot", default="false")
    args = parser.parse_args()
    cfg = Pi0Ur5eConfig()
    policy = PicklePolicyClient(args.policy) if args.policy else DryRunPolicyClient()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    real = str(args.allow_real_robot).lower() == "true"
    if real:
        raise SystemExit("Real robot rollout requires a project-specific adapter and emergency stop verification. This script currently stays dry-run.")
    records = []
    for step in range(args.steps):
        obs = {"observation.state": np.zeros((7,), dtype=np.float32)}
        action = safe_action(policy, obs, cfg.action_limits)
        print(f"step={step} action={action.tolist()}")
        records.append({"step": step, "action": action.tolist()})
    write_json(args.output_dir / "dry_run_actions.json", records)
    print(f"Dry-run actions written to {args.output_dir / 'dry_run_actions.json'}")


if __name__ == "__main__":
    main()
