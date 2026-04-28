#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi0_ur5e.io_utils import write_json
from pi0_ur5e.policy_client import DryRunPolicyClient, PicklePolicyClient, WebsocketPolicyClient, safe_action
from pi0_ur5e.schema import Pi0Ur5eConfig


def main():
    parser = argparse.ArgumentParser(description="Dry-run or execute pi0 policy rollout with explicit real-robot opt-in.")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--server-host", default=None, help="OpenPI policy server host/IP. Uses websocket transport.")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--allow-real-robot", default="false")
    args = parser.parse_args()
    cfg = Pi0Ur5eConfig()
    if args.server_host and args.policy:
        raise SystemExit("Use either --server-host or --policy, not both.")
    if args.server_host:
        policy = WebsocketPolicyClient(args.server_host, args.server_port)
    elif args.policy:
        policy = PicklePolicyClient(args.policy)
    else:
        policy = DryRunPolicyClient()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    real = str(args.allow_real_robot).lower() == "true"
    if real:
        raise SystemExit("Real robot rollout requires a project-specific adapter and emergency stop verification. This script currently stays dry-run.")
    records = []
    for step in range(args.steps):
        obs = {
            "base_rgb": np.zeros((*cfg.image_size, 3), dtype=np.uint8),
            "wrist_rgb": np.zeros((*cfg.image_size, 3), dtype=np.uint8),
            "state": np.zeros((7,), dtype=np.float32),
            "prompt": cfg.default_prompt,
        }
        action = safe_action(policy, obs, cfg.action_limits)
        print(f"step={step} action={action.tolist()}")
        records.append({"step": step, "action": action.tolist()})
    write_json(args.output_dir / "dry_run_actions.json", records)
    print(f"Dry-run actions written to {args.output_dir / 'dry_run_actions.json'}")


if __name__ == "__main__":
    main()
