#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi0_ur5e.policy_client import WebsocketPolicyClient, build_policy_observation, check_policy_server_health
from pi0_ur5e.schema import Pi0Ur5eConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one dummy UR5e observation to an OpenPI policy server.")
    parser.add_argument("--host", required=True, help="GPU computer IP/hostname running the OpenPI server.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--state-dim", type=int, default=7)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--skip-health-check", default="false")
    args = parser.parse_args()

    cfg = Pi0Ur5eConfig()
    if args.prompt is not None:
        cfg.default_prompt = args.prompt
    if str(args.skip_health_check).lower() != "true" and not check_policy_server_health(args.host, args.port):
        raise SystemExit(f"Policy server health check failed at {args.host}:{args.port}")

    client = WebsocketPolicyClient(args.host, args.port)
    obs = build_policy_observation(
        base_rgb=np.zeros((*cfg.image_size, 3), dtype=np.uint8),
        wrist_rgb=np.zeros((*cfg.image_size, 3), dtype=np.uint8),
        state=np.zeros((args.state_dim,), dtype=np.float32),
        prompt=cfg.default_prompt,
    )
    result = client.infer(obs)
    actions = np.asarray(result["actions"], dtype=np.float32)
    print(
        json.dumps(
            {
                "metadata": client.metadata,
                "action_shape": list(actions.shape),
                "first_action": actions[0].tolist() if actions.ndim > 1 else actions.tolist(),
                "policy_timing": result.get("policy_timing"),
                "server_timing": result.get("server_timing"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
