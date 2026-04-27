from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .transforms import clip_pi0_action


class DryRunPolicyClient:
    def __init__(self, action_dim: int = 7):
        self.action_dim = action_dim

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        return np.zeros((self.action_dim,), dtype=np.float32)


class PicklePolicyClient:
    """Minimal local policy wrapper for smoke tests.

    Real OpenPI inference should be launched from an OpenPI checkout; this class
    intentionally avoids importing OpenPI inside the robot repository.
    """

    def __init__(self, policy_path: str | Path):
        with open(policy_path, "rb") as f:
            self.policy = pickle.load(f)

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        if hasattr(self.policy, "act"):
            return np.asarray(self.policy.act(observation), dtype=np.float32)
        if callable(self.policy):
            return np.asarray(self.policy(observation), dtype=np.float32)
        raise TypeError("Loaded policy must be callable or expose act(observation)")


def safe_action(policy, observation: dict[str, Any], limits: dict[str, float]) -> np.ndarray:
    return clip_pi0_action(policy.act(observation), limits)
