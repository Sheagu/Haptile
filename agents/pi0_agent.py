from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from agents.dp_agent import get_eef_pose, get_reset_joints
from learning.pi0_ur5e.pi0_ur5e.policy_client import (
    WebsocketPolicyClient,
    build_policy_observation,
)
from learning.pi0_ur5e.pi0_ur5e.transforms import clip_pi0_action
from learning.pi0_ur5e.pi0_ur5e.schema import Pi0Ur5eConfig


class Pi0Agent:
    """Single-arm pi0 deployment agent for run_env.py.

    The trained pi0_ur5e_cup policy predicts EEF deltas:
    [dx, dy, dz, droll, dpitch, dyaw, gripper].
    """

    def __init__(
        self,
        host: str,
        port: int = 8000,
        *,
        predict_eef_delta: bool = True,
        state_dim: int = 7,
        image_size: tuple[int, int] = (224, 224),
        base_camera_index: int = 1,
        wrist_camera_index: int = 0,
        prompt: str | None = None,
        gripper_min: float = 0.0,
        gripper_max: float = 1.0,
        reject_translation_delta: float = 0.25,
    ):
        self.client = WebsocketPolicyClient(host, port)
        self.config = Pi0Ur5eConfig()
        self.predict_eef_delta = predict_eef_delta
        self.state_dim = state_dim
        self.image_size = image_size
        self.base_camera_index = base_camera_index
        self.wrist_camera_index = wrist_camera_index
        self.prompt = prompt or self.config.default_prompt
        self.config.action_limits["min_gripper"] = gripper_min
        self.config.action_limits["max_gripper"] = gripper_max
        self.reject_translation_delta = reject_translation_delta
        self.control = get_reset_joints(ur_eef=predict_eef_delta)
        self.trigger_state = True
        self.control_active = True

        metadata_action_format = self.client.metadata.get("action_format")
        if metadata_action_format and metadata_action_format != "ee_delta_6d_gripper":
            print(
                "[pi0] warning: server metadata action_format="
                f"{metadata_action_format!r}; this agent expects 'ee_delta_6d_gripper'."
            )

    def reset_temporal_state(self) -> None:
        self.control = get_reset_joints(ur_eef=self.predict_eef_delta)
        reset = getattr(self.client, "reset", None)
        if callable(reset):
            reset()

    def _select_image(self, obs: Dict[str, Any], canonical: str) -> np.ndarray:
        direct_keys = {
            "base": ("base_rgb", "third_view_camera_rgb", "base_camera_rgb_1"),
            "wrist": ("wrist_rgb", "wrist_camera_rgb", "camera_wrist_rgb", "base_camera_rgb_0"),
        }[canonical]
        for key in direct_keys:
            if key in obs and obs[key] is not None:
                return self._resize_image(obs[key])

        if "base_camera_rgb" not in obs or obs["base_camera_rgb"] is None:
            raise KeyError(f"Missing image for {canonical}; expected one of {direct_keys} or base_camera_rgb")

        images = np.asarray(obs["base_camera_rgb"])
        if images.ndim == 3:
            return self._resize_image(images)
        if images.ndim != 4:
            raise ValueError(f"Expected base_camera_rgb shape [N,H,W,C] or [H,W,C], got {images.shape}")

        index = self.base_camera_index if canonical == "base" else self.wrist_camera_index
        if index >= images.shape[0]:
            index = min(images.shape[0] - 1, 0)
        return self._resize_image(images[index])

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.moveaxis(image, 0, -1)
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0) * 255.0
        image = image.astype(np.uint8)
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size[::-1], interpolation=cv2.INTER_AREA)
        return image

    def _state(self, obs: Dict[str, Any]) -> np.ndarray:
        if "ee_pos_quat" in obs and obs["ee_pos_quat"] is not None:
            state = np.asarray(obs["ee_pos_quat"], dtype=np.float32).reshape(-1)[:6]
        elif "joint_positions" in obs and obs["joint_positions"] is not None:
            state = np.asarray(obs["joint_positions"], dtype=np.float32).reshape(-1)[:6]
        else:
            raise KeyError("pi0 requires obs['ee_pos_quat'] or obs['joint_positions']")

        if self.state_dim <= 6:
            return state[: self.state_dim].astype(np.float32)

        gripper = 0.0
        if "gripper_position" in obs and obs["gripper_position"] is not None:
            gripper = float(np.asarray(obs["gripper_position"]).reshape(-1)[0])
        elif "joint_positions" in obs and obs["joint_positions"] is not None:
            joints = np.asarray(obs["joint_positions"], dtype=np.float32).reshape(-1)
            if len(joints) > 6:
                gripper = float(joints[-1])
        return np.concatenate([state, np.asarray([gripper], dtype=np.float32)])[: self.state_dim].astype(np.float32)

    def _policy_observation(self, obs: Dict[str, Any]) -> dict[str, Any]:
        return build_policy_observation(
            base_rgb=self._select_image(obs, "base"),
            wrist_rgb=self._select_image(obs, "wrist"),
            state=self._state(obs),
            prompt=self.prompt,
        )

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        raw_action = np.asarray(self.client.act(self._policy_observation(obs)), dtype=np.float32).reshape(-1)
        if self.predict_eef_delta and raw_action.shape[0] >= 3:
            max_xyz = float(np.max(np.abs(raw_action[:3])))
            if max_xyz > self.reject_translation_delta:
                raise RuntimeError(
                    "pi0 policy produced an unsafe EEF delta before clipping: "
                    f"xyz={raw_action[:3].tolist()}. This usually means the checkpoint was trained "
                    "with absolute/joint actions instead of ee_delta_6d_gripper. Stop robot rollout, "
                    "reconvert the dataset, and retrain."
                )
        if self.predict_eef_delta:
            action = clip_pi0_action(raw_action, self.config.action_limits)
            curr_eef_pose = np.asarray(obs["ee_pos_quat"], dtype=np.float32).reshape(-1)[:6]
            return get_eef_pose(curr_eef_pose, action)

        action = raw_action
        if len(action) != len(np.asarray(obs["joint_positions"]).reshape(-1)):
            raise ValueError(
                "pi0 without _eef expects a joint-space policy output matching joint_positions. "
                f"Got action shape {action.shape}; use --agent pi0_eef for the current ee_delta_6d_gripper policy."
            )
        action[-1] = np.clip(action[-1], self.config.action_limits["min_gripper"], self.config.action_limits["max_gripper"])
        return action.astype(np.float32)
