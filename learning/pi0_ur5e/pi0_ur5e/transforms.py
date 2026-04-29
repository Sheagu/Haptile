from __future__ import annotations

import numpy as np


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float32)


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float32)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32) + _skew(rotvec)
    axis = rotvec / angle
    axis_skew = _skew(axis)
    return (
        np.eye(3, dtype=np.float32)
        + np.sin(angle) * axis_skew
        + (1.0 - np.cos(angle)) * (axis_skew @ axis_skew)
    ).astype(np.float32)


def _matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    cos_angle = float((np.trace(matrix) - 1.0) * 0.5)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if angle < 1e-8:
        return np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) * 0.5,
                (matrix[0, 2] - matrix[2, 0]) * 0.5,
                (matrix[1, 0] - matrix[0, 1]) * 0.5,
            ],
            dtype=np.float32,
        )
    if np.pi - angle < 1e-5:
        axis = np.sqrt(np.maximum(np.diag(matrix) + 1.0, 0.0) * 0.5)
        axis[0] = np.copysign(axis[0], matrix[2, 1] - matrix[1, 2])
        axis[1] = np.copysign(axis[1], matrix[0, 2] - matrix[2, 0])
        axis[2] = np.copysign(axis[2], matrix[1, 0] - matrix[0, 1])
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            return np.zeros(3, dtype=np.float32)
        return (axis / norm * angle).astype(np.float32)
    return (
        angle
        / (2.0 * np.sin(angle))
        * np.array(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ],
            dtype=np.float32,
        )
    ).astype(np.float32)


def ensure_2d(array: np.ndarray | None, width: int | None = None) -> np.ndarray | None:
    if array is None:
        return None
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None] if width == 1 else arr[None, :]
    return arr.astype(np.float32)


def absolute_pose_to_delta_action(poses: np.ndarray, gripper: np.ndarray | None = None) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] < 6:
        raise ValueError(f"Expected absolute poses with shape [T, >=6], got {poses.shape}")
    delta = np.zeros((poses.shape[0], 6), dtype=np.float32)
    delta[:-1, :3] = poses[1:, :3] - poses[:-1, :3]
    for idx in range(len(poses) - 1):
        current_rot = _rotvec_to_matrix(poses[idx, 3:6])
        next_rot = _rotvec_to_matrix(poses[idx + 1, 3:6])
        delta[idx, 3:6] = _matrix_to_rotvec(next_rot @ current_rot.T)
    if gripper is None:
        grip = poses[:, 6:7] if poses.shape[1] > 6 else np.zeros((poses.shape[0], 1), dtype=np.float32)
    else:
        grip = np.asarray(gripper, dtype=np.float32).reshape(poses.shape[0], -1)[:, :1]
    return np.concatenate([delta, grip], axis=1).astype(np.float32)


def delta_action_to_absolute_poses(start_pose: np.ndarray, actions: np.ndarray) -> np.ndarray:
    start = np.asarray(start_pose, dtype=np.float32)[:6]
    actions = np.asarray(actions, dtype=np.float32)
    poses = np.zeros((len(actions), 6), dtype=np.float32)
    pose = start.copy()
    for i, action in enumerate(actions):
        poses[i] = pose
        pose = pose + action[:6]
    return poses


def joint_position_to_delta_action(joints: np.ndarray, gripper: np.ndarray | None = None) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    arm = joints[:, :-1] if joints.shape[1] > 6 else joints
    delta = np.zeros_like(arm, dtype=np.float32)
    delta[:-1] = arm[1:] - arm[:-1]
    if gripper is None:
        grip = joints[:, -1:].astype(np.float32) if joints.shape[1] > 6 else np.zeros((len(joints), 1), dtype=np.float32)
    else:
        grip = np.asarray(gripper, dtype=np.float32).reshape(len(joints), -1)[:, :1]
    return np.concatenate([delta, grip], axis=1).astype(np.float32)


def build_action(raw_action: np.ndarray | None, state: np.ndarray, gripper: np.ndarray | None, action_mode: str) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if raw_action is not None:
        action = np.asarray(raw_action, dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
    else:
        action = state
    if action_mode == "ee_delta_6d_gripper":
        # For this repository's teleop logs, the saved `control` field may be a
        # joint command or an absolute target. The reliable source for EEF deltas
        # is the consecutive TCP pose series stored in state.
        return absolute_pose_to_delta_action(state[:, :6], gripper)
    if action_mode == "ee_absolute_6d_gripper":
        grip = gripper if gripper is not None else action[:, 6:7] if action.shape[1] > 6 else np.zeros((len(action), 1), dtype=np.float32)
        return np.concatenate([action[:, :6], np.asarray(grip, dtype=np.float32).reshape(len(action), -1)[:, :1]], axis=1).astype(np.float32)
    if action_mode == "joint_position_gripper":
        return action.astype(np.float32)
    if action_mode == "joint_delta_gripper":
        return joint_position_to_delta_action(action, gripper)
    raise ValueError(f"Unsupported action_mode: {action_mode}")


def clip_pi0_action(action: np.ndarray, limits: dict) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).copy()
    if action.shape[-1] >= 3:
        action[..., :3] = np.clip(action[..., :3], -limits.get("max_translation_delta", 0.05), limits.get("max_translation_delta", 0.05))
    if action.shape[-1] >= 6:
        action[..., 3:6] = np.clip(action[..., 3:6], -limits.get("max_rotation_delta", 0.35), limits.get("max_rotation_delta", 0.35))
    if action.shape[-1] >= 7:
        action[..., 6] = np.clip(action[..., 6], limits.get("min_gripper", 0.0), limits.get("max_gripper", 1.0))
    return action.astype(np.float32)
