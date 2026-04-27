import numpy as np

from pi0_ur5e.transforms import absolute_pose_to_delta_action, delta_action_to_absolute_poses


def test_absolute_pose_to_delta_action_keeps_gripper_absolute():
    poses = np.array([[0, 0, 0, 0, 0, 0], [1, 2, 3, 0.1, 0.2, 0.3], [2, 3, 4, 0.2, 0.3, 0.4]], dtype=np.float32)
    gripper = np.array([[0.0], [1.0], [1.0]], dtype=np.float32)
    action = absolute_pose_to_delta_action(poses, gripper)
    np.testing.assert_allclose(action[0, :6], poses[1] - poses[0])
    np.testing.assert_allclose(action[1, :6], poses[2] - poses[1])
    np.testing.assert_allclose(action[:, 6], gripper[:, 0])


def test_delta_action_to_absolute_replay_consistency():
    poses = np.array([[0, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0], [0.2, 0.2, 0, 0, 0, 0]], dtype=np.float32)
    actions = absolute_pose_to_delta_action(poses, np.zeros((3, 1), dtype=np.float32))
    replay = delta_action_to_absolute_poses(poses[0], actions)
    np.testing.assert_allclose(replay, poses, atol=1e-6)
