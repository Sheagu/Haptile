import numpy as np

from pi0_ur5e.sanity_checks import action_state_alignment
from pi0_ur5e.transforms import absolute_pose_to_delta_action


def test_action_delta_aligns_with_linear_state_diff():
    state = np.zeros((10, 6), dtype=np.float32)
    state[:, 0] = np.arange(10, dtype=np.float32) * 0.01
    action = absolute_pose_to_delta_action(state, np.zeros((10, 1), dtype=np.float32))
    report = action_state_alignment(action, state)
    assert report["available"]
    assert report["xyz_corr"][0] is None or report["xyz_corr"][0] > 0.99
    np.testing.assert_allclose(action[:-1, 0], 0.01, atol=1e-6)
