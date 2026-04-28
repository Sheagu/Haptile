import numpy as np

from pi0_ur5e.policy_client import DryRunPolicyClient, build_policy_observation, safe_action


def test_build_policy_observation_uses_openpi_raw_keys():
    obs = build_policy_observation(
        base_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        wrist_rgb=np.ones((224, 224, 3), dtype=np.uint8),
        state=np.arange(7),
        prompt="test prompt",
    )
    assert sorted(obs) == ["base_rgb", "prompt", "state", "wrist_rgb"]
    assert obs["base_rgb"].dtype == np.uint8
    assert obs["wrist_rgb"].dtype == np.uint8
    assert obs["state"].dtype == np.float32
    assert obs["prompt"] == "test prompt"


def test_safe_action_clips_network_style_action():
    class Policy(DryRunPolicyClient):
        def act(self, observation):
            return np.array([1.0, -1.0, 0.2, 1.0, -1.0, 0.5, 2.0], dtype=np.float32)

    action = safe_action(
        Policy(),
        {},
        {
            "max_translation_delta": 0.05,
            "max_rotation_delta": 0.35,
            "min_gripper": 0.0,
            "max_gripper": 1.0,
        },
    )
    np.testing.assert_allclose(action, [0.05, -0.05, 0.05, 0.35, -0.35, 0.35, 1.0])
