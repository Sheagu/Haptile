import pickle

import numpy as np

from pi0_ur5e.dataset_reader import DatasetReader


def test_dataset_reader_reads_fake_episode(tmp_path):
    ep = tmp_path / "episode_000"
    ep.mkdir()
    for i in range(3):
        frame = {
            "base_rgb": np.zeros((8, 8, 3), dtype=np.uint8) + i,
            "wrist_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "ee_pos_quat": np.array([i, 0, 0, 0, 0, 0], dtype=np.float32),
            "gripper_position": np.array([float(i > 1)], dtype=np.float32),
            "language_instruction": "test prompt",
        }
        with open(ep / f"{i}.pkl", "wb") as f:
            pickle.dump(frame, f)
    episodes = DatasetReader(tmp_path).episodes()
    assert len(episodes) == 1
    out = episodes[0]
    assert out.base_rgb.shape == (3, 8, 8, 3)
    assert out.wrist_rgb.shape == (3, 8, 8, 3)
    assert out.robot_state.shape == (3, 6)
    assert out.action.shape == (3, 7)
    assert out.language_instruction == "test prompt"
