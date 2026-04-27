import pickle

import numpy as np
import pytest

from pi0_ur5e.dataset_reader import DatasetReader
from pi0_ur5e.io_utils import read_json
from pi0_ur5e.lerobot_writer import write_lerobot_dataset


def test_fake_raw_dataset_converts_to_lerobot(tmp_path):
    pytest.importorskip("lerobot")
    raw = tmp_path / "raw" / "episode_000"
    raw.mkdir(parents=True)
    for i in range(4):
        with open(raw / f"{i}.pkl", "wb") as f:
            pickle.dump(
                {
                    "base_rgb": np.zeros((16, 16, 3), dtype=np.uint8),
                    "wrist_rgb": np.zeros((16, 16, 3), dtype=np.uint8),
                    "ee_pos_quat": np.array([i * 0.01, 0, 0, 0, 0, 0], dtype=np.float32),
                    "gripper_position": np.array([0.0], dtype=np.float32),
                },
                f,
            )
    episodes = DatasetReader(tmp_path / "raw").episodes()
    out = tmp_path / "lerobot"
    report = write_lerobot_dataset(episodes, out, task_name="cup_pick_place", repo_id="local/test_pi0_ur5e")
    assert (out / "meta" / "info.json").exists()
    assert (out / "data").exists()
    assert report["episode_count"] == 1
    info = read_json(out / "meta" / "info.json")
    assert "observation.images.base_rgb" in info["features"]
    assert info["features"]["action"]["shape"] == [7]
    reread = DatasetReader(out).episodes()
    assert len(reread) == 1
    assert reread[0].action.shape == (4, 7)
