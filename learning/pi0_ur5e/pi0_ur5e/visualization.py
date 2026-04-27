from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_replay(episode, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(episode.timestamps))
    action = np.asarray(episode.action)
    state = np.asarray(episode.robot_state)
    rows = 4 if episode.tactile is not None else 3
    fig, axes = plt.subplots(rows, 1, figsize=(10, 2.8 * rows), sharex=True)
    axes[0].plot(t, state[:, :3])
    axes[0].set_title("ee/state position first 3 dims")
    axes[1].plot(t, action[:, :3])
    axes[1].set_title("action dx/dy/dz")
    axes[2].plot(t, action[:, -1])
    axes[2].set_title("gripper command")
    if episode.tactile is not None:
        axes[3].plot(t, np.asarray(episode.tactile))
        axes[3].set_title("tactile features")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
