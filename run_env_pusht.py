import datetime
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import termcolor
import tyro
from pynput import keyboard

from agents.quest_agent_Akey import SingleArmQuestAgent
from camera_node import ZMQClientCamera
from env import RobotEnv
from robot_node import ZMQClientRobot
from writers.h5_trajectory_writer import H5TrajectoryWriter

trigger_state = {"r": False, "l": False}


def listen_key(key):
    global trigger_state
    try:
        trigger_state[key.char] = True
    except Exception:
        pass


def reset_key(key):
    global trigger_state
    try:
        trigger_state[key.char] = False
    except Exception:
        pass


listener = keyboard.Listener(on_press=listen_key)
listener2 = keyboard.Listener(on_release=reset_key)
listener.start()
listener2.start()


def print_color(*args, color=None, attrs=(), **kwargs):
    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


def _normalize_tcp_pose(pose: np.ndarray | None, fallback: np.ndarray) -> np.ndarray:
    if pose is None:
        return np.asarray(fallback, dtype=np.float64)
    pose_arr = np.asarray(pose, dtype=np.float64)
    if pose_arr.shape != fallback.shape:
        return np.asarray(fallback, dtype=np.float64)
    return pose_arr


def _count_folders(path: str) -> int:
    folder_count = 0
    for _, dirs, _ in os.walk(path):
        folder_count += len(dirs)
        break
    return folder_count


def _quest_stream_ready(agent: SingleArmQuestAgent, which_hand: str = "r") -> tuple[bool, str]:
    reader = getattr(agent, "oculus_reader", None)
    if reader is None:
        return True, "agent has no oculus reader"

    pose_data, button_data = reader.get_transformations_and_buttons()
    if len(pose_data) == 0 or len(button_data) == 0:
        return False, "no pose/button packets received yet"
    if which_hand not in pose_data:
        return False, f"missing {which_hand}-hand pose"

    trigger_key = "rightTrig" if which_hand == "r" else "leftTrig"
    if trigger_key not in button_data:
        return False, f"missing {trigger_key} in button packets"
    return True, "quest stream ready"


@dataclass
class Args:
    robot_port: int = 6000
    camera_port: int = 5000
    hostname: str = "127.0.0.1"
    hz: int = 15
    show_camera_view: bool = True
    save_depth: bool = True
    data_dir: str = "./shared/data/pusht"
    camera_obs_key: str = "multi_realsense"
    expected_camera_count: int = 0
    compression: str = "gzip"
    compression_level: int = 4
    robot_type: str = "ur5"
    verbose: bool = False


def _build_record(obs: dict, action: np.ndarray, agent: SingleArmQuestAgent, camera_obs_key: str) -> dict:
    robot_eef_pose = np.asarray(obs["ee_pos_quat"], dtype=np.float64)
    target_tcp_pose = _normalize_tcp_pose(agent.last_target_tcp_pose, robot_eef_pose)
    camera_rgb_key = f"{camera_obs_key}_rgb"
    camera_depth_key = f"{camera_obs_key}_depth"
    record = {
        f"{camera_obs_key}_rgb": np.asarray(obs[camera_rgb_key]),
        "robot_eef_pose": robot_eef_pose,
        "target_tcp_pose": target_tcp_pose,
        "control_joint_target": np.asarray(action, dtype=np.float64),
        "joint_positions": np.asarray(obs["joint_positions"], dtype=np.float64),
        "joint_velocities": np.asarray(obs["joint_velocities"], dtype=np.float64),
        "gripper_position": np.asarray(obs["gripper_position"], dtype=np.float64),
        "teleop_active": np.asarray(agent.control_active, dtype=np.bool_),
    }
    if camera_depth_key in obs:
        record[f"{camera_obs_key}_depth"] = np.asarray(obs[camera_depth_key])
    return record


def main(args: Args):
    print("Connecting to camera node...")
    camera_clients = {
        args.camera_obs_key: ZMQClientCamera(port=args.camera_port, host=args.hostname),
    }

    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(
        robot_client,
        control_rate_hz=args.hz,
        camera_dict=camera_clients,
        show_camera_view=args.show_camera_view,
        save_depth=args.save_depth,
    )

    agent = SingleArmQuestAgent(
        robot_type=args.robot_type,
        which_hand="r",
        verbose=args.verbose,
    )
    reset_joints = [-1.05247194, -1.71826806, -2.3567884,  -0.63558467,  1.66984439,  3.50998449,  0. ]
    # reset_joints = np.deg2rad([-87, -88, -112, -67, 90, 0, 0])
    curr_joints = env.get_obs()["joint_positions"]
    print("Current joints:", curr_joints)
    print("Reset joints:", reset_joints)
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = max(1, min(int(max_delta / 0.01), 20))
    for joints in np.linspace(curr_joints, reset_joints, steps):
        env.step(joints)

    obs = env.get_obs()
    print("Waiting for Quest controller stream...")
    last_quest_status = None
    while True:
        quest_ready, quest_status = _quest_stream_ready(agent, which_hand="r")
        if quest_ready:
            print_color("Quest stream ready.", color="green")
            break
        if quest_status != last_quest_status:
            print(
                ">>> Waiting for Quest data:",
                quest_status,
            )
            print(
                ">>> Put on the headset, allow USB debugging, open the Android panel if needed, "
                "and wake the right controller with the index trigger."
            )
            last_quest_status = quest_status
        time.sleep(0.5)

    print(f"Collecting traj no.{_count_folders(args.data_dir) + 1}")
    while not trigger_state["r"]:
        print(">>> Press keyboard [r] on this computer to start")
        time.sleep(0.2)

    print_color("\nReady to go", color="green", attrs=("bold",))
    print(
        ">>> Teleop bind: hold the Quest right index trigger to move the robot, "
        "release it to freeze."
    )
    print(
        ">>> Gripper: squeeze the right grip trigger to open, press Quest [A] to close."
    )
    time_str = datetime.datetime.now().strftime("%m%d_%H%M%S")
    save_path = Path(args.data_dir).expanduser() / time_str
    save_path.mkdir(parents=True, exist_ok=True)
    initial_rgb = np.asarray(obs[f"{args.camera_obs_key}_rgb"])
    camera_count = initial_rgb.shape[0] if initial_rgb.ndim == 4 else 1
    if args.expected_camera_count and camera_count != args.expected_camera_count:
        raise ValueError(
            f"Camera node returned {camera_count} camera(s), "
            f"but expected {args.expected_camera_count}."
        )
    writer = H5TrajectoryWriter(
        save_path / "trajectory.h5",
        video_fps=args.hz,
        compression=args.compression,
        compression_level=args.compression_level,
        metadata={
            "schema_name": "multicam_tcp_pose_v1",
            "camera_name": args.camera_obs_key,
            "camera_count": camera_count,
            "camera_source": "launch_nodes_zmq",
            "camera_port": args.camera_port,
        },
    )
    print(f"Saving to {save_path}")

    start_time = time.time()
    is_first_frame = True
    frame_freq = []
    try:
        while True:
            loop_start = time.time()
            elapsed = loop_start - start_time
            print_color(
                f"\rTime passed: {round(elapsed, 2)}          ",
                color="white",
                attrs=("bold",),
                end="",
                flush=True,
            )

            action = agent.act(obs)
            timestamp = datetime.datetime.now()
            if is_first_frame:
                is_first_frame = False
            else:
                writer.append(timestamp, _build_record(obs, action, agent, args.camera_obs_key))

            obs = env.step(action)
            frame_freq.append(1 / max(time.time() - loop_start, 1e-6))

            if trigger_state["l"]:
                print_color(
                    "\nStopping because keyboard trigger_state['l'] became True",
                    color="red",
                    attrs=("bold",),
                )
                break
    except KeyboardInterrupt:
        print_color("\nInterrupted!", color="red", attrs=("bold",))
    finally:
        print("Done")
        writer.close()

        with open(save_path / "freq.txt", "w") as f:
            if len(frame_freq) > 1:
                freq_slice = np.asarray(frame_freq[1:], dtype=np.float64)
            else:
                freq_slice = np.asarray(frame_freq, dtype=np.float64)
            if freq_slice.size > 0:
                f.write(
                    f"Average FPS: {np.mean(freq_slice)}\n"
                    f"Max FPS: {np.max(freq_slice)}\n"
                    f"Min FPS: {np.min(freq_slice)}\n"
                    f"Std FPS: {np.std(freq_slice)}\n\n"
                )
            for step, freq in enumerate(frame_freq):
                f.write(f"{step}: {freq}\n")


if __name__ == "__main__":
    main(tyro.cli(Args))
