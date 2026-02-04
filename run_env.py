import datetime
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import termcolor
import tyro

# foot pedal
from pynput import keyboard

from agents.agent import BimanualAgent, SafetyWrapper
from camera_node import ZMQClientCamera
from cameras.opencv_camera import OpenCVCamera
from cameras.realsense_camera import RealSenseCamera
from env import RobotEnv
from robot_node import ZMQClientRobot

trigger_state = {"r": False,"l":False}


def listen_key(key):
    global trigger_state
    try:
        trigger_state[key.char] = True
    except:
        pass


def reset_key(key):
    global trigger_state
    try:
        trigger_state[key.char] = False
    except:
        pass


listener = keyboard.Listener(on_press=listen_key)
listener2 = keyboard.Listener(on_release=reset_key)
listener.start()
listener2.start()

###


def count_folders(path):
    """Counts the number of folders under the given path."""
    folder_count = 0
    for root, dirs, files in os.walk(path):
        folder_count += len(dirs)  # Count directories only at current level
        break  # Prevents descending into subdirectories
    return folder_count


def print_color(*args, color=None, attrs=(), **kwargs):
    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


def save_frame(
    folder: Path,
    timestamp: datetime.datetime,
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
    activated=True,
    save_png=False,
    save_tactile_png=False,
    use_tactile=True,
) -> None:
    obs["activated"] = activated
    obs["control"] = action  # add action to obs
    recorded_file = folder / (
        timestamp.isoformat().replace(":", "-").replace(".", "-") + ".pkl"
    )
    with open(recorded_file, "wb") as f:
        pickle.dump(obs, f)

    # save rgb image as png
    if save_png:
        if "base_camera_rgb" in obs:
            rgb = obs["base_camera_rgb"]
            # Handle different dimensions from RealSense (can have multiple cameras)
            if rgb.ndim == 4:  # (num_cameras, H, W, 3)
                for i in range(rgb.shape[0]):
                    rgbi = cv2.cvtColor(rgb[i], cv2.COLOR_RGB2BGR)
                    fn = str(recorded_file)[:-4] + f"-base_{i}.png"
                    cv2.imwrite(fn, rgbi)
            elif rgb.ndim == 3:  # (H, W, 3) - single camera
                rgbi = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                fn = str(recorded_file)[:-4] + f"-base.png"
                cv2.imwrite(fn, rgbi)
    
    # save tactile images as png (only if use_tactile is True)
    if save_tactile_png and use_tactile:
        # Save left tactile sensor image
        if "tactile_left_rgb" in obs:
            tactile_left = obs["tactile_left_rgb"]
            if tactile_left.ndim == 4 and tactile_left.shape[0] == 1:
                tactile_left = tactile_left[0]
            if tactile_left.ndim == 3:
                tactile_left_bgr = cv2.cvtColor(tactile_left, cv2.COLOR_RGB2BGR)
                fn_left = str(recorded_file)[:-4] + f"-tactile_left.png"
                cv2.imwrite(fn_left, tactile_left_bgr)
        
        # Save right tactile sensor image
        if "tactile_right_rgb" in obs:
            tactile_right = obs["tactile_right_rgb"]
            if tactile_right.ndim == 4 and tactile_right.shape[0] == 1:
                tactile_right = tactile_right[0]
            if tactile_right.ndim == 3:
                tactile_right_bgr = cv2.cvtColor(tactile_right, cv2.COLOR_RGB2BGR)
                fn_right = str(recorded_file)[:-4] + f"-tactile_right.png"
                cv2.imwrite(fn_right, tactile_right_bgr)


@dataclass
class Args:
    robot_port: int = 6000
    wrist_camera_port: int = 5001
    base_camera_port: int = 5000
    tactile_left_camera_id: int = 2  # OpenCV camera ID for left tactile
    tactile_right_camera_id: int = 4  # OpenCV camera ID for right tactile
    hostname: str = "127.0.0.1"
    hz: int = 50
    show_camera_view: bool = True
    agent: str = "quest"
    robot_type: str = "ur5"
    save_data: bool = False
    save_depth: bool = True 
    save_png: bool = False
    use_tactile: bool = True  # whether to use tactile sensors
    save_tactile_png: bool = True  # save tactile images as PNG
    realsense_width: int = 640  # RealSense camera resolution width
    realsense_height: int = 480  # RealSense camera resolution height
    realsense_fps: int = 30  # RealSense camera FPS
    tactile_width: int = 640  # Tactile camera resolution width
    tactile_height: int = 480  # Tactile camera resolution height
    data_dir: str = "./shared/data/bc_data"
    verbose: bool = False
    safe: bool = False
    use_vel_ik: bool = False

    use_camera_node: bool = True # use camera node

    num_diffusion_iters_compile: int = 15  # used for compilation only for now
    jit_compile: bool = False  # send the compilation signal to the server (only need to do this once per inference server run).
    use_jit_agent: bool = False  # use the inference server to get actions. The inference_agent_port and the inference_agent_host need to be set to the proper values.
    inference_agent_port: str = (
        "1234"  # port must be the same as the inference server port
    )
    inference_agent_host = "127.0.0.2"  # ip of the inference server (localhost if running locally; currently defaults to bt) inference server needs to use the same checkpoint folder when launching the inference node (args need to match)

    dp_ckpt_path: str = "./shared/ckpts/best.ckpt"

    temporal_ensemble_mode: str = "avg"
    temporal_ensemble_act_tau: float = 0.5


def main(args):

    # Initialize cameras
    if args.use_camera_node:
        print("Using camera node...")
        camera_clients = {
            "base_camera": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
        }
    else:
        print("Initializing RealSense base camera...")
        camera_clients = {
            "base_camera": RealSenseCamera(
                height=args.realsense_height,
                width=args.realsense_width,
                fps=args.realsense_fps,
                img_size=(args.realsense_width, args.realsense_height),
            ),
        }
    
    # Add tactile cameras if enabled (using OpenCV webcams)
    if args.use_tactile:
        print("Initializing tactile sensors with OpenCV...")
        camera_clients["tactile_left"] = OpenCVCamera(
            camera_id=args.tactile_left_camera_id,
            width=args.tactile_width,
            height=args.tactile_height
        )
        camera_clients["tactile_right"] = OpenCVCamera(
            camera_id=args.tactile_right_camera_id,
            width=args.tactile_width,
            height=args.tactile_height
        )
        print("Tactile sensors enabled")
    else:
        print("Tactile sensors disabled")
    
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(
        robot_client,
        control_rate_hz=args.hz,
        camera_dict=camera_clients,
        show_camera_view=args.show_camera_view,
        save_depth=args.save_depth,
    )

    if args.agent == "quest":
        from agents.quest_agent import SingleArmQuestAgent
        agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="r")
        print("Quest agent created")
    # elif args.agent in ["dp", "dp_eef"]:
    #     if args.use_jit_agent:
    #         from agents.dp_agent_zmq import BimanualDPAgent
    #         agent = BimanualDPAgent(
    #             ckpt_path=args.dp_ckpt_path,
    #             port=args.inference_agent_port,
    #             host=args.inference_agent_host,
    #             temporal_ensemble_act_tau=args.temporal_ensemble_act_tau,
    #             temporal_ensemble_mode=args.temporal_ensemble_mode,
    #         )
    #     else:
    #         from agents.dp_agent import BimanualDPAgent
    #         agent = BimanualDPAgent(ckpt_path=args.dp_ckpt_path)
    else:
        raise ValueError(f"Invalid agent name : {args.agent}")

    if args.agent == "quest":
        # using grippers
        #To-do   90
        reset_joints = np.deg2rad([-82, -102, -70, -98, 86, 90, 0])
    # else:
    #     # using Ability hands
    #     arm_joints_left = [-80, -140, -80, -85, -10, 80]
    #     arm_joints_right = [-270, -30, 70, -85, 10, 0]
    #     hand_joints = [0, 0, 0, 0, 0.5, 0.5]
    #     reset_joints_left = np.concatenate([np.deg2rad(arm_joints_left), hand_joints])
    #     reset_joints_right = np.concatenate([np.deg2rad(arm_joints_right), hand_joints])
    # reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    curr_joints = env.get_obs()["joint_positions"]
    # curr_joints[6:12] = hand_joints
    # curr_joints[18:] = hand_joints
    print("Current joints:", curr_joints)
    print("Reset joints:", reset_joints)
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.01), 20)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step(jnt)

    obs = env.get_obs()
    # if args.jit_compile:
    #     agent.compile_inference(
    #         obs, num_diffusion_iters=args.num_diffusion_iters_compile
    #     )
    # going to start position
    print("Going to start position")
    start_pos = agent.act(obs) # in mujoco
    obs = env.get_obs()
    joints = obs["joint_positions"]

    # if args.agent == "quest":
    #     ur_idx = [i for i in range(len(joints))]
    #     hand_idx = None
    # else:
    #     ur_idx = list(range(0, 6)) + list(range(12, 18))
    #     hand_idx = list(range(6, 12)) + list(range(18, 24))

    # if args.safe:
    #     max_joint_delta = 0.5
    #     max_hand_delta = 0.1
    #     safety_wrapper = SafetyWrapper(
    #         ur_idx, hand_idx, agent, delta=max_joint_delta, hand_delta=max_hand_delta
    #     )

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    print(f"Collecting traj no.{count_folders(args.data_dir) + 1}")

    # time.sleep(2.0)
    while not trigger_state["r"]:
        print(">>> Press on [r] to start")
        time.sleep(0.2)

    print_color("\nReady to go 🚀🚀🚀", color="green", attrs=("bold",))

    start_time = time.time()

    if args.save_data:
        time_str = datetime.datetime.now().strftime("%m%d_%H%M%S")
        # if args.agent.startswith("dp"):
        #     # eval
        #     save_path = (
        #         Path(args.data_dir).expanduser()
        #         / "_".join(
        #             [
        #                 args.dp_ckpt_path.split("/")[-2],
        #                 args.dp_ckpt_path.split("/")[-1][:-5],
        #             ]
        #         )
        #         / time_str
        #     )
        # else:
        save_path = Path(args.data_dir).expanduser() / time_str
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {save_path}")

    is_first_frame = True
    try:
        frame_freq = []
        while True:
            new_start_time = time.time()
            num = new_start_time - start_time
            message = f"\rTime passed: {round(num, 2)}          "
            print_color(
                message,
                color="white",
                attrs=("bold",),
                end="",
                flush=True,
            )
            # if args.safe:
            #     action = safety_wrapper.act_safe(
            #         agent, obs, eef=(args.agent.endswith("_eef"))
            #     )
            # else:
            action = agent.act(obs)
            dt = datetime.datetime.now()

            if args.save_data:
                if is_first_frame:
                    is_first_frame = False
                else:
                    save_frame(
                        save_path,
                        dt,
                        obs,
                        action,
                        # activated=agent.trigger_state,
                        save_png=args.save_png,
                        save_tactile_png=args.save_tactile_png,
                        use_tactile=args.use_tactile,
                    )

            # if args.agent.endswith("_eef"):
            #     obs = env.step_eef(action)
            # else:
            obs = env.step(action)

            ff = 1 / (time.time() - new_start_time)
            frame_freq.append(ff)

            if trigger_state["l"]:
                print_color("\nTriggered!", color="red", attrs=("bold",))
                break

    except KeyboardInterrupt:
        print_color("\nInterrupted!", color="red", attrs=("bold",))
    finally:
        # if "dp" in args.agent:
        #     import glob

        #     from moviepy.editor import ImageSequenceClip

        #     # find all the pkl files in the episode directory
        #     pkls = sorted(glob.glob(os.path.join(save_path, "*.pkl")))
        #     print("Total number of pkls: ", len(pkls))
        #     frames = []
        #     for pkl in pkls:
        #         with open(pkl, "rb") as f:
        #             try:
        #                 data = pickle.load(f)
        #             except:
        #                 continue
        #         rgb = data["base_rgb"]
        #         rgb = np.concatenate([rgb[i] for i in range(rgb.shape[0])], axis=1)
        #         frames.append(rgb)
        #     clip = ImageSequenceClip(frames, fps=5)
        #     ckpt_path = os.path.dirname(args.dp_ckpt_path)
        #     parent_name = os.path.basename(ckpt_path)
        #     clip.write_videofile(
        #         os.path.join(ckpt_path, f"{parent_name}_{time_str}.mp4")
        #     )

        #     # save frame freq as txt
        #     with open(os.path.join(ckpt_path, f"freq_{time_str}.txt"), "w") as f:
        #         for step, freq in enumerate(frame_freq):
        #             f.write(f"{step}: {freq}\n")
        # else:
        print("Done")

        # Release camera resources
        print("Releasing camera resources...")
        for camera_name, camera in camera_clients.items():
            if hasattr(camera, 'release'):
                camera.release()
                print(f"Released {camera_name}")

        # save frame freq as txt
        if args.save_data:
            with open(save_path / "freq.txt", "w") as f:
                f.write(
                    f"Average FPS: {np.mean(frame_freq[1:])}\n"
                    f"Max FPS: {np.max(frame_freq[1:])}\n"
                    f"Min FPS: {np.min(frame_freq[1:])}\n"
                    f"Std FPS: {np.std(frame_freq[1:])}\n\n"
                )
                for step, freq in enumerate(frame_freq):
                    f.write(f"{step}: {freq}\n")

        os._exit(0)


if __name__ == "__main__":
    main(tyro.cli(Args))
