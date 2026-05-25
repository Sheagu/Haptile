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
from marker_tracking.utils import (
    find_marker,
    find_marker_centers,
    plot_marker_delta,
)
from robot_node import ZMQClientRobot

trigger_state = {"r": False,"l":False}

# Mapping for tactile camera names to v4l2 by-path ports (persistent device paths)
# Change these paths to match your actual v4l/by-path devices
TACTILE_CAM_PORTS = {
    "left": "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:1.3.4:1.0-video-index0",
    "right": "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:2:1.0-video-index0",  # Update with your actual by-path
    # Fallback to int IDs if needed
    "2": 2,
    "4": 4,
}


def _resolve_tactile_warp_config(config_path: str, sensor_name: str):
    """Resolve tactile warp config path with simple project-local defaults."""
    sensor_suffix = sensor_name.replace("tactile_", "")
    candidates = [
        f"robo_test/sensor_config_{sensor_suffix}.json",
        f"robo_test/{sensor_name}_sensor_config.json",
        f"robo_test/{sensor_suffix}_sensor_config.json",
        f"robo_test/sensor_config_{sensor_name}.json",
        "robo_test/sensor_config.json",
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"No tactile warp config found for {sensor_name}. "
        "Expected one of: "
        + ", ".join(candidates)
    )

def _resolve_camera_id(camera_identifier):
    """Resolve camera identifier to either int device ID or v4l/by-path string.
    
    Args:
        camera_identifier: Either an int, or a string that could be:
            - A v4l/by-path path: "/dev/v4l/by-path/pci-..."
            - An int string: "2", "4", "22", etc.
            - A preset name: "left", "right"
    
    Returns:
        Either an int (for device ID) or str (for v4l/by-path path)
    """
    if isinstance(camera_identifier, int):
        return camera_identifier
    
    if isinstance(camera_identifier, str):
        # Check if it's a preset name
        if camera_identifier in TACTILE_CAM_PORTS:
            return TACTILE_CAM_PORTS[camera_identifier]
        
        # Check if it's a v4l/by-path path
        if camera_identifier.startswith("/dev/v4l/by-path/"):
            return camera_identifier
        
        # Check if it's a numeric string (convert to int)
        if camera_identifier.isdigit():
            return int(camera_identifier)
    
    raise ValueError(f"Invalid camera identifier: {camera_identifier}")

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


@dataclass
class MarkerTrackingState:
    name: str
    ref_gray: np.ndarray | None = None
    ref_points: np.ndarray | None = None
    display_window: str | None = None
    motion_ema: float = 0.0


def _build_marker_tracking_params(args: "Args") -> dict:
    return {
        "morphop_kernel": cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (args.marker_morph_open_size, args.marker_morph_open_size)
        ),
        "morphclose_kernel": cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (args.marker_morph_close_size, args.marker_morph_close_size),
        ),
        "dilate_kernel": cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (args.marker_dilate_size, args.marker_dilate_size)
        ),
        "mask_range": tuple(args.marker_mask_range),
        "min_value": args.marker_value_threshold,
        "morphop_iter": args.marker_morph_open_iter,
        "morphclose_iter": args.marker_morph_close_iter,
        "dilate_iter": args.marker_dilate_iter,
    }


def _init_marker_tracking_state(
    sensor_name: str,
    frame: np.ndarray | None,
    marker_tracking_params: dict,
    create_window: bool,
) -> MarkerTrackingState | None:
    if frame is None:
        return None

    marker_mask = find_marker(frame, **marker_tracking_params)
    centers = find_marker_centers(marker_mask)
    if not centers:
        print_color(
            f"[marker_tracking] {sensor_name}: no markers found in initial frame",
            color="yellow",
        )
        return MarkerTrackingState(name=sensor_name)

    state = MarkerTrackingState(
        name=sensor_name,
        ref_gray=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
        ref_points=np.array(centers, dtype=np.float32).reshape(-1, 1, 2),
        display_window=f"{sensor_name}_marker_tracking",
    )
    if create_window:
        cv2.namedWindow(state.display_window, cv2.WINDOW_NORMAL)
    print(
        f"[marker_tracking] {sensor_name}: initialized with {len(centers)} markers"
    )
    return state


def _mask_contains_points(mask: np.ndarray, points: np.ndarray, radius: int = 3) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    if radius > 0:
        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

    h, w = mask.shape[:2]
    rounded = np.rint(points).astype(np.int32)
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < w)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < h)
    )
    on_mask = np.zeros(len(points), dtype=bool)
    if np.any(in_bounds):
        valid_points = rounded[in_bounds]
        on_mask[in_bounds] = mask[valid_points[:, 1], valid_points[:, 0]] > 0
    return on_mask


def _update_marker_tracking(
    sensor_name: str,
    frame: np.ndarray | None,
    state: MarkerTrackingState | None,
    marker_tracking_params: dict,
    lk_params: dict,
    reset_on_loss: bool,
    arrow_scale: float,
    fb_max_error: float,
    motion_deadband: float,
    motion_smoothing: float,
    min_valid_points: int,
    compensate_global_drift: bool,
) -> tuple[MarkerTrackingState | None, np.ndarray | None]:
    if frame is None:
        return state, None

    if state is None or state.ref_points is None or state.ref_gray is None:
        state = _init_marker_tracking_state(
            sensor_name=sensor_name,
            frame=frame,
            marker_tracking_params=marker_tracking_params,
            create_window=False,
        )
        if state is None or state.ref_points is None or state.ref_gray is None:
            return state, None

    track_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        state.ref_gray,
        track_gray,
        state.ref_points,
        None,
        **lk_params,
    )
    if next_points is None or status is None:
        if reset_on_loss:
            return _init_marker_tracking_state(
                state.name, frame, marker_tracking_params, create_window=False
            ), None
        return state, None

    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
        track_gray,
        state.ref_gray,
        next_points,
        None,
        **lk_params,
    )
    if back_points is None or back_status is None:
        if reset_on_loss:
            return _init_marker_tracking_state(
                state.name, frame, marker_tracking_params, create_window=False
            ), None
        return state, None

    current_marker_mask = find_marker(frame, **marker_tracking_params)

    ref_points_all = state.ref_points.reshape(-1, 2)
    tracked_points_all = next_points.reshape(-1, 2)
    back_points_all = back_points.reshape(-1, 2)
    status = status.reshape(-1).astype(bool)
    back_status = back_status.reshape(-1).astype(bool)
    fb_error = np.linalg.norm(back_points_all - ref_points_all, axis=1)
    on_marker_mask = _mask_contains_points(current_marker_mask, tracked_points_all)
    valid = status & back_status & (fb_error <= fb_max_error) & on_marker_mask

    if np.count_nonzero(valid) < min_valid_points:
        if reset_on_loss:
            return _init_marker_tracking_state(
                state.name, frame, marker_tracking_params, create_window=False
            ), None
        return state, None

    ref_points = ref_points_all[valid]
    tracked_points = tracked_points_all[valid]
    deltas = tracked_points - ref_points

    if compensate_global_drift and len(deltas) > 0:
        deltas = deltas - np.median(deltas, axis=0, keepdims=True)

    delta_norms = np.linalg.norm(deltas, axis=1)
    delta_norms = np.clip(delta_norms - motion_deadband, 0.0, None)
    mean_motion = float(delta_norms.mean()) if len(delta_norms) > 0 else 0.0
    alpha = float(np.clip(motion_smoothing, 0.0, 1.0))
    state.motion_ema = (1.0 - alpha) * state.motion_ema + alpha * mean_motion
    overlay = plot_marker_delta(
        frame,
        tracked_points,
        deltas,
        scale=arrow_scale,
        arrow_color=(255, 0, 0),
    )
    state.ref_points = ref_points.reshape(-1, 1, 2).astype(np.float32)
    return state, overlay


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
    obs_to_save = dict(obs)
    obs_to_save["activated"] = activated
    obs_to_save["control"] = action  # add action to obs

    recorded_file = folder / (
        timestamp.isoformat().replace(":", "-").replace(".", "-") + ".pkl"
    )
    with open(recorded_file, "wb") as f:
        pickle.dump(obs_to_save, f)

    # save rgb image as png
    if save_png:
        if "base_camera_rgb" in obs_to_save:
            rgb = obs_to_save["base_camera_rgb"]
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
        if "tactile_left_rgb" in obs_to_save:
            tactile_left = obs_to_save["tactile_left_rgb"]
            if tactile_left.ndim == 4 and tactile_left.shape[0] == 1:
                tactile_left = tactile_left[0]
            if tactile_left.ndim == 3:
                tactile_left_bgr = cv2.cvtColor(tactile_left, cv2.COLOR_RGB2BGR)
                fn_left = str(recorded_file)[:-4] + f"-tactile_left.png"
                cv2.imwrite(fn_left, tactile_left_bgr)

        # Save right tactile sensor image
        if "tactile_right_rgb" in obs_to_save:
            tactile_right = obs_to_save["tactile_right_rgb"]
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
    tactile_left_camera_id: str = "left"  # v4l/by-path or int ID (supports: "left", "2", "/dev/v4l/by-path/...")
    tactile_right_camera_id: str = "right"  # v4l/by-path or int ID (supports: "right", "4", "/dev/v4l/by-path/...")
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
    enable_marker_tracking: bool = True
    marker_tracking_show_view: bool = True
    marker_tracking_reset_on_loss: bool = True
    marker_flow_win_size: tuple[int, int] = (15, 15)
    marker_flow_max_level: int = 2
    marker_flow_fb_max_error: float = 1.5
    marker_arrow_scale: float = 6.0
    marker_mask_range: tuple[int, int] = (145, 255)
    marker_value_threshold: int = 90
    marker_morph_open_size: int = 5
    marker_morph_open_iter: int = 1
    marker_morph_close_size: int = 5
    marker_morph_close_iter: int = 1
    marker_dilate_size: int = 3
    marker_dilate_iter: int = 0
    marker_motion_deadband: float = 0.2
    marker_motion_smoothing: float = 0.2
    marker_motion_min_valid_points: int = 8
    marker_motion_compensate_global_drift: bool = True
    # Amir: grip intensity
    grip_intens_annot: str = ""  # e.g. "soft", "medium", "hard". This is a one-time annotation for the whole trajectory, can be saved in a text file in the trajectory folder. If empty, no annotation will be saved.
    #
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
    marker_tracking_params = _build_marker_tracking_params(args)
    lk_params = {
        "winSize": tuple(args.marker_flow_win_size),
        "maxLevel": args.marker_flow_max_level,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            10,
            0.03,
        ),
    }

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
        # Resolve camera IDs/paths
        left_cam_id = _resolve_camera_id(args.tactile_left_camera_id)
        right_cam_id = _resolve_camera_id(args.tactile_right_camera_id)
        left_warp_config = _resolve_tactile_warp_config("", "tactile_left")
        right_warp_config = _resolve_tactile_warp_config("", "tactile_right")
        
        camera_clients["tactile_left"] = OpenCVCamera(
            camera_id=left_cam_id,
            width=args.tactile_width,
            height=args.tactile_height,
            perspective_config_path=left_warp_config,
            perspective_key="tactile_left",
        )
        camera_clients["tactile_right"] = OpenCVCamera(
            camera_id=right_cam_id,
            width=args.tactile_width,
            height=args.tactile_height,
            perspective_config_path=right_warp_config,
            perspective_key="tactile_right",
        )
        print(f"Tactile sensors enabled - Left: {left_cam_id}, Right: {right_cam_id}")
        print(
            "Tactile warp config - "
            f"Left: {left_warp_config or 'disabled'}, "
            f"Right: {right_warp_config or 'disabled'}"
        )
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

        # Amir: using A key to open the gripper
        from agents.quest_agent_Akey import SingleArmQuestAgent
        ## Uncomment below for otherwise
        # from agents.quest_agent import SingleArmQuestAgent
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
        # reset_joints = np.deg2rad([-82, -102, -70, -98, 86, 90, 0])
        reset_joints = np.deg2rad([-87, -88, -112, -67, 90, 0, 0])
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
    marker_tracking_states: dict[str, MarkerTrackingState | None] = {}
    if args.enable_marker_tracking and args.use_tactile:
        for sensor_name in ("tactile_left", "tactile_right"):
            obs_key = f"{sensor_name}_rgb"
            marker_tracking_states[sensor_name] = _init_marker_tracking_state(
                sensor_name=sensor_name,
                frame=obs.get(obs_key),
                marker_tracking_params=marker_tracking_params,
                create_window=args.marker_tracking_show_view,
            )
    else:
        marker_tracking_states = {}

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

        # Amir:
        ## saving grip intensity annotation - this is a one time save
        grip_intens_annot = "soft"
        if args.grip_intens_annot:
            grip_intens_annot_file = save_path / "grip_intensity_annot.txt"
            if not grip_intens_annot_file.exists():
                with open(grip_intens_annot_file, "w") as f:
                    f.write(f"{grip_intens_annot}\n")

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

            if args.enable_marker_tracking and args.use_tactile:
                for sensor_name in ("tactile_left", "tactile_right"):
                    obs_key = f"{sensor_name}_rgb"
                    tracking_key = f"{sensor_name}_marker_tracking"
                    state, overlay = _update_marker_tracking(
                        sensor_name=sensor_name,
                        frame=obs.get(obs_key),
                        state=marker_tracking_states.get(sensor_name),
                        marker_tracking_params=marker_tracking_params,
                        lk_params=lk_params,
                        reset_on_loss=args.marker_tracking_reset_on_loss,
                        arrow_scale=args.marker_arrow_scale,
                        fb_max_error=args.marker_flow_fb_max_error,
                        motion_deadband=args.marker_motion_deadband,
                        motion_smoothing=args.marker_motion_smoothing,
                        min_valid_points=args.marker_motion_min_valid_points,
                        compensate_global_drift=args.marker_motion_compensate_global_drift,
                    )
                    marker_tracking_states[sensor_name] = state
                    if overlay is not None:
                        obs[tracking_key] = overlay
                        if (
                            args.marker_tracking_show_view
                            and state is not None
                            and state.display_window is not None
                        ):
                            cv2.imshow(
                                state.display_window,
                                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                            )
                            cv2.waitKey(1)

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
