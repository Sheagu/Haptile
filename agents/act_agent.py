import collections
import json
import os
from typing import Any, Dict

import numpy as np
import torch

from agents.dp_agent import get_eef_pose, get_reset_joints, parse_txt_to_json
from learning.act.pipeline import Agent as ACTAgent

IMAGE_KEYS = {"img", "tactile_img"}


def _load_training_args(ckpt_path):
    ckpt_dir = os.path.dirname(ckpt_path)
    args_json = os.path.join(ckpt_dir, "args_log.json")
    args_txt = os.path.join(ckpt_dir, "args_log.txt")
    if os.path.exists(args_json):
        with open(args_json, "r") as f:
            return json.load(f)
    if os.path.exists(args_txt):
        return parse_txt_to_json(args_txt, args_json)
    return {}


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value == "True"
    return bool(value)


def _as_camera_indices(value):
    if isinstance(value, str):
        return [int(ch) for ch in value]
    if isinstance(value, int):
        return [value]
    return list(value)


def _as_repr(value):
    if isinstance(value, str):
        return value.split("-")
    return list(value)


class BimanualACTAgent:
    """Single-arm ACT deployment agent.

    The class mirrors the diffusion-policy deployment interface used by run_env.py.
    """

    def __init__(
        self,
        ckpt_path,
        act_args=None,
        gripper_min=0.0,
        gripper_max=1.0,
    ):
        if act_args is None:
            act_args = self.get_default_act_args()

        train_args = _load_training_args(ckpt_path)
        for key in list(act_args.keys()):
            if key == "output_sizes":
                image_output_size = train_args.get("image_output_size")
                if image_output_size is not None:
                    act_args[key]["img"] = image_output_size
                    act_args[key]["tactile_img"] = image_output_size
            elif key in train_args:
                act_args[key] = train_args[key]

        act_args["representation_type"] = _as_repr(act_args["representation_type"])
        act_args["camera_indices"] = _as_camera_indices(act_args["camera_indices"])
        for bool_key in (
            "identity_encoder",
            "predict_eef_delta",
            "predict_pos_delta",
            "binarize_touch",
            "act_use_vae",
        ):
            if bool_key in act_args:
                act_args[bool_key] = _as_bool(act_args[bool_key])

        ckpt_dir = os.path.dirname(ckpt_path)
        args_path = os.path.join(ckpt_dir, "act_args.json")
        if os.path.exists(args_path):
            with open(args_path, "r") as f:
                saved_args = json.load(f)
            merged_args = self.get_default_act_args()
            if "output_sizes" in saved_args:
                merged_args["output_sizes"].update(saved_args["output_sizes"])
            merged_args.update({k: v for k, v in saved_args.items() if k != "output_sizes"})
            act_args = merged_args
        else:
            with open(args_path, "w") as f:
                json.dump(act_args, f, indent=4)
        act_args["representation_type"] = _as_repr(act_args["representation_type"])
        act_args["camera_indices"] = _as_camera_indices(act_args["camera_indices"])
        for bool_key in (
            "identity_encoder",
            "predict_eef_delta",
            "predict_pos_delta",
            "binarize_touch",
            "act_use_vae",
        ):
            if bool_key in act_args:
                act_args[bool_key] = _as_bool(act_args[bool_key])

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        self.dp = ACTAgent(
            output_sizes=act_args["output_sizes"],
            representation_type=act_args["representation_type"],
            identity_encoder=act_args["identity_encoder"],
            camera_indices=act_args["camera_indices"],
            pred_horizon=act_args["pred_horizon"],
            obs_horizon=act_args["obs_horizon"],
            action_horizon=act_args["action_horizon"],
            predict_eef_delta=act_args["predict_eef_delta"],
            predict_pos_delta=act_args["predict_pos_delta"],
            joint_state_dim=act_args["joint_state_dim"],
            action_dim=act_args["action_dim"],
            eef_state_dim=act_args["eef_dim"],
            touch_dim=act_args["touch_dim"],
            hand_pos_dim=act_args["hand_pos_dim"],
            binarize_touch=act_args["binarize_touch"],
            act_hidden_dim=act_args["act_hidden_dim"],
            act_nheads=act_args["act_nheads"],
            act_encoder_layers=act_args["act_encoder_layers"],
            act_decoder_layers=act_args["act_decoder_layers"],
            act_dim_feedforward=act_args["act_dim_feedforward"],
            act_dropout=act_args["act_dropout"],
            act_latent_dim=act_args["act_latent_dim"],
            act_kl_weight=act_args["act_kl_weight"],
            act_use_vae=act_args["act_use_vae"],
        )
        self.dp_args = act_args
        self.obsque = collections.deque(maxlen=act_args["obs_horizon"])
        self.dp.load(ckpt_path)
        self.action_queue = collections.deque(maxlen=act_args["action_horizon"])
        self.predict_eef_delta = act_args["predict_eef_delta"]
        self.predict_pos_delta = act_args["predict_pos_delta"]
        assert not (self.predict_eef_delta and self.predict_pos_delta)
        self.control = get_reset_joints(ur_eef=self.predict_eef_delta)
        self.num_diffusion_iters = 1
        self.trigger_state = True
        self.gripper_min = gripper_min
        self.gripper_max = gripper_max

    @staticmethod
    def get_default_act_args():
        return {
            "output_sizes": {
                "eef": 64,
                "hand_pos": 64,
                "img": 32,
                "tactile_img": 32,
                "pos": 128,
                "touch": 64,
            },
            "representation_type": ["img", "eef", "pos", "touch"],
            "identity_encoder": False,
            "camera_indices": [0, 1],
            "obs_horizon": 1,
            "pred_horizon": 16,
            "action_horizon": 8,
            "clip_far": False,
            "predict_eef_delta": False,
            "predict_pos_delta": False,
            "joint_state_dim": 7,
            "action_dim": 7,
            "eef_dim": 6,
            "touch_dim": 30,
            "hand_pos_dim": 12,
            "binarize_touch": False,
            "act_hidden_dim": 256,
            "act_nheads": 8,
            "act_encoder_layers": 4,
            "act_decoder_layers": 4,
            "act_dim_feedforward": 1024,
            "act_dropout": 0.1,
            "act_latent_dim": 32,
            "act_kl_weight": 10.0,
            "act_use_vae": True,
        }

    def compile_inference(self, example_obs, precision="high", num_inference_iters=None):
        del num_inference_iters
        torch.set_float32_matmul_precision(precision)
        self.dp.policy.forward = torch.compile(torch.no_grad(self.dp.policy.forward))
        for _ in range(25):
            self.act(example_obs)

    def _preprocess_obs(self, obs: Dict[str, Any]):
        obs = self.dp.get_observation([obs], load_img=True)
        for image_key in IMAGE_KEYS:
            if image_key in obs:
                obs[image_key] = self.dp.eval_transform(obs[image_key].squeeze(0))
        return obs

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        curr_joint_pos = np.asarray(obs["joint_positions"], dtype=np.float32)
        curr_eef_pose = np.asarray(obs["ee_pos_quat"], dtype=np.float32)
        obs = self._preprocess_obs(obs)

        if len(self.obsque) == 0:
            self.obsque.extend([obs] * self.dp_args["obs_horizon"])
        else:
            self.obsque.append(obs)

        if len(self.action_queue) > 0:
            act = self.action_queue.popleft()
        else:
            pred = self.dp.predict(self.obsque)
            for i in range(self.dp_args["action_horizon"]):
                self.action_queue.append(pred[i])
            act = self.action_queue.popleft()

        act = np.asarray(act, dtype=np.float32)

        if self.predict_pos_delta:
            self.control[: len(curr_joint_pos)] = curr_joint_pos
            self.control = self.control + act
            act = self.control
        elif self.predict_eef_delta:
            act = get_eef_pose(curr_eef_pose, act)

        if not self.predict_eef_delta:
            act[-1] = np.clip(act[-1], self.gripper_min, self.gripper_max)

        return act
