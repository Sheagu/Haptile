import numpy as np
import pickle
import time
import torch
import zmq
from typing import Any, Dict

from agents.dp_agent import BimanualDPAgent as LocalDPAgent, get_reset_joints
from inference_node import (
    DEFAULT_INFERENCE_PORT,
    ZMQInferenceClient,
    ZMQInferenceServer,
)


class BimanualDPAgentServer(ZMQInferenceServer):
    """Single-arm async inference server.

    The class name is kept for compatibility with existing launch scripts.
    """

    def __init__(
        self,
        ckpt_path,
        dp_args=None,
        gripper_min=0.0,
        gripper_max=1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.agent = LocalDPAgent(
            ckpt_path=ckpt_path,
            dp_args=dp_args,
            gripper_min=gripper_min,
            gripper_max=gripper_max,
        )
        self.num_diffusion_iters = self.agent.num_diffusion_iters

    @staticmethod
    def get_default_dp_args():
        return LocalDPAgent.get_default_dp_args()

    def reset_temporal_state(self) -> None:
        self.agent.reset_temporal_state()

    def compile_inference(self, precision="high"):
        message = self._socket.recv()
        start_time = time.time()
        state_dict = pickle.loads(message)
        self.num_diffusion_iters = state_dict["num_diffusion_iters"]
        example_obs = state_dict["example_obs"]
        print(
            f"received compilation request: # diff iters = {state_dict['num_diffusion_iters']}"
        )
        torch.set_float32_matmul_precision(precision)
        self.agent.dp.policy.forward = torch.compile(
            torch.no_grad(self.agent.dp.policy.forward)
        )
        self.agent.num_diffusion_iters = self.num_diffusion_iters
        for _ in range(25):
            self.agent.act(example_obs)
        self.agent.reset_temporal_state()
        print("success, compile time: " + str(time.time() - start_time))
        self._socket.send_string("success")

    def infer(self, obs: Dict[str, Any]) -> np.ndarray:
        return self.agent.dp.predict(
            [obs], num_diffusion_iters=self.num_diffusion_iters
        )

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        self.agent.num_diffusion_iters = self.num_diffusion_iters
        return self.agent.act(obs)


class BimanualDPAgent(ZMQInferenceClient):
    """Single-arm async inference client.

    The class name is kept for compatibility with existing deployment scripts.
    """

    def __init__(
        self,
        ckpt_path,
        dp_args=None,
        port=DEFAULT_INFERENCE_PORT,
        host="127.0.0.1",
        temporal_ensemble_mode="new",
        temporal_ensemble_act_tau=0.5,
    ):
        local_agent = LocalDPAgent(ckpt_path=ckpt_path, dp_args=dp_args)
        super().__init__(
            default_action=get_reset_joints(
                ur_eef=local_agent.dp_args["predict_eef_delta"]
            ),
            port=port,
            host=host,
            ensemble_mode=temporal_ensemble_mode,
            act_tau=temporal_ensemble_act_tau,
        )
        self.dp_args = local_agent.dp_args
        self.predict_eef_delta = self.dp_args["predict_eef_delta"]
        self.predict_pos_delta = self.dp_args["predict_pos_delta"]
        self.control = get_reset_joints(ur_eef=self.predict_eef_delta)
        self.num_diffusion_iters = self.dp_args["num_diffusion_iters"]
        self.trigger_state = True

    @staticmethod
    def get_default_dp_args():
        return LocalDPAgent.get_default_dp_args()

    def reset_temporal_state(self) -> None:
        self.act_q.clear()
        self.t = 0
        self.last_act = get_reset_joints(ur_eef=self.predict_eef_delta)
        self.control = get_reset_joints(ur_eef=self.predict_eef_delta)
        self._socket.send(pickle.dumps({"reset_temporal_state": True}))
        message = self._socket.recv()
        assert message == b"success"
        while True:
            try:
                self._socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

    def compile_inference(self, example_obs, num_diffusion_iters):
        message = pickle.dumps(
            {"example_obs": example_obs, "num_diffusion_iters": num_diffusion_iters}
        )
        self._socket.send(message)
        message = self._socket.recv()
        assert message == b"success"

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        curr_joint_pos = np.asarray(obs["joint_positions"], dtype=np.float32)
        act = np.asarray(super().act(obs), dtype=np.float32)
        if self.predict_pos_delta:
            self.control[: len(curr_joint_pos)] = curr_joint_pos
            self.control = self.control + act
            act = self.control
        if not self.predict_eef_delta:
            act[-1] = np.clip(act[-1], 0.0, 1.0)
        return act
