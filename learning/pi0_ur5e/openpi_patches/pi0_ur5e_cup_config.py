# BEGIN TELE_GSY_PI0_UR5E_CUP
# Paste this block into openpi/src/openpi/training/config.py immediately before:
#   if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
#
# It is written as an append-only patch so it does not require modifying
# OpenPI's existing config list entries.

import os as _tele_gsy_os
import numpy as _tele_gsy_np


def _tele_gsy_env_bool(name: str, default: bool) -> bool:
    value = _tele_gsy_os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _tele_gsy_env_int(name: str, default: int | None = None) -> int | None:
    value = _tele_gsy_os.environ.get(name)
    if value in {None, ""}:
        return default
    return int(value)


def _tele_gsy_parse_image(image) -> _tele_gsy_np.ndarray:
    image = _tele_gsy_np.asarray(image)
    if _tele_gsy_np.issubdtype(image.dtype, _tele_gsy_np.floating):
        image = _tele_gsy_np.clip(image, 0.0, 1.0) * 255.0
        image = image.astype(_tele_gsy_np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = _tele_gsy_np.moveaxis(image, 0, -1)
    return image.astype(_tele_gsy_np.uint8)


@dataclasses.dataclass(frozen=True)
class TeleGsyUR5eInputs(_transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI0
    camera_padding_strategy: str = "duplicate_base"

    def __call__(self, data: dict) -> dict:
        base_image = _tele_gsy_parse_image(data["base_rgb"])
        wrist_image = _tele_gsy_parse_image(data["wrist_rgb"])
        if self.camera_padding_strategy == "duplicate_wrist":
            right_wrist_image = wrist_image
            right_wrist_mask = _tele_gsy_np.True_
        elif self.camera_padding_strategy == "zeros":
            right_wrist_image = _tele_gsy_np.zeros_like(base_image)
            right_wrist_mask = _tele_gsy_np.True_ if self.model_type == _model.ModelType.PI0_FAST else _tele_gsy_np.False_
        else:
            right_wrist_image = base_image
            right_wrist_mask = _tele_gsy_np.True_ if self.camera_padding_strategy == "duplicate_base" else _tele_gsy_np.False_

        inputs = {
            "state": _tele_gsy_np.asarray(data["state"], dtype=_tele_gsy_np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": _tele_gsy_np.True_,
                "left_wrist_0_rgb": _tele_gsy_np.True_,
                "right_wrist_0_rgb": right_wrist_mask,
            },
        }
        if "actions" in data:
            inputs["actions"] = _tele_gsy_np.asarray(data["actions"], dtype=_tele_gsy_np.float32)
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class TeleGsyUR5eOutputs(_transforms.DataTransformFn):
    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        # UR5e action convention:
        # [dx, dy, dz, droll, dpitch, dyaw, gripper]
        # The gripper value is absolute and is not delta-converted.
        return {"actions": _tele_gsy_np.asarray(data["actions"][:, : self.action_dim])}


@dataclasses.dataclass(frozen=True)
class TeleGsyLeRobotUR5eDataConfig(DataConfigFactory):
    expected_state_dim: int | None = None
    action_dim: int = 7
    camera_padding_strategy: str = "duplicate_base"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "base_rgb": "observation.images.base_rgb",
                        "wrist_rgb": "observation.images.wrist_rgb",
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "task",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                TeleGsyUR5eInputs(
                    model_type=model_config.model_type,
                    camera_padding_strategy=self.camera_padding_strategy,
                )
            ],
            outputs=[TeleGsyUR5eOutputs(action_dim=self.action_dim)],
        )
        model_transforms = ModelTransformFactory(
            default_prompt=_tele_gsy_os.environ.get(
                "PI0_UR5E_DEFAULT_PROMPT",
                "pick up the paper cup and place it on the target",
            )
        )(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )


_TELE_GSY_PI0_UR5E_LORA = _tele_gsy_env_bool("PI0_UR5E_LORA", True)
_TELE_GSY_PI0_UR5E_REPO_ID = _tele_gsy_os.environ.get("PI0_UR5E_LEROBOT_REPO_ID", "local/pi0_ur5e_cup")
_TELE_GSY_PI0_UR5E_MODEL = (
    pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    if _TELE_GSY_PI0_UR5E_LORA
    else pi0_config.Pi0Config()
)

_CONFIGS.append(
    TrainConfig(
        name="pi0_ur5e_cup",
        # Keep the base pi0 action dimension for checkpoint compatibility.
        # The dataset/policy action dimension is 7 and is sliced in TeleGsyUR5eOutputs.
        model=_TELE_GSY_PI0_UR5E_MODEL,
        data=TeleGsyLeRobotUR5eDataConfig(
            repo_id=_TELE_GSY_PI0_UR5E_REPO_ID,
            assets=AssetsConfig(asset_id=_tele_gsy_os.environ.get("PI0_UR5E_ASSET_ID", _TELE_GSY_PI0_UR5E_REPO_ID)),
            base_config=DataConfig(prompt_from_task=False),
            expected_state_dim=_tele_gsy_env_int("PI0_UR5E_STATE_DIM"),
            action_dim=7,
            camera_padding_strategy=_tele_gsy_os.environ.get("PI0_UR5E_CAMERA_PADDING", "duplicate_base"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=_tele_gsy_env_int("PI0_UR5E_TRAIN_STEPS", 3000),
        batch_size=_tele_gsy_env_int("PI0_UR5E_BATCH_SIZE", 16),
        assets_base_dir=_tele_gsy_os.environ.get("PI0_UR5E_ASSETS_BASE_DIR", "./assets"),
        checkpoint_base_dir=_tele_gsy_os.environ.get("PI0_UR5E_CHECKPOINT_BASE_DIR", "./checkpoints"),
        freeze_filter=_TELE_GSY_PI0_UR5E_MODEL.get_freeze_filter() if _TELE_GSY_PI0_UR5E_LORA else nnx.Nothing(),
        ema_decay=None if _TELE_GSY_PI0_UR5E_LORA else 0.99,
        policy_metadata={
            "robot_type": "ur5e",
            "action_dim": 7,
            "action_format": "ee_delta_6d_gripper",
            "action_order": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
            "gripper_is_delta": False,
            "camera_padding_strategy": _tele_gsy_os.environ.get("PI0_UR5E_CAMERA_PADDING", "duplicate_base"),
        },
    )
)
# END TELE_GSY_PI0_UR5E_CUP
