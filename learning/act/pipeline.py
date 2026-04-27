import argparse
import importlib.util
import os
import pickle
import re
import sys
import time

import numpy as np
import torch

ACT_PATH = os.path.dirname(os.path.realpath(__file__))
DP_PATH = os.path.abspath(os.path.join(ACT_PATH, "..", "dp"))
REPO_ROOT = os.path.abspath(os.path.join(ACT_PATH, "..", ".."))
for path in (REPO_ROOT, DP_PATH):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

_dp_spec = importlib.util.spec_from_file_location(
    "dp_pipeline_for_act", os.path.join(DP_PATH, "pipeline.py")
)
_dp_pipeline = importlib.util.module_from_spec(_dp_spec)
_dp_spec.loader.exec_module(_dp_pipeline)

from learning.act.learner import ACTPolicy  # noqa: E402

data_processing = _dp_pipeline.data_processing
DiffusionAgent = _dp_pipeline.Agent
boolean_string = _dp_pipeline.boolean_string
WandBLogger = _dp_pipeline.WandBLogger
generate_random_string = _dp_pipeline.generate_random_string
save_args = _dp_pipeline.save_args


class Agent(DiffusionAgent):
    def __init__(
        self,
        act_hidden_dim=256,
        act_nheads=8,
        act_encoder_layers=4,
        act_decoder_layers=4,
        act_dim_feedforward=1024,
        act_dropout=0.1,
        act_latent_dim=32,
        act_kl_weight=10.0,
        act_use_vae=True,
        weight_decay=1e-6,
        binarize_touch=False,
        **kwargs,
    ):
        requested_compile_train = kwargs.get("compile_train", False)
        kwargs["compile_train"] = False
        super().__init__(
            without_sampling=True,
            weight_decay=weight_decay,
            binarize_touch=binarize_touch,
            **kwargs,
        )
        self.compile_train = requested_compile_train
        old_policy = self.policy
        encoders = {
            rt: old_policy.nets[f"{rt}_encoder"] for rt in self.representation_type
        }
        self.policy = ACTPolicy(
            obs_horizon=self.obs_horizon,
            obs_dim=old_policy.obs_dim,
            pred_horizon=self.pred_horizon,
            action_horizon=self.action_horizon,
            action_dim=self.rt_dim["action"],
            representation_type=self.representation_type,
            encoders=encoders,
            hidden_dim=act_hidden_dim,
            nheads=act_nheads,
            num_encoder_layers=act_encoder_layers,
            num_decoder_layers=act_decoder_layers,
            dim_feedforward=act_dim_feedforward,
            dropout=act_dropout,
            latent_dim=act_latent_dim,
            kl_weight=act_kl_weight,
            use_vae=act_use_vae,
            weight_decay=weight_decay,
            binarize_touch=binarize_touch,
        )
        self.policy.to(self.device)

        if self.compile_train:
            self.policy.nets["act_actor"].forward = torch.compile(
                self.policy.nets["act_actor"].forward
            )


def build_arg_parser():
    args = argparse.ArgumentParser()
    args.add_argument("--batch_size", type=int, default=32)
    args.add_argument("--obs_horizon", type=int, default=1)
    args.add_argument("--action_horizon", type=int, default=8)
    args.add_argument("--pred_horizon", type=int, default=16)
    args.add_argument("--epochs", type=int, default=300)
    args.add_argument("--traj_type", type=str, default="plain")
    args.add_argument("--prefix", type=str, default=None)
    args.add_argument("--save_path", type=str, default=None)
    args.add_argument("--load_path", type=str, default=None)
    args.add_argument("--eval", type=boolean_string, default=False)
    args.add_argument("--representation_type", type=str, default="img-depth-eef-pos-touch")
    args.add_argument("--base_path", type=str, default="./shared")
    args.add_argument("--data_name", type=str, default="test_data")
    args.add_argument("--data_path", type=str, default=None)
    args.add_argument("--data_prefix", type=str, default=None)
    args.add_argument("--model_save_path", type=str, default=None)
    args.add_argument("--clip_far", type=boolean_string, default=False)
    args.add_argument("--color_jitter", type=boolean_string, default=False)
    args.add_argument("--predict_eef_delta", type=boolean_string, default=False)
    args.add_argument("--predict_pos_delta", type=boolean_string, default=False)
    args.add_argument("--dropout_rate", type=float, default=0.0)
    args.add_argument("--img_dropout_rate", type=float, default=0.0)
    args.add_argument("--weight_decay", type=float, default=1e-5)
    args.add_argument("--image_output_size", type=int, default=32)
    args.add_argument("--joint_state_dim", type=int, default=7)
    args.add_argument("--action_dim", type=int, default=7)
    args.add_argument("--eef_dim", type=int, default=6)
    args.add_argument("--touch_dim", type=int, default=60)
    args.add_argument("--hand_pos_dim", type=int, default=12)
    args.add_argument("--state_noise", type=float, default=0.0)
    args.add_argument("--img_gaussian_noise", type=float, default=0.0)
    args.add_argument("--img_masking_prob", type=float, default=0.0)
    args.add_argument("--img_patch_size", type=int, default=16)
    args.add_argument("--num_workers", type=int, default=16)
    args.add_argument("--eval_path", type=str, default=None)
    args.add_argument("--identity_encoder", type=boolean_string, default=False)
    args.add_argument("--gpu", type=int, default=0)
    args.add_argument("--camera_indices", type=str, default="01")
    args.add_argument("--auto_infer_data_shapes", type=boolean_string, default=True)
    args.add_argument("--save_freq", type=int, default=10)
    args.add_argument("--eval_freq", type=int, default=10)
    args.add_argument("--add_model_save_path_suffix", type=boolean_string, default=True)
    args.add_argument("--use_wandb", type=boolean_string, default=False)
    args.add_argument("--binarize_touch", type=boolean_string, default=False)
    args.add_argument("--wandb_exp_name", type=str, default=None)
    args.add_argument("--load_img", type=boolean_string, default=False)
    args.add_argument("--train_suffix", type=str, default="")
    args.add_argument("--use_train_test_split", type=boolean_string, default=True)
    args.add_argument("--use_memmap_cache", type=boolean_string, default=False)
    args.add_argument("--memmap_loader_path", type=str, default=None)
    args.add_argument("--prepare_cache_only", type=boolean_string, default=False)
    args.add_argument("--compile_train", type=boolean_string, default=False)
    args.add_argument("--wandb_entity_name", type=str, default=None)
    args.add_argument("--wandb_project_name", type=str, default=None)
    args.add_argument("--act_hidden_dim", type=int, default=256)
    args.add_argument("--act_nheads", type=int, default=8)
    args.add_argument("--act_encoder_layers", type=int, default=4)
    args.add_argument("--act_decoder_layers", type=int, default=4)
    args.add_argument("--act_dim_feedforward", type=int, default=1024)
    args.add_argument("--act_dropout", type=float, default=0.1)
    args.add_argument("--act_latent_dim", type=int, default=32)
    args.add_argument("--act_kl_weight", type=float, default=10.0)
    args.add_argument("--act_use_vae", type=boolean_string, default=True)
    return args


def _resolve_data_paths(args):
    if args.data_path is not None:
        data_path = args.data_path
    elif args.data_prefix is not None:
        data_path = os.path.join(args.base_path, args.data_prefix, args.data_name)
    else:
        data_path = os.path.join(args.base_path, args.data_name)

    if args.use_train_test_split:
        train_path = data_path + "_train" + args.train_suffix
        test_path = data_path + "_test"
    else:
        train_path = test_path = None
    return data_path, train_path, test_path


def _first_episode(path, traj_type, prefix):
    if path is None or not os.path.exists(path):
        return None
    if os.path.isfile(path):
        return os.path.dirname(path)
    if os.path.exists(os.path.join(path, "trajectory.h5")) or _first_pickle_path(path) is not None:
        return path
    episodes = data_processing.get_epi_dir(path, traj_type=traj_type, prefix=prefix)
    return episodes[0] if episodes else None


def _first_pickle_path(episode_path):
    for name in sorted(os.listdir(episode_path)):
        if name.endswith(".pkl"):
            return os.path.join(episode_path, name)
    return None


def _first_frame(episode_path):
    if episode_path is None:
        return None
    h5_path = os.path.join(episode_path, "trajectory.h5")
    if os.path.exists(h5_path):
        try:
            data = data_processing.from_h5(h5_path, load_img=False)
        except RuntimeError as exc:
            print(f"Auto shape inference skipped for {h5_path}: {exc}")
            return None
        return data[0] if data else None
    pkl_path = _first_pickle_path(episode_path)
    if pkl_path is None:
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _last_dim(frame, key):
    if frame is None or key not in frame:
        return None
    value = np.asarray(frame[key])
    if value.ndim == 0:
        return 1
    return int(value.shape[-1])


def _camera_count_from_h5(episode_path):
    h5_path = os.path.join(episode_path, "trajectory.h5")
    if not os.path.exists(h5_path):
        return None
    try:
        import h5py
    except ImportError:
        return None
    with h5py.File(h5_path, "r") as f:
        counts = []
        for group_name in ("videos", "frames"):
            if group_name not in f:
                continue
            indices = []
            for key in f[group_name].keys():
                match = re.match(r"^(base_rgb|base_camera_rgb|base_depth|base_camera_depth)_(\d+)$", key)
                if match:
                    indices.append(int(match.group(2)))
            if indices:
                counts.append(max(indices) + 1)
        return max(counts) if counts else None


def _camera_count_from_pickle(episode_path, frame):
    if frame is not None:
        for key in ("base_rgb", "base_camera_rgb", "base_depth", "base_camera_depth"):
            if key in frame:
                value = np.asarray(frame[key])
                if value.ndim >= 4:
                    return int(value.shape[0])
    pkl_path = _first_pickle_path(episode_path)
    if pkl_path is None:
        return None
    count = 0
    stem, _ = os.path.splitext(pkl_path)
    while os.path.exists(f"{stem}-{count}.png"):
        count += 1
    return count or None


def _auto_infer_data_shapes(args, data_path, train_path):
    if not args.auto_infer_data_shapes:
        return

    source_path = train_path if train_path is not None and os.path.exists(train_path) else data_path
    episode_path = _first_episode(source_path, args.traj_type, args.prefix)
    if episode_path is None:
        print("Auto shape inference skipped: no episode found.")
        return

    frame = _first_frame(episode_path)
    inferred = {}
    for attr, key in (
        ("joint_state_dim", "joint_positions"),
        ("action_dim", "control"),
        ("eef_dim", "ee_pos_quat"),
        ("touch_dim", "touch"),
        ("hand_pos_dim", "hand_pos"),
    ):
        dim = _last_dim(frame, key)
        if dim is not None:
            setattr(args, attr, dim)
            inferred[attr] = dim

    camera_count = _camera_count_from_h5(episode_path)
    if camera_count is None:
        camera_count = _camera_count_from_pickle(episode_path, frame)
    if camera_count is not None:
        requested = list(map(int, args.camera_indices))
        if requested and max(requested) >= camera_count:
            args.camera_indices = "".join(str(i) for i in range(camera_count))
            inferred["camera_indices"] = args.camera_indices

    if inferred:
        print("Auto-inferred data shapes:", inferred)


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    if args.gpu is not None:
        torch.cuda.set_device(f"cuda:{args.gpu}")
        print(f"Using gpu: {args.gpu}")

    curr_time = time.strftime("%m%d_%H%M%S", time.localtime())
    model_id = f"{curr_time}_{generate_random_string()}"

    if args.use_wandb:
        wandb_config = WandBLogger.get_default_config()
        wandb_config.entity = args.wandb_entity_name
        wandb_config.project = args.wandb_project_name
        wandb_config.exp_name = (
            args.wandb_exp_name + "_" + model_id
            if args.wandb_exp_name is not None
            else model_id
        )
        wandb_logger = WandBLogger(config=wandb_config, variant=vars(args), prefix="logging")
    else:
        wandb_logger = None

    data_path, train_path, test_path = _resolve_data_paths(args)
    infer_data_path = args.eval_path if args.eval and args.eval_path is not None else data_path
    infer_train_path = None if args.eval else train_path
    _auto_infer_data_shapes(args, infer_data_path, infer_train_path)

    agent = Agent(
        dropout={
            "eef": args.dropout_rate,
            "hand_pos": args.dropout_rate,
            "img": args.img_dropout_rate,
            "tactile_img": args.img_dropout_rate,
            "pos": args.dropout_rate,
            "touch": args.dropout_rate,
        },
        output_sizes={
            "eef": 64,
            "hand_pos": 64,
            "img": args.image_output_size,
            "tactile_img": args.image_output_size,
            "pos": 128,
            "touch": 64,
        },
        representation_type=args.representation_type.split("-"),
        identity_encoder=args.identity_encoder,
        camera_indices=list(map(int, args.camera_indices)),
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_horizon=args.action_horizon,
        predict_eef_delta=args.predict_eef_delta,
        predict_pos_delta=args.predict_pos_delta,
        clip_far=args.clip_far,
        color_jitter=args.color_jitter,
        load_img=args.load_img,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        binarize_touch=args.binarize_touch,
        state_noise=args.state_noise,
        img_gaussian_noise=args.img_gaussian_noise,
        img_masking_prob=args.img_masking_prob,
        img_patch_size=args.img_patch_size,
        compile_train=args.compile_train,
        joint_state_dim=args.joint_state_dim,
        action_dim=args.action_dim,
        eef_state_dim=args.eef_dim,
        touch_dim=args.touch_dim,
        hand_pos_dim=args.hand_pos_dim,
        act_hidden_dim=args.act_hidden_dim,
        act_nheads=args.act_nheads,
        act_encoder_layers=args.act_encoder_layers,
        act_decoder_layers=args.act_decoder_layers,
        act_dim_feedforward=args.act_dim_feedforward,
        act_dropout=args.act_dropout,
        act_latent_dim=args.act_latent_dim,
        act_kl_weight=args.act_kl_weight,
        act_use_vae=args.act_use_vae,
    )
    if args.load_path is not None:
        agent.load(args.load_path)

    if args.eval:
        agent.eval(args.eval_path, save_path=args.save_path)
        sys.exit(0)

    if args.model_save_path is None:
        data_name = args.data_name.replace("/", "-")
        model_path = os.path.join(args.base_path, "ckpts", data_name)
    else:
        model_path = args.model_save_path

    model_path_suffix = f"{model_id}"
    if args.add_model_save_path_suffix:
        args_str = "-".join(
            [
                f"camera={args.camera_indices}",
                f"identity={args.identity_encoder}",
                f"repr={''.join([(x[0]).upper() for x in args.representation_type.split('-')])}",
                f"oh={args.obs_horizon}",
                f"ah={args.action_horizon}",
                f"ph={args.pred_horizon}",
                f"prefix={args.prefix}",
                f"h={args.act_hidden_dim}",
                f"latent={args.act_latent_dim}",
                f"kl={args.act_kl_weight}",
            ]
        )
        if args.predict_pos_delta:
            args_str += "-posdelta"
        if args.predict_eef_delta:
            args_str += "-eefdelta"
        model_path_suffix += "-" + args_str
    model_path = os.path.join(model_path, model_path_suffix)
    os.makedirs(model_path, exist_ok=True)
    save_args(args, model_path)

    use_depth = "depth" in args.representation_type.split("-")
    if args.use_memmap_cache:
        if args.memmap_loader_path is not None:
            memmap_loader_path = args.memmap_loader_path
            if test_path is not None:
                _, memmap_name = os.path.split(memmap_loader_path)
                eval_memmap_loader_path = os.path.join(test_path, memmap_name)
            else:
                eval_memmap_loader_path = memmap_loader_path
        else:
            memmap_base_path = train_path if train_path is not None else data_path
            memmap_loader_path = os.path.join(
                memmap_base_path, f"{args.camera_indices}-{use_depth}-mem.dat"
            )
            eval_memmap_base_path = test_path if test_path is not None else memmap_base_path
            eval_memmap_loader_path = os.path.join(
                eval_memmap_base_path, f"{args.camera_indices}-{use_depth}-mem.dat"
            )
    else:
        memmap_loader_path = ""
        eval_memmap_loader_path = ""

    print(f"Saving to model path {model_path}")
    print(f"using data path {data_path}")
    print("using memmap loader path:", memmap_loader_path)
    print("using eval memmap loader path:", eval_memmap_loader_path)

    if args.prepare_cache_only:
        cache_source_path = train_path if train_path is not None else data_path
        print(f"Preparing cache from {cache_source_path}")
        agent.epi_dir = data_processing.get_epi_dir(
            cache_source_path, traj_type=args.traj_type, prefix=args.prefix
        )
        agent.get_train_loader(
            batch_size=args.batch_size, memmap_loader_path=memmap_loader_path
        )
        print("Cache preparation finished.")
        sys.exit(0)

    agent.train(
        data_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        traj_type=args.traj_type,
        prefix=args.prefix,
        save_path=model_path,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        wandb_logger=wandb_logger,
        train_path=train_path,
        test_path=test_path,
        memmap_loader_path=memmap_loader_path,
        eval_memmap_loader_path=eval_memmap_loader_path,
    )
