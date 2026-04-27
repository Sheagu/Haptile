import collections
import copy
import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

try:
    from .models import ACTModel, kl_divergence
except ImportError:
    from models import ACTModel, kl_divergence

IMAGE_KEYS = {"img", "tactile_img"}
FEATURE_ORDER = ["eef", "hand_pos", "img", "tactile_img", "pos", "touch"]


def normalize_data(data, stats):
    ndata = (data - stats["min"]) / ((stats["max"] - stats["min"]) + 1e-8)
    return ndata * 2 - 1


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    return ndata * (stats["max"] - stats["min"] + 1e-8) + stats["min"]


class ACTPolicy:
    def __init__(
        self,
        obs_horizon,
        obs_dim,
        pred_horizon,
        action_horizon,
        action_dim,
        representation_type,
        encoders,
        hidden_dim=256,
        nheads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        latent_dim=32,
        kl_weight=10.0,
        use_vae=True,
        weight_decay=1e-6,
        binarize_touch=False,
    ):
        for rt in representation_type:
            assert rt in encoders, f"{rt} not in encoders"
        self.representation_type = representation_type
        self.obs_horizon = obs_horizon
        self.obs_dim = obs_dim
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.data_stat = None
        self.writer = None
        self.binarize_touch = binarize_touch
        self.kl_weight = kl_weight

        actor = ACTModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            hidden_dim=hidden_dim,
            nheads=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            latent_dim=latent_dim,
            use_vae=use_vae,
        )
        self.nets = nn.ModuleDict({"act_actor": actor})
        for rt in representation_type:
            self.nets[f"{rt}_encoder"] = encoders[rt]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ema_nets = copy.deepcopy(self.nets)
        self.optimizer = torch.optim.AdamW(
            params=self.nets.parameters(), lr=1e-4, weight_decay=weight_decay
        )
        self.lr_scheduler = None

    def set_lr_scheduler(self, num_training_steps):
        warmup_steps = min(500, max(1, num_training_steps // 20))

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def to(self, device):
        self.device = device
        self.nets.to(device)
        self.ema_nets.to(device)

    def to_ema(self):
        self.ema_nets.load_state_dict(self.nets.state_dict())

    def _encode_batch(self, nbatch, nets):
        features = []
        for data_key in [dk for dk in FEATURE_ORDER if dk in self.representation_type]:
            nsample = nbatch[data_key][:, : self.obs_horizon].to(self.device)
            if data_key in IMAGE_KEYS:
                images = [nsample[:, :, i] for i in range(nsample.shape[2])]
                image_features = [
                    nets[f"{data_key}_encoder"][i](image.flatten(end_dim=1))
                    for i, image in enumerate(images)
                ]
                image_features = torch.stack(image_features, dim=2)
                image_features = image_features.reshape(*nsample.shape[:2], -1)
                features.append(image_features)
            else:
                nfeat = nets[f"{data_key}_encoder"](nsample.flatten(end_dim=1))
                features.append(nfeat.reshape(*nsample.shape[:2], -1))
        return torch.cat(features, dim=-1)

    def train(
        self,
        num_epochs,
        dataloader,
        eval_data=None,
        save_path=None,
        save_freq=10,
        eval_freq=10,
        wandb_logger=None,
        eval=False,
    ):
        nets = self.ema_nets if eval else self.nets
        nets.eval() if eval else nets.train()
        action_mse = []

        if self.writer is None and save_path is not None:
            self.writer = SummaryWriter(os.path.join("./runs", os.path.basename(save_path)))

        with tqdm(range(num_epochs), desc="Epoch") as tglobal:
            for epoch_idx in tglobal:
                epoch_loss = []
                with tqdm(dataloader, desc="Batch", leave=False) as tepoch:
                    for nbatch in tepoch:
                        naction = nbatch["action"].to(self.device)
                        obs_features = self._encode_batch(nbatch, nets)
                        pred_action, mu, logvar = nets["act_actor"](obs_features, naction)
                        recon_loss = mse_loss(pred_action, naction)
                        kl_loss = kl_divergence(mu, logvar)
                        loss = recon_loss + self.kl_weight * kl_loss

                        if not eval:
                            loss.backward()
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            if self.lr_scheduler is not None:
                                self.lr_scheduler.step()
                        else:
                            unnormalized_naction = unnormalize_data(
                                naction.detach().cpu().numpy(), self.data_stat["action"]
                            )
                            unnormalized_pred_action = unnormalize_data(
                                pred_action.detach().cpu().numpy(), self.data_stat["action"]
                            )
                            action_mse.append(
                                mse_loss(
                                    torch.tensor(unnormalized_naction),
                                    torch.tensor(unnormalized_pred_action),
                                ).item()
                            )

                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu, recon=recon_loss.item(), kl=kl_loss.item())

                mean_loss = float(np.mean(epoch_loss))
                tglobal.set_postfix(loss=mean_loss)
                if self.writer is not None:
                    self.writer.add_scalar("Loss", mean_loss, epoch_idx)
                if eval:
                    return mean_loss, float(np.mean(action_mse))
                if wandb_logger is not None:
                    wandb_logger.step()
                    wandb_logger.log({"Loss": mean_loss, "epoch": epoch_idx})
                if save_path is not None and epoch_idx % save_freq == 0 and epoch_idx != 0:
                    self.save(os.path.join(save_path, f"model_epoch_{epoch_idx}.ckpt"))
                if save_path is not None:
                    self.save(os.path.join(save_path, "last.ckpt"))

                if eval_data is not None and epoch_idx % eval_freq == 0:
                    self.to_ema()
                    if callable(eval_data):
                        eval_label, current_eval_data = eval_data(epoch_idx)
                    else:
                        eval_label, current_eval_data = "eval", eval_data
                    print(f"Evaluating trajectory: {eval_label}")
                    obs, action = current_eval_data
                    _, mse, normalized_mse = self.eval(obs, action)
                    self.writer.add_scalar("Action_MSE", mse, epoch_idx)
                    self.writer.add_scalar("Normalized_MSE", normalized_mse, epoch_idx)
                    if wandb_logger is not None:
                        wandb_logger.log({"Action_MSE": mse, "Normalized_MSE": normalized_mse})
                    print(f"Action_MSE: {mse}, Normalized_MSE: {normalized_mse}")

    def _get_data_forward(self, stats, obs_deque, data_key):
        sample = np.stack([x[data_key] for x in obs_deque])
        if data_key not in IMAGE_KEYS and (data_key != "touch" or not self.binarize_touch):
            sample = normalize_data(sample, stats=stats[data_key])
        return torch.from_numpy(sample).to(self.device, dtype=torch.float32).unsqueeze(0)

    def forward(self, stats, obs_deque, num_diffusion_iters=None):
        del num_diffusion_iters
        self.ema_nets.eval()
        with torch.no_grad():
            features = []
            for data_key in [dk for dk in FEATURE_ORDER if dk in self.representation_type]:
                sample = self._get_data_forward(stats, obs_deque, data_key)
                if data_key in IMAGE_KEYS:
                    images = [sample[:, :, i] for i in range(sample.shape[2])]
                    image_features = [
                        self.ema_nets[f"{data_key}_encoder"][i](image.flatten(end_dim=1))
                        for i, image in enumerate(images)
                    ]
                    image_features = torch.stack(image_features, dim=2)
                    features.append(image_features.reshape(*sample.shape[:2], -1))
                else:
                    feat = self.ema_nets[f"{data_key}_encoder"](sample.flatten(end_dim=1))
                    features.append(feat.reshape(*sample.shape[:2], -1))
            obs_features = torch.cat(features, dim=-1)
            naction, _, _ = self.ema_nets["act_actor"](obs_features, actions=None)

        naction = naction.detach().cpu().numpy()[0]
        action_pred = unnormalize_data(naction, stats=stats["action"])
        start = self.obs_horizon - 1
        return action_pred[start : start + self.action_horizon]

    def eval(self, obs, action):
        obs_deque = collections.deque([obs[0]] * self.obs_horizon, maxlen=self.obs_horizon)
        actions_pred = []
        i = 0
        while i < len(obs) - self.action_horizon:
            action_pred = self.forward(self.data_stat, obs_deque)
            for j in range(self.action_horizon):
                actions_pred.append(action_pred[j])
                obs_deque.append(obs[i + j])
            i += self.action_horizon

        actions_pred = np.array(actions_pred)
        action = np.array(action)
        mse = mse_loss(torch.tensor(actions_pred), torch.tensor(action[: len(actions_pred)]))
        normalized_action = normalize_data(action, self.data_stat["action"])
        normalized_action_pred = normalize_data(actions_pred, self.data_stat["action"])
        normalized_mse = mse_loss(
            torch.tensor(normalized_action_pred),
            torch.tensor(normalized_action[: len(actions_pred)]),
        )
        return actions_pred, mse, normalized_mse

    def eval_loader(self, eval_loader):
        self.ema_nets.eval()
        return self.train(num_epochs=1, dataloader=eval_loader, eval=True)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.nets.load_state_dict(state_dict)
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        ema_path = os.path.join(dirname, "ema_" + basename)
        if os.path.exists(ema_path):
            self.ema_nets.load_state_dict(torch.load(ema_path, map_location=self.device))
        else:
            self.ema_nets.load_state_dict(state_dict)

    def save(self, path):
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self.to_ema()
        torch.save(self.ema_nets.state_dict(), os.path.join(dirname, "ema_" + basename))
        torch.save(self.nets.state_dict(), path)
