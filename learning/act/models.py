import math
import os
import sys

import torch
from torch import nn

DP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dp"))
if DP_PATH not in sys.path:
    sys.path.insert(0, DP_PATH)

from models import GaussianNoise, ImageEncoder, StateEncoder  # noqa: E402


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]].to(dtype=x.dtype)


class ACTModel(nn.Module):
    """Action Chunking Transformer with an optional CVAE latent bottleneck."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        obs_horizon,
        pred_horizon,
        hidden_dim=256,
        nheads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        latent_dim=32,
        use_vae=True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.latent_dim = latent_dim
        self.use_vae = use_vae

        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_encoding = SinusoidalPositionEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.query_embed = nn.Embedding(pred_horizon, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

        if use_vae:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.latent_encoder = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                ),
                num_layers=max(1, num_encoder_layers // 2),
            )
            self.latent_mu = nn.Linear(hidden_dim, latent_dim)
            self.latent_logvar = nn.Linear(hidden_dim, latent_dim)

    def encode_latent(self, obs_features, actions=None):
        batch_size = obs_features.shape[0]
        if not self.use_vae:
            zeros = obs_features.new_zeros(batch_size, self.latent_dim)
            return zeros, zeros, zeros

        if actions is None:
            zeros = obs_features.new_zeros(batch_size, self.latent_dim)
            return zeros, zeros, zeros

        cls = self.cls_token.expand(batch_size, -1, -1)
        latent_tokens = torch.cat(
            [cls, self.obs_proj(obs_features), self.action_proj(actions)], dim=1
        )
        latent_tokens = self.pos_encoding(latent_tokens)
        latent_state = self.latent_encoder(latent_tokens)[:, 0]
        mu = self.latent_mu(latent_state)
        logvar = self.latent_logvar(latent_state)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

    def forward(self, obs_features, actions=None):
        batch_size = obs_features.shape[0]
        z, mu, logvar = self.encode_latent(obs_features, actions)

        memory = self.obs_proj(obs_features)
        latent_token = self.latent_proj(z).unsqueeze(1)
        memory = torch.cat([latent_token, memory], dim=1)
        memory = self.encoder(self.pos_encoding(memory))

        query = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(query, memory)
        pred_actions = self.action_head(decoded)
        return pred_actions, mu, logvar


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
