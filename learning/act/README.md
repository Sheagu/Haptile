# ACT policy

`learning/act` implements an Action Chunking Transformer policy that reuses the
same trajectory loading, observation preprocessing, normalization, action chunk
training, checkpointing, and evaluation flow as `learning/dp`.

## Training

Example:

```bash
python learning/act/pipeline.py \
  --data_path /path/to/dataset \
  --representation_type img-depth-eef-pos-touch \
  --obs_horizon 1 \
  --pred_horizon 16 \
  --action_horizon 8 \
  --batch_size 32 \
  --epochs 300 \
  --model_save_path /path/to/act_ckpts
```

The data directory conventions and most command-line flags match
`learning/dp/pipeline.py`. Defaults target the current single-arm UR5e data
layout: two RGB/RGBD camera slots (`--camera_indices 01`) and 7D joint/action
vectors. `--auto_infer_data_shapes True` inspects the first episode and updates
`action_dim`, `joint_state_dim`, `eef_dim`, `touch_dim`, and camera indices when
the dataset shape is different.

ACT-specific flags include:

- `--act_hidden_dim`: transformer width, default `256`.
- `--act_nheads`: attention heads, default `8`.
- `--act_encoder_layers`: observation transformer encoder depth, default `4`.
- `--act_decoder_layers`: action-query decoder depth, default `4`.
- `--act_latent_dim`: CVAE latent dimension, default `32`.
- `--act_kl_weight`: KL loss weight, default `10.0`.
- `--act_use_vae`: enable ACT-style CVAE training, default `True`.

For legacy dual-arm datasets, pass the old dimensions explicitly, for example:

```bash
python learning/act/pipeline.py \
  --data_path /path/to/dual_arm_dataset \
  --representation_type img-depth-eef-hand_pos-touch \
  --camera_indices 012 \
  --joint_state_dim 24 \
  --action_dim 24 \
  --eef_dim 12 \
  --hand_pos_dim 12
```

## Evaluation

```bash
python learning/act/pipeline.py \
  --eval True \
  --load_path /path/to/act_ckpts/last.ckpt \
  --eval_path /path/to/eval_episode \
  --save_path /path/to/eval_outputs
```

## Deployment

`run_env.py` supports local ACT inference with the same environment loop used by
diffusion policy:

```bash
python run_env.py \
  --agent act \
  --act_ckpt_path /path/to/act_ckpts/last.ckpt
```

Use `--agent act_eef` for checkpoints trained with `--predict_eef_delta True`.
If `--act_ckpt_path` is omitted, `run_env.py` falls back to `--dp_ckpt_path` for
backward-compatible launch scripts.
