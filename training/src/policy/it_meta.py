"""InteractionTransformerMeta — Flax port.

Architecture matches the PyTorch reference in
``human-following-robot/rl/networks/interaction_transformer{,_meta}.py``:

    OGM_CNN (3 conv + adaptive max pool + FC)              → 256
    robot_embedding (10 → 128 → 256)        [meta: +1 dim]
    human_embedding (12 → 128 → 256)
    obstacle_embedding (256 → 256, linear)
    target_embedding (12 → 128 → 256)
    TransformerEncoder (4 layers, 8 heads, dim_ff=1024)
    transformer_output_layer (256 → 128 → 64)
    actor head (64 → 64 → 64, tanh)
    critic head (64 → 64 → 64, tanh)
    critic_linear (64 → 1)
    DiagGaussian (64 → 2) — mean + learnable log_std

Note: actor outputs `hidden_actor` (64-dim) that gets passed to a
DiagGaussian head which produces action mean + log_std. This matches the
original `Policy` wrapper in `rl/networks/distributions.py`.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn


# PPO best-practice initializers (matches PyTorch original's
# `orthogonal_(sqrt(2))` + zero bias used throughout actor/critic).
_ortho2 = nn.initializers.orthogonal(jnp.sqrt(2.0))
_ortho1 = nn.initializers.orthogonal(1.0)
_ortho_small = nn.initializers.orthogonal(0.01)   # action-mean: small init for stable initial policy
_zeros = nn.initializers.zeros


def _sinusoidal_pe(seq_len: int, d_model: int) -> jax.Array:
    """Standard sinusoidal positional encoding matching PyTorch original
    (`interaction_transformer.py:PositionalEncoding`)."""
    pos = jnp.arange(seq_len)[:, None].astype(jnp.float32)
    i = jnp.arange(0, d_model, 2).astype(jnp.float32)
    div = jnp.exp(i * (-jnp.log(10000.0) / d_model))
    pe = jnp.zeros((seq_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(pos * div))
    pe = pe.at[:, 1::2].set(jnp.cos(pos * div))
    return pe[None]


# ----- CNN -----------------------------------------------------------------
class OGMCNN(nn.Module):
    """3-stack OGM → 256-dim feature vector.

    Input ``(B, T=3, H=50, W=50)`` int8 → cast to float, conv stack,
    adaptive max pool, FC.
    """
    output_dim: int = 256
    temporal_frames: int = 3

    @nn.compact
    def __call__(self, x):
        # x: (B, T, H, W); cast to float and move T to channel dim
        x = x.astype(jnp.float32)
        # Flax conv expects NHWC; original PyTorch is NCHW (T as C).
        # Transpose: (B, T, H, W) -> (B, H, W, T)
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = nn.Conv(features=64, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        # Adaptive max pool to (2, 2)
        # Flax doesn't have AdaptiveMaxPool; use jax.image.resize on max-pool result
        # Simpler: global max with reduce over spatial blocks
        H_out, W_out = 2, 2
        H_in, W_in = x.shape[1], x.shape[2]
        # Use nn.max_pool with window = ceil(H_in / H_out), stride = floor(H_in / H_out)
        wh, ww = H_in // H_out, W_in // W_out
        x = nn.max_pool(x, window_shape=(wh, ww), strides=(wh, ww), padding='VALID')
        # Now (B, 2, 2, 256). Flatten and FC.
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x  # (B, 256)


# ----- Transformer ---------------------------------------------------------
class TransformerEncoderBlock(nn.Module):
    feature_dim: int = 256
    nhead: int = 8
    dim_ff: int = 1024
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, mask=None):
        # x: (B, S, D)
        # Self-attention
        attn_out = nn.SelfAttention(
            num_heads=self.nhead, qkv_features=self.feature_dim,
            out_features=self.feature_dim, dropout_rate=self.dropout,
            deterministic=True,
        )(x, mask=mask)
        x = nn.LayerNorm()(x + attn_out)
        # Feed-forward
        y = nn.Dense(self.dim_ff)(x)
        y = nn.relu(y)
        y = nn.Dense(self.feature_dim)(y)
        x = nn.LayerNorm()(x + y)
        return x


class TransformerEncoder(nn.Module):
    feature_dim: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_ff: int = 1024

    @nn.compact
    def __call__(self, x, mask=None):
        for _ in range(self.num_layers):
            x = TransformerEncoderBlock(
                feature_dim=self.feature_dim,
                nhead=self.nhead,
                dim_ff=self.dim_ff,
            )(x, mask=mask)
        x = nn.Dense(self.feature_dim)(x)
        return x


# ----- Policy --------------------------------------------------------------
class ITMetaPolicy(nn.Module):
    """Full meta-guided policy: obs dict → (value, action_mean, action_log_std).

    Output dims:
      value: (B, 1)
      action_mean: (B, action_dim)
      action_log_std: (action_dim,) — learnable, shared across batch
    """
    action_dim: int = 2
    max_human_num: int = 45
    feature_dim: int = 256
    output_size: int = 64

    @nn.compact
    def __call__(self, obs):
        B = obs['robot_node'].shape[0]

        # 1) Embed each modality to feature_dim=256 (orthogonal(sqrt(2)) init throughout)
        # robot_states = concat(following_preference[1], temporal_edges[2], robot_node[7]) = 10
        robot_states = jnp.concatenate([
            obs['following_preference'].reshape(B, 1, 1),
            obs['temporal_edges'].reshape(B, 1, 2),
            obs['robot_node'].reshape(B, 1, 7),
        ], axis=-1)  # (B, 1, 10)
        robot_embed = nn.Dense(128, kernel_init=_ortho2, bias_init=_zeros)(robot_states)
        robot_embed = nn.relu(robot_embed)
        robot_embed = nn.Dense(self.feature_dim, kernel_init=_ortho2, bias_init=_zeros)(robot_embed)

        # human_embedding 12 → 128 → 256
        human_embed = nn.Dense(128, kernel_init=_ortho2, bias_init=_zeros)(obs['spatial_edges'])
        human_embed = nn.relu(human_embed)
        human_embed = nn.Dense(self.feature_dim, kernel_init=_ortho2, bias_init=_zeros)(human_embed)

        # target_embedding 12 → 128 → 256
        target_embed = nn.Dense(128, kernel_init=_ortho2, bias_init=_zeros)(obs['target_human_traj'])
        target_embed = nn.relu(target_embed)
        target_embed = nn.Dense(self.feature_dim, kernel_init=_ortho2, bias_init=_zeros)(target_embed)
        target_embed = target_embed[:, None, :]  # (B, 1, 256)

        # obstacle: CNN(local_ogm) then linear 256 → 256
        ogm_feat = OGMCNN(output_dim=self.feature_dim)(obs['local_ogm'])  # (B, 256)
        obstacle_embed = nn.Dense(self.feature_dim, kernel_init=_ortho2, bias_init=_zeros)(ogm_feat)[:, None, :]

        # 2) Concat to sequence (B, 3+M, 256): robot, target, obstacle, humans
        sequence = jnp.concatenate(
            [robot_embed, target_embed, obstacle_embed, human_embed], axis=1
        )

        # 2b) Add sinusoidal positional encoding (paper interaction_transformer.py:61-82)
        sequence = sequence + _sinusoidal_pe(sequence.shape[1], self.feature_dim)

        # 3) Attention mask: robot/target/obstacle always valid, humans by detected count
        det = obs['detected_human_num'].astype(jnp.int32).reshape(B)
        idx = jnp.arange(self.max_human_num)[None, :]
        human_mask = idx < det[:, None]
        prefix = jnp.ones((B, 3), dtype=bool)
        attn_mask = jnp.concatenate([prefix, human_mask], axis=1)
        kv_mask = attn_mask[:, None, None, :]

        # 4) Transformer
        seq_out = TransformerEncoder(
            feature_dim=self.feature_dim, nhead=8, num_layers=4, dim_ff=1024
        )(sequence, mask=kv_mask)

        # 5) Robot token (index 0) → output projection
        robot_out = seq_out[:, 0, :]
        h = nn.Dense(self.feature_dim // 2, kernel_init=_ortho2, bias_init=_zeros)(robot_out)
        h = nn.relu(h)
        h = nn.Dense(self.output_size, kernel_init=_ortho2, bias_init=_zeros)(h)

        # 6) Actor + critic heads (each 64→64→64 with tanh, orthogonal init)
        actor_h = nn.tanh(nn.Dense(self.output_size, kernel_init=_ortho2, bias_init=_zeros)(h))
        actor_h = nn.tanh(nn.Dense(self.output_size, kernel_init=_ortho2, bias_init=_zeros)(actor_h))
        critic_h = nn.tanh(nn.Dense(self.output_size, kernel_init=_ortho2, bias_init=_zeros)(h))
        critic_h = nn.tanh(nn.Dense(self.output_size, kernel_init=_ortho2, bias_init=_zeros)(critic_h))

        # 7) Value (gain=1.0) + action mean (gain=0.01, PPO small-final-layer trick) + log_std
        value = nn.Dense(1, kernel_init=_ortho1, bias_init=_zeros)(critic_h)
        action_mean = nn.Dense(self.action_dim, kernel_init=_ortho_small, bias_init=_zeros)(actor_h)
        action_log_std = self.param('action_log_std', _zeros, (self.action_dim,))
        return value, action_mean, action_log_std


def init_policy(key, obs_template, max_human_num=45, action_dim=2):
    """Initialize policy params on a single-batch dummy obs."""
    model = ITMetaPolicy(action_dim=action_dim, max_human_num=max_human_num)
    params = model.init(key, obs_template)
    return model, params


def make_dummy_obs(B=1, max_human_num=45, predict_steps=5, ogm_size=50, history=3):
    return {
        'robot_node': jnp.zeros((B, 1, 7)),
        'temporal_edges': jnp.zeros((B, 1, 2)),
        'spatial_edges': jnp.zeros((B, max_human_num, 2 * (predict_steps + 1))),
        'detected_human_num': jnp.zeros((B, 1)),
        'target_human_traj': jnp.zeros((B, 2 * (predict_steps + 1))),
        'local_ogm': jnp.zeros((B, history, ogm_size, ogm_size), dtype=jnp.int8),
        'following_preference': jnp.zeros((B, 1, 1)),
    }
