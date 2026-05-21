"""All-on-GPU PPO training loop (PureJaxRL-style).

The entire training loop — env rollouts, GAE, multi-epoch minibatch updates,
optimizer step — is JIT-compiled into a single XLA program with ``jax.lax.scan``.
This means there's zero Python overhead per step once the loop is launched.

Hyperparams default to match the original ``arguments.py``:
  num_processes = 128
  num_steps = 30        (rollout length)
  ppo_epoch = 5
  num_mini_batch = 8
  clip_param = 0.02     (small — matches paper's settings)
  value_loss_coef = 0.5
  entropy_coef = 0.0
  lr = 4e-5
  gamma = 0.99
  gae_lambda = 0.95
"""

from __future__ import annotations

from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
import optax


class PPOConfig(NamedTuple):
    num_envs: int = 128
    num_steps: int = 30
    total_timesteps: int = 5_000_000
    ppo_epoch: int = 5
    num_mini_batch: int = 8
    clip_param: float = 0.02
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0
    lr: float = 4e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5


class Rollout(NamedTuple):
    obs: Any            # pytree of (T, B, ...)
    actions: jax.Array  # (T, B, A)
    log_probs: jax.Array  # (T, B)
    values: jax.Array   # (T, B)
    rewards: jax.Array  # (T, B)
    dones: jax.Array    # (T, B)


# ----- Distribution helpers -----------------------------------------------------
def gaussian_log_prob(mean, log_std, action):
    """Log prob of multivariate diagonal Gaussian. ``mean, action: (..., A)``."""
    std = jnp.exp(log_std)
    z = (action - mean) / std
    return (-0.5 * jnp.sum(z ** 2, axis=-1)
            - jnp.sum(log_std, axis=-1)
            - 0.5 * action.shape[-1] * jnp.log(2 * jnp.pi))


def sample_action(key, mean, log_std):
    """Sample action from diagonal Gaussian; return (action, log_prob)."""
    eps = jax.random.normal(key, mean.shape)
    action = mean + jnp.exp(log_std) * eps
    log_prob = gaussian_log_prob(mean, log_std, action)
    return action, log_prob


# ----- GAE --------------------------------------------------------------------
def compute_gae(rewards, values, dones, last_value, cfg: PPOConfig):
    """Generalised Advantage Estimation.

    All inputs ``(T, B)`` except ``last_value: (B,)``.
    Returns advantages ``(T, B)`` and returns ``(T, B)``.
    """
    T = rewards.shape[0]
    advantages = jnp.zeros_like(rewards)
    gae = jnp.zeros(rewards.shape[1])
    # We reverse-scan over time
    def body(carry, x):
        gae, next_value = carry
        rew, val, done = x
        not_done = 1.0 - done.astype(jnp.float32)
        delta = rew + cfg.gamma * next_value * not_done - val
        gae = delta + cfg.gamma * cfg.gae_lambda * not_done * gae
        return (gae, val), gae

    (gae_final, _), advantages = jax.lax.scan(
        body, (gae, last_value), (rewards, values, dones), reverse=True
    )
    returns = advantages + values
    return advantages, returns


# ----- PPO loss ---------------------------------------------------------------
def ppo_loss(params, apply_fn, obs, action, old_log_prob, advantage, ret,
             value_old, cfg: PPOConfig):
    """Standard PPO clipped objective."""
    value, mean, log_std = apply_fn(params, obs)
    value = value.squeeze(-1)
    log_prob = gaussian_log_prob(mean, log_std, action)
    ratio = jnp.exp(log_prob - old_log_prob)
    surrogate1 = ratio * advantage
    surrogate2 = jnp.clip(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * advantage
    policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

    # Value loss (clipped — match original ppo behavior)
    value_clipped = value_old + jnp.clip(value - value_old, -cfg.clip_param, cfg.clip_param)
    value_loss1 = (value - ret) ** 2
    value_loss2 = (value_clipped - ret) ** 2
    value_loss = 0.5 * jnp.mean(jnp.maximum(value_loss1, value_loss2))

    # Entropy bonus (diagonal Gaussian)
    entropy = jnp.mean(0.5 * (1 + jnp.log(2 * jnp.pi)) + log_std).sum()

    total = policy_loss + cfg.value_loss_coef * value_loss - cfg.entropy_coef * entropy
    return total, (policy_loss, value_loss, entropy)


# ----- Rollout collection ------------------------------------------------------
def collect_rollout(env_step_fn, env_reset_fn, apply_fn, params, state, obs,
                    key, cfg: PPOConfig):
    """Roll out ``cfg.num_steps`` steps across ``cfg.num_envs`` parallel envs.

    ``env_step_fn(key, state, action)`` and ``env_reset_fn(key)`` must be
    pre-vmap'd and jit'd by caller. Auto-resets done envs.

    Returns ``(new_state, new_obs, key, rollout)`` where ``rollout`` is a
    ``Rollout`` pytree of shape ``(T, B, ...)``.
    """
    def body(carry, _):
        state, obs, key = carry
        key, k_action, k_step, k_reset = jax.random.split(key, 4)

        # Policy forward
        value, mean, log_std = apply_fn(params, obs)
        value = value.squeeze(-1)  # (B,)

        # Sample action — log_std is shared across batch, so don't vmap over it
        action_key = jax.random.split(k_action, cfg.num_envs)
        actions, log_probs = jax.vmap(sample_action, in_axes=(0, 0, None))(
            action_key, mean, log_std
        )

        # Env step (vmapped + jit'd by caller)
        step_keys = jax.random.split(k_step, cfg.num_envs)
        state, next_obs, reward, done, info = env_step_fn(state, step_keys, actions)

        # Auto-reset done envs
        reset_keys = jax.random.split(k_reset, cfg.num_envs)
        reset_state, reset_obs = env_reset_fn(reset_keys)
        # Where done, replace state/obs with reset
        state = jax.tree_util.tree_map(
            lambda new, reset: jnp.where(done.reshape(-1, *([1] * (new.ndim - 1))),
                                         reset, new),
            state, reset_state
        )
        next_obs = jax.tree_util.tree_map(
            lambda new, reset: jnp.where(done.reshape(-1, *([1] * (new.ndim - 1))),
                                         reset, new),
            next_obs, reset_obs
        )

        per_step = (obs, actions, log_probs, value, reward, done)
        return (state, next_obs, key), per_step

    (final_state, final_obs, key), per_steps = jax.lax.scan(
        body, (state, obs, key), None, length=cfg.num_steps
    )
    obs_T, actions_T, log_probs_T, values_T, rewards_T, dones_T = per_steps
    rollout = Rollout(obs=obs_T, actions=actions_T, log_probs=log_probs_T,
                      values=values_T, rewards=rewards_T, dones=dones_T)
    return final_state, final_obs, key, rollout


# ----- Train step --------------------------------------------------------------
def update_minibatch(opt_state, params, apply_fn, optimizer, batch, cfg):
    """One gradient step on a minibatch."""
    obs, action, old_log_prob, advantage, ret, value_old = batch
    (loss, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
        params, apply_fn, obs, action, old_log_prob, advantage, ret, value_old, cfg
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux


def ppo_update(params, opt_state, apply_fn, optimizer, rollout, last_value, cfg):
    """One PPO update phase: compute GAE then ppo_epoch × num_mini_batch grad steps."""
    advantages, returns = compute_gae(
        rollout.rewards, rollout.values, rollout.dones, last_value, cfg
    )
    # Normalize advantages across all (T, B)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    T, B = rollout.rewards.shape
    N = T * B

    # Flatten (T, B, ...) → (T*B, ...)
    def flat(x):
        return x.reshape(N, *x.shape[2:])

    obs_flat = jax.tree_util.tree_map(flat, rollout.obs)
    actions_flat = flat(rollout.actions)
    log_probs_flat = flat(rollout.log_probs)
    values_flat = flat(rollout.values)
    advantages_flat = flat(advantages)
    returns_flat = flat(returns)

    mb_size = N // cfg.num_mini_batch

    def epoch_body(carry, _):
        params, opt_state, key = carry
        key, kshuf = jax.random.split(key)
        perm = jax.random.permutation(kshuf, N)
        # Make mini-batches
        def mb_body(carry, mb_idx):
            params, opt_state = carry
            idx = jax.lax.dynamic_slice(perm, (mb_idx * mb_size,), (mb_size,))
            batch = (
                jax.tree_util.tree_map(lambda x: x[idx], obs_flat),
                actions_flat[idx],
                log_probs_flat[idx],
                advantages_flat[idx],
                returns_flat[idx],
                values_flat[idx],
            )
            params, opt_state, loss, aux = update_minibatch(
                opt_state, params, apply_fn, optimizer, batch, cfg
            )
            return (params, opt_state), (loss, aux)
        (params, opt_state), out = jax.lax.scan(
            mb_body, (params, opt_state), jnp.arange(cfg.num_mini_batch)
        )
        return (params, opt_state, key), out

    init_key = jax.random.PRNGKey(0)
    (params, opt_state, _), epoch_out = jax.lax.scan(
        epoch_body, (params, opt_state, init_key), None, length=cfg.ppo_epoch
    )
    losses, aux = epoch_out
    return params, opt_state, losses.mean(), aux


def make_train(env_reset, env_step, apply_fn, cfg: PPOConfig):
    """Build the full jitted train function.

    ``env_reset(keys)`` and ``env_step(state, keys, actions)`` should already
    be vmapped + jitted by caller (over the batch of num_envs).

    Returns a function ``train(init_params, init_state, init_obs, key, n_updates)``
    that runs n_updates rollout+update cycles and returns the final params +
    a list of (mean_reward, loss).
    """
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.lr, eps=1e-5),
    )

    def train_one_update(carry, _):
        params, opt_state, state, obs, key = carry
        # 1) Collect rollout
        state, obs, key, rollout = collect_rollout(
            env_step, env_reset, apply_fn, params, state, obs, key, cfg
        )
        # 2) Bootstrap value from final obs for GAE
        value_last, _, _ = apply_fn(params, obs)
        last_value = value_last.squeeze(-1)
        # 3) PPO update
        params, opt_state, mean_loss, _aux = ppo_update(
            params, opt_state, apply_fn, optimizer, rollout, last_value, cfg
        )
        mean_reward = rollout.rewards.mean()
        return (params, opt_state, state, obs, key), (mean_reward, mean_loss)

    def train(init_params, init_state, init_obs, key, n_updates):
        opt_state = optimizer.init(init_params)
        carry = (init_params, opt_state, init_state, init_obs, key)
        carry, hist = jax.lax.scan(train_one_update, carry, None, length=n_updates)
        return carry[0], hist  # final params, (mean_rewards, mean_losses)

    return train, optimizer
