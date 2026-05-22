"""End-to-end smoke test: env + policy + PPO + lax.scan + autoreset.

Runs 2 updates with tiny configs to make sure the whole pipeline is wired
correctly (shapes match, no NaN, params actually change).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp

from env.crowd_follow_env import EnvConfig, env_reset, env_step
from env.lidar import LidarConfig, OgmConfig
from env.human_dynamics import HumanConfig
from policy.it_meta import ITMetaPolicy, make_dummy_obs
from training.ppo import PPOConfig, make_train


def test_train_loop_e2e():
    env_cfg = EnvConfig(max_human_num=8, human_num=4, n_boxes=3, max_steps=20)
    lcfg = LidarConfig(n_rays=128)
    ocfg = OgmConfig()
    hcfg = HumanConfig()
    cfg = PPOConfig(num_envs=8, num_steps=6, ppo_epoch=2, num_mini_batch=2)

    @jax.jit
    def env_reset_v(keys):
        return jax.vmap(env_reset, in_axes=(0, None, None, None, None))(
            keys, env_cfg, lcfg, ocfg, hcfg
        )

    @jax.jit
    def env_step_v(state, keys, actions):
        return jax.vmap(env_step, in_axes=(0, 0, 0, None, None, None, None))(
            keys, state, actions, env_cfg, lcfg, ocfg, hcfg
        )

    model = ITMetaPolicy(action_dim=2, max_human_num=env_cfg.max_human_num)
    dummy = make_dummy_obs(B=1, max_human_num=env_cfg.max_human_num,
                           predict_steps=env_cfg.predict_steps,
                           ogm_size=ocfg.grid_size,
                           history=ocfg.history_len)
    params = model.init(jax.random.PRNGKey(0), dummy)

    def apply_fn(p, o):
        return model.apply(p, o)

    train_fn, _opt = make_train(env_reset_v, env_step_v, apply_fn, cfg)
    train_jit = jax.jit(train_fn, static_argnames=('n_updates',))

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, cfg.num_envs)
    state, obs = env_reset_v(keys)

    # 2 updates
    new_params, hist = train_jit(params, state, obs, jax.random.PRNGKey(7), n_updates=2)
    rewards, losses = hist

    assert rewards.shape == (2,), f"rewards shape {rewards.shape}"
    assert losses.shape == (2,), f"losses shape {losses.shape}"
    assert not jnp.any(jnp.isnan(rewards))
    assert not jnp.any(jnp.isnan(losses))

    # Params should have actually been updated (some leaf must differ)
    diffs = jax.tree_util.tree_map(
        lambda a, b: float(jnp.abs(a - b).max()), params, new_params
    )
    leaves = jax.tree_util.tree_leaves(diffs)
    max_diff = max(leaves)
    assert max_diff > 0, f"params unchanged across train step (max_diff={max_diff})"


if __name__ == '__main__':
    test_train_loop_e2e()
    print('✓ train loop e2e smoke test passed')
