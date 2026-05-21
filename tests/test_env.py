"""Unit tests for full env: reset/step shapes, jit, vmap."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
import numpy as np

from env.crowd_follow_env import EnvConfig, env_reset, env_step
from env.lidar import LidarConfig, OgmConfig
from env.human_dynamics import HumanConfig


def make_cfgs(n_rays=128, max_human_num=10, human_num=5, n_boxes=4):
    """Small configs for unit tests (we don't need 1080 rays here)."""
    cfg = EnvConfig(max_human_num=max_human_num, human_num=human_num, n_boxes=n_boxes,
                    max_steps=20)
    lcfg = LidarConfig(n_rays=n_rays)
    ocfg = OgmConfig()
    hcfg = HumanConfig()
    return cfg, lcfg, ocfg, hcfg


def test_reset_shapes():
    cfg, lcfg, ocfg, hcfg = make_cfgs()
    key = jax.random.PRNGKey(0)
    state, obs = env_reset(key, cfg, lcfg, ocfg, hcfg)

    assert state.robot_xy.shape == (2,)
    assert state.human_xy.shape == (cfg.max_human_num, 2)
    assert state.boxes.shape == (cfg.n_boxes, 4)
    assert state.ogm_history.shape == (ocfg.history_len, ocfg.grid_size, ocfg.grid_size)

    assert obs['robot_node'].shape == (1, 7)
    assert obs['temporal_edges'].shape == (1, 2)
    assert obs['spatial_edges'].shape == (cfg.max_human_num, 2 * (cfg.predict_steps + 1))
    assert obs['target_human_traj'].shape == (2 * (cfg.predict_steps + 1),)
    assert obs['local_ogm'].shape == state.ogm_history.shape
    assert obs['following_preference'].shape == (1, 1)


def test_step_runs():
    cfg, lcfg, ocfg, hcfg = make_cfgs()
    key = jax.random.PRNGKey(0)
    state, _obs = env_reset(key, cfg, lcfg, ocfg, hcfg)
    k1, k2 = jax.random.split(key)
    state2, obs2, rew, done, info = env_step(k1, state, jnp.array([0.5, 0.0]), cfg, lcfg, ocfg, hcfg)
    assert obs2['robot_node'].shape == (1, 7)
    assert rew.shape == ()
    assert done.shape == ()
    # robot should have moved approximately by action * dt
    delta = state2.robot_xy - state.robot_xy
    assert abs(float(delta[0]) - 0.5 * cfg.time_step) < 1e-3


def test_step_jit():
    cfg, lcfg, ocfg, hcfg = make_cfgs()

    @jax.jit
    def step_fn(key, s, a):
        return env_step(key, s, a, cfg, lcfg, ocfg, hcfg)

    key = jax.random.PRNGKey(42)
    state, _ = env_reset(key, cfg, lcfg, ocfg, hcfg)
    new_state, obs, rew, done, info = step_fn(jax.random.PRNGKey(1), state, jnp.zeros(2))
    assert obs['local_ogm'].shape == (ocfg.history_len, ocfg.grid_size, ocfg.grid_size)


def test_vmap_over_envs():
    """Batch 16 envs through reset + 10 steps."""
    cfg, lcfg, ocfg, hcfg = make_cfgs()
    B = 16
    keys = jax.random.split(jax.random.PRNGKey(0), B)

    @jax.jit
    def reset_batch(keys):
        return jax.vmap(env_reset, in_axes=(0, None, None, None, None))(
            keys, cfg, lcfg, ocfg, hcfg
        )

    state, obs = reset_batch(keys)
    assert state.robot_xy.shape == (B, 2)
    assert obs['local_ogm'].shape == (B, ocfg.history_len, ocfg.grid_size, ocfg.grid_size)

    @jax.jit
    def step_batch(keys, state, action):
        return jax.vmap(env_step, in_axes=(0, 0, 0, None, None, None, None))(
            keys, state, action, cfg, lcfg, ocfg, hcfg
        )

    for _ in range(10):
        step_keys = jax.random.split(jax.random.PRNGKey(7), B)
        actions = jnp.zeros((B, 2))
        state, obs, rew, done, info = step_batch(step_keys, state, actions)
        assert rew.shape == (B,)
        assert done.shape == (B,)


def test_done_at_max_steps():
    """Episodes should terminate at max_steps (handled by caller via auto-reset)."""
    cfg, lcfg, ocfg, hcfg = make_cfgs()
    cfg = cfg._replace(max_steps=3)
    key = jax.random.PRNGKey(0)
    state, _ = env_reset(key, cfg, lcfg, ocfg, hcfg)
    for i in range(3):
        state, obs, rew, done, info = env_step(jax.random.PRNGKey(i), state,
                                               jnp.zeros(2), cfg, lcfg, ocfg, hcfg)
    assert bool(done), "expected done after max_steps"


if __name__ == '__main__':
    test_reset_shapes(); print('✓ reset shapes')
    test_step_runs(); print('✓ step runs + robot moves')
    test_step_jit(); print('✓ step under jit')
    test_vmap_over_envs(); print('✓ vmap over 16 envs')
    test_done_at_max_steps(); print('✓ done at max_steps')
    print('\nAll env tests passed!')
