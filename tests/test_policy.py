"""Unit tests for ITMetaPolicy."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp

from policy.it_meta import ITMetaPolicy, init_policy, make_dummy_obs


def test_init_and_forward():
    """Forward pass on dummy obs produces (value, action_mean, log_std) of expected shape."""
    obs = make_dummy_obs(B=4)
    model, params = init_policy(jax.random.PRNGKey(0), obs)
    value, action_mean, log_std = model.apply(params, obs)
    assert value.shape == (4, 1), f"value shape {value.shape}"
    assert action_mean.shape == (4, 2), f"action_mean shape {action_mean.shape}"
    assert log_std.shape == (2,), f"log_std shape {log_std.shape}"


def test_no_nan_on_realistic_obs():
    """Realistic-valued obs shouldn't produce NaNs."""
    obs = {
        'robot_node': jnp.array([[[1.0, -0.5, 0.3, 2.0, 1.5, 1.2, 0.7]]]),
        'temporal_edges': jnp.array([[[0.3, 0.1]]]),
        'spatial_edges': jnp.ones((1, 45, 12)) * 0.5,
        'detected_human_num': jnp.array([[3.0]]),
        'target_human_traj': jnp.array([[1.0, 2.0, 1.1, 2.1, 1.2, 2.2, 1.3, 2.3, 1.4, 2.4, 1.5, 2.5]]),
        'local_ogm': jnp.zeros((1, 3, 50, 50), dtype=jnp.int8).at[0, 0, 25, 25].set(1),
        'following_preference': jnp.array([[[-2.0]]]),
    }
    model, params = init_policy(jax.random.PRNGKey(1), obs)
    value, action_mean, log_std = model.apply(params, obs)
    assert not jnp.any(jnp.isnan(value))
    assert not jnp.any(jnp.isnan(action_mean))


def test_jit_works():
    obs = make_dummy_obs(B=8)
    model, params = init_policy(jax.random.PRNGKey(0), obs)

    @jax.jit
    def fwd(params, obs):
        return model.apply(params, obs)

    v, a, ls = fwd(params, obs)
    assert v.shape == (8, 1)


def test_param_count_in_range():
    """Sanity check on total param count (~2-5M for this architecture)."""
    obs = make_dummy_obs(B=1)
    _, params = init_policy(jax.random.PRNGKey(0), obs)
    n = sum(p.size for p in jax.tree_util.tree_leaves(params))
    # PyTorch reference: ~3.5M params
    assert 1_000_000 < n < 10_000_000, f"param count {n:,} out of expected range"


def test_attention_mask_affects_output():
    """Different detected_human_num values should yield different outputs."""
    obs1 = make_dummy_obs(B=1)
    obs1['detected_human_num'] = jnp.array([[1.0]])
    obs1['spatial_edges'] = jax.random.normal(jax.random.PRNGKey(7), (1, 45, 12))

    obs2 = {**obs1, 'detected_human_num': jnp.array([[10.0]])}

    model, params = init_policy(jax.random.PRNGKey(0), obs1)
    v1, _, _ = model.apply(params, obs1)
    v2, _, _ = model.apply(params, obs2)
    # Values should differ because more humans are attended to
    assert float(jnp.abs(v1 - v2).max()) > 1e-6


if __name__ == '__main__':
    test_init_and_forward(); print('✓ init + forward shapes')
    test_no_nan_on_realistic_obs(); print('✓ no NaNs on realistic obs')
    test_jit_works(); print('✓ jit')
    test_param_count_in_range(); print('✓ param count in range')
    test_attention_mask_affects_output(); print('✓ attention mask is effective')
    print('\nAll policy tests passed!')
