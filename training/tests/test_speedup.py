"""Benchmark JAX env vs the original 218 ms/step PyTorch baseline.

Reference baseline (measured 2026-05-21 on the same laptop, RTX 3070 8GB):
    Original `crowd_sim_following.py` step time = 218 ms/single env
    Original 128-env ShmemVecEnv throughput   ≈ 600 steps/sec
    Original ~4 days for full training (5M env steps)

We measure:
    A) Single env JAX, jit-compiled (CPU)
    B) Vmapped over N envs on GPU (cycle N = 64, 256, 1024, 4096)
    C) Compute "speedup vs original" and "GPU utilization estimate"
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import jax
import jax.numpy as jnp

from env.crowd_follow_env import EnvConfig, env_reset, env_step
from env.lidar import LidarConfig, OgmConfig
from env.human_dynamics import HumanConfig


ORIGINAL_BASELINE_MS = 218.0  # ms / single env step
ORIGINAL_VEC_THROUGHPUT = 600  # steps/sec across 128 envs


def benchmark_jax_single(n_steps=1000, n_rays=1080):
    """Single env, jit'd, on default device (likely GPU)."""
    cfg = EnvConfig(max_human_num=10, human_num=5, n_boxes=4, max_steps=10_000)
    lcfg = LidarConfig(n_rays=n_rays)
    ocfg = OgmConfig()
    hcfg = HumanConfig()

    @jax.jit
    def step(state, key, action):
        return env_step(key, state, action, cfg, lcfg, ocfg, hcfg)

    key = jax.random.PRNGKey(0)
    state, _ = env_reset(key, cfg, lcfg, ocfg, hcfg)
    # Warmup (compile)
    state, *_ = step(state, jax.random.PRNGKey(1), jnp.zeros(2))
    jax.block_until_ready(state.robot_xy)

    t0 = time.perf_counter()
    for i in range(n_steps):
        state, *_ = step(state, jax.random.PRNGKey(i), jnp.array([0.5, 0.0]))
    jax.block_until_ready(state.robot_xy)
    elapsed = time.perf_counter() - t0
    per_step_ms = elapsed / n_steps * 1000
    return per_step_ms


def benchmark_jax_vmap(N, n_steps=500, n_rays=1080):
    """Vmap N parallel envs."""
    cfg = EnvConfig(max_human_num=10, human_num=5, n_boxes=4, max_steps=10_000)
    lcfg = LidarConfig(n_rays=n_rays)
    ocfg = OgmConfig()
    hcfg = HumanConfig()

    @jax.jit
    def reset_batch(keys):
        return jax.vmap(env_reset, in_axes=(0, None, None, None, None))(
            keys, cfg, lcfg, ocfg, hcfg
        )

    @jax.jit
    def step_batch(state, keys, actions):
        return jax.vmap(env_step, in_axes=(0, 0, 0, None, None, None, None))(
            keys, state, actions, cfg, lcfg, ocfg, hcfg
        )

    keys = jax.random.split(jax.random.PRNGKey(0), N)
    state, _ = reset_batch(keys)

    # Warmup
    step_keys = jax.random.split(jax.random.PRNGKey(7), N)
    actions = jnp.zeros((N, 2))
    state, *_ = step_batch(state, step_keys, actions)
    jax.block_until_ready(state.robot_xy)

    t0 = time.perf_counter()
    for i in range(n_steps):
        step_keys = jax.random.split(jax.random.PRNGKey(i + 100), N)
        state, *_ = step_batch(state, step_keys, actions)
    jax.block_until_ready(state.robot_xy)
    elapsed = time.perf_counter() - t0

    total_env_steps = N * n_steps
    per_batch_ms = elapsed / n_steps * 1000
    throughput = total_env_steps / elapsed
    return per_batch_ms, throughput


def report():
    print(f"=== JAX env throughput benchmark ===")
    print(f"GPU: {jax.devices()}")
    print()
    print(f"Baseline (original Python+C++ env): {ORIGINAL_BASELINE_MS:.0f} ms/step single, "
          f"~{ORIGINAL_VEC_THROUGHPUT} steps/sec vec128")
    print()
    single_ms = benchmark_jax_single(n_steps=500)
    print(f"--- single env, jit'd ---")
    print(f"  {single_ms:.2f} ms/step  ({1000/single_ms:.0f} steps/sec)")
    print(f"  speedup vs original single = {ORIGINAL_BASELINE_MS / single_ms:.1f}x")
    print()
    print(f"--- vmap'd batched envs (GPU) ---")
    print(f"  {'N':>5} {'ms/batch':>10} {'steps/sec':>14} {'vs orig vec128':>16}")
    for N in [64, 256, 1024, 4096]:
        try:
            ms, tps = benchmark_jax_vmap(N, n_steps=200)
            speedup = tps / ORIGINAL_VEC_THROUGHPUT
            print(f"  {N:>5} {ms:>9.1f}ms {tps:>13.0f} {speedup:>15.1f}x")
        except Exception as e:
            print(f"  {N:>5} OOM or failed: {type(e).__name__}")
            break


def test_speedup_at_least_10x():
    """Assert JAX vmap'd env beats original by ≥10x."""
    _, tps_1024 = benchmark_jax_vmap(1024, n_steps=100)
    speedup = tps_1024 / ORIGINAL_VEC_THROUGHPUT
    assert speedup >= 10.0, f"speedup {speedup:.1f}x < 10x target"


if __name__ == '__main__':
    report()
