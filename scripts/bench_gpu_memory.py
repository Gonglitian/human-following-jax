#!/usr/bin/env python3
"""Sweep num_envs, measure GPU memory + throughput.

Finds the largest num_envs that fits 8 GB (RTX 3070) and shows the
throughput / wall-clock-to-5M-steps tradeoff.
"""
import os
import sys
import subprocess
import time
from pathlib import Path

# Prevent JAX from grabbing all GPU memory upfront so we can observe usage
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import jax
import jax.numpy as jnp

from env.crowd_follow_env import EnvConfig, env_reset, env_step
from env.lidar import LidarConfig, OgmConfig
from env.human_dynamics import HumanConfig


def gpu_memory_mb():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return int(out.split('\n')[0])
    except Exception:
        return -1


def bench(N, n_rays=1080, n_steps=200):
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

    mem_before = gpu_memory_mb()
    keys = jax.random.split(jax.random.PRNGKey(0), N)
    state, _ = reset_batch(keys)
    # warmup
    actions = jnp.zeros((N, 2))
    step_keys = jax.random.split(jax.random.PRNGKey(1), N)
    state, *_ = step_batch(state, step_keys, actions)
    jax.block_until_ready(state.robot_xy)
    mem_after = gpu_memory_mb()

    t0 = time.perf_counter()
    for i in range(n_steps):
        step_keys = jax.random.split(jax.random.PRNGKey(i + 1000), N)
        state, *_ = step_batch(state, step_keys, actions)
    jax.block_until_ready(state.robot_xy)
    dt = time.perf_counter() - t0
    sps = N * n_steps / dt
    return mem_after - mem_before, sps


def main():
    print(f"GPU: {jax.devices()}")
    print(f"{'N':>6} {'GPU MB':>10} {'env steps/sec':>16} {'5M-step ETA':>14}")
    for N in [128, 512, 2048, 8192, 16384, 32768]:
        try:
            mem, sps = bench(N, n_steps=100)
            eta = 5_000_000 / sps
            eta_str = f"{eta:.0f}s" if eta < 60 else f"{eta/60:.1f}min"
            print(f"{N:>6} {mem:>9}MB {sps:>15.0f} {eta_str:>14}")
        except RuntimeError as e:
            print(f"{N:>6} OOM: {str(e)[:60]}")
            break


if __name__ == '__main__':
    main()
