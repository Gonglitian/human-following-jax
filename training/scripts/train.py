#!/usr/bin/env python3
"""Full training driver — env + policy + PPO.

Usage:
    /usr/bin/python3 scripts/train.py --num-envs 1024 --total-timesteps 5000000

The entire env+policy+PPO update is fused into one ``lax.scan`` JIT program
that runs entirely on GPU. CPU is only used for initial compilation, logging,
and checkpoint I/O.

Outputs:
    runs/<timestamp>/
      params.pkl     — final policy parameters
      log.csv        — per-update mean reward + loss
      args.json      — config used for this run
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

# Allow ``import env, policy, training`` from src/
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import jax
import jax.numpy as jnp

from env.crowd_follow_env import EnvConfig, env_reset, env_step
from env.lidar import LidarConfig, OgmConfig
from env.human_dynamics import HumanConfig
from policy.it_meta import ITMetaPolicy, init_policy, make_dummy_obs
from training.ppo import PPOConfig, make_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-envs', type=int, default=1024)
    ap.add_argument('--num-steps', type=int, default=30, help='rollout length per update')
    ap.add_argument('--ppo-epoch', type=int, default=5)
    ap.add_argument('--num-mini-batch', type=int, default=8)
    ap.add_argument('--total-timesteps', type=int, default=5_000_000)
    ap.add_argument('--lr', type=float, default=4e-5)
    ap.add_argument('--clip-param', type=float, default=0.02)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--gae-lambda', type=float, default=0.95)
    ap.add_argument('--n-rays', type=int, default=720, help='LiDAR rays (1080 for full A1, 720 for fast training)')
    ap.add_argument('--max-human-num', type=int, default=45)
    ap.add_argument('--human-num', type=int, default=10)
    ap.add_argument('--n-boxes', type=int, default=6)
    ap.add_argument('--max-steps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output', type=str, default='runs',
                    help='dir for checkpoint + log. Will create subdir by timestamp.')
    ap.add_argument('--log-interval', type=int, default=10, help='print every N updates')
    args = ap.parse_args()

    # ---- Configs ----
    env_cfg = EnvConfig(
        max_human_num=args.max_human_num,
        human_num=args.human_num,
        n_boxes=args.n_boxes,
        max_steps=args.max_steps,
    )
    lidar_cfg = LidarConfig(n_rays=args.n_rays)
    ogm_cfg = OgmConfig()
    human_cfg = HumanConfig()

    ppo_cfg = PPOConfig(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        ppo_epoch=args.ppo_epoch,
        num_mini_batch=args.num_mini_batch,
        clip_param=args.clip_param,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )

    # ---- Output dir ----
    # IMPORTANT: resolve to absolute path so we survive CWD changes / dir
    # moves during long runs (got bitten by this on v3 — see git history).
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.output).resolve() / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f'[train] output → {out_dir}')

    # ---- Vmapped env fns ----
    @jax.jit
    def env_reset_v(keys):
        return jax.vmap(env_reset, in_axes=(0, None, None, None, None))(
            keys, env_cfg, lidar_cfg, ogm_cfg, human_cfg
        )

    @jax.jit
    def env_step_v(state, keys, actions):
        return jax.vmap(env_step, in_axes=(0, 0, 0, None, None, None, None))(
            keys, state, actions, env_cfg, lidar_cfg, ogm_cfg, human_cfg
        )

    # ---- Init policy on dummy obs ----
    dummy = make_dummy_obs(B=1, max_human_num=args.max_human_num,
                           predict_steps=env_cfg.predict_steps,
                           ogm_size=ogm_cfg.grid_size,
                           history=ogm_cfg.history_len)
    model = ITMetaPolicy(action_dim=2, max_human_num=args.max_human_num)
    key = jax.random.PRNGKey(args.seed)
    key, k_init = jax.random.split(key)
    params = model.init(k_init, dummy)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f'[train] policy params: {n_params:,}')

    # Adapter so PPO sees apply_fn(params, obs) only
    def apply_fn(params, obs):
        v, m, ls = model.apply(params, obs)
        return v, m, ls

    # Wrap env_step_v to match PPO's expected signature (state, keys, action) — same already
    def env_step_for_ppo(state, keys, actions):
        new_state, obs, rew, done, info = env_step_v(state, keys, actions)
        return new_state, obs, rew, done, info

    train_fn, optimizer = make_train(env_reset_v, env_step_for_ppo, apply_fn, ppo_cfg)
    train_jit = jax.jit(train_fn, static_argnames=('n_updates',))

    # ---- Initial reset ----
    key, k_reset = jax.random.split(key)
    keys = jax.random.split(k_reset, args.num_envs)
    state, obs = env_reset_v(keys)
    print(f'[train] initial state ready, num_envs={args.num_envs}')

    # ---- Train! ----
    steps_per_update = args.num_envs * args.num_steps
    total_updates = args.total_timesteps // steps_per_update
    print(f'[train] {total_updates} updates × {steps_per_update} env steps = {total_updates * steps_per_update:,} total')

    # Compile + run first update to measure warmup
    print('[train] compiling first update...')
    t0 = time.perf_counter()
    final_params, hist = train_jit(params, state, obs, key, n_updates=1)
    jax.block_until_ready(hist[0])
    compile_time = time.perf_counter() - t0
    print(f'[train] first update + compile: {compile_time:.1f}s')

    # Run remaining updates in chunks for logging
    chunk = args.log_interval
    log_rows = []
    overall_t0 = time.perf_counter()
    n_remaining = total_updates - 1
    params = final_params
    update_idx = 1
    # Save mid-run checkpoints every ckpt_every updates so a crash near the end
    # doesn't wipe hours of training (the v3 incident).
    ckpt_every = max(100, chunk)
    last_ckpt_at = 0
    while n_remaining > 0:
        this = min(chunk, n_remaining)
        # IMPORTANT: re-key each chunk so the rollout RNG advances
        key, k_run = jax.random.split(key)
        t_chunk = time.perf_counter()
        params, hist = train_jit(params, state, obs, k_run, n_updates=this)
        rewards, losses = hist
        jax.block_until_ready(rewards)
        dt = time.perf_counter() - t_chunk
        env_steps_done = this * steps_per_update
        sps = env_steps_done / dt
        mean_r = float(rewards.mean())
        mean_l = float(losses.mean())
        print(f'[train] update {update_idx:5d}-{update_idx+this-1:5d} | '
              f'rew={mean_r:+.3f} loss={mean_l:+.3f} | '
              f'{sps:.0f} env steps/sec | chunk {dt:.1f}s')
        log_rows.append((update_idx, mean_r, mean_l, sps))
        update_idx += this
        n_remaining -= this

        # Mid-run checkpoint
        if update_idx - last_ckpt_at >= ckpt_every:
            ckpt_path = out_dir / 'params.ckpt'
            tmp_path = out_dir / 'params.ckpt.tmp'
            with open(tmp_path, 'wb') as f:
                pickle.dump(params, f)
            tmp_path.replace(ckpt_path)  # atomic rename
            import csv as _csv
            with open(out_dir / 'log.csv', 'w', newline='') as f:
                _w = _csv.writer(f)
                _w.writerow(['update', 'mean_reward', 'mean_loss', 'env_steps_per_sec'])
                _w.writerows(log_rows)
            print(f'[train] ckpt saved → {ckpt_path} ({update_idx} updates done)')
            last_ckpt_at = update_idx

    total_dt = time.perf_counter() - overall_t0
    total_env_steps = (total_updates - 1) * steps_per_update
    print(f'\n[train] ===== DONE =====')
    print(f'[train] {total_updates} updates, {total_env_steps:,} env steps, {total_dt:.1f}s total')
    print(f'[train] effective throughput: {total_env_steps / total_dt:.0f} env steps/sec')

    # Save params + log
    with open(out_dir / 'params.pkl', 'wb') as f:
        pickle.dump(params, f)
    import csv
    with open(out_dir / 'log.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['update', 'mean_reward', 'mean_loss', 'env_steps_per_sec'])
        w.writerows(log_rows)
    print(f'[train] saved params → {out_dir / "params.pkl"}')
    print(f'[train] saved log → {out_dir / "log.csv"}')


if __name__ == '__main__':
    main()
