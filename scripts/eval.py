#!/usr/bin/env python3
"""Evaluate a JAX policy in the JAX env, compute paper metrics.

Outputs MDE / AFDE / WRP / SR / HCR / OCR / TLR — matches the metric
definitions in `scripts/analyze_metrics.py` of the real-robot deployment.

Usage:
    /usr/bin/python3 scripts/eval.py --params runs/<TS>/params.pkl \\
        --n-episodes 100 --max-steps 200

Or eval a RANDOM policy (no checkpoint) as baseline:
    /usr/bin/python3 scripts/eval.py --random
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import jax
import jax.numpy as jnp
import numpy as np

from env.crowd_follow_env import EnvConfig, env_reset, env_step, PREFERENCE_DISTANCES
from env.lidar import LidarConfig, OgmConfig
from env.human_dynamics import HumanConfig
from policy.it_meta import ITMetaPolicy, make_dummy_obs


# Paper thresholds (matches scripts/analyze_metrics.py on real-robot side)
WRP_BAND = 0.35
TLR_VALID = 5.0
HCR_THRESH = 0.40
OCR_THRESH = 0.30


def eval_run(model, params, env_cfg, lcfg, ocfg, hcfg,
             n_episodes=100, max_steps=200, deterministic=True, seed=42):
    """Run n_episodes in the JAX env and collect per-step metrics.

    Returns dict of per-episode metrics + raw arrays for inspection.
    """
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

    @jax.jit
    def step_policy(params, obs, key):
        v, mean, log_std = model.apply(params, obs)
        if deterministic:
            return mean
        eps = jax.random.normal(key, mean.shape)
        return mean + jnp.exp(log_std) * eps

    # Run n_episodes in PARALLEL via vmap — one env per episode
    keys = jax.random.split(jax.random.PRNGKey(seed), n_episodes)
    state, obs = env_reset_v(keys)

    # Tracking arrays — collected per step then aggregated per episode
    target_dist_buf = []
    desired_dist_buf = []
    min_human_dist_buf = []
    min_obs_dist_buf = []
    done_buf = []
    collision_buf = []

    key = jax.random.PRNGKey(seed + 1)
    active = jnp.ones(n_episodes, dtype=bool)  # which episodes still running

    for t in range(max_steps):
        key, k_act, k_env = jax.random.split(key, 3)
        action = step_policy(params, obs, k_act)
        env_keys = jax.random.split(k_env, n_episodes)
        state, obs, rew, done, info = env_step_v(state, env_keys, action)

        target_dist_buf.append(np.array(info['target_dist']))
        desired_dist_buf.append(np.array(info['desired_dist']))
        min_human_dist_buf.append(np.array(info['min_human_dist']))
        min_obs_dist_buf.append(np.array(info['min_obs_dist']))
        done_buf.append(np.array(done))
        collision_buf.append(np.array(info['collision']))
        # Track which episodes have already ended; freeze their stats
        active = active & ~np.array(done)

    target_dist = np.stack(target_dist_buf, axis=0)        # (T, N)
    desired_dist = np.stack(desired_dist_buf, axis=0)
    min_human = np.stack(min_human_dist_buf, axis=0)
    min_obs = np.stack(min_obs_dist_buf, axis=0)
    dones = np.stack(done_buf, axis=0)
    collisions = np.stack(collision_buf, axis=0)

    # Compute episode-truncation index (first done step per episode)
    first_done = np.argmax(dones, axis=0)  # (N,)
    has_done = dones.any(axis=0)
    ep_len = np.where(has_done, first_done + 1, max_steps)

    # Per-episode metrics
    metrics = {
        'MDE': [],
        'AFDE': [],
        'WRP': [],
        'SR': [],
        'HCR': [],
        'OCR': [],
        'TLR': [],
        'ep_length': ep_len,
    }
    for n in range(n_episodes):
        L = int(ep_len[n])
        td = target_dist[:L, n]
        dd = desired_dist[:L, n]
        err = np.abs(td - dd)
        mh = min_human[:L, n]
        mo = min_obs[:L, n]

        metrics['MDE'].append(float(err.mean()))
        metrics['AFDE'].append(float(abs(td.mean() - dd.mean())))
        metrics['WRP'].append(float(np.mean(err <= WRP_BAND)))
        tlr = int(np.any(td > TLR_VALID))
        hcr = int(np.any(mh < HCR_THRESH))
        ocr = int(np.any(mo < OCR_THRESH))
        metrics['TLR'].append(tlr)
        metrics['HCR'].append(hcr)
        metrics['OCR'].append(ocr)
        metrics['SR'].append(int(not (tlr or hcr or ocr)))

    return {k: np.array(v) for k, v in metrics.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', type=str, default=None,
                    help='Path to pickled policy params (omit for random policy)')
    ap.add_argument('--random', action='store_true',
                    help='Use a randomly-initialized policy (no training)')
    ap.add_argument('--n-episodes', type=int, default=100)
    ap.add_argument('--max-steps', type=int, default=200)
    ap.add_argument('--n-rays', type=int, default=360)
    ap.add_argument('--max-human-num', type=int, default=45)
    ap.add_argument('--human-num', type=int, default=10)
    ap.add_argument('--n-boxes', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--deterministic', action='store_true', default=True)
    args = ap.parse_args()

    env_cfg = EnvConfig(
        max_human_num=args.max_human_num,
        human_num=args.human_num,
        n_boxes=args.n_boxes,
        max_steps=args.max_steps,
    )
    lcfg = LidarConfig(n_rays=args.n_rays)
    ocfg = OgmConfig()
    hcfg = HumanConfig()

    model = ITMetaPolicy(action_dim=2, max_human_num=args.max_human_num)

    if args.random or args.params is None:
        print('[eval] using RANDOMLY initialized policy (no training)')
        dummy = make_dummy_obs(B=1, max_human_num=args.max_human_num,
                               predict_steps=env_cfg.predict_steps,
                               ogm_size=ocfg.grid_size,
                               history=ocfg.history_len)
        params = model.init(jax.random.PRNGKey(args.seed), dummy)
    else:
        print(f'[eval] loading params from {args.params}')
        with open(args.params, 'rb') as f:
            params = pickle.load(f)

    print(f'[eval] running {args.n_episodes} episodes × {args.max_steps} max steps...')
    t0 = time.perf_counter()
    metrics = eval_run(
        model, params, env_cfg, lcfg, ocfg, hcfg,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        seed=args.seed,
    )
    dt = time.perf_counter() - t0
    print(f'[eval] done in {dt:.1f}s ({args.n_episodes / dt:.1f} eps/sec)')

    print('\n=== Paper metrics (mean ± std across episodes) ===')
    arrows = {'MDE': '↓', 'AFDE': '↓', 'WRP': '↑', 'SR': '↑',
              'HCR': '↓', 'OCR': '↓', 'TLR': '↓'}
    for k in ['MDE', 'AFDE', 'WRP', 'SR', 'HCR', 'OCR', 'TLR']:
        v = metrics[k]
        unit = '' if k in ('MDE', 'AFDE') else '%'
        scale = 1.0 if k in ('MDE', 'AFDE') else 100.0
        print(f'  {arrows[k]} {k:5s}: {v.mean() * scale:6.2f}{unit} ± {v.std() * scale:5.2f}{unit}')

    print(f'\n  mean ep_length: {metrics["ep_length"].mean():.1f} steps '
          f'(max {args.max_steps})')


if __name__ == '__main__':
    main()
