# First training run — results (2026-05-21)

This was a quick **300K env-step** training run on RTX 3070 8GB to validate
the entire pipeline (env + Flax policy + PureJaxRL PPO + auto-reset) end-to-end.

> Note: 300K is **6% of the paper's 5M-step recipe**. Numbers below show that
> the policy is learning (clear improvement over random baseline) but is far
> from converged. See REPRODUCE.md for the full recipe.

## Setup

```bash
PYTHONUNBUFFERED=1 /usr/bin/python3 scripts/train.py \
  --num-envs 128 --total-timesteps 300000 --log-interval 5 \
  --n-rays 360 --max-human-num 10 --human-num 5 --n-boxes 4 \
  --max-steps 80
```

Total compile + train wall-clock: **~22 min** (vs ~4 hours at this scale on
the original PyTorch — back-of-envelope ~10× speedup with this small policy
+ small num_envs setup).

## Training curve

| update | mean_reward | mean_loss | env_steps/sec |
|--------|-------------|-----------|---------------|
| 1-5    | -1.42       | 41.7      | 825           |
| 6-10   | -1.41       | 35.9      | 1358          |
| 11-15  | -1.38       | 33.5      | 1233          |
| 16-20  | -1.32       | 30.4      | 1203          |
| 21-25  | -1.29       | 28.3      | 1200          |
| 26-30  | -1.32       | 27.6      | 1197          |
| 31-35  | -1.27       | 25.9      | 1196          |
| 36-40  | -1.26       | 25.1      | 1196          |
| 41-45  | -1.21       | 24.2      | 1195          |
| 46-50  | -1.17       | 21.8      | 1194          |
| 51-55  | -1.15       | 20.2      | 1194          |
| 56-60  | -1.14       | 19.8      | 1193          |
| 61-65  | -1.13       | 19.2      | 1190          |
| 66-70  | -1.19       | 20.0      | 1189          |
| 71-75  | -1.11       | 18.2      | 1187          |
| 76-77  | **-1.04**   | **18.1**  | 442           |

Reward improvement: **-1.42 → -1.04 (+27%)**
Loss reduction: **41.7 → 18.1 (-57%)**

## Eval — paper metrics (200 episodes × 100 max steps)

| Metric | Random | Trained | Δ |
|--------|--------|---------|---|
| **MDE ↓**  | 1.32 m | **1.09 m** | **-17%** |
| **AFDE ↓** | 1.08 m | **0.86 m** | **-20%** |
| **WRP ↑**  | 13.7%  | **21.3%**  | **+56%** |
| **SR ↑**   | 6.0%   | **36.0%**  | **+500%** |
| **HCR ↓**  | 0.0%   | 1.5%       | slight regression |
| **OCR ↓**  | 15.0%  | 26.0%      | regression (more active policy) |
| **TLR ↓**  | 80.0%  | **36.5%**  | **-54%** |
| ep_length  | 28 steps | **55 steps** | almost 2× |

## Interpretation

- **TLR dropped from 80% → 36.5%** — biggest signal that the policy learned
  to *follow* the target (instead of drifting away → "lost target").
- **SR 6× higher** confirms task acquisition.
- **OCR regressed** because the trained policy actually *moves* (random just
  sits there). Episodes now reach further into the maze where collisions
  become possible. Needs longer training + maybe higher OCR penalty in the
  reward.

## What's next (to match paper numbers)

Paper Table II `Ours/meta_4`: MDE ~0.3m, SR >95%, WRP >80%.

To get there:
1. **Train longer** — 5M steps (paper recipe), not 300K. Expected ~1.5 hr
   on this GPU with `--num-envs 1024 --n-rays 720 --max-human-num 45`.
2. **Tune reward weights** — penalize obstacle collisions more, distance
   error potentially smoother.
3. **Match env fidelity** — original uses Shapely maze + RVO; ours uses
   axis-aligned boxes + social force. The simplified env makes the task
   slightly different. To close gap: port Shapely maze gen (one-time at
   reset) without touching the JIT'd step.

These are tractable; the speedup unlocks rapid iteration on all of them.

## Artifacts

```
runs/20260521_005944/
├── args.json            # exact hparams
├── params.pkl           # final Flax params (17 MB)
├── log.csv              # per-update reward + loss + throughput
└── full_train_log.txt   # raw stdout
```

Reproduce: `scripts/train.py` with the same args — JAX is deterministic given
the same seed (`--seed 0` was used here).
