# human-following-jax

A monorepo for the RSS 2026 paper **Learning Customizable Human Following**.
Two halves, one repo:

- [`training/`](training/) — JAX rewrite of the original PyTorch training stack.
  Entire env + PPO loop runs **on the GPU**; CPU stops being the bottleneck
  (~100× end-to-end speedup vs the original).
- [`deployment/`](deployment/) — ROS 2 stack that runs the trained policy on
  the **Yahboom Rosmaster X3** mecanum robot (LiDAR + UWB + RL decider).

The PyTorch source we ported from lives separately at
[`human-following-robot`](https://github.com/tasl-lab/human-following-robot)
(referenced, not vendored).

> 中文版见 [README.zh.md](README.zh.md)

---

## TL;DR — benchmarks

Reference baseline (original `human-following-robot` codebase, RTX 3070 8GB,
profiled 2026-05-21):

| Step             | Original (PyTorch + C++ env, 128 forks) | This repo (JAX, GPU) |
|------------------|----------------------------------------|----------------------|
| Single env step  | **218 ms**                              | **1.16 ms** (188×)   |
| 128 envs / sec   | ~600                                    | 24,540 (**41×**)     |
| 1024 envs / sec  | n/a                                     | 281,092 (**468×**)   |
| 4096 envs / sec  | n/a                                     | 558,199 (**930×**)   |
| 32 768 envs / sec | n/a                                    | 711,711 (**~1190×**) |
| Full 5 M env steps | **~4 days**                            | **~7 seconds** (pure env) |

PPO update (4.5 M-param Transformer-policy, 128 envs × 30 steps × 5 epochs
× 8 minibatches, all in one `lax.scan`):

| Setup                      | Env-only steps/sec | End-to-end (env+update) steps/sec |
|----------------------------|---------------------|-----------------------------------|
| Original (4 days for 5 M)  | ~600                | ~14.5                              |
| JAX, 128 envs              | 24 540              | 1 622                              |
| JAX, 1024 envs             | 281 092             | 1 337                              |
| **Speedup over original**  | **41–1190×** env-only | **92–112× end-to-end**          |

(End-to-end gets capped by policy compute, not env. With a smaller policy /
A100 / fewer PPO epochs the end-to-end number jumps further.)

---

## Why this rewrite

The original code's training is CPU-bound:

- `ShmemVecEnv` with 128 forked CPU processes
- Each fork runs ~218 ms of Python+C++ per env step
- 62 % of that time is in `lidar_ogm_cpp.render_polygon_edges` called **64×/step
  per env** in a Python loop

JAX moves the entire env + policy + optimizer onto the GPU and `vmap`s across
thousands of envs, with the whole rollout+update cycle fused into one
`jax.lax.scan` JIT program. CPU sees only `train_jit(...)` calls.

---

## Install (training side)

```bash
git clone https://github.com/Gonglitian/human-following-jax.git
cd human-following-jax/training
/usr/bin/python3 -m pip install --user -e .
```

For **Python 3.8** (most lab machines are still on Foxy / Ubuntu 20.04):

```bash
/usr/bin/python3 -m pip install --user \
  jax==0.4.13 jaxlib==0.4.13+cuda12.cudnn89 \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
/usr/bin/python3 -m pip install --user flax==0.7.2 optax==0.1.7 chex==0.1.7 distrax==0.1.3
```

For Python ≥ 3.10 use latest JAX (`jax[cuda12]`) — ~20 % faster XLA.
See [`training/SETUP.md`](training/SETUP.md) for the full Jenkins-friendly
recipe (CUDA version table, GPU memory tuning, troubleshooting).

Verify GPU:
```bash
/usr/bin/python3 -c "import jax; print(jax.devices())"
# expect: [gpu(id=0)] or similar
```

---

## Quick train

```bash
cd training
# 5M env steps, default ~6 GB GPU, RTX 3070 finishes in <2 hours wall-clock
/usr/bin/python3 scripts/train.py \
  --num-envs 1024 --total-timesteps 5000000 \
  --n-rays 720 --max-human-num 45 --human-num 10 \
  --output runs/
```

Output: `runs/<TS>/{params.pkl, log.csv, args.json}`.

---

## Deployment

See [`deployment/README.md`](deployment/README.md) and
[`DEPLOY.md`](DEPLOY.md) for the ROS 2 robot bringup pipeline (DR-SPAAM ◦ SORT
◦ predictor ◦ decider ◦ /cmd_vel). Heavyweight 3rd-party deps (HuNavSim, AWS
worlds) and PyTorch model checkpoints are pulled in by helper scripts, not
tracked in git.

### Tuning num_envs

| GPU    | Recommended num_envs | Notes              |
|--------|----------------------|--------------------|
| 3070 8GB | 1024–2048           | env-only fits 32K, but PPO update bottlenecks ~2K |
| 4090 24GB | 4096–8192          | bigger minibatch → faster convergence |
| A100 40GB | 8192–16384         | scale up minibatch + lower clip_param if needed |

Bigger num_envs ≠ always faster in wall-clock — the PPO update phase becomes
GPU-bound on policy compute. Profile with `scripts/bench_gpu_memory.py`.

---

## Repo layout

```
human-following-jax/
├── README.md / README.zh.md
├── DEPLOY.md                     # robot-side bringup recipe
├── .gitignore
│
├── training/                     # ── JAX training stack ──
│   ├── pyproject.toml
│   ├── SETUP.md                  # GPU server / Jenkins setup
│   ├── src/
│   │   ├── env/                  # geometry, lidar, human_dynamics, crowd_follow_env
│   │   ├── policy/it_meta.py     # Flax ITMetaPolicy
│   │   └── training/ppo.py       # PureJaxRL PPO (lax.scan fused)
│   ├── scripts/                  # train.py, eval.py, bench_gpu_memory.py
│   ├── tests/                    # test_env, test_policy, test_training_smoke, …
│   ├── docs/                     # PORT_SCOPE, ARCHITECTURE, AUDIT, REPRODUCE
│   └── runs/                     # gitignored
│
└── deployment/                   # ── ROS 2 robot stack ──
    ├── README.md
    ├── src/
    │   ├── core/                 # 13 vendored lab-developed ROS 2 packages
    │   │   ├── decider/          # RL inference + control (loads policy)
    │   │   ├── predictor/        # constant-velocity trajectory prediction
    │   │   ├── target_tracker/   # single-target lock
    │   │   ├── sort_tracker/     # SORT MOT
    │   │   ├── camera_detector/  # RGB-D detector
    │   │   ├── uwb_tracking/     # LinkTrack UWB Kalman
    │   │   ├── command_listener/ # CLI mode/preference
    │   │   ├── depth_costmap/    # depth → costmap
    │   │   ├── occupancy_generation/  # LiDAR → OGM
    │   │   ├── frequency_monitor/     # Hz + latency CSV logging
    │   │   ├── tf_republisher/   # TF bridge
    │   │   ├── fake_detection/   # sim-only test source
    │   │   └── following_sim/    # lightweight Gazebo URDF
    │   └── third_party/          # populated by install_third_party.sh
    └── scripts/
        ├── install_third_party.sh  # clones HuNavSim/AWS worlds/etc
        └── pull_models.sh          # downloads PyTorch ckpts from Google Drive
```

---

## Differences vs the original env

To make the env GPU-friendly we made these **deliberate simplifications**.
The policy only sees LiDAR + relative agent positions so it should generalize
back to the original Python env / real robot.

1. **Static obstacles** are axis-aligned boxes (no Shapely rotated rectangles).
   → closed-form ray-box intersection. ~free in JAX.
2. **Human dynamics** = Helbing social force, not RVO2.
   → vmappable; RVO2 is C++ with mutable per-agent state, not portable.
3. **GST trajectory prediction** → constant-velocity extrapolation for
   `predict_steps` frames (still feeds `spatial_edges`).
4. **Maze topology** → procedural random boxes (no shapely walls/corridors).

Same as original:
- Observation schema (`robot_node` / `temporal_edges` / `spatial_edges` /
  `target_human_traj` / `local_ogm` / `detected_human_num` / `following_preference`)
- Discrete preference distances `{-2: 1.37, -1: 1.90, 0: 2.29, 1: 3.31, 2: 3.80}`
- Policy architecture (OGM CNN → Transformer → actor/critic)
- PPO hyperparameters (`clip_param=0.02`, `lr=4e-5`, `gae_lambda=0.95`, …)

---

## Tests

```bash
cd training
for t in tests/test_*.py; do /usr/bin/python3 $t; done
```

All five suites should pass (~20 sec total on GPU).

---

## Roadmap

- [x] JAX env (geometry, LiDAR, OGM, human dynamics, full step/reset/reward)
- [x] Flax ITMetaPolicy (matches PyTorch architecture)
- [x] PureJaxRL PPO with `lax.scan` fused training
- [x] Benchmarks (10–1000× env speedup, 100× end-to-end)
- [x] Tests + docs
- [ ] **Convergence run** to reproduce paper MDE/SR metrics (deferred — needs
      ~1-2 hr GPU; see `docs/REPRODUCE.md`)
- [ ] Param-level diff test vs PyTorch reference (weight transfer to verify
      forward parity within 1e-4)
- [ ] Train one ckpt per discrete preference (0.5/1/1.5m) as baselines for
      the real-robot deployment

---

## License

BSD-3-Clause. Source paper:
*Learning Customizable Human Following*, RSS 2026 (TASL Lab).
