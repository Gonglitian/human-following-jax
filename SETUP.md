# SETUP.md — Deploying `human-following-jax` on a new GPU server

> **Audience**: anyone with SSH access to a fresh Linux GPU box (lab, Jenkins
> runner, cloud VM) who wants to start single-GPU training immediately.
>
> **Time-to-first-training**: ~10 min (mostly waiting for pip).

---

## TL;DR — copy-paste recipe

```bash
# 1) System prereqs (Ubuntu 20.04 / 22.04 / 24.04)
sudo apt-get update && sudo apt-get install -y \
    python3 python3-pip python3-venv git build-essential

# 2) Verify NVIDIA driver + CUDA
nvidia-smi   # need CUDA 12.x driver; we use the JAX cuda12 wheels
# Expect output like: "CUDA Version: 12.x" in the header

# 3) Clone
git clone https://github.com/Gonglitian/human-following-jax.git
cd human-following-jax

# 4) Install
python3 -m pip install --user -e . \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 5) Verify JAX sees GPU
python3 -c "import jax; print(jax.devices())"
# Expect: [gpu(id=0)] or similar — NOT [CpuDevice(id=0)]

# 6) Smoke-test (≈30s)
for t in tests/test_*.py; do python3 $t || break; done

# 7) Kick off paper-recipe training (≈5h on RTX 3070, ≈2h on A100-40)
PYTHONUNBUFFERED=1 python3 -u scripts/train.py \
  --num-envs 256 --total-timesteps 10000000 \
  --n-rays 720 --max-human-num 45 --human-num 40 \
  --n-boxes 6 --max-steps 200 --seed 215 \
  --output runs/my_first_run

# 8) Eval the resulting ckpt
python3 scripts/eval.py \
  --params runs/my_first_run/*/params.pkl \
  --n-episodes 500 --max-steps 200 \
  --n-rays 720 --max-human-num 45 --human-num 40 --n-boxes 6
```

---

## 1. Hardware / OS requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA, Compute Capability ≥ 7.0, 8 GB VRAM | A100/L40/4090, 24+ GB |
| GPU driver | CUDA 12.0 compatible (`nvidia-smi` shows CUDA 12.x) | latest stable |
| CPU | 4 cores | 8+ cores (only used for compile + logging) |
| RAM | 8 GB | 16+ GB |
| Disk | 5 GB (deps) + ~50 MB / training run | — |
| OS | Ubuntu 20.04 LTS | Ubuntu 22.04 LTS |
| Python | 3.8 (pinned wheels available) | 3.10–3.12 (~20% faster XLA) |

**Network**: only needed during install (pip downloads ~1.5 GB of JAX + CUDA libs).
Training itself is fully offline.

---

## 2. Step-by-step install

### 2.1 System packages

```bash
# Ubuntu (Debian-family)
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git build-essential

# RHEL/CentOS/Rocky
sudo yum install -y python3 python3-pip git gcc gcc-c++ make
```

### 2.2 NVIDIA driver / CUDA — only the driver, NOT the CUDA toolkit

JAX ships pre-built CUDA libs in its wheel; you only need the GPU driver. Verify:

```bash
nvidia-smi
```

Expected output includes `CUDA Version: 12.x`. If you see `CUDA Version: 11.x`, upgrade the driver (or install jax with `cuda11_pip` wheels instead).

### 2.3 Clone the repo

```bash
git clone https://github.com/Gonglitian/human-following-jax.git
cd human-following-jax
```

### 2.4 Install Python deps

**Python 3.8** (common on Ubuntu 20.04 / ROS Foxy systems — pinned wheels):

```bash
python3 -m pip install --user \
  jax==0.4.13 jaxlib==0.4.13+cuda12.cudnn89 \
  flax==0.7.2 optax==0.1.7 chex==0.1.7 distrax==0.1.3 \
  numpy gymnasium tyro matplotlib tqdm \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Python ≥ 3.10** (newer Ubuntu / Conda envs):

```bash
python3 -m pip install --user "jax[cuda12]" flax optax distrax chex \
  numpy gymnasium tyro matplotlib tqdm
```

**Alternative — editable install of this repo** (gets you the right deps from `pyproject.toml`):

```bash
python3 -m pip install --user -e .
```

### 2.5 Verify GPU is visible to JAX

```bash
python3 -c "import jax, jax.numpy as jnp; \
            print('devices:', jax.devices()); \
            print('matmul ok:', float(jnp.ones((1000,1000)) @ jnp.ones((1000,1000))).is_integer())"
```

Expected:
```
devices: [gpu(id=0)]
matmul ok: True
```

If you see `[CpuDevice(id=0)]`, JAX did not find a CUDA wheel — re-run the install with `-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html` and check driver version.

### 2.6 Run tests (~30 seconds)

```bash
for t in tests/test_*.py; do
  python3 $t
done
```

All 5 suites should pass. If `test_lidar.py` fails, your JAX install is broken;
re-check 2.4.

---

## 3. Training

### 3.1 Default paper-recipe (full alignment)

```bash
PYTHONUNBUFFERED=1 python3 -u scripts/train.py \
  --num-envs 256 \
  --total-timesteps 10000000 \
  --num-steps 30 \
  --ppo-epoch 5 \
  --num-mini-batch 8 \
  --lr 4e-5 \
  --clip-param 0.02 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --n-rays 720 \
  --max-human-num 45 \
  --human-num 40 \
  --n-boxes 6 \
  --max-steps 200 \
  --seed 215 \
  --output runs/$(date +%Y%m%d_%H%M%S)
```

Wall-clock estimates (10M steps):
| GPU | Time |
|---|---|
| RTX 3070 8GB | ~5h |
| RTX 4090 24GB | ~2h |
| A100-40GB | ~1.5h |

**Auto-snapshot**: outputs go to `runs/<TS>/{args.json, log.csv, params.pkl}`.

### 3.2 Quick smoke-train (5 min, verifies pipeline only)

```bash
python3 scripts/train.py \
  --num-envs 64 --total-timesteps 50000 \
  --n-rays 360 --max-human-num 10 --human-num 5 \
  --n-boxes 4 --max-steps 80 --log-interval 5 \
  --output runs/smoke
```

### 3.3 Background + log streaming

```bash
nohup python3 -u scripts/train.py [args] > train.log 2>&1 &
tail -f train.log | grep -E "rew=|DONE"
```

### 3.4 GPU memory tuning

If you OOM (RTX 3070 = 8GB cap), drop `--num-envs`:

| GPU VRAM | Recommended `--num-envs` |
|---|---|
| 8 GB (3070, T4) | 256 |
| 12 GB (3080 Ti) | 512 |
| 16 GB (4080) | 1024 |
| 24 GB (4090, A6000) | 2048 |
| 40-80 GB (A100, H100) | 3072–4096 |

> **Note**: bigger `--num-envs` ≠ proportional speedup. Past ~2048 the bottleneck
> is the Transformer's attention compute, not memory. See `docs/AUDIT.md`
> section O5 for details.

---

## 4. Evaluation

```bash
python3 scripts/eval.py \
  --params runs/<TS>/params.pkl \
  --n-episodes 500 \
  --max-steps 200 \
  --n-rays 720 \
  --max-human-num 45 \
  --human-num 40 \
  --n-boxes 6 \
  --seed 2026
```

Outputs the 7 paper metrics (MDE, AFDE, WRP, SR, HCR, OCR, TLR). Also writes
CSV-style structured logs to stdout. For a random-policy baseline use `--random`.

---

## 5. Jenkins / CI integration

### 5.1 Jenkins pipeline (declarative)

```groovy
pipeline {
    agent { label 'gpu' }   // pick GPU runner
    stages {
        stage('checkout') {
            steps { checkout scm }
        }
        stage('install') {
            steps {
                sh 'python3 -m pip install --user -e . ' +
                   '-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
            }
        }
        stage('test') {
            steps {
                sh 'for t in tests/test_*.py; do python3 $t; done'
            }
        }
        stage('train') {
            steps {
                sh '''
                    PYTHONUNBUFFERED=1 python3 -u scripts/train.py \
                      --num-envs 256 --total-timesteps 10000000 \
                      --n-rays 720 --max-human-num 45 --human-num 40 \
                      --n-boxes 6 --max-steps 200 --seed 215 \
                      --output runs/jenkins-$BUILD_NUMBER
                '''
            }
        }
        stage('eval') {
            steps {
                sh '''
                    python3 scripts/eval.py \
                      --params runs/jenkins-$BUILD_NUMBER/*/params.pkl \
                      --n-episodes 500 --max-steps 200 \
                      --n-rays 720 --max-human-num 45 --human-num 40 --n-boxes 6 \
                      | tee runs/jenkins-$BUILD_NUMBER/eval.txt
                '''
            }
        }
        stage('archive') {
            steps {
                archiveArtifacts artifacts: 'runs/jenkins-*/**', fingerprint: true
            }
        }
    }
}
```

### 5.2 Bare-metal cron / systemd timer (no Jenkins)

```bash
# /etc/cron.daily/retrain-jax-policy
#!/usr/bin/env bash
set -e
cd /opt/human-following-jax
git pull -q
python3 scripts/train.py \
  --num-envs 256 --total-timesteps 10000000 \
  --n-rays 720 --max-human-num 45 --human-num 40 \
  --n-boxes 6 --max-steps 200 --seed $((RANDOM)) \
  --output runs/nightly-$(date +%F)
```

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `[CpuDevice(id=0)]` instead of `gpu` | Re-install with `-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`. Verify `nvidia-smi` shows driver. |
| `RESOURCE_EXHAUSTED: Out of memory` | Lower `--num-envs`. See §3.4 table. |
| Compile takes > 5 min | Normal first call. Bigger configs (max_human_num=45 + n_rays=720) take ~30-60s. Subsequent identical runs use XLA cache. |
| Training reward not improving past update 100 | See `docs/AUDIT.md`. Common: wrong hyperparams, OOM-driven param corruption, missing CUDA cudnn. |
| Test `test_lidar.py` fails on `jnp.min(... initial=jnp.inf)` | jax < 0.4.0 doesn't support `initial=`. Upgrade. |
| `ImportError: libcudnn.so.8 not found` | Driver too old. Need CUDA 12 + cudnn 8.9 (bundled in jaxlib==0.4.13+cuda12.cudnn89). |
| `cuDNN version doesn't match` warning | Cosmetic; training still works. |

For more issues, see `docs/AUDIT.md` (what we know is wrong but works), or
file an issue at https://github.com/Gonglitian/human-following-jax/issues.

---

## 7. Repo structure recap

```
human-following-jax/
├── src/
│   ├── env/                  # GPU-resident env (geometry, lidar, dynamics)
│   ├── policy/it_meta.py     # Flax InteractionTransformerMeta
│   └── training/ppo.py       # PureJaxRL-style PPO
├── tests/                    # Run all: for t in tests/test_*.py; do python3 $t; done
├── scripts/
│   ├── train.py              # Main training driver
│   ├── eval.py               # Paper-metrics eval
│   └── bench_gpu_memory.py   # Sweep num_envs for max throughput
├── docs/
│   ├── AUDIT.md              # Mismatches between this repo and the PyTorch original
│   ├── ARCHITECTURE.md       # Design rationale
│   ├── PORT_SCOPE.md         # Per-function port mapping
│   ├── REPRODUCE.md          # Recipe for paper Table II
│   └── TRAINING_RESULTS.md   # Prior run records
├── pyproject.toml
├── README.md / README.zh.md  # English / Chinese overview
└── SETUP.md                  # ← you are here
```

---

## 8. What I changed vs the PyTorch original (high-level)

(See `docs/AUDIT.md` for the full diff.)

- **Env on GPU** — every step is JIT'd into one XLA program; no Python loops
- **Helbing social force** replaces RVO2 (RVO is C++, can't vmap)
- **Axis-aligned boxes** replace Shapely rotated obstacle layouts
- **Constant-velocity human prediction** replaces GST neural network
- **Per-axis radius collision math**, **L2-norm action clip**, **paper-matched
  constants** (target_speed 1.0, collision_penalty -20, etc.)
- **Sinusoidal positional encoding** + **orthogonal weight init** match the
  paper's PyTorch policy exactly
- **PPO**: max_grad_norm=0.1, clip=0.02, lr=4e-5, gamma=0.99, λ=0.95, 5 epoch ×
  8 minibatch (paper exact)

---

## 9. License & support

BSD-3-Clause. Original paper: *Learning Customizable Human Following*, RSS
2026 (TASL Lab). Contact: [@Gonglitian](https://github.com/Gonglitian)
