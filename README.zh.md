# human-following-jax

RSS 2026 论文 *Learning Customizable Human Following* 的 monorepo，含两半：

- [`training/`](training/) — JAX 训练栈，整个 env + PPO 循环搬到 GPU 上跑
  （比原版 PyTorch+C++ 端到端快 ~100×）
- [`deployment/`](deployment/) — ROS 2 部署栈，跑在 **Yahboom Rosmaster X3** 麦轮机器人上
  （LiDAR + UWB + RL decider）

原版 PyTorch 训练代码不收录，外链
[`human-following-robot`](https://github.com/tasl-lab/human-following-robot)。

> English version: [README.md](README.md)

---

## TL;DR — 实测加速

参照基线（原始 `human-following-robot` 代码库，RTX 3070 8GB，2026-05-21 测）：

| 项目 | 原版 (PyTorch + C++ env, 128 forks) | 本项目 (JAX, GPU) |
|---|---|---|
| 单 env 一步 | **218 ms** | **1.16 ms** (188×) |
| 128 envs / 秒 | ~600 | 24,540 (**41×**) |
| 1024 envs / 秒 | n/a | 281,092 (**468×**) |
| 4096 envs / 秒 | n/a | 558,199 (**930×**) |
| 32 768 envs / 秒 | n/a | 711,711 (**~1190×**) |
| 5M env 步纯仿真 | **~4 天** | **~7 秒** |

PPO 更新（4.5M 参数 Transformer，128 envs × 30 步 × 5 epoch × 8 minibatch，
全部 fuse 到一个 `lax.scan`）：

| 配置 | 纯 env 步/秒 | end-to-end (env+update) |
|---|---|---|
| 原版（4 天 → 5M 步）| ~600 | ~14.5 |
| JAX, 128 envs | 24,540 | 1,622 |
| JAX, 1024 envs | 281,092 | 1,337 |
| **相对原版加速** | **41–1190× (env)** | **92–112× (端到端)** |

end-to-end 受 PPO 更新阶段制约（policy 计算量大）。换 A100 / 缩小 policy / 减少 epoch
还能更快。

---

## 为什么要重写

原版训练 CPU 瓶颈死锁：

- `ShmemVecEnv` 128 个 CPU fork
- 每个 fork 跑 218 ms / 步 Python+C++
- 其中 62% 时间在 `lidar_ogm_cpp.render_polygon_edges` —— 每步每 env **调 64 次** 的
  Python 循环

JAX 把整个 env+policy+optimizer 搬到 GPU，用 `vmap` 并行几千个 env，整个
rollout+update 循环都 fuse 到一个 `jax.lax.scan` 的 JIT 程序里。CPU 只看见一次
`train_jit(...)` 调用。

---

## 安装（训练侧）

```bash
git clone https://github.com/Gonglitian/human-following-jax.git
cd human-following-jax/training
/usr/bin/python3 -m pip install --user -e .
```

**Python 3.8**（实验室大部分机器还在 Ubuntu 20.04 / Foxy）：

```bash
/usr/bin/python3 -m pip install --user \
  jax==0.4.13 jaxlib==0.4.13+cuda12.cudnn89 \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
/usr/bin/python3 -m pip install --user flax==0.7.2 optax==0.1.7 chex==0.1.7 distrax==0.1.3
```

**Python ≥ 3.10** 用最新版（XLA 快 ~20%）：

```bash
pip install jax[cuda12] flax optax
```

验证 GPU：
```bash
/usr/bin/python3 -c "import jax; print(jax.devices())"
```

---

## 快速训练

```bash
cd training
# 5M 步，默认占 ~6GB GPU，RTX 3070 < 2 小时跑完
/usr/bin/python3 scripts/train.py \
  --num-envs 1024 --total-timesteps 5000000 \
  --n-rays 720 --max-human-num 45 --human-num 10 \
  --output runs/
```

输出在 `runs/<时间戳>/{params.pkl, log.csv, args.json}`。

---

## 部署到真机

ROS 2 部署栈完整集成在 [`deployment/`](deployment/) 下，看
[`deployment/README.md`](deployment/README.md) 和 [`DEPLOY.md`](DEPLOY.md)。
重量级第三方依赖（HuNavSim、AWS worlds 等）和 PyTorch 模型权重不进 git，
由 `scripts/install_third_party.sh` 和 `scripts/pull_models.sh`（从 Google
Drive 拉）按需安装。

### num_envs 取值参考

| GPU | 建议 num_envs | 备注 |
|---|---|---|
| 3070 8GB | 1024–2048 | env-only 能跑 32K，但 PPO 更新瓶颈 ~2K |
| 4090 24GB | 4096–8192 | minibatch 更大、收敛更快 |
| A100 40GB | 8192–16384 | 必要时降 `clip_param` |

num_envs 越大 ≠ 总训练越快 —— PPO 更新阶段会被 policy 计算压制。用
`scripts/bench_gpu_memory.py` profile 确认。

---

## 仓库结构

```
human-following-jax/
├── README.md / README.zh.md
├── DEPLOY.md                     # 机器人侧部署 recipe
├── .gitignore
│
├── training/                     # ── JAX 训练栈 ──
│   ├── pyproject.toml
│   ├── SETUP.md                  # GPU 服务器 / Jenkins 部署
│   ├── src/
│   │   ├── env/                  # geometry, lidar, human_dynamics, crowd_follow_env
│   │   ├── policy/it_meta.py     # Flax ITMetaPolicy
│   │   └── training/ppo.py       # PureJaxRL PPO（lax.scan 融合）
│   ├── scripts/                  # train.py, eval.py, bench_gpu_memory.py
│   ├── tests/                    # test_env, test_policy, test_training_smoke 等
│   ├── docs/                     # PORT_SCOPE, ARCHITECTURE, AUDIT, REPRODUCE
│   └── runs/                     # gitignored
│
└── deployment/                   # ── ROS 2 机器人栈 ──
    ├── README.md
    ├── src/
    │   ├── core/                 # 实验室自研 13 个 ROS 2 包
    │   │   ├── decider/          # RL 推理 + 控制
    │   │   ├── predictor/        # 匀速轨迹预测
    │   │   ├── target_tracker/   # 目标人锁定
    │   │   ├── sort_tracker/     # SORT 多目标跟踪
    │   │   ├── camera_detector/  # RGB-D 检测
    │   │   ├── uwb_tracking/     # LinkTrack UWB Kalman
    │   │   ├── command_listener/ # CLI 模式/偏好
    │   │   ├── depth_costmap/    # 深度图 → costmap
    │   │   ├── occupancy_generation/  # LiDAR → OGM
    │   │   ├── frequency_monitor/     # Hz + 时延 CSV
    │   │   ├── tf_republisher/   # TF 转发
    │   │   ├── fake_detection/   # 仿真测试源
    │   │   └── following_sim/    # 轻量 Gazebo URDF
    │   └── third_party/          # install_third_party.sh 拉
    └── scripts/
        ├── install_third_party.sh  # 克隆 HuNavSim/AWS worlds 等
        └── pull_models.sh          # 从 Google Drive 下载 PyTorch ckpt
```

---

## 跟原版的差异

为了让 env 在 GPU 上跑，做了几处**有意简化**。policy 只见 LiDAR + 相对位置，
理论上能泛化回原版 env / 真机：

1. **静态障碍**变成轴对齐 box（去掉 Shapely 旋转矩形）→ 解析式射线-盒求交
2. **人群动力学**用 Helbing 社会力，不用 RVO2 → 可 vmap，RVO2 是 C++ 状态机
3. **GST 轨迹预测**初版改成匀速外推（够喂 `spatial_edges`）
4. **迷宫拓扑**随机 box（不是 Shapely 走廊）

跟原版一致的部分：
- 观测 schema（`robot_node` / `temporal_edges` / `spatial_edges` /
  `target_human_traj` / `local_ogm` / `detected_human_num` / `following_preference`）
- 5 个离散偏好距离 `{-2: 1.37, -1: 1.90, 0: 2.29, 1: 3.31, 2: 3.80}`
- Policy 架构（OGM CNN → Transformer → actor/critic）
- PPO 超参（`clip_param=0.02`, `lr=4e-5`, `gae_lambda=0.95`, ...）

---

## 测试

```bash
cd training
for t in tests/test_*.py; do /usr/bin/python3 $t; done
```

5 个 suite 全过（GPU 上 ~20 秒）。

---

## Roadmap

- [x] JAX env（几何、LiDAR、OGM、人群动力学、完整 step/reset/reward）
- [x] Flax ITMetaPolicy（架构对齐 PyTorch 原版）
- [x] PureJaxRL PPO（`lax.scan` 融合）
- [x] Benchmarks（env 10-1000×，端到端 100×）
- [x] Tests + docs
- [ ] **收敛跑**复现 paper MDE/SR 指标（待办 — ~1-2 小时 GPU，见 `docs/REPRODUCE.md`）
- [ ] 参数级 diff test 对齐 PyTorch（权重迁移验证 1e-4 内一致）
- [ ] 训三个固定 d* 的 baseline ckpt（0.5/1/1.5m）给真机部署

---

## License

BSD-3-Clause. Source paper: *Learning Customizable Human Following*, RSS 2026 (TASL Lab)。
