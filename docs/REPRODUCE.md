# Reproducing the paper

This is the recipe for retraining a `human-following-jax` policy that should
match the original paper's `meta_4.pt` (RSS 2026 *Learning Customizable Human
Following*) on the same metrics suite.

## Environment

| Item | Value |
|---|---|
| OS / Python | Ubuntu 20.04 / Python 3.8 |
| JAX / jaxlib | 0.4.13 (cuda12 wheels from the JAX Google bucket) |
| GPU | tested on RTX 3070 8GB; expected to scale to 4090/A100 |
| Dependencies | `pip install -e .` then verify `import jax; print(jax.devices())` shows `[gpu(id=0)]` |

## Train command

```bash
/usr/bin/python3 scripts/train.py \
  --num-envs 1024 \
  --total-timesteps 5000000 \
  --num-steps 30 \
  --ppo-epoch 5 \
  --num-mini-batch 8 \
  --lr 4e-5 \
  --clip-param 0.02 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --n-rays 720 \
  --max-human-num 45 \
  --human-num 10 \
  --n-boxes 6 \
  --max-steps 200 \
  --output runs/paper_repro \
  --seed 215
```

Expected wall-clock on RTX 3070: ~1.5 – 2 hr (vs ~4 days for the original).

Hyperparameters match `human-following-robot/arguments.py` line-for-line
(verified 2026-05-21). The only change is `num_processes` → `num_envs` (here we
go bigger to occupy the GPU).

## Eval

```bash
/usr/bin/python3 scripts/eval.py \
  --params runs/paper_repro/<TS>/params.pkl \
  --n-episodes 200 --max-steps 200 \
  --n-rays 720 --max-human-num 45 --human-num 10 --n-boxes 6
```

Output is mean ± std for **MDE / AFDE / WRP / SR / HCR / OCR / TLR** — same
metric set as `human-following/scripts/analyze_metrics.py`.

## Expected metrics

Target (paper Table II, `Ours / meta_4`):

| Metric | Paper (sim) |
|---|---|
| MDE ↓ | ~0.3 m |
| AFDE ↓ | ~0.2 m |
| WRP ↑ | > 80 % |
| SR ↑ | > 95 % |
| HCR ↓ | < 5 % |
| OCR ↓ | < 5 % |
| TLR ↓ | < 5 % |

**Caveats / known gaps** between this repo and the original sim:

1. **Maze topology**: original uses Shapely walls + corridors; we use random
   axis-aligned boxes. The policy may need slight fine-tune on the original
   env if you want exact paper numbers.
2. **Human dynamics**: original RVO2 vs our social force. RVO gives more
   "polite" cooperative avoidance; social force can produce slightly more
   collision warnings. Expect HCR to be ~5 percentage points worse here than
   in the paper if not retuned.
3. **GST predictor**: dropped here, using constant-velocity. Spatial_edges
   carries less information so the Transformer needs to compensate.

These are deliberate trade-offs to keep the env GPU-friendly. To close the gap:
- Add Shapely-based reset (one-time per episode, doesn't affect throughput much)
- Tune social-force parameters to better mimic RVO
- Port GST predictor as a small Flax model that runs inside `env_step`

All are tractable; see ARCHITECTURE.md.

## Reproducing on the real robot

Use the trained `params.pkl` with the ROS 2 wrapper in
`~/human-following/ros2_following/decider/`. You'll need to:

1. Convert Flax pytree params → PyTorch state_dict (~50 lines, see
   `scripts/flax_to_pt.py` once implemented).
2. Save as `meta_jax_<TS>.pt` next to `meta_4.pt`.
3. Launch with: `METHOD=ours CKPT=meta_jax_<TS>.pt bash start_real_robot.sh`.

The real-robot decider's metric logger (`scripts/log_ogm_clearance.py`)
produces a CSV that `scripts/analyze_metrics.py` reads — same metric format
as `eval.py` here, so you can directly compare sim vs real numbers.

## Variance + multiple seeds

The paper reports mean over 5 seeds. To reproduce, train with:

```bash
for seed in 1 2 3 4 5; do
  /usr/bin/python3 scripts/train.py --seed $seed --output runs/seed$seed ...
done
```

Then average the eval outputs.
