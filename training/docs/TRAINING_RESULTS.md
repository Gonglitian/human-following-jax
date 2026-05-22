# Training results — paper reproduction

## v3 (final, 2026-05-22) — **basically matches paper**

10M env-step paper-aligned recipe on RTX 3070 8GB. All Phase 2+3 alignment
fixes applied (paper constants, weight init, sinusoidal PE, L2-norm action
clip, atomic mid-run checkpointing).

### Setup

```bash
PYTHONUNBUFFERED=1 /usr/bin/python3 scripts/train.py \
  --num-envs 256 --total-timesteps 10000000 --num-steps 30 \
  --ppo-epoch 5 --num-mini-batch 8 --lr 4e-5 --clip-param 0.02 \
  --gamma 0.99 --gae-lambda 0.95 \
  --n-rays 720 --max-human-num 45 --human-num 40 --n-boxes 6 \
  --max-steps 200 --output runs/paper_repro_v3 --seed 215 --log-interval 20
```

Wall-clock: **5.25h** (18907s). 9.99M env steps / 1302 updates.

### Convergence

| Phase | Update range | Mean reward | Mean loss |
|---|---|---|---|
| Cold start | 1-100 | −1.27 → −0.97 | 31.2 → 14.9 |
| Recovery | 100-420 | −0.97 → +0.003 | 14.9 → 5.9 |
| Productive | 420-1100 | +0.003 → +0.064 | 5.9 → 5.4 |
| Polish | 1100-1300 | +0.030 → +0.034 | 5.4 → 5.4 |

Cross-zero at **update ~440 (34% of training)**. Peak reward +0.064 at update 880.

### Eval — 500 episodes, seed 2026

```bash
/usr/bin/python3 scripts/eval.py \
  --params runs/paper_repro_v3/20260522_084351/params.pkl \
  --n-episodes 500 --max-steps 200 \
  --n-rays 720 --max-human-num 45 --human-num 40 --n-boxes 6 --seed 2026
```

| Metric | v3 | Paper (Ours) | Status |
|---|---|---|---|
| ↑ SR (Success Rate) | **92.80%** | ~95% | within 2.2pp |
| ↓ OCR (Obstacle Collision Rate) | **5.80%** | <10% | ✅ |
| ↓ HCR (Human Collision Rate) | **1.40%** | ~2% | ✅ |
| ↓ TLR (Target Lost Rate) | **0.00%** | ~0% | ✅ |
| ↓ MDE (Mean Distance Error) | 0.94 m | — | — |
| ↓ AFDE (Average Final Distance Error) | 0.82 m | — | — |
| ↑ WRP (Within Reasonable Proximity) | 21.42% | — | — |
|   Mean episode length | 185.6 / 200 | — | — |

**Verdict**: SR is 2.2pp short of the 95% target. OCR/HCR/TLR all hit or beat
paper. With ~10% more training or a different seed sweep, SR should clear 95%.

---

## v3 first run (2026-05-21) — **lost due to save bug**

Same recipe + same seed (215). Trained to completion (1302 updates / 5.6h)
but the end-of-run `params.pkl` save crashed because the relative output path
broke when the `runs/` directory was moved during a mid-flight repo
restructure (see commit `e09f49b`). Convergence trajectory shown in
`runs/paper_repro_v3/20260521_174854/log.csv` (reconstructed from stdout
log). Final reward +0.059, very similar shape to v3 redo.

Fix applied to `scripts/train.py`:
- Output dir resolved to absolute path at startup
- Mid-run `params.ckpt` saved every ~100 updates via atomic rename

---

## v2 (2026-05-21) — **5M-step preview, did not converge**

5M-step recipe (half of paper budget). Reward stayed negative; SR 31% / OCR
35%. This run motivated the move to full 10M budget + Phase 2+3 alignment
fixes for v3.

---

## v0 (300K-step smoke, 2026-05-21)

300K env-step pipeline-validation run on a small env config (n_rays=360,
human_num=5, n_boxes=4, max_steps=80). Just confirms that env + policy +
PPO + auto-reset works end-to-end. Not a meaningful policy.
