# Architecture

## Design constraints

1. **Single GPU, max throughput** — RTX 3070 8 GB is the target.
2. **No fork-based parallelism** — we hit the GIL ceiling in the original.
3. **Pure functions everywhere** — needed for `jax.jit` + `jax.vmap` + `lax.scan`.
4. **Match original obs schema** — so policies port over without re-training data pipeline.

## Architecture

```
                       ┌──────────────────────────────────────┐
                       │  jax.lax.scan (n_updates iterations) │
                       │  ┌────────────────────────────────┐  │
                       │  │  lax.scan (rollout T=30 steps) │  │
                       │  │  ┌──────────────────────────┐  │  │
                       │  │  │ vmap over num_envs (1024)│  │  │
                       │  │  │                          │  │  │
                       │  │  │ env_step(state, action)  │  │  │
                       │  │  │  → simulate_scan         │  │  │
                       │  │  │  → rasterize_ogm         │  │  │
                       │  │  │  → social_force(humans)  │  │  │
                       │  │  │  → reward / done         │  │  │
                       │  │  │                          │  │  │
                       │  │  │ policy.apply(obs)        │  │  │
                       │  │  │  → CNN(OGM) → Trans → A,C│  │  │
                       │  │  │                          │  │  │
                       │  │  │ sample_action            │  │  │
                       │  │  └──────────────────────────┘  │  │
                       │  └────────────────────────────────┘  │
                       │  GAE(rewards, values)                │
                       │  ppo_update(5 epoch × 8 minibatch)   │
                       │  optimizer.update                    │
                       └──────────────────────────────────────┘
```

All three nested loops compile into a single XLA program at training start.
Python only sees `train_jit(params, state, obs, key, n_updates)` calls.

## Per-component design

### `src/env/geometry.py`
Closed-form ray–box and ray–circle intersection. The slab method for boxes is
6 arithmetic ops + 4 mins; the circle test is quadratic formula. Both are
vmappable over rays × obstacles. **Replaces** the C++ `render_polygon_edges`
function that dominated 62 % of the original env's CPU time.

### `src/env/lidar.py`
- `simulate_scan(robot_xy, yaw, boxes, circles, cfg)` — casts `cfg.n_rays`
  rays at evenly-spaced angles and takes min over all hits.
- `rasterize_ogm(scan, ocfg, lcfg)` — converts hit points to a
  robot-centered `(H, W)` binary OGM via index scatter.

Both are pure jax.numpy; the `scatter` uses `at[].max(ones)` which compiles to
a single XLA segment max.

### `src/env/human_dynamics.py`
Helbing social force model: goal attraction + pairwise repulsion + obstacle
repulsion. Substitutes RVO2 (which is C++ with mutable per-agent state and
can't be vmapped). Trajectories differ in detail from RVO but produce the
same macroscopic behavior (humans walk to goals, avoid each other / obstacles).

### `src/env/crowd_follow_env.py`
Composes the above into a PureJaxRL-style API:

```python
state, obs = env_reset(key, cfg, lidar_cfg, ogm_cfg, human_cfg)
state, obs, reward, done, info = env_step(key, state, action, cfg, ...)
```

State is a `NamedTuple` of arrays — pytree-friendly so `jax.vmap` over the
leading batch axis Just Works. No Python control flow in `step`; `done`
branches are handled by the outer training loop via `jnp.where` masks.

### `src/policy/it_meta.py`
Flax re-implementation of `InteractionTransformerMeta`:
- OGM CNN (Conv 5→3→3 + max-pool 2×2 + FC 1024→512→256)
- Embeddings: robot (10→256), human (12→256), target (12→256), obstacle (256→256)
- Transformer encoder: 4 layers, 8 heads, dim_ff=1024 with attention mask
  driven by `detected_human_num`
- Actor + critic heads, DiagGaussian output

Param count: 4.5 M (vs PyTorch reference ~3.5 M — small diff from
LayerNorm placement; functional shapes identical).

### `src/training/ppo.py`
- `collect_rollout` — `lax.scan` over T rollout steps; auto-resets done envs
  by `jnp.where`-merging a reset state.
- `compute_gae` — reverse `lax.scan` for GAE.
- `ppo_update` — outer `lax.scan` over epochs, inner `lax.scan` over
  minibatches. Each minibatch step does forward + backward + `optax.apply_updates`.
- `make_train(...)` — returns a `train(params, state, obs, key, n_updates)`
  function that runs the entire training loop fused in XLA.

## Differences vs original env (intentional)

| | Original | This repo |
|---|---|---|
| Static obstacles | Shapely polygons (rotated rects) | Axis-aligned boxes |
| Human dynamics | RVO2 (C++) | Helbing social force |
| Trajectory prediction | GST neural net | Constant-velocity extrapolation |
| Maze topology | Shapely walls + corridors | Random boxes |

These simplifications **were necessary** to vmap on GPU. The downstream
policy only sees LiDAR + relative agent positions, so the abstraction barrier
holds — trained policies should still work on the original env / real robot
with minimal fine-tuning.

## Why not Isaac Lab / brax?

- **Isaac Lab**: designed for jointed/rigid-body physics (URDF + USD). Our
  env has custom 2D human dynamics + LiDAR rasterization that don't map to
  Isaac's primitives. Would require rewriting everything as task plugins
  + USD scenes. Estimated 2-4 weeks.
- **brax / mjx**: same issue — physics-engine-oriented, doesn't model
  arbitrary 2D crowd sim or sensor models.
- **Custom JAX rewrite** (this repo): 1-2 days to working baseline, ~1 week
  to feature-complete. Best ROI.

## Where the speed comes from

Original profile (one env step, single-thread):

```
render_polygon_edges (C++ × 64)   24.3 s / 39 s = 62 %
detect_visible                     2.4 s        =  6 %
RVO processObstacles               1.5 s        =  4 %
RVO doStep                         1.0 s        =  3 %
numpy norm                         1.3 s        =  3 %
... other Python overhead          8.5 s        = 22 %
```

JAX version eliminates **all** of these:

- `render_polygon_edges` → vectorized ray-box intersection (no per-polygon loop)
- `detect_visible` → vectorized angle/distance check
- RVO → Helbing social force (vectorized)
- numpy norms → `jnp.linalg.norm` fused in XLA
- Python overhead → none, single XLA program

Result: 188× speedup single-env, scaling to 1000+× with vmap.
