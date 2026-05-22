# Port Scope — PyTorch/numpy → JAX

Audited from `~/human-following/human-following-robot/` (branch `pure_rl_meta_guided`).
Total ~28K LOC; ~3.7K LOC in env, ~3K in RL networks.

## Observation space (per env)

| Key | Shape | Dtype | Source |
|---|---|---|---|
| `robot_node` | (1, 7) | float32 | px, py, r, gx, gy, v_pref, theta |
| `temporal_edges` | (1, 2) | float32 | vx, vy |
| `spatial_edges` | (max_human_num=45, 12) | float32 | per-human relative xy at t..t+5 (predict_steps=5) |
| `detected_human_num` | (1,) | float32 | count |
| `target_human_traj` | (12,) | float32 | target xy at t..t+5 |
| `local_ogm` | (3, 50, 50) | int8 | stacked OGM history (10m × 10m, 0.2m/cell) |
| `following_preference` | (1, 1) | float32 | scalar ∈ [-2, 2] |

## Action space

`Box(-inf, inf, shape=(2,))` — continuous (vx, vy). Clipped to `robot.v_pref=1.2` downstream.

## Reward

```
collision (obs/human/lost_target):  collision_penalty (= -20 default)
success (target reached):           success_reward     (= +10 default)
otherwise:
  reward = 0
  + (dmin - discomfort_dist) * factor * dt   if dmin < discomfort_dist
  + (obs_dmin - obs_disc_dist) * factor * dt if obs_dmin < obs_disc_dist
  + future_trajectory_collision_penalty (predict steps × 1/2^k decay)
```

## Env config (defaults)

- `time_step = 0.25` s
- `arena_size = 18` m
- `human_num = 40` (range 0)
- `predict_steps = 5`
- `robot.v_pref = 1.2`, `radius = 0.3`, `kinematics = holonomic`
- `sensor_range = 5`, `FOV = 2 rad`

## Policy architecture

`InteractionTransformerMeta` (extends `InteractionTransformer`):

| Layer | Shape |
|---|---|
| OGM CNN: 3-stack → Conv(5×5, 64) → Conv(3×3, 128) → Conv(3×3, 256) → AdaptiveMaxPool(2,2) → FC(1024→512→256) | 256 |
| robot_embedding: 10 → 128 → 256 (meta adds following_preference) | 256 |
| human_embedding: 12 → 128 → 256 | 256 |
| target_embedding: 12 → 128 → 256 | 256 |
| obstacle_embedding: 256 → 256 (linear) | 256 |
| TransformerEncoder: 4 layers, 8 heads, dim_feedforward=1024 | 256 |
| transformer_output_layer: 256 → 128 → 64 (= `human_node_output_size`) | 64 |
| Critic + Actor (DiagGaussian) | 1 + 2 |

## Per-function port plan

| Original | LOC | Plan | Notes |
|---|---|---|---|
| `crowd_sim_following.reset()` | 200 | **Keep in numpy** | Procedural maze gen w/ shapely; runs once per episode, not on hot path |
| `crowd_sim_following.step()` | 80 | **Port to JAX** | Core hot loop |
| `crowd_sim_following.generate_ob()` | 120 | **Port to JAX** | Constructs obs dict each step |
| `crowd_sim_following.calc_reward()` | 100 | **Port to JAX** | Pure numpy math; straightforward |
| `crowd_sim_following.generate_lidar_scan()` | 60 | **Port to JAX** | Vectorize ray-polygon intersect |
| `crowd_sim_following.generate_lidar_ogm()` | 80 | **Port to JAX** | OGM rasterization |
| `crowd_sim_following.generate_obstacles()` | 200 | **Keep in numpy** | Reset path, shapely-heavy |
| `crowd_sim.get_human_actions()` | 60 | **Replace with social-force JAX** | RVO2 is C++ no-vmap; substitute with Helbing social force |
| `lidar_sensor.generate_scan()` | 100 | **Port to JAX** | Hot path (62% of CPU per profile) |
| `lidar_sensor._render_static_obstacles()` | 50 | **Drop / inline into step** | Was cpp; replace with closed-form ray-box intersect |
| `lidar_sensor.generate_filtered_ogm()` | 80 | **Port to JAX** | Hot path |
| `human.act()` | 50 | **Replace** | Currently calls RVO; replace with social-force step |
| GST trajectory prediction | external | **Drop initially** | Use constant-velocity extrapolation for predict_steps |
| `InteractionTransformerMeta` policy | 118 | **Port to Flax** | Match shapes for diff-test |
| `InteractionTransformer` base | 326 | **Port to Flax** | Same |
| `OGM_CNN` | 50 | **Port to Flax** | Same shapes |
| PPO training loop | 350 | **Rewrite (PureJaxRL pattern)** | `lax.scan` for rollouts, all-on-GPU |
| ShmemVecEnv | 154 | **Drop** | Replaced by jax.vmap |

## Simplifications (vs original)

To get a working JAX env quickly, we **simplify**:

1. **Obstacles** — axis-aligned boxes only (no shapely rotations). Reset still uses numpy to generate them; step uses closed-form ray-box intersection.
2. **Human dynamics** — social force (Helbing) instead of RVO. Trajectories will differ from training original but the policy should generalize (it consumes only LiDAR + relative human pos).
3. **GST predictor** — initially constant velocity. Can swap later if needed for parity.
4. **Maze topology** — fixed procedural (random walls + N inner boxes), same family as original.

## Out of scope (initial)

- HuNav integration
- ROS wrappers
- Visualizer (port only what's needed for unit testing)
- Baseline policies (MPC/ORCA/CRL/RLPC) — separate concern

## Acceptance criteria

1. JAX env step time on GPU ≥ **10× faster** than current 218ms/single-env baseline
2. Policy forward pass on dummy input produces output within 1e-4 of PyTorch reference (after weight transfer)
3. Full PPO training loop runs end-to-end (1+ updates) with no NaN
4. Tests pass (`tests/test_env.py`, `tests/test_policy.py`, `tests/test_speedup.py`)
5. README has copy-pasteable train command + expected throughput
