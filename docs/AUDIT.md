# JAX rewrite alignment audit vs PyTorch original

> Source: 3 parallel sub-agent audits + manual cross-check (2026-05-21).
> Symptom that triggered audit: trained policy got MDE=0.27m ✓ but OCR=72% ✗ (paper <5%).
> A 4th sub-agent for obs/action ran out of session limit — section deferred (marked 🚧).

## Severity legend

🔴 **CRITICAL** — wrong gradient direction, broken objective, exploit-able
🟠 **HIGH** — significant training signal loss, wrong distribution
🟡 **MEDIUM** — affects convergence speed / final metrics, recoverable
🔵 **LOW** — minor, docs / metric-collection only

---

## 🔴 CRITICAL — fix FIRST

### C1. Missing `success_reward` on timeout
- **Original** `crowd_sim_following.py:1004-1007`: when `global_time >= time_limit - 1`, emits `reward = success_reward = 25`, `Success()`. Surviving the full episode without collision is the **only positive terminal signal**.
- **JAX** `crowd_follow_env.py:340`: `done = collision | (step_count >= 200)`, but **no positive reward** at timeout. `success_reward = 10` is in config but never paid out.
- **Effect**: Reward landscape inverted. With per-step `-distance_error*0.5` always negative, optimal policy is to **lose target on purpose** (one -20 hit beats 200×-1 ≈ -200). Directly explains OCR=72%.
- **Fix**:
  ```python
  timeout = new_state.step_count >= cfg.max_steps
  reward_success = jnp.where(timeout & ~collision, cfg.success_reward, 0.0)
  ```
  Plus bump `success_reward` default 10 → 25.

### C2. Always-negative `reward_distance` term (not in original)
- **Original** `crowd_sim_following.py:1009-1013`: the `# crl single cost for following control` branch sets `reward = 0` for the non-terminal step. The previous "potential-based" distance term is intentionally **commented out**. Following is enforced by `lost_target` termination + discomfort terms.
- **JAX** `crowd_follow_env.py:336`: `reward_distance = -|target_dist - desired_dist| * 0.5` per step. Always negative, dominates reward signal, incentivizes early termination (see C1).
- **Effect**: Combined with C1, this is the primary reason policy prefers to lose target.
- **Fix**: Remove this term entirely. Following is enforced by `lost_target` (target_dist > 4.5m) → -20.

### C3. Missing human discomfort penalty
- **Original** `crowd_sim_following.py:1017-1019`:
  ```python
  if dmin < discomfort_dist (=0.25):
      reward += (dmin - 0.25) * 10 * 0.25  # max -0.625 at dmin=0
  ```
- **JAX**: completely missing.
- **Effect**: No soft gradient to push robot away from non-target humans. Only the abrupt -20 collision penalty. Drives HCR (when present) and contributes to general aggressive behavior.
- **Fix**:
  ```python
  human_radius = 0.2
  edge_dists = jnp.where(state.human_valid,
                         jnp.linalg.norm(new_h_xy - new_robot_xy, axis=-1) - cfg.robot_radius - human_radius,
                         jnp.inf)
  dmin = jnp.min(edge_dists, initial=jnp.inf)
  reward_discomfort_h = jnp.where(dmin < 0.25,
                                  (dmin - 0.25) * 10.0 * cfg.time_step,
                                  0.0)
  ```

### C4. Missing obstacle discomfort penalty
- **Original** `crowd_sim_following.py:1022-1024`:
  ```python
  if obstacle_dmin < obstacle_discomfort_dist (=0.50):
      reward += (obstacle_dmin - 0.5) * 10 * 0.25  # max -1.25 at dmin=0
  ```
- **JAX**: completely missing.
- **Effect**: **Direct cause of OCR=72%**. No gradient pushing robot away from boxes.
- **Fix**:
  ```python
  reward_discomfort_o = jnp.where(min_box < 0.5,
                                  (min_box - 0.5) * 10.0 * cfg.time_step,
                                  0.0)
  ```

### C5. Holonomic action clip: per-axis vs L2-norm
- **Original** `crowd_nav/policy/srnn.py:28-33`: `clip_action(action, v_pref)` rescales by L2 norm → max speed = 1.2 m/s in any direction.
- **JAX** `crowd_follow_env.py:267`: `jnp.clip(action, -v_pref, v_pref)` per-axis → max speed = 1.2×√2 ≈ 1.70 m/s on diagonals.
- **Effect**: 41% speed cheat on diagonals. Policy learns to exploit. **Sim-to-real broken** (mecanum drive caps total speed, not per-axis).
- **Fix**:
  ```python
  speed = jnp.linalg.norm(action) + 1e-6
  v_cmd = action * jnp.minimum(1.0, cfg.robot_v_pref / speed)
  ```

### C6. PPO entropy formula wrong
- **Original** `distributions.py:38-44`: `dist.entropy().mean()` = `Σᵢ [0.5·log(2πe) + log_stdᵢ]` then `.mean()` over batch.
- **JAX** `ppo.py:116`: `jnp.mean(0.5 * (1 + jnp.log(2*pi)) + log_std).sum()` — **averaged across dims instead of summed**, and trailing `.sum()` on scalar is no-op.
- **Effect**: Off by factor `1/A=0.5` for action_dim=2. **Currently latent** because `entropy_coef=0` (default), but will bite the moment exploration is needed.
- **Fix**:
  ```python
  A = action.shape[-1]
  entropy = 0.5 * A * (1.0 + jnp.log(2.0 * jnp.pi)) + jnp.sum(log_std)
  ```

### C7. `max_grad_norm` 5× too loose
- **Original** `arguments.py:108-112`: `--max-grad-norm 0.1` (very aggressive — paired with tiny lr=4e-5 and clip=0.02).
- **JAX** `ppo.py:41`: `max_grad_norm: float = 0.5`.
- **Effect**: Noisier value-function fits. Could destabilize PPO especially early.
- **Fix**: `max_grad_norm: float = 0.1`.

### C8. Transformer missing positional encoding
- **Original** `interaction_transformer.py:61-82,114`: sinusoidal `PositionalEncoding(d_model=256)` added before encoder.
- **JAX** `it_meta.py:158-178`: no PE.
- **Effect**: Original sequence is `[robot, target, obstacle, human_1, ..., human_M]` with positional info matters for the model to know which token is the robot (which it picks at index 0 downstream). Without PE the model can still learn via content but it's weakened — exactly the architectural change that breaks weight transfer if we ever want to do parameter porting.
- **Fix**:
  ```python
  def _sinusoidal_pe(seq_len, d_model):
      pos = jnp.arange(seq_len)[:, None].astype(jnp.float32)
      i = jnp.arange(0, d_model, 2).astype(jnp.float32)
      div = jnp.exp(i * (-jnp.log(10000.0) / d_model))
      pe = jnp.zeros((seq_len, d_model))
      pe = pe.at[:, 0::2].set(jnp.sin(pos * div))
      pe = pe.at[:, 1::2].set(jnp.cos(pos * div))
      return pe[None]
  # In ITMetaPolicy.__call__ before transformer:
  sequence = sequence + _sinusoidal_pe(sequence.shape[1], self.feature_dim)
  ```

---

## 🟠 HIGH — fix after CRITICAL

### H1. Missing future-trajectory collision penalty
- **Original** `crowd_sim_following.py:1051-1062`:
  ```python
  rel = human_future_traj[1:, :, :2] - robot_xy   # [pred_steps, M, 2]
  collision_idx = ||rel|| < 0.5                    # robot.r + human.r
  coeffs = 2^[2..6]                                 # 4, 8, 16, 32, 64
  penalties = -20 / coeffs                          # [-5, -2.5, -1.25, -0.625, -0.3125]
  reward_future = min(collision_idx * penalties)
  reward += reward_future                            # ALWAYS added
  ```
- **JAX**: missing.
- **Effect**: No predictive social shaping. Policy doesn't decelerate / sidestep before contact.
- **Fix**:
  ```python
  ks = jnp.arange(1, cfg.predict_steps + 1) * cfg.time_step  # (K,)
  pred_h = new_h_xy[:, None, :] + new_h_vel[:, None, :] * ks[None, :, None]
  rel = pred_h - new_robot_xy[None, None, :]
  pred_dists = jnp.linalg.norm(rel, axis=-1)
  pred_collision = pred_dists < (cfg.robot_radius + 0.2)
  coeffs = 2.0 ** jnp.arange(2, cfg.predict_steps + 2)
  penalties = cfg.collision_penalty / coeffs
  masked = jnp.where(pred_collision & state.human_valid[:, None],
                     penalties[None, :], 0.0)
  reward_future = jnp.min(masked, initial=0.0)
  ```

### H2. Target human is not an ORCA agent
- **Original**: target IS `humans[0]`, runs ORCA policy with v_pref=1.0, reacts to robot + other humans + obstacles. Has 3s/0.5m stuck detection.
- **JAX** `crowd_follow_env.py:272-284`: target at constant `target_speed=0.8 m/s` straight-line through walls, ignoring robot.
- **Effect**: Target plows through obstacles, ignores robot, gets stuck in corners. Trajectory distribution very different from training.
- **Fix**: Merge target into `step_all_humans` as human[0], use human_xy[0] as target. Bump target speed to 1.0.

### H3. No free-space rejection spawn check
- **Original** `crowd_sim_following.py:385-471`: rejection sampling for every spawn, min-distance enforcement.
- **JAX** `_spawn_agents`: pure uniform, no rejection.
- **Effect**: ~14% of episodes terminate at step 0 (robot spawned inside box) → wasted samples + biased policy.
- **Fix**: Fixed-attempt scan over N candidates, pick farthest from obstacles:
  ```python
  def _sample_free(key, boxes, cfg, n_attempts=16):
      keys = jax.random.split(key, n_attempts)
      pts = jax.vmap(lambda k: jax.random.uniform(
          k, (2,), minval=-cfg.arena_size*0.9, maxval=cfg.arena_size*0.9))(keys)
      nx = jnp.clip(pts[:, 0:1], boxes[None, :, 0], boxes[None, :, 2])
      ny = jnp.clip(pts[:, 1:2], boxes[None, :, 1], boxes[None, :, 3])
      d = jnp.linalg.norm(pts[:, None, :] - jnp.stack([nx, ny], -1), axis=-1).min(-1)
      return pts[jnp.argmax(d)]
  ```

### H4. `human_collision` uses stale `state.human_xy`
- **JAX** `crowd_follow_env.py:318`: `diffs = state.human_xy - new_robot_xy` — pre-step humans vs post-step robot. Off by up to `(v_r + v_h)*dt = 0.675m`.
- **Fix**: `diffs = new_h_xy - new_robot_xy`.

### H5. Weight initialization mismatch
- **Original**: all actor/critic/critic_linear `Linear` use `orthogonal_(√2)` + zero bias. Action mean head uses gain=0.01 (small final-layer trick).
- **JAX**: Flax default = `lecun_normal`. Known PPO best-practice violation; slows convergence.
- **Fix**: Wrap every `nn.Dense` with explicit `kernel_init=nn.initializers.orthogonal(sqrt(2))`, `bias_init=zeros`. Special: action_mean uses `orthogonal(0.01)`, value head uses `orthogonal(1.0)`.

### H6. Half the gradient updates at 5M steps
- **Original**: 128 envs × 30 steps = 3840 transitions per update → at 30M total = 7800 updates.
- **JAX** @256 envs: 7680/update → at 5M = 651 updates. **12× fewer gradient steps** vs original recipe.
- **Effect**: Severely under-optimized vs paper. Explains plateau.
- **Fix**: Either drop `num_envs` to 128 OR raise `total_timesteps` to 30M+ (use the GPU speedup).

### H7. PPO minibatch shuffle key hardcoded `PRNGKey(0)`
- **JAX** `ppo.py:239`: `init_key = PRNGKey(0)` inside `ppo_update` → every update sees the same shuffle pattern. Dramatically reduces shuffle diversity.
- **Fix**: Thread key from `train_one_update`:
  ```python
  def ppo_update(..., key):
      ...
      (params, opt_state, _), out = jax.lax.scan(epoch_body, (params, opt_state, key), ...)
  # In train_one_update:
  key, k_upd = jax.random.split(key)
  params, opt_state, loss, _ = ppo_update(..., k_upd)
  ```

---

## 🟡 MEDIUM

### M1. Default constants drifted from `config.py`
| Field | Original | JAX | Action |
|---|---|---|---|
| `success_reward` | 25 | 10 | bump to 25 (covered by C1) |
| `discomfort_dist` | 0.25 (human) | 0.5 | split into `human_discomfort=0.25`, `obstacle_discomfort=0.5` |
| `discomfort_penalty_factor` | 10 | 5 | bump to 10 |
| `follow_distance_max` | 4.5 | 5.0 | set to 4.5 |
| `human_num` | 40 | 10 | bump to 40 (or rationale) |
| `arena_size` | 18 | 10 | bump to 18 |

### M2. Episode info missing per-type flags
- **Original**: `episode_info` distinguishes `Success / HumanCollision / ObstacleCollision / TargetLost / Danger / Nothing`.
- **JAX**: only `{collision: bool}`.
- **Effect**: Can't compute SR/HCR/OCR/TLR properly per paper definition.
- **Fix**: Add booleans `success, human_collision, box_collision, target_lost, timeout, danger` to `info`.

### M3. Random environment rotation missing
- **Original** `crowd_sim_following.py:791-793`: rotates entire scene by `~U(0, 2π)` after generation. Big data augmentation.
- **JAX**: no rotation. Boxes always axis-aligned.
- **Effect**: Policy may learn axis-aligned shortcuts. Reduces generalization.
- **Fix**: At minimum rotate spawn distributions; ideally rotate boxes too (requires switching from AABB to oriented boxes → rewrite `lidar.py` ray-box intersect to handle rotation, or compose rotation into ray_orig/yaw).

### M4. No human goal-changing mid-episode
- **Original** `config.py:96-103`: `random_goal_changing=True, goal_change_chance=0.5` every 5s. Plus `end_goal_changing=True, end_goal_change_chance=1.0`.
- **JAX**: goals fixed per episode.
- **Effect**: Less varied human trajectories. Trained policy may overfit to predictable paths.
- **Fix**: In `step_all_humans`, when `||pos - goal|| < radius`, sample new goal. Also probabilistic mid-episode change.

### M5. Action input semantics (`step` integration)
- **Original** `agent.py` + `crowd_sim_following.py:843-847`: `desiredVelocity[0] = clip(desiredVelocity[0] + action.v, -v_pref, v_pref)` — action is INCREMENTAL (delta velocity).
- **JAX**: action is direct velocity, integrated to position with `xy + v*dt`.
- **Effect**: Different control modality. Original is acceleration-like, JAX is direct kinematic. Policy can't transfer.
- **Status**: Documented as intentional simplification but worth confirming.

---

## 🔵 LOW

### L1. `detected_human_num` missing floor of 1
- **Original** `crowd_sim_following.py:670-671`: if `detected==0` set to 1.
- **JAX**: no floor. Could be 0.
- **Effect**: Downstream NaN risk if policy divides by it (it doesn't currently, but brittle).
- **Fix**: `detected = jnp.maximum(detected, 1.0)`.

### L2. `spatial_edges` not masked by valid + in-range
- **Original** `crowd_sim_following.py:650, 667`: invalid humans get sentinel value 15.
- **JAX** `crowd_follow_env.py:217-220`: invalid humans get garbage (uniform-random spawn positions never overwritten).
- **Effect**: Network sees fake humans. Could affect attention weights.
- **Fix**:
  ```python
  in_range = jnp.linalg.norm(state.human_xy - state.robot_xy, axis=-1) < lidar_cfg.max_range
  mask = state.human_valid & in_range
  spatial_edges = jnp.where(mask[:, None], rel.reshape(M, -1).astype(jnp.float32), 15.0)
  ```

### L3. No `bad_masks` for time-limit truncation
- **Original** `storage.py:104-115` supports `bad_masks` but `arguments.py:149` defaults `use_proper_time_limits=False`. So matches JAX behavior. Documentation only.

### L4. Collision uses `2*robot_radius` instead of `robot_radius + human_radius`
- **JAX** `crowd_follow_env.py:323`: `human_collision = min_h < (cfg.robot_radius * 2) = 0.6`.
- **Original** uses `robot_radius (0.3) + human_radius (0.2) = 0.5`.
- **Fix**: Add `human_radius=0.2` to `EnvConfig`, use `cfg.robot_radius + cfg.human_radius`. Also use everywhere (in `_humans_as_circles`, in C3 fix).

---

## 🚧 PENDING — obs+action audit (sub-agent ran out of session)

Will manually verify:
- Coordinate frame consistency (robot frame vs odom frame) per obs key
- OGM value range: original `int8 ∈ {-1, 0, 100}`, JAX `int8 ∈ {0, 1}` — **possible CNN-input issue**
- `local_ogm` history is correctly stacked oldest-to-newest in both
- `following_preference` value range vs preference_index mapping
- Reset deterministic seeding (train/val/test seed disjoint)

---

# Recommended fix order

Phase 1 (CRITICAL — should fix OCR from 72% → ~10%):
- C1 + C2 + C3 + C4 (reward shape + missing discomfort terms) — **highest impact, expect SR jump >70%**

Phase 2 (CRITICAL — affects transfer + correctness):
- C5 (action clip), C6 (entropy), C7 (grad clip), C8 (positional encoding)

Phase 3 (HIGH):
- H1 (future-traj), H2 (target as ORCA), H3 (spawn rejection), H4 (collision frame)
- H5 (weight init), H6 (training budget), H7 (shuffle key)

Phase 4 (MEDIUM):
- M1 (constants), M2 (info flags), M3 (random rotation), M4 (goal-changing), M5 (action integration)

Phase 5 (LOW):
- L1, L2, L4

Phase 6: re-train + re-eval, compare to paper Table II.

---

## Files

- `/home/lee/human-following/human-following-jax/src/env/crowd_follow_env.py`
- `/home/lee/human-following/human-following-jax/src/env/human_dynamics.py`
- `/home/lee/human-following/human-following-jax/src/policy/it_meta.py`
- `/home/lee/human-following/human-following-jax/src/training/ppo.py`
- `/home/lee/human-following/human-following-jax/scripts/train.py`
- `/home/lee/human-following/human-following-robot/crowd_sim/envs/crowd_sim_following.py`
- `/home/lee/human-following/human-following-robot/crowd_sim/envs/crowd_sim.py`
- `/home/lee/human-following/human-following-robot/crowd_nav/configs/config.py`
- `/home/lee/human-following/human-following-robot/rl/networks/interaction_transformer.py`
- `/home/lee/human-following/human-following-robot/rl/networks/distributions.py`
- `/home/lee/human-following/human-following-robot/rl/ppo/ppo.py`
- `/home/lee/human-following/human-following-robot/arguments.py`
