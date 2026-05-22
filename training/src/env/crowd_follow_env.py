"""GPU-resident human-following env, JAX rewrite.

API (PureJaxRL-style functional, NOT gym.Env):

    cfg = EnvConfig()
    key, sub = jax.random.split(key)
    env_state, obs = env_reset(sub, cfg)
    for _ in range(...):
        key, sub = jax.random.split(key)
        action = policy(obs)                 # shape (2,) — vx, vy
        env_state, obs, reward, done, info = env_step(sub, env_state, action, cfg)

Vectorize via ``jax.vmap`` over the first axis of state arrays. The whole
``env_step`` is one ``jit``-compiled function — the entire rollout can be
fused into a single XLA program with ``lax.scan`` for max throughput.

Differences vs the original PyTorch env (see docs/PORT_SCOPE.md):
  * Obstacles are axis-aligned boxes (no shapely rotations) — drastically simpler
    geometry, ~free in JAX.
  * Human dynamics use Helbing social force, not RVO2 (vmappable).
  * Trajectory prediction uses constant velocity for ``predict_steps`` frames
    (good enough for spatial_edges; original used GST).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .lidar import LidarConfig, OgmConfig, simulate_scan, rasterize_ogm, make_angles
from .human_dynamics import HumanConfig, step_all_humans, step_one_human


def _step_one_with_social_force(pos, vel, goal, others_pos, others_valid,
                                robot_pos, boxes, dt, cfg):
    """Step the target via Helbing social force (sees other humans + robot +
    boxes as repulsion sources). Thin wrapper around ``step_one_human``."""
    return step_one_human(pos, vel, goal, others_pos, others_valid,
                          robot_pos, boxes, dt, cfg)


# ----- Config -----------------------------------------------------------------
class EnvConfig(NamedTuple):
    arena_size: float = 18.0      # m (half-width). Matches paper sim.arena_size=18.
    n_boxes: int = 6              # number of axis-aligned inner obstacles
    box_min_size: float = 1.5
    box_max_size: float = 3.0
    max_human_num: int = 45       # match original obs_space
    human_num: int = 40           # paper sim.human_num=40
    predict_steps: int = 5
    time_step: float = 0.25
    max_steps: int = 200          # 200 * 0.25s = 50s episode (matches paper time_limit)
    robot_radius: float = 0.3
    human_radius: float = 0.2     # NEW: matches paper config.humans.radius
    robot_v_pref: float = 1.2
    follow_distance_min: float = 0.4              # collision with target
    follow_distance_max: float = 4.5              # target lost (paper: 4.5)
    collision_penalty: float = -20.0              # paper: -20
    success_reward: float = 25.0                  # paper: +25 on timeout-without-collision
    # Discomfort penalty fires per step when within zone:
    #   penalty = (d - zone) * factor * dt  (negative since d < zone)
    discomfort_dist_human: float = 0.25           # paper: 0.25
    discomfort_dist_obstacle: float = 0.50        # paper: 0.50
    discomfort_penalty_factor: float = 10.0       # paper: 10
    target_speed: float = 1.0                     # paper humans.v_pref = 1.0
    # Following preference: which discrete distance to follow at
    # -2: 1.37 | -1: 1.90 | 0: 2.29 | 1: 3.31 | 2: 3.80
    preference_dists: tuple = (1.37, 1.90, 2.29, 3.31, 3.80)


PREFERENCE_DISTANCES = jnp.array([1.37, 1.90, 2.29, 3.31, 3.80])


# ----- State -----------------------------------------------------------------
class EnvState(NamedTuple):
    """Per-env dynamic state. Vmap over leading batch axis."""
    # Robot
    robot_xy: jax.Array          # (2,)
    robot_vel: jax.Array         # (2,)
    robot_yaw: jax.Array         # scalar
    # Target human (just one)
    target_xy: jax.Array         # (2,)
    target_vel: jax.Array        # (2,)
    target_goal: jax.Array       # (2,) — where the target is heading
    # Other humans (M = max_human_num; first `human_num` are valid)
    human_xy: jax.Array          # (M, 2)
    human_vel: jax.Array         # (M, 2)
    human_goal: jax.Array        # (M, 2)
    human_valid: jax.Array       # (M,) bool — currently active
    # Static obstacles (axis-aligned boxes)
    boxes: jax.Array             # (n_boxes, 4)
    # OGM history rolling buffer
    ogm_history: jax.Array       # (history_len, H, W) int8
    # Following preference (which discrete d* to target)
    pref_index: jax.Array        # scalar int in [0..4]
    # Bookkeeping
    step_count: jax.Array        # scalar
    key: jax.Array               # PRNG (for stochastic target re-goal)


# ----- Reset -----------------------------------------------------------------
def _spawn_boxes(key: jax.Array, cfg: EnvConfig) -> jax.Array:
    """Sample ``cfg.n_boxes`` random inner obstacles.

    Returns axis-aligned ``(N, 4)`` boxes spread in the arena.
    No overlap check for simplicity — overlap is fine, just produces larger
    composite shapes which still work as obstacles.
    """
    k_cx, k_cy, k_sz = jax.random.split(key, 3)
    cx = jax.random.uniform(k_cx, (cfg.n_boxes,),
                            minval=-cfg.arena_size + 1,
                            maxval=cfg.arena_size - 1)
    cy = jax.random.uniform(k_cy, (cfg.n_boxes,),
                            minval=-cfg.arena_size + 1,
                            maxval=cfg.arena_size - 1)
    sz = jax.random.uniform(k_sz, (cfg.n_boxes,),
                            minval=cfg.box_min_size,
                            maxval=cfg.box_max_size)
    half = sz / 2.0
    return jnp.stack([cx - half, cy - half, cx + half, cy + half], axis=-1)


def _sample_free(key: jax.Array, boxes: jax.Array, cfg: EnvConfig,
                 n_attempts: int = 16, margin: float = 0.4) -> jax.Array:
    """JAX-friendly rejection sampler: pick the point (out of n_attempts uniform
    samples) farthest from any obstacle. Used for robot/target/human spawns to
    avoid the original env's ~14% "spawn-inside-obstacle" episodes.
    """
    keys = jax.random.split(key, n_attempts)
    def _one(k):
        return jax.random.uniform(k, (2,),
                                  minval=-cfg.arena_size + margin,
                                  maxval=cfg.arena_size - margin)
    pts = jax.vmap(_one)(keys)                                    # (n_attempts, 2)
    nx = jnp.clip(pts[:, 0:1], boxes[None, :, 0], boxes[None, :, 2])  # (n_attempts, n_boxes)
    ny = jnp.clip(pts[:, 1:2], boxes[None, :, 1], boxes[None, :, 3])
    nearest = jnp.stack([nx, ny], axis=-1)                        # (n_attempts, n_boxes, 2)
    d_per_box = jnp.linalg.norm(pts[:, None, :] - nearest, axis=-1)
    d_min = jnp.min(d_per_box, axis=-1)                           # (n_attempts,)
    return pts[jnp.argmax(d_min)]


def _spawn_agents(key: jax.Array, boxes: jax.Array, cfg: EnvConfig):
    """Spawn robot + target + other humans in free space (H3 fix).

    Each agent uses ``_sample_free`` to land away from box obstacles. Target
    additionally spawns within 2-3m of robot for a sensible initial follow
    geometry.

    Returns (robot_xy, target_xy, target_goal, human_xys, human_goals).
    """
    M = cfg.max_human_num
    # Split keys: 1 robot, 1 target-direction, 1 target-goal, M human-xy, M human-goal
    keys = jax.random.split(key, 3 + 2 * M)
    robot_xy = _sample_free(keys[0], boxes, cfg)
    # Target spawns within 2-3m of robot in a random direction
    k_dir = keys[1]
    angle = jax.random.uniform(k_dir, (), minval=0, maxval=2 * jnp.pi)
    dist = jax.random.uniform(k_dir, (), minval=2.0, maxval=3.0)
    target_xy = robot_xy + jnp.array([jnp.cos(angle), jnp.sin(angle)]) * dist
    target_goal = _sample_free(keys[2], boxes, cfg)

    # Per-human free-space spawn + free-space goal — vmap over M
    h_xys = jax.vmap(lambda k: _sample_free(k, boxes, cfg))(keys[3:3 + M])
    h_goals = jax.vmap(lambda k: _sample_free(k, boxes, cfg))(keys[3 + M:3 + 2 * M])
    return robot_xy, target_xy, target_goal, h_xys, h_goals


def env_reset(key: jax.Array, cfg: EnvConfig,
              lidar_cfg: LidarConfig, ogm_cfg: OgmConfig,
              human_cfg: HumanConfig) -> tuple[EnvState, dict]:
    """Initial state + first observation.

    Note: lidar_cfg, ogm_cfg, human_cfg are passed separately so they can be
    static_argnames=()-style baked into jit.
    """
    key, k_box, k_agent, k_pref, k_state = jax.random.split(key, 5)
    boxes = _spawn_boxes(k_box, cfg)
    robot_xy, target_xy, target_goal, h_xys, h_goals = _spawn_agents(k_agent, boxes, cfg)
    pref_index = jax.random.randint(k_pref, (), 0, len(cfg.preference_dists))
    valid = jnp.arange(cfg.max_human_num) < cfg.human_num

    # Initial OGM history: 3 copies of the first scan (no history yet)
    scan = simulate_scan(robot_xy, jnp.array(0.0),
                         boxes, _humans_as_circles(target_xy, h_xys, valid, cfg),
                         lidar_cfg)
    ogm0 = rasterize_ogm(scan, ogm_cfg, lidar_cfg)
    ogm_hist = jnp.stack([ogm0] * ogm_cfg.history_len, axis=0)

    state = EnvState(
        robot_xy=robot_xy,
        robot_vel=jnp.zeros(2),
        robot_yaw=jnp.array(0.0),
        target_xy=target_xy,
        target_vel=jnp.zeros(2),
        target_goal=target_goal,
        human_xy=h_xys,
        human_vel=jnp.zeros((cfg.max_human_num, 2)),
        human_goal=h_goals,
        human_valid=valid,
        boxes=boxes,
        ogm_history=ogm_hist,
        pref_index=pref_index,
        step_count=jnp.array(0),
        key=k_state,
    )
    obs = build_obs(state, cfg, lidar_cfg, ogm_cfg)
    return state, obs


def _humans_as_circles(target_xy, human_xys, human_valid, cfg):
    """Stack target + valid humans as ``(M+1, 3)`` circle obstacles for LiDAR.

    Invalid humans get radius 0 (no LiDAR return).
    """
    target_circle = jnp.array([[target_xy[0], target_xy[1], cfg.robot_radius]])  # treat target like a person
    radii = jnp.where(human_valid, cfg.robot_radius, 0.0)
    human_circles = jnp.concatenate([human_xys, radii[:, None]], axis=-1)
    return jnp.concatenate([target_circle, human_circles], axis=0)


# ----- Observation -----------------------------------------------------------
def build_obs(state: EnvState, cfg: EnvConfig,
              lidar_cfg: LidarConfig, ogm_cfg: OgmConfig) -> dict:
    """Build the obs dict matching the original env's schema."""
    # robot_node: (1, 7) px, py, r, gx, gy, v_pref, theta
    robot_node = jnp.array([
        state.robot_xy[0], state.robot_xy[1],
        cfg.robot_radius,
        state.target_xy[0], state.target_xy[1],  # gx, gy = current target pos
        cfg.robot_v_pref,
        state.robot_yaw,
    ])[None, :]

    # temporal_edges: (1, 2) — robot vel
    temporal_edges = state.robot_vel[None, :]

    # spatial_edges: (max_human_num, 2*(predict_steps+1)) — sorted by distance (O1)
    # constant-velocity extrapolation: pos + vel * k * dt
    ks = jnp.arange(cfg.predict_steps + 1) * cfg.time_step
    rel = (state.human_xy[:, None, :] +
           state.human_vel[:, None, :] * ks[None, :, None] -
           state.robot_xy[None, None, :])  # (M, K, 2)
    spatial_edges_unsorted = rel.reshape(cfg.max_human_num, -1).astype(jnp.float32)

    # Distance for sort + in-range gate. Invalid humans get +inf so they sink
    # to the end. Out-of-range humans also pushed to end and replaced with
    # sentinel 15.0 (matches paper crowd_sim_following.py:667-668).
    cur_dist = jnp.linalg.norm(state.human_xy - state.robot_xy, axis=-1)
    in_range = cur_dist < lidar_cfg.max_range
    visible = state.human_valid & in_range
    sort_key = jnp.where(visible, cur_dist, jnp.inf)
    order = jnp.argsort(sort_key)
    spatial_edges_sorted = spatial_edges_unsorted[order]
    # Replace invisible (now at the end) with sentinel 15
    visible_sorted = visible[order]
    spatial_edges = jnp.where(visible_sorted[:, None], spatial_edges_sorted, 15.0)

    # detected_human_num: count of valid + within sensor_range (floor at 1, L1 fix)
    detected = jnp.maximum(jnp.sum(visible).astype(jnp.float32), 1.0)[None]

    # target_human_traj: (12,) predicted target xy
    target_rel = (state.target_xy[None, :] +
                  state.target_vel[None, :] * ks[:, None] -
                  state.robot_xy[None, :])
    target_traj = target_rel.reshape(-1).astype(jnp.float32)

    # local_ogm: (history_len, H, W) — already in state
    local_ogm = state.ogm_history

    # following_preference: (1, 1) — scalar index mapped to [-2, 2]
    pref = state.pref_index.astype(jnp.float32) - 2.0
    following_pref = pref[None, None]

    return {
        'robot_node': robot_node,
        'temporal_edges': temporal_edges,
        'spatial_edges': spatial_edges,
        'detected_human_num': detected,
        'target_human_traj': target_traj,
        'local_ogm': local_ogm,
        'following_preference': following_pref,
    }


# ----- Step -----------------------------------------------------------------
def env_step(key: jax.Array, state: EnvState, action: jax.Array, cfg: EnvConfig,
             lidar_cfg: LidarConfig, ogm_cfg: OgmConfig,
             human_cfg: HumanConfig) -> tuple[EnvState, dict, jax.Array, jax.Array, dict]:
    """Single env step.

    Args:
        key: PRNG for this step (target re-goal sampling).
        state: previous EnvState.
        action: ``(2,)`` continuous vx, vy in robot frame (we use holonomic).
        cfg, *_cfg: static configs.

    Returns ``(new_state, obs, reward, done, info)``.
    """
    dt = cfg.time_step

    # 1) Robot: L2-norm clip to v_pref (C5 — matches paper srnn.py:28-33,
    #    NOT per-axis clip which would allow ||v||=v_pref*sqrt(2) on diagonals)
    speed = jnp.linalg.norm(action) + 1e-6
    scale = jnp.minimum(1.0, cfg.robot_v_pref / speed)
    new_robot_vel = action * scale
    new_robot_xy = state.robot_xy + new_robot_vel * dt
    new_robot_yaw = jnp.arctan2(new_robot_vel[1], new_robot_vel[0])

    # 2) Target human: now stepped via social force like other humans (H2 fix).
    #    Previously was a constant-velocity drone that ignored obstacles/robot.
    k1, k_humans = jax.random.split(key, 2)
    # Re-goal if reached: matches paper end_goal_changing=True behavior
    diff_tg = state.target_goal - state.target_xy
    dist_to_goal = jnp.linalg.norm(diff_tg) + 1e-6
    reached = dist_to_goal < 0.5
    new_goal_proposal = _sample_free(k1, state.boxes, cfg)
    new_target_goal = jnp.where(reached, new_goal_proposal, state.target_goal)

    # Run target through social force: it sees the robot + ALL other humans
    # as repulsion sources, plus boxes. Output velocity is capped at desired_speed.
    new_target_xy, new_target_vel = _step_one_with_social_force(
        state.target_xy, state.target_vel, new_target_goal,
        state.human_xy, state.human_valid, new_robot_xy,
        state.boxes, dt, human_cfg,
    )

    # 3) Other humans: social force (target_xy is repulsion source for them too,
    #    so they don't run through the target)
    new_h_xy, new_h_vel = step_all_humans(
        state.human_xy, state.human_vel, state.human_goal, state.human_valid,
        new_robot_xy, state.boxes, dt, human_cfg,
    )
    # 4) Recompute LiDAR + OGM
    circles = _humans_as_circles(new_target_xy, new_h_xy, state.human_valid, cfg)
    scan = simulate_scan(new_robot_xy, new_robot_yaw, state.boxes, circles, lidar_cfg)
    new_ogm = rasterize_ogm(scan, ogm_cfg, lidar_cfg)
    # Roll history: drop oldest, append new
    new_ogm_history = jnp.concatenate([state.ogm_history[1:], new_ogm[None]], axis=0)

    new_state = state._replace(
        robot_xy=new_robot_xy,
        robot_vel=new_robot_vel,
        robot_yaw=new_robot_yaw,
        target_xy=new_target_xy,
        target_vel=new_target_vel,
        target_goal=new_target_goal,
        human_xy=new_h_xy,
        human_vel=new_h_vel,
        ogm_history=new_ogm_history,
        step_count=state.step_count + 1,
        key=k_humans,
    )

    # 5) Reward + done — MATCHES paper crowd_sim_following.calc_reward()
    target_dist = jnp.linalg.norm(new_target_xy - new_robot_xy)
    desired_dist = PREFERENCE_DISTANCES[state.pref_index]
    distance_error = jnp.abs(target_dist - desired_dist)

    # --- Collision detection (uses POST-step positions for both robot & humans) ---
    # Human collision: edge-to-edge distance ≤ 0 means touching
    # diffs uses NEW human positions (was bug: previously used stale state.human_xy)
    diffs_h = new_h_xy - new_robot_xy
    h_norm = jnp.linalg.norm(diffs_h, axis=-1)
    h_edge = jnp.where(state.human_valid,
                       h_norm - cfg.robot_radius - cfg.human_radius,
                       jnp.inf)
    min_h_edge = jnp.min(h_edge, initial=jnp.inf)
    min_h_center = jnp.min(jnp.where(state.human_valid, h_norm, jnp.inf), initial=jnp.inf)
    human_collision = min_h_edge < 0.0

    # Target lost (> follow_distance_max=4.5 m)
    target_lost = target_dist > cfg.follow_distance_max

    # Obstacle collision: distance from robot center to nearest box edge
    nx = jnp.clip(new_robot_xy[0], state.boxes[:, 0], state.boxes[:, 2])
    ny = jnp.clip(new_robot_xy[1], state.boxes[:, 1], state.boxes[:, 3])
    box_dists = jnp.linalg.norm(new_robot_xy[None] - jnp.stack([nx, ny], axis=-1), axis=-1)
    min_box = jnp.min(box_dists, initial=jnp.inf)
    obstacle_dmin = min_box - cfg.robot_radius   # edge-to-edge clearance
    box_collision = obstacle_dmin < 0.0

    collision = human_collision | box_collision | target_lost
    timeout = new_state.step_count >= cfg.max_steps
    success = timeout & ~collision

    # --- Reward terms (paper crowd_sim_following.py:992-1062) ---
    # (a) terminal penalty / reward
    reward_terminal = jnp.where(collision, cfg.collision_penalty,
                                jnp.where(success, cfg.success_reward, 0.0))

    # (b) human-discomfort penalty: continuous soft zone
    reward_discomfort_h = jnp.where(
        min_h_edge < cfg.discomfort_dist_human,
        (min_h_edge - cfg.discomfort_dist_human) * cfg.discomfort_penalty_factor * cfg.time_step,
        0.0,
    )

    # (c) obstacle-discomfort penalty
    reward_discomfort_o = jnp.where(
        obstacle_dmin < cfg.discomfort_dist_obstacle,
        (obstacle_dmin - cfg.discomfort_dist_obstacle) * cfg.discomfort_penalty_factor * cfg.time_step,
        0.0,
    )

    # (d) predictive future-trajectory collision (paper §V eq w/ exp decay)
    #   Predict humans at constant velocity for predict_steps frames ahead.
    #   If any predicted human enters robot's collision zone, penalty = -20 / 2^(k+2)
    ks = (jnp.arange(cfg.predict_steps) + 1).astype(jnp.float32) * cfg.time_step  # (K,)
    pred_h = new_h_xy[:, None, :] + new_h_vel[:, None, :] * ks[None, :, None]     # (M, K, 2)
    rel_future = pred_h - new_robot_xy[None, None, :]
    pred_dists = jnp.linalg.norm(rel_future, axis=-1)                              # (M, K)
    pred_collision = pred_dists < (cfg.robot_radius + cfg.human_radius)
    coeffs = 2.0 ** (jnp.arange(cfg.predict_steps).astype(jnp.float32) + 2.0)      # 4, 8, 16, 32, 64
    pen_per_step = cfg.collision_penalty / coeffs                                  # negative
    masked = jnp.where(pred_collision & state.human_valid[:, None],
                       pen_per_step[None, :], 0.0)
    reward_future = jnp.min(masked, initial=0.0)

    reward = reward_terminal + reward_discomfort_h + reward_discomfort_o + reward_future
    done = collision | timeout

    info = {
        'target_dist': target_dist,
        'desired_dist': desired_dist,
        'distance_error': distance_error,
        'min_human_dist': min_h_center,
        'min_obs_dist': min_box,
        # Per-type termination flags (matches paper's Info classes)
        'success': success,
        'human_collision': human_collision,
        'box_collision': box_collision,
        'target_lost': target_lost,
        'timeout': timeout,
        'collision': collision,
    }
    obs = build_obs(new_state, cfg, lidar_cfg, ogm_cfg)
    return new_state, obs, reward, done, info
