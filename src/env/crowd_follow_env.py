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
from .human_dynamics import HumanConfig, step_all_humans


# ----- Config -----------------------------------------------------------------
class EnvConfig(NamedTuple):
    arena_size: float = 10.0      # m (half-width). Total arena = 20×20.
    n_boxes: int = 6              # number of axis-aligned inner obstacles
    box_min_size: float = 1.5
    box_max_size: float = 3.0
    max_human_num: int = 45       # match original obs_space
    human_num: int = 10           # ACTIVE humans this episode (rest invalid)
    predict_steps: int = 5
    time_step: float = 0.25
    max_steps: int = 200
    robot_radius: float = 0.3
    robot_v_pref: float = 1.2
    follow_distance_min: float = 0.4   # collision with target
    follow_distance_max: float = 5.0   # target lost
    collision_penalty: float = -20.0
    success_reward: float = 10.0
    discomfort_dist: float = 0.5
    discomfort_penalty_factor: float = 5.0
    target_speed: float = 0.8
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


def _spawn_agents(key: jax.Array, boxes: jax.Array, cfg: EnvConfig):
    """Spawn robot + target + other humans in free space (not inside boxes).

    Naive uniform sampling — collisions with boxes are tolerated and the social
    force will push agents out. Robust enough for training.

    Returns (robot_xy, target_xy, target_goal, human_xys, human_goals).
    """
    k_r, k_ta, k_td, k_tg, k_hxy, k_hg = jax.random.split(key, 6)
    robot_xy = jax.random.uniform(k_r, (2,),
                                  minval=-cfg.arena_size * 0.5,
                                  maxval=cfg.arena_size * 0.5)
    # Target spawns within 2-3m of robot (sensible initial follow scenario)
    angle = jax.random.uniform(k_ta, (), minval=0, maxval=2 * jnp.pi)
    dist = jax.random.uniform(k_td, (), minval=2.0, maxval=3.0)
    target_xy = robot_xy + jnp.array([jnp.cos(angle), jnp.sin(angle)]) * dist
    # Target goal: far point in arena
    target_goal = jax.random.uniform(k_tg, (2,),
                                     minval=-cfg.arena_size * 0.8,
                                     maxval=cfg.arena_size * 0.8)
    # Other humans uniformly scattered
    M = cfg.max_human_num
    h_xys = jax.random.uniform(k_hxy, (M, 2),
                               minval=-cfg.arena_size + 0.5,
                               maxval=cfg.arena_size - 0.5)
    h_goals = jax.random.uniform(k_hg, (M, 2),
                                 minval=-cfg.arena_size + 0.5,
                                 maxval=cfg.arena_size - 0.5)
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

    # spatial_edges: (max_human_num, 2*(predict_steps+1))
    # constant-velocity extrapolation: pos + vel * k * dt
    ks = jnp.arange(cfg.predict_steps + 1) * cfg.time_step
    # (M, K, 2) relative human positions over time
    rel = (state.human_xy[:, None, :] +
           state.human_vel[:, None, :] * ks[None, :, None] -
           state.robot_xy[None, None, :])
    spatial_edges = rel.reshape(cfg.max_human_num, -1).astype(jnp.float32)

    # detected_human_num: count of valid + within sensor_range
    in_range = jnp.linalg.norm(state.human_xy - state.robot_xy, axis=-1) < lidar_cfg.max_range
    detected = jnp.sum(state.human_valid & in_range).astype(jnp.float32)[None]

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

    # 1) Robot: clip action to v_pref and integrate
    v_cmd = jnp.clip(action, -cfg.robot_v_pref, cfg.robot_v_pref)
    new_robot_vel = v_cmd
    new_robot_xy = state.robot_xy + new_robot_vel * dt
    new_robot_yaw = jnp.arctan2(new_robot_vel[1], new_robot_vel[0])

    # 2) Target human: walk toward target_goal at constant target_speed
    diff = state.target_goal - state.target_xy
    dist_to_goal = jnp.linalg.norm(diff) + 1e-6
    target_vel_unit = diff / dist_to_goal
    new_target_vel = target_vel_unit * cfg.target_speed
    new_target_xy = state.target_xy + new_target_vel * dt
    # When target reaches goal, pick a new random one
    reached = dist_to_goal < 0.5
    k1, k2 = jax.random.split(key)
    new_goal_proposal = jax.random.uniform(k1, (2,),
                                           minval=-cfg.arena_size * 0.8,
                                           maxval=cfg.arena_size * 0.8)
    new_target_goal = jnp.where(reached, new_goal_proposal, state.target_goal)

    # 3) Other humans: social force
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
        key=k2,
    )

    # 5) Reward + done
    target_dist = jnp.linalg.norm(new_target_xy - new_robot_xy)
    desired_dist = PREFERENCE_DISTANCES[state.pref_index]
    distance_error = jnp.abs(target_dist - desired_dist)

    # Collision: robot vs nearest human (excluding target since target is who we follow)
    diffs = state.human_xy - new_robot_xy  # use old state.human_xy is fine
    h_dists = jnp.where(state.human_valid,
                        jnp.linalg.norm(diffs, axis=-1),
                        jnp.inf)
    min_h = jnp.min(h_dists, initial=jnp.inf)
    human_collision = min_h < (cfg.robot_radius * 2)
    target_lost = target_dist > cfg.follow_distance_max

    # Obstacle collision: distance to nearest box edge
    nx = jnp.clip(new_robot_xy[0], state.boxes[:, 0], state.boxes[:, 2])
    ny = jnp.clip(new_robot_xy[1], state.boxes[:, 1], state.boxes[:, 3])
    box_dists = jnp.linalg.norm(new_robot_xy[None] - jnp.stack([nx, ny], axis=-1), axis=-1)
    min_box = jnp.min(box_dists, initial=jnp.inf)
    box_collision = min_box < cfg.robot_radius

    # Reward shaping
    collision = human_collision | box_collision | target_lost
    reward_collision = jnp.where(collision, cfg.collision_penalty, 0.0)
    reward_distance = -distance_error * 0.5  # MDE-aligned
    reward_progress = -jnp.linalg.norm(new_robot_vel) * 0.01  # mild penalty for motion
    reward = reward_collision + reward_distance + reward_progress

    done = collision | (new_state.step_count >= cfg.max_steps)

    info = {
        'target_dist': target_dist,
        'desired_dist': desired_dist,
        'distance_error': distance_error,
        'min_human_dist': min_h,
        'min_obs_dist': min_box,
        'collision': collision,
    }
    obs = build_obs(new_state, cfg, lidar_cfg, ogm_cfg)
    return new_state, obs, reward, done, info
