"""Social-force human dynamics (replaces RVO2).

Why not RVO2?
  RVO2 is a C++ library (``rvo2-python``) with mutable per-agent state held
  in C++ objects — impossible to vmap across thousands of envs on GPU.

What we do instead:
  Helbing-style social force model. Each human computes acceleration as the
  sum of:
    (i)  a goal-pulling force toward their assigned waypoint, and
    (ii) repulsive forces from other humans, from the robot, and from
         static obstacles (axis-aligned boxes).

  All ops are pure JAX, vmappable over (n_envs, n_humans).

Fidelity vs. RVO:
  Trajectories will differ from RVO in detail (RVO gives more "polite"
  cooperative avoidance), but the macroscopic behavior — humans walk
  toward goals, deflect around obstacles and each other — is preserved.
  The downstream policy only sees LiDAR + relative human positions,
  so it should generalize.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class HumanConfig(NamedTuple):
    radius: float = 0.3            # m
    desired_speed: float = 1.0     # m/s
    relax_time: float = 0.5        # s — how fast they reach desired velocity
    # Social-force parameters (Helbing & Molnár 1995, tuned for indoor crowd)
    repulse_strength: float = 2.0  # peak repulsion magnitude
    repulse_range: float = 0.5     # m — exp falloff scale
    obstacle_repulse: float = 5.0
    obstacle_range: float = 0.3
    max_force: float = 5.0         # clamp to avoid explosions in tight spaces


def goal_force(pos: jax.Array, vel: jax.Array, goal: jax.Array,
               cfg: HumanConfig) -> jax.Array:
    """``pos, vel, goal: (2,)`` → ``(2,)`` force."""
    diff = goal - pos
    dist = jnp.linalg.norm(diff) + 1e-6
    desired_vel = (diff / dist) * cfg.desired_speed
    return (desired_vel - vel) / cfg.relax_time


def pairwise_repulse(self_pos: jax.Array, other_pos: jax.Array,
                     cfg: HumanConfig) -> jax.Array:
    """Exponential repulsion away from one neighbor. ``(2,)`` input/output."""
    diff = self_pos - other_pos
    dist = jnp.linalg.norm(diff) + 1e-6
    direction = diff / dist
    magnitude = cfg.repulse_strength * jnp.exp(-(dist - 2 * cfg.radius) / cfg.repulse_range)
    return direction * magnitude


def repulse_from_humans(self_pos: jax.Array, others_pos: jax.Array,
                        valid_mask: jax.Array, cfg: HumanConfig) -> jax.Array:
    """Sum repulsion from all other humans.

    ``self_pos: (2,)``, ``others_pos: (M, 2)``, ``valid_mask: (M,)``.
    """
    def per_other(opos, valid):
        f = pairwise_repulse(self_pos, opos, cfg)
        return jnp.where(valid, f, jnp.zeros(2))
    forces = jax.vmap(per_other)(others_pos, valid_mask)
    return forces.sum(axis=0)


def repulse_from_boxes(self_pos: jax.Array, boxes: jax.Array,
                       cfg: HumanConfig) -> jax.Array:
    """Repulsion away from the nearest point on each axis-aligned box.

    ``boxes: (N, 4)`` ``[xmin, ymin, xmax, ymax]``. Returns ``(2,)`` total force.
    """
    # Nearest point on box to self_pos
    nx = jnp.clip(self_pos[0], boxes[:, 0], boxes[:, 2])
    ny = jnp.clip(self_pos[1], boxes[:, 1], boxes[:, 3])
    nearest = jnp.stack([nx, ny], axis=-1)  # (N, 2)
    diff = self_pos - nearest  # (N, 2)
    dist = jnp.linalg.norm(diff, axis=-1) + 1e-6  # (N,)
    direction = diff / dist[:, None]
    magnitude = cfg.obstacle_repulse * jnp.exp(-(dist - cfg.radius) / cfg.obstacle_range)
    return (direction * magnitude[:, None]).sum(axis=0)


def step_one_human(self_pos: jax.Array, self_vel: jax.Array, goal: jax.Array,
                   others_pos: jax.Array, others_valid: jax.Array,
                   robot_pos: jax.Array, boxes: jax.Array,
                   dt: float, cfg: HumanConfig) -> tuple[jax.Array, jax.Array]:
    """Integrate one human by dt under social force.

    Returns ``(new_pos, new_vel)``.
    """
    f_goal = goal_force(self_pos, self_vel, goal, cfg)
    f_humans = repulse_from_humans(self_pos, others_pos, others_valid, cfg)
    f_robot = pairwise_repulse(self_pos, robot_pos, cfg)
    f_boxes = repulse_from_boxes(self_pos, boxes, cfg)
    f_total = f_goal + f_humans + f_robot + f_boxes
    # Clamp magnitude to avoid blow-ups in tight maze
    mag = jnp.linalg.norm(f_total) + 1e-6
    f_total = f_total * jnp.minimum(1.0, cfg.max_force / mag)
    new_vel = self_vel + f_total * dt
    # cap speed too
    sp = jnp.linalg.norm(new_vel) + 1e-6
    new_vel = new_vel * jnp.minimum(1.0, cfg.desired_speed * 1.5 / sp)
    new_pos = self_pos + new_vel * dt
    return new_pos, new_vel


def step_all_humans(positions: jax.Array, velocities: jax.Array, goals: jax.Array,
                    valid: jax.Array, robot_pos: jax.Array, boxes: jax.Array,
                    dt: float, cfg: HumanConfig) -> tuple[jax.Array, jax.Array]:
    """Step ALL humans (``(M, 2)`` arrays) one timestep.

    Each human sees all OTHER humans as repulsion sources (mask self out).
    """
    M = positions.shape[0]
    # For human i, build (M-1) others view by setting valid[i]=False before pass
    def per_human(i):
        mask = valid.at[i].set(False)
        return step_one_human(
            positions[i], velocities[i], goals[i],
            positions, mask,
            robot_pos, boxes, dt, cfg,
        )
    new_pos_vel = jax.vmap(per_human)(jnp.arange(M))
    new_pos, new_vel = new_pos_vel
    # Keep invalid humans frozen (so they don't drift to NaN-land)
    new_pos = jnp.where(valid[:, None], new_pos, positions)
    new_vel = jnp.where(valid[:, None], new_vel, velocities)
    return new_pos, new_vel
