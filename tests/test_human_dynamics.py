"""Unit tests for social-force human dynamics."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
import numpy as np

from env.human_dynamics import (
    HumanConfig, goal_force, step_one_human, step_all_humans,
)


def test_goal_pull_toward_goal():
    """Human at origin with goal to right should accelerate +x."""
    cfg = HumanConfig()
    pos = jnp.array([0.0, 0.0])
    vel = jnp.zeros(2)
    goal = jnp.array([5.0, 0.0])
    f = goal_force(pos, vel, goal, cfg)
    assert f[0] > 0, f"expected +x force, got {f}"
    assert abs(float(f[1])) < 1e-5


def test_single_human_walks_to_goal():
    """Integrate one human for ~5s, check it makes progress toward goal."""
    cfg = HumanConfig()
    pos = jnp.array([0.0, 0.0])
    vel = jnp.zeros(2)
    goal = jnp.array([5.0, 0.0])
    others_pos = jnp.zeros((0, 2))
    others_valid = jnp.zeros(0, dtype=bool)
    robot_pos = jnp.array([100.0, 100.0])  # far away
    boxes = jnp.zeros((0, 4))
    dt = 0.1

    for _ in range(50):  # 5s
        pos, vel = step_one_human(pos, vel, goal, others_pos, others_valid,
                                  robot_pos, boxes, dt, cfg)
    # After 5s of ~1 m/s walking, should be ≥3m from start toward (5,0)
    assert float(pos[0]) > 3.0, f"expected progress, got pos={pos}"


def test_humans_avoid_each_other():
    """Two humans walking toward each other (slightly off-axis to break symmetry)
    should deflect or at least not penetrate one another's radius."""
    cfg = HumanConfig()
    # Tiny y offset to break perfect symmetry (matches real-world noise).
    positions = jnp.array([[-3.0, 0.05], [3.0, -0.05]])
    velocities = jnp.zeros((2, 2))
    goals = jnp.array([[3.0, 0.05], [-3.0, -0.05]])
    valid = jnp.array([True, True])
    robot_pos = jnp.array([100.0, 100.0])
    boxes = jnp.zeros((0, 4))
    dt = 0.1
    min_dist = float('inf')
    for _ in range(80):
        positions, velocities = step_all_humans(
            positions, velocities, goals, valid, robot_pos, boxes, dt, cfg
        )
        d = float(jnp.linalg.norm(positions[0] - positions[1]))
        min_dist = min(min_dist, d)
    assert not jnp.any(jnp.isnan(positions))
    # Bodies (radius 0.3 each) shouldn't overlap during the swap
    assert min_dist > 0.4, f"humans penetrated each other: min_dist={min_dist:.3f}"


def test_vmap_over_envs():
    """Run step_all_humans over 4 parallel envs."""
    cfg = HumanConfig()
    B = 4
    M = 5
    positions = jnp.zeros((B, M, 2))
    velocities = jnp.zeros((B, M, 2))
    goals = jnp.ones((B, M, 2)) * 5.0
    valid = jnp.ones((B, M), dtype=bool)
    robot_pos = jnp.zeros((B, 2))
    boxes = jnp.zeros((B, 0, 4))

    @jax.jit
    def step_batch(p, v, g, va, rp, bx):
        return jax.vmap(step_all_humans, in_axes=(0, 0, 0, 0, 0, 0, None, None))(
            p, v, g, va, rp, bx, 0.1, cfg
        )

    new_p, new_v = step_batch(positions, velocities, goals, valid, robot_pos, boxes)
    assert new_p.shape == (B, M, 2)
    assert not jnp.any(jnp.isnan(new_p))


if __name__ == '__main__':
    test_goal_pull_toward_goal(); print('✓ goal force points to goal')
    test_single_human_walks_to_goal(); print('✓ single human walks to goal')
    test_humans_avoid_each_other(); print('✓ humans deflect on collision')
    test_vmap_over_envs(); print('✓ vmap over envs')
    print('\nAll human dynamics tests passed!')
