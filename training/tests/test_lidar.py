"""Unit tests for JAX LiDAR + OGM."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
import numpy as np

from env.geometry import ray_box_intersect, ray_circle_intersect, cast_rays
from env.lidar import LidarConfig, OgmConfig, simulate_scan, rasterize_ogm


def test_ray_box_hit():
    """A ray firing right (+x) should hit a box at x=5 at distance 5."""
    o = jnp.array([0.0, 0.0])
    d = jnp.array([1.0, 0.0])
    boxes = jnp.array([[5.0, -1.0, 6.0, 1.0]])
    t = ray_box_intersect(o, d, boxes)
    assert np.isclose(float(t[0]), 5.0, atol=1e-4), f"expected 5.0, got {float(t[0])}"


def test_ray_box_miss():
    """A ray firing up (+y) shouldn't hit a box that's to the right."""
    o = jnp.array([0.0, 0.0])
    d = jnp.array([0.0, 1.0])
    boxes = jnp.array([[5.0, -1.0, 6.0, 1.0]])
    t = ray_box_intersect(o, d, boxes)
    assert np.isinf(float(t[0])), f"expected inf, got {float(t[0])}"


def test_ray_circle_hit():
    """Ray right hits unit-radius circle at (5, 0) at distance 4."""
    o = jnp.array([0.0, 0.0])
    d = jnp.array([1.0, 0.0])
    circles = jnp.array([[5.0, 0.0, 1.0]])
    t = ray_circle_intersect(o, d, circles)
    assert np.isclose(float(t[0]), 4.0, atol=1e-4), f"expected 4.0, got {float(t[0])}"


def test_cast_rays_360():
    """Cast 360 rays around origin in a box-bounded arena."""
    # 10m × 10m arena (boxes pointing inward)
    boxes = jnp.array([
        [-100., 5., 100., 6.],   # top wall y=5
        [-100., -6., 100., -5.], # bottom wall y=-5
        [5., -100., 6., 100.],   # right wall x=5
        [-6., -100., -5., 100.], # left wall x=-5
    ])
    circles = jnp.zeros((0, 3))
    o = jnp.array([0.0, 0.0])
    angles = jnp.linspace(0, 2 * jnp.pi, 360, endpoint=False)
    t = cast_rays(o, angles, boxes, circles, 100.0)
    # closest distances along ±x, ±y axes should be 5
    # angle 0 → +x → t = 5
    assert np.isclose(float(t[0]), 5.0, atol=0.01), f"+x hit {float(t[0])}, expected 5.0"
    # angle 90 deg → +y
    idx_90 = 360 // 4
    assert np.isclose(float(t[idx_90]), 5.0, atol=0.01)


def test_simulate_scan_runs():
    cfg = LidarConfig()
    boxes = jnp.array([[5., -1., 6., 1.]])
    circles = jnp.zeros((0, 3))
    scan = simulate_scan(jnp.zeros(2), jnp.array(0.0), boxes, circles, cfg)
    assert scan.shape == (cfg.n_rays,)
    # most rays miss → max_range; some hit the box
    assert float(jnp.min(scan)) < 6.0
    assert float(jnp.max(scan)) == cfg.max_range


def test_ogm_rasterize_shape():
    lcfg = LidarConfig()
    ocfg = OgmConfig()
    # Box at 2.5m → solidly within 10x10 robot-centered OGM (extents ±5m)
    boxes = jnp.array([[2.5, -1., 3., 1.]])
    circles = jnp.zeros((0, 3))
    scan = simulate_scan(jnp.zeros(2), jnp.array(0.0), boxes, circles, lcfg)
    grid = rasterize_ogm(scan, ocfg, lcfg)
    assert grid.shape == (ocfg.grid_size, ocfg.grid_size)
    # Some cell should be occupied (the box endpoint)
    assert int(grid.sum()) > 0, f"expected hits, got grid.sum()={int(grid.sum())}, scan.min()={float(scan.min())}"
    # Result dtype is int8
    assert grid.dtype == jnp.int8


def test_jit_works():
    """All ops should compile under jit + vmap."""
    cfg = LidarConfig()
    boxes = jnp.array([[5., -1., 6., 1.]])
    circles = jnp.zeros((0, 3))

    @jax.jit
    def step(xy, yaw):
        return simulate_scan(xy, yaw, boxes, circles, cfg)

    s1 = step(jnp.zeros(2), jnp.array(0.0))
    s2 = step(jnp.array([1.0, 0.0]), jnp.array(0.0))
    assert s1.shape == s2.shape == (cfg.n_rays,)


def test_vmap_works():
    """Batch over 32 envs."""
    cfg = LidarConfig()
    boxes_b = jnp.tile(jnp.array([[5., -1., 6., 1.]])[None], (32, 1, 1))  # (32, 1, 4)
    circles_b = jnp.zeros((32, 0, 3))
    xys = jnp.zeros((32, 2))
    yaws = jnp.zeros(32)
    scans = jax.vmap(simulate_scan, in_axes=(0, 0, 0, 0, None))(
        xys, yaws, boxes_b, circles_b, cfg
    )
    assert scans.shape == (32, cfg.n_rays)


if __name__ == '__main__':
    import time
    test_ray_box_hit(); print('✓ ray-box hit')
    test_ray_box_miss(); print('✓ ray-box miss')
    test_ray_circle_hit(); print('✓ ray-circle hit')
    test_cast_rays_360(); print('✓ 360-ray sweep')
    test_simulate_scan_runs(); print('✓ simulate_scan')
    test_ogm_rasterize_shape(); print('✓ ogm rasterize')
    test_jit_works(); print('✓ jit')
    test_vmap_works(); print('✓ vmap')
    print('\nAll lidar tests passed!')
