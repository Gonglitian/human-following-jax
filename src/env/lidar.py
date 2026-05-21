"""JAX LiDAR scan + OGM rasterization.

Replaces ``crowd_sim/envs/utils/lidar_sensor.py`` (682 LOC of Python that called
into C++ ``lidar_ogm_cpp`` once per polygon per LiDAR per env per step — that
loop was 62% of CPU time in the original training).

API:

* ``simulate_scan(robot_xy, robot_yaw, boxes, circles, cfg)`` →
      ``(n_rays,)`` distances. JIT-able + vmap-able over batch.
* ``rasterize_ogm(scan, robot_yaw, cfg)`` →
      ``(H, W)`` binary OGM in robot frame, dtype=int8.

The pipeline matches the real-robot OGM published by ``occupancy_generation``:
50x50 cells, 0.2m/cell, robot-centered, binary.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .geometry import cast_rays


class LidarConfig(NamedTuple):
    n_rays: int = 1080         # SLAMTEC A1 native angular resolution
    max_range: float = 12.0    # m (matches A1)
    angle_min: float = -jnp.pi
    angle_max: float = jnp.pi


class OgmConfig(NamedTuple):
    grid_size: int = 50        # cells per side
    resolution: float = 0.2    # m per cell
    history_len: int = 3       # observation stack


def make_angles(cfg: LidarConfig) -> jax.Array:
    """``(n_rays,)`` evenly spaced ray angles in robot frame."""
    return jnp.linspace(cfg.angle_min, cfg.angle_max, cfg.n_rays, endpoint=False)


def simulate_scan(robot_xy: jax.Array, robot_yaw: jax.Array,
                  boxes: jax.Array, circles: jax.Array,
                  cfg: LidarConfig) -> jax.Array:
    """Generate one LiDAR scan from given robot pose.

    Args:
        robot_xy:  ``(2,)`` robot world position.
        robot_yaw: scalar yaw (rad).
        boxes:     ``(N, 4)`` static obstacles ``[xmin, ymin, xmax, ymax]``.
        circles:   ``(M, 3)`` dynamic obstacles ``[cx, cy, r]`` (humans / target).
        cfg:       LidarConfig.

    Returns:
        ``(n_rays,)`` distances, clipped to ``cfg.max_range``.
    """
    angles_local = make_angles(cfg)
    angles_world = angles_local + robot_yaw
    return cast_rays(robot_xy, angles_world, boxes, circles, cfg.max_range)


def rasterize_ogm(scan: jax.Array, cfg: OgmConfig, lidar_cfg: LidarConfig) -> jax.Array:
    """Convert LiDAR scan to a robot-centered binary OGM.

    Cells the laser passes through stay empty (0), cells where the laser
    terminates become occupied (1). Matches the real-robot OGM convention
    used by ``occupancy_generation`` (centered on robot, axis-aligned).

    Args:
        scan: ``(n_rays,)`` distances in robot frame (from ``simulate_scan``).
        cfg:  OgmConfig.
        lidar_cfg: LidarConfig (needed for ray angles).

    Returns:
        ``(H, W)`` int8 grid; 0 = free, 1 = occupied.
    """
    angles = make_angles(lidar_cfg)
    # Endpoint in robot frame (robot at (0, 0))
    ex = scan * jnp.cos(angles)
    ey = scan * jnp.sin(angles)

    # Grid: cell (i, j) covers [origin + i*res, origin + (i+1)*res). origin = -grid/2.
    half = cfg.grid_size * cfg.resolution / 2.0
    cx = ((ex + half) / cfg.resolution).astype(jnp.int32)
    cy = ((ey + half) / cfg.resolution).astype(jnp.int32)

    # In bounds + ray didn't reach max_range (a hit, not max-out)
    valid = (cx >= 0) & (cx < cfg.grid_size) & (cy >= 0) & (cy < cfg.grid_size)
    valid = valid & (scan < lidar_cfg.max_range)

    flat_idx = cy * cfg.grid_size + cx  # (n_rays,)
    flat_idx = jnp.where(valid, flat_idx, -1)

    # Scatter 1's at the hit cells.  ``segment_max`` gives us "any ray hit
    # this cell" without needing a Python loop.
    grid_flat = jnp.zeros(cfg.grid_size * cfg.grid_size, dtype=jnp.int8)
    ones = jnp.where(valid, jnp.int8(1), jnp.int8(0))
    grid_flat = grid_flat.at[jnp.where(valid, flat_idx, 0)].max(ones)
    return grid_flat.reshape(cfg.grid_size, cfg.grid_size)


# Convenience batched variants ---------------------------------------------------
def simulate_scan_batch(robot_xys, robot_yaws, boxes, circles, cfg):
    """``(B, 2), (B,), (B, N, 4), (B, M, 3) → (B, n_rays)``."""
    return jax.vmap(simulate_scan, in_axes=(0, 0, 0, 0, None))(
        robot_xys, robot_yaws, boxes, circles, cfg
    )


def rasterize_ogm_batch(scans, cfg, lidar_cfg):
    """``(B, n_rays) → (B, H, W)``."""
    return jax.vmap(rasterize_ogm, in_axes=(0, None, None))(scans, cfg, lidar_cfg)
