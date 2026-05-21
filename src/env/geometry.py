"""Vectorized 2D geometry primitives in JAX.

All functions are pure and vmappable over batch dimensions. The conventions:

* Boxes are stored as `(N, 4)` arrays `[x_min, y_min, x_max, y_max]` (axis-aligned).
* Rays are stored as `(R, 2, 2)` arrays `[[ox, oy], [dx, dy]]` with unit direction.
* Circles are `(M, 3)` arrays `[cx, cy, radius]`.

These primitives are the building blocks for the LiDAR scan (ray vs world) and
OGM rasterization. They replace the per-polygon Python loop that the original
C++ helper got called from (which dominated 62% of CPU time per `profile_train_env.py`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def ray_box_intersect(ray_orig: jax.Array, ray_dir: jax.Array, boxes: jax.Array) -> jax.Array:
    """Closed-form ray vs axis-aligned box (slab method).

    Args:
        ray_orig: ``(2,)`` ray origin.
        ray_dir:  ``(2,)`` ray unit direction.
        boxes:    ``(N, 4)`` boxes ``[xmin, ymin, xmax, ymax]``.

    Returns:
        ``(N,)`` ``t`` values where ``t > 0`` means hit at distance ``t``;
        ``inf`` for misses. Caller takes ``min`` over the box axis to pick
        the nearest hit per ray.
    """
    # Guard against zero-direction components — replace with tiny value to avoid div-by-zero.
    eps = 1e-8
    safe_dir = jnp.where(jnp.abs(ray_dir) < eps, jnp.sign(ray_dir) * eps + eps, ray_dir)
    inv = 1.0 / safe_dir

    t1 = (boxes[:, 0] - ray_orig[0]) * inv[0]
    t2 = (boxes[:, 2] - ray_orig[0]) * inv[0]
    t3 = (boxes[:, 1] - ray_orig[1]) * inv[1]
    t4 = (boxes[:, 3] - ray_orig[1]) * inv[1]

    tmin = jnp.maximum(jnp.minimum(t1, t2), jnp.minimum(t3, t4))
    tmax = jnp.minimum(jnp.maximum(t1, t2), jnp.maximum(t3, t4))

    # Hit iff tmax >= max(tmin, 0). Distance returned is max(tmin, 0).
    valid = (tmax >= 0.0) & (tmax >= tmin)
    t_hit = jnp.where(tmin < 0.0, tmax, tmin)
    return jnp.where(valid, t_hit, jnp.inf)


def ray_circle_intersect(ray_orig: jax.Array, ray_dir: jax.Array, circles: jax.Array) -> jax.Array:
    """Ray vs circle (analytic).

    Args:
        ray_orig: ``(2,)`` ray origin.
        ray_dir:  ``(2,)`` ray unit direction.
        circles:  ``(M, 3)`` circles ``[cx, cy, r]``.

    Returns:
        ``(M,)`` distances; ``inf`` for misses.
    """
    oc = ray_orig - circles[:, :2]  # (M, 2)
    b = jnp.sum(oc * ray_dir, axis=-1)  # (M,)
    c = jnp.sum(oc * oc, axis=-1) - circles[:, 2] ** 2  # (M,)
    disc = b * b - c
    valid = disc >= 0.0
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = -b - sq
    t2 = -b + sq
    t_hit = jnp.where(t1 > 0, t1, jnp.where(t2 > 0, t2, jnp.inf))
    return jnp.where(valid, t_hit, jnp.inf)


def cast_one_ray(ray_orig: jax.Array, angle: jax.Array, boxes: jax.Array,
                 circles: jax.Array, max_range: float) -> jax.Array:
    """Cast a single ray, return distance to nearest hit (clipped to ``max_range``).

    Empty obstacle arrays (shape ``(0, ...)``) are allowed — the reduction
    short-circuits to ``+inf`` via ``initial=`` argument.
    """
    ray_dir = jnp.array([jnp.cos(angle), jnp.sin(angle)])
    t_box = jnp.min(ray_box_intersect(ray_orig, ray_dir, boxes), initial=jnp.inf)
    t_cir = jnp.min(ray_circle_intersect(ray_orig, ray_dir, circles), initial=jnp.inf)
    return jnp.minimum(jnp.minimum(t_box, t_cir), max_range)


def cast_rays(ray_orig: jax.Array, angles: jax.Array, boxes: jax.Array,
              circles: jax.Array, max_range: float) -> jax.Array:
    """Cast many rays from a single origin.

    Args:
        ray_orig: ``(2,)`` origin
        angles:   ``(R,)`` ray angles (rad)
        boxes:    ``(N, 4)``
        circles:  ``(M, 3)``
        max_range: scalar

    Returns:
        ``(R,)`` hit distances.
    """
    return jax.vmap(cast_one_ray, in_axes=(None, 0, None, None, None))(
        ray_orig, angles, boxes, circles, max_range
    )
