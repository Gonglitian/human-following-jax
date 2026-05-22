"""Depth Anything V2 wrapper for metric depth estimation."""

from __future__ import annotations

import numpy as np


class DepthEstimator:
    """Wraps Depth Anything V2 (metric indoor variant) for per-frame depth.

    The model is loaded via ``torch.hub`` from the official
    ``DepthAnything/Depth-Anything-V2`` repository on first instantiation.
    Subsequent runs use the cached weights.

    Parameters
    ----------
    model_name:
        Logical model identifier (currently unused beyond documentation;
        the metric-indoor ViT-S checkpoint is always loaded).
    device:
        Torch device string (``"cuda"`` or ``"cpu"``).
    """

    def __init__(
        self,
        model_name: str = 'depth-anything-v2-small',
        device: str = 'cuda',
    ) -> None:
        import torch  # type: ignore[import]

        self.device = device
        self.model = torch.hub.load(
            'DepthAnything/Depth-Anything-V2',
            'depth_anything_v2_vits14_metric_indoor',
            trust_repo=True,
        )
        self.model.to(device)
        self.model.eval()

    def estimate(self, image_bgr) -> np.ndarray:
        """Estimate metric depth for a BGR image.

        Parameters
        ----------
        image_bgr:
            OpenCV BGR image as a ``numpy.ndarray`` with shape ``(H, W, 3)``.

        Returns
        -------
        Depth map in metres as a ``float32`` array with shape ``(H, W)``.
        """
        depth: np.ndarray = self.model.infer_image(image_bgr)
        return depth.astype(np.float32)

    def get_depth_at_point(
        self,
        depth_map: np.ndarray,
        u: int,
        v: int,
        patch_size: int = 5,
    ) -> float:
        """Return a robust depth estimate at pixel ``(u, v)``.

        A small square patch centred on the pixel is sampled and the
        median is returned to suppress noisy outliers.

        Parameters
        ----------
        depth_map:
            Metric depth map returned by :meth:`estimate`.
        u:
            Horizontal pixel coordinate (column index).
        v:
            Vertical pixel coordinate (row index).
        patch_size:
            Side length of the sampling patch in pixels.

        Returns
        -------
        Median depth in metres as a Python ``float``.
        """
        h, w = depth_map.shape
        half = patch_size // 2
        u_min = max(0, u - half)
        u_max = min(w, u + half + 1)
        v_min = max(0, v - half)
        v_max = min(h, v + half + 1)
        patch = depth_map[v_min:v_max, u_min:u_max]
        return float(np.median(patch))
