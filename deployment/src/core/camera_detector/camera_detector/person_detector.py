"""YOLO11-Pose person detector with keypoint-based position estimation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# COCO 17-keypoint indices
KP_NOSE, KP_LEYE, KP_REYE, KP_LEAR, KP_REAR = 0, 1, 2, 3, 4
KP_LSHO, KP_RSHO, KP_LELB, KP_RELB, KP_LWRI, KP_RWRI = 5, 6, 7, 8, 9, 10
KP_LHIP, KP_RHIP, KP_LKNE, KP_RKNE, KP_LANK, KP_RANK = 11, 12, 13, 14, 15, 16

# Anchor joints for robust position estimation (Mono-RPF-style voting)
ANCHOR_JOINTS = [KP_LHIP, KP_RHIP, KP_LSHO, KP_RSHO, KP_LKNE, KP_RKNE, KP_LANK, KP_RANK]

# Torso joints for depth sampling (most stable body region)
TORSO_JOINTS = [KP_LSHO, KP_RSHO, KP_LHIP, KP_RHIP]

DetectionList = List[Tuple[int, int, int, int, float]]
KeypointsDict = Dict[str, object]  # {'xy': ndarray, 'conf': ndarray, 'n_visible': int, 'anchor_center': tuple}


class PersonDetector:
    """YOLO11-Pose detector: outputs bbox + 17 keypoints per person.

    Key advantage over bbox-only:
      Under partial occlusion, visible joints still provide position estimates
      via anchor joint voting (same principle as Mono-RPF, but 10x faster).
    """

    def __init__(
        self,
        model_name: str = 'yolo11n-pose.pt',
        conf_threshold: float = 0.5,
        device: str = 'cuda',
        kp_conf_threshold: float = 0.5,
    ) -> None:
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.device = device
        self.kp_conf_threshold = kp_conf_threshold

    def detect(self, image_bgr) -> DetectionList:
        """Bbox-only detection (backward compatible)."""
        results = self.model(
            image_bgr, conf=self.conf_threshold,
            device=self.device, verbose=False,
        )
        dets: DetectionList = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                dets.append((int(x1), int(y1), int(x2), int(y2), conf))
        return dets

    def detect_with_keypoints(self, image_bgr) -> Tuple[DetectionList, List[KeypointsDict]]:
        """Detection with keypoints: returns (detections, keypoints_list)."""
        results = self.model(
            image_bgr, conf=self.conf_threshold,
            device=self.device, verbose=False,
        )
        dets: DetectionList = []
        kps_list: List[KeypointsDict] = []

        for r in results:
            if r.boxes is None or r.keypoints is None:
                continue
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                dets.append((int(x1), int(y1), int(x2), int(y2), conf))

                xy = r.keypoints.xy[i].cpu().numpy()        # (17, 2)
                kp_conf = r.keypoints.conf[i].cpu().numpy()  # (17,)
                n_visible = int((kp_conf > self.kp_conf_threshold).sum())
                acx, acy = self._anchor_joint_center(xy, kp_conf)

                kps_list.append({
                    'xy': xy,
                    'conf': kp_conf,
                    'n_visible': n_visible,
                    'anchor_center': (acx, acy),
                })
        return dets, kps_list

    def _anchor_joint_center(self, xy: np.ndarray, kp_conf: np.ndarray) -> Tuple[float, float]:
        """Weighted center from visible anchor joints (Mono-RPF-style voting)."""
        weights, positions = [], []
        for j in ANCHOR_JOINTS:
            if kp_conf[j] > self.kp_conf_threshold and xy[j][0] > 0 and xy[j][1] > 0:
                weights.append(float(kp_conf[j]))
                positions.append(xy[j])
        # Fallback: any visible keypoint
        if not positions:
            for j in range(17):
                if kp_conf[j] > self.kp_conf_threshold and xy[j][0] > 0 and xy[j][1] > 0:
                    weights.append(float(kp_conf[j]))
                    positions.append(xy[j])
        if not positions:
            return 0.0, 0.0
        w = np.array(weights)
        p = np.array(positions)
        w /= w.sum()
        center = (w[:, None] * p).sum(axis=0)
        return float(center[0]), float(center[1])

    @staticmethod
    def get_depth_at_keypoints(
        depth_map: np.ndarray,
        kp_xy: np.ndarray,
        kp_conf: np.ndarray,
        kp_threshold: float = 0.5,
        patch_size: int = 5,
    ) -> float:
        """Sample depth at visible torso keypoints (more robust than bbox center)."""
        h, w = depth_map.shape
        half = patch_size // 2
        depths = []
        # Try torso joints first
        for j in TORSO_JOINTS:
            if kp_conf[j] > kp_threshold:
                u, v = int(kp_xy[j][0]), int(kp_xy[j][1])
                if 0 <= u < w and 0 <= v < h:
                    r1, r2 = max(0, v - half), min(h, v + half + 1)
                    c1, c2 = max(0, u - half), min(w, u + half + 1)
                    patch = depth_map[r1:r2, c1:c2]
                    if patch.size > 0:
                        d = float(np.median(patch))
                        if d > 0.1:
                            depths.append(d)
        # Fallback: any visible keypoint
        if not depths:
            for j in range(17):
                if kp_conf[j] > kp_threshold:
                    u, v = int(kp_xy[j][0]), int(kp_xy[j][1])
                    if 0 <= u < w and 0 <= v < h:
                        r1, r2 = max(0, v - half), min(h, v + half + 1)
                        c1, c2 = max(0, u - half), min(w, u + half + 1)
                        patch = depth_map[r1:r2, c1:c2]
                        if patch.size > 0:
                            d = float(np.median(patch))
                            if d > 0.1:
                                depths.append(d)
        return float(np.median(depths)) if depths else 0.0
