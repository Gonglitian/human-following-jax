#!/usr/bin/env python3
"""
Simple appearance-based Re-ID feature extractor.
Uses ResNet18 CNN features + HSV color histogram for robust person matching.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from collections import deque


class ReIDFeatureExtractor:
    """Extracts appearance features from person crops for Re-ID matching."""

    def __init__(self, device='cuda', history_size=10, update_rate=0.1):
        self.device = device
        self.history_size = history_size
        self.update_rate = update_rate  # EMA weight for template update

        # CNN feature extractor (ResNet18, remove final FC)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # output: (1, 512, 1, 1)
        self.cnn.to(device)
        self.cnn.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Target template
        self.target_cnn_feature = None  # (512,) float32
        self.target_color_hist = None   # (180,) float32
        self.feature_history = deque(maxlen=history_size)

    def extract_cnn_feature(self, crop_bgr):
        """Extract 512-dim CNN feature from a person crop (BGR image)."""
        if crop_bgr.size == 0 or crop_bgr.shape[0] < 10 or crop_bgr.shape[1] < 5:
            return None

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.cnn(tensor)
        feat = feat.squeeze().cpu().numpy()  # (512,)
        feat = feat / (np.linalg.norm(feat) + 1e-8)  # L2 normalize
        return feat

    def extract_color_histogram(self, crop_bgr):
        """Extract HSV color histogram from a person crop."""
        if crop_bgr.size == 0 or crop_bgr.shape[0] < 10 or crop_bgr.shape[1] < 5:
            return None

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        # Use upper body region (top 60%) for more stable colors
        h = hsv.shape[0]
        upper = hsv[:int(h * 0.6), :, :]
        hist = cv2.calcHist([upper], [0], None, [180], [0, 180])
        hist = hist.flatten().astype(np.float32)
        hist = hist / (hist.sum() + 1e-8)  # normalize
        return hist

    def extract_feature(self, crop_bgr):
        """Extract combined feature from person crop."""
        cnn_feat = self.extract_cnn_feature(crop_bgr)
        color_hist = self.extract_color_histogram(crop_bgr)
        return cnn_feat, color_hist

    def set_target(self, crop_bgr):
        """Initialize target template from a person crop."""
        cnn_feat, color_hist = self.extract_feature(crop_bgr)
        if cnn_feat is None or color_hist is None:
            return False
        self.target_cnn_feature = cnn_feat
        self.target_color_hist = color_hist
        self.feature_history.clear()
        self.feature_history.append(cnn_feat)
        return True

    def match(self, crop_bgr):
        """
        Match a person crop against the target template.
        Returns similarity score in [0, 1]. Higher = more similar.
        """
        if self.target_cnn_feature is None:
            return 0.0

        cnn_feat, color_hist = self.extract_feature(crop_bgr)
        if cnn_feat is None:
            return 0.0

        # CNN cosine similarity (primary signal)
        cnn_sim = float(np.dot(cnn_feat, self.target_cnn_feature))

        # Color histogram correlation (secondary signal)
        color_sim = 0.0
        if color_hist is not None and self.target_color_hist is not None:
            color_sim = float(cv2.compareHist(
                color_hist, self.target_color_hist, cv2.HISTCMP_CORREL
            ))
            color_sim = max(0.0, color_sim)  # clamp negative correlations

        # Weighted combination: 70% CNN + 30% color
        similarity = 0.7 * cnn_sim + 0.3 * color_sim
        return similarity

    def update_template(self, crop_bgr):
        """Update target template with EMA using a confirmed match."""
        cnn_feat, color_hist = self.extract_feature(crop_bgr)
        if cnn_feat is None:
            return

        # EMA update
        alpha = self.update_rate
        self.target_cnn_feature = (1 - alpha) * self.target_cnn_feature + alpha * cnn_feat
        self.target_cnn_feature /= (np.linalg.norm(self.target_cnn_feature) + 1e-8)

        if color_hist is not None and self.target_color_hist is not None:
            self.target_color_hist = (1 - alpha) * self.target_color_hist + alpha * color_hist
            self.target_color_hist /= (self.target_color_hist.sum() + 1e-8)

        self.feature_history.append(cnn_feat)

    def has_target(self):
        """Check if a target template is set."""
        return self.target_cnn_feature is not None

    def clear_target(self):
        """Clear the target template."""
        self.target_cnn_feature = None
        self.target_color_hist = None
        self.feature_history.clear()
