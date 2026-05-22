#!/usr/bin/env python3
"""
Target Tracker Node: Manual target selection + appearance-based Re-ID.

Replaces UWB-based target identification with camera-based Re-ID.
Subscribes to camera image + SORT tracked objects + detection bboxes.
Publishes target person position as geometry_msgs/Point (drop-in for UWB).

Target selection commands (via /command topic):
  - "target:nearest"       → auto-select nearest detected person
  - "target:select:<id>"   → lock onto SORT track with given ID
  - "target:reset"         → clear target, stop publishing
"""

import json
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge

from .reid_feature import ReIDFeatureExtractor


class TargetTrackerNode(Node):

    def __init__(self):
        super().__init__('target_tracker_node')

        # Declare parameters
        self.declare_parameters(namespace='', parameters=[
            ('camera_topic', '/camera/image_raw'),
            ('person_crops_topic', '/camera/person_crops'),
            ('tracked_objects_topic', '/tracked_objects_json'),
            ('command_topic', '/command'),
            ('target_position_topic', '/camera/target_position'),
            ('target_viz_topic', '/camera/target_viz'),
            ('reid_similarity_threshold', 0.55),
            ('reid_feature_update_rate', 0.1),
            ('reid_history_size', 10),
            ('max_lost_frames', 30),
            ('spatial_fallback_radius', 1.5),
        ])

        # Read parameters
        self.similarity_threshold = self.get_parameter('reid_similarity_threshold').value
        self.max_lost_frames = self.get_parameter('max_lost_frames').value
        self.spatial_fallback_radius = self.get_parameter('spatial_fallback_radius').value
        update_rate = self.get_parameter('reid_feature_update_rate').value
        history_size = self.get_parameter('reid_history_size').value

        # Re-ID feature extractor
        device = 'cuda'
        try:
            import torch
            if not torch.cuda.is_available():
                device = 'cpu'
                self.get_logger().warn('[TargetTracker] CUDA not available, using CPU')
        except ImportError:
            device = 'cpu'
        self.reid = ReIDFeatureExtractor(
            device=device,
            history_size=history_size,
            update_rate=update_rate,
        )
        self.get_logger().info(f'[TargetTracker] Re-ID initialized on {device}')

        # CV bridge
        self.bridge = CvBridge()

        # State
        self.target_sort_id = None       # SORT track ID of target
        self.target_position = None      # (x, y) in odom frame
        self.last_target_position = None # last known valid position
        self.lost_count = 0
        self.latest_image = None
        self.latest_crops_data = None    # parsed JSON from camera_detector
        self.latest_tracks_data = None   # parsed JSON from sort_tracker
        self.selecting_nearest = False   # flag to auto-select on next frame
        self.selecting_id = None         # specific SORT ID to select

        # Subscribers
        camera_topic = self.get_parameter('camera_topic').value
        crops_topic = self.get_parameter('person_crops_topic').value
        tracks_topic = self.get_parameter('tracked_objects_topic').value
        command_topic = self.get_parameter('command_topic').value

        self.image_sub = self.create_subscription(
            Image, camera_topic, self.image_callback, 10)
        self.crops_sub = self.create_subscription(
            String, crops_topic, self.crops_callback, 10)
        self.tracks_sub = self.create_subscription(
            String, tracks_topic, self.tracks_callback, 10)
        self.command_sub = self.create_subscription(
            String, command_topic, self.command_callback, 10)

        # Publishers
        target_pos_topic = self.get_parameter('target_position_topic').value
        target_viz_topic = self.get_parameter('target_viz_topic').value

        self.target_pub = self.create_publisher(Point, target_pos_topic, 10)
        self.target_viz_pub = self.create_publisher(Marker, target_viz_topic, 10)

        self.get_logger().info('[TargetTracker] Node started. Waiting for target selection command.')

    # ========== Callbacks ==========

    def image_callback(self, msg: Image):
        """Store latest camera image for crop extraction."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'[TargetTracker] Image conversion error: {e}')

    def crops_callback(self, msg: String):
        """Store latest detection crop data from camera_detector."""
        try:
            self.latest_crops_data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'[TargetTracker] Crops JSON parse error: {e}')

    def tracks_callback(self, msg: String):
        """Main processing loop — triggered by new tracking data from SORT."""
        try:
            self.latest_tracks_data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'[TargetTracker] Tracks JSON parse error: {e}')
            return

        tracks = self.latest_tracks_data.get('tracks', [])
        if not tracks:
            self._handle_no_detections()
            return

        # Handle pending selection commands
        if self.selecting_nearest:
            self._select_nearest(tracks)
            self.selecting_nearest = False
            return

        if self.selecting_id is not None:
            self._select_by_id(tracks, self.selecting_id)
            self.selecting_id = None
            return

        # If no target set, do nothing
        if not self.reid.has_target():
            return

        # Match target across current detections using Re-ID
        self._match_and_publish(tracks)

    def command_callback(self, msg: String):
        """Handle target selection commands."""
        cmd = msg.data.strip()

        if cmd == 'target:nearest':
            self.selecting_nearest = True
            self.get_logger().info('[TargetTracker] Will select nearest person on next frame')

        elif cmd.startswith('target:select:'):
            try:
                sort_id = int(cmd.split(':')[2])
                self.selecting_id = sort_id
                self.get_logger().info(f'[TargetTracker] Will select SORT track ID {sort_id}')
            except (IndexError, ValueError):
                self.get_logger().warn(f'[TargetTracker] Invalid select command: {cmd}')

        elif cmd == 'target:reset':
            self._reset_target()
            self.get_logger().info('[TargetTracker] Target cleared')

    # ========== Target Selection ==========

    def _select_nearest(self, tracks):
        """Select the nearest detected person as target."""
        if not tracks:
            self.get_logger().warn('[TargetTracker] No tracks to select from')
            return

        # Find nearest track (smallest distance from robot origin in odom frame)
        best = None
        best_dist = float('inf')
        for t in tracks:
            x, y = t.get('x', 0), t.get('y', 0)
            dist = np.hypot(x, y)
            if dist < best_dist:
                best_dist = dist
                best = t

        if best is None:
            return

        sort_id = best.get('id', -1)
        self._lock_target(sort_id, best.get('x', 0), best.get('y', 0))

    def _select_by_id(self, tracks, target_id):
        """Select a specific SORT track ID as target."""
        for t in tracks:
            if t.get('id') == target_id:
                self._lock_target(target_id, t.get('x', 0), t.get('y', 0))
                return
        self.get_logger().warn(f'[TargetTracker] Track ID {target_id} not found in current tracks')

    def _lock_target(self, sort_id, x, y):
        """Lock onto a target: extract Re-ID features from their crop."""
        crop = self._get_crop_for_track(sort_id)
        if crop is not None and self.reid.set_target(crop):
            self.target_sort_id = sort_id
            self.target_position = (x, y)
            self.last_target_position = (x, y)
            self.lost_count = 0
            self.get_logger().info(
                f'[TargetTracker] LOCKED target: SORT ID={sort_id} at ({x:.2f}, {y:.2f})')
        else:
            # No crop available — lock by ID only, will extract features when crop appears
            self.target_sort_id = sort_id
            self.target_position = (x, y)
            self.last_target_position = (x, y)
            self.lost_count = 0
            self.get_logger().warn(
                f'[TargetTracker] Locked SORT ID={sort_id} WITHOUT Re-ID features (no crop)')

    def _get_crop_for_track(self, sort_id):
        """Extract image crop for a given SORT track ID using detection bboxes."""
        if self.latest_image is None or self.latest_crops_data is None:
            return None

        detections = self.latest_crops_data.get('detections', [])
        tracks = self.latest_tracks_data.get('tracks', []) if self.latest_tracks_data else []

        # Find the detection bbox closest to this track's odom position
        target_track = None
        for t in tracks:
            if t.get('id') == sort_id:
                target_track = t
                break

        if target_track is None:
            return None

        tx, ty = target_track.get('x', 0), target_track.get('y', 0)

        # Find closest detection (by odom position) to this track
        best_det = None
        best_dist = float('inf')
        for det in detections:
            pos = det.get('position', {})
            dx = pos.get('x', 0) - tx
            dy = pos.get('y', 0) - ty
            dist = np.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best_det = det

        if best_det is None or best_dist > 1.0:
            return None

        bbox = best_det.get('bbox', [])
        if len(bbox) != 4:
            return None

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = self.latest_image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return self.latest_image[y1:y2, x1:x2].copy()

    # ========== Matching ==========

    def _match_and_publish(self, tracks):
        """Match target against current detections and publish position."""
        best_match_id = None
        best_similarity = 0.0
        best_position = None

        for t in tracks:
            sort_id = t.get('id', -1)
            crop = self._get_crop_for_track(sort_id)
            if crop is None:
                continue

            sim = self.reid.match(crop)

            if sim > best_similarity:
                best_similarity = sim
                best_match_id = sort_id
                best_position = (t.get('x', 0), t.get('y', 0))

        if best_similarity >= self.similarity_threshold and best_position is not None:
            # Confirmed match
            self.target_sort_id = best_match_id
            self.target_position = best_position
            self.last_target_position = best_position
            self.lost_count = 0

            # Update Re-ID template with confirmed match
            crop = self._get_crop_for_track(best_match_id)
            if crop is not None:
                self.reid.update_template(crop)

            self._publish_target(best_position[0], best_position[1])
            self.get_logger().debug(
                f'[TargetTracker] MATCH: ID={best_match_id} sim={best_similarity:.3f} '
                f'at ({best_position[0]:.2f}, {best_position[1]:.2f})')
        else:
            # No confident Re-ID match — try spatial fallback
            self._spatial_fallback(tracks)

    def _spatial_fallback(self, tracks):
        """Fall back to spatial proximity when Re-ID matching fails."""
        self.lost_count += 1

        if self.last_target_position is None:
            self._handle_lost()
            return

        # Find closest track to last known position
        lx, ly = self.last_target_position
        best_track = None
        best_dist = float('inf')
        for t in tracks:
            tx, ty = t.get('x', 0), t.get('y', 0)
            dist = np.hypot(tx - lx, ty - ly)
            if dist < best_dist:
                best_dist = dist
                best_track = t

        if best_track is not None and best_dist < self.spatial_fallback_radius:
            # Accept spatial match
            x, y = best_track.get('x', 0), best_track.get('y', 0)
            self.target_position = (x, y)
            self.last_target_position = (x, y)
            self.target_sort_id = best_track.get('id', -1)
            self._publish_target(x, y)
            self.get_logger().debug(
                f'[TargetTracker] SPATIAL fallback: ID={self.target_sort_id} '
                f'dist={best_dist:.2f}m (lost {self.lost_count} frames)')

            # Try to re-initialize Re-ID features
            crop = self._get_crop_for_track(self.target_sort_id)
            if crop is not None and self.lost_count > 5:
                self.reid.set_target(crop)
                self.lost_count = 0
                self.get_logger().info('[TargetTracker] Re-initialized Re-ID template from spatial match')
        else:
            self._handle_lost()

    def _handle_lost(self):
        """Handle case when target is completely lost."""
        if self.lost_count > self.max_lost_frames:
            self.get_logger().warn(
                f'[TargetTracker] Target LOST for {self.lost_count} frames. '
                f'Still publishing last known position.')
        # Keep publishing last known position (decider handles staleness)
        if self.last_target_position is not None:
            self._publish_target(self.last_target_position[0], self.last_target_position[1])

    def _handle_no_detections(self):
        """Handle frame with zero detections."""
        if self.reid.has_target():
            self.lost_count += 1
            if self.last_target_position is not None:
                self._publish_target(self.last_target_position[0], self.last_target_position[1])

    # ========== Publishing ==========

    def _publish_target(self, x, y):
        """Publish target position as Point (same format as UWB)."""
        msg = Point()
        msg.x = float(x)
        msg.y = float(y)
        msg.z = 0.0
        self.target_pub.publish(msg)

        # Publish visualization marker
        marker = Marker()
        marker.header.frame_id = 'odom'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'target_person'
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.5
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 1.0
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
        marker.lifetime.sec = 1
        self.target_viz_pub.publish(marker)

    def _reset_target(self):
        """Reset all target tracking state."""
        self.reid.clear_target()
        self.target_sort_id = None
        self.target_position = None
        self.last_target_position = None
        self.lost_count = 0


def main(args=None):
    rclpy.init(args=args)
    node = TargetTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
