#!/usr/bin/env python3
"""
Occupancy Grid Generation with Static Map Overlay (方案A).

Merges two sources into a single 50×50 OGM for the RL policy:
  1. Real-time LiDAR scan → dynamic obstacles (people, moving objects)
  2. Pre-built static map  → walls, furniture (loaded from .pgm via SLAM)

The static map is loaded once at startup. Each frame, a robot-centered crop
is extracted and merged with the LiDAR OGM using OR logic:
  - Either source says occupied → merged cell = occupied

Requires AMCL (or similar) to provide the map→odom TF for static map alignment.
"""

import json
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, PoseArray
from tf2_ros import Buffer, TransformListener, TransformException
import tf_transformations
import yaml


class OccupancyGridNode(Node):
    """
    Builds a robot-centered 10m×10m occupancy grid (50×50, 0.2m/cell).
    Optionally overlays a pre-built static map for complete obstacle coverage.
    """

    def __init__(self):
        super().__init__('occupancy_grid_node')

        # ---- parameters ----
        self.declare_parameter('grid_size', 10.0)
        self.declare_parameter('resolution', 0.2)
        self.declare_parameter('use_static_map', False)
        self.declare_parameter('static_map_yaml', '')
        self.declare_parameter('human_filter_enabled', False)
        self.declare_parameter('human_filter_radius', 0.4)

        self.grid_size = self.get_parameter('grid_size').value
        self.resolution = self.get_parameter('resolution').value
        self.width = int(self.grid_size / self.resolution)
        self.height = int(self.grid_size / self.resolution)
        self.use_static_map = self.get_parameter('use_static_map').value
        self.human_filter_enabled = self.get_parameter('human_filter_enabled').value
        self.human_filter_radius = self.get_parameter('human_filter_radius').value

        # ---- static map (loaded once) ----
        self.static_map = None          # full-res numpy array (0=free, 1=occupied)
        self.static_map_resolution = 0.0
        self.static_map_origin_x = 0.0
        self.static_map_origin_y = 0.0
        self.static_map_h = 0
        self.static_map_w = 0

        if self.use_static_map:
            map_yaml = self.get_parameter('static_map_yaml').value
            if map_yaml and os.path.exists(map_yaml):
                self._load_static_map(map_yaml)
            else:
                self.get_logger().warn(
                    f'use_static_map=True but static_map_yaml not found: "{map_yaml}". '
                    'Will also try /map topic.')
                # Subscribe to /map as fallback (published by map_server)
                map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
                self.create_subscription(
                    OccupancyGrid, '/map', self._map_topic_callback, map_qos)

        # ---- subscribers ----
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        self.latest_human_poses = []
        self.human_sub = self.create_subscription(
            PoseArray, '/dr_spaam_detections', self.human_callback, 10)

        # ---- publishers ----
        self.occ_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)
        self.occ_json_pub = self.create_publisher(String, '/occupancy_grid_json', 10)

        # ---- TF ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        mode = "static_map + LiDAR" if self.use_static_map else "LiDAR only"
        self.get_logger().info(
            f'OccupancyGridNode started. mode={mode}, '
            f'grid={self.width}x{self.height} @ {self.resolution}m')

    # ================================================================
    # Static map loading
    # ================================================================

    def _load_static_map(self, yaml_path: str):
        """Load a static map from SLAM-generated .yaml + .pgm files."""
        self.get_logger().info(f'Loading static map: {yaml_path}')
        with open(yaml_path, 'r') as f:
            meta = yaml.safe_load(f)

        pgm_path = meta.get('image', '')
        if not os.path.isabs(pgm_path):
            pgm_path = os.path.join(os.path.dirname(yaml_path), pgm_path)

        if not os.path.exists(pgm_path):
            self.get_logger().error(f'Static map image not found: {pgm_path}')
            return

        self.static_map_resolution = float(meta.get('resolution', 0.05))
        origin = meta.get('origin', [0, 0, 0])
        self.static_map_origin_x = float(origin[0])
        self.static_map_origin_y = float(origin[1])
        occupied_thresh = float(meta.get('occupied_thresh', 0.65))
        free_thresh = float(meta.get('free_thresh', 0.25))
        negate = int(meta.get('negate', 0))

        # Load PGM (grayscale, 0-255)
        img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().error(f'Failed to read: {pgm_path}')
            return

        # ROS map convention: image row 0 = top of image = max Y in world
        # Flip vertically so row 0 = min Y
        img = np.flipud(img)

        # Convert to probability [0, 1]
        if negate:
            prob = img.astype(np.float32) / 255.0
        else:
            prob = (255 - img).astype(np.float32) / 255.0

        # Threshold to binary: 1=occupied, 0=free
        self.static_map = np.zeros_like(prob, dtype=np.int8)
        self.static_map[prob >= occupied_thresh] = 1
        # Cells between free_thresh and occupied_thresh → unknown, treat as free for OGM
        self.static_map_h, self.static_map_w = self.static_map.shape

        self.get_logger().info(
            f'Static map loaded: {self.static_map_w}x{self.static_map_h} '
            f'@ {self.static_map_resolution}m, '
            f'origin=({self.static_map_origin_x:.1f}, {self.static_map_origin_y:.1f}), '
            f'occupied cells: {int(self.static_map.sum())}')

    def _map_topic_callback(self, msg: OccupancyGrid):
        """Load static map from /map topic (published by nav2_map_server)."""
        if self.static_map is not None:
            return  # already loaded

        w = msg.info.width
        h = msg.info.height
        self.static_map_resolution = msg.info.resolution
        self.static_map_origin_x = msg.info.origin.position.x
        self.static_map_origin_y = msg.info.origin.position.y

        data = np.array(msg.data, dtype=np.int8).reshape((h, w))
        # ROS OccupancyGrid: -1=unknown, 0=free, 100=occupied
        self.static_map = np.zeros((h, w), dtype=np.int8)
        self.static_map[data >= 50] = 1  # occupied
        self.static_map_h, self.static_map_w = h, w

        self.get_logger().info(
            f'Static map from /map: {w}x{h} @ {self.static_map_resolution}m, '
            f'occupied: {int(self.static_map.sum())}')

    def _crop_static_map(self, robot_x_map: float, robot_y_map: float) -> np.ndarray:
        """Extract a robot-centered crop from the static map, resampled to OGM resolution.

        Returns a (self.height, self.width) int8 array with 0=free, 1=occupied.
        """
        crop = np.zeros((self.height, self.width), dtype=np.int8)
        if self.static_map is None:
            return crop

        # OGM origin in map frame
        ogm_origin_x = robot_x_map - self.grid_size / 2.0
        ogm_origin_y = robot_y_map - self.grid_size / 2.0

        # For each OGM cell, find the corresponding static map cell
        for row in range(self.height):
            for col in range(self.width):
                # World coordinate of this OGM cell center
                wx = ogm_origin_x + (col + 0.5) * self.resolution
                wy = ogm_origin_y + (row + 0.5) * self.resolution

                # Static map pixel coordinate
                sx = int((wx - self.static_map_origin_x) / self.static_map_resolution)
                sy = int((wy - self.static_map_origin_y) / self.static_map_resolution)

                if 0 <= sx < self.static_map_w and 0 <= sy < self.static_map_h:
                    if self.static_map[sy, sx] == 1:
                        crop[row, col] = 1

        return crop

    def _crop_static_map_fast(self, robot_x_map: float, robot_y_map: float) -> np.ndarray:
        """Vectorized version of static map cropping (much faster)."""
        crop = np.zeros((self.height, self.width), dtype=np.int8)
        if self.static_map is None:
            return crop

        ogm_origin_x = robot_x_map - self.grid_size / 2.0
        ogm_origin_y = robot_y_map - self.grid_size / 2.0

        # Build coordinate arrays for all OGM cells
        cols = np.arange(self.width, dtype=np.float32)
        rows = np.arange(self.height, dtype=np.float32)
        cc, rr = np.meshgrid(cols, rows)

        wx = ogm_origin_x + (cc + 0.5) * self.resolution
        wy = ogm_origin_y + (rr + 0.5) * self.resolution

        sx = ((wx - self.static_map_origin_x) / self.static_map_resolution).astype(np.int32)
        sy = ((wy - self.static_map_origin_y) / self.static_map_resolution).astype(np.int32)

        valid = (sx >= 0) & (sx < self.static_map_w) & (sy >= 0) & (sy < self.static_map_h)
        sx_v = np.clip(sx, 0, self.static_map_w - 1)
        sy_v = np.clip(sy, 0, self.static_map_h - 1)

        crop[valid] = self.static_map[sy_v[valid], sx_v[valid]]
        return crop

    # ================================================================
    # Human callback
    # ================================================================

    def human_callback(self, msg: PoseArray):
        self.latest_human_poses = [(p.position.x, p.position.y) for p in msg.poses]

    # ================================================================
    # Main scan callback
    # ================================================================

    def scan_callback(self, msg: LaserScan):
        scan = np.array(msg.ranges)
        scan[scan == 0.0] = 29.99
        scan[np.isinf(scan)] = 29.99
        scan[np.isnan(scan)] = 29.99

        # TF: laser → odom
        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                "odom", msg.header.frame_id, rclpy.time.Time())
            trans = transform_stamped.transform.translation
            rot = transform_stamped.transform.rotation
            tf_mat = tf_transformations.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
            tf_mat[0, 3] = trans.x
            tf_mat[1, 3] = trans.y
            tf_mat[2, 3] = trans.z
        except TransformException as ex:
            self.get_logger().warn(f"TF laser→odom failed: {ex}")
            return

        robot_x = trans.x
        robot_y = trans.y
        origin_x = robot_x - self.grid_size / 2.0
        origin_y = robot_y - self.grid_size / 2.0

        # ---- Build LiDAR OGM ----
        data_binary = [0] * (self.width * self.height)
        data_ros = [-1] * (self.width * self.height)

        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        min_len = min(len(angles), len(scan))
        angles = angles[:min_len]
        scan = scan[:min_len]

        for i in range(min_len):
            r = scan[i]
            angle = angles[i]
            lx = r * np.cos(angle)
            ly = r * np.sin(angle)

            point_laser = np.array([lx, ly, 0.0, 1.0], dtype=np.float32)
            point_odom = tf_mat @ point_laser
            ox, oy = point_odom[0], point_odom[1]

            # Human filtering (optional)
            if self.human_filter_enabled:
                is_human = False
                for hx, hy in self.latest_human_poses:
                    if np.hypot(ox - hx, oy - hy) < self.human_filter_radius:
                        is_human = True
                        break
                if is_human:
                    continue

            if (origin_x <= ox < origin_x + self.grid_size) and \
               (origin_y <= oy < origin_y + self.grid_size):
                grid_x = int((ox - origin_x) / self.resolution)
                grid_y = int((oy - origin_y) / self.resolution)
                if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                    idx = grid_y * self.width + grid_x
                    data_binary[idx] = 1
                    data_ros[idx] = 100

                    # Ray-trace free space
                    steps = int(r / self.resolution)
                    for step in range(steps):
                        t = step / float(steps) if steps > 0 else 0
                        ray_x = robot_x + t * (ox - robot_x)
                        ray_y = robot_y + t * (oy - robot_y)
                        if (origin_x <= ray_x < origin_x + self.grid_size) and \
                           (origin_y <= ray_y < origin_y + self.grid_size):
                            rgx = int((ray_x - origin_x) / self.resolution)
                            rgy = int((ray_y - origin_y) / self.resolution)
                            if 0 <= rgx < self.width and 0 <= rgy < self.height:
                                ridx = rgy * self.width + rgx
                                if data_binary[ridx] == 0:
                                    data_ros[ridx] = 0

        # ---- Merge with static map (方案A) ----
        lidar_grid = np.array(data_binary, dtype=np.int8).reshape((self.height, self.width))

        if self.use_static_map and self.static_map is not None:
            # Get robot position in map frame for static map cropping
            robot_x_map, robot_y_map = self._get_robot_in_map_frame()
            if robot_x_map is not None:
                static_crop = self._crop_static_map_fast(robot_x_map, robot_y_map)
                # OR merge: either source says occupied → occupied
                merged = np.maximum(lidar_grid, static_crop)
            else:
                merged = lidar_grid
        else:
            merged = lidar_grid

        # ---- Convert merged grid back to publish formats ----
        merged_binary = merged.flatten().tolist()
        merged_ros = [-1] * (self.width * self.height)
        for i, val in enumerate(merged_binary):
            if val == 1:
                merged_ros[i] = 100
            elif data_ros[i] == 0:
                merged_ros[i] = 0

        # ---- Publish OccupancyGrid (ROS standard) ----
        occ_msg = OccupancyGrid()
        occ_msg.header.stamp = self.get_clock().now().to_msg()
        occ_msg.header.frame_id = "odom"
        occ_msg.info.resolution = self.resolution
        occ_msg.info.width = self.width
        occ_msg.info.height = self.height
        occ_origin = Pose()
        occ_origin.position.x = origin_x
        occ_origin.position.y = origin_y
        occ_origin.position.z = 0.0
        occ_msg.info.origin = occ_origin
        occ_msg.data = merged_ros
        self.occ_pub.publish(occ_msg)

        # ---- Publish JSON (for RL policy) ----
        grid_dict = {
            "header": {
                "frame_id": "odom",
                "stamp_sec": occ_msg.header.stamp.sec,
                "stamp_nsec": occ_msg.header.stamp.nanosec
            },
            "info": {
                "resolution": self.resolution,
                "width": self.width,
                "height": self.height,
                "origin": {
                    "x": origin_x,
                    "y": origin_y,
                    "z": 0.0
                }
            },
            "data": merged_binary
        }
        json_msg = String()
        json_msg.data = json.dumps(grid_dict, ensure_ascii=False)
        self.occ_json_pub.publish(json_msg)

    # ================================================================
    # Helpers
    # ================================================================

    def _get_robot_in_map_frame(self):
        """Get robot position in the map frame (requires AMCL providing map→odom TF)."""
        try:
            tf = self.tf_buffer.lookup_transform("map", "base_footprint", rclpy.time.Time())
            return tf.transform.translation.x, tf.transform.translation.y
        except TransformException:
            # Fallback: try map→odom→base_link chain
            try:
                tf = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
                return tf.transform.translation.x, tf.transform.translation.y
            except TransformException:
                return None, None


def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
