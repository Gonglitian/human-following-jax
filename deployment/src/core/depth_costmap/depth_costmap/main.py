#!/usr/bin/env python3
"""
depth_costmap/main.py

Generates a robot-centered 50x50 occupancy grid (10 m x 10 m, 0.2 m/cell)
from monocular depth estimation (Depth Anything V2 metric indoor model).

Publishes on the same topics and in the exact same formats as the
occupancy_generation LiDAR node:
  /occupancy_grid      — nav_msgs/OccupancyGrid  (-1 unknown, 0 free, 100 occ)
  /occupancy_grid_json — std_msgs/String  (JSON with 0/1 binary data)
"""

import json
import math
from typing import Optional

import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.time import Time as RosTime

from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_depth_model(device: torch.device) -> torch.nn.Module:
    """Load Depth Anything V2 ViT-S metric indoor model from torch.hub."""
    model = torch.hub.load(
        'DepthAnything/Depth-Anything-V2',
        'depth_anything_v2_vits14_metric_indoor',
        trust_repo=True,
    )
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class DepthCostmapNode(Node):
    """
    Subscribes to a monocular RGB camera, estimates metric depth with
    Depth Anything V2, then projects the depth map to a bird's-eye-view
    occupancy grid aligned with the odom frame.

    Output format is identical to the occupancy_generation LiDAR node.
    """

    def __init__(self) -> None:
        super().__init__('depth_costmap')

        # ------------------------------------------------------------------ #
        # Parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('occupancy_grid_topic', '/occupancy_grid')
        self.declare_parameter('occupancy_grid_json_topic', '/occupancy_grid_json')
        self.declare_parameter('depth_model', 'depth-anything-v2-small')
        self.declare_parameter('grid_size', 10.0)
        self.declare_parameter('resolution', 0.2)
        self.declare_parameter('min_obstacle_height', 0.10)
        self.declare_parameter('max_obstacle_height', 1.50)
        self.declare_parameter('max_depth', 8.0)
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('target_frame', 'odom')

        camera_topic           = self.get_parameter('camera_topic').value
        camera_info_topic      = self.get_parameter('camera_info_topic').value
        occupancy_grid_topic   = self.get_parameter('occupancy_grid_topic').value
        occupancy_grid_json_topic = self.get_parameter('occupancy_grid_json_topic').value
        self.grid_size         = float(self.get_parameter('grid_size').value)
        self.resolution        = float(self.get_parameter('resolution').value)
        self.min_obs_height    = float(self.get_parameter('min_obstacle_height').value)
        self.max_obs_height    = float(self.get_parameter('max_obstacle_height').value)
        self.max_depth         = float(self.get_parameter('max_depth').value)
        self.camera_frame      = self.get_parameter('camera_frame').value
        self.target_frame      = self.get_parameter('target_frame').value

        self.width  = int(self.grid_size / self.resolution)   # 50
        self.height = int(self.grid_size / self.resolution)   # 50

        # ------------------------------------------------------------------ #
        # Depth model
        # ------------------------------------------------------------------ #
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Loading Depth Anything V2 on {self.device} …')
        try:
            self.depth_model = _load_depth_model(self.device)
            self.get_logger().info('Depth Anything V2 loaded successfully.')
        except Exception as exc:
            self.get_logger().error(f'Failed to load depth model: {exc}')
            raise

        # ------------------------------------------------------------------ #
        # Camera intrinsics (filled when CameraInfo arrives)
        # ------------------------------------------------------------------ #
        self.camera_info: Optional[CameraInfo] = None
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None
        self.img_width: Optional[int] = None
        self.img_height: Optional[int] = None

        # ------------------------------------------------------------------ #
        # TF
        # ------------------------------------------------------------------ #
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ------------------------------------------------------------------ #
        # CV Bridge
        # ------------------------------------------------------------------ #
        self.bridge = CvBridge()

        # ------------------------------------------------------------------ #
        # Subscriptions
        # ------------------------------------------------------------------ #
        self.create_subscription(CameraInfo, camera_info_topic,
                                 self._camera_info_callback, 10)
        self.create_subscription(Image, camera_topic,
                                 self._image_callback, 10)

        # ------------------------------------------------------------------ #
        # Publishers
        # ------------------------------------------------------------------ #
        self.occ_pub = self.create_publisher(
            OccupancyGrid, occupancy_grid_topic, 10)
        self.occ_json_pub = self.create_publisher(
            String, occupancy_grid_json_topic, 10)

        self.get_logger().info(
            f'DepthCostmapNode started. Grid: {self.width}x{self.height} '
            f'cells @ {self.resolution} m/cell ({self.grid_size} m x {self.grid_size} m).'
        )

    # ---------------------------------------------------------------------- #
    # Callbacks
    # ---------------------------------------------------------------------- #

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        """Cache intrinsics from the first CameraInfo message."""
        if self.camera_info is not None:
            return  # already stored
        self.camera_info = msg
        self.fx         = msg.k[0]   # K[0,0]
        self.fy         = msg.k[4]   # K[1,1]
        self.cx         = msg.k[2]   # K[0,2]
        self.cy         = msg.k[5]   # K[1,2]
        self.img_width  = msg.width
        self.img_height = msg.height
        self.get_logger().info(
            f'Camera intrinsics cached: fx={self.fx:.2f} fy={self.fy:.2f} '
            f'cx={self.cx:.2f} cy={self.cy:.2f} '
            f'({self.img_width}x{self.img_height})'
        )

    def _image_callback(self, msg: Image) -> None:
        """Main processing pipeline triggered on each incoming image."""
        # ---- Guard: need intrinsics before processing ---- #
        if self.camera_info is None:
            self.get_logger().warn(
                'No CameraInfo received yet — skipping frame.', throttle_duration_sec=5.0)
            return

        # ---- 1. ROS Image → OpenCV BGR ---- #
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'cv_bridge conversion failed: {exc}')
            return

        # ---- 2. Depth estimation ---- #
        depth_map = self._run_depth_model(bgr)  # (H, W) float32, metres

        # ---- 3. TF: camera_frame → target_frame (odom) ---- #
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                RosTime(),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f'TF lookup {self.camera_frame}→{self.target_frame} failed: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        # Robot position in odom frame (translation of camera frame ≈ robot pos)
        robot_x = tf_stamped.transform.translation.x
        robot_y = tf_stamped.transform.translation.y

        # Build 4x4 homogeneous transform matrix
        t  = tf_stamped.transform.translation
        q  = tf_stamped.transform.rotation
        tf_mat = self._quat_to_mat(q.x, q.y, q.z, q.w,
                                   t.x, t.y, t.z)  # (4, 4)

        # ---- 4–7. Project depth → grid ---- #
        data_binary, data_ros = self._depth_to_grid(
            depth_map, tf_mat, robot_x, robot_y)

        # ---- 8. Publish ---- #
        now = self.get_clock().now().to_msg()
        origin_x = robot_x - self.grid_size / 2.0
        origin_y = robot_y - self.grid_size / 2.0

        self._publish_occupancy_grid(
            now, origin_x, origin_y, data_ros)
        self._publish_json(
            now, origin_x, origin_y, data_binary)

        self.get_logger().info(
            f'Published occupancy grid at robot ({robot_x:.2f}, {robot_y:.2f}).')

    # ---------------------------------------------------------------------- #
    # Depth inference
    # ---------------------------------------------------------------------- #

    def _run_depth_model(self, bgr: np.ndarray) -> np.ndarray:
        """
        Run Depth Anything V2 on a BGR OpenCV image.

        Returns:
            depth_map: float32 array (H, W) in metres.
        """
        # Convert BGR → RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Depth Anything V2 hub model accepts a numpy RGB image and handles
        # its own preprocessing internally via infer_image().
        with torch.no_grad():
            depth = self.depth_model.infer_image(rgb)  # (H, W) numpy float32
        return depth.astype(np.float32)

    # ---------------------------------------------------------------------- #
    # Grid projection
    # ---------------------------------------------------------------------- #

    def _depth_to_grid(
        self,
        depth_map: np.ndarray,
        tf_mat: np.ndarray,
        robot_x: float,
        robot_y: float,
    ):
        """
        Project the depth map into a 2-D bird's-eye occupancy grid.

        Steps:
          a) Subsample depth map (stride=4) for speed.
          b) Backproject pixels → 3-D points in camera frame.
          c) Transform to odom frame via tf_mat.
          d) Height-filter to keep obstacle-range points.
          e) Project to grid and mark cells as occupied.
          f) Mark cells outside camera FOV as unknown (-1).

        Returns:
            data_binary : list[int] length W*H, values {0, 1}    (JSON format)
            data_ros    : list[int] length W*H, values {-1, 0, 100} (ROS format)
        """
        H, W = depth_map.shape
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        # ---- (a) Subsample ---- #
        stride = 4
        depth_sub = depth_map[::stride, ::stride]  # (H//stride, W//stride)
        hs, ws = depth_sub.shape

        # Pixel coordinate grids for the subsampled image
        u_idx = np.arange(ws, dtype=np.float32) * stride   # original u coords
        v_idx = np.arange(hs, dtype=np.float32) * stride   # original v coords
        uu, vv = np.meshgrid(u_idx, v_idx)                 # (hs, ws) each

        d = depth_sub.ravel()          # (N,)
        uu = uu.ravel()
        vv = vv.ravel()

        # ---- (b) Backproject to camera frame ---- #
        # Standard pinhole: X_cam right, Y_cam down, Z_cam forward
        X_cam = (uu - cx) * d / fx
        Y_cam = (vv - cy) * d / fy
        Z_cam = d

        # Stack as homogeneous (4, N)
        ones = np.ones_like(d)
        pts_cam = np.stack([X_cam, Y_cam, Z_cam, ones], axis=0)  # (4, N)

        # ---- (c) Transform to odom frame ---- #
        pts_odom = tf_mat @ pts_cam  # (4, N)
        X_odom = pts_odom[0]
        Y_odom = pts_odom[1]
        Z_odom = pts_odom[2]        # height above odom ground plane

        # ---- (d) Filter by depth and height ---- #
        valid_depth  = (d > 0.0) & (d <= self.max_depth)
        valid_height = (Z_odom >= self.min_obs_height) & (Z_odom <= self.max_obs_height)
        valid = valid_depth & valid_height

        X_occ = X_odom[valid]
        Y_occ = Y_odom[valid]

        # ---- (e) Project to grid ---- #
        origin_x = robot_x - self.grid_size / 2.0
        origin_y = robot_y - self.grid_size / 2.0

        # Initialise grids
        # data_binary: 0=free, 1=occupied  (no explicit unknown; FOV mask applied later)
        # data_ros   : -1=unknown, 0=free, 100=occupied
        grid_binary = np.zeros((self.height, self.width), dtype=np.int8)
        grid_ros    = np.full((self.height, self.width), -1, dtype=np.int8)

        # Compute grid indices for occupied points (vectorised)
        col = ((X_occ - origin_x) / self.resolution).astype(np.int32)
        row = ((Y_occ - origin_y) / self.resolution).astype(np.int32)

        in_bounds = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        col = col[in_bounds]
        row = row[in_bounds]

        grid_binary[row, col] = 1
        grid_ros[row, col]    = 100

        # Mark free-space along depth rays (ray-casting on subsampled points,
        # only for valid-depth rays regardless of height filter)
        # Use the same subsample grid. For each ray that has a valid depth
        # reading we walk from robot to the end-point and mark intermediate
        # cells as free.
        self._raycast_free_space(
            d, X_odom, Y_odom, valid_depth,
            robot_x, robot_y,
            origin_x, origin_y,
            grid_binary, grid_ros,
        )

        # ---- (f) FOV masking ---- #
        self._apply_fov_mask(
            fx, W,
            tf_mat,
            robot_x, robot_y,
            origin_x, origin_y,
            grid_binary, grid_ros,
        )

        # Flatten to row-major lists
        data_binary = grid_binary.flatten().tolist()
        data_ros    = grid_ros.flatten().tolist()

        return data_binary, data_ros

    # ---------------------------------------------------------------------- #
    # Ray-casting: free-space marking
    # ---------------------------------------------------------------------- #

    def _raycast_free_space(
        self,
        d: np.ndarray,
        X_odom: np.ndarray,
        Y_odom: np.ndarray,
        valid_depth: np.ndarray,
        robot_x: float,
        robot_y: float,
        origin_x: float,
        origin_y: float,
        grid_binary: np.ndarray,
        grid_ros: np.ndarray,
    ) -> None:
        """
        For each subsampled ray with a valid depth reading, walk from the
        robot to the measured end-point and mark intermediate grid cells
        as free space (unless already marked occupied).

        Uses integer Bresenham-style stepping in grid space for efficiency.
        """
        robot_col = int((robot_x - origin_x) / self.resolution)
        robot_row = int((robot_y - origin_y) / self.resolution)

        # Work only on rays with valid depth readings
        X_valid = X_odom[valid_depth]
        Y_valid = Y_odom[valid_depth]

        end_col = ((X_valid - origin_x) / self.resolution).astype(np.int32)
        end_row = ((Y_valid - origin_y) / self.resolution).astype(np.int32)

        # For each ray, take several steps from robot toward end-point
        # and mark free. We use fractional stepping (vectorised per ray is
        # not straightforward, so we use a coarse step count).
        n_rays = len(end_col)
        for i in range(n_rays):
            ec, er = end_col[i], end_row[i]
            dc = ec - robot_col
            dr = er - robot_row
            steps = max(abs(dc), abs(dr), 1)
            # Walk from robot (exclusive) to end-point (exclusive)
            for step in range(1, steps):
                t = step / steps
                c = int(robot_col + t * dc)
                r = int(robot_row + t * dr)
                if 0 <= c < self.width and 0 <= r < self.height:
                    # Only mark as free if NOT occupied
                    if grid_binary[r, c] == 0:
                        grid_ros[r, c] = 0

    # ---------------------------------------------------------------------- #
    # FOV masking
    # ---------------------------------------------------------------------- #

    def _apply_fov_mask(
        self,
        fx: float,
        img_width: int,
        tf_mat: np.ndarray,
        robot_x: float,
        robot_y: float,
        origin_x: float,
        origin_y: float,
        grid_binary: np.ndarray,
        grid_ros: np.ndarray,
    ) -> None:
        """
        Mark grid cells that lie OUTSIDE the camera's horizontal FOV cone
        as unknown (-1 in ROS grid, 0 in binary grid).

        The camera FOV cone is computed from the intrinsics, rotated into
        the odom frame, and applied as a half-space test on each cell centre.

        Cells inside the FOV that are still -1 (not yet assigned) keep their
        -1 value; this function only sets cells OUTSIDE the cone to -1 and
        leaves cells inside the cone as-is (they may be free=0 or occ=100).
        """
        half_fov = math.atan2(img_width / 2.0, fx)  # radians

        # Camera optical-axis direction in camera frame: (0, 0, 1, 0)
        # Left boundary ray in camera frame: (-tan(half_fov), 0, 1, 0)
        # Right boundary ray: (+tan(half_fov), 0, 1, 0)
        tan_hfov = math.tan(half_fov)

        def _ray_odom(dx_cam: float) -> tuple:
            """Project a camera-frame XZ ray direction into odom XY."""
            r_cam = np.array([dx_cam, 0.0, 1.0, 0.0], dtype=np.float64)
            r_odom = tf_mat @ r_cam
            return float(r_odom[0]), float(r_odom[1])

        # Optical axis and boundary rays in odom XY
        ax_ox, ax_oy   = _ray_odom(0.0)
        left_ox, left_oy  = _ray_odom(-tan_hfov)
        right_ox, right_oy = _ray_odom(+tan_hfov)

        # For each grid cell determine if it is inside the FOV cone.
        # The cone is defined by two half-planes whose normals are the
        # perpendiculars to the boundary rays.
        # A point P is inside iff:
        #   cross(left_ray, P - robot) >= 0   (left boundary)
        #   cross(right_ray, P - robot) <= 0  (right boundary)
        # where cross(a, b) = a.x*b.y - a.y*b.x  (2-D)

        # Build cell-centre coordinate arrays (vectorised)
        col_idx = np.arange(self.width,  dtype=np.float64)
        row_idx = np.arange(self.height, dtype=np.float64)
        cc, rr  = np.meshgrid(col_idx, row_idx)  # (H, W)

        cell_x = origin_x + (cc + 0.5) * self.resolution  # odom X
        cell_y = origin_y + (rr + 0.5) * self.resolution  # odom Y

        dx = cell_x - robot_x
        dy = cell_y - robot_y

        # Cross products with boundary rays
        cross_left  = left_ox  * dy - left_oy  * dx   # >=0 means left of left boundary (inside)
        cross_right = right_ox * dy - right_oy * dx   # <=0 means right of right boundary (inside)

        # Also require the cell to be in front of the robot (dot with optical axis > 0)
        dot_ax = ax_ox * dx + ax_oy * dy

        inside_fov = (cross_left >= 0) & (cross_right <= 0) & (dot_ax > 0)
        outside_fov = ~inside_fov

        # Cells outside FOV become unknown
        # (even if a stray occupied point landed there, we reset it)
        grid_binary[outside_fov] = 0
        grid_ros[outside_fov]    = -1

    # ---------------------------------------------------------------------- #
    # Publishing helpers
    # ---------------------------------------------------------------------- #

    def _publish_occupancy_grid(
        self,
        stamp,
        origin_x: float,
        origin_y: float,
        data_ros: list,
    ) -> None:
        """Publish nav_msgs/OccupancyGrid."""
        occ_msg = OccupancyGrid()
        occ_msg.header.stamp    = stamp
        occ_msg.header.frame_id = self.target_frame

        occ_msg.info.resolution = self.resolution
        occ_msg.info.width      = self.width
        occ_msg.info.height     = self.height

        occ_origin = Pose()
        occ_origin.position.x = origin_x
        occ_origin.position.y = origin_y
        occ_origin.position.z = 0.0
        occ_msg.info.origin = occ_origin

        occ_msg.data = data_ros
        self.occ_pub.publish(occ_msg)

    def _publish_json(
        self,
        stamp,
        origin_x: float,
        origin_y: float,
        data_binary: list,
    ) -> None:
        """
        Publish std_msgs/String with JSON payload.

        JSON schema (matches occupancy_generation exactly):
        {
          "header": {"frame_id": "odom", "stamp_sec": int, "stamp_nsec": int},
          "info": {
            "resolution": 0.2,
            "width": 50,
            "height": 50,
            "origin": {"x": float, "y": float, "z": 0.0}
          },
          "data": [0, 1, 0, ...]   // 2500 ints, 0=free, 1=occupied
        }
        """
        grid_dict = {
            "header": {
                "frame_id": self.target_frame,
                "stamp_sec":  stamp.sec,
                "stamp_nsec": stamp.nanosec,
            },
            "info": {
                "resolution": self.resolution,
                "width":      self.width,
                "height":     self.height,
                "origin": {
                    "x": origin_x,
                    "y": origin_y,
                    "z": 0.0,
                },
            },
            "data": data_binary,
        }

        json_msg      = String()
        json_msg.data = json.dumps(grid_dict, ensure_ascii=False)
        self.occ_json_pub.publish(json_msg)

    # ---------------------------------------------------------------------- #
    # Utility
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _quat_to_mat(
        qx: float, qy: float, qz: float, qw: float,
        tx: float, ty: float, tz: float,
    ) -> np.ndarray:
        """Build a 4x4 homogeneous transform from quaternion + translation."""
        # Rotation matrix from quaternion
        x2, y2, z2 = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz

        mat = np.array([
            [1 - 2*(y2 + z2),   2*(xy - wz),     2*(xz + wy),   tx],
            [2*(xy + wz),        1 - 2*(x2 + z2), 2*(yz - wx),   ty],
            [2*(xz - wy),        2*(yz + wx),      1-2*(x2+y2),  tz],
            [0.0,                0.0,              0.0,           1.0],
        ], dtype=np.float64)
        return mat


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = DepthCostmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
