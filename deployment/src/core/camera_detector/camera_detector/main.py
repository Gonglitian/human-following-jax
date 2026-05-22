"""ROS2 node: camera-based person detector with dual depth mode.

Detects persons via YOLO11-Pose (keypoints + bbox), estimates their 3D
position in the odom frame, and publishes a PoseArray on /dr_spaam_detections
as a drop-in replacement for the LiDAR-based DR-SPAAM detector.

Depth modes (set via `depth_mode` parameter):
  "hardware"  — Read depth from Astra Pro /camera/depth/image_raw (0 GPU cost)
  "neural"    — Run Depth Anything V2 on RGB image (37ms GPU cost)
"""

from __future__ import annotations

import json
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from std_msgs.msg import String

import tf2_ros
from tf2_ros import Buffer, TransformListener

from camera_detector.person_detector import PersonDetector


class CameraDetectorNode(Node):
    """Person detection + depth → 3D odom-frame PoseArray.

    Supports two depth backends:
      - ``hardware``: reads Astra depth topic (zero GPU, ±1-3mm accuracy)
      - ``neural``: runs Depth Anything V2 (GPU, ~38ms, works without depth camera)

    The YOLO11-Pose detector provides 17 keypoints per person. Depth is sampled
    at torso keypoints (shoulders/hips) for robust distance estimation even
    under partial occlusion.
    """

    def __init__(self) -> None:
        super().__init__('camera_detector')

        # ---- parameters ----
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('depth_image_topic', '/camera/depth/image_raw')
        self.declare_parameter('detection_topic', '/dr_spaam_detections')
        self.declare_parameter('detection_image_topic', '/camera/detections_image')
        self.declare_parameter('person_crops_topic', '/camera/person_crops')
        self.declare_parameter('yolo_model', 'yolo11n-pose.pt')
        self.declare_parameter('yolo_conf_threshold', 0.5)
        self.declare_parameter('depth_mode', 'hardware')  # "hardware" or "neural"
        self.declare_parameter('depth_scale', 0.001)  # hardware depth unit→meters (Astra: mm→m)
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('target_frame', 'odom')

        camera_topic = self.get_parameter('camera_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.depth_image_topic = self.get_parameter('depth_image_topic').value
        detection_topic = self.get_parameter('detection_topic').value
        detection_image_topic = self.get_parameter('detection_image_topic').value
        person_crops_topic = self.get_parameter('person_crops_topic').value
        yolo_model = self.get_parameter('yolo_model').value
        yolo_conf = float(self.get_parameter('yolo_conf_threshold').value)
        self.depth_mode = self.get_parameter('depth_mode').value
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.camera_frame = self.get_parameter('camera_frame').value
        self.target_frame = self.get_parameter('target_frame').value

        # ---- camera intrinsics ----
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None

        # ---- hardware depth image (updated asynchronously) ----
        self.hw_depth_map: Optional[np.ndarray] = None

        # ---- TF2 ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- cv_bridge ----
        self.bridge = CvBridge()

        # ---- load YOLO11-Pose ----
        self.get_logger().info(f'Loading YOLO11-Pose: {yolo_model}')
        self.person_detector = PersonDetector(
            model_name=yolo_model,
            conf_threshold=yolo_conf,
            device='cuda',
        )
        self.get_logger().info('YOLO11-Pose loaded.')

        # ---- load depth backend ----
        self.depth_estimator = None
        if self.depth_mode == 'neural':
            self.get_logger().info('Depth mode: NEURAL (Depth Anything V2)')
            from camera_detector.depth_estimator import DepthEstimator
            self.depth_estimator = DepthEstimator(device='cuda')
            self.get_logger().info('Depth Anything V2 loaded.')
        elif self.depth_mode == 'hardware':
            self.get_logger().info(f'Depth mode: HARDWARE (subscribing to {self.depth_image_topic})')
            self.create_subscription(
                Image, self.depth_image_topic, self._hw_depth_callback, 10)
        else:
            raise ValueError(f'Unknown depth_mode: {self.depth_mode}. Use "hardware" or "neural".')

        # ---- publishers ----
        self.pub_detections = self.create_publisher(PoseArray, detection_topic, 10)
        self.pub_image = self.create_publisher(Image, detection_image_topic, 10)
        self.pub_crops = self.create_publisher(String, person_crops_topic, 10)

        # ---- subscribers ----
        self.create_subscription(CameraInfo, camera_info_topic, self._camera_info_callback, 10)
        self.create_subscription(Image, camera_topic, self._image_callback, 10)

        # ---- FPS tracking ----
        self._frame_count = 0
        self._fps_log_interval = 30
        self._t_last_fps = time.monotonic()

        self.get_logger().info(f'CameraDetectorNode ready. depth_mode={self.depth_mode}')

    # ================================================================ callbacks

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def _hw_depth_callback(self, msg: Image) -> None:
        """Store latest hardware depth image (Astra Pro publishes 16-bit mm)."""
        try:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.hw_depth_map = raw.astype(np.float32) * self.depth_scale  # → meters
        except Exception as exc:
            self.get_logger().error(f'Depth image conversion failed: {exc}')

    def _image_callback(self, msg: Image) -> None:
        """Main pipeline: detect → depth → project → publish."""
        if self.fx is None:
            self.get_logger().warn(
                'Camera info not received yet, skipping.', throttle_duration_sec=5.0)
            return

        # ---- convert to BGR ----
        try:
            image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'cv_bridge failed: {exc}')
            return

        # ---- TF lookup ----
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, self.camera_frame,
                Time(), timeout=rclpy.duration.Duration(seconds=0.05))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn(
                f'TF not available: {exc}', throttle_duration_sec=5.0)
            return

        # ---- get depth map ----
        depth_map = self._get_depth_map(image_bgr)
        if depth_map is None:
            self.get_logger().warn(
                'No depth data available, skipping.', throttle_duration_sec=2.0)
            return

        # ---- detect with keypoints ----
        try:
            detections, kps_list = self.person_detector.detect_with_keypoints(image_bgr)
        except Exception as exc:
            self.get_logger().error(f'YOLO inference failed: {exc}')
            return

        # ---- build outputs ----
        pose_array = PoseArray()
        pose_array.header.stamp = msg.header.stamp
        pose_array.header.frame_id = self.target_frame

        detection_records = []
        annotated = image_bgr.copy()

        for idx, ((x1, y1, x2, y2, conf), kps) in enumerate(zip(detections, kps_list)):
            # Sample depth at keypoints (robust to occlusion)
            Z = PersonDetector.get_depth_at_keypoints(
                depth_map, kps['xy'], kps['conf'])

            # Fallback: bbox center depth
            if Z <= 0.0:
                u, v = (x1 + x2) // 2, (y1 + y2) // 2
                h, w = depth_map.shape
                if 0 <= u < w and 0 <= v < h:
                    Z = float(depth_map[v, u])

            if Z <= 0.0:
                continue

            # Back-project using anchor center (more robust than bbox center)
            acx, acy = kps['anchor_center']
            if acx > 0 and acy > 0:
                u_proj, v_proj = acx, acy
            else:
                u_proj = float((x1 + x2) / 2)
                v_proj = float((y1 + y2) / 2)

            X_cam = (u_proj - self.cx) * Z / self.fx
            Y_cam = (v_proj - self.cy) * Z / self.fy

            x_odom, y_odom, z_odom = _transform_point(X_cam, Y_cam, Z, transform)

            pose = Pose()
            pose.position = Point(x=x_odom, y=y_odom, z=z_odom)
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            pose_array.poses.append(pose)

            n_vis = kps['n_visible']
            detection_records.append({
                'id': idx,
                'bbox': [x1, y1, x2, y2],
                'position': {'x': round(x_odom, 4), 'y': round(y_odom, 4)},
                'confidence': round(conf, 4),
                'n_keypoints': n_vis,
                'depth_m': round(Z, 3),
            })

            # annotate
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f'{conf:.2f} Z={Z:.1f}m [{n_vis}/17]'
            cv2.putText(annotated, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            # Draw anchor center
            if acx > 0:
                cv2.drawMarker(annotated, (int(acx), int(acy)), color,
                               cv2.MARKER_DIAMOND, 10, 2)

        # ---- publish ----
        self.pub_detections.publish(pose_array)

        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            img_msg.header = msg.header
            self.pub_image.publish(img_msg)
        except Exception as exc:
            self.get_logger().error(f'Failed to publish image: {exc}')

        stamp = msg.header.stamp
        self.pub_crops.publish(String(data=json.dumps({
            'header': {
                'frame_id': self.target_frame,
                'stamp_sec': stamp.sec,
                'stamp_nsec': stamp.nanosec,
            },
            'detections': detection_records,
        })))

        # ---- FPS ----
        self._frame_count += 1
        if self._frame_count % self._fps_log_interval == 0:
            now = time.monotonic()
            elapsed = now - self._t_last_fps
            fps = self._fps_log_interval / elapsed if elapsed > 0.0 else 0.0
            self._t_last_fps = now
            self.get_logger().info(
                f'FPS: {fps:.1f} | persons: {len(detections)} | '
                f'depth_mode: {self.depth_mode}')

    # ================================================================ helpers

    def _get_depth_map(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Get depth map from the configured backend."""
        if self.depth_mode == 'hardware':
            if self.hw_depth_map is None:
                return None
            depth = self.hw_depth_map
            # Resize if resolution differs from RGB
            h_rgb, w_rgb = image_bgr.shape[:2]
            h_d, w_d = depth.shape[:2]
            if (h_d, w_d) != (h_rgb, w_rgb):
                depth = cv2.resize(depth, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)
            return depth
        else:  # neural
            return self.depth_estimator.estimate(image_bgr)


def _transform_point(
    x_cam: float, y_cam: float, z_cam: float, transform,
) -> Tuple[float, float, float]:
    """Apply TF2 TransformStamped to a camera-frame point."""
    t = transform.transform.translation
    q = transform.transform.rotation
    qw, qx, qy, qz = q.w, q.x, q.y, q.z
    px, py, pz = x_cam, y_cam, z_cam
    tx = 2.0 * (qy * pz - qz * py)
    ty = 2.0 * (qz * px - qx * pz)
    tz = 2.0 * (qx * py - qy * px)
    rx = px + qw * tx + (qy * tz - qz * ty)
    ry = py + qw * ty + (qz * tx - qx * tz)
    rz = pz + qw * tz + (qx * ty - qy * tx)
    return rx + t.x, ry + t.y, rz + t.z


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
