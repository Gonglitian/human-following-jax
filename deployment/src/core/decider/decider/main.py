#!/usr/bin/env python3

import json
import math
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

import tf2_ros
import tf_transformations

import numpy as np
import os
import torch
import torch.nn as nn
import time
import gym

from rl.networks.model import Policy
from rl.networks.model_meta import PolicyMeta
from ament_index_python.packages import get_package_share_directory

import logging
from datetime import datetime

MANUAL_CMD_TIMEOUT = 0.2  # 200 ms override window
MAX_HUMANS = 50

# P Controller for Meta Policy
class PController:
    """
    Proportional feedback controller with adaptive adjustment.
    
    Maps continuous distance error to discrete preference values (-2, -1, 0, 1, 2).
    
    Preference → Distance: -2→1.37m, -1→1.90m, 0→2.29m, 1→3.31m, 2→3.80m
    """
    
    def __init__(self, kp=4.0, target_distance=2.0, tolerance=0.10, adaptive=True):
        self.kp = kp
        self.target_distance = target_distance
        self.tolerance = tolerance  # Small error threshold
        # adaptive=False reproduces paper's Meta-NoMap baseline: skip closed-loop
        # error correction and always map target_distance to nearest discrete preference.
        self.adaptive = adaptive

        self.preference_to_distance = {
            -2: 1.37,
            -1: 1.90,
             0: 2.29,
             1: 3.31,
             2: 3.80
        }
    
    def reset(self):
        pass
    
    def set_target_distance(self, distance):
        self.target_distance = distance
    
    def compute(self, current_distance, dt=0.25):
        """P control with tolerance check and adaptive adjustment."""
        # Meta-NoMap baseline: bypass closed-loop entirely.
        if not self.adaptive:
            return self._nearest_preference(self.target_distance)

        error = current_distance - self.target_distance
        abs_error = abs(error)

        # Small error within tolerance: choose preference matching target
        if abs_error < self.tolerance:
            return self._nearest_preference(self.target_distance)
        
        # Large error: aggressive P control
        # Ensure minimum adjustment to actually change preference
        adjustment = max(self.kp * abs_error, 0.3)
        
        if error > 0:  # Too far → need closer preference
            desired_distance = self.target_distance - adjustment
        else:  # Too close → need farther preference
            desired_distance = self.target_distance + adjustment
        
        return self._nearest_preference(desired_distance)
    
    def _nearest_preference(self, distance):
        """Find preference with closest target distance."""
        min_diff = float('inf')
        best_pref = 0
        for pref, pref_dist in self.preference_to_distance.items():
            diff = abs(distance - pref_dist)
            if diff < min_diff:
                min_diff = diff
                best_pref = pref
        return best_pref


class Decider(Node):
    def __init__(self):
        super().__init__('decider_node')

        # Meta-NoMap toggle: when false, P-controller uses direct mapping only
        # (paper baseline V.B "Meta-NoMap"). True = full closed-loop adaptive (Ours).
        self.declare_parameter('adaptive_mapping', True)
        self.adaptive_mapping = bool(self.get_parameter('adaptive_mapping').value)

        # Which ckpt to load from `share/decider/model_weight/`. Lets us swap
        # `meta_4.pt` (Ours) ↔ `meta_nomap.pt` (paper-trained Meta-NoMap)
        # without rebuild. Filename only — full path is resolved against share.
        self.declare_parameter('model_weight_file', 'meta_4.pt')
        self.model_weight_file = str(self.get_parameter('model_weight_file').value)

        # Target acquisition source.
        #   'uwb_camera'    — default; use UWB tag_0 + camera fallback (paper path)
        #   'closest_lidar' — pick nearest DR-SPAAM-tracked human (no UWB/camera needed)
        self.declare_parameter('target_source', 'uwb_camera')
        self.target_source = str(self.get_parameter('target_source').value)

        # Real-robot safety overrides (training defaults: 1.0 / 0.5 / 2.0)
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('max_delta_v', 0.5)
        self.declare_parameter('default_target_distance', 2.0)
        # APF safety repulsion (against OGM-occupied cells).
        #   safety_enabled    — toggle whole feature (default true)
        #   safety_radius     — cells within this distance (m) push the robot away (default 0.6 m)
        #   safety_strength   — caps |repulsion vector| (m/s) added to policy action (default 0.5)
        self.declare_parameter('safety_enabled', True)
        self.declare_parameter('safety_radius', 0.6)
        self.declare_parameter('safety_strength', 0.5)
        # Hard-cutoff: if min cell distance < safety_hard_dist AND velocity
        # has component toward closest obstacle, the inward component is zeroed
        # (policy command into wall is BLOCKED entirely, not just nudged).
        self.declare_parameter('safety_hard_dist', 0.35)
        self.declare_parameter('safety_hard_enabled', True)
        self.safety_enabled = bool(self.get_parameter('safety_enabled').value)
        self.safety_radius = float(self.get_parameter('safety_radius').value)
        self.safety_strength = float(self.get_parameter('safety_strength').value)
        self.safety_hard_dist = float(self.get_parameter('safety_hard_dist').value)
        self.safety_hard_enabled = bool(self.get_parameter('safety_hard_enabled').value)
        self._safety_log_count = 0  # rate-limit safety log spam

        # ============= 1. File Logging Setup =============
        self.log_dir = 'decider_logs'
        os.makedirs(self.log_dir, exist_ok=True)
        
        log_filename = datetime.now().strftime("decider_following_%Y%m%d-%H%M%S.log")
        log_path = os.path.join(self.log_dir, log_filename)
        
        self.file_logger = logging.getLogger("DeciderFileLog")
        self.file_logger.setLevel(logging.INFO)
        
        if not self.file_logger.handlers:
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
            fh.setFormatter(formatter)
            self.file_logger.addHandler(fh)
            
        self.get_logger().info(f"[Decider] File logging initialized: {log_path}")

        # ============= 2. Distance Statistics Variables =============
        self.is_logging_active = False
        self.stats_lidar_sum = 0.0
        self.stats_lidar_count = 0
        self.stats_uwb_sum = 0.0
        self.stats_uwb_count = 0

        # ============= Subscriptions =============
        self.command_sub_ = self.create_subscription(
            String, '/command', self.command_callback, 10
        )
        self.joint_state_sub_ = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.tracked_objects_sub_ = self.create_subscription(
            String, '/tracked_objects_json', self.tracked_objects_json_callback, 10
        )
        self.predictions_sub_ = self.create_subscription(
            String, '/predictions_json', self.predictions_callback, 10
        )
        self.odom_sub_ = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 20
        )
        self.occ_json_sub_ = self.create_subscription(
            String, '/occupancy_grid_json', self.occupancy_grid_json_callback, 10
        )

        # === UWB Target Tracking (Tag-to-Tag distance) ===
        # Tag 0 = 人, Tag 1 = 小车
        self.uwb_human_sub_ = self.create_subscription(
            Point, '/uwb/tag_0/position', self.uwb_human_callback, 10
        )
        self.uwb_robot_sub_ = self.create_subscription(
            Point, '/uwb/tag_1/position', self.uwb_robot_callback, 10
        )
        self.get_logger().info("[Decider] UWB Tag-to-Tag mode: tag_0=human, tag_1=robot")

        # === Camera Target Tracking (replaces UWB when enabled) ===
        self.use_camera_target = True  # Set to True to use camera-based target tracking
        self.camera_target_x = None
        self.camera_target_y = None
        self.camera_target_last_update = 0.0
        self.camera_target_sub_ = self.create_subscription(
            Point, '/camera/target_position', self.camera_target_callback, 10
        )
        if self.use_camera_target:
            self.get_logger().info("[Decider] Camera target mode ENABLED: subscribing to /camera/target_position")

        # ============= TF2 for Robot Pose =============
        self.tf_buffer_ = tf2_ros.Buffer()
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self)

        # ============= Publishers =============
        self.cmd_vel_pub_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.action_marker_pub_ = self.create_publisher(Marker, '/decider_action_marker', 10)
        self.goal_marker_pub_ = self.create_publisher(Marker, '/decider_goal_marker', 10)
        self.robot_marker_pub_ = self.create_publisher(Marker, '/decider_robot_marker', 10)
        # /decider_target: JSON {"id": int|null, "x": float, "y": float} in odom
        # frame. Published every tick so external metric loggers can compute
        # dist_to_target + min_human_dist (closest other-than-target) WITHOUT
        # needing ground truth. id=null = no current lock (COAST or no humans).
        self.target_pub_ = self.create_publisher(String, '/decider_target', 10)

        # ============= Mode and override logic =============
        self.current_mode_ = None
        self.in_override_ = False
        self.last_manual_cmd_time_ = self.get_clock().now()
        self.timer_ = self.create_timer(0.1, self.check_manual_timeout)

        # Fix #6: Fixed 4Hz RL timer — decouples RL from variable-rate tracker callback.
        # Training uses dt=0.25s (4Hz). Running RL at tracker rate (~4-15Hz) causes
        # trajectory prediction and OGM history to operate on mismatched time scales.
        self.rl_timer_ = self.create_timer(0.25, self.rl_timer_callback)
        self.pending_rl_obs = None  # latest observation built from tracker callback

        # ============= RL Setup =============
        from config.arguments import get_args
        self.algo_args = get_args()
        self.get_logger().info("[Decider] Successfully imported algo_args.")

        from config.config import Config
        self.config = Config()
        self.get_logger().info("[Decider] Successfully imported config.")

        self.predict_steps = 5
        self.ogm_history_len = 3  # Must be defined before set_ob_act_space()
        self.set_ob_act_space()

        # Subclasses (e.g. main_rlpc.OursRlpc) override `_create_actor_critic`
        # to swap network class / base name without re-implementing the rest
        # of __init__.
        self.actor_critic = self._create_actor_critic()

        decider_share_dir = get_package_share_directory('decider')
        load_path = os.path.join(decider_share_dir, 'model_weight', self.model_weight_file)
        self.get_logger().info(f"[Decider] Loading model_weight: {load_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _sd = torch.load(load_path, map_location=self.device)
        # Some legacy ckpts (meta_original.pt) embed optimizer state under
        # 'model_state_dict' key. Strip if present.
        if isinstance(_sd, dict) and 'model_state_dict' in _sd:
            _sd = _sd['model_state_dict']
        self.actor_critic.load_state_dict(_sd)
        self.actor_critic.base.nenv = 1
        nn.DataParallel(self.actor_critic).to(self.device)
        self.get_logger().info("[Decider] Meta RL model initialized.")
        
        # P Controller for following distance
        self.p_controller = PController(
            kp=4.0, target_distance=2.0, tolerance=0.10,
            adaptive=self.adaptive_mapping,
        )
        self.get_logger().info(
            f"[Decider] adaptive_mapping={self.adaptive_mapping} "
            f"({'Ours' if self.adaptive_mapping else 'Meta-NoMap baseline'})"
        )
        self.target_following_distance = float(self.get_parameter('default_target_distance').value)
        self.current_following_preference = 0  # 当前 following preference (-2 to 2)

        # Recurrent states for RL
        self.eval_recurrent_hidden_states = {}
        self.eval_masks = torch.zeros(1, 1, device=self.device)

        # Robot data
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.robot_vpref = 1.2  # Fix #3: must match training config (robot.v_pref = 1.2)
        self.robot_radius = 0.3
        self.robot_vx = 0.0
        self.robot_vy = 0.0

        # Movement/goal logic
        self.forward_mode = False
        self.goal_x = None
        self.goal_y = None
        self.last_known_pose = (0.0, 0.0, 0.0)

        # Tracked data
        self.current_predictions_ = None
        self.tracked_humans = []
        # Effectively unbounded: target acquisition shouldn't depend on robot
        # being within 5m of target. LiDAR's natural range (~30m) is the real
        # upper limit; this only filters tracker output by distance.
        self.detect_range = 50.0

        # Mecanum smoothing/clipping
        # Fix #5: training has NO acceleration limit and clips at v_pref=1.2.
        # Relax delta to 0.5 (reach full speed in ~2 steps) and raise max_speed to 1.0.
        # Keeps safety margin vs training's 1.2 but reduces the sim2real gap.
        self.motion_mode = "mecanum"
        _max_dv = float(self.get_parameter('max_delta_v').value)
        self.max_delta_vx = _max_dv
        self.max_delta_vy = _max_dv
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.last_rl_vel = np.array([0.0, 0.0], dtype=np.float32)

        # Allow runtime tuning via `ros2 param set` (used by
        # scripts/tune_dr_spaam.py). Without this, edits after node start
        # are silently dropped.
        self.add_on_set_parameters_callback(self._on_param_change)

        # Occupancy grid
        self.occupancy_grid_2d = None
        self.ogm_history = None
        # ogm_history_len is defined earlier (before set_ob_act_space)
        
        # Target human tracking
        self.target_human_id = None
        self.target_human_position = None
        self.last_target_human_position = None  # 上一次有效的目标位置
        self.last_target_human_position_time = 0.0  # Fix #4: timestamp for timeout
        self.target_position_timeout = 2.0  # seconds — stale position expires
        self.target_lost_count = 0
        self.max_target_lost_frames = 10

        # UWB target tracking (Tag-to-Tag mode)
        # Human tag (tag_0) position
        self.uwb_human_x = None
        self.uwb_human_y = None
        # Robot tag (tag_1) position
        self.uwb_robot_x = None
        self.uwb_robot_y = None
        # Computed distance between tags
        self.uwb_tag_distance = None
        self.uwb_last_update = 0.0
        # Height compensation (set to 0 to disable)
        self.uwb_height_diff = 0.9  # meters (human ~1.2m, robot ~0.3m)

        # Legacy compatibility
        self.uwb_target_x = None
        self.uwb_target_y = None
        
        # Protection logic
        self.target_human_confirmed = False
        self.min_confirmations = 3
        self.confirmation_count = 0

        # 使用 UWB 跟踪真实人类（本文件的核心配置）
        self.use_fixed_target = False  # 不使用固定坐标，使用 UWB 跟踪

    def reset_stats(self):
        """重置统计数据"""
        self.stats_lidar_sum = 0.0
        self.stats_lidar_count = 0
        self.stats_uwb_sum = 0.0
        self.stats_uwb_count = 0
        self.log("[Decider] Statistics have been RESET for new run.")

    def log(self, msg, level='info'):
        if level == 'info':
            self.file_logger.info(msg)
            self.get_logger().info(msg)
        elif level == 'warn':
            self.file_logger.warning(msg)
            self.get_logger().warn(msg)
        elif level == 'error':
            self.file_logger.error(msg)
            self.get_logger().error(msg)

    def uwb_human_callback(self, msg: Point):
        """Receive human UWB tag position (tag_0)"""
        if not (np.isfinite(msg.x) and np.isfinite(msg.y)):
            self.get_logger().warn(f"[Decider] Invalid human UWB: ({msg.x}, {msg.y})")
            return

        self.uwb_human_x = msg.x
        self.uwb_human_y = msg.y
        self.uwb_last_update = self.get_clock().now().nanoseconds / 1e9

        # Legacy compatibility: also set uwb_target for existing code
        self.uwb_target_x = msg.x
        self.uwb_target_y = msg.y

        # Calculate tag-to-tag distance if both positions available
        self._update_tag_distance()

    def uwb_robot_callback(self, msg: Point):
        """Receive robot UWB tag position (tag_1)"""
        if not (np.isfinite(msg.x) and np.isfinite(msg.y)):
            self.get_logger().warn(f"[Decider] Invalid robot UWB: ({msg.x}, {msg.y})")
            return

        self.uwb_robot_x = msg.x
        self.uwb_robot_y = msg.y

        # Calculate tag-to-tag distance if both positions available
        self._update_tag_distance()

    def camera_target_callback(self, msg: Point):
        """Receive target person position from camera-based target tracker."""
        if not (np.isfinite(msg.x) and np.isfinite(msg.y)):
            self.get_logger().warn(f"[Decider] Invalid camera target: ({msg.x}, {msg.y})")
            return

        self.camera_target_x = msg.x
        self.camera_target_y = msg.y
        self.camera_target_last_update = self.get_clock().now().nanoseconds / 1e9

        # When camera mode is enabled, also update UWB fields for compatibility
        if self.use_camera_target:
            self.uwb_target_x = msg.x
            self.uwb_target_y = msg.y
            self.uwb_last_update = self.camera_target_last_update

    def _update_tag_distance(self):
        """Calculate distance between human and robot UWB tags"""
        if (self.uwb_human_x is None or self.uwb_human_y is None or
            self.uwb_robot_x is None or self.uwb_robot_y is None):
            return

        # 2D distance (horizontal plane)
        dx = self.uwb_human_x - self.uwb_robot_x
        dy = self.uwb_human_y - self.uwb_robot_y
        dist_2d = np.sqrt(dx*dx + dy*dy)

        # Apply height compensation if enabled
        # Measured 3D distance = sqrt(horizontal² + height²)
        # So horizontal = sqrt(measured² - height²)
        if self.uwb_height_diff > 0 and dist_2d > self.uwb_height_diff:
            # This is already 2D from trilateration, no compensation needed
            # Height compensation is only needed for direct tag-to-tag ranging
            pass

        self.uwb_tag_distance = dist_2d
        self.get_logger().debug(
            f"[UWB] Human:({self.uwb_human_x:.2f},{self.uwb_human_y:.2f}) "
            f"Robot:({self.uwb_robot_x:.2f},{self.uwb_robot_y:.2f}) "
            f"Dist:{dist_2d:.2f}m"
        )

    def occupancy_grid_json_callback(self, msg: String):
        """处理占用栅格消息"""
        try:
            grid_dict = json.loads(msg.data)
        except json.JSONDecodeError as ex:
            self.get_logger().error(f"[Decider] Could not parse occupancy grid JSON: {ex}")
            return

        info = grid_dict.get("info", {})
        w = info.get("width", 0)
        h = info.get("height", 0)
        data_flat = grid_dict.get("data", [])

        if w == 0 or h == 0 or len(data_flat) != w*h:
            self.get_logger().warn("[Decider] Received malformed occupancy grid JSON.")
            return

        grid_2d = np.array(data_flat, dtype=np.int8).reshape((h, w))
        self.occupancy_grid_2d = grid_2d
        self.update_ogm_history(self.occupancy_grid_2d)

    def update_ogm_history(self, current_ogm):
        """更新 OGM 历史"""
        nn_ogm = current_ogm.astype(np.int8)

        if self.ogm_history is None:
            h, w = nn_ogm.shape
            # 使用 np.int8 与 simulation 一致
            self.ogm_history = np.zeros((self.ogm_history_len, h, w), dtype=np.int8)
            for i in range(self.ogm_history_len):
                self.ogm_history[i] = nn_ogm
        else:
            self.ogm_history[:-1] = self.ogm_history[1:]
            self.ogm_history[-1] = nn_ogm

    def get_uwb_target_position(self):
        """获取目标位置（Camera模式或UWB模式）"""
        # Camera mode: camera_target_callback already updates uwb_target_x/y
        # so this method works for both modes without further changes
        if self.uwb_target_x is not None and self.uwb_target_y is not None:
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - self.uwb_last_update < 2.0:
                return self.uwb_target_x, self.uwb_target_y
            else:
                self.get_logger().debug(f"[Decider] UWB data stale: {current_time - self.uwb_last_update:.1f}s old")
        return None, None

    def get_uwb_tag_distance(self):
        """获取 UWB Tag-to-Tag 距离（人和小车之间）"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.uwb_tag_distance is not None and current_time - self.uwb_last_update < 2.0:
            return self.uwb_tag_distance
        return None

    def find_uwb_corresponding_human_id(self, tracked_humans):
        """找到与 UWB 位置对应的人类 ID"""
        uwb_x, uwb_y = self.get_uwb_target_position()
        if uwb_x is None or uwb_y is None:
            self.get_logger().info("[Decider] No UWB data available for human matching")
            return None
        
        min_dist = float('inf')
        closest_human_id = None
        self.get_logger().info(f"[Decider] UWB position: ({uwb_x:.2f}, {uwb_y:.2f}), checking {len(tracked_humans)} humans")
        
        for hid, hx, hy, dist in tracked_humans:
            try:
                if np.isnan(hx) or np.isnan(hy) or np.isinf(hx) or np.isinf(hy):
                    continue
                uwb_dist = np.sqrt((hx - uwb_x)**2 + (hy - uwb_y)**2)
                self.get_logger().info(f"[Decider] Human {hid} at ({hx:.2f}, {hy:.2f}), distance to UWB: {uwb_dist:.2f}m")
                if uwb_dist < min_dist:
                    min_dist = uwb_dist
                    closest_human_id = hid
            except (ValueError, TypeError) as e:
                self.get_logger().warn(f"[Decider] Error calculating distance for human {hid}: {e}")
                continue
        
        if min_dist < 1:
            self.get_logger().info(f"[Decider] ✅ MATCH FOUND: Human {closest_human_id} at {min_dist:.2f}m from UWB")
            return closest_human_id
        else:
            self.get_logger().info(f"[Decider] ❌ NO MATCH: No human within 1.0m of UWB (closest: {min_dist:.2f}m)")
            return None

    def find_target_human(self, tracked_humans):
        """Pick target human. Mode controlled by `target_source` ROS param."""
        # closest_lidar mode: no UWB / camera. Two-phase ID-locking logic:
        #   1) ACQUIRE — no current lock, pick closest, lock that SORT ID.
        #   2) HOLD    — keep returning the locked ID's position even if some
        #                other human is closer (intruder won't steal the target).
        #   3) COAST   — locked ID temporarily missing from tracker output.
        #                Trust SORT's Kalman to re-associate (max_age frames).
        #                We just don't switch target during this gap.
        #   4) RELEASE — locked ID not seen for > lock_release_timeout sec.
        #                Drop lock, re-acquire on next call.
        # See [[robot-tracking-id-lock]] memory for rationale.
        if self.target_source == 'closest_lidar':
            now = self.get_clock().now().nanoseconds / 1e9
            lock_release_timeout = 3.0  # sec — match sort.max_age @ 10Hz

            if not tracked_humans:
                # No humans visible at all. Don't drop lock yet — wait for timeout.
                if self.target_human_id is not None:
                    age = now - self.last_target_human_position_time
                    if age > lock_release_timeout:
                        self.get_logger().info(
                            f"[Decider] lock released — ID={self.target_human_id} "
                            f"absent {age:.1f}s > {lock_release_timeout}s"
                        )
                        self.target_human_id = None
                        self.target_human_confirmed = False
                self.confirmation_count = 0
                return None, None, None

            # Build id→entry map for O(1) lookup
            by_id = {hid: (hid, hx, hy, dist) for hid, hx, hy, dist in tracked_humans}

            # Phase HOLD: locked ID still visible — use its position even if not closest
            if self.target_human_id is not None and self.target_human_id in by_id:
                hid, hx, hy, dist = by_id[self.target_human_id]
                self.target_lost_count = 0
                self.target_human_position = (hx, hy)
                self.last_target_human_position = (hx, hy)
                self.last_target_human_position_time = now
                return hid, hx, hy

            # Phase COAST: locked ID temporarily gone (SORT is coasting via Kalman).
            # Hold lock until timeout, return None so decider keeps last goal stale.
            if self.target_human_id is not None:
                age = now - self.last_target_human_position_time
                if age <= lock_release_timeout:
                    return None, None, None
                # Timeout exceeded — fall through to re-acquire
                self.get_logger().info(
                    f"[Decider] lock released — ID={self.target_human_id} "
                    f"missing {age:.1f}s > {lock_release_timeout}s, re-acquiring..."
                )
                self.target_human_id = None
                self.target_human_confirmed = False

            # Phase ACQUIRE: no lock, pick closest (tracked_humans is sorted by dist)
            closest_id, hx, hy, _dist = tracked_humans[0]
            self.get_logger().info(
                f"[Decider] lock ACQUIRED: ID={closest_id} at ({hx:.2f},{hy:.2f}) "
                f"dist={_dist:.2f}m (out of {len(tracked_humans)} humans)"
            )
            self.target_human_id = closest_id
            self.target_human_confirmed = True
            self.target_lost_count = 0
            self.target_human_position = (hx, hy)
            self.last_target_human_position = (hx, hy)
            self.last_target_human_position_time = now
            return closest_id, hx, hy

        # default 'uwb_camera' path
        uwb_x, uwb_y = self.get_uwb_target_position()
        if uwb_x is not None and uwb_y is not None:
            corresponding_human_id = self.find_uwb_corresponding_human_id(tracked_humans)
            if corresponding_human_id is not None:
                self.target_human_id = corresponding_human_id
                self.target_lost_count = 0
                self.confirmation_count += 1
                
                if self.confirmation_count >= self.min_confirmations:
                    if not self.target_human_confirmed:
                        self.target_human_confirmed = True
                        self.get_logger().info(f"[Decider] Target human CONFIRMED after {self.confirmation_count} matches!")
                
                for hid, hx, hy, dist in tracked_humans:
                    if hid == corresponding_human_id:
                        self.target_human_position = (hx, hy)
                        self.last_target_human_position = (hx, hy)
                        self.last_target_human_position_time = self.get_clock().now().nanoseconds / 1e9
                        self.get_logger().info(f"[Decider] Target human match #{self.confirmation_count}: ID={corresponding_human_id} at ({hx:.2f}, {hy:.2f})")
                        return corresponding_human_id, hx, hy
            else:
                if self.confirmation_count > 0:
                    self.get_logger().info(f"[Decider] Lost UWB-human match, resetting confirmation count from {self.confirmation_count}")
                self.confirmation_count = 0
                self.target_human_confirmed = False
        else:
            if self.confirmation_count > 0:
                self.get_logger().info(f"[Decider] No UWB data, resetting confirmation count from {self.confirmation_count}")
            self.confirmation_count = 0
            self.target_human_confirmed = False
            
        return None, None, None

    def reset_target_tracking(self):
        """重置目标追踪状态"""
        self.target_human_id = None
        self.target_lost_count = 0
        self.target_human_position = None
        self.last_target_human_position = None
        self.target_human_confirmed = False
        self.confirmation_count = 0
        self.get_logger().info("[Decider] Target tracking reset")

    def get_target_human_trajectory(self, sorted_humans, predictions):
        """获取目标轨迹"""
        target_traj = np.ones((self.spatial_edge_dim,), dtype=np.float32) * 15
        
        if self.target_human_id is None:
            return target_traj
        
        target_human_found = False
        target_x, target_y = None, None
        
        for hid, hx, hy, dist in sorted_humans:
            if hid == self.target_human_id:
                target_x, target_y = hx, hy
                target_human_found = True
                break
        
        if target_human_found and target_x is not None and target_y is not None:
            target_traj = np.zeros((self.spatial_edge_dim,), dtype=np.float32)
            dx = target_x - self.robot_x
            dy = target_y - self.robot_y
            
            for i in range(6):
                target_traj[i*2] = dx
                target_traj[i*2+1] = dy
            
            if predictions and self.target_human_id in predictions:
                pred_traj = predictions[self.target_human_id].get("predicted_trajectory", [])
                for step_idx in range(self.predict_steps):
                    if step_idx < len(pred_traj):
                        px = pred_traj[step_idx].get("x", 15.0) - self.robot_x
                        py = pred_traj[step_idx].get("y", 15.0) - self.robot_y
                        idx2 = 2*(step_idx+1)
                        target_traj[idx2] = px
                        target_traj[idx2+1] = py
            
        return target_traj

    # ----- subclass hooks (Meta default → override for RL-PC etc.) -----
    def _create_actor_critic(self):
        """Construct the policy network. Override in subclasses to swap arch."""
        return PolicyMeta(
            self.observation_space,
            self.action_space,
            base_kwargs=self.algo_args,
            base='interaction_transformer_meta',
            config=self.config,
        )

    def _add_preference_obs_space(self, d):
        """Insert the preference-related Box into the obs dict. Meta uses a
        discrete preference index ∈ [-2, 2] under key 'following_preference';
        RL-PC overrides to use raw `preference_distance` instead."""
        d['following_preference'] = gym.spaces.Box(low=-2, high=2, shape=(1, 1), dtype=np.float32)

    def _build_preference_obs(self, current_distance):
        """Compute and return {obs_key: np.array} for the preference channel.
        Default = closed-loop P-controller mapping current_distance → discrete
        preference index. Overridden by RL-PC to pass raw distance straight
        through (which is also the user's `auto:distance:N` setpoint)."""
        if current_distance is not None and self.adaptive_mapping:
            self.current_following_preference = self.p_controller.compute(current_distance)
        elif current_distance is not None:
            # Meta-NoMap fallback: direct mapping (no closed loop).
            self.current_following_preference = self.p_controller.compute(current_distance)
        else:
            self.current_following_preference = 0
        return {
            "following_preference": np.array([[self.current_following_preference]], dtype=np.float32),
        }

    def set_ob_act_space(self):
        d = {}
        d['robot_node'] = gym.spaces.Box(-np.inf, np.inf, shape=(1,7), dtype=np.float32)
        d['temporal_edges'] = gym.spaces.Box(-np.inf, np.inf, shape=(1,2), dtype=np.float32)

        self.spatial_edge_dim = int(2*(self.predict_steps+1))
        d['spatial_edges'] = gym.spaces.Box(
            -np.inf, np.inf,
            shape=(self.config.sim.human_num + self.config.sim.human_num_range, self.spatial_edge_dim),
            dtype=np.float32
        )

        d['visible_masks'] = gym.spaces.Box(
            0, 1, shape=(self.config.sim.human_num + self.config.sim.human_num_range,), dtype=np.bool_
        )
        d['detected_human_num'] = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        d['aggressiveness_factor'] = gym.spaces.Box(-np.inf, np.inf, shape=(1,1), dtype=np.float32)

        local_map_size = 50
        d['local_ogm'] = gym.spaces.Box(low=0, high=1, shape=(self.ogm_history_len, local_map_size, local_map_size), dtype=np.int8)
        d['target_human_traj'] = gym.spaces.Box(-np.inf, np.inf, shape=(self.spatial_edge_dim,), dtype=np.float32)
        self._add_preference_obs_space(d)

        self.observation_space = gym.spaces.Dict(d)
        high = np.inf * np.ones([2,])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def check_manual_timeout(self):
        now = self.get_clock().now()
        elapsed = (now - self.last_manual_cmd_time_).nanoseconds * 1e-9

        if self.current_mode_ == "manual":
            if elapsed > MANUAL_CMD_TIMEOUT:
                self.publish_stop_command()
        elif self.current_mode_ == "combined":
            if self.in_override_ and elapsed > MANUAL_CMD_TIMEOUT:
                self.in_override_ = False
                self.get_logger().info("[Decider] Combined override timed out => revert to RL.")
                self.publish_stop_command()

    def rl_reset(self):
        self.eval_recurrent_hidden_states = {}
        self.eval_masks = torch.zeros(1,1, device=self.device)
        self.goal_x = None
        self.goal_y = None
        self.forward_mode = False
        self.last_rl_vel[:] = 0.0
        self.ogm_history = None
        self.reset_target_tracking()
        self.get_logger().info("[Decider] RL reset complete.")

    def reset_recurrent_states(self):
        self.eval_recurrent_hidden_states = {}
        self.eval_masks = torch.zeros(1,1, device=self.device)
        self.last_rl_vel[:] = 0.0
        self.reset_target_tracking()
        self.get_logger().warn("[Decider] Recurrent states reset due to NaN or exception.")

    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        rot = msg.pose.pose.orientation
        q = (rot.x, rot.y, rot.z, rot.w)
        _, _, yaw = tf_transformations.euler_from_quaternion(q)
        self.robot_theta = yaw

        # Fix #1: /odom twist is in body frame (base_link).
        # Policy was trained with global-frame velocities, so rotate body→global.
        vx_body = msg.twist.twist.linear.x
        vy_body = msg.twist.twist.linear.y
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        self.robot_vx = cos_yaw * vx_body - sin_yaw * vy_body
        self.robot_vy = sin_yaw * vx_body + cos_yaw * vy_body

        self.publish_robot_marker()

    def joint_state_callback(self, msg: JointState):
        pass

    def tracked_objects_json_callback(self, msg: String):
        """Parse tracker data and build observation. RL step runs in rl_timer_callback at 4Hz."""
        if self.current_mode_ not in ("automatic", "combined"):
            return
        if self.current_mode_ == "combined" and self.in_override_:
            return
        if not self.forward_mode and (self.goal_x is None or self.goal_y is None):
            return

        try:
            data = json.loads(msg.data)
            tracks = data.get("tracks", [])
            if not tracks:
                return
            robot_pose = self.get_robot_pose(require_tf_for_rl=True)
            if robot_pose is None:
                self.get_logger().warn("[Decider] No TF => skip RL step.")
                return

            self.robot_x, self.robot_y, self.robot_theta = robot_pose

            tmp = []
            for agent in tracks:
                hid = agent.get("id", -1)
                if hid < 0:
                    continue
                hx = agent.get("x", 15.0)
                hy = agent.get("y", 15.0)
                dist = np.hypot(hx - self.robot_x, hy - self.robot_y)
                if dist <= self.detect_range:
                    tmp.append((hid, hx, hy, dist))
            tmp.sort(key=lambda x: x[3])
            self.tracked_humans = tmp

            # UWB-LiDAR 融合跟踪
            if self.forward_mode:
                target_human_id, target_x, target_y = self.find_target_human(tmp)
                uwb_x, uwb_y = self.get_uwb_target_position()
                uwb_status = f"UWB: ({uwb_x:.2f}, {uwb_y:.2f})" if uwb_x is not None else "UWB: NO DATA"
                self.get_logger().info(f"[Decider] STATUS: {uwb_status} | Humans: {len(tmp)} | Confirmed: {self.target_human_confirmed} | Target ID: {self.target_human_id} | Confirmations: {self.confirmation_count}/{self.min_confirmations}")

                # Broadcast target for external metric logging
                tgt_msg = String()
                if target_human_id is not None and target_x is not None:
                    tgt_msg.data = json.dumps({
                        "id": int(target_human_id),
                        "x": float(target_x), "y": float(target_y),
                    })
                else:
                    tgt_msg.data = json.dumps({"id": None, "x": 0.0, "y": 0.0})
                self.target_pub_.publish(tgt_msg)

            # 等待 UWB-人类匹配确认
            if self.forward_mode and not self.target_human_confirmed:
                self.get_logger().info(f"[Decider] WAITING for UWB-human match confirmation (confirmations: {self.confirmation_count}/{self.min_confirmations})")
                self.pending_rl_obs = None
                return

            # Build observation and store for the 4Hz timer (Fix #6)
            self.pending_rl_obs = self.build_observation_from_humans()

        except json.JSONDecodeError as ex:
            self.get_logger().error(f"[Decider] JSON parse error in tracked_objects: {ex}")

    def rl_timer_callback(self):
        """Fixed 4Hz RL execution — matches training dt=0.25s (Fix #6)."""
        if self.pending_rl_obs is None:
            return
        obs = self.pending_rl_obs
        self.pending_rl_obs = None  # consume once per timer tick
        self.run_rl_step(obs)

    def predictions_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            predictions_list = data.get("predictions", [])
            if not predictions_list:
                return
            pred_dict = {}
            for hum in predictions_list:
                hid = hum["id"]
                pred_dict[hid] = {
                    "predicted_trajectory": hum["predicted_trajectory"],
                    "uncertainty": hum.get("uncertainty", [])
                }
            self.current_predictions_ = pred_dict
        except json.JSONDecodeError as ex:
            self.get_logger().error(f"[Decider] JSON parse error in predictions: {ex}")

    def build_observation_from_humans(self):
        rx, ry, rtheta = self.robot_x, self.robot_y, self.robot_theta

        # Fix #4: expire stale last_target_human_position to prevent chasing old goals (wall collision)
        if self.last_target_human_position is not None:
            now = self.get_clock().now().nanoseconds / 1e9
            age = now - self.last_target_human_position_time
            if age > self.target_position_timeout:
                self.get_logger().warn(
                    f"[Decider] Target position expired ({age:.1f}s > {self.target_position_timeout}s), clearing stale goal")
                self.last_target_human_position = None

        if self.target_human_position is not None:
            gx, gy = self.target_human_position
        elif self.last_target_human_position is not None:
            gx, gy = self.last_target_human_position
        elif self.goal_x is not None and self.goal_y is not None:
            gx, gy = self.goal_x, self.goal_y
        elif self.forward_mode:
            gx, gy = rx + 5.0, ry
        else:
            gx, gy = rx + 2.0, ry

        robot_node = np.array([[rx, ry, self.robot_radius, gx, gy, self.robot_vpref, rtheta]], dtype=np.float32)
        temporal_edges = np.array([[self.robot_vx, self.robot_vy]], dtype=np.float32)

        nHumansMax = 50
        spatial_edges = np.ones((nHumansMax, self.spatial_edge_dim), dtype=np.float32)*15
        visible_masks = np.zeros((nHumansMax,), dtype=bool)

        sorted_humans = self.tracked_humans[:nHumansMax]
        detected_num = len(sorted_humans)

        for i,(hid,hx,hy,dist) in enumerate(sorted_humans):
            dx = hx - rx
            dy = hy - ry
            
            for t in range(6):
                spatial_edges[i, t*2] = dx
                spatial_edges[i, t*2+1] = dy

            if self.current_predictions_ and (hid in self.current_predictions_):
                pred_traj = self.current_predictions_[hid].get("predicted_trajectory",[])
                for step_idx in range(self.predict_steps):
                    if step_idx < len(pred_traj):
                        px = pred_traj[step_idx].get("x",15.0) - rx
                        py = pred_traj[step_idx].get("y",15.0) - ry
                        idx2 = 2*(step_idx+1)
                        spatial_edges[i, idx2] = px
                        spatial_edges[i, idx2+1] = py

            visible_masks[i] = True

        target_human_traj = self.get_target_human_trajectory(sorted_humans, self.current_predictions_)

        local_ogm = np.zeros((3, 50, 50), dtype=np.int8)
        if self.ogm_history is not None:
            local_ogm = self.ogm_history.copy()

        # Compute preference channel (Meta uses discrete index via P-controller;
        # subclasses like RL-PC override to use raw distance).
        if self.target_human_position is not None:
            current_distance = np.sqrt(
                (self.target_human_position[0] - rx)**2 +
                (self.target_human_position[1] - ry)**2
            )
        else:
            current_distance = None
        preference_obs_kv = self._build_preference_obs(current_distance)

        if self.is_logging_active:
            current_lidar_dist = float('nan')
            if self.target_human_position is not None:
                current_lidar_dist = np.sqrt(
                    (self.target_human_position[0] - rx)**2 + 
                    (self.target_human_position[1] - ry)**2
                )
                self.stats_lidar_sum += current_lidar_dist
                self.stats_lidar_count += 1

            current_uwb_dist = float('nan')
            # Use Tag-to-Tag distance
            tag_dist = self.get_uwb_tag_distance()
            if tag_dist is not None:
                current_uwb_dist = tag_dist
                self.stats_uwb_sum += current_uwb_dist
                self.stats_uwb_count += 1

            avg_lidar = self.stats_lidar_sum / self.stats_lidar_count if self.stats_lidar_count > 0 else 0.0
            avg_uwb = self.stats_uwb_sum / self.stats_uwb_count if self.stats_uwb_count > 0 else 0.0
            
            lidar_str = f"{current_lidar_dist:.3f}" if not np.isnan(current_lidar_dist) else "N/A"
            uwb_str = f"{current_uwb_dist:.3f}" if not np.isnan(current_uwb_dist) else "N/A"
            
            log_msg = (
                f"[DistanceStats] "
                f"LiDAR_Cur: {lidar_str} | LiDAR_Avg: {avg_lidar:.3f} | "
                f"UWB_Cur: {uwb_str} | UWB_Avg: {avg_uwb:.3f} | "
                f"Target_Dist: {self.target_following_distance:.2f} | "
                f"Current_Pref: {self.current_following_preference}"
            )
            self.log(log_msg)

        obs_dict = {
            "robot_node": robot_node,
            "temporal_edges": temporal_edges,
            "spatial_edges": spatial_edges,
            "visible_masks": visible_masks,
            "detected_human_num": np.array([detected_num], dtype=np.float32),
            "aggressiveness_factor": np.zeros((1,1), dtype=np.float32),
            "local_ogm": local_ogm,
            "target_human_traj": target_human_traj,
        }
        obs_dict.update(preference_obs_kv)

        obs = {}
        for k,v in obs_dict.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 1:
                    obs[k] = torch.from_numpy(v).unsqueeze(0).to(self.device)
                elif v.ndim == 2:
                    obs[k] = torch.from_numpy(v).unsqueeze(0).to(self.device)
                elif v.ndim == 3:
                    obs[k] = torch.from_numpy(v).unsqueeze(0).to(self.device)
                else:
                    obs[k] = torch.from_numpy(v).to(self.device)
            
        return obs

    def run_rl_step(self, obs):
        try:
            with torch.no_grad():
                rl_start_time = time.time()
                result = self.actor_critic.act(
                    obs,
                    self.eval_recurrent_hidden_states,
                    self.eval_masks,
                    deterministic=True
                )
                _, action_tensor, _, self.eval_recurrent_hidden_states = result
                rl_elapsed = time.time() - rl_start_time
                self.get_logger().info(f"[Decider] Time for RL inference: {rl_elapsed}")

            # ---- Obs/action dump for sim↔real distribution comparison ----
            # Each forward() writes one JSONL line to /tmp/decider_obs.jsonl.
            # Enabled by env var DUMP_OBS=1 (off by default to avoid disk I/O).
            try:
                if os.environ.get('DUMP_OBS', '0') == '1':
                    if not hasattr(self, '_obs_dump_f'):
                        import json as _json
                        self._obs_dump_f = open('/tmp/decider_obs.jsonl', 'w')
                        self._obs_dump_json = _json
                        self.get_logger().info("[Decider] Obs dump enabled → /tmp/decider_obs.jsonl")
                    _ogm = obs['local_ogm'].cpu().numpy()
                    _se  = obs['spatial_edges'].cpu().numpy()
                    rec = {
                        't': time.time(),
                        'robot_node': obs['robot_node'].cpu().numpy().flatten().tolist(),
                        'temporal_edges': obs['temporal_edges'].cpu().numpy().flatten().tolist(),
                        'following_preference': float(obs['following_preference'].cpu().numpy().flatten()[0]),
                        'detected_human_num': int(obs['detected_human_num'].cpu().numpy().flatten()[0]),
                        'target_human_traj': obs['target_human_traj'].cpu().numpy().flatten().tolist(),
                        # spatial_edges: dump first 3 humans (the closest), each is 12 values
                        'spatial_edges_top3': _se.reshape(-1, _se.shape[-1])[:3].tolist(),
                        # OGM: shape + occupancy stats (full OGM too big for JSON every tick)
                        'ogm_shape': list(_ogm.shape),
                        'ogm_min': float(_ogm.min()),
                        'ogm_max': float(_ogm.max()),
                        'ogm_mean': float(_ogm.mean()),
                        'ogm_sum': int(_ogm.sum()),       # in binary {0,1} this = occupied cell count
                        'ogm_n_eq_1': int((_ogm == 1).sum()),
                        'ogm_n_eq_0': int((_ogm == 0).sum()),
                        'ogm_n_lt_0': int((_ogm < 0).sum()),
                        'ogm_n_ge_50': int((_ogm >= 50).sum()),
                        # raw action (pre-clip)
                        'action_raw': action_tensor.cpu().numpy().flatten().tolist(),
                        # value (critic prediction)
                        'value': float(result[0].cpu().numpy().flatten()[0]),
                    }
                    self._obs_dump_f.write(self._obs_dump_json.dumps(rec) + '\n')
                    self._obs_dump_f.flush()
            except Exception as _e:
                self.get_logger().warn(f"[Decider] obs dump failed: {_e}")
            # ---- end dump ----

            if torch.isnan(action_tensor).any():
                self.get_logger().warn("[Decider] NaN in action_tensor => resetting states.")
                self.reset_recurrent_states()
                return

            raw_action = action_tensor.cpu().numpy()[0]
            vx_raw, vy_raw = raw_action[0], raw_action[1]

            # APF safety: pre-normalize raw action to max_speed scale so the
            # safety contribution is at the SAME order of magnitude as the
            # policy command. Otherwise the policy can output ||v||=3m/s and
            # the safety's 0.5m/s repulse is lost in the noise.
            vx_raw, vy_raw, _safety_dbg = self.apply_safety_repulsion(
                vx_raw, vy_raw, max_scale=self.max_speed
            )

            new_vel = np.array([vx_raw, vy_raw], dtype=np.float32)
            vx_smooth, vy_smooth = self.smooth_and_clip_mecanum(new_vel, self.last_rl_vel)
            self.last_rl_vel[:] = [vx_smooth, vy_smooth]

            if not self.in_override_:
                self.publish_velocity_command_global(vx_smooth, vy_smooth, require_tf_for_rl=True)

            self.publish_goal_marker()

        except Exception as ex:
            self.get_logger().error(f"[Decider] RL step exception: {ex}")
            self.reset_recurrent_states()

    def _on_param_change(self, params):
        """Apply runtime parameter updates from `ros2 param set`.

        Tunable at runtime:
          - safety_enabled, safety_radius, safety_strength,
            safety_hard_enabled, safety_hard_dist
          - max_speed, max_delta_v
          - default_target_distance
        """
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == 'safety_enabled':
                self.safety_enabled = bool(p.value)
            elif p.name == 'safety_radius':
                self.safety_radius = float(p.value)
            elif p.name == 'safety_strength':
                self.safety_strength = float(p.value)
            elif p.name == 'safety_hard_enabled':
                self.safety_hard_enabled = bool(p.value)
            elif p.name == 'safety_hard_dist':
                self.safety_hard_dist = float(p.value)
            elif p.name == 'max_speed':
                self.max_speed = float(p.value)
            elif p.name == 'max_delta_v':
                v = float(p.value)
                self.max_delta_vx = v
                self.max_delta_vy = v
            elif p.name == 'default_target_distance':
                # update PController target if it exists
                self.target_following_distance = float(p.value)
                if hasattr(self, 'p_controller'):
                    try:
                        self.p_controller.set_target_distance(float(p.value))
                    except Exception:
                        pass
            else:
                continue
            self.get_logger().info(f"[tune] {p.name} → {p.value}")
        return SetParametersResult(successful=True)

    def apply_safety_repulsion(self, vx_raw, vy_raw, max_scale=None):
        """APF (soft) + forced-reverse (hard) repulsion against OGM-occupied
        cells.

        Returns (vx_new, vy_new, dbg_tuple).

        max_scale: optional. If given, pre-clips the raw policy action to this
        magnitude BEFORE applying safety. Critical for safety to actually
        dominate — without pre-clip, policy ||v|| can be 3 m/s and safety's
        0.5 m/s repulse gets swamped after the downstream max_speed clip.
        Pass max_scale=self.max_speed at the call site.

        Hard cut: when the closest cell is within safety_hard_dist, we keep
        the velocity's perpendicular component but REPLACE the inward
        component with an outward push proportional to (1 - d/hard_dist) ×
        safety_strength. This forces the robot AWAY from the wall regardless
        of what the policy commanded (the user's original Plan B "硬反向").

        Frame: OGM is robot-centered (grid origin = robot - 5m in odom).
        So cell_xy_in_robot_frame = (j+0.5)*res - grid_size/2.
        action_raw is in odom/global frame; OGM grid axes are odom-aligned
        (NOT rotated by yaw) — matches action_raw's frame. ✓
        """
        if not self.safety_enabled or self.ogm_history is None:
            return vx_raw, vy_raw, None

        ogm = self.ogm_history[-1]  # latest frame, shape (H, W) int8
        res = 0.2
        grid_half = ogm.shape[0] * res / 2.0  # 5.0 m

        occ_ys, occ_xs = np.where(ogm == 1)
        if len(occ_xs) == 0:
            return vx_raw, vy_raw, None

        # ---- pre-clip policy action to working scale ----
        # Otherwise policy ||v||=3m/s drowns out safety's 0.5m/s repulse.
        if max_scale is not None:
            v_mag_in = math.hypot(vx_raw, vy_raw)
            if v_mag_in > max_scale and v_mag_in > 1e-3:
                scale = max_scale / v_mag_in
                vx_raw *= scale
                vy_raw *= scale

        # Cell centers relative to robot, in odom-aligned frame
        cell_dx = (occ_xs + 0.5) * res - grid_half
        cell_dy = (occ_ys + 0.5) * res - grid_half
        dists = np.hypot(cell_dx, cell_dy)

        vx_new, vy_new = vx_raw, vy_raw
        hard_cut_engaged = False
        outward_push = 0.0

        # ---- LAYER 1: APF soft repulsion ----
        R = self.safety_radius
        influence = np.clip(1.0 - dists / R, 0.0, 1.0) ** 2
        rep_x = rep_y = 0.0
        if influence.sum() > 0:
            dists_safe = np.maximum(dists, 0.05)
            rep_x = float((influence * (-cell_dx / dists_safe)).sum())
            rep_y = float((influence * (-cell_dy / dists_safe)).sum())
            rep_mag = math.hypot(rep_x, rep_y)
            if rep_mag > self.safety_strength:
                scale = self.safety_strength / rep_mag
                rep_x *= scale
                rep_y *= scale
            vx_new += rep_x
            vy_new += rep_y

        # ---- LAYER 2: HARD REVERSE (forces outward motion) ----
        # When closest cell is within hard_dist: replace inward velocity
        # component with an outward push (proportional to 1 - d/hard_dist).
        # Perpendicular component is preserved (robot can slide along wall).
        if self.safety_hard_enabled and len(dists) > 0:
            i_min = int(np.argmin(dists))
            d_min = float(dists[i_min])
            if d_min < self.safety_hard_dist:
                if d_min > 1e-3:
                    # Unit vector from robot TOWARD nearest cell (= "inward")
                    nx = cell_dx[i_min] / d_min
                    ny = cell_dy[i_min] / d_min
                    # Decompose velocity: parallel (away/toward) + perpendicular
                    v_in = vx_new * nx + vy_new * ny
                    vx_perp = vx_new - v_in * nx
                    vy_perp = vy_new - v_in * ny
                    # Force outward push (always negative-of-normal direction):
                    # outward_push grows from 0 at hard_dist to safety_strength at d=0
                    outward_push = self.safety_strength * (1.0 - d_min / self.safety_hard_dist)
                    vx_new = vx_perp - outward_push * nx
                    vy_new = vy_perp - outward_push * ny
                    hard_cut_engaged = True

        # Throttle log to ~1/sec when active
        self._safety_log_count = (self._safety_log_count + 1) % 4
        any_action = (math.hypot(rep_x, rep_y) > 0.05) or hard_cut_engaged
        if any_action and self._safety_log_count == 0:
            n_close = int((dists < R).sum())
            mark = "🚨HARD" if hard_cut_engaged else "soft"
            self.get_logger().info(
                f"[Safety/{mark}] {n_close} cells <{R:.2f}m, closest={float(dists.min()):.2f}m → "
                f"soft=({rep_x:+.2f},{rep_y:+.2f}) outward_push={outward_push:.2f} | "
                f"action ({vx_raw:+.2f},{vy_raw:+.2f}) → ({vx_new:+.2f},{vy_new:+.2f})"
            )
        return vx_new, vy_new, (rep_x, rep_y, hard_cut_engaged)

    def smooth_and_clip_mecanum(self, new_vel, last_vel):
        vx_current, vy_current = last_vel
        vx_target, vy_target = new_vel

        delta_vx = np.clip(vx_target - vx_current, -self.max_delta_vx, self.max_delta_vx)
        delta_vy = np.clip(vy_target - vy_current, -self.max_delta_vy, self.max_delta_vy)

        vx_candidate = vx_current + delta_vx
        vy_candidate = vy_current + delta_vy
        speed = np.hypot(vx_candidate, vy_candidate)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            vx_candidate *= scale
            vy_candidate *= scale

        return float(vx_candidate), float(vy_candidate)

    def command_callback(self, msg: String):
        cmd_str = msg.data
        self.get_logger().info(f"[Decider] Command: {cmd_str}")

        if cmd_str == "stop":
            self.publish_stop_command()
            self.current_mode_ = None
            if self.is_logging_active:
                self.is_logging_active = False
                self.log("[Decider] Stop command received. Logging PAUSED.")
            return

        if cmd_str.startswith("mode:"):
            new_mode = cmd_str.split(":")[1]
            self.current_mode_ = new_mode
            self.get_logger().info(f"[Decider] Switched to mode: {self.current_mode_}")
            if new_mode in ("automatic", "combined"):
                self.rl_reset()
            else:
                self.in_override_ = False
                self.publish_stop_command()
            return

        if self.current_mode_ == "manual":
            if cmd_str.startswith("manual:"):
                self.handle_manual_command(cmd_str)
            return

        if self.current_mode_ in ("automatic","combined"):
            if cmd_str == "auto:forward":
                self.forward_mode = True
                self.goal_x = None
                self.goal_y = None
            elif cmd_str == "auto:human_following":
                self.get_logger().info("[Decider] Human following mode activated (UWB + Controller).")
                self.forward_mode = True
                self.reset_stats()
                self.is_logging_active = True
                self.log("[Decider] Human following STARTED. Logging ACTIVATED.")
                self.goal_x = None
                self.goal_y = None
            elif cmd_str == "auto:reset_target":
                self.reset_target_tracking()
                self.get_logger().info("[Decider] Target tracking manually reset.")
            elif cmd_str.startswith("auto:goal:"):
                try:
                    s = cmd_str.split("auto:goal:")[-1]
                    gx_str, gy_str = s.split(",")
                    self.goal_x = float(gx_str)
                    self.goal_y = float(gy_str)
                    self.forward_mode = False
                except:
                    self.get_logger().error("[Decider] auto:goal parse error.")
            elif cmd_str.startswith("auto:distance:"):
                # 设置目标跟随距离
                try:
                    distance = float(cmd_str.split("auto:distance:")[-1])
                    self.target_following_distance = max(1.0, min(6.0, distance))
                    self.p_controller.set_target_distance(self.target_following_distance)
                    self.get_logger().info(f"[Decider] Target following distance set to: {self.target_following_distance:.2f}m")
                except:
                    self.get_logger().error("[Decider] auto:distance parse error. Usage: auto:distance:2.0")
            elif cmd_str.startswith("manual:") and self.current_mode_ == "combined":
                self.handle_manual_command(cmd_str)
            else:
                self.get_logger().warn(f"[Decider] Unknown command: {cmd_str}")

    def handle_manual_command(self, cmd_str: str):
        self.last_manual_cmd_time_ = self.get_clock().now()
        if self.current_mode_ == "combined":
            self.in_override_ = True

        key = cmd_str.split(":")[1]
        if key == 'w':
            vx_global, vy_global = 0.5, 0.0
        elif key == 's':
            vx_global, vy_global = -0.5, 0.0
        elif key == 'a':
            vx_global, vy_global = 0.0, 0.5
        elif key == 'd':
            vx_global, vy_global = 0.0, -0.5
        else:
            self.get_logger().warn(f"[Decider] Unrecognized manual key: {key}")
            return

        self.publish_velocity_command_global(vx_global, vy_global, require_tf_for_rl=False)

    def publish_goal_marker(self):
        if self.forward_mode:
            gx = self.robot_x + 5.0
            gy = 0.0
        elif self.goal_x is not None and self.goal_y is not None:
            gx, gy = self.goal_x, self.goal_y
        else:
            gx, gy = self.robot_x, self.robot_y

        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "decider_goal"
        marker.id = 1
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        diameter = 2.0 * (1.5 * self.robot_radius)
        marker.scale.x = diameter
        marker.scale.y = diameter
        marker.scale.z = 0.02

        marker.color = ColorRGBA(r=1.0, g=0.65, b=0.0, a=0.6)
        marker.pose.position.x = gx
        marker.pose.position.y = gy
        marker.pose.orientation.w = 1.0

        self.goal_marker_pub_.publish(marker)

    def publish_robot_marker(self):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "decider_robot"
        marker.id = 2
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4
        marker.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=1.0)

        marker.pose.position.x = self.robot_x
        marker.pose.position.y = self.robot_y
        marker.pose.orientation.w = 1.0

        self.robot_marker_pub_.publish(marker)

    def publish_velocity_command_global(self, vx_global: float, vy_global: float, require_tf_for_rl: bool=False):
        pose = self.get_robot_pose(require_tf_for_rl)
        if pose is None and require_tf_for_rl:
            self.get_logger().warn("[Decider] RL velocity not published, no TF.")
            return

        if pose is not None:
            rx, ry, ryaw = pose
        else:
            rx, ry, ryaw = self.last_known_pose

        local_vx = np.cos(ryaw)*vx_global + np.sin(ryaw)*vy_global
        local_vy = -np.sin(ryaw)*vx_global + np.cos(ryaw)*vy_global

        twist_msg = Twist()
        twist_msg.linear.x = float(local_vx)
        twist_msg.linear.y = float(local_vy)
        twist_msg.angular.z = 0.0
        self.cmd_vel_pub_.publish(twist_msg)

        self.get_logger().info(
            f"[Decider] => G=({vx_global:.2f},{vy_global:.2f}), L=({local_vx:.2f},{local_vy:.2f})"
        )
        self.publish_action_marker_global(rx, ry, vx_global, vy_global)

    def publish_stop_command(self):
        twist_msg = Twist()
        self.cmd_vel_pub_.publish(twist_msg)
        self.get_logger().info("[Decider] Stop command.")

        rx, ry, ryaw = self.last_known_pose
        self.publish_action_marker_global(rx, ry, 0.0, 0.0)

    def get_robot_pose(self, require_tf_for_rl=False):
        try:
            tf_stamped = self.tf_buffer_.lookup_transform('odom','base_link', rclpy.time.Time())
            trans = tf_stamped.transform.translation
            rot = tf_stamped.transform.rotation
            q = (rot.x, rot.y, rot.z, rot.w)
            _, _, yaw = tf_transformations.euler_from_quaternion(q)
            rx, ry = trans.x, trans.y
            self.last_known_pose = (rx, ry, yaw)
            return (rx, ry, yaw)
        except Exception as e:
            if require_tf_for_rl:
                self.get_logger().error(f"[Decider] TF lookup failed for RL: {e}")
                return None
            else:
                return None

    def publish_action_marker_global(self, rx, ry, vx_g, vy_g):
        vel_mag = np.hypot(vx_g, vy_g)
        arrow_theta = np.arctan2(vy_g, vx_g) if vel_mag > 0.001 else 0.0

        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "decider_action"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.scale.x = float(vel_mag)
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=1.0)

        marker.pose.position.x = rx
        marker.pose.position.y = ry
        q = tf_transformations.quaternion_from_euler(0, 0, arrow_theta)
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]

        self.action_marker_pub_.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = Decider()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[Decider] KeyboardInterrupt -> shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

