#!/usr/bin/env python3
"""
Fixed Human 版本 - 手动调节 preference (-2, -1, 0, 1, 2)
使用固定坐标作为目标，通过 auto:preference:X 命令切换 preference
"""

import json
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

MANUAL_CMD_TIMEOUT = 0.2
MAX_HUMANS = 50


class Decider(Node):
    def __init__(self):
        super().__init__('decider_node')

        # ============= 1. File Logging Setup =============
        self.log_dir = 'decider_logs'
        os.makedirs(self.log_dir, exist_ok=True)
        
        log_filename = datetime.now().strftime("decider_fixed_human_%Y%m%d-%H%M%S.log")
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

        # ============= TF2 for Robot Pose =============
        self.tf_buffer_ = tf2_ros.Buffer()
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self)

        # ============= Publishers =============
        self.cmd_vel_pub_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.action_marker_pub_ = self.create_publisher(Marker, '/decider_action_marker', 10)
        self.goal_marker_pub_ = self.create_publisher(Marker, '/decider_goal_marker', 10)
        self.robot_marker_pub_ = self.create_publisher(Marker, '/decider_robot_marker', 10)

        # ============= Mode and override logic =============
        self.current_mode_ = None
        self.in_override_ = False
        self.last_manual_cmd_time_ = self.get_clock().now()
        self.timer_ = self.create_timer(0.1, self.check_manual_timeout)

        # ============= RL Setup =============
        from config.arguments import get_args
        self.algo_args = get_args()
        self.get_logger().info("[Decider] Successfully imported algo_args.")

        from config.config import Config
        self.config = Config()
        self.get_logger().info("[Decider] Successfully imported config.")

        self.predict_steps = 5
        self.set_ob_act_space()

        # Meta Policy with following preference input
        self.actor_critic = PolicyMeta(
            self.observation_space,
            self.action_space,
            base_kwargs=self.algo_args,
            base='interaction_transformer_meta',
            config=self.config
        )

        decider_share_dir = get_package_share_directory('decider')
        load_path = os.path.join(decider_share_dir, 'model_weight', 'meta_2.pt')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic.load_state_dict(torch.load(load_path, map_location=self.device))
        self.actor_critic.base.nenv = 1
        nn.DataParallel(self.actor_critic).to(self.device)
        self.get_logger().info("[Decider] Meta RL model initialized.")
        
        # 手动调节的 following preference (-2 to 2)
        self.current_following_preference = 0  # 默认值

        # Recurrent states for RL
        self.eval_recurrent_hidden_states = {}
        self.eval_masks = torch.zeros(1, 1, device=self.device)

        # Robot data
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.robot_vpref = 1.0
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
        self.detect_range = 5.0

        # Mecanum smoothing/clipping
        self.motion_mode = "mecanum"
        self.max_delta_vx = 0.25
        self.max_delta_vy = 0.25
        self.max_speed = 0.6
        self.last_rl_vel = np.array([0.0, 0.0], dtype=np.float32)

        # Occupancy grid
        self.occupancy_grid_2d = None
        self.ogm_history = None
        self.ogm_history_len = 3
        
        # Target human tracking
        self.target_human_id = None
        self.target_human_position = None
        self.target_human_confirmed = False

        # 固定 target 坐标（本文件的核心配置）
        self.use_fixed_target = True  # 使用固定坐标
        self.fixed_target_position = (2.4, 0)  # 默认固定坐标 (x, y)

        # 移动目标配置
        self.moving_target_enabled = True  # 启用移动目标
        self.moving_target_velocity = 0.6  # m/s 沿 x 轴
        self.moving_target_duration = 15.0  # 秒
        self.moving_target_start_time = None  # 开始时间
        self.moving_target_initial_x = 2.4  # 初始 x 坐标
        self.moving_target_y = 0.0  # y 坐标保持不变

    def reset_stats(self):
        """重置统计数据"""
        self.stats_lidar_sum = 0.0
        self.stats_lidar_count = 0
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
            self.ogm_history = np.zeros((self.ogm_history_len, h, w), dtype=np.int8)
            for i in range(self.ogm_history_len):
                self.ogm_history[i] = nn_ogm
        else:
            self.ogm_history[:-1] = self.ogm_history[1:]
            self.ogm_history[-1] = nn_ogm

    def find_target_human(self, tracked_humans):
        """使用固定坐标或移动目标作为目标"""
        if self.use_fixed_target:
            if self.moving_target_enabled and self.moving_target_start_time is not None:
                # 计算移动目标位置
                current_time = time.time()
                elapsed = current_time - self.moving_target_start_time

                if elapsed <= self.moving_target_duration:
                    # 匀速移动中
                    fx = self.moving_target_initial_x + self.moving_target_velocity * elapsed
                else:
                    # 移动结束，保持最终位置
                    fx = self.moving_target_initial_x + self.moving_target_velocity * self.moving_target_duration

                fy = self.moving_target_y
                self.fixed_target_position = (fx, fy)  # 更新位置
            else:
                fx, fy = self.fixed_target_position

            self.target_human_position = (fx, fy)
            self.target_human_confirmed = True
            self.target_human_id = -1
            return -1, fx, fy
        return None, None, None

    def reset_target_tracking(self):
        """重置目标追踪状态"""
        self.target_human_id = None
        self.target_human_position = None
        self.target_human_confirmed = False
        self.moving_target_start_time = None  # 重置移动目标计时
        self.fixed_target_position = (self.moving_target_initial_x, self.moving_target_y)
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
        d['following_preference'] = gym.spaces.Box(low=-2, high=2, shape=(1, 1), dtype=np.float32)
        
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
        self.robot_vx = msg.twist.twist.linear.x
        self.robot_vy = msg.twist.twist.linear.y
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        rot = msg.pose.pose.orientation
        q = (rot.x, rot.y, rot.z, rot.w)
        _, _, yaw = tf_transformations.euler_from_quaternion(q)
        self.robot_theta = yaw
        self.publish_robot_marker()

    def joint_state_callback(self, msg: JointState):
        pass

    def tracked_objects_json_callback(self, msg: String):
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

            if self.forward_mode:
                self.find_target_human(tmp)

            if self.forward_mode and not self.target_human_confirmed:
                return

            obs = self.build_observation_from_humans()
            self.run_rl_step(obs)

        except json.JSONDecodeError as ex:
            self.get_logger().error(f"[Decider] JSON parse error in tracked_objects: {ex}")

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
        # 固定目标模式：用 fixed_target_position 作为 goal
        if self.target_human_position is not None:
            gx, gy = self.target_human_position
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

        # 固定模式下：创建一个假的 target human
        if self.use_fixed_target:
            fx, fy = self.fixed_target_position
            dist = np.sqrt((fx - rx)**2 + (fy - ry)**2)
            sorted_humans = [(-1, fx, fy, dist)]
        else:
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

        # 使用手动设置的 preference（不用 P Controller）
        following_preference = np.array([[self.current_following_preference]], dtype=np.float32)

        if self.is_logging_active:
            current_lidar_dist = float('nan')
            if self.target_human_position is not None:
                current_lidar_dist = np.sqrt(
                    (self.target_human_position[0] - rx)**2 + 
                    (self.target_human_position[1] - ry)**2
                )
                self.stats_lidar_sum += current_lidar_dist
                self.stats_lidar_count += 1

            avg_lidar = self.stats_lidar_sum / self.stats_lidar_count if self.stats_lidar_count > 0 else 0.0
            lidar_str = f"{current_lidar_dist:.3f}" if not np.isnan(current_lidar_dist) else "N/A"
            
            log_msg = (
                f"[DistanceStats] "
                f"LiDAR_Cur: {lidar_str} | LiDAR_Avg: {avg_lidar:.3f} | "
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
            "following_preference": following_preference
        }

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

            if torch.isnan(action_tensor).any():
                self.get_logger().warn("[Decider] NaN in action_tensor => resetting states.")
                self.reset_recurrent_states()
                return

            raw_action = action_tensor.cpu().numpy()[0]
            vx_raw, vy_raw = raw_action[0], raw_action[1]

            new_vel = np.array([vx_raw, vy_raw], dtype=np.float32)
            vx_smooth, vy_smooth = self.smooth_and_clip_mecanum(new_vel, self.last_rl_vel)
            self.last_rl_vel[:] = [vx_smooth, vy_smooth]

            if not self.in_override_:
                self.publish_velocity_command_global(vx_smooth, vy_smooth, require_tf_for_rl=True)

            self.publish_goal_marker()

        except Exception as ex:
            self.get_logger().error(f"[Decider] RL step exception: {ex}")
            self.reset_recurrent_states()

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
                self.get_logger().info("[Decider] Human following mode activated (Fixed Human).")
                self.forward_mode = True
                self.reset_stats()
                self.is_logging_active = True
                self.log("[Decider] Human following STARTED. Logging ACTIVATED.")
                self.goal_x = None
                self.goal_y = None
                # 启动移动目标
                if self.moving_target_enabled:
                    self.moving_target_start_time = time.time()
                    self.fixed_target_position = (self.moving_target_initial_x, self.moving_target_y)
                    self.log(f"[Decider] Moving target started: v={self.moving_target_velocity}m/s, duration={self.moving_target_duration}s")
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
            elif cmd_str.startswith("auto:preference:"):
                # 手动设置 following preference: -2, -1, 0, 1, 2
                try:
                    pref = int(cmd_str.split("auto:preference:")[-1])
                    self.current_following_preference = max(-2, min(2, pref))
                    self.get_logger().info(f"[Decider] ★ Following preference set to: {self.current_following_preference}")
                except:
                    self.get_logger().error("[Decider] auto:preference parse error. Usage: auto:preference:0 (valid: -2, -1, 0, 1, 2)")
            elif cmd_str.startswith("auto:fixed_target:"):
                # 设置固定目标位置
                try:
                    s = cmd_str.split("auto:fixed_target:")[-1]
                    fx_str, fy_str = s.split(",")
                    self.fixed_target_position = (float(fx_str), float(fy_str))
                    self.get_logger().info(f"[Decider] Fixed target position set to: {self.fixed_target_position}")
                except:
                    self.get_logger().error("[Decider] auto:fixed_target parse error. Usage: auto:fixed_target:2.4,0")
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

