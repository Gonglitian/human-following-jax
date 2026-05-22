#!/usr/bin/env python3
"""
SG-ORCA baseline (paper §V.B): Subgoal-Guided ORCA.

Thin ROS 2 wrapper around the official Python-RVO2 binding
(github.com/sybrenstuvel/Python-RVO2). Mirrors the exact API surface used
by the paper's training-side baseline at
  human-following-robot/crowd_nav/policy/orca.py
so this is the same reference implementation, not a re-write.

Two-step pipeline:
  1. Subgoal: place a goal d* metres behind the target along the
     robot→target line (paper §V.B "fixed offset behind the target").
  2. Velocity: build an RVO2 simulator each step with the robot as agent 0
     and every other tracked human as agent i+1; set the robot's
     pref_velocity toward the subgoal and ORCA returns a collision-free
     velocity that we publish to /cmd_vel.

Subscriptions / command API match decider.main and main_mpc_adc, so the
same launch and the same /command sequence drive all three baselines.
"""

import json
import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, ColorRGBA
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

import tf2_ros
import tf_transformations

try:
    import rvo2
    RVO2_AVAILABLE = True
except ImportError as e:
    RVO2_AVAILABLE = False
    _RVO2_IMPORT_ERR = e


# RVO2 simulator parameters — taken from the training-side ORCA baseline at
# human-following-robot/crowd_nav/configs/config.py (orca section). Same
# values as the paper's training environment so behaviour matches Table I.
DT = 0.25                # time_step (matches decider 4 Hz loop)
NEIGHBOR_DIST = 10.0     # consider all visible humans
TIME_HORIZON = 5.0       # collision look-ahead [s]
TIME_HORIZON_OBST = 5.0  # only used if static obstacles added
SAFETY_SPACE = 0.15      # extra inflation on agent radii


class OrcaFollower(Node):
    def __init__(self):
        super().__init__('decider_node')

        if not RVO2_AVAILABLE:
            self.get_logger().error(
                f'[ORCA] rvo2 not importable: {_RVO2_IMPORT_ERR}. '
                'Install via: pip install --user '
                'git+https://github.com/sybrenstuvel/Python-RVO2'
            )
            raise RuntimeError('rvo2 missing')

        # ---- params ----
        self.declare_parameter('v_max', 1.0)
        self.declare_parameter('robot_radius', 0.3)
        self.declare_parameter('human_radius', 0.35)
        # Effectively unbounded — see decider.main for rationale.
        self.declare_parameter('detect_range', 50.0)
        self.declare_parameter('subgoal_distance', 2.0)
        self.declare_parameter('time_horizon', TIME_HORIZON)
        self.declare_parameter('safety_space', SAFETY_SPACE)
        self.v_max = float(self.get_parameter('v_max').value)
        self.robot_radius = float(self.get_parameter('robot_radius').value)
        self.human_radius = float(self.get_parameter('human_radius').value)
        self.detect_range = float(self.get_parameter('detect_range').value)
        self.target_following_distance = float(self.get_parameter('subgoal_distance').value)
        self.time_horizon = float(self.get_parameter('time_horizon').value)
        self.safety_space = float(self.get_parameter('safety_space').value)

        # ---- subs ----
        self.create_subscription(String, '/command', self.command_callback, 10)
        self.create_subscription(String, '/tracked_objects_json', self.tracked_callback, 10)
        self.create_subscription(String, '/predictions_json', self.predictions_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 20)
        self.create_subscription(Point, '/uwb/tag_0/position', self.uwb_human_callback, 10)
        self.create_subscription(Point, '/uwb/tag_1/position', self.uwb_robot_callback, 10)
        self.create_subscription(Point, '/camera/target_position', self.camera_target_callback, 10)

        # ---- pubs ----
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.action_marker_pub = self.create_publisher(Marker, '/decider_action_marker', 10)
        self.goal_marker_pub = self.create_publisher(Marker, '/decider_goal_marker', 10)

        # ---- tf ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- state ----
        self.current_mode = None
        self.forward_mode = False
        self.tracked_humans = []           # list[(id, x, y, dist)]
        self.predictions = {}              # id -> (vx, vy) approx
        self.target_human_id = None
        self.target_human_position = None
        self.target_human_confirmed = False
        self.confirmation_count = 0
        self.min_confirmations = 3

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.robot_vx = 0.0
        self.robot_vy = 0.0
        self.last_cmd_vel = np.zeros(2, dtype=np.float64)

        # uwb
        self.uwb_human_x = None
        self.uwb_human_y = None
        self.uwb_robot_x = None
        self.uwb_robot_y = None
        self.uwb_last_update = 0.0

        # smoothing — match main.py
        self.max_delta = 0.5

        # 4 Hz control loop
        self.create_timer(DT, self.control_step)

        self.get_logger().info(
            f'[ORCA] SG-ORCA baseline initialized | rvo2={rvo2.__file__} '
            f'v_max={self.v_max} d*={self.target_following_distance} '
            f'tau={self.time_horizon}'
        )

    # ----- callbacks -----
    def command_callback(self, msg: String):
        cmd = msg.data
        if cmd == 'stop':
            self.publish_stop()
            self.current_mode = None
            return
        if cmd.startswith('mode:'):
            self.current_mode = cmd.split(':', 1)[1]
            self.forward_mode = False
            self.publish_stop()
            return
        if self.current_mode not in ('automatic', 'combined'):
            return
        if cmd == 'auto:human_following':
            self.forward_mode = True
            self.get_logger().info('[ORCA] Human following ENABLED.')
        elif cmd.startswith('auto:distance:'):
            try:
                d = float(cmd.split(':')[-1])
                self.target_following_distance = max(1.0, min(6.0, d))
                self.get_logger().info(
                    f'[ORCA] d*={self.target_following_distance:.2f} m'
                )
            except ValueError:
                self.get_logger().error('[ORCA] auto:distance parse error')
        elif cmd == 'auto:reset_target':
            self.reset_target()

    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        rot = msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion(
            (rot.x, rot.y, rot.z, rot.w)
        )
        self.robot_theta = yaw
        # body → global (Fix #1 in main.py)
        vx_b = msg.twist.twist.linear.x
        vy_b = msg.twist.twist.linear.y
        c, s = np.cos(yaw), np.sin(yaw)
        self.robot_vx = c * vx_b - s * vy_b
        self.robot_vy = s * vx_b + c * vy_b

    def uwb_human_callback(self, msg: Point):
        if not (np.isfinite(msg.x) and np.isfinite(msg.y)):
            return
        self.uwb_human_x, self.uwb_human_y = msg.x, msg.y
        self.uwb_last_update = self.get_clock().now().nanoseconds / 1e9

    def uwb_robot_callback(self, msg: Point):
        if not (np.isfinite(msg.x) and np.isfinite(msg.y)):
            return
        self.uwb_robot_x, self.uwb_robot_y = msg.x, msg.y

    def camera_target_callback(self, msg: Point):
        if not (np.isfinite(msg.x) and np.isfinite(msg.y)):
            return
        self.uwb_human_x, self.uwb_human_y = msg.x, msg.y
        self.uwb_last_update = self.get_clock().now().nanoseconds / 1e9

    def tracked_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        tmp = []
        for a in data.get('tracks', []):
            hid = a.get('id', -1)
            if hid < 0:
                continue
            hx = float(a.get('x', 15.0))
            hy = float(a.get('y', 15.0))
            d = float(np.hypot(hx - self.robot_x, hy - self.robot_y))
            if d <= self.detect_range:
                tmp.append((hid, hx, hy, d))
        tmp.sort(key=lambda x: x[3])
        self.tracked_humans = tmp
        self._match_target(tmp)

    def predictions_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        out = {}
        for h in data.get('predictions', []):
            traj = h.get('predicted_trajectory', [])
            if not traj:
                continue
            x0 = traj[0].get('x', 0.0)
            y0 = traj[0].get('y', 0.0)
            x1 = traj[1].get('x', x0) if len(traj) > 1 else x0
            y1 = traj[1].get('y', y0) if len(traj) > 1 else y0
            out[h['id']] = ((x1 - x0) / DT, (y1 - y0) / DT)
        self.predictions = out

    # ----- target matching (mirrors main.py simplified) -----
    def _match_target(self, tracked):
        now = self.get_clock().now().nanoseconds / 1e9
        if (self.uwb_human_x is None or self.uwb_human_y is None
                or now - self.uwb_last_update > 2.0 or not tracked):
            self.confirmation_count = 0
            self.target_human_confirmed = False
            return
        best_id, best_d = None, float('inf')
        for hid, hx, hy, _ in tracked:
            d = np.hypot(hx - self.uwb_human_x, hy - self.uwb_human_y)
            if d < best_d:
                best_d, best_id = d, hid
        if best_d < 1.0:
            self.target_human_id = best_id
            self.confirmation_count = min(
                self.confirmation_count + 1, self.min_confirmations + 1
            )
            if self.confirmation_count >= self.min_confirmations:
                self.target_human_confirmed = True
            for hid, hx, hy, _ in tracked:
                if hid == best_id:
                    self.target_human_position = (hx, hy)
                    return
        else:
            self.confirmation_count = 0
            self.target_human_confirmed = False

    def reset_target(self):
        self.target_human_id = None
        self.target_human_position = None
        self.target_human_confirmed = False
        self.confirmation_count = 0

    # ----- main control loop -----
    def control_step(self):
        if self.current_mode not in ('automatic', 'combined'):
            return
        if not self.forward_mode or not self.target_human_confirmed:
            return
        if self.target_human_position is None:
            return

        tx, ty = self.target_human_position
        rx, ry = self.robot_x, self.robot_y

        # Subgoal: d* metres behind target along robot→target line.
        dx, dy = tx - rx, ty - ry
        dist = float(np.hypot(dx, dy))
        if dist < 1e-3:
            self.publish_stop()
            return
        ux, uy = dx / dist, dy / dist
        sg_x = tx - self.target_following_distance * ux
        sg_y = ty - self.target_following_distance * uy
        self.publish_goal_marker(sg_x, sg_y)

        # Preferred velocity: unit vector toward subgoal at v_max (clipped if
        # already close to subgoal). Matches the training-side ORCA logic at
        # crowd_nav/policy/orca.py:104-107 (unit if speed > 1, else as-is).
        gx, gy = sg_x - rx, sg_y - ry
        gdist = float(np.hypot(gx, gy))
        if gdist < 0.05:
            v_pref = (0.0, 0.0)
        else:
            speed_pref = min(self.v_max, gdist / 0.5)
            v_pref = (gx / gdist * speed_pref, gy / gdist * speed_pref)

        # Build a fresh RVO2 simulator each step (training-side does the same
        # at orca.py:86-96 — re-creates when agent count changes; we re-create
        # unconditionally because tracker IDs/visibility flip frequently).
        # Robot is agent 0; every visible human (target included; ORCA will
        # naturally yield by going around) is an agent.
        humans = list(self.tracked_humans)
        n_humans = len(humans)
        sim = rvo2.PyRVOSimulator(
            DT,
            NEIGHBOR_DIST,
            max(n_humans, 1),         # max_neighbors
            self.time_horizon,
            TIME_HORIZON_OBST,
            self.robot_radius + 0.01 + self.safety_space,
            self.v_max,
        )
        # Robot
        sim.addAgent(
            (rx, ry),
            NEIGHBOR_DIST, max(n_humans, 1),
            self.time_horizon, TIME_HORIZON_OBST,
            self.robot_radius + 0.01 + self.safety_space,
            self.v_max,
            (float(self.robot_vx), float(self.robot_vy)),
        )
        # Humans (use predicted velocity if available; fall back to 0)
        for hid, hx, hy, _ in humans:
            hvx, hvy = self.predictions.get(hid, (0.0, 0.0))
            sim.addAgent(
                (float(hx), float(hy)),
                NEIGHBOR_DIST, max(n_humans, 1),
                self.time_horizon, TIME_HORIZON_OBST,
                self.human_radius + 0.01 + self.safety_space,
                self.v_max,
                (float(hvx), float(hvy)),
            )

        sim.setAgentPrefVelocity(0, (float(v_pref[0]), float(v_pref[1])))
        # Humans get pref_velocity=0 (training-side does this — orca.py:117-118)
        # so RVO2 treats them as goal-less neighbours that won't yield.
        for i in range(n_humans):
            sim.setAgentPrefVelocity(i + 1, (0.0, 0.0))

        sim.doStep()
        v_cmd_global = np.asarray(sim.getAgentVelocity(0), dtype=np.float64)

        # Smoothing identical to main.py.
        delta = v_cmd_global - self.last_cmd_vel
        delta = np.clip(delta, -self.max_delta, self.max_delta)
        v_cmd_global = self.last_cmd_vel + delta
        self.last_cmd_vel = v_cmd_global

        self.publish_velocity_global(float(v_cmd_global[0]), float(v_cmd_global[1]))

    # ----- output helpers -----
    def publish_velocity_global(self, vx_g, vy_g):
        c, s = np.cos(self.robot_theta), np.sin(self.robot_theta)
        local_vx = c * vx_g + s * vy_g
        local_vy = -s * vx_g + c * vy_g
        t = Twist()
        t.linear.x = float(local_vx)
        t.linear.y = float(local_vy)
        self.cmd_vel_pub.publish(t)
        self.publish_action_marker(vx_g, vy_g)

    def publish_stop(self):
        self.cmd_vel_pub.publish(Twist())
        self.last_cmd_vel[:] = 0.0

    def publish_goal_marker(self, gx, gy):
        m = Marker()
        m.header.frame_id = 'odom'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'decider_goal'
        m.id = 1
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.scale.x = 0.6
        m.scale.y = 0.6
        m.scale.z = 0.02
        m.color = ColorRGBA(r=0.2, g=1.0, b=0.4, a=0.6)
        m.pose.position.x = float(gx)
        m.pose.position.y = float(gy)
        m.pose.orientation.w = 1.0
        self.goal_marker_pub.publish(m)

    def publish_action_marker(self, vx_g, vy_g):
        mag = float(np.hypot(vx_g, vy_g))
        m = Marker()
        m.header.frame_id = 'odom'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'decider_action'
        m.id = 0
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale.x = mag
        m.scale.y = 0.1
        m.scale.z = 0.1
        m.color = ColorRGBA(r=0.2, g=1.0, b=0.4, a=1.0)
        m.pose.position.x = self.robot_x
        m.pose.position.y = self.robot_y
        theta = float(np.arctan2(vy_g, vx_g)) if mag > 1e-3 else 0.0
        q = tf_transformations.quaternion_from_euler(0, 0, theta)
        m.pose.orientation.x = q[0]
        m.pose.orientation.y = q[1]
        m.pose.orientation.z = q[2]
        m.pose.orientation.w = q[3]
        self.action_marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = OrcaFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
