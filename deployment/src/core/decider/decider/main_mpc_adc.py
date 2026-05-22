#!/usr/bin/env python3
"""
MPC-ADC baseline (paper §V.B): MPC with Adaptive Distance Constraint.

Thin ROS 2 wrapper around the paper's reference MPC implementation lifted
verbatim from
  github.com/tasl-lab/human-following-robot @ baseline_mpc_orca
into `decider/baselines/mpc_lifted.py`. Solver: CasADi + IPOPT (with
warm-start + 0.2s CPU-time limit per call). Cost terms: distance to goal,
preference-distance to target, control effort, soft repulsion against
predicted humans, soft repulsion against OGM-derived obstacle points.

ADC = Adaptive Distance Constraint: every time we receive
`auto:distance:<d>`, we call `mpc.set_preference_distance(d)`. The MPC
formulation already has a preference_distance soft constraint
(weight `w_preference`) so this gives the dynamic-d behavior described
in paper §V.B for the meta-policy comparison.

Subscriptions / command API match decider.main and main_orca, so the
same launch and the same /command sequence drive all three baselines.
"""

import json
import time
from types import SimpleNamespace

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, ColorRGBA
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

import tf2_ros
import tf_transformations

from decider.baselines.mpc_lifted import MPC


class MpcAdcFollower(Node):
    def __init__(self):
        super().__init__('decider_node')

        # ---- params ----
        self.declare_parameter('v_max', 1.0)
        self.declare_parameter('robot_radius', 0.3)
        self.declare_parameter('human_radius', 0.35)
        # Effectively unbounded — see decider.main for rationale.
        self.declare_parameter('detect_range', 50.0)
        self.declare_parameter('target_following_distance', 2.0)
        self.declare_parameter('horizon', 5)
        # Solve budget per call (s). Lifted MPC default is 0.2 s; keep it
        # below DT so /cmd_vel never starves.
        self.declare_parameter('solver_timeout', 0.20)
        # MPC cost weights — defaults match the source branch defaults.
        self.declare_parameter('w_goal', 1.0)
        self.declare_parameter('w_control', 0.1)
        self.declare_parameter('w_human', 100.0)
        self.declare_parameter('w_obstacle', 150.0)
        self.declare_parameter('w_preference', 5.0)
        self.declare_parameter('safety_margin', 0.5)
        self.declare_parameter('obstacle_safety_margin', 0.3)
        self.declare_parameter('preference_tolerance', 0.2)

        gp = self.get_parameter
        self.v_max = float(gp('v_max').value)
        self.robot_radius = float(gp('robot_radius').value)
        self.human_radius = float(gp('human_radius').value)
        self.detect_range = float(gp('detect_range').value)
        self.target_following_distance = float(gp('target_following_distance').value)
        self.horizon = int(gp('horizon').value)
        self.solver_timeout = float(gp('solver_timeout').value)

        # Build a SimpleNamespace config that the lifted MPC class understands.
        # Mirrors the keys it reads via `getattr` in __init__.
        mpc_cfg = SimpleNamespace(
            horizon=self.horizon,
            max_speed=self.v_max,
            w_goal=float(gp('w_goal').value),
            w_control=float(gp('w_control').value),
            w_human=float(gp('w_human').value),
            w_obstacle=float(gp('w_obstacle').value),
            w_preference=float(gp('w_preference').value),
            safety_margin=float(gp('safety_margin').value),
            obstacle_safety_margin=float(gp('obstacle_safety_margin').value),
            solver_timeout=self.solver_timeout,
        )
        self.dt_plan = 0.25  # planning step matches our 4 Hz control loop
        cfg = SimpleNamespace(
            env=SimpleNamespace(time_step=self.dt_plan),
            robot=SimpleNamespace(radius=self.robot_radius),
            mpc=mpc_cfg,
            preference_distance=self.target_following_distance,
            preference_tolerance=float(gp('preference_tolerance').value),
        )
        self.mpc = MPC(cfg)

        # ---- subs ----
        self.create_subscription(String, '/command', self.command_callback, 10)
        self.create_subscription(String, '/tracked_objects_json', self.tracked_callback, 10)
        self.create_subscription(String, '/predictions_json', self.predictions_callback, 10)
        self.create_subscription(String, '/occupancy_grid_json', self.occ_callback, 10)
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
        self.tracked_humans = []   # list[(id, x, y, dist)]
        # Predictions: id -> list[(x, y)] in odom frame, len = predict_steps.
        self.predictions = {}
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

        # occupancy grid (robot-centred 50x50 @ 0.2 m). Lifted MPC's
        # _get_obstacle_points_from_ogm assumes this exact convention.
        self.ogm = None

        # smoothing — same as decider.main
        self.max_delta = 0.5

        self.create_timer(self.dt_plan, self.control_step)
        self.get_logger().info(
            f'[MPC-ADC] initialized | v_max={self.v_max} '
            f'd*={self.target_following_distance} horizon={self.horizon} '
            f'solver_timeout={self.solver_timeout}'
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
            self.get_logger().info('[MPC-ADC] Human following ENABLED.')
        elif cmd.startswith('auto:distance:'):
            try:
                d = float(cmd.split(':')[-1])
                self.target_following_distance = max(1.0, min(6.0, d))
                # ADC: forward to MPC's preference-distance soft constraint.
                self.mpc.set_preference_distance(self.target_following_distance)
                self.get_logger().info(
                    f'[MPC-ADC] preference_distance ← {self.target_following_distance:.2f} m'
                )
            except ValueError:
                self.get_logger().error('[MPC-ADC] auto:distance parse error')
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
            traj = [(p.get('x', 0.0), p.get('y', 0.0))
                    for p in h.get('predicted_trajectory', [])]
            out[h['id']] = traj
        self.predictions = out

    def occ_callback(self, msg: String):
        try:
            grid_dict = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        info = grid_dict.get('info', {})
        w = info.get('width', 0)
        h = info.get('height', 0)
        data_flat = grid_dict.get('data', [])
        if w == 0 or h == 0 or len(data_flat) != w * h:
            return
        # Lifted MPC expects 0/1 binary, robot-centred. Threshold any value > 0.
        self.ogm = (np.array(data_flat, dtype=np.int8).reshape((h, w)) > 0).astype(np.int8)

    # ----- target matching -----
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

    # ----- assemble the State object the lifted MPC consumes -----
    def _build_state(self):
        if self.target_human_position is None:
            return None

        # self_state — robot kinematics + goal (= target's current position;
        # the MPC's preference-distance soft constraint will pull it back to
        # d* metres behind, so we do NOT pre-offset here).
        tx, ty = self.target_human_position
        self_state = SimpleNamespace(
            px=float(self.robot_x), py=float(self.robot_y),
            vx=float(self.robot_vx), vy=float(self.robot_vy),
            gx=float(tx), gy=float(ty),
            radius=self.robot_radius,
            sensor_range=self.detect_range,
        )

        # human_states & human_future_traj.
        # Lifted MPC expects future_traj shape [T+1, num_humans, 4] where
        # T = horizon. The first index t=0 is the current position; t>0 are
        # predicted positions. Final dim is (x, y, vx, vy) — but the MPC
        # only uses the first two coords (see _get_human_trajectories).
        T = self.horizon + 1
        humans = list(self.tracked_humans)
        human_states = []
        future_traj = np.zeros((T, max(len(humans), 1), 4), dtype=np.float32)

        for h_idx, (hid, hx, hy, _) in enumerate(humans):
            human_states.append(SimpleNamespace(
                px=float(hx), py=float(hy),
                vx=0.0, vy=0.0,
                radius=self.human_radius,
            ))
            traj = self.predictions.get(hid, [])
            future_traj[0, h_idx, 0] = hx
            future_traj[0, h_idx, 1] = hy
            for t in range(1, T):
                if t - 1 < len(traj):
                    px, py = traj[t - 1]
                else:
                    # Hold last known position if predictions run short.
                    last = traj[-1] if traj else (hx, hy)
                    px, py = last
                future_traj[t, h_idx, 0] = px
                future_traj[t, h_idx, 1] = py

        # Edge case: no visible humans. We still need a non-empty traj to
        # avoid downstream divide-by-zero; fill a far-away dummy.
        if not humans:
            future_traj[:, 0, 0] = 1e3
            future_traj[:, 0, 1] = 1e3
            human_states.append(SimpleNamespace(
                px=1e3, py=1e3, vx=0.0, vy=0.0, radius=self.human_radius
            ))

        return SimpleNamespace(
            self_state=self_state,
            human_states=human_states,
            human_future_traj=future_traj,
            ogm=self.ogm if self.ogm is not None else None,
            preference_distance=self.target_following_distance,
        )

    # ----- main control loop -----
    def control_step(self):
        if self.current_mode not in ('automatic', 'combined'):
            return
        if not self.forward_mode or not self.target_human_confirmed:
            return
        state = self._build_state()
        if state is None:
            return

        try:
            action = self.mpc.predict(state)  # ActionXY(vx, vy)
            v_cmd_global = np.array([float(action.vx), float(action.vy)])
        except Exception as ex:
            self.get_logger().error(f'[MPC-ADC] solver exception: {ex}')
            v_cmd_global = np.zeros(2)

        # Slew + speed limit (matches main.py).
        delta = v_cmd_global - self.last_cmd_vel
        delta = np.clip(delta, -self.max_delta, self.max_delta)
        v_cmd_global = self.last_cmd_vel + delta
        speed = np.linalg.norm(v_cmd_global)
        if speed > self.v_max:
            v_cmd_global = v_cmd_global * (self.v_max / speed)
        self.last_cmd_vel = v_cmd_global

        # Subgoal marker = d* metres behind target along robot→target line
        # (visualization only; MPC handles the geometry internally).
        tx, ty = self.target_human_position
        ux = tx - self.robot_x
        uy = ty - self.robot_y
        norm = np.hypot(ux, uy) + 1e-6
        sg_x = tx - self.target_following_distance * ux / norm
        sg_y = ty - self.target_following_distance * uy / norm
        self.publish_goal_marker(sg_x, sg_y)

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
        m.color = ColorRGBA(r=0.4, g=0.6, b=1.0, a=0.6)
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
        m.color = ColorRGBA(r=0.4, g=0.6, b=1.0, a=1.0)
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
    node = MpcAdcFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
