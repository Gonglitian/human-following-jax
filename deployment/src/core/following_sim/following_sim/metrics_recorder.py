#!/usr/bin/env python3
"""
metrics_recorder

Complements hunav_evaluator (which tracks social-navigation metrics) with a
following-specific metric: the error between actual robot<->target distance
and the distance prescribed by the current following_preference.

Output: CSV at <output_dir>/<run_name>.csv with columns
    t_sec, robot_x, robot_y, target_x, target_y, distance,
    preference, desired_distance, distance_error, linear_vel, angular_vel
"""
import csv
import math
import os
from pathlib import Path
from datetime import datetime

import math
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from hunav_msgs.msg import Agents


# Mirror of decider's PController.preference_to_distance map (main.py).
# Keeping a local copy avoids importing the decider package just for one dict.
PREFERENCE_TO_DISTANCE = {
    -2: 1.37,
    -1: 1.90,
     0: 2.29,
     1: 3.31,
     2: 3.80,
}


class MetricsRecorder(Node):
    def __init__(self):
        super().__init__('metrics_recorder')

        self.declare_parameter('target_name', 'target')
        self.declare_parameter('target_id', -1)
        self.declare_parameter('output_dir', '/tmp/following_sim_metrics')
        self.declare_parameter('run_name', '')
        self.declare_parameter('sample_rate_hz', 10.0)
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('human_states_topic', '/human_states')
        self.declare_parameter('command_topic', '/command')

        self.target_name = self.get_parameter('target_name').value
        self.target_id = int(self.get_parameter('target_id').value)

        output_dir = Path(self.get_parameter('output_dir').value)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = self.get_parameter('run_name').value or \
                   f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._csv_path = output_dir / f"{run_name}.csv"

        self._csv_file = open(self._csv_path, 'w', newline='')
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow([
            't_sec', 'robot_x', 'robot_y', 'target_x', 'target_y',
            'distance', 'preference', 'desired_distance', 'distance_error',
            'linear_vel', 'angular_vel',
            # paper-style safety metrics (per timestep):
            #   min_human_dist  = robot to nearest of 9 agents (collision if < 0.4m)
            #   min_obs_dist    = robot to nearest LiDAR return after subtracting
            #                     points near humans (collision if < 0.3m)
            'min_human_dist', 'min_obs_dist',
        ])

        self._robot_xy = None
        self._robot_vel = (0.0, 0.0)
        self._robot_yaw = 0.0
        self._target_xy = None
        self._current_pref = 0  # default = 0 -> 2.29 m
        self._all_humans = []   # list of (x, y) for all 9 agents (map frame)
        self._scan = None       # latest LaserScan

        self.create_subscription(Odometry,
                                 self.get_parameter('odom_topic').value,
                                 self._on_odom, 10)
        self.create_subscription(Agents,
                                 self.get_parameter('human_states_topic').value,
                                 self._on_agents, 10)
        self.create_subscription(LaserScan, '/scan', self._on_scan, 10)
        self.create_subscription(String,
                                 self.get_parameter('command_topic').value,
                                 self._on_command, 10)

        rate = float(self.get_parameter('sample_rate_hz').value)
        self._t0 = self.get_clock().now()
        self._timer = self.create_timer(1.0 / max(rate, 1e-3), self._tick)

        self.get_logger().info(
            f"[metrics_recorder] writing {self._csv_path} @ {rate:.1f} Hz"
        )

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        self._robot_xy = (p.x, p.y)
        # yaw from quaternion (z-axis only — planar)
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._robot_yaw = math.atan2(siny, cosy)
        v = msg.twist.twist
        self._robot_vel = (
            math.hypot(v.linear.x, v.linear.y),
            v.angular.z,
        )

    def _on_agents(self, msg):
        humans = []
        for a in msg.agents:
            humans.append((a.position.position.x, a.position.position.y))
            if (self.target_id >= 0 and a.id == self.target_id) or \
               (self.target_id < 0 and a.name == self.target_name):
                self._target_xy = (a.position.position.x,
                                   a.position.position.y)
        self._all_humans = humans

    def _on_scan(self, msg):
        self._scan = msg

    def _on_command(self, msg):
        # Match the subset of /command strings the decider handles:
        #   "auto:preference:<int>"      -> sets preference directly
        #   "auto:distance:<float>"      -> decider maps via PController; we
        #                                   approximate by picking the nearest
        #                                   preference bucket.
        data = msg.data.strip()
        if data.startswith('auto:preference:'):
            try:
                self._current_pref = int(data.split(':')[2])
            except (ValueError, IndexError):
                pass
        elif data.startswith('auto:distance:'):
            try:
                target_d = float(data.split(':')[2])
            except (ValueError, IndexError):
                return
            self._current_pref = min(
                PREFERENCE_TO_DISTANCE.keys(),
                key=lambda k: abs(PREFERENCE_TO_DISTANCE[k] - target_d),
            )

    def _tick(self):
        if self._robot_xy is None or self._target_xy is None:
            return
        rx, ry = self._robot_xy
        tx, ty = self._target_xy
        dist = math.hypot(tx - rx, ty - ry)
        desired = PREFERENCE_TO_DISTANCE.get(self._current_pref, 2.29)
        err = dist - desired
        t = (self.get_clock().now() - self._t0).nanoseconds * 1e-9
        lv, av = self._robot_vel

        # HCR proxy: min distance from robot to any of the 9 hunav agents
        min_h = float('nan')
        if self._all_humans:
            min_h = min(math.hypot(h[0] - rx, h[1] - ry) for h in self._all_humans)

        # OCR proxy: min LiDAR return after subtracting points that fall within
        # 0.4m of any tracked human (so we don't double-count humans as obstacles).
        min_o = float('nan')
        if self._scan is not None:
            ranges = self._scan.ranges
            a0 = self._scan.angle_min
            da = self._scan.angle_increment
            r_min = self._scan.range_min
            r_max = self._scan.range_max
            cy, sy = math.cos(self._robot_yaw), math.sin(self._robot_yaw)
            min_dist = float('inf')
            n = len(ranges)
            # subsample every 3 rays for speed; A1 has ~360+ rays
            for i in range(0, n, 3):
                r = ranges[i]
                if not (r_min < r < r_max) or math.isnan(r) or math.isinf(r):
                    continue
                ang = a0 + i * da
                # laser-frame point
                lx = r * math.cos(ang)
                ly = r * math.sin(ang)
                # rotate to map frame (laser ≈ base_link for our setup)
                wx = rx + cy * lx - sy * ly
                wy = ry + sy * lx + cy * ly
                # skip if near any human
                is_human = False
                for hx, hy in self._all_humans:
                    if (wx - hx) ** 2 + (wy - hy) ** 2 < 0.16:  # 0.4m radius
                        is_human = True
                        break
                if is_human:
                    continue
                if r < min_dist:
                    min_dist = r
            if math.isfinite(min_dist):
                min_o = min_dist

        self._writer.writerow([
            f"{t:.3f}", f"{rx:.3f}", f"{ry:.3f}", f"{tx:.3f}", f"{ty:.3f}",
            f"{dist:.3f}", self._current_pref, f"{desired:.3f}", f"{err:.3f}",
            f"{lv:.3f}", f"{av:.3f}",
            f"{min_h:.3f}", f"{min_o:.3f}",
        ])
        self._csv_file.flush()

    def destroy_node(self):
        try:
            self._csv_file.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MetricsRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
