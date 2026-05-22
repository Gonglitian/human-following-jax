#!/usr/bin/env python3
"""
detections_merger

Gazebo <actor> (what HuNavSim spawns) ships no <collision> geometry, so the
simulated 2D LiDAR never hits pedestrian legs and DR-SPAAM's NN rejects
every raw scan at conf_thresh 0.5. The only thing that makes it through
into /dr_spaam_detections is the UWB pseudo-human (== target), leaving
sort_tracker with no bystander tracks in crowd/junction scenarios.

This node fuses:
  - /dr_spaam_detections (PoseArray, odom frame) — whatever DR-SPAAM
    actually produces (typically just the UWB-injected target)
  - /human_states (hunav_msgs/Agents, map frame) — ground-truth pose of
    every HuNav actor, transformed into odom

into a single /combined_detections PoseArray. sort_tracker should be
pointed at this topic instead of /dr_spaam_detections when running in sim.

Deduplication: if a HuNav agent is already within `dedup_radius` of a
DR-SPAAM detection (typically the target, already seen via UWB), the HuNav
copy is dropped so SORT doesn't oscillate between two IDs for the same
person.
"""
import math

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

import tf2_ros
from tf2_ros import TransformException

from geometry_msgs.msg import Pose, PoseArray
from hunav_msgs.msg import Agents


class DetectionsMerger(Node):
    def __init__(self):
        super().__init__('detections_merger')
        self.declare_parameter('dr_spaam_topic', '/dr_spaam_detections')
        self.declare_parameter('human_states_topic', '/human_states')
        self.declare_parameter('output_topic', '/combined_detections')
        self.declare_parameter('source_frame', 'map')
        self.declare_parameter('output_frame', 'odom')
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('dedup_radius', 0.6)  # m

        self.src_frame = self.get_parameter('source_frame').value
        self.out_frame = self.get_parameter('output_frame').value
        self.dedup_radius = float(self.get_parameter('dedup_radius').value)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._last_drspaam = None        # list of (x, y) in odom
        self._last_drspaam_hdr = None    # std_msgs/Header of most recent PoseArray
        self._last_humans = None         # list of (x, y) in odom

        self._pub = self.create_publisher(
            PoseArray, self.get_parameter('output_topic').value, 10)
        self.create_subscription(
            PoseArray, self.get_parameter('dr_spaam_topic').value,
            self._on_drspaam, 10)
        self.create_subscription(
            Agents, self.get_parameter('human_states_topic').value,
            self._on_humans, 10)

        rate = float(self.get_parameter('publish_rate_hz').value)
        self._timer = self.create_timer(1.0 / max(rate, 1e-3), self._publish_tick)

        self.get_logger().info(
            f"[detections_merger] merging {self.get_parameter('dr_spaam_topic').value} + "
            f"{self.get_parameter('human_states_topic').value} -> "
            f"{self.get_parameter('output_topic').value}"
        )

    def _on_drspaam(self, msg: PoseArray):
        self._last_drspaam_hdr = msg.header
        self._last_drspaam = [(p.position.x, p.position.y) for p in msg.poses]

    def _map_to_odom(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                self.out_frame, self.src_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.05))
        except TransformException:
            return None
        q = tf.transform.rotation
        t = tf.transform.translation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp), (t.x, t.y)

    def _on_humans(self, msg: Agents):
        tf = self._map_to_odom()
        if tf is None:
            # treat map == odom at startup before first TF is cached
            yaw, (tx, ty) = 0.0, (0.0, 0.0)
        else:
            yaw, (tx, ty) = tf
        c, s = math.cos(yaw), math.sin(yaw)
        xs = []
        for a in msg.agents:
            x = a.position.position.x
            y = a.position.position.y
            xo = c * x - s * y + tx
            yo = s * x + c * y + ty
            xs.append((xo, yo))
        self._last_humans = xs

    def _publish_tick(self):
        if self._last_drspaam is None and self._last_humans is None:
            return

        out = PoseArray()
        if self._last_drspaam_hdr is not None:
            out.header = self._last_drspaam_hdr
        else:
            out.header.frame_id = self.out_frame
            out.header.stamp = self.get_clock().now().to_msg()

        detected = []  # (x, y) tuples

        # 1) pass through everything DR-SPAAM already gave us (includes UWB target)
        if self._last_drspaam:
            detected.extend(self._last_drspaam)

        # 2) add HuNav bystanders, skipping anyone already within dedup_radius of a
        #    DR-SPAAM detection (so the target counted once, not twice)
        if self._last_humans:
            r2 = self.dedup_radius * self.dedup_radius
            for (hx, hy) in self._last_humans:
                dup = False
                for (dx, dy) in detected:
                    if (hx - dx) ** 2 + (hy - dy) ** 2 < r2:
                        dup = True
                        break
                if not dup:
                    detected.append((hx, hy))

        for (x, y) in detected:
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            p.position.z = 0.0
            p.orientation.w = 1.0
            out.poses.append(p)

        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionsMerger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
