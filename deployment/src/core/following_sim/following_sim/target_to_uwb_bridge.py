#!/usr/bin/env python3
"""
target_to_uwb_bridge

HuNavSim publishes every human agent on /human_states (hunav_msgs/Agents) in
the global frame (default: map). Decider consumes the 'target' human via
/uwb/tag_0/position (geometry_msgs/Point, odom frame) through DR-SPAAM's
UWB pseudo-human injection path.

This bridge finds the HuNav agent matching the configured target name,
transforms its pose from map into odom, and republishes it as a Point. That
way the decider sees the simulated target with zero code change.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import tf2_ros
from tf2_ros import TransformException

from geometry_msgs.msg import Point
from hunav_msgs.msg import Agents


class TargetToUwbBridge(Node):
    def __init__(self):
        super().__init__('target_to_uwb_bridge')

        self.declare_parameter('target_name', 'target')
        self.declare_parameter('target_id', -1)
        self.declare_parameter('source_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('human_states_topic', '/human_states')
        self.declare_parameter('uwb_topic', '/uwb/tag_0/position')
        self.declare_parameter('publish_rate_hz', 10.0)

        self.target_name = self.get_parameter('target_name').value
        self.target_id = int(self.get_parameter('target_id').value)
        self.source_frame = self.get_parameter('source_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        human_topic = self.get_parameter('human_states_topic').value
        uwb_topic = self.get_parameter('uwb_topic').value
        rate = float(self.get_parameter('publish_rate_hz').value)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._pub = self.create_publisher(Point, uwb_topic, 10)
        self._sub = self.create_subscription(
            Agents, human_topic, self._on_agents, 10
        )

        # Decouple incoming /human_states rate (can be 100+ Hz) from the UWB
        # rate real hardware gives us (~10 Hz). Hold last resolved odom-frame
        # target position and re-publish on a timer.
        self._last_point_odom = None
        self._timer = self.create_timer(1.0 / max(rate, 1e-3), self._publish_tick)

        self.get_logger().info(
            f"[target_to_uwb_bridge] target='{self.target_name}' id={self.target_id} "
            f"{self.source_frame}->{self.odom_frame} -> {uwb_topic} @ {rate:.1f} Hz"
        )

    def _select_target(self, agents):
        """Return the Agent matching id (if >=0) or name. None if absent."""
        if self.target_id >= 0:
            for a in agents:
                if a.id == self.target_id:
                    return a
        if self.target_name:
            for a in agents:
                if a.name == self.target_name:
                    return a
        return None

    def _on_agents(self, msg):
        target = self._select_target(msg.agents)
        if target is None:
            return

        x_src = target.position.position.x
        y_src = target.position.position.y
        z_src = target.position.position.z

        # Transform (x, y) from source_frame (map) into odom_frame.
        try:
            tf = self._tf_buffer.lookup_transform(
                self.odom_frame,          # target
                self.source_frame,        # source
                rclpy.time.Time(),        # latest
                timeout=Duration(seconds=0.05),
            )
        except TransformException as ex:
            self.get_logger().warn(
                f"[target_to_uwb_bridge] TF {self.source_frame}->{self.odom_frame} "
                f"unavailable: {ex}; passing pose through unchanged",
                throttle_duration_sec=2.0,
            )
            # Without TF, source and odom are assumed coincident. Better than
            # dropping the message, since at t=0 they usually are.
            self._last_point_odom = (x_src, y_src, z_src)
            return

        tx = tf.transform.translation
        rot = tf.transform.rotation
        # Rotate (x, y) by yaw extracted from quaternion (planar only).
        # qw*qz terms dominate for a yaw-only rotation, which is typical for
        # a static map->odom transform.
        siny_cosp = 2.0 * (rot.w * rot.z + rot.x * rot.y)
        cosy_cosp = 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        cy, sy = math.cos(yaw), math.sin(yaw)
        xo = cy * x_src - sy * y_src + tx.x
        yo = sy * x_src + cy * y_src + tx.y
        zo = z_src + tx.z
        self._last_point_odom = (xo, yo, zo)

    def _publish_tick(self):
        if self._last_point_odom is None:
            return
        x, y, z = self._last_point_odom
        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = float(z)
        self._pub.publish(p)


def main(args=None):
    rclpy.init(args=args)
    node = TargetToUwbBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
