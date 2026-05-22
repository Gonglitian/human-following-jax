#!/usr/bin/env python3
"""
human_states_viz

HuNavSim publishes pedestrian ground truth on /human_states (hunav_msgs/Agents)
but has no RViz display. Convert to a MarkerArray so we can eyeball every
actor in RViz alongside what the decider actually sees (via /dr_spaam_rviz
+ /tracked_objects_viz). The "target" agent gets a distinct colour so the
person the policy should be tracking pops out.
"""
import math

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

import tf2_ros
from tf2_ros import TransformException

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from hunav_msgs.msg import Agents


TARGET_COLOR = ColorRGBA(r=0.0, g=0.95, b=0.3, a=0.85)      # bright green
BYSTANDER_COLOR = ColorRGBA(r=0.95, g=0.35, b=0.0, a=0.85)  # amber
RADIUS = 0.35  # m — matches `radius` in agents_*.yaml
HEIGHT = 1.7   # m — visual cylinder height for actors


def _rotate_xy(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * x - s * y, s * x + c * y


class HumanStatesViz(Node):
    def __init__(self):
        super().__init__('human_states_viz')
        self.declare_parameter('target_name', 'target')
        self.declare_parameter('target_id', -1)
        self.declare_parameter('source_frame', 'map')
        self.declare_parameter('output_frame', 'odom')
        self.declare_parameter('input_topic', '/human_states')
        self.declare_parameter('output_topic', '/human_states_viz')

        self.target_name = self.get_parameter('target_name').value
        self.target_id = int(self.get_parameter('target_id').value)
        self.src_frame = self.get_parameter('source_frame').value
        self.out_frame = self.get_parameter('output_frame').value

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._pub = self.create_publisher(
            MarkerArray, self.get_parameter('output_topic').value, 10)
        self._sub = self.create_subscription(
            Agents, self.get_parameter('input_topic').value, self._on_agents, 10)

        self.get_logger().info(
            f"[human_states_viz] subscribing {self.get_parameter('input_topic').value}, "
            f"publishing {self.get_parameter('output_topic').value} "
            f"in frame '{self.out_frame}'"
        )

    def _get_yaw_and_trans(self):
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
        return math.atan2(siny_cosp, cosy_cosp), (t.x, t.y, t.z)

    def _on_agents(self, msg):
        tfinfo = self._get_yaw_and_trans()
        yaw, trans = (0.0, (0.0, 0.0, 0.0)) if tfinfo is None else tfinfo

        markers = MarkerArray()

        # a leading DELETEALL resets namespaces so vanished agents disappear
        clear = Marker()
        clear.header.frame_id = self.out_frame
        clear.header.stamp = msg.header.stamp
        clear.ns = 'human_states_viz'
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for a in msg.agents:
            is_target = (
                (self.target_id >= 0 and a.id == self.target_id)
                or (self.target_id < 0 and a.name == self.target_name)
            )
            color = TARGET_COLOR if is_target else BYSTANDER_COLOR

            x_src = a.position.position.x
            y_src = a.position.position.y
            xo, yo = _rotate_xy(x_src, y_src, yaw)
            xo += trans[0]
            yo += trans[1]

            # capsule body
            body = Marker()
            body.header.frame_id = self.out_frame
            body.header.stamp = msg.header.stamp
            body.ns = 'human_states_viz'
            body.id = 2 * int(a.id)
            body.type = Marker.CYLINDER
            body.action = Marker.ADD
            body.pose.position.x = float(xo)
            body.pose.position.y = float(yo)
            body.pose.position.z = HEIGHT / 2.0
            body.pose.orientation.w = 1.0
            body.scale.x = 2 * RADIUS
            body.scale.y = 2 * RADIUS
            body.scale.z = HEIGHT
            body.color = color
            markers.markers.append(body)

            # name / id label floating above the head
            label = Marker()
            label.header.frame_id = self.out_frame
            label.header.stamp = msg.header.stamp
            label.ns = 'human_states_viz'
            label.id = 2 * int(a.id) + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(xo)
            label.pose.position.y = float(yo)
            label.pose.position.z = HEIGHT + 0.25
            label.pose.orientation.w = 1.0
            label.scale.z = 0.25
            label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            label.text = f"{a.name}#{int(a.id)}{'*' if is_target else ''}"
            markers.markers.append(label)

        self._pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = HumanStatesViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
