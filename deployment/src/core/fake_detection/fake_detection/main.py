#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import Point, Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header


class Person:
    """
    Simple container to hold a "person"'s position and velocity.
    Bounces within a given bounding box.
    """
    def __init__(self, x, y, vx, vy, min_xy=-3.0, max_xy=3.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.min_xy = min_xy
        self.max_xy = max_xy

    def update(self):
        # Update position
        self.x += self.vx
        self.y += self.vy

        # Check boundaries and reverse velocity if needed
        if self.x < self.min_xy or self.x > self.max_xy:
            self.vx = -self.vx
        if self.y < self.min_xy or self.y > self.max_xy:
            self.vy = -self.vy


class FakeDetectionNode(Node):
    """
    A fake detection node that publishes detections (PoseArray and MarkerArray)
    mimicking the format of the DrSpaamROS node, but without subscribing to
    LaserScan or using any real detector.
    """
    def __init__(self):
        super().__init__('fake_detection')

        # Declare the same parameters as the original node (some are unused here).
        self.declare_parameter('publisher.detections.topic', '/dr_spaam_detections')
        self.declare_parameter('publisher.detections.queue_size', 10)
        self.declare_parameter('publisher.detections.latch', False)

        self.declare_parameter('publisher.rviz.topic', '/dr_spaam_rviz')
        self.declare_parameter('publisher.rviz.queue_size', 10)
        self.declare_parameter('publisher.rviz.latch', False)

        # Read the relevant parameters
        self.det_topic = self.get_parameter('publisher.detections.topic').value
        self.det_qsize = self.get_parameter('publisher.detections.queue_size').value
        self.det_latch = self.get_parameter('publisher.detections.latch').value

        self.rviz_topic = self.get_parameter('publisher.rviz.topic').value
        self.rviz_qsize = self.get_parameter('publisher.rviz.queue_size').value
        self.rviz_latch = self.get_parameter('publisher.rviz.latch').value

        # Set up QoS
        det_qos = QoSProfile(depth=self.det_qsize)
        if self.det_latch:
            det_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        rviz_qos = QoSProfile(depth=self.rviz_qsize)
        if self.rviz_latch:
            rviz_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        # Publishers
        self._dets_pub = self.create_publisher(PoseArray, self.det_topic, det_qos)
        self._rviz_pub = self.create_publisher(MarkerArray, self.rviz_topic, rviz_qos)

        # Create a timer to publish at 10 Hz
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self._timer_callback)

        # Create some random persons
        np.random.seed(42)
        self.persons = []
        for _ in range(6):
            x = np.random.uniform(-2.0, 2.0)
            y = np.random.uniform(-2.0, 2.0)
            vx = np.random.uniform(-0.5, 0.5)
            vy = np.random.uniform(-0.5, 0.5)
            self.persons.append(Person(x, y, vx, vy, min_xy=-3.0, max_xy=3.0))

        self.get_logger().info("Fake Detection Node started.")

    def _timer_callback(self):
        """
        Periodically update person positions and publish PoseArray and MarkerArray.
        """
        # Update all persons
        for p in self.persons:
            p.update()

        # Convert to NumPy arrays for convenience
        dets_xy = np.array([[p.x, p.y] for p in self.persons])
        # (Optional) use random or constant 'confidence'
        dets_cls = np.array([1.0 for _ in self.persons])
        radii = [0.4 for _ in self.persons]

        # Prepare a header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'odom'  # or any relevant frame you want

        # Clear old markers before publishing new ones (same approach as original)
        self._clear_old_markers(header)

        # Publish new PoseArray
        pose_array_msg = detections_to_pose_array(dets_xy, dets_cls)
        pose_array_msg.header = header
        self._dets_pub.publish(pose_array_msg)

        # Publish new MarkerArray
        rviz_msg = detections_to_rviz_marker(dets_xy, dets_cls, header, radii=radii)
        self._rviz_pub.publish(rviz_msg)

    def _clear_old_markers(self, header):
        """Publish a DELETEALL Marker to clear previously displayed markers."""
        clear_msg = MarkerArray()
        clear_marker = Marker()
        clear_marker.header = header
        clear_marker.ns = "dr_spaam_ros"
        clear_marker.id = 0
        clear_marker.action = Marker.DELETEALL
        clear_msg.markers.append(clear_marker)
        self._rviz_pub.publish(clear_msg)


def detections_to_pose_array(dets_xy, dets_cls):
    """
    Create a PoseArray from the given detection points.
    We ignore confidence here except to maintain the function signature consistency.
    """
    pose_array = PoseArray()
    # If no detections, optionally add a dummy pose below ground (mimicking the original code).
    if len(dets_xy) == 0:
        p = Pose()
        p.position.x = 0.0
        p.position.y = 0.0
        p.position.z = -1.0
        pose_array.poses.append(p)
    else:
        for d_xy in dets_xy:
            p = Pose()
            p.position.x = float(d_xy[0])
            p.position.y = float(d_xy[1])
            p.position.z = 0.0
            pose_array.poses.append(p)
    return pose_array


def detections_to_rviz_marker(dets_xy, dets_cls, header, radii=None):
    """
    Create a MarkerArray for RViz visualization (circles plus text).
    Largely follows the logic from the original detection code.
    """
    if radii is None:
        radii = [0.4]*len(dets_xy)

    markers = MarkerArray()
    for idx, ((x, y), d_cls, r) in enumerate(zip(dets_xy, dets_cls, radii)):
        # Draw a circle (LINE_STRIP) around the detection
        circle_marker = Marker()
        circle_marker.header = header
        circle_marker.ns = "dr_spaam_ros"
        circle_marker.id = idx * 2
        circle_marker.type = Marker.LINE_STRIP
        circle_marker.action = Marker.ADD
        circle_marker.pose.orientation.w = 1.0
        circle_marker.scale.x = 0.03  # line thickness
        circle_marker.color.r = 1.0
        circle_marker.color.a = 1.0

        ang = np.linspace(0, 2 * math.pi, 20)
        xy_offsets = r * np.stack((np.cos(ang), np.sin(ang)), axis=1)

        for i in range(len(xy_offsets)):
            p = Point()
            p.x = float(x + xy_offsets[i, 0])
            p.y = float(y + xy_offsets[i, 1])
            p.z = 0.0
            circle_marker.points.append(p)
        # Close the circle
        circle_marker.points.append(circle_marker.points[0])
        markers.markers.append(circle_marker)

        # Text marker for confidence
        text_marker = Marker()
        text_marker.header = header
        text_marker.ns = "dr_spaam_ros"
        text_marker.id = idx * 2 + 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = float(x)
        text_marker.pose.position.y = float(y)
        text_marker.pose.position.z = 0.5
        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z = 0.3
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.text = "{:.2f}".format(d_cls)

        markers.markers.append(text_marker)

    return markers


def main(args=None):
    rclpy.init(args=args)
    node = FakeDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
