#!/usr/bin/env python3

import json
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import String
import time
from visualization_msgs.msg import Marker, MarkerArray

# Import the Sort tracker
from .sort_max_hit_streak import Sort

from rclpy.qos import QoSProfile, DurabilityPolicy


class SortTrackerNode(Node):
    """
    ROS 2 node that subscribes to detections (PoseArray) and uses a SORT tracker
    to track objects over time. It publishes:
      1) A PoseArray for visualization or debugging
      2) A MarkerArray for RViz
      3) A JSON string (std_msgs/String) with each tracked agent's ID, position, and timestamp
    """

    def __init__(self):
        super().__init__('sort_tracker_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('publisher.tracks.topic', '/tracked_objects'),
                ('publisher.tracks.queue_size', 10),
                ('publisher.tracks.latch', False),
                ('publisher.tracks_viz.topic', '/tracked_objects_viz'),
                ('publisher.tracks_viz.queue_size', 10),
                ('publisher.tracks_viz.latch', False),
                ('publisher.json_tracks.topic', '/tracked_objects_json'),  # <== new param for JSON
                ('publisher.json_tracks.queue_size', 10),
                ('subscriber.detections.topic', '/minkunet/poses'),
                ('subscriber.detections.queue_size', 10),
                ('sort.max_age', 10),
                ('sort.min_hits', 1),
                ('sort.iou_threshold', 0.01),
            ]
        )

        self._read_params()

        # Initialize the SORT tracker
        self.mot_tracker = Sort(
            max_age=self.max_age,
            min_hits=self.min_hits,
            iou_threshold=self.iou_threshold
        )

        self._init_pub_sub()

    def _read_params(self):
        """Reads parameters from the node."""
        self.tracks_topic = self.get_parameter('publisher.tracks.topic').value
        self.tracks_qsize = self.get_parameter('publisher.tracks.queue_size').value
        self.tracks_latch = self.get_parameter('publisher.tracks.latch').value

        self.tracks_viz_topic = self.get_parameter('publisher.tracks_viz.topic').value
        self.tracks_viz_qsize = self.get_parameter('publisher.tracks_viz.queue_size').value
        self.tracks_viz_latch = self.get_parameter('publisher.tracks_viz.latch').value

        self.json_tracks_topic = self.get_parameter('publisher.json_tracks.topic').value
        self.json_tracks_qsize = self.get_parameter('publisher.json_tracks.queue_size').value

        self.detections_topic = self.get_parameter('subscriber.detections.topic').value
        self.detections_qsize = self.get_parameter('subscriber.detections.queue_size').value

        self.max_age = self.get_parameter('sort.max_age').value
        self.min_hits = self.get_parameter('sort.min_hits').value
        self.iou_threshold = self.get_parameter('sort.iou_threshold').value

    def _init_pub_sub(self):
        """Initialize publishers and subscribers."""
        # Publishers for PoseArray and MarkerArray
        tracks_qos = QoSProfile(depth=self.tracks_qsize)
        if self.tracks_latch:
            tracks_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._tracks_pub = self.create_publisher(PoseArray, self.tracks_topic, tracks_qos)

        tracks_viz_qos = QoSProfile(depth=self.tracks_viz_qsize)
        if self.tracks_viz_latch:
            tracks_viz_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._tracks_viz_pub = self.create_publisher(MarkerArray, self.tracks_viz_topic, tracks_viz_qos)

        # Publisher for JSON-based tracking info
        json_qos = QoSProfile(depth=self.json_tracks_qsize)
        self._json_pub = self.create_publisher(String, self.json_tracks_topic, json_qos)

        # Subscriber for detections
        detections_qos = QoSProfile(depth=self.detections_qsize)
        self._detections_sub = self.create_subscription(
            PoseArray, self.detections_topic, self._detection_callback, detections_qos
        )
        self.get_logger().info(f"Subscribing to detections topic: {self.detections_topic}")

    def _detection_callback(self, msg: PoseArray):
        # If no subscribers to tracks or viz, do nothing for them
        # But we still might want to publish JSON if _json_pub has subscribers
        if (
            self._tracks_pub.get_subscription_count() == 0
            and self._tracks_viz_pub.get_subscription_count() == 0
            and self._json_pub.get_subscription_count() == 0
        ):
            return

        # Clear previous markers
        self._clear_old_markers(msg.header)

        # Convert PoseArray (detections) to bounding boxes for SORT
        box_size = 1.2
        dets = []
        for p in msg.poses:
            x = p.position.x
            y = p.position.y
            x1 = x - box_size / 2.0
            y1 = y - box_size / 2.0
            x2 = x + box_size / 2.0
            y2 = y + box_size / 2.0
            score = 1.0  # If desired, could be replaced with detection confidence if available
            dets.append([x1, y1, x2, y2, score])
        # Fix #7: removed phantom (5,5) injection — pass empty array instead.
        # The old dummy caused the policy to always see a fake "human" at (5,5) in odom frame.
        if len(dets) == 0:
            dets = np.empty((0, 5))
        else:
            dets = np.array(dets)

        # Update tracker
        start_time = time.time()
        trackers = self.mot_tracker.update_with_valid_unmatched_trks(dets)
        end_time = time.time()
        self.get_logger().info(f"SORT updating time: {end_time - start_time} s")
        # self.get_logger().info(f"Number of tracked agents: {len(trackers)}")  # <-- Added


        # 1) Convert trackers to PoseArray for old usage
        tracked_poses = trackers_to_pose_array(trackers)
        tracked_poses.header.frame_id = msg.header.frame_id  # Retain original frame_id ('map')
        tracked_poses.header.stamp = msg.header.stamp  # Retain original timestamp
        self._tracks_pub.publish(tracked_poses)

        # 2) Convert trackers to MarkerArray for RViz
        tracking_markers = trackers_to_rviz_markers(trackers, header=msg.header)

        # **Set frame_id for each marker**
        for marker in tracking_markers.markers:
            marker.header.frame_id = msg.header.frame_id  # Retain original frame_id ('map')

        self._tracks_viz_pub.publish(tracking_markers)

        # 3) **Publish JSON string with ID, position, timestamp**
        if self._json_pub.get_subscription_count() > 0:
            json_msg = self._build_json_message(trackers, msg.header)
            self._json_pub.publish(json_msg)

    def _build_json_message(self, trackers, header):
        """
        Build a JSON string with each track's {id, x, y, timestamp}.
        We'll publish it in a std_msgs/String.
        """
        # Convert ROS time to float seconds
        stamp_sec = header.stamp.sec
        stamp_nsec = header.stamp.nanosec
        timestamp_sec = stamp_sec + stamp_nsec * 1e-9

        track_list = []
        for d in trackers:
            x1, y1, x2, y2, track_id = d
            x_center = float((x1 + x2) / 2.0)
            y_center = float((y1 + y2) / 2.0)

            track_info = {
                "id": int(track_id),
                "x": x_center,
                "y": y_center,
                "timestamp": timestamp_sec
            }
            track_list.append(track_info)

        # Additional header info if desired
        # frame_id: header.frame_id
        # stamp (sec, nsec) or a single float
        data_dict = {
            "header": {
                "frame_id": header.frame_id,  # Retain original frame_id ('map')
                "stamp_sec": stamp_sec,
                "stamp_nsec": stamp_nsec
            },
            "tracks": track_list
        }

        # Turn into a JSON string
        json_str = json.dumps(data_dict, ensure_ascii=False)
        msg = String()
        msg.data = json_str
        return msg

    def _clear_old_markers(self, header):
        """Publish a DELETEALL Marker to clear previous visualization markers."""
        clear_msg = MarkerArray()
        clear_marker = Marker()
        clear_marker.header.frame_id = header.frame_id  # Retain original frame_id ('laser')
        clear_marker.header.stamp = header.stamp
        clear_marker.ns = "tracked_objects"
        clear_marker.action = Marker.DELETEALL
        clear_msg.markers.append(clear_marker)
        self._tracks_viz_pub.publish(clear_msg)


def trackers_to_pose_array(trackers):
    """Convert tracker outputs to PoseArray for backward compatibility."""
    pose_array = PoseArray()
    pose_array.header.frame_id = 'laser'  # Ensure frame_id is set to 'laser'
    for d in trackers:
        x1, y1, x2, y2, track_id = d
        p = Pose()
        p.position.x = float((x1 + x2) / 2.0)
        p.position.y = float((y1 + y2) / 2.0)
        p.position.z = 0.0
        pose_array.poses.append(p)
    return pose_array


def trackers_to_rviz_markers(trackers, header):
    """Convert tracker outputs to RViz MarkerArray messages."""
    marker_array = MarkerArray()
    for d in trackers:
        x1, y1, x2, y2, track_id = d

        # Box marker as cube
        box_marker = Marker()
        box_marker.header.frame_id = header.frame_id  # Retain original frame_id ('laser')
        box_marker.header.stamp = header.stamp
        box_marker.ns = "tracked_objects"
        box_marker.id = int(track_id)
        box_marker.type = Marker.CUBE
        box_marker.action = Marker.ADD
        box_marker.pose.position.x = float((x1 + x2) / 2.0)
        box_marker.pose.position.y = float((y1 + y2) / 2.0)
        box_marker.pose.position.z = 0.0
        box_marker.pose.orientation.w = 1.0
        box_marker.scale.x = float(x2 - x1)
        box_marker.scale.y = float(y2 - y1)
        box_marker.scale.z = 0.1
        box_marker.color.r = 0.0
        box_marker.color.g = 0.0
        box_marker.color.b = 1.0
        box_marker.color.a = 0.0 # Set to 0.0 for invisible boxes # yjp mark
        marker_array.markers.append(box_marker)

        # Text marker for ID
        text_marker = Marker()
        text_marker.header.frame_id = header.frame_id  # Retain original frame_id ('laser')
        text_marker.header.stamp = header.stamp
        text_marker.ns = "tracked_objects_ids"
        text_marker.id = int(track_id) + 1000
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = float((x1 + x2) / 2.0)
        text_marker.pose.position.y = float((y1 + y2) / 2.0)
        text_marker.pose.position.z = 0.5
        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z = 0.5
        text_marker.color.r = 1.0
        text_marker.color.g = 0.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.text = str(int(track_id))
        marker_array.markers.append(text_marker)

    return marker_array


def main(args=None):
    rclpy.init(args=args)
    node = SortTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

