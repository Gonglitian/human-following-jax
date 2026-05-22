#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import os
import csv
import time

# According to your actual topics and message types:
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseArray


class FrequencyMonitorNode(Node):
    def __init__(self):
        super().__init__('frequency_monitor')

        # ---- (A) Which topics to track for frequency ----
        # The dictionary maps topic_name -> message_type
        self.topics = {
            '/dr_spaam_detections': PoseArray,      # PoseArray in your tracker node
            '/tracked_objects_json': String,
            '/predictions_json': String,
            '/cmd_vel': Twist
        }

        # Create a unique log directory (include timestamp to avoid overwriting)
        self.log_dir = f'ros2_frequency_log_{time.time():.0f}'
        os.makedirs(self.log_dir, exist_ok=True)

        # We'll compute average frequencies every 3 seconds
        self.monitor_interval = 1.0

        # (B) Store arrival times (last few seconds) for each topic
        # For example: self.arrival_times['/dr_spaam_detections'] = [t1, t2, ...]
        self.arrival_times = {topic: [] for topic in self.topics}

        # (C) Prepare CSV writers for each topic's frequency log
        self.csv_files = {}
        self.csv_writers = {}
        for topic_name in self.topics:
            filename = self._topic_to_filename(topic_name)
            filepath = os.path.join(self.log_dir, filename)

            file_exists = os.path.exists(filepath)
            f = open(filepath, mode='a', newline='', encoding='utf-8')
            writer = csv.writer(f)

            if not file_exists:
                # Write a header row: [timestamp, frequency_hz]
                writer.writerow(['timestamp', 'frequency_hz'])

            self.csv_files[topic_name] = f
            self.csv_writers[topic_name] = writer

        # (D) Initialize the subscription for each topic
        for topic_name, msg_type in self.topics.items():
            self.create_subscription(
                msg_type,
                topic_name,
                lambda msg, t=topic_name: self._callback(msg, t),
                10
            )

        # (E) Additional: Track last detection time for /dr_spaam_detections to measure delay
        self.last_dr_spaam_time = None

        # (F) Create an extra CSV for storing dr_spaam->cmd_vel delays
        delay_filename = "dr_spaam_cmd_vel_delay.csv"
        self.delay_filepath = os.path.join(self.log_dir, delay_filename)
        delay_file_exists = os.path.exists(self.delay_filepath)
        self.delay_file = open(self.delay_filepath, mode='a', newline='', encoding='utf-8')
        self.delay_writer = csv.writer(self.delay_file)

        if not delay_file_exists:
            # For the delay CSV, let's store [timestamp_of_cmd_vel, delay_s]
            self.delay_writer.writerow(["cmd_vel_timestamp", "delay_s"])

        # (G) Create a timer for computing average frequency every 3 seconds
        self.timer_ = self.create_timer(self.monitor_interval, self._timer_callback)

    def _callback(self, msg, topic_name):
        """
        When a message is received on any subscribed topic, record the current time.
        Additionally, if it's /cmd_vel, compute the time delay from the last
        /dr_spaam_detections (if available).
        """
        now = time.time()
        self.arrival_times[topic_name].append(now)

        # If this is /dr_spaam_detections, update our 'last detection time'
        if topic_name == '/dr_spaam_detections':
            self.last_dr_spaam_time = now

        # If this is /cmd_vel, compute the delay from last dr_spaam_detections (if any)
        elif topic_name == '/cmd_vel':
            if self.last_dr_spaam_time is not None:
                delay = now - self.last_dr_spaam_time
                # Write that delay to our special CSV
                self.delay_writer.writerow([f"{now:.6f}", f"{delay:.6f}"])
                self.delay_file.flush()
                self.get_logger().info(
                    f"[Delay] dr_spaam_detections -> cmd_vel = {delay:.6f} s"
                )

    def _timer_callback(self):
        """
        Every 3 seconds, compute how many messages arrived in that window
        for each topic, compute average frequency, and log it to CSV.
        If no messages arrived for a topic, we skip writing (i.e. don't record 0).
        """
        now = time.time()
        window_start = now - self.monitor_interval  # e.g. now - 3s

        for topic_name in self.topics:
            # Keep only the times >= window_start
            old_list = self.arrival_times[topic_name]
            recent_times = [t for t in old_list if t >= window_start]
            self.arrival_times[topic_name] = recent_times

            count = len(recent_times)
            if count > 0:
                freq = count / self.monitor_interval
                # Write to the per-topic frequency CSV
                writer = self.csv_writers[topic_name]
                writer.writerow([f"{now:.3f}", f"{freq:.3f}"])
                self.csv_files[topic_name].flush()

                # Also print to terminal
                self.get_logger().info(
                    f"[{topic_name}] average frequency (past {self.monitor_interval} s): {freq:.3f} Hz"
                )
            # else: no messages in the last 3s → skip writing

    def _topic_to_filename(self, topic_name: str) -> str:
        """
        Convert a ROS topic name into a CSV filename, e.g.:
          '/dr_spaam_detections' -> 'dr_spaam_detections_frequency.csv'
        """
        clean_topic = topic_name.lstrip('/').replace('/', '_')
        return f"{clean_topic}_frequency.csv"

    def destroy_node(self):
        """
        When shutting down, close all CSV files to avoid resource leaks.
        """
        for topic_name, f in self.csv_files.items():
            f.close()
        if self.delay_file:
            self.delay_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FrequencyMonitorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down frequency monitor.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
