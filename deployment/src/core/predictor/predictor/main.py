#!/usr/bin/env python3

import json
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from collections import deque
import numpy as np

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

from .human import Human
import time

from std_srvs.srv import Empty  # Import Empty service


class CVPredictor:
    """
    Constant Velocity (CV) Predictor
    简单的常速模型预测器，假设人以恒定速度移动
    """
    def __init__(self, predict_steps=5, dt=0.4):
        """
        Args:
            predict_steps: 预测未来的步数
            dt: 每步的时间间隔 (秒)
        """
        self.predict_steps = predict_steps
        self.dt = dt
    
    def predict(self, past_positions):
        """
        基于历史位置进行常速预测
        
        Args:
            past_positions: 历史位置列表 [(x1, y1), (x2, y2), ...]，时间顺序排列
        
        Returns:
            predicted_positions: 预测的未来位置 [(x1, y1), ..., (xn, yn)]
        """
        if len(past_positions) < 2:
            # 没有足够的历史数据，返回当前位置的重复
            current_pos = past_positions[-1] if past_positions else np.array([0.0, 0.0])
            return [current_pos for _ in range(self.predict_steps)]
        
        # 计算速度 (使用最近两个位置)
        pos_recent = np.array(past_positions[-1])
        pos_prev = np.array(past_positions[-2])
        
        velocity = (pos_recent - pos_prev) / self.dt
        
        # 预测未来位置
        predictions = []
        for i in range(1, self.predict_steps + 1):
            future_pos = pos_recent + velocity * self.dt * i
            predictions.append(future_pos)
        
        return predictions


class Predictor(Node):
    def __init__(self):
        super().__init__('predictor_node')

        # ============= Parameter Declarations =============
        self.declare_parameters(
            namespace='',
            parameters=[
                # Publisher for predicted trajectories (Visualization)
                ('publisher.predicted_trajectories.topic', '/predicted_trajectories_viz'),  # Existing topic
                ('publisher.predicted_trajectories.queue_size', 10),
                ('publisher.predicted_trajectories.latch', False),

                # Subscriber for tracked_objects_json
                ('subscriber.tracked_objects_json.topic', '/tracked_objects_json'),
                ('subscriber.tracked_objects_json.queue_size', 10),

                # Model parameters
                ('history_length', 5),  # Number of past positions to store
                ('predict_steps', 5),   # Number of future steps to predict
                ('dt', 0.25),           # Time interval between steps (matches RL training)

                # Publisher for predictions JSON to decider node
                ('publisher.predictions_json.topic', '/predictions_json'),  # Existing publisher
                ('publisher.predictions_json.queue_size', 10),
                ('publisher.predictions_json.latch', False),
            ]
        )

        # ============= Read Parameters =============
        self._read_params()

        # ============= Publishers =============
        self.marker_array_pub = self.create_publisher(
            MarkerArray,
            self.publisher_predicted_trajectories_topic,
            self.publisher_predicted_trajectories_queue_size  # Queue size as integer
        )

        # New Publisher for Predictions JSON to Decider Node
        self.predictions_json_pub = self.create_publisher(
            String,
            self.publisher_predictions_json_topic,
            self.publisher_predictions_json_queue_size  # Queue size as integer
        )

        # ============= Subscriptions =============
        self.tracked_objects_json_sub_ = self.create_subscription(
            String,
            self.subscriber_tracked_objects_json_topic,
            self.tracked_objects_json_callback,
            self.subscriber_tracked_objects_json_queue_size  # Queue size as integer
        )

        # ============= Initialize CV Predictor =============
        self.cv_predictor = CVPredictor(
            predict_steps=self.predict_steps_param,
            dt=self.dt_param
        )
        self.get_logger().info("[Predictor] CV (Constant Velocity) Predictor initialized.")

        self.reset()

    def _read_params(self):
        """Reads parameters from node."""
        # Publishers
        self.publisher_predicted_trajectories_topic = self.get_parameter(
            'publisher.predicted_trajectories.topic').value
        self.publisher_predicted_trajectories_queue_size = self.get_parameter(
            'publisher.predicted_trajectories.queue_size').value
        self.publisher_predicted_trajectories_latch = self.get_parameter(
            'publisher.predicted_trajectories.latch').value

        self.publisher_predictions_json_topic = self.get_parameter(
            'publisher.predictions_json.topic').value
        self.publisher_predictions_json_queue_size = self.get_parameter(
            'publisher.predictions_json.queue_size').value
        self.publisher_predictions_json_latch = self.get_parameter(
            'publisher.predictions_json.latch').value

        # Subscribers
        self.subscriber_tracked_objects_json_topic = self.get_parameter(
            'subscriber.tracked_objects_json.topic').value
        self.subscriber_tracked_objects_json_queue_size = self.get_parameter(
            'subscriber.tracked_objects_json.queue_size').value

        # Model parameters
        self.history_length = self.get_parameter('history_length').value
        self.predict_steps_param = self.get_parameter('predict_steps').value
        self.dt_param = self.get_parameter('dt').value

    def reset(self):
        # Create the Human objects (IDs from 0..49)
        self.max_human_num = 50
        self.humans = [Human(id=i, x=15.0, y=15.0) for i in range(self.max_human_num)]
        self.step_counter = 0

    def tracked_objects_json_callback(self, msg: String):
        """
        Callback function to process incoming JSON messages from /tracked_objects_json.
        Expected JSON structure:
        {
          "header": {
             "frame_id": "laser",
             "stamp_sec": 1733440988,
             "stamp_nsec": 246987811
          },
          "tracks": [
             {"id": 1, "x": 1.23, "y": 4.56, "timestamp": 1733440988.246987811},
             {"id": 2, "x": 7.89, "y": 0.12, "timestamp": 1733440988.246987811},
             ...
          ]
        }
        """
        # Clear old markers before publishing new ones
        self.clear_old_markers(msg.data)

        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        tracks = data.get("tracks", [])
        header = data.get("header", {})
        frame_id = header.get("frame_id", "odom")  # Default to 'odom' if not specified
        stamp_sec = header.get("stamp_sec", 0)
        stamp_nsec = header.get("stamp_nsec", 0)

        if not tracks:
            return

        self.step_counter += 1

        # Create a boolean mask for which humans are visible in the current step.
        self.visible_mask = np.zeros((self.max_human_num,), dtype=bool)

        # Prepare MarkerArrays to collect all markers
        markers = MarkerArray()

        for agent in tracks:
            agent_id = agent.get("id", -1)
            orig_x = agent.get("x", 15.0)
            orig_y = agent.get("y", 15.0)
            t = agent.get("timestamp", 0.0)

            # Check the range to avoid out-of-bounds
            if agent_id < 0 or agent_id >= self.max_human_num:
                self.get_logger().warning(
                    f"[Predictor] Skipping agent with out-of-range ID={agent_id} "
                    f"(position=({orig_x:.2f},{orig_y:.2f}))"
                )
                continue

            x, y = orig_x, orig_y

            # Update the Human object's position
            self.humans[agent_id].set_attributes(x, y, t)

            # Mark this agent as visible
            self.visible_mask[agent_id] = True

        # Visualize current positions for visible humans
        for human in self.humans:
            # Check if this human was visible
            if not self.visible_mask[human.id]:
                continue
            current_marker = self.create_current_position_marker(
                human.id,
                human.get_position(),
                frame_id=frame_id
            )
            markers.markers.append(current_marker)

        # Perform CV prediction for each visible human
        cv_start_time = time.time()
        for human in self.humans:
            if not self.visible_mask[human.id]:
                continue
            
            # Get past positions for CV prediction
            past_positions = list(human.past_locations)
            if len(past_positions) < 1:
                past_positions = [human.get_position()]
            
            # Perform CV prediction
            predictions = self.cv_predictor.predict(past_positions)
            
            # Store predictions
            is_valid = np.ones(len(predictions), dtype=bool)
            human.store_predictions(
                np.array(predictions),
                is_valid=is_valid
            )
            
            # Visualize the predictions
            trajectory = list(human.past_predictions[-1])
            trajectory_marker = self.create_trajectory_marker(human.id, trajectory, frame_id=frame_id)
            markers.markers.append(trajectory_marker)
        
        cv_elapsed = time.time() - cv_start_time
        self.get_logger().info(f"[Predictor] Time for CV inference: {cv_elapsed:.4f}s")

        # Build and publish predictions JSON to decider node
        predictions_json = self.build_predictions_json(header, self.humans)
        predictions_msg = String()
        predictions_msg.data = predictions_json
        self.predictions_json_pub.publish(predictions_msg)

        # Publish trajectory markers
        self.marker_array_pub.publish(markers)

    def clear_old_markers(self, json_data):
        """
        Publishes a DELETEALL Marker to clear previously displayed markers.
        Extracts header information from the incoming JSON data.
        """
        try:
            data = json.loads(json_data)
            header = data.get("header", {})
            frame_id = header.get("frame_id", "odom")
            stamp_sec = header.get("stamp_sec", 0)
            stamp_nsec = header.get("stamp_nsec", 0)
        except json.JSONDecodeError:
            self.get_logger().error("[Predictor] Failed to decode JSON for header information.")
            return

        # Clear trajectory markers
        clear_msg = MarkerArray()
        clear_marker = Marker()
        clear_marker.header.frame_id = frame_id
        clear_marker.header.stamp.sec = stamp_sec
        clear_marker.header.stamp.nanosec = stamp_nsec
        clear_marker.ns = "predictor"  # Namespace should match the markers being published
        clear_marker.id = 0
        clear_marker.action = Marker.DELETEALL
        clear_msg.markers.append(clear_marker)
        self.marker_array_pub.publish(clear_msg)

    def build_predictions_json(self, header, humans):
        """
        Constructs a JSON string containing predictions for each tracked agent.
        """
        predictions = []
        for human in humans:
            if not self.visible_mask[human.id]:
                continue
            # Assuming 'past_predictions' holds the predicted trajectories
            if len(human.past_predictions) == 0:
                continue
            trajectory = [
                {"x": float(pos[0]), "y": float(pos[1])}
                for pos in human.past_predictions[-1]  # Latest prediction
            ]
            predictions.append({
                "id": human.id,
                "predicted_trajectory": trajectory
            })
        data_dict = {
            "header": {
                "frame_id": header.get("frame_id", "odom"),
                "stamp_sec": header.get("stamp_sec", 0),
                "stamp_nsec": header.get("stamp_nsec", 0)
            },
            "predictions": predictions
        }

        json_str = json.dumps(data_dict, ensure_ascii=False)
        return json_str

    def create_trajectory_marker(self, agent_id, trajectory, frame_id='odom'):
        """
        Creates a Marker message for the predicted trajectory of an agent
        as dots (SPHERE_LIST) in the specified frame with blue color.
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "predictor"
        marker.id = agent_id + 1000  # Offset ID to differentiate from position markers

        marker.type = Marker.SPHERE_LIST  # Changed from LINE_STRIP to SPHERE_LIST for dots
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.5  # Reduced scale for better visualization
        marker.scale.y = 0.5
        marker.scale.z = 0.5

        # Set color to blue
        marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.4)  # Blue with higher alpha

        for pos in trajectory:
            p = Point()
            p.x = float(pos[0])
            p.y = float(pos[1])
            p.z = 0.0
            marker.points.append(p)

        return marker

    def create_current_position_marker(self, agent_id, position, frame_id='odom'):
        """
        Creates a Marker message for the current position of an agent
        in the specified frame.
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "predictor"
        marker.id = agent_id  # Unique ID for each agent's current position

        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = 0.0

        marker.color = ColorRGBA(r=0.5, g=0.5, b=1.0, a=0.8)  # Light blue
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5

        return marker

    def id_to_color(self, agent_id):
        """
        Generates a unique color for each agent based on their ID.
        """
        np.random.seed(agent_id)
        return np.random.rand(1).tolist()


def main(args=None):
    rclpy.init(args=args)
    node = Predictor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[Predictor] KeyboardInterrupt, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
