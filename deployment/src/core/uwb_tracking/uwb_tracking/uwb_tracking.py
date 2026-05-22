#!/usr/bin/env python3

import serial
import serial.tools.list_ports
import json
import time
import math
import traceback
import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


class UWBKalmanFilter:
    """
    Kalman filter for UWB position tracking.
    State vector: [x, y, vx, vy]
    Measurement: [x, y]
    Uses constant velocity motion model.
    """
    def __init__(self, dt=0.05, process_noise=0.5, measurement_noise=1.0):
        """
        Initialize Kalman filter.

        Args:
            dt: Time step between updates (default 0.05s = 20Hz)
            process_noise: Process noise covariance (higher = trust measurements more)
            measurement_noise: Measurement noise covariance (higher = trust predictions more)
        """
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.dt = dt

        # State Transition Matrix F (constant velocity model)
        # [x, y, vx, vy] -> [x + vx*dt, y + vy*dt, vx, vy]
        self.kf.F = np.array([
            [1, 0, dt, 0],   # x = x + vx*dt
            [0, 1, 0, dt],   # y = y + vy*dt
            [0, 0, 1, 0],    # vx = vx
            [0, 0, 0, 1]     # vy = vy
        ])

        # Measurement Matrix H (we only measure x, y)
        self.kf.H = np.array([
            [1, 0, 0, 0],    # measure x
            [0, 1, 0, 0]     # measure y
        ])

        # Measurement Noise Covariance R
        # UWB typically has 10-30cm accuracy, so we set this accordingly
        self.kf.R = np.eye(2) * measurement_noise

        # Process Noise Covariance Q
        # Models uncertainty in the motion model
        q = process_noise
        self.kf.Q = np.array([
            [q*dt**4/4, 0, q*dt**3/2, 0],
            [0, q*dt**4/4, 0, q*dt**3/2],
            [q*dt**3/2, 0, q*dt**2, 0],
            [0, q*dt**3/2, 0, q*dt**2]
        ])

        # Initial State Covariance P (high uncertainty initially)
        self.kf.P = np.eye(4) * 10.0
        self.kf.P[2:, 2:] *= 100.0  # Higher uncertainty for velocity

        # Initial state
        self.kf.x = np.zeros((4, 1))

        self.initialized = False
        self.last_update_time = None

    def initialize(self, x, y):
        """Initialize filter with first measurement."""
        self.kf.x = np.array([[x], [y], [0], [0]])
        self.initialized = True
        self.last_update_time = time.time()

    def predict(self):
        """Predict next state."""
        if not self.initialized:
            return None, None

        # Update dt based on actual time elapsed
        current_time = time.time()
        if self.last_update_time is not None:
            actual_dt = current_time - self.last_update_time
            if actual_dt > 0.001:  # Avoid division issues
                self.kf.F[0, 2] = actual_dt
                self.kf.F[1, 3] = actual_dt

        self.kf.predict()
        return float(self.kf.x[0]), float(self.kf.x[1])

    def update(self, x, y):
        """Update filter with new measurement."""
        current_time = time.time()

        if not self.initialized:
            self.initialize(x, y)
            return x, y

        # Update dt based on actual time elapsed
        if self.last_update_time is not None:
            actual_dt = current_time - self.last_update_time
            if actual_dt > 0.001:
                self.kf.F[0, 2] = actual_dt
                self.kf.F[1, 3] = actual_dt

        # Predict
        self.kf.predict()

        # Update with measurement
        z = np.array([[x], [y]])
        self.kf.update(z)

        self.last_update_time = current_time

        return float(self.kf.x[0]), float(self.kf.x[1])

    def get_position(self):
        """Get current estimated position."""
        if not self.initialized:
            return None, None
        return float(self.kf.x[0]), float(self.kf.x[1])

    def get_velocity(self):
        """Get current estimated velocity."""
        if not self.initialized:
            return None, None
        return float(self.kf.x[2]), float(self.kf.x[3])

class UWB:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.x = 0
        self.y = 0
        self.status = False
        self.range_list = []
        self.history_x = deque(maxlen=20)  # Reduced from 50 to 20 for better real-time display
        self.history_y = deque(maxlen=20)
        self.last_update = time.time()  # Track when data was last updated

        # Velocity filtering parameters
        self.max_human_speed = 200.0  # Disabled (allow all movements)
        self.last_position_time = None
        self.last_valid_x = None
        self.last_valid_y = None

        # Kalman filter for this tag (only for tags, not anchors)
        self.use_kalman_filter = True
        self.kalman_filter = None
        if self.type == 1:  # Tag
            self.kalman_filter = UWBKalmanFilter(dt=0.05, process_noise=0.1, measurement_noise=3.0)
            self.color = 'red'
            self.marker = 'o'
        else:  # Anchor
            self.color = 'black'
            self.marker = 's'

    def set_location(self, x, y):
        self.x = x
        self.y = y
        self.status = True
        if self.type == 1:  # Only track tag movement
            self.history_x.append(x)
            self.history_y.append(y)

    def calculate_position(self):
        """Calculate position using proper trilateration"""
        count = 0
        anchor_positions = []
        distances = []

        anchor_details = []
        
        # Get anchors list from the global node instance (will be set during init)
        if hasattr(self, '_anchors_ref'):
            anchors_list = self._anchors_ref
        else:
            # Fallback for standalone usage
            anchors_list = []
        
        for i, range_val in enumerate(self.range_list):
            if range_val != 0 and i < len(anchors_list):
                anchor_positions.append((anchors_list[i].x, anchors_list[i].y))
                distances.append(range_val / 100.0)  # Convert cm to meters
                count += 1

                dist_meters = range_val / 100.0
                anchor_name = anchors_list[i].name
                anchor_details.append(f"{anchor_name}: {dist_meters:.2f}m")

        details_str = ", ".join(anchor_details)

        if count >= 3:
            try:
                # Use proper trilateration with first 3 anchors
                x, y = self.trilaterate(
                    anchor_positions[0], distances[0],
                    anchor_positions[1], distances[1], 
                    anchor_positions[2], distances[2]
                )
                
                # Validate result by checking distances to all anchors
                max_error = 0
                for i in range(count):
                    calc_dist = math.sqrt((x - anchor_positions[i][0])**2 + (y - anchor_positions[i][1])**2)
                    error = abs(calc_dist - distances[i])
                    max_error = max(max_error, error)
                
                # Accept result if error is reasonable
                max_error_threshold = getattr(self, 'max_error', 3.0)
                if max_error < max_error_threshold:
                    # Apply velocity filtering before smoothing
                    if self.is_valid_velocity(x, y):
                        # Apply Kalman filter or simple low-pass filter
                        if self.use_kalman_filter and self.kalman_filter is not None:
                            # Use Kalman filter for better smoothing
                            x_filtered, y_filtered = self.kalman_filter.update(x, y)
                            x, y = x_filtered, y_filtered
                        elif hasattr(self, 'x') and self.x != 0 and self.y != 0:
                            # Fallback: Simple low-pass filter
                            alpha = getattr(self, 'smoothing_alpha', 0.3)
                            x = alpha * x + (1 - alpha) * self.x
                            y = alpha * y + (1 - alpha) * self.y

                        self.set_location(x, y)
                        self.status = True

                        # Publish ROS2 messages
                        if hasattr(self, 'publisher') and self.publisher:
                            filter_type = "KF" if (self.use_kalman_filter and self.kalman_filter) else "LPF"
                            print(f"Tag {self.name} [{filter_type}]: ({x:.2f}, {y:.2f}) err:{max_error:.2f}m | {details_str}")
                            self.publish_position(x, y)

                        # Update velocity filter state
                        self.update_velocity_filter(x, y)
                    else:
                        # Velocity too high - use Kalman prediction instead if available
                        if self.use_kalman_filter and self.kalman_filter is not None and self.kalman_filter.initialized:
                            pred_x, pred_y = self.kalman_filter.predict()
                            if pred_x is not None:
                                self.set_location(pred_x, pred_y)
                                if hasattr(self, 'publisher') and self.publisher:
                                    print(f"Tag {self.name} [KF-PRED]: ({pred_x:.2f}, {pred_y:.2f}) - measurement rejected (speed > {self.max_human_speed}m/s)")
                                    self.publish_position(pred_x, pred_y)
                        else:
                            print(f"Tag {self.name} rejected: velocity too high (> {self.max_human_speed}m/s) | {details_str}")
                else:
                    print(f"Tag {self.name} trilateration error too high: {max_error:.2f}m > {max_error_threshold:.1f}m, skipping")
                    
            except Exception as e:
                print(f"[Error]: {e}")
                # pass  # Silently ignore trilateration failures
                
        elif count == 2:
            # Fallback to two-point method if only 2 anchors available
            try:
                x, y = self.two_point_positioning(
                    anchor_positions[0][0], anchor_positions[0][1],
                    anchor_positions[1][0], anchor_positions[1][1],
                    distances[0], distances[1]
                )

                # Apply velocity filtering for 2-point method as well
                if self.is_valid_velocity(x, y):
                    # Apply Kalman filter or simple low-pass filter
                    if self.use_kalman_filter and self.kalman_filter is not None:
                        x_filtered, y_filtered = self.kalman_filter.update(x, y)
                        x, y = x_filtered, y_filtered

                    self.set_location(x, y)
                    self.status = True

                    if hasattr(self, 'publisher') and self.publisher:
                        filter_type = "KF" if (self.use_kalman_filter and self.kalman_filter) else "LPF"
                        print(f"Tag {self.name} 2-pt [{filter_type}]: ({x:.2f}, {y:.2f}) | {details_str}")
                        self.publish_position(x, y)
                    self.update_velocity_filter(x, y)
                else:
                    # Use Kalman prediction if available
                    if self.use_kalman_filter and self.kalman_filter is not None and self.kalman_filter.initialized:
                        pred_x, pred_y = self.kalman_filter.predict()
                        if pred_x is not None:
                            self.set_location(pred_x, pred_y)
                            if hasattr(self, 'publisher') and self.publisher:
                                print(f"Tag {self.name} 2-pt [KF-PRED]: ({pred_x:.2f}, {pred_y:.2f}) - rejected")
                                self.publish_position(pred_x, pred_y)
                    else:
                        print(f"Tag {self.name} 2-pt rejected: velocity too high")
            except Exception as e:
                print(f"2-point positioning failed for tag {self.name}: {e}")

    def two_point_uwb(self, a_id, b_id):
        """Calculate position using two anchors"""
        x, y = self.two_point_positioning(
            anchors[a_id].x, anchors[a_id].y, 
            anchors[b_id].x, anchors[b_id].y, 
            self.range_list[a_id] / 100.0,  # Convert cm to meters
            self.range_list[b_id] / 100.0
        )
        return x, y

    def two_point_positioning(self, x1, y1, x2, y2, r1, r2):
        """Two point positioning algorithm from position.py"""
        temp_x = 0.0
        temp_y = 0.0
        
        # Distance between circle centers
        p2p = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        p2p = math.sqrt(p2p)

        # Check if circles intersect
        if r1 + r2 <= p2p:
            temp_x = x1 + (x2 - x1) * r1 / (r1 + r2)
            temp_y = y1 + (y2 - y1) * r1 / (r1 + r2)
        else:
            dr = p2p / 2 + (r1 * r1 - r2 * r2) / (2 * p2p)
            temp_x = x1 + (x2 - x1) * dr / p2p
            temp_y = y1 + (y2 - y1) * dr / p2p

        return temp_x, temp_y

    def trilaterate(self, p1, r1, p2, r2, p3, r3):
        """
        Proper 3-point trilateration algorithm
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        A = 2*(x2 - x1)
        B = 2*(y2 - y1)
        C = r1**2 - r2**2 - x1**2 + x2**2 - y1**2 + y2**2
        D = 2*(x3 - x2)
        E = 2*(y3 - y2)
        F = r2**2 - r3**2 - x2**2 + x3**2 - y2**2 + y3**2

        denom = A*E - B*D
        if abs(denom) < 1e-10:  # Avoid division by zero
            raise ValueError("Degenerate trilateration case")

        x = (C*E - B*F) / denom
        y = (A*F - C*D) / denom
        return x, y
    
    def is_valid_velocity(self, new_x, new_y):
        """Check if the new position represents a realistic human movement speed"""
        current_time = time.time()
        
        # Always accept first position
        if self.last_position_time is None or self.last_valid_x is None or self.last_valid_y is None:
            return True
        
        # Calculate time difference
        dt = current_time - self.last_position_time
        
        # Avoid division by zero for very fast updates
        if dt < 0.01:  # Less than 10ms
            return True
        
        # Calculate distance moved
        dx = new_x - self.last_valid_x
        dy = new_y - self.last_valid_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Calculate speed
        speed = distance / dt
        
        # Check if speed is reasonable for human movement
        if speed <= self.max_human_speed:
            return True
        else:
            print(f"  Rejected: moved {distance:.2f}m in {dt:.2f}s = {speed:.2f}m/s (max: {self.max_human_speed}m/s)")
            return False
    
    def update_velocity_filter(self, x, y):
        """Update the velocity filter state with the accepted position"""
        self.last_position_time = time.time()
        self.last_valid_x = x
        self.last_valid_y = y

    def set_publisher(self, publisher, marker_publisher):
        """Set ROS2 publishers for this tag"""
        self.publisher = publisher

    def set_anchors_reference(self, anchors_list):
        """Set reference to anchors list for positioning calculations"""
        self._anchors_ref = anchors_list

    def publish_position(self, x, y):
        """Publish tag position as ROS2 Point message"""
        if self.publisher:
            msg = Point()
            msg.x = float(x)
            msg.y = float(y)
            msg.z = 0.0  # Always 0 for 2D positioning
            self.publisher.publish(msg)
    


def parse_uwb_data(line):
    """Parse UWB data from AT+RANGE format"""
    try:
        if 'AT+RANGE=' in line:
            # Extract tid (tag ID)
            tid_start = line.find('tid:') + 4
            tid_end = line.find(',', tid_start)
            if tid_start == 3 or tid_end == -1:
                return None
            
            tag_id = int(line[tid_start:tid_end])
            
            # Extract range data
            range_start = line.find('range:(') + 7
            range_end = line.find(')', range_start)
            if range_start == 6 or range_end == -1:
                return None
            
            range_str = line[range_start:range_end]
            ranges = [int(x.strip()) for x in range_str.split(',')]
            
            return {'id': tag_id, 'range': ranges[:4]}  # Only take first 4 ranges
        
    except (ValueError, IndexError) as e:
        print(f"Parse error: {e}")
        return None
    
    return None

def read_data(ser):
    """Read and parse data from serial port - optimized for real-time tracking"""
    data_processed = False
    lines_read = 0
    max_lines_per_frame = 15  # Process multiple lines per frame to catch up
    
    try:
        # Aggressively clear old data if buffer is getting full
        if ser.in_waiting > 500:  # If buffer has >500 bytes of data
            ser.reset_input_buffer()
            print("Buffer cleared - too much old data")
            return False
            
        # Read multiple lines per frame to reduce lag
        while ser.in_waiting > 0 and lines_read < max_lines_per_frame:
            line = ser.readline().decode('UTF-8').replace('\n', '').strip()
            lines_read += 1
            
            if not line:
                continue
                
            print(f"Raw: {line}")
            
            data = parse_uwb_data(line)
            if data:
                print(f"Parsed: {data}")
                tag_id = data['id']
                if 0 <= tag_id < len(tags):
                    tags[tag_id].range_list = data['range']
                    tags[tag_id].calculate_position()
                    data_processed = True
        
    except Exception as e:
        print(f"Read error: {e}")
        
    return data_processed

class UWBTrackingNode(Node):
    def __init__(self):
        # Set default parameters BEFORE calling super().__init__()
        self.set_default_parameters()
        
        # Initialize the node
        super().__init__('uwb_tracking_node')
        
        # Try to load parameters from config file (non-blocking)
        self.load_parameters_safe()
        
        # Initialize serial connection
        self.ser = None
        try:
            self.ser = serial.Serial(
                self.serial_port, 
                self.serial_baudrate, 
                timeout=self.serial_timeout
            )
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.get_logger().info(f"Connected to {self.serial_port} with optimized settings")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to serial port: {e}")
            self.get_logger().info("Running without serial connection")
    
        # Create UWB objects
        self.anchors = []
        self.tags = []
        
        # Create anchor objects from config
        for i in range(4):  # Always 4 anchors
            anchor_name = f"anchor_{i}"
            if anchor_name in self.anchor_configs:
                config = self.anchor_configs[anchor_name]
                anchor = UWB(config['name'], 0)
                anchor.set_location(config['x'], config['y'])
                self.anchors.append(anchor)
                self.get_logger().info(f"  {config['name']}: ({config['x']:.2f}, {config['y']:.2f})")
            else:
                self.get_logger().error(f"Missing configuration for {anchor_name}")
                return
        
        # Create ROS2 publishers for each tag
        self.tag_publishers = []
        for i in range(self.tag_count):
            name = f"TAG {i}"
            tag = UWB(name, 1)
            # Create publishers for this tag
            publisher = self.create_publisher(Point, f'/uwb/tag_{i}/position', 10)
            tag.set_publisher(publisher, None)  # We'll use the shared marker array publisher
            tag.set_anchors_reference(self.anchors)  # Set anchor reference
            # Set configurable parameters
            tag.max_error = self.max_positioning_error
            tag.smoothing_alpha = self.smoothing_alpha
            tag.max_human_speed = self.max_human_speed
            # Configure Kalman filter
            tag.use_kalman_filter = self.use_kalman_filter
            if tag.use_kalman_filter and tag.kalman_filter is not None:
                tag.kalman_filter = UWBKalmanFilter(
                    dt=1.0/self.read_frequency,
                    process_noise=self.kalman_process_noise,
                    measurement_noise=self.kalman_measurement_noise
                )
                self.get_logger().info(f"  Tag {i}: Kalman filter enabled (Q={self.kalman_process_noise}, R={self.kalman_measurement_noise})")
            else:
                self.get_logger().info(f"  Tag {i}: Using low-pass filter (alpha={self.smoothing_alpha})")
            self.tags.append(tag)
            self.tag_publishers.append(publisher)
        
        # Create MarkerArray publishers
        self.tag_markers_pub = self.create_publisher(MarkerArray, '/uwb/tags/markers', 10)
        self.anchor_markers_pub = self.create_publisher(MarkerArray, '/uwb/anchors/markers', 10)
        
        # Publish anchor markers once (they don't move)
        self.publish_anchor_markers()

        # Clear serial buffer
        if self.ser:
            self.ser.reset_input_buffer()
        
        # Create timer for periodic data reading
        timer_period = 1.0 / self.read_frequency
        self.timer = self.create_timer(timer_period, self.read_and_process_data)
        self.buffer_clear_counter = 0
        
        self.get_logger().info(f"UWB tracking node started at {self.read_frequency}Hz")
        self.get_logger().info(f"  Filter config: Kalman={self.use_kalman_filter}, max_speed={self.max_human_speed}m/s")
        self.get_logger().info(f"  Publishing to /uwb/tag_X/position topics")
        self.get_logger().info("  Anchor markers: /uwb/anchors/markers")

    def set_default_parameters(self):
        """Set default parameter values for ROS2 Foxy compatibility"""
        # Serial configuration defaults
        self.serial_port = '/dev/ttyUSB0'
        self.serial_baudrate = 115200
        self.serial_timeout = 0.01
        
        # Data processing defaults
        self.read_frequency = 20.0
        self.max_lines_per_frame = 15
        self.buffer_clear_threshold = 500
        self.periodic_clear_interval = 40
        
        # Positioning defaults
        self.max_positioning_error = 3.0
        self.smoothing_alpha = 0.3  # Lower = smoother
        self.max_human_speed = 200.0  # Disable velocity filtering

        # Kalman filter defaults (stronger smoothing)
        self.use_kalman_filter = True
        self.kalman_process_noise = 0.1  # Lower = smoother
        self.kalman_measurement_noise = 3.0  # Higher = smoother

        # Tag configuration defaults
        self.tag_count = 2  # tag_0=human, tag_1=robot
        
        # Anchor configuration defaults (updated to match config.yaml)
        self.anchor_configs = {
            'anchor_0': {'x': 0, 'y': 4.8, 'z': 1.2, 'name': 'ANC 0'},
            'anchor_1': {'x': -0.48, 'y': -4.8, 'z': 1.2, 'name': 'ANC 1'},
            # 'anchor_2': {'x': 14.67, 'y': 4.8, 'z': 1.2, 'name': 'ANC 2'},
            'anchor_2': {'x': 5.91, 'y': 4.8, 'z': 1.2, 'name': 'ANC 2'},
            # 'anchor_3': {'x': 14.67, 'y': -3.67, 'z': 1.2, 'name': 'ANC 3'}
            'anchor_3': {'x': 5.91, 'y': -5.54, 'z': 1.2, 'name': 'ANC 3'}
        }
        
        # Logging defaults
        self.log_level = 'info'
        self.log_raw_data = False
        self.log_parsed_data = False

    def load_parameters_safe(self):
        """Safely load parameters from ROS2 parameter server with fallback to defaults"""
        self.get_logger().info("Loading parameters...")
        
        # Use a simpler parameter loading approach for ROS2 Foxy compatibility
        params_loaded = False
        
        try:
            # Try to load parameters after a small delay to ensure the node is fully initialized
            import time
            time.sleep(0.1)
            
            # Check if we have any parameters available (simple check)
            if hasattr(self, '_parameters') and self._parameters:
                params_loaded = True
                self.get_logger().info("Parameters available from config file")
            else:
                self.get_logger().info("No config parameters found, using defaults")
                
        except Exception as e:
            self.get_logger().warn(f"Parameter loading error: {e}")
            
        if not params_loaded:
            self.get_logger().info("Using default configuration values")
            
        self.get_logger().info(f"Configuration loaded:")
        self.get_logger().info(f"  Serial port: {self.serial_port}")
        self.get_logger().info(f"  Read frequency: {self.read_frequency}Hz")
        self.get_logger().info(f"  Tag count: {self.tag_count}")
        self.get_logger().info(f"  Anchor positions:")
        for name, config in self.anchor_configs.items():
            self.get_logger().info(f"    {config['name']}: ({config['x']:.2f}, {config['y']:.2f})")

    def publish_anchor_markers(self):
        """Publish anchor positions as MarkerArray for RViz visualization"""
        marker_array = MarkerArray()
        
        for i, (anchor_name, config) in enumerate(self.anchor_configs.items()):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "uwb_anchors"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = float(config['x'])
            marker.pose.position.y = float(config['y'])
            marker.pose.position.z = 0.5  # Slightly above ground
            
            # Orientation (no rotation)
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # Scale (cube size)
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 1.0  # Tall cube like a beacon
            
            # Color (blue for anchors)
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            
            # Lifetime (0 = permanent)
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 0
            
            # Text label
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "uwb_anchor_labels"
            text_marker.id = i + 100  # Offset ID for text
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(config['x'])
            text_marker.pose.position.y = float(config['y'])
            text_marker.pose.position.z = 1.2  # Above the cube
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.3  # Text size
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = config['name']
            text_marker.lifetime.sec = 0
            text_marker.lifetime.nanosec = 0
            
            marker_array.markers.append(marker)
            marker_array.markers.append(text_marker)
        
        self.anchor_markers_pub.publish(marker_array)
        self.get_logger().info(f"Published {len(self.anchor_configs)} anchor markers to /uwb/anchors/markers")

    def publish_tag_markers(self):
        """Publish all tag positions as MarkerArray for RViz visualization"""
        marker_array = MarkerArray()
        
        for i, tag in enumerate(self.tags):
            if tag.status and tag.x != 0 and tag.y != 0:  # Only publish if tag has valid position
                # Tag marker
                marker = Marker()
                marker.header.frame_id = "odom"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "uwb_tags"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                # Position
                marker.pose.position.x = float(tag.x)
                marker.pose.position.y = float(tag.y)
                marker.pose.position.z = 0.5  # Slightly above ground for visibility
                
                # Orientation (no rotation)
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                
                # Scale (sphere size)
                marker.scale.x = 0.3
                marker.scale.y = 0.3
                marker.scale.z = 0.3
                
                # Color (red for tags)
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0  # Alpha (opacity)
                
                # Lifetime (0 = permanent until replaced)
                marker.lifetime.sec = 0
                marker.lifetime.nanosec = 0
                
                # Text label
                text_marker = Marker()
                text_marker.header = marker.header
                text_marker.ns = "uwb_tag_labels"
                text_marker.id = i + 200  # Offset ID for text
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.pose.position.x = float(tag.x)
                text_marker.pose.position.y = float(tag.y)
                text_marker.pose.position.z = 0.8  # Above the sphere
                text_marker.pose.orientation.w = 1.0
                text_marker.scale.z = 0.3  # Text size
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 1.0
                text_marker.text = tag.name
                text_marker.lifetime.sec = 0
                text_marker.lifetime.nanosec = 0
                
                marker_array.markers.append(marker)
                marker_array.markers.append(text_marker)
        
        # Publish the marker array
        self.tag_markers_pub.publish(marker_array)

    def read_and_process_data(self):
        """Timer callback to read and process UWB data"""
        if not self.ser or not self.ser.is_open:
            return
            
        # Periodic buffer clearing
        self.buffer_clear_counter += 1
        if self.buffer_clear_counter >= self.periodic_clear_interval:
            self.ser.reset_input_buffer()
            self.get_logger().debug("Periodic buffer clear")
            self.buffer_clear_counter = 0
        
        # Read data from serial
        self.read_data()
        
        # Publish tag markers for RViz
        self.publish_tag_markers()

    def read_data(self):
        """Read and parse data from serial port - optimized for real-time tracking"""
        data_processed = False
        lines_read = 0
        
        try:
            # Aggressively clear old data if buffer is getting full
            if self.ser.in_waiting > self.buffer_clear_threshold:
                self.ser.reset_input_buffer()
                self.get_logger().debug("Buffer cleared - too much old data")
                return False
                
            # Read multiple lines per frame to reduce lag
            while self.ser.in_waiting > 0 and lines_read < self.max_lines_per_frame:
                line = self.ser.readline().decode('UTF-8').replace('\n', '').strip()
                lines_read += 1
                
                if not line:
                    continue
                    
                if self.log_raw_data:
                    self.get_logger().info(f"Raw: {line}")
                
                data = parse_uwb_data(line)
                if data:
                    if self.log_parsed_data:
                        self.get_logger().info(f"Parsed: {data}")
                    tag_id = data['id']
                    if 0 <= tag_id < len(self.tags):
                        self.tags[tag_id].range_list = data['range']
                        self.tags[tag_id].calculate_position()
                        data_processed = True
            
        except Exception as e:
            self.get_logger().error(f"Read error: {e}")
            
        return data_processed

    def destroy_node(self):
        """Clean shutdown"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.get_logger().info("Serial connection closed")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    uwb_node = UWBTrackingNode()
    
    try:
        rclpy.spin(uwb_node)
    except KeyboardInterrupt:
        uwb_node.get_logger().info("Keyboard interrupt received")
    finally:
        uwb_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 