#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import yaml
import math
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PointStamped, PoseStamped
import json
import os
from tf2_ros import Buffer, TransformListener, TransformException, TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration

import tf2_geometry_msgs

class StaticMapNode(Node):
    def __init__(self):
        super().__init__('static_map_node')
        
        # Grid parameters - keep these the same
        self.grid_size = 10.0  # 10m x 10m window
        self.resolution = 0.2  # 0.2m resolution
        self.width = int(self.grid_size / self.resolution)  # 50 cells
        self.height = int(self.grid_size / self.resolution)  # 50 cells
        
        # Publishers - keep these the same
        self.occ_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)
        self.occ_json_pub = self.create_publisher(String, '/occupancy_grid_json', 10)
        
        # TF buffer and broadcaster
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)  # ✅ ADDED: TF broadcaster
        
        # Subscribe to the GLOBAL costmap instead of local costmap
        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',  # Global costmap topic
            self.costmap_callback,
            10)
        
        # Store the global costmap for sliding window extraction
        self.global_costmap_data = None
        self.global_costmap_meta = None
        
        # Timer to publish map at regular intervals
        self.timer = self.create_timer(0.1, self.publish_sliding_window)  # 10Hz
        
        # ✅ FIXED: Timer for TF broadcasting (50Hz for smooth transforms)
        #  self.tf_timer = self.create_timer(0.02, self.publish_map_odom_tf)  # 50Hz
        
        self.get_logger().info('Robot-centered map node initialized using global costmap sliding window')

    def publish_map_odom_tf(self):
        """Publish a real-time transform between map and odom frames"""
        try:
            # Get robot position in map frame (ground truth from simulation)
            map_to_base_transform = self.tf_buffer.lookup_transform(
                "map", 
                "base_link", 
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1))
            
            # Get robot position in odom frame
            odom_to_base_transform = self.tf_buffer.lookup_transform(
                "odom", 
                "base_link", 
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1))
            
            # Calculate map->odom transform
            # This represents the correction needed to align odom with map
            
            # Extract positions
            map_x = map_to_base_transform.transform.translation.x
            map_y = map_to_base_transform.transform.translation.y
            map_rot = map_to_base_transform.transform.rotation
            
            odom_x = odom_to_base_transform.transform.translation.x  
            odom_y = odom_to_base_transform.transform.translation.y
            odom_rot = odom_to_base_transform.transform.rotation
            
            # Calculate the offset between map and odom
            # map_to_odom = map_to_base - odom_to_base
            offset_x = map_x - odom_x
            offset_y = map_y - odom_y
            
            # For rotation, calculate the difference in quaternions
            # This is simplified - assumes small rotation differences
            offset_rot = map_rot  # Use map rotation as reference
            
            # Create and publish map->odom transform
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'map'
            t.child_frame_id = 'odom'
            
            # Set translation (offset between map and odom origins)
            t.transform.translation.x = offset_x
            t.transform.translation.y = offset_y
            t.transform.translation.z = 0.0
            
            # Set rotation (align odom with map)
            t.transform.rotation = offset_rot
            
            # Broadcast the transform
            self.tf_broadcaster.sendTransform(t)
            
            self.get_logger().debug(f"Published map->odom TF: offset=({offset_x:.3f}, {offset_y:.3f})")
            
        except TransformException as e:
            self.get_logger().debug(f"Could not compute map->odom transform: {e}")
            # Publish identity transform as fallback
            self._publish_identity_map_odom_tf()
    
    def _publish_identity_map_odom_tf(self):
        """Publish an identity transform between map and odom (no offset)"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        
        # Identity translation
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        
        # Identity rotation
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().debug("Published identity map->odom TF")

    def costmap_callback(self, costmap_msg):
        """Store the global costmap for sliding window extraction"""
        # Check costmap frame - global costmap is usually in map frame
        if costmap_msg.header.frame_id != "map":
            self.get_logger().warn(f"Global costmap frame is {costmap_msg.header.frame_id}, expected 'map'")
        
        # Extract and store the global costmap data
        costmap_resolution = costmap_msg.info.resolution
        costmap_width = costmap_msg.info.width
        costmap_height = costmap_msg.info.height
        
        self.get_logger().debug(f"Received global costmap: {costmap_width}x{costmap_height} at {costmap_resolution}m resolution")
        
        # Reshape the costmap data to 2D grid
        costmap_data = np.array(costmap_msg.data, dtype=np.int8).reshape(
            costmap_height, costmap_width)
        
        # Handle -1 (unknown) values - convert to 0 (free) as requested
        # In costmaps: 0=free, >0=obstacle cost, -1=unknown
        processed_costmap = costmap_data.copy()
        processed_costmap[processed_costmap == -1] = 0  # Unknown -> Free
        
        # Downsample if needed
        if costmap_resolution != self.resolution:
            # Calculate downsampling factor
            scale_factor = self.resolution / costmap_resolution
            
            self.get_logger().info(f"Downsampling costmap from {costmap_resolution}m to {self.resolution}m resolution (scale factor: {scale_factor})")
            
            # Calculate new dimensions
            new_width = int(costmap_width / scale_factor)
            new_height = int(costmap_height / scale_factor)
            
            # Ensure positive values for OpenCV
            costmap_for_resize = np.clip(processed_costmap, 0, 255).astype(np.uint8)
            
            # Downsample using OpenCV with INTER_NEAREST to preserve occupancy values
            downsampled_costmap = cv2.resize(
                costmap_for_resize,
                (new_width, new_height),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int8)
            
            # Update the metadata for the downsampled costmap
            updated_meta = costmap_msg.info
            updated_meta.resolution = self.resolution
            updated_meta.width = new_width
            updated_meta.height = new_height
            # Note: origin stays the same as it's in world coordinates
            
            self.global_costmap_data = downsampled_costmap
            self.global_costmap_meta = updated_meta
            
            self.get_logger().debug(f"Downsampled to: {new_width}x{new_height} at {self.resolution}m resolution")
        else:
            # No downsampling needed
            self.global_costmap_data = processed_costmap
            self.global_costmap_meta = costmap_msg.info
            
            self.get_logger().debug(f"No downsampling needed: {costmap_width}x{costmap_height} at {costmap_resolution}m resolution")

    def publish_sliding_window(self):
        """Extract a robot-centered window from the global costmap and publish it"""
        if self.global_costmap_data is None or self.global_costmap_meta is None:
            self.get_logger().debug("No global costmap data available yet")
            return
        
        try:
            # ✅ STEP 1: Get robot position in MAP frame (AMCL-corrected) for accurate cropping
            map_transform = self.tf_buffer.lookup_transform(
                "map", 
                "base_link", 
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0))
            
            robot_x_map = map_transform.transform.translation.x
            robot_y_map = map_transform.transform.translation.y
            
            # ✅ STEP 2: Also get robot position in ODOM frame for grid origin
            odom_transform = self.tf_buffer.lookup_transform(
                "odom", 
                "base_link", 
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0))
            
            robot_x_odom = odom_transform.transform.translation.x
            robot_y_odom = odom_transform.transform.translation.y
            
            self.get_logger().debug(f"Robot position - Map: ({robot_x_map:.2f}, {robot_y_map:.2f}), Odom: ({robot_x_odom:.2f}, {robot_y_odom:.2f})")
            
            # ✅ STEP 3: Calculate window bounds in MAP frame for accurate cropping
            window_origin_x_map = robot_x_map - (self.grid_size / 2.0)
            window_origin_y_map = robot_y_map - (self.grid_size / 2.0)
            window_end_x_map = robot_x_map + (self.grid_size / 2.0)
            window_end_y_map = robot_y_map + (self.grid_size / 2.0)
            
            # ✅ STEP 4: Convert MAP coordinates to pixel coordinates for cropping
            map_origin_x = self.global_costmap_meta.origin.position.x
            map_origin_y = self.global_costmap_meta.origin.position.y
            map_resolution = self.global_costmap_meta.resolution
            
            # Calculate pixel bounds for extraction (using MAP coordinates)
            min_px = int((window_origin_x_map - map_origin_x) / map_resolution)
            max_px = int((window_end_x_map - map_origin_x) / map_resolution)
            min_py = int((window_origin_y_map - map_origin_y) / map_resolution)
            max_py = int((window_end_y_map - map_origin_y) / map_resolution)
            
            # Make sure we don't go outside the global costmap
            min_px = max(0, min_px)
            max_px = min(self.global_costmap_data.shape[1] - 1, max_px)
            min_py = max(0, min_py)
            max_py = min(self.global_costmap_data.shape[0] - 1, max_py)
            
            self.get_logger().debug(f"Cropping from map region: ({window_origin_x_map:.2f}, {window_origin_y_map:.2f}) to ({window_end_x_map:.2f}, {window_end_y_map:.2f})")
            self.get_logger().debug(f"Pixel region: ({min_px}, {min_py}) to ({max_px}, {max_py})")
            
            # ✅ STEP 5: Extract data from global costmap (using MAP-based coordinates)
            if max_px > min_px and max_py > min_py:
                extracted_data = self.global_costmap_data[min_py:max_py+1, min_px:max_px+1]
                
                # Resize to exactly our target grid size (50x50)
                resized_data = cv2.resize(
                    extracted_data.astype(np.uint8),
                    (self.width, self.height),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int8)
                
                # Debug: Log what we extracted
                unique_values = np.unique(resized_data)
                max_value = np.max(resized_data)
                self.get_logger().debug(f"Extracted data - Max: {max_value}, Unique values: {unique_values[:10]}")
                
            else:
                # No valid data to extract - create empty grid
                resized_data = np.zeros((self.height, self.width), dtype=np.int8)
                self.get_logger().debug("No valid region to extract - using empty grid")
            
            # ✅ STEP 6: Process the data (convert to binary)
            binary_data = np.zeros_like(resized_data, dtype=np.int8)
            binary_data[resized_data > 20] = 1  # Threshold for obstacles
            
            # Optional: Apply rectangle filling
            # processed_grid = self.fill_obstacle_rectangles(binary_data)
            processed_grid = binary_data  # Skip rectangle filling for now
            # ✅ STEP 7: Create occupancy grid message in ODOM frame
            occ_msg = OccupancyGrid()
            occ_msg.header.stamp = self.get_clock().now().to_msg()
            occ_msg.header.frame_id = "odom"  # Grid is published in odom frame
            
            # Set grid metadata
            occ_msg.info.resolution = self.resolution
            occ_msg.info.width = self.width
            occ_msg.info.height = self.height
            
            # ✅ STEP 8: Calculate grid origin in ODOM frame
            # The grid represents the area around the robot in odom frame
            window_origin_x_odom = robot_x_odom - (self.grid_size / 2.0)
            window_origin_y_odom = robot_y_odom - (self.grid_size / 2.0)
            
            occ_msg.info.origin.position.x = window_origin_x_odom
            occ_msg.info.origin.position.y = window_origin_y_odom
            occ_msg.info.origin.position.z = 0.0
            
            # Identity orientation (grid aligned with odom frame)
            occ_msg.info.origin.orientation.w = 1.0
            occ_msg.info.origin.orientation.x = 0.0
            occ_msg.info.origin.orientation.y = 0.0
            occ_msg.info.origin.orientation.z = 0.0
            
            # ✅ STEP 9: Convert to ROS format and publish
            processed_ros = np.zeros_like(processed_grid, dtype=np.int8)
            processed_ros[processed_grid == 1] = 100
            
            occ_msg.data = processed_ros.flatten().tolist()
            self.occ_pub.publish(occ_msg)
            
            # ✅ STEP 10: Publish JSON format
            grid_dict = {
                "header": {
                    "frame_id": occ_msg.header.frame_id,
                    "stamp_sec": occ_msg.header.stamp.sec,
                    "stamp_nsec": occ_msg.header.stamp.nanosec
                },
                "info": {
                    "resolution": occ_msg.info.resolution,
                    "width": occ_msg.info.width,
                    "height": occ_msg.info.height,
                    "origin": {
                        "x": occ_msg.info.origin.position.x,
                        "y": occ_msg.info.origin.position.y,
                        "z": occ_msg.info.origin.position.z,
                        "orientation": {
                            "x": occ_msg.info.origin.orientation.x,
                            "y": occ_msg.info.origin.orientation.y,
                            "z": occ_msg.info.origin.orientation.z,
                            "w": occ_msg.info.origin.orientation.w
                        }
                    }
                },
                "data": processed_grid.flatten().tolist()
            }
            
            grid_json_str = json.dumps(grid_dict)
            json_msg = String()
            json_msg.data = grid_json_str
            self.occ_json_pub.publish(json_msg)
            
            self.get_logger().debug(f"Published AMCL-corrected crop in odom frame")
            
        except TransformException as e:
            self.get_logger().warn(f"Transform failed: {e}")

    def fill_obstacle_rectangles(self, grid):
        """
        Find clusters of obstacles where cells are within 4 grid cells of each other,
        then fill bounding rectangles for each cluster, and add one more lap around each cluster
        ONLY if the cluster is not on the edge of the global map.
        
        Args:
            grid: 2D numpy array where 1 = occupied, 0 = free/unknown
            
        Returns:
            Processed grid with filled rectangles for each obstacle cluster plus extra border (where applicable)
        """
        # Make a copy to avoid modifying the original during processing
        processed_grid = grid.copy()
        height, width = grid.shape
        
        # Keep track of cells we've already assigned to clusters
        visited = np.zeros_like(grid, dtype=bool)
        
        # Find all clusters of occupied cells
        clusters = []
        
        for y in range(height):
            for x in range(width):
                # Skip if cell is free or already processed
                if grid[y, x] == 0 or visited[y, x]:
                    continue
                    
                # Found an unprocessed obstacle cell - start a new cluster
                cluster = []
                queue = [(y, x)]
                visited[y, x] = True
                
                # BFS to find all connected and nearby (within 4 cells) obstacle cells
                while queue:
                    cy, cx = queue.pop(0)
                    cluster.append((cy, cx))
                    
                    # Check cells within ±4 grid cells in both directions
                    for ny in range(max(0, cy-2), min(height, cy+3)):
                        for nx in range(max(0, cx-2), min(width, cx+3)):
                            # Calculate Manhattan distance
                            distance = abs(ny - cy) + abs(nx - cx)
                            
                            # Only consider cells within 4 grid cells distance
                            if distance <= 4:
                                # Skip if already visited or not an obstacle
                                if visited[ny, nx] or grid[ny, nx] == 0:
                                    continue
                                    
                                # Mark as visited and add to queue
                                visited[ny, nx] = True
                                queue.append((ny, nx))
                
                # Add this cluster to our list
                if cluster:
                    clusters.append(cluster)
        
        # Fill bounding rectangles for each cluster
        for cluster in clusters:
            if not cluster:
                continue
                
            # Find bounding rectangle
            min_y = min(y for y, _ in cluster)
            max_y = max(y for y, _ in cluster)
            min_x = min(x for _, x in cluster)
            max_x = max(x for _, x in cluster)
            
            # Fill the entire bounding rectangle
            processed_grid[min_y:max_y+1, min_x:max_x+1] = 1
        
        return processed_grid


def main(args=None):
    rclpy.init(args=args)
    node = StaticMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()