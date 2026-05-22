import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros
from nav_msgs.msg import OccupancyGrid


class MapOdomRepublisher(Node):
    def __init__(self):
        super().__init__('map_odom_republisher')
        
        # TF broadcaster to publish our corrected transform
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Subscribe to the map topic to know when SLAM updates
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10)
        
        # Create a transform listener to get the transform from slam_toolbox
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Timer to regularly publish the transform
        self.timer = self.create_timer(0.1, self.publish_transform)
        
        self.get_logger().info('Map-odom transform republisher started')

    def map_callback(self, msg):
        # Map was updated, but we don't need to do anything specific here
        pass

    def publish_transform(self):
        try:
            # Try to get the transform from slam_toolbox's internal calculations
            trans = self.tf_buffer.lookup_transform('map', 'odom', rclpy.time.Time())
            
            # Create a new transform with current timestamp
            new_transform = TransformStamped()
            new_transform.header.stamp = self.get_clock().now().to_msg()
            new_transform.header.frame_id = 'map'
            new_transform.child_frame_id = 'odom'
            
            # Copy the transform values
            new_transform.transform.translation = trans.transform.translation
            new_transform.transform.rotation = trans.transform.rotation
            
            # Publish with correct timestamp
            self.tf_broadcaster.sendTransform(new_transform)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            # If slam_toolbox hasn't published a transform yet, publish identity
            self.get_logger().debug(f'Could not get transform: {e}')
            
            # Publish identity transform as fallback
            identity = TransformStamped()
            identity.header.stamp = self.get_clock().now().to_msg()
            identity.header.frame_id = 'map'
            identity.child_frame_id = 'odom'
            identity.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(identity)


def main(args=None):
    rclpy.init(args=args)
    node = MapOdomRepublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()