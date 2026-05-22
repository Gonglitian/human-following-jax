from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('depth_costmap')
    config_file = os.path.join(pkg_share, 'config', 'depth_costmap.yaml')

    depth_costmap_node = Node(
        package='depth_costmap',
        executable='depth_costmap',
        name='depth_costmap',
        output='screen',
        parameters=[config_file],
    )

    return LaunchDescription([
        depth_costmap_node,
    ])
