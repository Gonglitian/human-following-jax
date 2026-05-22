from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
def generate_launch_description():
    # Get the package directory
    package_dir = get_package_share_directory('occupancy_generation')

    return LaunchDescription([
        Node(
            package='occupancy_generation',
            executable='static_map_node',  # Matches the entry point in setup.py
            name='static_map_node',
            output='screen',
            parameters=[]
        )
    ])