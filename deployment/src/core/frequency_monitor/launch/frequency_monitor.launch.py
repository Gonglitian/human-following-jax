from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the package directory
    package_dir = get_package_share_directory('frequency_monitor')

    # # Paths to the parameter files
    # sort_tracker_params = os.path.join(package_dir, 'config', 'sort_tracker.yaml')
    # topics_params = os.path.join(package_dir, 'config', 'topics.yaml')

    return LaunchDescription([
        Node(
            package='frequency_monitor',
            executable='frequency_monitor',  # Matches the entry point in setup.py
            name='frequency_monitor_node',
            output='screen',
            parameters=[]#sort_tracker_params, topics_params]
        )
    ])

