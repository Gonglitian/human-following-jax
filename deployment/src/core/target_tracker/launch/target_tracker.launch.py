import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('target_tracker'),
        'config',
        'target_tracker.yaml'
    )

    return LaunchDescription([
        Node(
            package='target_tracker',
            executable='target_tracker',
            name='target_tracker',
            parameters=[config],
            output='screen',
        ),
    ])
