import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('camera_detector'),
        'config',
        'camera_detector.yaml',
    )

    camera_detector_node = Node(
        package='camera_detector',
        executable='camera_detector',
        name='camera_detector',
        output='screen',
        parameters=[config],
    )

    return LaunchDescription([
        camera_detector_node,
    ])
