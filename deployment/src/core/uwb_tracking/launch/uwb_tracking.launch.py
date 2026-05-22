#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('uwb_tracking')
    
    # Path to the config file
    config_file = os.path.join(pkg_dir, 'config', 'uwb_config.yaml')
    
    # Create the uwb_tracking node
    uwb_tracking_node = Node(
        package='uwb_tracking',
        executable='uwb_tracking',
        name='uwb_tracking_node',
        parameters=[config_file],
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        uwb_tracking_node
    ]) 