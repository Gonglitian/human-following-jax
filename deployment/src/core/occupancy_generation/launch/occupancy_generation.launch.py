from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    human_filter_enabled = LaunchConfiguration('human_filter_enabled')
    human_filter_radius = LaunchConfiguration('human_filter_radius')

    return LaunchDescription([
        DeclareLaunchArgument('human_filter_enabled', default_value='false',
                              description='Subtract DR-SPAAM-detected humans from OGM (matches training distribution).'),
        DeclareLaunchArgument('human_filter_radius', default_value='0.4'),
        Node(
            package='occupancy_generation',
            executable='occupancy_generation',
            name='occupancy_generation_node',
            output='screen',
            parameters=[{
                'human_filter_enabled': human_filter_enabled,
                'human_filter_radius': human_filter_radius,
            }],
        ),
    ])
