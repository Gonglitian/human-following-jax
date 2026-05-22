from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    target_source = LaunchConfiguration('target_source')
    max_speed = LaunchConfiguration('max_speed')
    max_delta_v = LaunchConfiguration('max_delta_v')
    default_target_distance = LaunchConfiguration('default_target_distance')
    model_weight_file = LaunchConfiguration('model_weight_file')
    adaptive_mapping = LaunchConfiguration('adaptive_mapping')

    return LaunchDescription([
        DeclareLaunchArgument('target_source', default_value='uwb_camera',
                              description="'uwb_camera' or 'closest_lidar'"),
        DeclareLaunchArgument('max_speed', default_value='1.0'),
        DeclareLaunchArgument('max_delta_v', default_value='0.5'),
        DeclareLaunchArgument('default_target_distance', default_value='2.0'),
        DeclareLaunchArgument('model_weight_file', default_value='meta_4.pt'),
        DeclareLaunchArgument('adaptive_mapping', default_value='true'),
        Node(
            package='decider',
            executable='decider',
            name='decider_node',
            output='screen',
            parameters=[{
                'target_source': target_source,
                'max_speed': max_speed,
                'max_delta_v': max_delta_v,
                'default_target_distance': default_target_distance,
                'model_weight_file': model_weight_file,
                'adaptive_mapping': adaptive_mapping,
            }],
        ),
    ])
