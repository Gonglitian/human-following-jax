from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rviz_arg = DeclareLaunchArgument('rviz', default_value='false',
        description='Pass through to bringup.launch.py')
    gui_arg = DeclareLaunchArgument('gui', default_value='true',
        description='Pass through to bringup.launch.py')
    bringup = PathJoinSubstitution([
        FindPackageShare('following_sim'), 'launch', 'bringup.launch.py',
    ])
    return LaunchDescription([
        rviz_arg, gui_arg,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(bringup),
            launch_arguments={
                'scenario': 'occlusion',
                'robot_x': '-9.5',
                'robot_y': '0.0',
                'robot_yaw': '0.0',
                'rviz': LaunchConfiguration('rviz'),
                'gui': LaunchConfiguration('gui'),
                'run_name': 'occlusion',
            }.items(),
        ),
    ])
