"""
AWS RoboMaker small warehouse scenario for following_sim.

The base world is `no_roof_small_warehouse.world` from
https://github.com/aws-robotics/aws-robomaker-small-warehouse-world, copied
verbatim into following_sim/worlds/warehouse.world so HuNavSim's generator can
rewrite it without touching the upstream clone. The AWS models/ directory lives
outside the install space, so we extend GAZEBO_MODEL_PATH here before handing
off to bringup.launch.py.
"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare


# Source checkout of aws-robomaker-small-warehouse-world. The repo is cloned
# under ros2_following/ (sibling of this package's source tree); we resolve it
# from this file's location so the launch keeps working whether it's run from
# the source dir or from install/.
_AWS_MODELS_DIR = (
    Path(__file__).resolve().parents[2]
    / 'aws-robomaker-small-warehouse-world' / 'models'
)


def generate_launch_description():
    rviz_arg = DeclareLaunchArgument('rviz', default_value='false')
    gui_arg = DeclareLaunchArgument('gui', default_value='true')

    # Prepend the AWS models dir so `<uri>model://aws_robomaker_warehouse_*</uri>`
    # resolves when gzserver loads warehouse.world.
    set_aws_models = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=[str(_AWS_MODELS_DIR), ':',
               EnvironmentVariable('GAZEBO_MODEL_PATH', default_value='')],
    )

    bringup = PathJoinSubstitution([
        FindPackageShare('following_sim'), 'launch', 'bringup.launch.py',
    ])

    return LaunchDescription([
        rviz_arg, gui_arg,
        set_aws_models,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(bringup),
            launch_arguments={
                'scenario': 'warehouse',
                # Main aisle, south end — inside warehouse walls. Target
                # spawns 2 m north at (-3, -5); detect_range=5 m so the
                # match fires on the first /tracked_objects_json tick.
                'robot_x': '-3.0',
                'robot_y': '-7.0',
                'robot_yaw': '1.5708',
                'rviz': LaunchConfiguration('rviz'),
                'gui': LaunchConfiguration('gui'),
                'run_name': 'warehouse',
                # AWS warehouse has 30+ heavy meshes; gzserver+gzclient
                # loading them starves the hunav<->generator service
                # handshake. Stagger everything further out.
                'worldgen_delay': '5.0',
                'gazebo_delay': '10.0',
                'spawn_delay': '28.0',
                'bridge_delay': '30.0',
                'perception_delay': '32.0',
            }.items(),
        ),
    ])
