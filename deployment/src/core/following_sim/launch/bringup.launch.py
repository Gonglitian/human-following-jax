#!/usr/bin/env python3
"""
following_sim master bringup.

Chain:
  hunav_loader (agents YAML -> ROS params)
    -> hunav_gazebo_world_generator (injects actors + HuNavPlugin into base world)
       -> gzserver + gzclient (loads generated world)
          -> spawn rosmaster_x3 URDF at requested pose
             -> target_to_uwb_bridge (/human_states -> /uwb/tag_0/position)
             -> metrics_recorder
             -> hunav_agent_manager, hunav_evaluator
             -> dr_spaam, sort_tracker, predictor, occupancy, decider, command_listener
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
    RegisterEventHandler, TimerAction, SetEnvironmentVariable, LogInfo, Shutdown,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration, PathJoinSubstitution, PythonExpression, Command,
    EnvironmentVariable,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_following_sim = FindPackageShare('following_sim')

    # -------------------- Launch args --------------------
    scenario_arg = DeclareLaunchArgument(
        'scenario', default_value='corridor',
        description='One of: corridor, junction, crowd, occlusion, sharp_turn')
    gui_arg = DeclareLaunchArgument(
        'gui', default_value='true',
        description='Launch gzclient (set false for headless)')
    launch_perception_arg = DeclareLaunchArgument(
        'launch_perception', default_value='true',
        description='Launch DR-SPAAM, tracker, predictor, occupancy, decider')
    robot_x_arg = DeclareLaunchArgument('robot_x', default_value='-5.0')
    robot_y_arg = DeclareLaunchArgument('robot_y', default_value='0.0')
    robot_yaw_arg = DeclareLaunchArgument('robot_yaw', default_value='0.0')
    target_name_arg = DeclareLaunchArgument('target_name', default_value='target')
    metrics_dir_arg = DeclareLaunchArgument(
        'metrics_dir', default_value='/tmp/following_sim_metrics')
    run_name_arg = DeclareLaunchArgument('run_name', default_value='')
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='false',
        description='Launch RViz2 with the following_sim layout')
    # Heavy worlds (AWS warehouse) need longer gaps so gzserver finishes
    # loading its meshes before the generator/agent_manager handshake and
    # before spawn_entity races the physics init.
    worldgen_delay_arg = DeclareLaunchArgument('worldgen_delay', default_value='2.0')
    gazebo_delay_arg = DeclareLaunchArgument('gazebo_delay', default_value='2.0')
    spawn_delay_arg = DeclareLaunchArgument('spawn_delay', default_value='6.0')
    bridge_delay_arg = DeclareLaunchArgument('bridge_delay', default_value='7.0')
    perception_delay_arg = DeclareLaunchArgument('perception_delay', default_value='8.0')
    # Observability layer: every default-on, all overridable.
    enable_freq_monitor_arg = DeclareLaunchArgument('enable_freq_monitor', default_value='true')
    enable_tf_health_arg = DeclareLaunchArgument('enable_tf_health', default_value='true')
    enable_cmd_watchdog_arg = DeclareLaunchArgument('enable_cmd_watchdog', default_value='true')
    enable_rosbag_arg = DeclareLaunchArgument('enable_rosbag', default_value='true')
    rosbag_dir_arg = DeclareLaunchArgument(
        'rosbag_dir', default_value='/tmp/following_sim_bags',
        description='Parent dir for rosbag2 recordings; one subdir per run.')
    # Baseline / ablation switches.
    # policy: meta (Ours / Meta-NoMap), orca (SG-ORCA), mpc (MPC-ADC), rlpc (RL-PC)
    policy_arg = DeclareLaunchArgument(
        'policy', default_value='meta',
        description='Decider policy: meta | orca | mpc | rlpc | crl')
    # adaptive_mapping only applies to policy=meta. False -> Meta-NoMap baseline.
    adaptive_mapping_arg = DeclareLaunchArgument(
        'adaptive_mapping', default_value='true',
        description='When policy=meta: enable closed-loop adaptive mapping')
    # Which ckpt to load from share/decider/model_weight/. Filename only.
    # meta_4.pt = Ours; meta_nomap.pt = paper-trained Meta-NoMap;
    # rl_pc.pt = RL-PC. Ignored for orca/mpc (no ckpt).
    model_weight_file_arg = DeclareLaunchArgument(
        'model_weight_file', default_value='meta_4.pt',
        description='ckpt filename inside decider/model_weight/')

    scenario = LaunchConfiguration('scenario')
    gui = LaunchConfiguration('gui')
    launch_perception = LaunchConfiguration('launch_perception')
    robot_x = LaunchConfiguration('robot_x')
    robot_y = LaunchConfiguration('robot_y')
    robot_yaw = LaunchConfiguration('robot_yaw')
    target_name = LaunchConfiguration('target_name')
    metrics_dir = LaunchConfiguration('metrics_dir')
    run_name = LaunchConfiguration('run_name')
    worldgen_delay = LaunchConfiguration('worldgen_delay')
    gazebo_delay = LaunchConfiguration('gazebo_delay')
    spawn_delay = LaunchConfiguration('spawn_delay')
    bridge_delay = LaunchConfiguration('bridge_delay')
    perception_delay = LaunchConfiguration('perception_delay')
    enable_freq_monitor = LaunchConfiguration('enable_freq_monitor')
    enable_tf_health = LaunchConfiguration('enable_tf_health')
    enable_cmd_watchdog = LaunchConfiguration('enable_cmd_watchdog')
    enable_rosbag = LaunchConfiguration('enable_rosbag')
    rosbag_dir = LaunchConfiguration('rosbag_dir')
    policy = LaunchConfiguration('policy')
    adaptive_mapping = LaunchConfiguration('adaptive_mapping')
    model_weight_file = LaunchConfiguration('model_weight_file')
    rviz_cfg = PathJoinSubstitution([
        pkg_following_sim, 'rviz', 'following_sim.rviz',
    ])

    # -------------------- Paths --------------------
    agents_yaml = PathJoinSubstitution([
        pkg_following_sim, 'config',
        PythonExpression(["'agents_' + '", scenario, "' + '.yaml'"]),
    ])
    base_world = PathJoinSubstitution([
        pkg_following_sim, 'worlds',
        PythonExpression(["'", scenario, "' + '.world'"]),
    ])
    urdf_xacro = PathJoinSubstitution([
        pkg_following_sim, 'urdf', 'rosmaster_x3.urdf.xacro',
    ])

    # Expose our worlds as Gazebo resources so `<include>` can find them.
    worlds_dir = PathJoinSubstitution([pkg_following_sim, 'worlds'])
    meshes_dir = PathJoinSubstitution([pkg_following_sim, 'meshes'])
    # HuNav actor skins (.dae) live in hunav_gazebo_wrapper source tree, not in
    # its install share. Add the source media dir so Gazebo can find them.
    hunav_media = '/home/lee/human-following/ros2_following/hunav_gazebo_wrapper/media'
    hunav_models = '/home/lee/human-following/ros2_following/hunav_gazebo_wrapper/media/models'
    set_gazebo_resource = SetEnvironmentVariable(
        name='GAZEBO_RESOURCE_PATH',
        value=[EnvironmentVariable('GAZEBO_RESOURCE_PATH', default_value=''),
               ':', worlds_dir, ':', meshes_dir,
               ':', hunav_media, ':', hunav_models],
    )
    set_gazebo_model = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=[EnvironmentVariable('GAZEBO_MODEL_PATH', default_value=''),
               ':', meshes_dir, ':', hunav_media, ':', hunav_models],
    )

    # -------------------- HuNavSim loader + world generator --------------------
    hunav_loader = Node(
        package='hunav_agent_manager',
        executable='hunav_loader',
        output='screen',
        parameters=[agents_yaml],
    )

    hunav_worldgen = Node(
        package='hunav_gazebo_wrapper',
        executable='hunav_gazebo_world_generator',
        output='screen',
        parameters=[{
            'base_world': base_world,
            'use_gazebo_obs': True,
            # 100 Hz tick piles up async service requests faster than
            # hunav_agent_manager's executor answers them in our env. 15 Hz is
            # enough for smooth actor motion.
            'update_rate': 15.0,
            'robot_name': 'rosmaster_x3',
            'global_frame_to_publish': 'map',
            'use_navgoal_to_start': False,
            'navgoal_topic': '/goal_pose',
            # upstream declares this as a single string of space-separated model names
            'ignore_models': 'ground_plane',
        }],
    )
    hunav_worldgen_event = RegisterEventHandler(OnProcessStart(
        target_action=hunav_loader,
        on_start=[
            LogInfo(msg='hunav_loader up; starting world generator...'),
            TimerAction(period=worldgen_delay, actions=[hunav_worldgen]),
        ],
    ))

    # hunav_gazebo_world_generator writes `generatedWorld.world` into the
    # directory containing the base_world it was given. Since our base_world
    # lives in following_sim/share/following_sim/worlds/, the generated world
    # ends up there too.
    generated_world = PathJoinSubstitution([
        pkg_following_sim, 'worlds', 'generatedWorld.world',
    ])

    # Gazebo's own shader/media path is exported by /usr/share/gazebo/setup.sh.
    # Wrap the launch so the subshell sources it before gzserver/gzclient.
    gzserver = ExecuteProcess(
        cmd=['bash', '-c',
             'source /usr/share/gazebo/setup.sh && '
             'exec gzserver --verbose "$1" '
             '-s libgazebo_ros_init.so -s libgazebo_ros_factory.so',
             '_', generated_world],
        output='screen', on_exit=Shutdown(),
    )
    gzclient = ExecuteProcess(
        cmd=['bash', '-c', 'source /usr/share/gazebo/setup.sh && exec gzclient'],
        output='screen', condition=IfCondition(gui), on_exit=Shutdown(),
    )
    gazebo_event = RegisterEventHandler(OnProcessStart(
        target_action=hunav_worldgen,
        on_start=[
            LogInfo(msg='world generated; starting Gazebo...'),
            TimerAction(period=gazebo_delay, actions=[gzserver, gzclient]),
        ],
    ))

    # -------------------- Robot state + spawn --------------------
    # ParameterValue(value_type=str) stops launch from trying to parse the
    # xacro-expanded XML as YAML.
    robot_description = ParameterValue(
        Command(['xacro ', urdf_xacro]), value_type=str)

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description,
                     'use_sim_time': True}],
    )

    spawn_robot = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=[
            '-entity', 'rosmaster_x3',
            '-topic', 'robot_description',
            '-x', robot_x, '-y', robot_y, '-z', '0.05',
            '-Y', robot_yaw,
        ],
        output='screen',
    )

    # map->odom is assumed identity unless SLAM is active. HuNavSim default
    # global frame is 'map'; decider consumes odom. Publishing this static
    # transform lets the target-to-UWB bridge and downstream TF lookups work.
    static_map_odom = Node(
        package='tf2_ros', executable='static_transform_publisher',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
    )

    # -------------------- HuNav behavior + evaluator --------------------
    hunav_manager = Node(
        package='hunav_agent_manager', executable='hunav_agent_manager',
        name='hunav_agent_manager', output='screen',
        parameters=[{'use_sim_time': True}],
    )

    metrics_conf = PathJoinSubstitution([
        FindPackageShare('hunav_evaluator'), 'config', 'metrics.yaml'])
    hunav_evaluator = Node(
        package='hunav_evaluator', executable='hunav_evaluator_node',
        output='screen', parameters=[metrics_conf],
    )

    # -------------------- Our bridge + metrics --------------------
    target_bridge = Node(
        package='following_sim', executable='target_to_uwb_bridge',
        name='target_to_uwb_bridge', output='screen',
        parameters=[{
            'target_name': target_name,
            'source_frame': 'map',
            'odom_frame': 'odom',
        }],
    )

    human_states_viz = Node(
        package='following_sim', executable='human_states_viz',
        name='human_states_viz', output='screen',
        parameters=[{
            'target_name': target_name,
            'source_frame': 'map',
            'output_frame': 'odom',
        }],
    )

    rviz = Node(
        package='rviz2', executable='rviz2',
        name='rviz2', output='screen',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    metrics_node = Node(
        package='following_sim', executable='metrics_recorder',
        name='metrics_recorder', output='screen',
        parameters=[{
            'target_name': target_name,
            'output_dir': metrics_dir,
            'run_name': run_name,
        }],
    )

    # -------------------- Perception/decision pipeline --------------------
    # dr_spaam config lives in-tree. We forward /scan from the sim.
    dr_spaam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('dr_spaam_ros2'), 'launch',
            'dr_spaam_ros2.launch.py',
        ])),
        condition=IfCondition(launch_perception),
    )

    # sort_tracker, predictor, occupancy_generation, decider, command_listener
    # are assumed to have a launch file each. We start them as bare nodes so
    # this bringup stays robust even if some launch files are missing.
    # Executable names come from each package's setup.py console_scripts.
    sort_tracker = Node(
        package='sort_tracker', executable='sort_tracker',
        name='sort_tracker', output='screen',
        parameters=[{
            'use_sim_time': True,
            # In sim we cannot rely on DR-SPAAM alone (Gazebo <actor> has no
            # collision so the NN returns nothing; only the UWB-injected
            # target makes it into /dr_spaam_detections). The detections_merger
            # fuses DR-SPAAM + HuNav ground truth into /combined_detections.
            'subscriber.detections.topic': '/combined_detections',
        }],
        condition=IfCondition(launch_perception),
    )

    detections_merger = Node(
        package='following_sim', executable='detections_merger',
        name='detections_merger', output='screen',
        parameters=[{
            'source_frame': 'map',
            'output_frame': 'odom',
            'dedup_radius': 0.6,
        }],
        condition=IfCondition(launch_perception),
    )
    predictor = Node(
        package='predictor', executable='predictor',
        name='predictor', output='screen', parameters=[{'use_sim_time': True}],
        condition=IfCondition(launch_perception),
    )
    occupancy = Node(
        package='occupancy_generation', executable='occupancy_generation',
        name='occupancy_grid', output='screen', parameters=[{'use_sim_time': True}],
        condition=IfCondition(launch_perception),
    )
    # Decider: select baseline via `policy:=meta|orca|mpc`. Only one launches;
    # the other two are gated out by IfCondition. All three publish /cmd_vel
    # and consume the same upstream perception topics.
    decider_meta = Node(
        package='decider', executable='decider',
        name='decider', output='screen',
        parameters=[{
            'use_sim_time': True,
            'adaptive_mapping': adaptive_mapping,
            'model_weight_file': model_weight_file,
        }],
        condition=IfCondition(PythonExpression([
            "'", launch_perception, "' == 'true' and '", policy, "' == 'meta'",
        ])),
    )
    decider_orca = Node(
        package='decider', executable='decider_orca',
        name='decider', output='screen',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(PythonExpression([
            "'", launch_perception, "' == 'true' and '", policy, "' == 'orca'",
        ])),
    )
    decider_mpc = Node(
        package='decider', executable='decider_mpc',
        name='decider', output='screen',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(PythonExpression([
            "'", launch_perception, "' == 'true' and '", policy, "' == 'mpc'",
        ])),
    )
    decider_rlpc = Node(
        package='decider', executable='decider_rlpc',
        name='decider', output='screen',
        parameters=[{
            'use_sim_time': True,
            'model_weight_file': model_weight_file,
        }],
        condition=IfCondition(PythonExpression([
            "'", launch_perception, "' == 'true' and '", policy, "' == 'rlpc'",
        ])),
    )
    decider_crl = Node(
        package='decider', executable='decider_crl',
        name='decider', output='screen',
        parameters=[{
            'use_sim_time': True,
            'model_weight_file': model_weight_file,
        }],
        condition=IfCondition(PythonExpression([
            "'", launch_perception, "' == 'true' and '", policy, "' == 'crl'",
        ])),
    )
    command_listener = Node(
        package='command_listener', executable='command_listener',
        name='command_listener', output='screen',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(launch_perception),
    )

    # -------------------- Observability layer --------------------
    # Hz logger for the four hot topics + dr_spaam->cmd_vel latency.
    freq_monitor = Node(
        package='frequency_monitor', executable='frequency_monitor',
        name='frequency_monitor', output='screen',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(enable_freq_monitor),
    )
    tf_health = Node(
        package='following_sim', executable='tf_health_monitor',
        name='tf_health_monitor', output='screen',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(enable_tf_health),
    )
    cmd_watchdog = Node(
        package='following_sim', executable='cmd_vel_watchdog',
        name='cmd_vel_watchdog', output='screen',
        parameters=[{'use_sim_time': True, 'timeout': 1.0}],
        condition=IfCondition(enable_cmd_watchdog),
    )
    # rosbag2 record: per-run subdir, only topics we'd want to replay.
    # Default sqlite3 storage; switch to mcap by installing
    # ros-humble-rosbag2-storage-mcap and adding `-s mcap` if you prefer.
    rosbag_record = ExecuteProcess(
        cmd=['bash', '-c',
             'mkdir -p "$0" && cd "$0" && '
             'exec ros2 bag record '
             '-o "run_$(date +%Y%m%d_%H%M%S)" '
             '/scan /odom /cmd_vel /tf /tf_static '
             '/human_states /uwb/tag_0/position /uwb/tag_1/position '
             '/dr_spaam_detections /combined_detections '
             '/tracked_objects /tracked_objects_json '
             '/predictions_json /occupancy_grid /command '
             '/cmd_vel_watchdog/state',
             rosbag_dir],
        output='screen', condition=IfCondition(enable_rosbag),
    )

    return LaunchDescription([
        scenario_arg, gui_arg, launch_perception_arg,
        robot_x_arg, robot_y_arg, robot_yaw_arg,
        target_name_arg, metrics_dir_arg, run_name_arg, rviz_arg,
        worldgen_delay_arg, gazebo_delay_arg, spawn_delay_arg,
        bridge_delay_arg, perception_delay_arg,
        enable_freq_monitor_arg, enable_tf_health_arg,
        enable_cmd_watchdog_arg, enable_rosbag_arg, rosbag_dir_arg,
        policy_arg, adaptive_mapping_arg, model_weight_file_arg,

        set_gazebo_resource, set_gazebo_model,

        hunav_loader,
        hunav_worldgen_event,
        gazebo_event,

        rsp,
        static_map_odom,
        TimerAction(period=spawn_delay, actions=[spawn_robot]),

        hunav_manager,
        hunav_evaluator,

        TimerAction(period=bridge_delay, actions=[target_bridge, human_states_viz, metrics_node]),

        TimerAction(period=perception_delay, actions=[
            dr_spaam_launch, detections_merger, sort_tracker, predictor, occupancy,
            # All three decider variants gated on `policy:=`; only the matching
            # one actually launches.
            decider_meta, decider_orca, decider_mpc, decider_rlpc, decider_crl, command_listener,
            # Observability — start when perception comes up so they actually
            # have publishers to monitor; rosbag itself starts a bit earlier so
            # we capture spawn-time TF too.
            freq_monitor, tf_health, cmd_watchdog,
        ]),

        # rosbag2 starts as soon as Gazebo is up so we capture TF setup.
        TimerAction(period=gazebo_delay, actions=[rosbag_record]),

        # RViz can come up immediately — it will connect to topics as they appear.
        rviz,
    ])
