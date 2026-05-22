# following_sim

Gazebo Classic + HuNavSim evaluation harness for the human-following Meta RL
policy (`decider` / `meta_4.pt`). Provides:

- A classic-Gazebo URDF of the Yahboom ROSMASTER X3 with a 2D LiDAR whose
  parameters match the real RPLidar S2 (`panoramic_scan: True`, 720 rays
  over 360°, 0.2–30 m, 10 Hz, `frame_id: laser_frame`, topic `/scan`). So
  `dr_spaam_ros2` needs zero changes between sim and real.
- Five scenario worlds: `corridor`, `junction`, `crowd`, `occlusion`,
  `sharp_turn`.
- HuNavSim agent configs per scenario — one "target" actor plus bystanders
  where appropriate.
- `target_to_uwb_bridge`: subscribes `/human_states`, picks the target
  agent, transforms its pose `map → odom`, republishes as
  `/uwb/tag_0/position`. Reuses the existing UWB pseudo-human injection path
  in DR-SPAAM; the decider sees the target exactly the way it does on the
  real robot.
- `metrics_recorder`: CSV logger for following-distance-error vs current
  preference. Complements `hunav_evaluator` (social metrics).
- Per-scenario launch files + a master `bringup.launch.py`.

## Environment

| Component | Required |
| - | - |
| Ubuntu | 22.04 |
| ROS 2 | **Humble** |
| Gazebo | **Classic 11** (HuNavSim's `HuNavPlugin` is built against it) |
| GPU | optional |

## Install

```bash
# 1. clone HuNavSim + its Gazebo wrapper alongside this package
cd ros2_following
./following_sim/scripts/install_sim_deps.sh

# 2. system deps
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-xacro \
                 ros-humble-robot-state-publisher ros-humble-tf2-ros

# 3. place the trained checkpoint where the decider expects it
mkdir -p decider/model_weight
cp <your-downloaded>/meta_4.pt decider/model_weight/

# 4. build
cd ..                  # back to repo root
colcon build --symlink-install --base-paths ros2_following
source ros2_following/install/setup.bash
```

> **Training/deployment consistency note**. The repo-wide
> `sim.human_num` in `ros2_following/decider/config/config.py` is 50, but
> the training config (`human-following-robot/crowd_nav/configs/config.py`)
> is 40. `meta_4.pt`'s `spatial_edges` transformer was sized for whichever
> value was set at training time. Match them before the first run (edit
> `ros2_following/decider/config/config.py`), or your policy will silently
> mask the wrong number of humans.

## Run

```bash
# corridor (simplest, one target actor walking end-to-end)
ros2 launch following_sim corridor.launch.py

# T-junction with one crossing bystander
ros2 launch following_sim junction.launch.py

# 20x20 m hall with ~10 SFM actors
ros2 launch following_sim crowd.launch.py

# line-of-sight broken by three off-axis pillars
ros2 launch following_sim occlusion.launch.py

# L-shaped corridor with a 90 deg target turn
ros2 launch following_sim sharp_turn.launch.py
```

After Gazebo settles, drive the policy from another terminal:

```bash
# arm the decider
ros2 topic pub --once /command std_msgs/String "data: 'mode:automatic'"
ros2 topic pub --once /command std_msgs/String "data: 'auto:human_following'"

# pick a distance bucket: -2 (closest) .. +2 (farthest)
ros2 topic pub --once /command std_msgs/String "data: 'auto:preference:0'"
# or request a continuous distance; PController maps it to the nearest bucket
ros2 topic pub --once /command std_msgs/String "data: 'auto:distance:2.0'"
```

## Metrics output

Two streams are written in parallel per run:

| File | Produced by | Contents |
| - | - | - |
| `~/hunav_metrics.txt` (default, configurable in `metrics.yaml`) | `hunav_evaluator` | Social-nav metrics: min dist to people, collisions, intimate/personal/social intrusions, path length, SFM forces, ... |
| `/tmp/following_sim_metrics/<scenario>.csv` | `metrics_recorder` | Per-timestep: `t, robot_xy, target_xy, distance, preference, desired_distance, distance_error, v_lin, w_ang`. Allows plotting the follow-distance error against the preference the policy was asked to follow. |

Override the output dir and run name:

```bash
ros2 launch following_sim corridor.launch.py \
     metrics_dir:=$HOME/exp_logs run_name:=corridor_pref1_seed0
```

## Launch args (bringup.launch.py)

| Arg | Default | Description |
| - | - | - |
| `scenario` | `corridor` | one of the five scenario names |
| `gui` | `true` | set `false` for headless `gzserver` only |
| `launch_perception` | `true` | when `false`, only the sim + bridge + metrics run (useful for sanity-checking the world by itself) |
| `robot_x`, `robot_y`, `robot_yaw` | per scenario | initial robot pose in the world frame |
| `target_name` | `target` | HuNav agent name the bridge tracks; must match `agents_*.yaml` |
| `metrics_dir` | `/tmp/following_sim_metrics` | CSV output directory |
| `run_name` | auto timestamp | CSV file stem |

## Topology cheat-sheet

```
 Gazebo /scan (LaserScan, 360°) ─┐
                                 ▼
                       dr_spaam_ros2
                         │     ▲
                         │     └─── /uwb/tag_0/position  ← target_to_uwb_bridge
                         ▼                                      ▲
                   /dr_spaam_detections                          │
                         │                                       │
                         ▼                                       │
                    sort_tracker                                 │
                         │                                       │
                         ▼                                       │
                     predictor                                   │
                         │                                       │
                         ▼                                       │
                      decider (meta_4.pt) ─► /cmd_vel ───► Gazebo│
                                                                 │
  HuNavSim (/human_states hunav_msgs/Agents)  ───────────────────┘
                   │
                   ▼
            hunav_evaluator (social metrics)
            metrics_recorder (follow-distance CSV)
```

## Adding a new scenario

1. Drop a new `worlds/foo.world` (static obstacles only — HuNavSim injects
   actors at runtime).
2. Drop a new `config/agents_foo.yaml` with at minimum one agent named
   `target`.
3. Create `launch/foo.launch.py` copying one of the existing ones and
   editing `scenario: foo`.
