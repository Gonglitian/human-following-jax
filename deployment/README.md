# Deployment — ROS 2 stack for Rosmaster X3

This is the colcon workspace that runs the human-following policy on a real
**Yahboom Rosmaster X3** mecanum-drive robot. The pipeline is:

```
LiDAR /scan ─► DR-SPAAM ─► SORT tracker ─► CV predictor ─► decider ─► /cmd_vel
                  │                              │
            UWB pseudo-human               occupancy_generation
            injection                      (OGM from /scan)
```

## What is and isn't in this repo

**Vendored (in `src/core/`)** — 13 lab-developed packages:

| Package | Purpose |
|---|---|
| `decider` | Central node. Loads the RL policy, fuses obs, outputs `/cmd_vel`. |
| `predictor` | Constant-velocity trajectory prediction for tracked agents. |
| `target_tracker` | Lock onto a single target human across SORT IDs. |
| `sort_tracker` | SORT multi-object tracker (Kalman + Hungarian). |
| `camera_detector` | RGB-D detector (optional, currently second-source to LiDAR). |
| `uwb_tracking` | LinkTrack UWB serial → Kalman → `/uwb/tag_*/position`. |
| `command_listener` | CLI for mode/preference (`mode:automatic`, `auto:distance:2.0`, …). |
| `depth_costmap` | Depth image → local costmap. |
| `occupancy_generation` | LiDAR → robot-centered 10 m × 10 m OGM (binary). |
| `frequency_monitor` | Logs topic Hz + dr_spaam→cmd_vel latency to CSV. |
| `tf_republisher` | TF bridge (map ↔ odom). |
| `fake_detection` | Sim-only test source. |
| `following_sim` | Lightweight Gazebo + URDF for desktop testing. |

**NOT in this repo** (install separately — too large or third-party):

| Dependency | Where to get | Why excluded |
|---|---|---|
| `dr_spaam_ros2` + `DR-SPAAM-Detector` | https://github.com/VisualComputingInstitute/2D_lidar_person_detection | ~34 MB, third-party |
| `hunav_sim` | https://github.com/robotics-upo/hunav_sim | 172 MB, only needed for crowd simulation |
| `hunav_gazebo_wrapper` | https://github.com/robotics-upo/hunav_gazebo_wrapper | 475 MB Gazebo plugin |
| `lightsfm` | https://github.com/robotics-upo/lightsfm | Social-force C++ lib |
| `people` stack | https://github.com/wg-perception/people | leg_detector + people_msgs |
| `aws-robomaker-*-world` | https://github.com/aws-robotics | ~450 MB of Gazebo worlds |
| `yahboomcar_nav` | Vendor Yahboom (or grab from upstream ros2_following) | Robot URDF/Nav2 config |
| PyTorch model weights (`*.pt`) | **Google Drive** (lab share) | 263 MB; managed out-of-band |
| JAX training checkpoints (`params.pkl`) | Google Drive after each run | Large, per-experiment |

A helper script lives at `scripts/install_third_party.sh` — see "First-time
setup" below.

## First-time setup on a fresh robot

> ROS 2 Foxy or Galactic, Ubuntu 20.04. Robot's NUC/Pi is `pi@192.168.1.11`.

```bash
# 1) Clone main repo
git clone https://github.com/Gonglitian/human-following-jax.git
cd human-following-jax/deployment

# 2) Pull third-party packages into src/third_party/ (script clones them at pinned commits)
bash scripts/install_third_party.sh

# 3) Pull model weights from Google Drive into src/core/decider/model_weight/
bash scripts/pull_models.sh   # prompts for the shared GDrive folder URL

# 4) Build
colcon build --symlink-install
source install/setup.bash
```

## Running

### Real robot

```bash
# On robot (Pi/NUC): bringup
ssh pi@192.168.1.11 'cd ~/yahboomcar_ws && ros2 launch yahboomcar_bringup bringup.launch.py'

# On laptop: full pipeline
ros2 launch deployment start_real_robot.launch.py method:=meta speed:=0.5 pref:=0
```

`method`, `speed`, `pref` env-style args control which policy variant, speed
cap, and following-distance preference are used. See
[`docs/operating.md`](docs/operating.md) for the full matrix.

### Sim only (no robot)

```bash
ros2 launch following_sim sim.launch.py
ros2 launch deployment start_real_robot.launch.py use_sim:=true
```

## Where the policy lives

- **Current production**: PyTorch checkpoint `decider/model_weight/meta_4.pt`,
  loaded by `decider/main.py:260-271`.
- **In flight (JAX)**: trained in `training/` (this repo), produces
  `params.pkl`. Loading into the ROS node is **not yet wired** — see
  [`../DEPLOY.md`](../DEPLOY.md) for the planned bridge (Flax forward call from
  a Python rclpy node).

## Topics — quick reference

| Topic | Type | Notes |
|---|---|---|
| `/scan` | LaserScan | RPLidar A1, 8 Hz |
| `/dr_spaam_detections` | PoseArray | Pedestrian positions (odom frame) |
| `/tracked_objects_json` | String (JSON) | SORT tracks with IDs |
| `/predictions_json` | String (JSON) | CV-predicted trajectories |
| `/occupancy_grid_json` | String (JSON) | Binary 100×100 OGM |
| `/uwb/tag_0/position` | Point | Target human (UWB pseudo-detection) |
| `/cmd_vel` | Twist | Mecanum velocity command |
| `/command` | String | Mode/pref control |

## Troubleshooting

See [`../training/SETUP.md`](../training/SETUP.md) for GPU/training environment.
For robot-side issues:

- **LiDAR motor won't start** → USB reset `usbreset 10c4:ea60` (2–3 retries
  typical).
- **`/voltage` shows 0 V or TF flicker** → multiple bringup procs running on
  the Pi. `pkill -f Mcnamu; pkill -f base_node; pkill -f ekf` then relaunch.
- **`/cmd_vel` silent** → check `frequency_monitor` CSV under
  `decider_logs/` for which link in the pipeline died.

## Build artifacts

`build/`, `install/`, `log/`, `decider_logs/`, `ros2_frequency_log_*/` are all
gitignored — they're regenerated by `colcon` and runtime logging.
