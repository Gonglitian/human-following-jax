# DEPLOY — robot bringup recipe

Two-host setup: a **laptop** (runs perception/decider/RViz) connects over Wi-Fi
to the **Rosmaster X3** Pi (`pi@192.168.1.11`, runs only bringup + sensors).

> The training side lives in [`training/`](training/) and runs on GPU servers.
> See [`deployment/README.md`](deployment/README.md) for the package list and
> third-party install flow.

## Hardware

- Yahboom Rosmaster X3 (mecanum, Raspberry Pi 4)
- RPLidar A1 on `/dev/ttyUSB0` (USB ID `10c4:ea60`)
- LinkTrack UWB pair on `/dev/ttyUSB1`
- Laptop (Ubuntu 20.04, ROS 2 Foxy, optional CUDA for any GPU-bound nodes)

## Network

- Robot Wi-Fi AP `Mecnamu` (or robot joined to lab AP) → `192.168.1.11`
- Laptop on same subnet
- `ROS_DOMAIN_ID=42` on both sides
- (Optional) Tailscale for remote access

## One-shot launch from laptop

```bash
cd deployment
source install/setup.bash

# method ∈ {meta, baseline_rl, orca, mpc}
# speed  ∈ {0.3, 0.5, 0.8, 1.0} m/s
# pref   ∈ {-2, -1, 0, 1, 2}  (following distance preference)
ros2 launch deployment start_real_robot.launch.py method:=meta speed:=0.5 pref:=0
```

The launch file:

1. SSHes to `pi@192.168.1.11` and runs `bringup.launch.py` in the robot's docker
2. On the laptop, starts: `dr_spaam_ros2 → sort_tracker → predictor → decider →
   occupancy_generation → frequency_monitor`
3. Opens RViz with the lab's preset (target ID overlay, OGM, predicted trajectories)

## JAX policy bridge (planned, not wired yet)

Today's `decider/main.py` loads a PyTorch `.pt` from
`decider/model_weight/`. To swap in a JAX policy trained by `training/`:

1. **Export** params on the GPU host:
   ```bash
   cd training
   python scripts/eval.py --params runs/<TS>/params.pkl --export-onnx /tmp/policy.onnx
   ```
   *(ONNX export script not yet written — see task list.)*

2. **Bridge node**: write `deployment/src/core/decider/decider/main_jax.py`
   that re-implements the obs-construction + forward-call against either:
   - `flax + jax` directly (requires JAX on the robot — heavy), OR
   - `onnxruntime` (light, CPU-only inference works for 50 Hz control)

3. **Launch with**:
   ```bash
   ros2 launch deployment start_real_robot.launch.py method:=jax_meta \
        policy_format:=onnx policy_path:=/tmp/policy.onnx
   ```

The obs schema (`robot_node` / `temporal_edges` / `spatial_edges` /
`target_human_traj` / `local_ogm` / `detected_human_num` /
`following_preference`) is identical between the two repos by design, so the
bridge is mechanical — just rewire the loader.

## Common issues

| Symptom | First check |
|---|---|
| `/scan` silent | LiDAR motor — `usbreset 10c4:ea60`, 2–3 retries |
| `/voltage` = 0 V or TF flickers | Stale bringup procs on Pi — `pkill Mcnamu base_node ekf robot_state` |
| `/cmd_vel` silent but tracking works | `frequency_monitor` CSV reveals which link dropped |
| Policy outputs zeros | Wrong `model_weight_file` ROS param — check `ros2 param get /decider model_weight_file` |
| Robot reboot needed | `sudo shutdown -h now` before flipping main switch (hard power-off corrupted libstdc++ once) |

## Shutdown

```bash
ros2 service call /command std_msgs/msg/String "data: 'stop'"   # stop policy
ssh pi@192.168.1.11 'sudo shutdown -h now'                       # graceful robot off
```
