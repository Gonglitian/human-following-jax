#!/usr/bin/env bash
# Generic one-shot startup for any following_sim scenario.
#   usage: run_scenario.sh <scenario>            # gui + rviz + perception
#          run_scenario.sh <scenario> headless   # gzserver only, no rviz
#
# Handles all the landmines we hit:
#   - conda base in PATH hijacks spawn_entity.py (rclpy is py3.10)
#   - stale generatedWorld.world from previous run loads the wrong scenario
#   - `ros2 topic pub -1` drops first 1-2 messages during DDS discovery
#   - decider rejects auto:* unless mode:automatic was sent first
#   - leaked monitor processes (frequency_monitor, tf_health, watchdog, bag)
# `set -u` clashes with /opt/ros/humble/setup.bash (uses unbound vars).
set -eo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $(basename "$0") <scenario> [headless] [policy]" >&2
  echo "       policy = meta (default) | orca | mpc" >&2
  echo "       env: ADAPTIVE_MAPPING=true|false (only honored when policy=meta)" >&2
  echo >&2
  echo "available scenarios:" >&2
  ls "$(dirname "${BASH_SOURCE[0]}")/../launch"/*.launch.py 2>/dev/null \
    | xargs -n1 basename | sed 's/\.launch\.py$//' | grep -v '^bringup$' \
    | sed 's/^/  /' >&2
  exit 2
fi

SCENARIO="$1"
MODE="${2:-gui}"  # gui or headless
POLICY="${3:-meta}"  # meta, orca, mpc, rlpc
ADAPTIVE_MAPPING="${ADAPTIVE_MAPPING:-true}"
# Auto-pick a sensible default ckpt per policy. User can override via env.
case "$POLICY" in
  meta) DEFAULT_CKPT=meta_4.pt ;;
  rlpc) DEFAULT_CKPT=rl_pc.pt ;;
  crl)  DEFAULT_CKPT=baselines/crl_22_75_7e_5.pt ;;
  *)    DEFAULT_CKPT=meta_4.pt ;;
esac
MODEL_WEIGHT_FILE="${MODEL_WEIGHT_FILE:-$DEFAULT_CKPT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LAUNCH_FILE="$WS_ROOT/install/following_sim/share/following_sim/launch/${SCENARIO}.launch.py"
LOG_DIR=/tmp/following_sim_logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SCENARIO}_$(date +%Y%m%d_%H%M%S).log"

if [ ! -f "$LAUNCH_FILE" ]; then
  echo "ERROR: $LAUNCH_FILE not found." >&2
  echo "Did you colcon build after adding the launch file? Looking for available scenarios:" >&2
  ls "$WS_ROOT/install/following_sim/share/following_sim/launch"/*.launch.py 2>/dev/null \
    | xargs -n1 basename | sed 's/\.launch\.py$//' | grep -v '^bringup$' \
    | sed 's/^/  /' >&2
  exit 1
fi

# 1. Strip conda so system py3.8 resolves for ROS Foxy.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER \
      CONDA_PYTHON_EXE _CONDA_ROOT CONDA_EXE _CONDA_EXE _CE_CONDA \
      PYTHONPATH PYTHONHOME || true
export PATH=/opt/ros/foxy/bin:/opt/ros/foxy/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export ROS_DOMAIN_ID=99  # force isolation from real-robot domain (32)
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /opt/ros/foxy/setup.bash
source "$WS_ROOT/install/setup.bash"

# 2. Kill residuals from any previous run (incl. monitor layer leaks).
echo "[1/5] killing residual sim processes..."
ps -eo pid,cmd \
  | grep -E "(rviz|gzserver|gzclient|ros2 launch following_sim|ros2 bag record|hunav|dr_spaam|decider|sort_tracker|predictor|occupancy_grid|target_to_uwb|detections_merger|robot_state_publisher|spawn_entity|command_listener|metrics_recorder|human_states_viz|frequency_monitor|tf_health_monitor|cmd_vel_watchdog)" \
  | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
sleep 3

# 3. Clear cached generated world (prevents loading previous scenario).
rm -f "$WS_ROOT/install/following_sim/share/following_sim/worlds/generatedWorld.world" \
      "$WS_ROOT/following_sim/worlds/generatedWorld.world"

# 4. Launch in background.
echo "[2/5] launching $SCENARIO ($MODE; log: $LOG)"
GUI_FLAG=true
RVIZ_FLAG=true
if [ "$MODE" = "headless" ]; then GUI_FLAG=false; RVIZ_FLAG=false; fi
nohup ros2 launch following_sim "${SCENARIO}.launch.py" \
      gui:=$GUI_FLAG rviz:=$RVIZ_FLAG launch_perception:=true \
      policy:=$POLICY adaptive_mapping:=$ADAPTIVE_MAPPING \
      model_weight_file:=$MODEL_WEIGHT_FILE \
      > "$LOG" 2>&1 &
disown

# 5. Wait for decider init. Heavy scenarios need longer; loop up to ~3min.
case "$POLICY" in
  meta) READY_GREP="Meta RL model initialized" ;;
  orca) READY_GREP="SG-ORCA baseline initialized" ;;
  mpc)  READY_GREP="MPC-ADC.*initialized" ;;
  rlpc) READY_GREP="Meta RL model initialized" ;;  # subclass uses same logger
  crl)  READY_GREP="Meta RL model initialized" ;;  # subclass uses same logger
  *)    echo "ERROR: unknown policy: $POLICY" >&2; exit 1 ;;
esac
echo "[3/5] waiting for decider ($POLICY, up to 180s)..."
for _ in $(seq 1 90); do
  grep -Eq "$READY_GREP" "$LOG" 2>/dev/null && break
  sleep 2
done
if ! grep -Eq "$READY_GREP" "$LOG"; then
  echo "ERROR: decider did not come up. tail of log:" >&2
  tail -40 "$LOG" >&2
  exit 1
fi
sleep 6  # let command_listener finish subscribing

# 6. Arm. Persistent publisher (`-r 2` for ~4s) avoids DDS discovery drops.
#    Order: mode -> human_following -> distance. Other orders drop commands.
echo "[4/5] arming policy..."
timeout 4 ros2 topic pub -r 2 /command std_msgs/msg/String "data: mode:automatic"        >/dev/null 2>&1 || true
sleep 1
timeout 4 ros2 topic pub -r 2 /command std_msgs/msg/String "data: auto:human_following"  >/dev/null 2>&1 || true
sleep 1
timeout 4 ros2 topic pub -r 2 /command std_msgs/msg/String "data: auto:distance:2.0"     >/dev/null 2>&1 || true

echo "[5/5] armed. /cmd_vel should be live (target may need to drift into detect range)."
echo
echo "  watch status  : tail -f $LOG | grep -E 'DistanceStats|MATCH|Human following|tf_health|watchdog'"
echo "  watch cmd_vel : ros2 topic hz /cmd_vel"
echo "  bag dir       : ls -lt /tmp/following_sim_bags/ | head -3"
echo "  freq csv dir  : ls -td ros2_frequency_log_* | head -1"
echo "  stop all      : kill \$(pgrep -f 'ros2 launch following_sim'); pkill -f gzserver"
