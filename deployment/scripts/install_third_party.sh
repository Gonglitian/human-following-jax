#!/usr/bin/env bash
# Clone third-party ROS 2 packages required by the deployment stack but
# excluded from this repo (too large or upstream-maintained).
#
# Pinned to commits known-good with the lab's Foxy/Galactic setup.

set -euo pipefail

HERE="$(cd "$(dirname "$0")"/.. && pwd)"
TP="$HERE/src/third_party"
mkdir -p "$TP"
cd "$TP"

clone_or_skip () {
  local name="$1" url="$2" ref="$3"
  if [ -d "$name/.git" ]; then
    echo "[skip] $name already cloned"
  else
    git clone "$url" "$name"
    git -C "$name" checkout "$ref" || echo "[warn] could not checkout $ref for $name"
  fi
}

# Pedestrian detection
clone_or_skip dr_spaam_detector https://github.com/VisualComputingInstitute/2D_lidar_person_detection.git master
# Note: dr_spaam_ros2 wrapper is small — copy from upstream lab repo manually if needed.

# Crowd simulation
clone_or_skip hunav_sim            https://github.com/robotics-upo/hunav_sim.git master
clone_or_skip hunav_gazebo_wrapper https://github.com/robotics-upo/hunav_gazebo_wrapper.git master
clone_or_skip lightsfm             https://github.com/robotics-upo/lightsfm.git master
clone_or_skip people               https://github.com/wg-perception/people.git ros2

# Gazebo worlds (large — comment out if not needed)
clone_or_skip aws-robomaker-bookstore-world       https://github.com/aws-robotics/aws-robomaker-bookstore-world.git ros2
clone_or_skip aws-robomaker-hospital-world        https://github.com/aws-robotics/aws-robomaker-hospital-world.git ros2
clone_or_skip aws-robomaker-small-house-world     https://github.com/aws-robotics/aws-robomaker-small-house-world.git ros2
clone_or_skip aws-robomaker-small-warehouse-world https://github.com/aws-robotics/aws-robomaker-small-warehouse-world.git ros2

# Yahboom robot description + Nav2 config (vendor-supplied; lab has a local copy)
echo "[manual] yahboomcar_nav: copy from your existing ros2_following workspace or vendor source."

echo "Done. Run: colcon build --symlink-install"
