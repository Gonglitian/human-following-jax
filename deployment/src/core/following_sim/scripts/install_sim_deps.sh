#!/usr/bin/env bash
# Clone every third-party repo following_sim depends on into the parent
# workspace dir so `colcon build` picks them up alongside following_sim.
#
# Idempotent: re-running pulls latest if already present. Safe on the dev
# box; do NOT run this on the rosmaster (real robot doesn't need any of
# these — see docs/rosmaster/install_real_deps.md).
#
# Total disk: ~1.1 GB after clone (AWS hospital alone is 200+ MB).

set -eo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ws_dir="$(cd "$script_dir/../.." && pwd)"   # ros2_following/

clone_or_pull() {
    local url="$1"
    local dest="$2"
    local branch="${3:-}"
    if [[ -d "$dest/.git" ]]; then
        echo "==> pulling $(basename "$dest")"
        git -C "$dest" pull --ff-only || echo "    (couldn't ff; skip)"
    else
        echo "==> cloning $url -> $dest"
        if [[ -n "$branch" ]]; then
            git clone --depth 1 --branch "$branch" "$url" "$dest"
        else
            git clone --depth 1 "$url" "$dest"
        fi
    fi
}

cd "$ws_dir"

# --- HuNav social-navigation simulator ----------------------------------
clone_or_pull https://github.com/robotics-upo/hunav_sim.git \
              "$ws_dir/hunav_sim"
clone_or_pull https://github.com/robotics-upo/hunav_gazebo_wrapper.git \
              "$ws_dir/hunav_gazebo_wrapper"

# --- AWS RoboMaker Gazebo Classic worlds --------------------------------
# Each is a ROS 1 package shape, but works fine because we only consume
# their <world>/<model> assets via GAZEBO_*_PATH; nothing is colcon-built.
clone_or_pull https://github.com/aws-robotics/aws-robomaker-small-warehouse-world.git \
              "$ws_dir/aws-robomaker-small-warehouse-world"
clone_or_pull https://github.com/aws-robotics/aws-robomaker-bookstore-world.git \
              "$ws_dir/aws-robomaker-bookstore-world"
clone_or_pull https://github.com/aws-robotics/aws-robomaker-hospital-world.git \
              "$ws_dir/aws-robomaker-hospital-world"
clone_or_pull https://github.com/aws-robotics/aws-robomaker-small-house-world.git \
              "$ws_dir/aws-robomaker-small-house-world"

# --- Optional: ROS people perception stack ------------------------------
# Used by some legacy nodes; not required for the Meta RL decider.
clone_or_pull https://github.com/wg-perception/people.git \
              "$ws_dir/people" \
              ros2 || echo "==> people clone failed (non-fatal); skipping"

# --- Make hunav_agent_manager find the BT XMLs ---------------------------
# It reads from `<ws>/src/hunav_gazebo_wrapper/behavior_trees/`, so symlink.
mkdir -p "$ws_dir/src"
if [[ ! -e "$ws_dir/src/hunav_gazebo_wrapper" ]]; then
    ln -s "$ws_dir/hunav_gazebo_wrapper" "$ws_dir/src/hunav_gazebo_wrapper"
    echo "==> symlinked src/hunav_gazebo_wrapper -> hunav_gazebo_wrapper"
fi

# --- Hint: apt deps -----------------------------------------------------
echo
echo "If you haven't yet, install Gazebo Classic + ROS Humble apt deps:"
echo "    sudo apt install -y ros-humble-gazebo-ros-pkgs ros-humble-xacro \\"
echo "                        ros-humble-robot-state-publisher ros-humble-tf2-ros \\"
echo "                        gazebo libgazebo-dev"
echo
echo "Done. Next:"
echo "  1) Build:        cd $ws_dir && colcon build --symlink-install"
echo "  2) Generate BTs: python3 following_sim/scripts/generate_bt_xmls.py \\"
echo "                          following_sim/config/agents_<scenario>.yaml"
echo "  3) Run scenario: bash following_sim/scripts/run_scenario.sh <scenario>"
