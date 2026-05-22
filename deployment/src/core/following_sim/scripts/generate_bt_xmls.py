#!/usr/bin/env python3
"""
Generate hunav_agent_manager behavior-tree XML files from our scenario YAMLs.

hunav_agent_manager looks up BT files at
    <hunav_gazebo_wrapper_src>/behavior_trees/<yaml_base>__agent_<id>_bt.xml
and the upstream only ships pre-baked files for cafe/house/warehouse. Our
scenarios (corridor, junction, crowd, occlusion, sharp_turn) all use
behavior type Regular, so the BT body is the same template with only
agent id + goal list changing. This script stamps out the XMLs using the
upstream generator's header template + a small inline footer per agent.
"""
import argparse
import os
import sys
from pathlib import Path
import yaml


HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
WS_ROOT = PKG_ROOT.parent
TEMPLATE_HEADER = (
    WS_ROOT / "hunav_sim" / "hunav_behavior_tree_generator"
    / "hunav_behavior_tree_generator" / "templates" / "bt_template_header.xml"
)
OUT_DIR = WS_ROOT / "hunav_gazebo_wrapper" / "behavior_trees"


def footer(goals):
    """Compose the <BehaviorTree> body for a Regular-behavior agent.

    Mirrors the pattern in agents_warehouse__agent_1_bt.xml:
    every goal except the last goes inside a <RunOnce>, the last one wraps
    in <Inverter><RunOnce>. The Fallback + UpdateGoal block then cycles.
    """
    set_goals = []
    for idx, g in enumerate(goals):
        if idx == len(goals) - 1 and len(goals) > 1:
            set_goals.append(
                f'        <Inverter>\n'
                f'          <RunOnce>\n'
                f'            <SetGoal agent_id="{{id}}" goal_id="{g}"/>\n'
                f'          </RunOnce>\n'
                f'        </Inverter>\n'
            )
        else:
            set_goals.append(
                f'        <RunOnce>\n'
                f'          <SetGoal agent_id="{{id}}" goal_id="{g}"/>\n'
                f'        </RunOnce>\n'
            )

    set_goals_block = ''.join(set_goals)

    return f'''
<BehaviorTree ID="DefaultTree">
  <Fallback name="MainFallback">
    <!-- Goal setting Sequence -->
    <Sequence name="SetGoals">
{set_goals_block}
    </Sequence>
    <!-- Navigation loop -->
    <Sequence name="RegularNavigation">
      <Inverter>
        <IsGoalReached agent_id="{{id}}"/>
      </Inverter>
      <RegularNav agent_id="{{id}}" time_step="{{dt}}"/>
    </Sequence>
    <!-- Update Goal -->
    <UpdateGoal agent_id="{{id}}"/>
  </Fallback>
</BehaviorTree>



</root>
'''


def generate(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    params = data["hunav_loader"]["ros__parameters"]
    agent_names = params["agents"]

    yaml_base = yaml_path.stem  # e.g. "agents_corridor"
    header = TEMPLATE_HEADER.read_text()

    generated = []
    for name in agent_names:
        a = params[name]
        agent_id = a["id"]
        goals = a.get("goals", [])
        if not goals:
            print(f"  skipping {name}: no goals", file=sys.stderr)
            continue
        xml = header + footer(goals)
        out_path = OUT_DIR / f"{yaml_base}__agent_{agent_id}_bt.xml"
        out_path.write_text(xml)
        generated.append(out_path)
        print(f"  wrote {out_path}")

    return generated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("yamls", nargs="*",
                   help="agents_*.yaml files (defaults to all under config/)")
    args = p.parse_args()

    if not TEMPLATE_HEADER.exists():
        print(f"template header missing: {TEMPLATE_HEADER}", file=sys.stderr)
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    yamls = [Path(y) for y in args.yamls] or sorted(
        (PKG_ROOT / "config").glob("agents_*.yaml"))

    total = 0
    for y in yamls:
        print(f"== {y.name} ==")
        total += len(generate(y))

    print(f"\nDone. Generated {total} BT XML file(s) in {OUT_DIR}")


if __name__ == "__main__":
    main()
