#!/usr/bin/env python3
"""
RL-PC baseline (paper §V.B): RL with Preference Conditioning.

Per paper §V.B:
  "A single RL policy that takes the desired following distance as an
   additional input and is trained with a shaped reward, without guiding
   policies or KL regularization."

ckpt = `rl_pc.pt` (lifted from
  github.com/tasl-lab/human-following-robot @ pure_rl_baseline branch
  /trained_models/rl_ood/checkpoints/07811.pt
verified by reading branch's train.py (uses `ppo.PPO` + plain `Policy`,
no `PPO_MetaGuided` / no KL distillation; env shapes reward as
`-abs(target_human_dist - preference_distance)`).

Network: `interaction_transformer_rlpc.InteractionTransformer` — identical
to the base IT except `robot_embedding` / `robot_linear` accept 10-dim
input and `forward()` concatenates `inputs['preference_distance']` into
`robot_states`.

This entry point is a thin subclass of `decider.main.Decider` overriding
only the network-construction and preference-encoding hooks. Everything
else (UWB matching, OGM history, /command API, mecanum smoothing) is
inherited unchanged.
"""

import numpy as np
import gym
import rclpy

from rl.networks.model import Policy

from decider.main import Decider


class RlpcDecider(Decider):
    """RL-PC: Policy + InteractionTransformerRLPC, raw preference_distance obs."""

    def _create_actor_critic(self):
        # `Policy` is the base wrapper (not `PolicyMeta`) — RL-PC was trained
        # without the meta-guided KL machinery, so its ckpt expects the
        # vanilla DiagGaussian head + IT base.
        return Policy(
            self.observation_space,
            self.action_space,
            base_kwargs=self.algo_args,
            base='interaction_transformer_rlpc',
            config=self.config,
        )

    def _add_preference_obs_space(self, d):
        # RL-PC trained on `preference_distance` ∈ [1.5, 3.0] m (config.py:
        # "Randomly sample preference distance for this episode (1.5m to 3.0m)").
        # Use shape (1,) — IT-RLPC.forward calls .unsqueeze(-1) inside.
        d['preference_distance'] = gym.spaces.Box(
            low=0.0, high=10.0, shape=(1,), dtype=np.float32,
        )

    def _build_preference_obs(self, current_distance):
        # No P-controller, no closed-loop adaptive mapping. RL-PC takes the
        # user's raw `auto:distance:N` setpoint straight to the network.
        # `current_distance` is unused for RL-PC (kept for sig parity).
        target = float(self.target_following_distance)
        # Mirror the Meta-side bookkeeping field for log compatibility.
        self.current_following_preference = target
        return {
            "preference_distance": np.array([target], dtype=np.float32),
        }


def main(args=None):
    rclpy.init(args=args)
    node = RlpcDecider()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[RLPC] KeyboardInterrupt -> shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
