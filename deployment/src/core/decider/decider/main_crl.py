#!/usr/bin/env python3
"""
CRL guiding-policy / fixed-distance pure-RL baseline (paper §V.C, Fig. 3).

Loads ckpts that share the same architecture:
  - Stage-1 CRL guiding policies (PPOLag, single cost_limit per ckpt):
    `baselines/crl_{16_5,17,22_75,23,24}*.pt`
  - Fixed-distance pure-RL (paper "RL-DRP"; one ckpt per d*):
    `baselines/pure_rl_d{1_4,1_8,3_2}.pt`

All of these share `Policy + interaction_transformer` (9-dim robot_node, NO
`following_preference` channel — preference / d* is *baked into the trained
weights*, not an observation input).

This entry point is a thin subclass of `decider.main.Decider` that:
  - Swaps `PolicyMeta` for vanilla `Policy`
  - Drops the `following_preference` obs key entirely
  - Bypasses the P-controller (it would compute a preference no one reads)

Everything else (UWB / closest_lidar matching, OGM history, /command API,
mecanum smoothing, RL-step timer) is inherited unchanged.
"""

import rclpy

from rl.networks.model import Policy

from decider.main import Decider


class CrlDecider(Decider):
    """CRL / fixed-d pure-RL: Policy + 9-dim InteractionTransformer, no preference obs."""

    def _create_actor_critic(self):
        # Vanilla `Policy` (not `PolicyMeta`) + `interaction_transformer`
        # (9-dim, no preference channel). Both Stage-1 CRL guiding policies
        # and fixed-d pure-RL ckpts load with this combination.
        return Policy(
            self.observation_space,
            self.action_space,
            base_kwargs=self.algo_args,
            base='interaction_transformer',
            config=self.config,
        )

    def _add_preference_obs_space(self, d):
        # CRL/pure_rl_d networks were trained without a preference observation;
        # the desired distance is implicit in the trained weights. Adding any
        # preference key here would break state_dict matching.
        return

    def _build_preference_obs(self, current_distance):
        # No preference obs key for CRL. Still bookkeep `current_following_preference`
        # for log compatibility (decider's DistanceStats line uses it).
        if current_distance is not None and self.adaptive_mapping:
            self.current_following_preference = self.p_controller.compute(current_distance)
        else:
            self.current_following_preference = 0
        return {}


def main(args=None):
    rclpy.init(args=args)
    node = CrlDecider()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[CRL] KeyboardInterrupt -> shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
