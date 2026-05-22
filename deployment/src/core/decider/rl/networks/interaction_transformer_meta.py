"""Guided meta policy."""

import torch
import torch.nn as nn
import numpy as np

from .interaction_transformer import InteractionTransformer, reshapeT
from .network_utils import init


class InteractionTransformerMeta(InteractionTransformer):
    """
    Meta version of InteractionTransformer.
    Adds following_preference input (1 dimension) to adapt behavior.
    
    Following preference values: -2=very close, -1=close, 0=medium, 1=far, 2=very far
    """
    
    def __init__(self, obs_space_dict, args, config=None):
        # Initialize parent class first
        super().__init__(obs_space_dict, args, config)
        
        # Override robot embedding to accept 10-dim input (2 temporal + 7 robot_node + 1 following_preference)
        self.robot_embedding = nn.Sequential(
            nn.Linear(10, 128),  # robot state dimension: vx(1), vy(1), px(1), py(1), r(1), gx(1), gy(1), v_pref(1), theta(1), following_preference(1)
            nn.ReLU(),
            nn.Linear(128, self.feature_dim)
        )
        
        # Override robot_linear to accept 10-dim input (for compatibility, even if not used in Transformer)
        init_ = lambda m: init(m, nn.init.orthogonal_, 
                              lambda x: nn.init.constant_(x, 0), 
                              np.sqrt(2))
        self.robot_linear = nn.Sequential(init_(nn.Linear(10, 256)), nn.ReLU())
        self.human_node_final_linear = init_(nn.Linear(self.output_size, 2))
    
    
    def forward(self, inputs, rnn_hxs, masks, infer=False):
        """
        Forward pass with following_preference input.
        Only difference from parent: includes following_preference in robot_states concatenation.
        """
        if infer:
            # test/unroll time
            seq_length = 1
            nenv = self.nenv
        else:
            # training time
            seq_length = self.seq_length
            nenv = self.nenv // self.nminibatch
        
        # Reshape inputs (same as parent)
        robot_node = reshapeT(inputs['robot_node'], seq_length, nenv) # [1, 128, 1, 7]
        temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv) # [1, 128, 1, 2]
        spatial_edges = reshapeT(inputs['spatial_edges'], seq_length, nenv) # [1, 128, 45, 12]
        obstacle_features = reshapeT(inputs['local_ogm'], seq_length, nenv) # [1, 128, 3, 50, 50]
        target_traj = reshapeT(inputs['target_human_traj'], seq_length, nenv) # [1, 128, 12]
        
        # Get human number
        if not hasattr(self.args, 'sort_humans'):
            self.args.sort_humans = True
        if self.args.sort_humans:
            detected_human_num = inputs['detected_human_num'].squeeze(-1).int() # [128]
        else:
            human_masks = reshapeT(inputs['visible_masks'], seq_length, nenv).float()
            human_masks[human_masks.sum(dim=-1)==0] = self.dummy_human_mask
            detected_human_num = human_masks.sum(dim=-1).int()
        
        # Process masks
        masks = reshapeT(masks, seq_length, nenv)
        
        # Add following_preference for guided meta policy
        following_preference = reshapeT(inputs['following_preference'], seq_length, nenv)  # [1, 128, 1, 1]
        robot_states = torch.cat((following_preference, temporal_edges, robot_node), dim=-1)  # [1, 128, 1, 10]
        
        # Process obstacle features through CNN
        obstacle_features = self.obstacle_cnn(obstacle_features)  # [1, 128, 256]
        
        max_human_num = spatial_edges.shape[2]

        # Embed all features
        robot_embed = self.robot_embedding(robot_states)  # [1, 128, 1, 256]
        human_embed = self.human_embedding(spatial_edges)  # [1, 128, 45, 256]
        obstacle_embed = self.obstacle_embedding(obstacle_features).unsqueeze(2)  # [1, 128, 1, 256]
        target_embed = self.target_embedding(target_traj).unsqueeze(2)  # [1, 128, 1, 256]
        
        # Create input sequence: robot, target, obstacle, humans
        sequence = torch.cat([
            robot_embed,
            target_embed,
            obstacle_embed,
            human_embed
        ], dim=2).view(seq_length * nenv, 3 + max_human_num, 256) # [128, 48, 256]
        
        # Create attention mask
        attention_mask = self.create_attention_mask(detected_human_num, max_human_num)  # [128, 3+max_human_num]
        
        # Pass through Transformer
        output_sequence = self.transformer(sequence, attention_mask)  # [128, 3+max_human_num, 256]
        
        # Extract robot representation (index 0 in sequence)
        robot_output = output_sequence[:, 0, :]  # [128, 256]
        
        # Generate transformer output
        outputs = self.transformer_output_layer(robot_output)  # [128, output_size]
        
        # Reshape back to original dimensions
        outputs = outputs.view(seq_length, nenv, -1)  # [1, 128, output_size]
        
        # Send outputs to Actor and Critic
        hidden_critic = self.critic(outputs)
        hidden_actor = self.actor(outputs)

        if infer:
            return self.critic_linear(hidden_critic).squeeze(0), hidden_actor.squeeze(0), rnn_hxs
        else:
            return self.critic_linear(hidden_critic).view(-1, 1), hidden_actor.view(-1, self.output_size), rnn_hxs
