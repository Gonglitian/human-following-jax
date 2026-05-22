import torch.nn.functional as F

from .selfAttn_srnn_merge import *
from .srnn_model import *

import torch.nn as nn

class OGM_CNN(nn.Module):
    def __init__(self, output_dim=256, temporal_frames=3):
        """
        Args:
            output_dim: Dimension of output features
            temporal_frames: Number of stacked static OGMs (early fusion)
        """
        super(OGM_CNN, self).__init__()

        self.temporal_frames = temporal_frames

        # 2D CNN with early fusion
        self.cnn_2d = nn.Sequential(
            nn.Conv2d(temporal_frames, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # Preserve local geometry (important for sparse OGM)
            nn.AdaptiveMaxPool2d((2, 2))   # -> [B, 256, 2, 2]
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.output_dim = output_dim

    def forward(self, x):
        """
        x: [seq_len, nenv, temporal_frames, H, W]
        """
        seq_len, nenv, T, H, W = x.shape
        assert T == self.temporal_frames, \
            f"Expected {self.temporal_frames} OGM frames, got {T}"

        # early fusion: temporal dimension -> channel dimension
        x = x.view(seq_len * nenv, T, H, W).float()  # [B, C, H, W]

        x = self.cnn_2d(x)                          # [B, 256, 2, 2]
        x = x.view(seq_len * nenv, -1)              # [B, 1024]
        x = self.fc(x)                              # [B, output_dim]

        return x.view(seq_len, nenv, -1)             # [seq_len, nenv, output_dim]


class EndRNNOGM(RNNBase):
    '''
    Class for the GRU
    '''
    def __init__(self, args):
        super(EndRNNOGM, self).__init__(args, edge=False)

        self.args = args

        self.args = args

        # Store required sizes
        self.rnn_size = args.human_node_rnn_size
        self.output_size = args.human_node_output_size
        self.embedding_size = args.human_node_embedding_size
        self.input_size = args.human_node_input_size
        self.edge_rnn_size = args.human_human_edge_rnn_size

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(256, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()

        # Linear layer to embed attention module output
        self.edge_attention_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)

        # Output linear layer
        self.output_linear = nn.Linear(self.rnn_size, self.output_size)

        self.obstacle_linear = nn.Linear(256, self.embedding_size)

        self.target_linear = nn.Linear(256, self.embedding_size)


    def forward(self, robot_s, target_traj, h_spatial_other, obstacle_features, h, masks):
        '''
        Forward pass for the model
        params:
        robot_s : input position [seq_len, nenv, 1, 256]
        target_traj : target human trajectory [seq_len, nenv, 2*(predict_steps+1)]
        h_spatial_other : output of the attention module [seq_len, nenv, 1, 256]
        obstacle_features : CNN processed obstacle features [seq_len, nenv, 256]
        h : hidden state of the current nodeRNN [1, nenv, 1, rnn_size]
        masks : masks for done states
        '''
        
        # Encode the input position
        encoded_input = self.encoder_linear(robot_s)
        encoded_input = self.relu(encoded_input)

        h_edges_embedded = self.relu(self.edge_attention_embed(h_spatial_other))

        # Process obstacle features
        obstacle_encoded = self.relu(self.obstacle_linear(obstacle_features))  # [seq_len, nenv, embedding_size]
        obstacle_encoded = obstacle_encoded.unsqueeze(2)  # [seq_len, nenv, 1, embedding_size]
        
        # Process target trajectory
        target_encoded = self.relu(self.target_linear(target_traj))  # [seq_len, nenv, embedding_size]
        target_encoded = target_encoded.unsqueeze(2)  # [seq_len, nenv, 1, embedding_size]
        
        # [seq_len, nenv, 1, 4*embedding_size]
        concat_encoded = torch.cat((encoded_input, target_encoded, obstacle_encoded, h_edges_embedded), dim=-1)

        # Forward through GRU
        x, h_new = self._forward_gru(concat_encoded, h, masks)
        outputs = self.output_linear(x)

        return outputs, h_new


class selfAttn_merge_SRNN_ogm(selfAttn_merge_SRNN):
    """
    Class for the proposed network
    """
    def __init__(self, obs_space_dict, args, infer=False):
        """
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        """
        # Call parent class constructor with required parameters
        super().__init__(obs_space_dict, args, infer)
        
        # Replace the original humanNodeRNN with our OGM version
        self.humanNodeRNN = EndRNNOGM(args)
        
        # Add OGM CNN for obstacle features
        self.obstacle_cnn = OGM_CNN(output_dim=256)
        
        # Add target trajectory processing
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.target_linear = nn.Sequential(init_(nn.Linear(12, 256)), nn.ReLU())

    def forward(self, inputs, rnn_hxs, masks, infer=False):
        if infer:
            # Test time
            seq_length = 1
            nenv = self.nenv

        else:
            # Training time
            seq_length = self.seq_length
            nenv = self.nenv // self.nminibatch

        robot_node = reshapeT(inputs['robot_node'], seq_length, nenv)
        temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv)
        spatial_edges = reshapeT(inputs['spatial_edges'], seq_length, nenv)
        obstacle_features = reshapeT(inputs['local_ogm'], seq_length, nenv)
        target_traj = reshapeT(inputs['target_human_traj'], seq_length, nenv)

        # to prevent errors in old models that does not have sort_humans argument
        if not hasattr(self.args, 'sort_humans'):
            self.args.sort_humans = True
        if self.args.sort_humans:
            detected_human_num = inputs['detected_human_num'].squeeze(-1).cpu().int()
        else:
            human_masks = reshapeT(inputs['visible_masks'], seq_length, nenv).float() # [seq_len, nenv, max_human_num]
            # if no human is detected (human_masks are all False, set the first human to True)
            human_masks[human_masks.sum(dim=-1)==0] = self.dummy_human_mask

        hidden_states_node_RNNs = reshapeT(rnn_hxs['human_node_rnn'], 1, nenv)

        masks = reshapeT(masks, seq_length, nenv)

        if self.args.no_cuda:
            all_hidden_states_edge_RNNs = Variable(
                torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cpu())
        else:
            all_hidden_states_edge_RNNs = Variable(
                torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cuda())

        robot_states = torch.cat((temporal_edges, robot_node), dim=-1)
        robot_states = self.robot_linear(robot_states)

        obstacle_features = self.obstacle_cnn(obstacle_features)

        target_traj = self.target_linear(target_traj)

        # attention modules
        if self.args.sort_humans:
            # human-human attention
            if self.args.use_self_attn:
                spatial_attn_out=self.spatial_attn(spatial_edges, detected_human_num).view(seq_length, nenv, self.human_num, -1)
            else:
                spatial_attn_out = spatial_edges
            output_spatial = self.spatial_linear(spatial_attn_out)

            # robot-human attention
            hidden_attn_weighted, _ = self.attn(robot_states, output_spatial, detected_human_num)
        else:
            # human-human attention
            if self.args.use_self_attn:
                spatial_attn_out = self.spatial_attn(spatial_edges, human_masks).view(seq_length, nenv, self.human_num, -1)
            else:
                spatial_attn_out = spatial_edges
            output_spatial = self.spatial_linear(spatial_attn_out)

            # robot-human attention
            hidden_attn_weighted, _ = self.attn(robot_states, output_spatial, human_masks)

        # Do a forward pass through GRU
        outputs, h_nodes = self.humanNodeRNN(robot_states, target_traj, hidden_attn_weighted, obstacle_features, hidden_states_node_RNNs, masks)

        # Update the hidden and cell states
        all_hidden_states_node_RNNs = h_nodes
        outputs_return = outputs

        rnn_hxs['human_node_rnn'] = all_hidden_states_node_RNNs
        rnn_hxs['human_human_edge_rnn'] = all_hidden_states_edge_RNNs

        # x is the output and will be sent to actor and critic
        x = outputs_return[:, :, 0, :]

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        for key in rnn_hxs:
            rnn_hxs[key] = rnn_hxs[key].squeeze(0)

        if infer:
            return self.critic_linear(hidden_critic).squeeze(0), hidden_actor.squeeze(0), rnn_hxs
        else:
            return self.critic_linear(hidden_critic).view(-1, 1), hidden_actor.view(-1, self.output_size), rnn_hxs

def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape))