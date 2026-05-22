import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .srnn_model import init

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

            # Preserve local geometry
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



class PositionalEncoding(nn.Module):
    """Positional encoding module, providing sequence position information for Transformer"""
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        # Create position encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Use sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as a buffer
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoder(nn.Module):
    """Transformer encoder for feature extraction"""
    def __init__(self, feature_dim=256, nhead=8, num_layers=3, dim_feedforward=1024):
        super(TransformerEncoder, self).__init__()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(feature_dim)
        
        # Standard Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0, 
            batch_first=True
        )
        
        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output processing
        self.output_layer = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, feature_dim]
        mask: attention mask [batch_size, seq_len]
        """
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # mask processing
        if mask is not None:
            src_key_padding_mask = ~mask.bool()
            
            # Use src_key_padding_mask format to pass to transformer_encoder
            output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        else:
            output = self.transformer_encoder(x)
        
        output = self.output_layer(output)
        
        return output


class InteractionTransformer(nn.Module):
    """Transformer for modeling interactions between entities (robot, humans, obstacles, target)"""
    def __init__(self, obs_space_dict, args, config=None, infer=False):
        super(InteractionTransformer, self).__init__()
        self.infer = infer
        self.is_recurrent = False
        self.args = args
        self.config = config
        
        self.human_num = obs_space_dict['spatial_edges'].shape[0]
        self.seq_length = args.seq_length
        self.nenv = args.num_processes
        self.nminibatch = args.num_mini_batch
        
        # store required size
        self.output_size = args.human_node_output_size
        self.feature_dim = 256
        
        # for compatibility interface, define these attributes
        self.human_node_rnn_size = 256
        self.human_human_edge_rnn_size = 256
        
        # initialize CNN to process obstacle features
        self.obstacle_cnn = OGM_CNN(output_dim=256)
        
        # feature embedding layers
        self.robot_embedding = nn.Sequential(
            nn.Linear(10, 128),  # robot state dimension (2 + 7 + 1 = 10, added preference_distance)
            nn.ReLU(),
            nn.Linear(128, self.feature_dim)
        )
        
        # Calculate human embedding input dimension based on config
        human_input_dim = 12  # base trajectory dimension
        
        self.human_embedding = nn.Sequential(
            nn.Linear(human_input_dim, 128),  # 12
            nn.ReLU(),
            nn.Linear(128, self.feature_dim)
        )
        
        self.obstacle_embedding = nn.Linear(self.feature_dim, self.feature_dim)
        
        # Calculate target embedding input dimension based on config
        target_input_dim = 12  # base trajectory dimension
        
        self.target_embedding = nn.Sequential(
            nn.Linear(target_input_dim, 128),  # 12
            nn.ReLU(),
            nn.Linear(128, self.feature_dim)
        )
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            feature_dim=self.feature_dim,
            nhead=8,
            num_layers=4,
            dim_feedforward=1024
        )
        
        # output processing layer
        self.transformer_output_layer = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, args.human_node_output_size)
        )
        
        # initialize Actor and Critic networks
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        hidden_size = self.output_size
        
        self.actor = nn.Sequential(
            init_(nn.Linear(self.output_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
            
        self.critic = nn.Sequential(
            init_(nn.Linear(self.output_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
            
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        
        self.robot_linear = nn.Sequential(init_(nn.Linear(10, 256)), nn.ReLU())  # 9 + 1 for preference_distance
        self.human_node_final_linear = init_(nn.Linear(self.output_size, 2))
        
        # initialize virtual mask
        dummy_human_mask = [0] * self.human_num
        dummy_human_mask[0] = 1
        if self.args.no_cuda:
            self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cpu())
        else:
            self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cuda())

    def create_attention_mask(self, detected_human_num, max_human_num):
        """Create attention mask for valid humans"""
        nenv = detected_human_num.shape[0]
        device = detected_human_num.device
        mask = torch.zeros(nenv, max_human_num + 3, device=device)  # +3 for robot, target and obstacle
        
        # robot, target and obstacle are always valid
        mask[:, 0:3] = 1
        
        # set mask for valid humans
        for i in range(nenv):
            n = min(int(detected_human_num[i].item()), max_human_num)
            mask[i, 3:3+n] = 1
            
        return mask

    def forward(self, inputs, rnn_hxs, masks, infer=False):
        if infer:
            # test/unroll time
            seq_length = 1
            nenv = self.nenv
        else:
            # training time
            seq_length = self.seq_length
            nenv = self.nenv // self.nminibatch
        
        # Reshape inputs
        robot_node = reshapeT(inputs['robot_node'], seq_length, nenv) # [1, 128, 1, 7]
        temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv) # [1, 128, 1, 2]
        spatial_edges = reshapeT(inputs['spatial_edges'], seq_length, nenv) # [1, 128, 45, 12]
        obstacle_features = reshapeT(inputs['local_ogm'], seq_length, nenv) # [1, 128, 3, 50, 50]
        target_traj = reshapeT(inputs['target_human_traj'], seq_length, nenv) # [1, 128, 12]
        preference_distance = reshapeT(inputs['preference_distance'], seq_length, nenv) # [1, 128, 1]
        
        # print(f"target_traj.shape: {target_traj.shape}")
        # print(f"spatial_edges.shape: {spatial_edges.shape}")
        
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
        
        # Combine robot state with preference_distance (put preference first as high-level intent)
        robot_states = torch.cat((preference_distance.unsqueeze(-1), temporal_edges, robot_node), dim=-1)  # [1, 128, 1, 10]
        
        # Process obstacle features through CNN
        obstacle_features = self.obstacle_cnn(obstacle_features)  # [1, 128, 256]
        
        max_human_num = spatial_edges.shape[2]
        # print(f"max_human_num: {max_human_num}")

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

    @property
    def recurrent_hidden_state_size(self):
        # For compatibility
        return {'human_node_rnn': 1, 'human_human_edge_rnn': 1}


def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape)) 