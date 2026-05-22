import numpy as np
import torch
import torch.nn as nn

from .distributions import Bernoulli, Categorical, DiagGaussian
from .srnn_model import SRNN
from .interaction_transformer_meta import InteractionTransformerMeta


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class PolicyMeta(nn.Module):
    """Meta Policy that accepts following_preference input"""
    def __init__(self, obs_shape, action_space, base_kwargs=None, base='interaction_transformer_meta', config=None):
        super(PolicyMeta, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        self.config = config

        base = InteractionTransformerMeta

        self.base = base(obs_shape, base_kwargs, config)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs, self.config)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs, self.config)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs, self.config)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hxs."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks, infer=False):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        """
        Decide the action for the given inputs and hidden states.
        
        Args:
            inputs: dict of observation tensors
            rnn_hxs: dict of hidden states
            masks: whether to reset RNN hidden states
            deterministic: if True, select the most likely action
        
        Returns:
            value, action, action_log_prob, rnn_hxs
        """
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, infer=True)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        """
        Get the value estimate for the given inputs and hidden states.
        """
        value, _, _ = self.base(inputs, rnn_hxs, masks, infer=True)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, infer=False):
        """
        Evaluate the value and log probability for the given inputs and action.
        Used during PPO training.
        
        Args:
            inputs: dict of observation tensors
            rnn_hxs: dict of hidden states
            masks: whether to reset RNN hidden states
            action: the action taken
            infer: if True, use inference mode
        
        Returns:
            value, action_log_probs, dist_entropy, rnn_hxs, dist
        """
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, infer=infer)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs, dist
