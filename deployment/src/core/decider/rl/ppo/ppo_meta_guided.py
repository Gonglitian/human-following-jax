import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.kl import kl_divergence


class PPO_MetaGuided():
    """ Class for the PPO optimizer """
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 guiding_policies=None):

        self.actor_critic = actor_critic
        self.guiding_policies = guiding_policies # yjp mark: actor_critic_backup is for guiding the training

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.all_aggr_values = [-3, -2, -1, 0, 1, 2, 3]
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)



    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        
        offline_loss_epoch = 0 # yjp mark: offline loss is for regularization loss

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, action_dist = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)
                
                # tag
                if self.guiding_policies:
                    # select policies
                    unique_factors = obs_batch["aggressiveness_factor"].unique(sorted=True).squeeze()
                    
                    if unique_factors.ndim == 0:
                        unique_factors = unique_factors.view(1)
                    
                    policy_kl_grouped = {}
                    group_masks = {}
                    
                    for factor in unique_factors:
                        factor = int(factor.item())   
                        guiding_model_index = self.all_aggr_values.index(factor)
                        
                        guiding_model_index = 0 # for testing
                        factor = -3 # for testing
                        
                        guiding_model = self.guiding_policies[guiding_model_index]
                        
                        _, _, _, _, action_dist_guiding = guiding_model.evaluate_actions(
                            obs_batch, recurrent_hidden_states_batch, masks_batch,
                            actions_batch)                   
                        
                        policy_kl_grouped[factor] = kl_divergence(action_dist, action_dist_guiding).sum(-1)

                        
                        # Create a mask for the current factor. Ensure it's 1D for broadcasting.
                        mask = (obs_batch["aggressiveness_factor"].squeeze() == factor)
                        group_masks[factor] = mask.clone()
                    
                    regularization_loss = torch.tensor(0.0).to(values.device)
                    for key, value in policy_kl_grouped.items():
                        regularization_loss += value[group_masks[key]].sum()       
                else:
                    regularization_loss = torch.tensor(0.0).to(values.device)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss: # yjp mark: calculation of value losses
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                total_loss=value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef + 0.01 * regularization_loss
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                offline_loss_epoch += regularization_loss.item()


        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        
        offline_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, offline_loss_epoch
