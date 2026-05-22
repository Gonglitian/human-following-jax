from . import VecEnvWrapper
import numpy as np
from .running_mean_std import RunningMeanStd
import torch
import os
from collections import deque

import copy
import pickle

# from gst_updated.src.gumbel_social_transformer.temperature_scheduler \
#                                         import Temp_Scheduler
# from gst_updated.scripts.wrapper.crowd_nav_interface_parallel \
#                                         import CrowdNavPredInterfaceMultiEnv


# yjp mark: this is to utilize pretrained prediction models
class VecPretextNormalizeDummy(VecEnvWrapper):
    """
    A vectorized wrapper that processes the observations and rewards
    for GST predictors, and returns from an environment.
    config: a Config object
    test: whether we are training or testing
    """

    def __init__(self, venv, ob=False, ret=False, clipob=10., cliprew=10.,
                 gamma=0.99, epsilon=1e-8, config=None, test=False):
        VecEnvWrapper.__init__(self, venv)

        self.config = config
        self.device = torch.device(self.config.training.device)
        if test:
            self.num_envs = 1
        else:
            self.num_envs = self.config.env.num_processes

        self.max_human_num = config.sim.human_num + config.sim.human_num_range

        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = torch.zeros(self.num_envs).to(self.device)
        self.gamma = gamma
        self.epsilon = epsilon

        # For CV prediction, we don't need GST model, use simple dummy values
        self.pred_interval = 1
        self.buffer_len = 5  # simple buffer length for CV prediction

        # self.predictor = CrowdNavPredInterfaceMultiEnv(load_path=load_path, device=self.device, config = self.args, num_env = self.num_envs)

        # temperature_scheduler = Temp_Scheduler(self.args.num_epochs, self.args.init_temp, self.args.init_temp, temp_min=0.03)
        # self.tau = temperature_scheduler.decay_whole_process(epoch=100)

    def talk2Env_async(self, data): # yjp mark: pass in predictions to envs
        self.venv.talk2Env_async(data)

    def talk2Env_wait(self):
        outs=self.venv.talk2Env_wait()
        return outs
    
    def update_monitor_async(self, data):
        self.venv.update_monitor_async(data)
        
    def update_monitor_wait(self):
        obs, reward, done, info = self.venv.update_monitor_wait()
        if isinstance(obs, dict):
            for key in obs:
                obs[key] = torch.from_numpy(obs[key]).to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float() # yjp mark: reward don't need to be allocated on GPU?
        return obs, reward, done, info

    def step_wait(self):
        obs, rews, done, infos = self.venv.step_wait()

        # process the observations and reward # yjp mark: add-on style to incorporate prediction models
        obs, rews, infos = self.process_obs_rew(obs, done, rews=rews, infos=infos) # the effect is on observation alone

        return obs, rews, done, infos

    def _obfilt(self, obs):
        if self.ob_rms and self.config.RLTrain:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        # For CV prediction, we don't need complex trajectory buffers
        self.step_counter = 0

        obs = self.venv.reset()
        obs, _, _ = self.process_obs_rew(obs, np.zeros(self.num_envs), ())

        return obs


    '''
    1. Process observations: 
    Run inference on pred model with past obs as inputs, fill in the predicted trajectory in O['spatial_edges']
    
    2. Process rewards (rews):
    Calculate reward for colliding with predicted future traj and add to the original reward, 
    same as calc_reward() function in crowd_sim_pred.py except the data are torch tensors
    '''
    def process_obs_rew(self, O, done, rews=0., infos=()): # yjp mark: O for observation
        # for CV prediction (const_vel), we don't need to process observations
        # just pass through the observation and info
        self.step_counter = self.step_counter + 1

        return O, rews, infos