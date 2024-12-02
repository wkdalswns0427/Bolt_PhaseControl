# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO, PPO_sym
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv

from .on_policy_runner import OnPolicyRunner

import matplotlib.pyplot as plt

import shutil
import pandas as pd

class OnPolicyRunnerHistory(OnPolicyRunner):

    def __init__(self,
                 env: VecEnv,
                 train_cfg, # this is a dict
                 log_dir=None,
                 device='cpu'):
        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        # Initialize algorithm runner
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env        
        # Initialize actor history length and history queue
        if "history_len" in self.cfg:
            history_len = self.cfg["history_len"]
        else:
            history_len = 1
        self.obs_history = deque(maxlen=history_len)

        # Initialize critic history length and history queue         
        if self.env.num_privileged_obs is not None:   
            num_critic_obs = self.env.num_privileged_obs
            if "critic_history_len" in self.cfg:
                critic_history_len = self.cfg["critic_history_len"]
            else:
                critic_history_len = 1       
        else: 
            num_critic_obs = self.env.num_obs
            critic_history_len = history_len
        self.critic_obs_history = deque(maxlen=critic_history_len)   
            

        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.num_obs * history_len,
                                                        num_critic_obs * critic_history_len,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg) # self.alg_cfg is dict?
        self.alg.history_len = history_len # Alg must know history len for logging
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model    
        storage_num_obs = self.env.num_obs * history_len
        storage_privileged_obs = self.env.num_privileged_obs
        if storage_privileged_obs is not None:
            storage_privileged_obs *= critic_history_len
        print("Observation history dimension : ", storage_num_obs)
        print("Privileged observation history dimension : ", storage_privileged_obs)
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [storage_num_obs], [storage_privileged_obs], [self.env.num_actions])
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        _, _ = self.env.reset() # reset means a single step after zero initialization
                     
        for _ in range(self.obs_history.maxlen):
            self.obs_history.append(torch.zeros(size=(self.env.num_envs, self.env.num_obs), device=self.device))
        for _ in range(self.critic_obs_history.maxlen):
            self.critic_obs_history.append(torch.zeros(size=(self.env.num_envs, num_critic_obs), device=self.device))
        print("History version of runner loaded")
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
            
        SAVE_PATH = os.path.join(self.log_dir, 'data/')
        os.makedirs(SAVE_PATH, exist_ok=True)
        URDF_PATH = os.path.join(self.resource_root, 'robots', self.cfg['experiment_name'], 'urdf', self.cfg['experiment_name']+".urdf")
        CONFIG_PATH = os.path.join(self.envs_root, self.cfg['experiment_name'])
        CSV_RETURN_PATH = os.path.join(SAVE_PATH, 'return.csv')
        CSV_EPS_PATH = os.path.join(SAVE_PATH, 'episode_length.csv')
        shutil.copy(URDF_PATH, os.path.join(SAVE_PATH, "log_" + self.cfg['experiment_name']+".urdf") )
        shutil.copy(os.path.join(CONFIG_PATH, self.cfg['experiment_name']+".py"), os.path.join(SAVE_PATH, "log_"+self.cfg['experiment_name']+".py") )
        shutil.copy(os.path.join(CONFIG_PATH, self.cfg['experiment_name']+"_config.py"), os.path.join(SAVE_PATH, "log_"+self.cfg['experiment_name']+"_config.py"))
            
        obs = self.env.get_observations().to(self.device)
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history.append(obs)
        if privileged_obs is not None:
            self.critic_obs_history.append(privileged_obs)
        else:
            self.critic_obs_history.append(obs)
        # history queues into tensors in self.device
        obs_history, critic_obs_history = deque_to_tensor(self.obs_history), deque_to_tensor(self.critic_obs_history)
                
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        returns = deque(maxlen=num_learning_iterations)
        epochs = deque(maxlen=num_learning_iterations)
        episode_lengths = deque(maxlen=num_learning_iterations)
        epochs2 = deque(maxlen=num_learning_iterations)
        # returnbuffer = deque(maxlen=self.num_steps_per_env)
        # episode_lengthbuffer = deque(maxlen=self.num_steps_per_env)
        # Buffer stores the average values of terminated environments over the past 100 steps
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs_history, critic_obs_history)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.obs_history.append(obs)
                    self.critic_obs_history.append(critic_obs)
                    obs_history = deque_to_tensor(self.obs_history) 
                    critic_obs_history = deque_to_tensor(self.critic_obs_history) 
                    self.alg.process_env_step(rewards, dones, infos)
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        # To buffer, add the mean of terminated environments. 
                        # if len(new_ids):
                        #     returnbuffer.append(cur_reward_sum[new_ids][:, 0].cpu().numpy().mean())
                        #     episode_lengthbuffer.append(cur_episode_length[new_ids][:, 0].cpu().numpy().mean())
                        
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        
                if len(rewbuffer):
                    # add returns data to csv file                    
                    returns.append(round(sum(rewbuffer)/len(rewbuffer), 3))
                    epochs.append(it)
                    # returnbuffer.clear()
                if len(lenbuffer):
                    # add returns data to csv file
                    episode_lengths.append(round(sum(lenbuffer)/len(lenbuffer), 3))
                    epochs2.append(it)
                    # episode_lengthbuffer.clear()         
                               
                stop = time.time()
                collection_time = stop - start
                
                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs_history)
                
            mean_value_loss, mean_surrogate_loss, mean_mirror_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        append_to_csv(CSV_RETURN_PATH, epoch=list(epochs), value=list(returns), isfirstappend=True, name='returns')
        append_to_csv(CSV_EPS_PATH, epoch=list(epochs2), value=list(episode_lengths), isfirstappend=True, name='episode length')
        plot_pandas(CSV_RETURN_PATH, os.path.join(SAVE_PATH, 'return.png'))
        plot_pandas(CSV_EPS_PATH, os.path.join(SAVE_PATH, 'episode_length.png'))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/mirror', locs['mean_mirror_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mirror loss:':>{pad}} {locs['mean_mirror_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mirror loss:':>{pad}} {locs['mean_mirror_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def set_resource_root(self, dir):
        if os.path.exists:
            self.resource_root = dir
        else:
            raise ValueError("Resource root does not exist")    
    
    def set_envs_root(self, dir):
        if os.path.exists:
            self.envs_root = dir
        else:
            raise ValueError("Envs root does not exist")   
            
##################HELPER###################
def deque_to_tensor(buffer : deque) -> torch.Tensor:
    if not buffer:
        raise ValueError("Deque is empty. No data to change into tensor.")
    if torch.is_tensor(buffer[0]) is not True:
        raise TypeError("Given deque does not contain torch tensors.")
    
    ret = torch.cat(list(buffer), dim=1)
    if ret.shape[0] != buffer[0].shape[0] or ret.shape[1] != buffer[0].shape[1]*len(buffer):
        raise ValueError("Conversion from deque to tensor is wrong.")
    return ret

def plot_pandas(csv_file, save_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    # Extract 'episode' and 'return' columns

    # Extract 'epoch' and 'return' columns
    name = df.columns[1]
    episode = df['epoch']
    return_value = df[name]

    # Plot episode-to-return graph
    plt.figure()
    plt.plot(episode, return_value)
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.title('Epoch-to-' +name+ ' Graph')
    plt.grid(True)
    plt.show
    plt.savefig(save_path)
    
def append_to_csv(file_path, epoch, value, isfirstappend = False, name=''):
    # Create a DataFrame with the new data
    data = {'epoch': epoch, name : value}
    df = pd.DataFrame(data)
    
    # Append the DataFrame to the CSV file
    df.to_csv(file_path, mode='a', header=isfirstappend, index=False)

