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

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils.helpers import get_load_path
import torch

from collections import deque
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, custom_task_registry, Logger
import matplotlib.pyplot as plt


def deque_to_tensor(buffer : deque) -> torch.Tensor:
    if not buffer:
        raise ValueError("Deque is empty. No data to change into tensor.")
    if torch.is_tensor(buffer[0]) is not True:
        raise TypeError("Given deque does not contain torch tensors.")
    for i, obs in enumerate(buffer):
        if i == 0:
            ret = obs
            ret.to(obs.device)
        else:
            ret = torch.cat((ret, obs), dim=1)
    if ret.shape[0] != buffer[0].shape[0] or ret.shape[1] != buffer[0].shape[1]*len(buffer):
        raise ValueError("Conversion from deque to tensor is wrong.")
    return ret

def train(args):
    if args.task in task_registry.task_classes:
        env, env_cfg = task_registry.make_env(name=args.task, args=args) # env is registed + args
        registry = task_registry
    elif args.task in custom_task_registry.task_classes:
        env, env_cfg = custom_task_registry.make_env(name=args.task, args=args) # env is registed + args
        registry = custom_task_registry
    
    env_cfg_play, train_cfg_play = registry.get_cfgs(name=args.task)
    history_len = train_cfg_play.runner.history_len
    ppo_runner, train_cfg = registry.make_alg_runner(env=env, name=args.task, args=args) # if task_cfg = [some dict file], alg_runner ignores nameCfgPPO and uses task_cfg file    
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    # env.destroy_sim()
    # torch.cuda.empty_cache()

    # # override some parameters for testing
    # env_cfg_play.env.num_envs = 1
    # env_cfg_play.terrain.num_rows = 5
    # env_cfg_play.terrain.num_cols = 5
    # env_cfg_play.terrain.curriculum = False
    # env_cfg_play.noise.add_noise = True
    # env_cfg_play.domain_rand.randomize_friction = False
    # env_cfg_play.domain_rand.push_robots = False
    # env_cfg_play.domain_rand.ext_force_robots = False
    # env_cfg_play.env.episode_length_s = 10
    # env_cfg_play.commands.num_commands = 4
    # env_cfg_play.commands.heading_command = True        
    # env_cfg_play.commands.ranges.lin_vel_x = [.6, .6]
    # env_cfg_play.commands.ranges.lin_vel_y = [0., 0.]
    # env_cfg_play.commands.ranges.heading = [0.,0.]
    # env_cfg_play.domain_rand.randomize_dof_friction = False
    # env_cfg_play.domain_rand.randomize_base_mass = False

    # obs_history = deque(maxlen = history_len)

    # # prepare environment
    # args.headless = False
    # env, _ = registry.make_env(name=args.task, args=args, env_cfg=env_cfg_play)
    # obs = env.get_observations()
    # for i in range(history_len):
    #     obs_history.append(torch.zeros(size=(env_cfg_play.env.num_envs, env_cfg_play.env.num_observations), device=env.device))
    # # load policy
    # train_cfg_play.runner.resume = True

    # ppo_runner, train_cfg_play = registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg_play) # env is reset here
    # policy = ppo_runner.get_inference_policy(device=env.device)
    
    # if hasattr(train_cfg_play.runner, "history_len"):
    #     exp_name = history_len
    #     if history_len == 0:
    #         raise ValueError("History length must be at least 1")
    #     exp_name = "_history_length_"+str(history_len)
    # else:
    #     exp_name = ""    
    # log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg_play.runner.experiment_name+exp_name) 
    # resume_dir = os.listdir(log_root)
    # resume_dir.sort()
    # if 'exported' in resume_dir: resume_dir.remove('exported')
    # SAVE_DIR = os.path.join(log_root, resume_dir[-1], 'data')
    
    # # export policy as a jit module (used to run it from C++)
    # if EXPORT_POLICY:
    #     path = os.path.join(log_root, 'exported', 'policies')
    #     export_policy_as_jit(ppo_runner.alg.actor_critic, path)
    #     export_policy_as_jit(ppo_runner.alg.actor_critic, SAVE_DIR)
    #     print('Exported policy as jit script to: ', path)

        

    # logger = Logger(env.dt)
    # robot_index = 0 # which robot is used for logging
    # joint_index = 3 # which joint is used for logging - knee pitch
    # start_state_log = np.ceil(2. / env.dt)
    # stop_state_log =np.ceil(5. / env.dt) # number of steps before plotting states
    # stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    # camera_direction = np.array([1., 1., 1.])
    # img_idx = 0

    # for i in range(1*int(env.max_episode_length)):
    #     obs_history.append(obs)
    #     obs_history_tensor = deque_to_tensor(obs_history)
    #     actions = policy(obs_history_tensor.detach())
    #     obs, _, rews, dones, infos = env.step(actions.detach())
        
    #     if RECORD_FRAMES:
    #         if i % 2:
    #             filename = os.path.join(log_root, 'exported', 'frames', f"{img_idx}.png")
    #             env.gym.write_viewer_image_to_file(env.viewer, filename)
    #             img_idx += 1 
    #     if MOVE_CAMERA:
    #         camera_position = env.rb_states[0, 0:3].to('cpu') + torch.from_numpy(camera_direction)   
    #         env.set_camera(camera_position, camera_position - torch.from_numpy(camera_direction))

    #     if i < stop_state_log and i > start_state_log:
    #         logger.log_states(
    #             {
    #                 'hip_pitch_vel': env.dof_vel[robot_index, 2].item(),
    #                 'knee_pitch_vel': env.dof_vel[robot_index, 3].item(),
    #                 'ankle_pitch_vel': env.dof_vel[robot_index, 4].item(),
    #                 'hip_pitch_pos': env.dof_pos[robot_index, 2].item(),
    #                 'knee_pitch_pos': env.dof_pos[robot_index, 3].item(),                    
    #                 'ankle_pitch_pos': env.dof_pos[robot_index, 4].item(),                    
    #                 'hip_pitch_torque': env.torques[robot_index, 2].item(),
    #                 'knee_pitch_torque': env.torques[robot_index, 3].item(),
    #                 'ankle_pitch_torque': env.torques[robot_index, 4].item(),
    #                 'command_x': env.commands[robot_index, 0].item(),
    #                 'command_y': env.commands[robot_index, 1].item(),
    #                 'command_yaw': env.commands[robot_index, 2].item(),
    #                 'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
    #                 'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
    #                 'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
    #                 'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
    #                 'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
    #             }
    #         )
    #     elif i==stop_state_log:
    #         logger.plot_states_mainthread(save_fig=True, save_dir= os.path.join(SAVE_DIR, "plot.png"))
    #         # logger.plot_states(save_fig=SAVE_FIG, save_dir= os.path.join(SAVE_DIR, "plot.png"))
    #     if  0 < i < stop_rew_log:
    #         if infos["episode"]:
    #             num_episodes = torch.sum(env.reset_buf).item()
    #             if num_episodes>0:
    #                 logger.log_rewards(infos["episode"], num_episodes)
    #     elif i==stop_rew_log:
    #         logger.print_rewards()
    env.destroy_sim()

if __name__ == '__main__':

    # Play once for logging
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    SAVE_FIG = True
    args = get_args()
    list_of_tasks=  ['bolt10fixed']
    for task in list_of_tasks:
        args.task = task
        print("TRAINING OF ", task, " has started")
        train(args)
