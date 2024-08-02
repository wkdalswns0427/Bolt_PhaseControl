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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from collections import deque

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, custom_task_registry, Logger
from legged_gym.utils.helpers import set_seed

import numpy as np
import torch

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

def write_tensor_to_txt(tensor, file_path, precision):
    """
    Write tensor data to a text file.
    
    Args:
    - tensor (torch.Tensor): Input tensor.
    - file_path (str): File path to save the tensor data.
    """
    # Convert tensor to numpy array
    tensor_data = tensor.detach().to('cpu').numpy()
    np.round(tensor_data)
    # Write tensor data to text file
    with open(file_path, 'a') as file:
        for row in tensor_data:
            # Convert each row of the tensor to a string and write to file
            row_str = ' '.join(map(lambda x: f"{x:.{precision}f}", row))
            file.write(row_str + '\n')


def custom_play_disturbance(args):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    set_seed(432432)
    list_of_robots = ['bolt6', 'bolt10inverted'] 
    list_of_angles = [0, 3.141592/2, 3.141592]
    list_of_comparisons = ['Forward', 'Y-Axis', 'Backward']
    list_of_forces = [(6, 12), (6, 12), (6, 12)]
    for compare_index, angles in enumerate(list_of_angles):
        for task in list_of_robots:
            if task in task_registry.task_classes:
                registry = task_registry
            elif task in custom_task_registry.task_classes:
                registry = custom_task_registry
            print("Experiment for ", task, " has started")
            env_cfg, train_cfg = registry.get_cfgs(name=task)
            # override some parameters for testing
            env_cfg.env.num_envs = 100
            env_cfg.env.episode_length_s = 5
            env_cfg.terrain.num_rows = 5
            env_cfg.terrain.num_cols = 5
            env_cfg.terrain.curriculum = False
            env_cfg.terrain.mesh_type = 'plane'
            env_cfg.noise.add_noise = True
            env_cfg.domain_rand.randomize_friction = False
            env_cfg.domain_rand.push_robots = False
            env_cfg.domain_rand.ext_force_robots = True
            env_cfg.domain_rand.ext_force_duration_s = [.5, .5]
            env_cfg.domain_rand.ext_force_interval_s = 3.
            env_cfg.commands.num_commands = 4
            env_cfg.commands.heading_command = True        
            env_cfg.commands.ranges.lin_vel_x = [0.5, 0.5]
            env_cfg.commands.ranges.lin_vel_y = [0., 0.]
            env_cfg.commands.ranges.heading = [0.,0.]

            if hasattr(train_cfg.runner, "history_len"):
                history_len = train_cfg.runner.history_len
            else:
                history_len = 1
            obs_history = deque(maxlen = history_len)

            # prepare environment
            env, _ = registry.make_env(name=task, args=args, env_cfg=env_cfg)
            obs = env.get_observations()
            for _ in range(history_len):
                obs_history.append(torch.zeros(size=(env_cfg.env.num_envs, env_cfg.env.num_observations), device=env.device))
            # obs_history[history_len-1][:, 2] = -1. #TODO change if observation changes
            # load policy
            train_cfg.runner.resume = True
            ppo_runner, train_cfg = registry.make_alg_runner(env=env, name=task, args=args, train_cfg=train_cfg) # env is reset here
            policy = ppo_runner.get_inference_policy(device=env.device)
            
            # export policy as a jit module (used to run it from C++)
            if EXPORT_POLICY:
                path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
                export_policy_as_jit(ppo_runner.alg.actor_critic, path)
                print('Exported policy as jit script to: ', path)

            logger = Logger(env.dt)
            robot_index = 0 # which robot is used for logging
            joint_index = 1 # which joint is used for logging
            start_state_log = np.ceil(4. / env.dt) 
            stop_state_log = np.ceil(6. / env.dt) # number of steps before plotting states
            stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
            # camera_position = np.array(env_cfg.viewer.pos, dtype=float64)
            # camera_vel = np.array([1., 1., 0.])
            # camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
            camera_direction = np.array([3,3, 3])
            img_idx = 0
            
            # SAVE_DIR = 'play_plot.png'
            # log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name+'_disturbance') 
            # resume_dir = os.listdir(log_root)
            # resume_dir.sort()
            # if 'exported' in resume_dir: resume_dir.remove('exported')
            # SAVE_DIR = os.path.join(log_root, resume_dir[-1], 'data', 'plot.png')
            # open('data.txt', 'w')
            env.curriculum_index = 1

            success_rate = []
            disturbances = np.linspace(*list_of_forces[compare_index], 50)
            for disturbance in disturbances:
                total_count = 0
                fail_count = 0
                env_cfg.domain_rand.ext_force_scale_range = (disturbance, disturbance)
                env_cfg.domain_rand.ext_force_direction_range = (angles, angles)
                for i in range(int(env.max_episode_length+1)):
                    # obs_history.append(obs)
                    # obs_history.append(torch.ones_like(obs) * (-1.))
                    obs_history_tensor = deque_to_tensor(obs_history)
                    actions = policy(obs_history_tensor.detach())
                    obs, _, rews, dones, fails, infos = env.step_count_failures(actions.detach())
                    fail_count += torch.sum(fails)
                    total_count += torch.sum(dones)
                    obs_history.append(obs)
                    
                    if RECORD_FRAMES:
                        if i % 2:
                            filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                            env.gym.write_viewer_image_to_file(env.viewer, filename)
                            img_idx += 1 
                    if MOVE_CAMERA:
                        camera_position = env.rb_states[0, 0:3].to('cpu') + torch.from_numpy(camera_direction)   
                        env.set_camera(camera_position, camera_position - torch.from_numpy(camera_direction))

                    # if i < stop_state_log and i > start_state_log:
                    #     logger.log_states(
                    #         {
                    #             'hip_pitch_vel': env.dof_vel[robot_index, 2].item(),
                    #             'knee_pitch_vel': env.dof_vel[robot_index, 3].item(),
                    #             'ankle_pitch_vel': env.dof_vel[robot_index, 4].item(),
                    #             'hip_pitch_pos': env.dof_pos[robot_index, 2].item(),
                    #             'knee_pitch_pos': env.dof_pos[robot_index, 3].item(),                    
                    #             'ankle_pitch_pos': env.dof_pos[robot_index, 4].item(),                    
                    #             'hip_pitch_torque': env.torques[robot_index, 2].item(),
                    #             'knee_pitch_torque': env.torques[robot_index, 3].item(),
                    #             'ankle_pitch_torque': env.torques[robot_index, 4].item(),
                    #             'command_x': env.commands[robot_index, 0].item(),
                    #             'command_y': env.commands[robot_index, 1].item(),
                    #             'command_yaw': env.commands[robot_index, 2].item(),
                    #             'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    #             'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    #             'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    #             'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    #             'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    #         }
                    #     )
                    #     if hasattr(env, 'commands_sinusoid'):
                    #         logger.log_states({
                    #             'sinusoid_command_x': env.commands_sinusoid[robot_index, 0].item(),
                    #         })
                    # elif i==stop_state_log:
                    #     print("Plot states to : ", SAVE_DIR)
                    #     logger.plot_states_mainthread(save_fig=True, save_dir=SAVE_DIR)
                    # if  0 < i < stop_rew_log:
                    #     if infos["episode"]:
                    #         num_episodes = torch.sum(env.reset_buf).item()
                    #         if num_episodes>0:
                    #             logger.log_rewards(infos["episode"], num_episodes)
                    # elif i==stop_rew_log:
                    #     logger.print_rewards()
                print("Disturbance : ", disturbance, ", SUCCESS RATE : ", 1.-float(fail_count / total_count))
                success_rate.append(1.-float(fail_count / total_count))

            # plot using success rate and disturbance
            if task == 'bolt6':
                label = 'Bolt'
            elif task == 'bolt10inverted':
                label = 'Bolt10'
            elif task == 'bolt10fixed':
                label = 'Bolt10 fixed'
            else:
                label = task
            a = axs[compare_index]
            a.plot(disturbances, success_rate, label = label)
            env.destroy_sim()
            torch.cuda.empty_cache()
        a.set(xlabel=list_of_comparisons[compare_index]+' External Force, [N]', ylabel='Survival Rate')
        a.legend()
    fig.suptitle('Robustness Comparison')
    plt.savefig('push experiment.png')
    plt.clf()


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    # play(args)
    custom_play_disturbance(args)