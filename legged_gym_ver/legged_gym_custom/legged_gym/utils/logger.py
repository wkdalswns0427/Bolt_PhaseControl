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

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states_mainthread(self, save_fig = False, save_dir = None):
        nb_rows = 3 # 3
        nb_cols = 3 # 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(4., 4.+len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot joint velocity 
        a = axs[1, 0]
        if log["hip_pitch_vel"]: a.plot(time, log["hip_pitch_vel"], label='measured')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Hip pitch Velocity')
        a.legend()
        # plot joint velocity        
        a = axs[1, 1]
        if log["knee_pitch_vel"]: a.plot(time, log["knee_pitch_vel"], label='measured')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Knee pitch Velocity')
        a.legend()        
        # plot joint velocity        
        a = axs[1, 2]
        if log["ankle_pitch_vel"]: a.plot(time, log["ankle_pitch_vel"], label='measured')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Ankle pitch Velocity')
        a.legend()  
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        if log["sinusoid_command_x"]: a.plot(time, log["sinusoid_command_x"], label='sinusoid')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()
        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend()
        # plot base vel yaw
        a = axs[0, 2]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.legend()
        # plot contact forces
        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
        a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        a.legend()
        # plot torque/vel curves
        # a = axs[2, 1]
        # if log["dof_vel"]!=[] and log["dof_torque"]!=[]: a.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
        # a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
        # a.legend()
        # plot torques
        a = axs[2, 1]
        if log["ankle_pitch_torque"]!=[]: a.plot(time, log["ankle_pitch_torque"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='ankle Pitch Torque')
        a.legend()     
        a = axs[2, 2]
        if log["knee_pitch_torque"]!=[]: a.plot(time, log["knee_pitch_torque"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Knee Pitch Torque')
        a.legend()        
        
        plt.show
        if save_fig:
            if save_dir is None:
                raise ValueError("Directory must be specified to save plots")
            plt.savefig(save_dir)
        

    def plot_states(self, save_fig = False, save_dir = None):
        self.plot_process = Process(target=self._plot, args=(save_fig, save_dir))
        self.plot_process.start()
        
        
    def _plot(self, save_fig = False, save_dir = None):
        nb_rows = 3 # 3
        nb_cols = 3 # 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        fig.set_size_inches(10, 10)
        for key, value in self.state_log.items():
            time = np.linspace(4., 4.+len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot joint velocity 
        a = axs[1, 0]
        if log["hip_pitch_vel"]: a.plot(time, log["hip_pitch_vel"], label='measured')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Hip pitch Velocity')
        a.legend()
        # plot joint velocity        
        a = axs[1, 1]
        if log["knee_pitch_vel"]: a.plot(time, log["knee_pitch_vel"], label='measured')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Knee pitch Velocity')
        a.legend()        
        # plot joint velocity        
        a = axs[1, 2]
        if log["ankle_pitch_vel"]: a.plot(time, log["ankle_pitch_vel"], label='measured')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Ankle pitch Velocity')
        a.legend()  
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()
        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend()
        # plot base vel yaw
        a = axs[0, 2]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.legend()
        # plot contact forces
        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
        a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        a.legend()
        # plot torque/vel curves
        # a = axs[2, 1]
        # if log["dof_vel"]!=[] and log["dof_torque"]!=[]: a.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
        # a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
        # a.legend()
        # plot torques
        a = axs[2, 1]
        if log["ankle_pitch_torque"]!=[]: a.plot(time, log["ankle_pitch_torque"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='ankle_pitch_torque')
        a.legend()     
        a = axs[2, 2]
        if log["knee_pitch_torque"]!=[]: a.plot(time, log["knee_pitch_torque"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='knee_pitch_torque')
        a.legend()        
        
        # plt.show
        if save_fig:
            if save_dir is None:
                raise ValueError("Directory must be specified to save plots")
            plt.savefig(save_dir)


    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()