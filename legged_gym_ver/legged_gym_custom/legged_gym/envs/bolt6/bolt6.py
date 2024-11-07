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

from time import time
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.custom_terrain import custom_Terrain

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
import math

class Bolt6(LeggedRobot):

    def _custom_init(self, cfg):
        self.control_tick = torch.zeros(
            self.num_envs, 1, dtype=torch.int,
            device=self.device, requires_grad=False)
        self.ext_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.ext_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.curriculum_index=0
        if self.num_privileged_obs is not None:
            self.dof_props = torch.zeros((self.num_dofs, 2), device=self.device, dtype=torch.float) # includes dof friction (0) and damping (1) for each environment
    
    ###############################################################################
    # Pseudo Code yet
    # phase setup
    
    def _get_phase_state(self):
        phase_time = 0.3
        self.phase_duration = 0.5
        phi = (phase_time % self.phase_duration) * 2 * math.pi
        swing_active = torch.exp(self.cfg.domain_rand.kappa * (torch.cos(torch.tensor([phi],device = self.device) - 1)))
        return swing_active
    
    # phase reward on swing and stance -> mark # nofly and standstill
    def _reward_phase(self):
        self.stance_weight = 0.5
        self.swing_weight = 0.5
        swing_phase = self._get_phase_state()
        # Reward for stance phase: minimize foot velocity, maximize foot force
        stance_reward = self.stance_weight * torch.norm(self.contact_forces[:, self.feet_indices, 2]) * (self.phase_duration - swing_phase) # 1step per 0.5sec
        # Reward for swing phase: minimize foot force, allow foot velocity
        swing_reward = self.swing_weight * (self.phase_duration-torch.norm(self.contact_forces[:, self.feet_indices, 2])) * swing_phase 
        
        return stance_reward + swing_reward
    ###############################################################################

    ###############################################################################
    # penalize energy usage
    def _reward_energy(self):
        return torch.sum(self.torques * self.dof_vel, dim=-1)
    ###############################################################################
    
    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.001
        single_contact = torch.sum(1.*contacts, dim=1)==1
        single_contact *= torch.norm(self.commands[:, :3], dim=1) > 0.1 #no reward for zero command
        return 1.*single_contact
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.001
        single_contact = torch.sum(1.*contacts, dim=1) >0
        contact_filt = torch.logical_or(contacts, self.last_contacts) 
        self.last_contacts = contacts
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum( torch.clip(self.feet_air_time - 0.3, min=0.0, max=0.7) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > 0.1 #no reward for zero command
        rew_airTime *= single_contact #no reward for flying or double support
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        joint_error = torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return torch.exp(-joint_error/self.cfg.rewards.tracking_sigma) * (torch.norm(self.commands[:, :3], dim=1) <= 0.1)

    def _reward_torques(self):
        # Penalize torques
        return torch.mean(torch.square(self.torques), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.mean(torch.square( (self.last_actions - self.actions)/self.dt ), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        orientation_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        # print("orientation_error: ", orientation_error[0])
        return torch.exp(-orientation_error/self.cfg.rewards.orientation_sigma)

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_base_height(self):
        # Reward tracking specified base height
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        error = (base_height-self.cfg.rewards.base_height_target)
        error = error.flatten()
        return torch.exp(-torch.square(error)/self.cfg.rewards.tracking_sigma)

    def _reward_joint_regularization(self):
        # Reward joint poses and symmetry
        error = 0.
        # Yaw joints regularization around 0
        # error += self.sqrdexp(
        #     (self.dof_pos[:, 0]) / self.cfg.normalization.obs_scales.dof_pos)
        # error += self.sqrdexp(
        #      (self.dof_pos[:, 5]) / self.cfg.normalization.obs_scales.dof_pos)
        # Ab/ad joint symmetry
        error += self.sqrdexp(
            ( (self.dof_pos[:, 0] ) - (self.dof_pos[:, 3]) )
            / self.cfg.normalization.obs_scales.dof_pos)
        # Pitch joint symmetry
        # print("self.dof_pos[0, 6]: ", self.dof_pos[0, 1], "// self.dof_pos[0, 6]: ", self.dof_pos[0, 6])
        # error += 0.8 * self.sqrdexp(
        #     ( self.dof_pos[:, 2] + self.dof_pos[:, 7])
        #     / self.cfg.normalization.obs_scales.dof_pos)        
        # error +=  self.sqrdexp(
        #     ( self.dof_vel[:, 4] + self.dof_vel[:, 9])
        #     / self.cfg.normalization.obs_scales.dof_vel) 
        # error += self.sqrdexp(
        #     ( self.dof_vel[:, 9])
        #     / self.cfg.normalization.obs_scales.dof_vel)         
        return error

    def _reward_feet_contact_forces(self):
        return 1.- torch.exp(-0.07 * torch.norm((self.contact_forces[:, self.feet_indices, 2] -  1.2 * 9.81 * self.robot_mass.mean()).clip(min=0.), dim=1)) 

    # Potential-based rewards
    
    def _reward_ori_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_orientation() - self.rwd_oriPrev)
        return delta_phi / self.dt

    def _reward_jointReg_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_joint_regularization() - self.rwd_jointRegPrev)
        return delta_phi / self.dt

    def _reward_baseHeight_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_base_height() - self.rwd_baseHeightPrev)
        return delta_phi / self.dt

    def _reward_action_rate_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_action_rate() - self.rwd_actionRatePrev)
        return delta_phi / self.dt

    def _reward_stand_still_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_stand_still() - self.rwd_standStillPrev)
        return delta_phi / self.dt

    def _reward_no_fly_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_no_fly() - self.rwd_noFlyPrev)
        return delta_phi / self.dt

    def _reward_feet_air_time_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_feet_air_time() - self.rwd_feetAirTimePrev)
        return delta_phi / self.dt

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        
        if hasattr(self, "_custom_reset"):
            self._custom_reset(env_ids)
            
        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s# / self.reward_scales[key]
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        if self.cfg.ERFI.isERFI:
            self.force_bias[env_ids, :] = torch.rand(env_ids.shape[0], self.num_actions, device = self.device) * self.cfg.ERFI.force_bias_scale
        
        if self.cfg.domain_rand.randomize_torque_constant:
            self.torque_constant[env_ids] = 1 + (2*torch.rand((env_ids.shape[0], self.num_actions), device = self.device) -1)* self.cfg.domain_rand.torque_constant_range


    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.torque_constant = torch.ones_like(self.torques)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) 
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.height_points = self._init_height_points()
        self.measured_heights = self._get_heights()

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
                    
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.cfg.asset.name)
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)        
        self.jacobian = gymtorch.wrap_tensor(_jacobian).flatten(1,2) # originally shape of (num_envs, num_bodies, 6, num_dofs+6)
        # The jacobian maps joint velocities (num_dofs + 6) to spatial velocities of CoM frame of each link in global frame
        # https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/docs/Articulations.html#jacobian
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        self.rb_inertia = gymtorch.torch.zeros((self.num_envs, self.num_bodies, 3, 3), device=self.device) # [Ix, Iy, Iz]
        self.rb_mass = gymtorch.torch.zeros((self.num_envs, self.num_bodies), device=self.device) # link mass
        self.rb_com = gymtorch.torch.zeros((self.num_envs, self.num_bodies, 3), device = self.device) # [comX, comY, comZ] in link's origin frame 
        self.com_position = gymtorch.torch.zeros((self.num_envs, 3), device=self.device) # robot-com position in global frame
        
        # Reconstruct rb_props as tensor        
        for env in range(self.num_envs):
            for key, N in self.body_names_dict.items():
                rb_props = self.gym.get_actor_rigid_body_properties(self.envs[env], 0)[N]
                # inertia tensors are about link's CoM frame
                self.rb_com[env, N, :] = gymtorch.torch.tensor([rb_props.com.x, rb_props.com.y, rb_props.com.z], device=self.device)
                self.rb_inertia[env, N, 0, :] = gymtorch.torch.tensor([rb_props.inertia.x.x, -rb_props.inertia.x.y, -rb_props.inertia.x.z], device=self.device)
                self.rb_inertia[env, N, 1, :] = gymtorch.torch.tensor([-rb_props.inertia.y.x, rb_props.inertia.y.y, -rb_props.inertia.y.z], device=self.device)
                self.rb_inertia[env, N, 2, :] = gymtorch.torch.tensor([-rb_props.inertia.z.x, -rb_props.inertia.z.y, rb_props.inertia.z.z], device=self.device)
                # see how inertia tensor is made : https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/dd277ec654440f4c2b5b07d6c286c3fd_MIT16_07F09_Lec26.pdf
                self.rb_mass[env, N] = rb_props.mass
        self.robot_mass = torch.sum(self.rb_mass, dim=1).unsqueeze(1)

        # self.rb_{property} can be used for dynamics calculation

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = custom_Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def destroy_sim(self):
        """Destroy simulation, terrain and environments
        """
        self.gym.destroy_sim(self.sim)
        print("Simulation destroyed")

    def compute_observations(self):
        """ Computes observations
        """
        self.contacts = self.contact_forces[:, self.feet_indices, 2] > 0.001
        self.obs_buf = torch.cat((  
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            self.obs_buf = torch.cat((self.obs_buf, torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements), dim=-1)
        # add noise if needed
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat((
                self.obs_buf,
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel  * self.obs_scales.ang_vel,
                # self.contacts,
                # self.contact_forces.squeeze(), # When trimesh, this is unreliable
                self.ext_forces[:, 0, :]* self.obs_scales.ext_forces / self.robot_mass,
                self.ext_torques[:, 0, :]* self.obs_scales.ext_torques/ self.robot_mass,
                self.friction_coeffs* self.obs_scales.friction_coeffs,
                self.dof_props[:, 0].repeat(self.num_envs, 1)*self.obs_scales.dof_friction,
                self.dof_props[:, 1].repeat(self.num_envs, 1)*self.obs_scales.dof_damping,
                torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
                ), dim=-1)
        # if self.add_noise:
        #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.
        noise_vec[9:19] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[19:29] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[29:39] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[39:] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    def pre_physics_step(self):
        self.rwd_oriPrev = self._reward_orientation()
        self.rwd_baseHeightPrev = self._reward_base_height()
        self.rwd_jointRegPrev = self._reward_joint_regularization()
        self.rwd_actionRatePrev = self._reward_action_rate()

        self.rwd_standStillPrev = self._reward_stand_still()
        self.rwd_noFlyPrev = self._reward_no_fly()
        self.rwd_feetAirTimePrev = self._reward_feet_air_time()
        if self.cfg.domain_rand.randomize_dof_friction and  (self.common_step_counter % self.cfg.domain_rand.dof_friction_interval == 0):
            props = self.gym.get_actor_dof_properties(self.envs[0], 0)
            for dof in range(self.num_dofs):
               props["friction"][dof] = np.random.uniform(*self.cfg.domain_rand.dof_friction)
               props["damping"][dof] = np.random.uniform(*self.cfg.domain_rand.dof_damping)
            for envs in self.envs:
                self.gym.set_actor_dof_properties(envs, 0, props)
            if self.num_privileged_obs is not None:
                self.dof_props[:, 0] =  torch.tensor(props["friction"], device=self.device)
                self.dof_props[:, 1] = torch.tensor(props["damping"], device=self.device)
        if self.cfg.domain_rand.ext_force_robots and (self.common_step_counter % self.cfg.domain_rand.ext_force_randomize_interval == 0):
            self.cfg.domain_rand.ext_force_duration = np.ceil(np.random.uniform(*self.cfg.domain_rand.ext_force_duration_s) / self.dt) 

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device) # at reset, action is zero.
        # step physics and render each frame
        self.pre_physics_step()
        self.render(sync_frame_time=True)

        for _ in range(self.cfg.control.decimation): # compute torque 4 times
            
            if self.cfg.domain_rand.ext_force_robots and  (self.common_step_counter % self.cfg.domain_rand.ext_force_interval < self.cfg.domain_rand.ext_force_duration) and self.curriculum_index > 0:  
                if self.common_step_counter % self.cfg.domain_rand.ext_force_interval == 0:
                    scale = np.random.uniform(*self.cfg.domain_rand.ext_force_scale_range)
                    angle = np.random.uniform(*self.cfg.domain_rand.ext_force_direction_range)
                    self.ext_forces[:, 0, :] = torch.tensor([scale*math.cos(angle), scale*math.sin(angle), 0], device=self.device, requires_grad=False)    #index: root, body, force axis(6)
                    self.ext_torques[:, 0, :] = torch.tensor([0, 0, 0], device=self.device, requires_grad=False)                
            else:
                self.ext_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
                self.ext_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)            
            
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.ext_forces), gymtorch.unwrap_tensor(self.ext_torques), gymapi.ENV_SPACE)
            
            self.torques = self._compute_torques(self.actions).view(self.torques.shape) 
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques)) 
            self.gym.simulate(self.sim)
            # print("------------TORQUES------------")
            # print(self.torques)
            # print("-------------------------------")
            
            # ## set custom state DEBUGGING
            # self.dof_pos[:, :] = 5
            # self.dof_vel[:, :] = -3
            # self.root_states[:, 0:3] = 2
            # self.root_states[:, 3:7] = torch.tensor([.5, .5, .5, .5]) # xyzw
            # self.root_states[:, 7:10] = -1
            # self.root_states[:, 10:13] = -0.2
            # self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
            # self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            
        self.post_physics_step()
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def step_count_failures(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device) # at reset, action is zero.
        # step physics and render each frame
        self.pre_physics_step()
        self.render()

        for _ in range(self.cfg.control.decimation): # compute torque decimation times
            if self.cfg.domain_rand.ext_force_robots and  (self.common_step_counter % self.cfg.domain_rand.ext_force_interval < self.cfg.domain_rand.ext_force_duration) and self.curriculum_index > 0:  
                if self.common_step_counter % self.cfg.domain_rand.ext_force_interval == 0:
                    scale = np.random.uniform(*self.cfg.domain_rand.ext_force_scale_range)
                    angle = np.random.uniform(*self.cfg.domain_rand.ext_force_direction_range)
                    self.ext_forces[:, 0, :] = torch.tensor([scale*math.cos(angle), scale*math.sin(angle), 0], device=self.device, requires_grad=False)    #index: root, body, force axis(6)
                    self.ext_torques[:, 0, :] = torch.tensor([0, 0, 0], device=self.device, requires_grad=False)                    
            else:
                self.ext_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
                self.ext_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)            
            
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.ext_forces), gymtorch.unwrap_tensor(self.ext_torques), gymapi.ENV_SPACE)
                        
            self.torques = self._compute_torques(self.actions).view(self.torques.shape) # zero action means its trying to go to default joint positions. What is the current joint position?
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques)) # Now its set as default joint position. So the torque should be 0? Because dof vel is 0
            self.gym.simulate(self.sim)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            
        self.post_physics_step()
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.failure_buf, self.extras 

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.4, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.4, 0., self.cfg.commands.max_curriculum)
            self.curriculum_index += 1
        # If the agent does not terminate

    def custom_post_physics_step(self):
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)        
        self.control_tick += 1
        self.measured_heights = self._get_heights()
            
    def _custom_reset(self, env_ids):
        self.control_tick[env_ids, 0] = 0   
        
    def _custom_create_envs(self):
        collision_mask = [3,8] # List of shapes for which collision must be detected
        if self.cfg.domain_rand.randomize_friction:
            # prepare friction randomization
            friction_range = self.cfg.domain_rand.friction_range
            self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)
        else:
            self.friction_coeffs = torch.ones((self.num_envs, 1), device=self.device)
        for i, env in enumerate(self.envs):
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env, 0)
            for j in range(len(rigid_shape_props)):
                if j not in collision_mask:
                    rigid_shape_props[j].filter=1
                rigid_shape_props[j].friction=self.friction_coeffs[i, 0]
            self.gym.set_actor_rigid_shape_properties(env, 0, rigid_shape_props)
        # for name, num in self.body_names_dict.items():
        #     shape_id = self.body_to_shapes[num]
        #     print("body : ", name, ", shape index start : ", shape_id.start, ", shape index count : ", shape_id.count)
        #     for i in range(shape_id.start, shape_id.start + shape_id.count):
        #         print("shape ", i, " filter : ", self.gym.get_actor_rigid_shape_properties(self.envs[0], 0)[i].filter)
        #         print("shape ", i, " contact offset : ", self.gym.get_actor_rigid_shape_properties(self.envs[0], 0)[i].contact_offset)
                # I guess the best I can try is set the shin's bitmasks as 0           

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        # actions_scaled = actions * self.cfg.control.action_scale      
        # actions_scaled = torch.clamp(actions, min=-1., max=1.)                                                                                 
        actions_scaled = torch.clamp(actions, min=-1., max=1.) #+ self.force_bias + torch.rand(actions.shape, device=self.device) * self.cfg.ERFI.force_noise_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled * self.torque_limits[:self.num_actions] * self.torque_constant
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)



    def _custom_parse_cfg(self, cfg):
        self.cfg.domain_rand.ext_force_interval = np.ceil(self.cfg.domain_rand.ext_force_interval_s / self.dt)
        self.cfg.domain_rand.ext_force_randomize_interval = np.ceil(self.cfg.domain_rand.ext_force_randomize_interval_s / self.dt)
        self.cfg.domain_rand.dof_friction_interval = np.ceil(self.cfg.domain_rand.dof_friction_interval_s / self.dt)      
        
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props
    
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1).unsqueeze(1)    
    
    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer,None, cam_pos, cam_target)
        
        # ##################### HELPER FUNCTIONS ################################## #

    def sqrdexp(self, x):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(x)/self.cfg.rewards.tracking_sigma)