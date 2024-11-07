# bolt6_jump.py

from time import time
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.custom_terrain import custom_Terrain

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobotJump
from legged_gym.envs import LeggedRobot
import math


class Bolt6J(LeggedRobotJump):

    def _custom_init(self, cfg):
        self.control_tick = torch.zeros(
            self.num_envs, 1, dtype=torch.int,
            device=self.device, requires_grad=False)
        self.ext_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.ext_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.curriculum_index = 0
        if self.num_privileged_obs is not None:
            self.dof_props = torch.zeros((self.num_dofs, 2), device=self.device, dtype=torch.float)  # includes dof friction (0) and damping (1) for each environment
    
    def _reward_energy(self):
        return -torch.mean(torch.matmul(self.torques,self.dof_vel.T), dim=1)

    def _reward_safe_landing(self):
        # Check if both feet are in contact with the ground
        feet_in_contact = torch.all(self.contact_forces[:, self.feet_indices, 2] > 0.001, dim=1)

        # Check if the robot is upright
        orientation_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        upright_orientation = orientation_error < 0.1

        # If both conditions are met, we consider it a proper landing
        proper_landing = feet_in_contact & upright_orientation

        return proper_landing
    
    def _reward_knee_straightness(self):
        knee_deviation = torch.abs(self.dof_pos[:, [2,5]] - self.knee_target_position)
        knee_straightness_penalty = torch.sum(knee_deviation, dim=1)

        # knee_extension_influence = 1.0  # Strength of the influence
        # self.actions[:, [2, 5]] = knee_extension_influence * self.knee_target_position + (1.0 - knee_extension_influence) * self.actions[:, [2, 5]]

        return -knee_straightness_penalty
    
    # def _reward_hip_cross(self):
    #     hip_distance = torch.abs(self.dof_pos[:, 0] - self.dof_pos[:, 3])
    #     crossing_penalty = torch.exp(-hip_distance)
    #     return -crossing_penalty


    def _reward_feet_air_time(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.001
        single_contact = torch.sum(1. * contacts, dim=1) > 0
        contact_filt = torch.logical_or(contacts, self.last_contacts)
        self.last_contacts = contacts
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(torch.clip(self.feet_air_time - 0.3, min=0.0, max=0.7) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= self.commands[:, 2].abs() > 0.1  # no reward for zero command
        rew_airTime *= single_contact  # no reward for flying or double support
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stand_still(self):
        joint_error = torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return torch.exp(-joint_error / self.cfg.rewards.tracking_sigma) * (torch.norm(self.commands[:, :3], dim=1) <= 0.1)

    # def _reward_jump_height(self):
    #     jump_height = self.root_states[:, 2]
    #     return jump_height

    def _reward_torques(self):
        return torch.mean(torch.square(self.torques), dim=1)

    def _reward_action_rate(self):
        return torch.mean(torch.square((self.last_actions - self.actions) / self.dt), dim=1)

    def _reward_orientation(self):
        orientation_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return torch.exp(-orientation_error / self.cfg.rewards.orientation_sigma)

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :1]), dim=1)

    def _reward_target_height(self):
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        error = (self.cfg.rewards.base_height_target - base_height)
        error = error.flatten()
        return torch.exp(-torch.square(error) / self.cfg.rewards.tracking_sigma)

    def _reward_joint_regularization(self):
        error = 0.
        error += self.sqrdexp(
            ((self.dof_pos[:, 0]) - (self.dof_pos[:, 3])) / self.cfg.normalization.obs_scales.dof_pos)
        return error

    def _reward_feet_contact_forces(self):
        return 1. - torch.exp(-0.07 * torch.norm((self.contact_forces[:, self.feet_indices, 2] - 1.2 * 9.81 * self.robot_mass.mean()).clip(min=0.), dim=1))

    def _reward_ori_pb(self):
        delta_phi = ~self.reset_buf \
                    * (self._reward_orientation() - self.rwd_oriPrev)
        return delta_phi / self.dt

    def _reward_jointReg_pb(self):
        delta_phi = ~self.reset_buf \
                    * (self._reward_joint_regularization() - self.rwd_jointRegPrev)
        return delta_phi / self.dt

    def _reward_targetHeight_pb(self):
        delta_phi = ~self.reset_buf \
                    * (self._reward_target_height() - self.rwd_targetHeightPrev)
        return delta_phi / self.dt

    def _reward_action_rate_pb(self):
        delta_phi = ~self.reset_buf \
                    * (self._reward_action_rate() - self.rwd_actionRatePrev)
        return delta_phi / self.dt

    def _reward_stand_still_pb(self):
        delta_phi = ~self.reset_buf \
                    * (self._reward_stand_still() - self.rwd_standStillPrev)
        return delta_phi / self.dt

    def _reward_feet_air_time_pb(self):
        delta_phi = ~self.reset_buf \
                    * (self._reward_feet_air_time() - self.rwd_feetAirTimePrev)
        return delta_phi / self.dt

    def _init_buffers(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.height_points = self._init_height_points()
        self.measured_heights = self._get_heights()

        self.jump_phase = [True]*self.num_envs
        self.knee_target_position = torch.tensor([self.cfg.init_state.default_joint_angles['FL_KFE'],
                                         self.cfg.init_state.default_joint_angles['FR_KFE']],
                                        device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.hip_target_position = torch.tensor([self.cfg.init_state.default_joint_angles['FL_HFE'],
                                        self.cfg.init_state.default_joint_angles['FR_HFE']],
                                       device=self.device).unsqueeze(0).repeat(self.num_envs, 1)


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
        self.jacobian = gymtorch.wrap_tensor(_jacobian).flatten(1, 2)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        self.rb_inertia = gymtorch.torch.zeros((self.num_envs, self.num_bodies, 3, 3), device=self.device)
        self.rb_mass = gymtorch.torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self.rb_com = gymtorch.torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.com_position = gymtorch.torch.zeros((self.num_envs, 3), device=self.device)

        for env in range(self.num_envs):
            for key, N in self.body_names_dict.items():
                rb_props = self.gym.get_actor_rigid_body_properties(self.envs[env], 0)[N]
                self.rb_com[env, N, :] = gymtorch.torch.tensor([rb_props.com.x, rb_props.com.y, rb_props.com.z], device=self.device)
                self.rb_inertia[env, N, 0, :] = gymtorch.torch.tensor([rb_props.inertia.x.x, -rb_props.inertia.x.y, -rb_props.inertia.x.z], device=self.device)
                self.rb_inertia[env, N, 1, :] = gymtorch.torch.tensor([-rb_props.inertia.y.x, rb_props.inertia.y.y, -rb_props.inertia.y.z], device=self.device)
                self.rb_inertia[env, N, 2, :] = gymtorch.torch.tensor([-rb_props.inertia.z.x, -rb_props.inertia.z.y, rb_props.inertia.z.z], device=self.device)
                self.rb_mass[env, N] = rb_props.mass
        self.robot_mass = torch.sum(self.rb_mass, dim=1).unsqueeze(1)

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = custom_Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognized. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def destroy_sim(self):
        self.gym.destroy_sim(self.sim)
        print("Simulation destroyed")

    def compute_observations(self):
        self.contacts = self.contact_forces[:, self.feet_indices, 2] > 0.001
        self.obs_buf = torch.cat((
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)

        if self.cfg.terrain.measure_heights:
            self.obs_buf = torch.cat((self.obs_buf, torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements), dim=-1)

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat((
                self.obs_buf,
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.ext_forces[:, 0, :] * self.obs_scales.ext_forces / self.robot_mass,
                self.ext_torques[:, 0, :] * self.obs_scales.ext_torques / self.robot_mass,
                self.friction_coeffs * self.obs_scales.friction_coeffs,
                self.dof_props[:, 0].repeat(self.num_envs, 1) * self.obs_scales.dof_friction,
                self.dof_props[:, 1].repeat(self.num_envs, 1) * self.obs_scales.dof_damping,
                torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            ), dim=-1)

    def pre_physics_step(self):
        self.rwd_oriPrev = self._reward_orientation()
        self.rwd_targetHeightPrev = self._reward_target_height()
        self.rwd_jointRegPrev = self._reward_joint_regularization()
        self.rwd_actionRatePrev = self._reward_action_rate()
        self.rwd_standStillPrev = self._reward_stand_still()
        self.rwd_feetAirTimePrev = self._reward_feet_air_time()

    def step(self, actions):
        self.actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions).to(self.device)
        self.pre_physics_step()
        self.render(sync_frame_time=True)
        
        # Vectorized control logic instead of looping through each environment
        jump_force = torch.zeros_like(self.ext_forces)  
        # Apply the jump force for a limited number of steps
        jump_phase = (self.control_tick < 10).unsqueeze(-1).float()
        jump_force[:, 0, 2] = jump_phase.squeeze() * 30.0
        self.ext_forces[:, 0, :] = jump_force[:, 0, :]

        # self.actions[:, 0] = self.actions[:, 3]  # Symmetrize hip roll (HAA)
        # self.actions[:, 1] = self.actions[:, 4]  # Symmetrize hip pitch (HFE) but in opposite directions
        # self.actions[:, 2] = self.actions[:, 5]  # Symmetrize knee pitch (KFE)

        straightness_factor = 0.8  # Strength of the straightening effect (0.0 to 1.0)
        self.actions[:,[2,5]] = straightness_factor * self.knee_target_position + (1.0 - straightness_factor) * self.actions[:, [2,5]]
        self.actions[:, [1,4]] = straightness_factor * self.hip_target_position + (1.0 - straightness_factor) * self.actions[:, [1,4]]

        for _ in range(self.cfg.control.decimation):
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.ext_forces), gymtorch.unwrap_tensor(self.ext_torques), gymapi.ENV_SPACE)
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        self.post_physics_step()
        self.obs_buf = torch.clip(self.obs_buf, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def update_command_curriculum(self, env_ids):
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.4, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_z"][1] = np.clip(self.command_ranges["lin_vel_z"][1] + 0.4, 0., self.cfg.commands.max_curriculum)
            self.curriculum_index += 1

    def custom_post_physics_step(self):
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.control_tick += 1
        self.measured_heights = self._get_heights()

    def _custom_reset(self, env_ids):
        self.control_tick[env_ids, 0] = 0

    def _custom_create_envs(self):
        collision_mask = [3, 8]
        if self.cfg.domain_rand.randomize_friction:
            friction_range = self.cfg.domain_rand.friction_range
            self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs, 1), device=self.device)
        else:
            self.friction_coeffs = torch.ones((self.num_envs, 1), device=self.device)
        for i, env in enumerate(self.envs):
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env, 0)
            for j in range(len(rigid_shape_props)):
                if j not in collision_mask:
                    rigid_shape_props[j].filter = 1
                rigid_shape_props[j].friction = self.friction_coeffs[i, 0]
            self.gym.set_actor_rigid_shape_properties(env, 0, rigid_shape_props)

    def _custom_parse_cfg(self, cfg):
        self.cfg.domain_rand.ext_force_interval = np.ceil(self.cfg.domain_rand.ext_force_interval_s / self.dt)
        self.cfg.domain_rand.ext_force_randomize_interval = np.ceil(self.cfg.domain_rand.ext_force_randomize_interval_s / self.dt)
        self.cfg.domain_rand.dof_friction_interval = np.ceil(self.cfg.domain_rand.dof_friction_interval_s / self.dt)

    def _process_rigid_shape_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["lin_vel_z"][0], self.command_ranges["lin_vel_z"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1).unsqueeze(1)

    def set_camera(self, position, lookat):
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def sqrdexp(self, x):
        return torch.exp(-torch.square(x) / self.cfg.rewards.tracking_sigma)

