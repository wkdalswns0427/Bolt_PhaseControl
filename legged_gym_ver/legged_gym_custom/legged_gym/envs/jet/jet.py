from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot

class Jet(LeggedRobot):
    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact
    
    def _reward_angular_momentum(self):
        self._compute_centroidal_dynamics()
        return torch.square(self.centroidal_momentum.reshape(self.num_envs, -1)[:, 5])
    
    def _reward_upper_motion(self):
        return torch.sum(torch.square(self.dof_pos[:, 12:] - self.default_dof_pos[:, 12:]), dim=1)
       
    def _reward_lower_motion(self):
        return torch.sum(torch.square((self.dof_vel[:, 0:12])), dim=1)
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # Assume right foot has been in contact for .3 seconds.
        # left foot has just landed on ground
        contact = self.contact_forces[:, self.feet_indices, 2] > 1. # [0, 1]
        contact_filt = torch.logical_or(contact, self.last_contacts) # [1, 1]
        self.last_contacts = contact # [1, 1]
        first_contact = (self.feet_air_time > 0.) * contact_filt # [1 1]
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt # set to 0 if feet was on ground or is on ground
        return rew_airTime    
    
    
        #######NOTE#######
        # The jacobian maps the joint velocities to the body's CoM frame velocities
        # The body's CoM frame is determined by the URDF files.
        # The spatial link velocities that the matrix maps to are with respect to the center of mass (COM) of the links, 
        # and are stacked [vx; vy; vz; wx; wy; wz], where vx and wx refer to the linear and rotational velocity IN WORLD FRAME, respectively.
        # The body's origin frame is the joint's frame which connects the link to its parent.
        # The body's CoM frame is described in the URDF file under the tag <inertial>
        # The <origin> tag describes the CoM frame's linear and angular offset from the body's origin frame


    def _compute_centroidal_dynamics(self):
        #reset robot-com position
        # local variables used for computation
        transform = torch.zeros((self.num_envs, 6*self.num_bodies, 6), device=self.device) # the projection matrix
        transform_from_inertial_to_com = torch.zeros((self.num_envs, 6*self.num_bodies, 6*self.num_bodies), device=self.device)
        # the jacobian from physX gives us spatial velocity of the link-CoM frame about the inertial frame
        # https://forums.developer.nvidia.com/t/differences-between-isaac-sim-and-mujoco-jacobians/252755
        # transform_from_inertial_to_com transforms the twist in inertial frame to in link-CoM frame.
        # This way, the inertia matrix are about the link-CoM frame, thus much more simple.
        unflatten = gymtorch.torch.nn.Unflatten(0, (self.num_envs, self.num_bodies))
        rb_states = unflatten(self.rb_states) # states are in inertial frame, about the joint frame
        link_r = torch.zeros((self.num_envs, len(self.body_names_dict), 3, 3), device=self.device) # link rotation matrix in inertial frame
        link_com = torch.zeros((self.num_envs, len(self.body_names_dict), 3), device=self.device) # link com position in inertial frame
        tmp_com = torch.zeros((self.num_envs, 3), device=self.device) # for CoM computation
        tmp_mass = torch.zeros((self.num_envs, 3), device=self.device) # shape is (num_envs, 3) to use in CoM computation
        inertia_mat = gymtorch.torch.zeros((self.num_envs, 6*self.num_bodies, 6*self.num_bodies), device = self.device)
        Sc = gymtorch.torch.zeros((self.num_envs, 3, 3), device=self.device) # for computation. Skew of some vector c
        
        for key, N in self.body_names_dict.items():
            mass = self.rb_mass[:, N]
            # inertia matrix
            inertia_mat[:, 6*N+3:6*N+6, 6*N+3:6*N+6] = self.rb_inertia[:, N, :, :]
            inertia_mat[:, 6*N, 6*N] = mass
            inertia_mat[:, 6*N+1, 6*N+1] = mass
            inertia_mat[:, 6*N+2, 6*N+2] = mass
            
            # link rotation matrix, link com position, robot com position computed
            quats = rb_states[:, N, 3:7] # G_R_i
            link_r[:, N, 0, 0] = 2*(quats[:, 0] * quats[:, 0] + quats[:, 1] * quats[:, 1])-1
            link_r[:, N, 0, 1] = 2*(quats[:, 1] * quats[:, 2] - quats[:, 0] * quats[:, 3])
            link_r[:, N, 0, 2] = 2*(quats[:, 1] * quats[:, 3] + quats[:, 0] * quats[:, 2])
            link_r[:, N, 1, 0] = 2*(quats[:, 1] * quats[:, 2] + quats[:, 0] * quats[:, 3])
            link_r[:, N, 1, 1] = 2*(quats[:, 0] * quats[:, 0] + quats[:, 2] * quats[:, 2])-1
            link_r[:, N, 1, 2] = 2*(quats[:, 2] * quats[:, 3] - quats[:, 0] * quats[:, 1])
            link_r[:, N, 2, 0] = 2*(quats[:, 1] * quats[:, 3] - quats[:, 0] * quats[:, 2])
            link_r[:, N, 2, 1] = 2*(quats[:, 2] * quats[:, 3] + quats[:, 0] * quats[:, 1])
            link_r[:, N, 2, 2] = 2*(quats[:, 0] * quats[:, 0] + quats[:, 3] * quats[:, 3])-1
            link_com[:, N, :] = rb_states[:, N, 0:3] + (link_r[:, N, :, :] @ self.rb_com[:, N, :].unsqueeze(2)).flatten(1,2) # link com position in inertial frame
            tmp_com += self.rb_mass[:, N].unsqueeze(1).repeat(1,3) * link_com[:, N, :]
            tmp_mass += self.rb_mass[:, N].unsqueeze(1).repeat(1,3)            
        self.com_position = tmp_com / tmp_mass  # robot COM computed in inertial frame   
              
        # get Projection matrix
        for key, N in self.body_names_dict.items():
            # c is the vector from robot-COM to link-COM
            c = link_com[:, N, :] - self.com_position # CoM frame has same orientation as inertial frame -> no rotation required
            Sc[:, 0, 1] = - c[:, 2]
            Sc[:, 0, 2] = c[:, 1]
            Sc[:, 1, 2] = -c[:, 0]
            Sc[:, 1, 0] = c[:, 2]
            Sc[:, 2, 0] = -c[:, 1]
            Sc[:, 2, 1] = c[:, 0]          
            tmp = gymtorch.torch.transpose(link_r[:, N, :, :], 1, 2) # i_R_G     
            transform[:, 6*N:6*N+3, 0:3] = tmp
            transform[:, 6*N:6*N+3, 3:6] = tmp @ Sc.transpose(1,2)
            transform[:, 6*N+3:6*N+6, 3:6] = tmp
            
            transform_from_inertial_to_com[:, 6*N:6*N+3, 6*N:6*N+3] = tmp # i_R_0                         
            transform_from_inertial_to_com[:, 6*N+3:6*N+6, 6*N+3:6*N+6] = tmp
                        
        CMM = transform.transpose(1,2) @ inertia_mat @ transform_from_inertial_to_com @ self.jacobian
        dofvel = gymtorch.torch.cat((rb_states[:, 0, 7:13], self.dof_vel), dim=1)
        self.centroidal_momentum = (CMM @ dofvel.unsqueeze(2)).reshape(self.num_envs, -1)          

    def _init_buffers(self):
        super()._init_buffers()
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.cfg.asset.name)
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)        
        self.jacobian = gymtorch.wrap_tensor(_jacobian).flatten(1,2) # originally shape of (num_envs, num_bodies, 6, num_dofs+6)
        # The jacobian maps joint velocities (num_dofs + 6) to spatial velocities of CoM frame of each link in inertial frame
        # https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/docs/Articulations.html#jacobian
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        self.rb_inertia = gymtorch.torch.zeros((self.num_envs, self.num_bodies, 3, 3), device=self.device) # [comX, comY, comZ], [Ix, Iy, Iz]
        self.rb_mass = gymtorch.torch.zeros((self.num_envs, self.num_bodies), device=self.device) # link mass
        self.rb_com = gymtorch.torch.zeros((self.num_envs, self.num_bodies, 3), device = self.device) # [comX, comY, comZ] in link's origin frame 
        self.com_position = gymtorch.torch.zeros((self.num_envs, 3), device=self.device) # robot-com position in inertial frame
        self.centroidal_momentum = gymtorch.torch.zeros((self.num_envs, 6), device=self.device)        
        
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
        # Update dynamics        

         
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed   
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
    
    def _create_envs(self):
        super()._create_envs()
        collision_mask = [ 6, 12] # shin 4, thigh 5, ankle 6, foot 7 must have collision
        for env in self.envs:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env, 0)
            for j in range(len(rigid_shape_props)):
                if j not in collision_mask:
                    rigid_shape_props[j].filter=1
            self.gym.set_actor_rigid_shape_properties(env, 0, rigid_shape_props)

        for name, num in self.body_names_dict.items():
            shape_id = self.body_to_shapes[num]
            print("body : ", name, ", shape index start : ", shape_id.start, ", shape index count : ", shape_id.count)
            for i in range(shape_id.start, shape_id.start + shape_id.count):
                print("shape ", i, " filter : ", self.gym.get_actor_rigid_shape_properties(self.envs[0], 0)[i].filter)
                print("shape ", i, " contact offset : ", self.gym.get_actor_rigid_shape_properties(self.envs[0], 0)[i].contact_offset)
                # I guess the best I can try is set the shin's bitmasks as 0    
    