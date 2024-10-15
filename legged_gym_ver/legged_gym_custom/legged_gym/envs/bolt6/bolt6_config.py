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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
class Bolt6Cfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096 # robot count 4096
        num_observations = 24
        '''
        self.projected_gravity:  torch.Size([4096, 3])
        self.commands[:, :3]:  torch.Size([4096, 3])
        (self.dof_pos - self.default_dof_pos):  torch.Size([4096, 6])
        self.dof_vel:  torch.Size([4096, 6])
        self.actions:  torch.Size([4096, 6])

        Observations should be stacked so that cartesian space vectors are in front, followed by joint space vectors.
        Values that do not need any mirroring (ex. terrain parameters) should be stacked in the back.
        
        3 + 3 + 6 + 6 + 6 = 24(num_observation)
        '''
        num_privileged_obs = 130 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        '''
        self.obs_buf: torch.Size([4096, 24])
        self.base_lin_vel: torch.Size([4096, 3])
        self.base_ang_vel: torch.Size([4096, 3])
        self.contacts:   torch.Size([4096, 2])
        self.ext_forces: torch.Size([4096, 3])
        self.ext_torques: torch.Size([4096, 3])
        self.friction_coeffs: torch.Size([4096, 1])
        self.dof_props['friction']: torch.Size([4096, 6])
        self.dof_props['damping']: torch.Size([4096, 6])
        self.measured_heights: torch.Size([4096, 36]) #
        
        24 + 3 + 3 + 2 + 3 + 3 + 1 + 6 + 6 + 81 = 130
        '''
        num_actions = 6 # robot actuation
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 10 # episode length in seconds

    class terrain( LeggedRobotCfg.terrain):
        mesh_type = mesh_type # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.001 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4] # 1mx1m rectangle (without center line)
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
        difficulty = [0.1, 0.2, 0.3]

    class commands( LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 10.
        num_commands = 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        
        class ranges( LeggedRobotCfg.commands.ranges ):
            # lin_vel_x = [-0.8, 0.8] # min max [m/s] seems like less than or equal to 0.2 it sends 0 command
            # lin_vel_y = [-0.8, 0.8]   # min max [m/s]
            # ang_vel_yaw = [-1., 1.]    # min max [rad/s]
            # heading = [-3.14, 3.14]
            lin_vel_x = [0.4, 0.4] # min max [m/s] seems like less than or equal to 0.2 it sends 0 command
            lin_vel_y = [-0., 0.]   # min max [m/s]
            ang_vel_yaw = [-0., 0.]    # min max [rad/s]
            heading = [-3.14, 3.14]
            

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.45] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_HAA': 0.,
            # 'hip_rotation_left': 0.,
            'FL_HFE': 0.2,
            # 'thigh_joint_left': -1.8,
            'FL_KFE': -0.4,
            'FL_ANKLE': 0.,

            'FR_HAA': 0.,
            # 'hip_rotation_right': 0.,
            'FR_HFE': 0.2,
            # 'thigh_joint_right': -1.8,
            'FR_KFE': -0.4,
            'FR_ANKLE': 0.
        }

    class control( LeggedRobotCfg.control ):
        control_type = 'T' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {   'FL_HAA': 0.1, 
                        # 'hip_rotation': 100.0,
                        'FL_HFE': 0.2,
                        'FL_KFE': 0.2,
                        
                        # 'thigh_joint': 200.,
                        'FR_HAA': 0.1,
                        'FR_HFE': 0.2,
                        'FR_KFE': 0.2,
                        
                        #'FL_ANKLE': 200.,
                        #'FR_ANKLE': 200.
                        # 'toe_joint': 40.
                        }  # [N*m/rad]
        damping = { 'FL_HAA': 0.02, 
                    'FL_HFE': 0.02,
                    'FL_KFE': 0.01,
                    'FR_HAA': 0.02, 
                    
                    'FR_HFE': 0.02,
                    
                    'FR_KFE': 0.01,
                    
                    #'FL_ANKLE': 6.,
                    #'FR_ANKLE': 6.
                   #'hip_rotation': 3.0,
                   #'hip_flexion': 6., 
                    # 'thigh_joint': 6., 
                    # 'ankle_joint': 6.,
                    # 'toe_joint': 1.
                    }   # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1.0 # 0.5 in pos control
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 2 #4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/bolt6/urdf/bolt6.urdf'
        name = "bolt6"
        foot_name = 'LOWER_LEG'
        penalize_contacts_on = []
        # penalize_contacts_on = ['bolt_lower_leg_right_side', 'bolt_body', 'bolt_hip_fe_left_side', 'bolt_hip_fe_right_side', ' bolt_lower_leg_left_side', 'bolt_shoulder_fe_left_side', 'bolt_shoulder_fe_right_side', 'bolt_trunk', 'bolt_upper_leg_left_side', 'bolt_upper_leg_right_side']
        terminate_after_contacts_on = ['base_link', 'FR_SHOULDER', 'FL_SHOULDER']
        
        
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        # fix_base_link = True
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 20.
        max_linear_velocity = 20.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True # Randomizes dof shape friction. Simulated ground friction is the mean of shape friction and terrain friction
        friction_range = [0.05, 3.0] #[0.2, 1.5]
        randomize_base_mass = False
        added_mass_range = [-.5, .5]
        push_robots = False
        push_interval_s = 3
        max_push_vel_xy = .5    
        ext_force_robots = False
        ext_force_randomize_interval_s = 5
        ext_force_direction_range = (0, 2*3.141592)
        ext_force_scale_range = (-10, 10)
        ext_force_interval_s = 2
        ext_force_duration_s = [0.3, 0.7]
        randomize_dof_friction = False
        dof_friction_interval_s = 5
        dof_friction = [0, 0.1] # https://forums.developer.nvidia.com/t/possible-bug-in-joint-friction-value-definition/208631
        dof_damping = [0, 0.003]
        
        
        

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 20.
        only_positive_rewards = False
        orientation_sigma = 0.25
        tracking_sigma = 0.5

        base_height_target = 0.5
        
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200.
            # traking
            tracking_lin_vel = 10.
            tracking_ang_vel = 10.

            # regulation in task space
            lin_vel_z = -0.
            ang_vel_xy = -0.0
            
            # regulation in joint space
            torques = 0. # -5.e-7
            dof_vel = 0 #-1.e-3
            dof_acc = 0 #-2.e-7 # -2.e-7
            action_rate = 0 #-0.001 # -0.000001

            # walking specific rewards
            feet_air_time = 0.
            collision = -0.
            feet_stumble = -0.0 
            stand_still = 0.0
            no_fly = 0.0
            feet_contact_forces = -5. #10?
            energy = -1e-1
            
            # joint limits
            torque_limits = -0.01
            dof_vel_limits = -0.
            dof_pos_limits = -10.
            
            # DRS
            orientation = 0.0 # Rui
            base_height = 0.0
            joint_regularization = 0.0

            # PBRS rewards
            ori_pb = 5.0
            baseHeight_pb = 2.0 #3.
            jointReg_pb =1.5
            action_rate_pb = 0.0

            stand_still_pb = 1.0
            no_fly_pb = 4.0
            feet_air_time_pb = 4. # 2.

    class normalization:
        class obs_scales:
            lin_vel = 1.0 # Rui
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            
            ext_forces = .1
            ext_torques = 1.
            friction_coeffs = 1.
            dof_friction = 10.
            dof_damping = 10.
        clip_observations = 100.
        clip_actions = 25.

    class noise:
        add_noise = True
        noise_level = 2.0 # scales other values
        class noise_scales:
            dof_pos = 0.005
            dof_vel = 0.01
            lin_vel = 0.1
            ang_vel = 0.05
            gravity = 0.05
            height_measurements = 0.02

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]
        # pos = [10, -1, 6]  # [m]
        # lookat = [-10., 0, 0.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 10.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class Bolt6CfgPPO( LeggedRobotCfgPPO ):
    seed = 1
    runner_class_name = 'OnPolicyRunnerHistoryEst'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims =[512, 256, 128]# 
        critic_hidden_dims = [512, 256, 128]#
        state_estimator_hiden_size = [256,128, 64]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        
        
        # Symmetric loss
        mirror = {'HipPitch': (1,4), 
                  'KneePitch': (2,5), 
                } # Joint pairs that need to be mirrored
        mirror_neg = {'HipRoll': (0,3), } # Joint pairs that need to be mirrored and signs must be changed
        mirror_weight = 0.5
        estimation_weight = 1.0
        # The following lists indicate the ranges in the observation vector indices, for which specific mirroring method should applied
        # For example, cartesian_angular_mirror = [(0,3), (6,12)] indicate that the cartesian angular mirror operation should be applied
        # to the 0th~2nd, and the 6th~8th, 9th~11th elements of the observation vector.
        cartesian_angular_mirror = []
        cartesian_linear_mirror = [(0,3)]
        cartesian_command_mirror = [(3,6)]
        # The following list indicate the ranges in the observation vector indices, for which switching places is necessary
        switch_mirror = []
        # The following list indicate the ranges in the observation vector indices, for which no mirroring is necessary.
        no_mirror = []
        
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO_sym_est'
        num_steps_per_env = 24 # per iteration
        max_iterations = 10000 # number of policy updates
        
        # Optional. Choose the length of state history for the algorithm to use.
        history_len = 10
        critic_history_len = 10

        # logging
        save_interval = 1000 # check for potential saves every this many iterations
        experiment_name = 'bolt6'
        run_name = 'bolt6'
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
        # symmetric loss
        