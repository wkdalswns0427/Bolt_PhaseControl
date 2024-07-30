from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets

from torch import optim
import torch 
import numpy as np


class A2CAgent(a2c_common.ContinuousA2CBase):

    def __init__(self, base_name, params):
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        obs_shape = self.obs_shape
        build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        
        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound'
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape, 
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'horizon_length' : self.horizon_length,
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_length' : self.seq_length,
                'normalize_value' : self.normalize_value,
                'network' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer,
                'max_epochs' : self.max_epochs,
                'multi_gpu' : self.multi_gpu,
                'zero_rnn_on_done' : self.zero_rnn_on_done
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std

        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        self.algo_observer.after_init(self)

        self.symmetry_loss_weight = 3.0

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num
        
    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn, set_epoch=True):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']            

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model,value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]
            symmetry_loss = self.compute_symmetry_loss(self.model,batch_dict)

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef + symmetry_loss*self.symmetry_loss_weight
            
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of they year
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)      

        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)

    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result

    def reg_loss(self, mu):
        if self.bounds_loss_coef is not None:
            reg_loss = (mu*mu).sum(axis=-1)
        else:
            reg_loss = 0
        return reg_loss

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss
    
###################################################################################################
################################     mirror symmetry loss   #######################################
###################################################################################################
    def compute_symmetry_loss(self, policy, batch_dict):
        num_obvs = 42
        self.history_len = 1
        mirror_dict = {'HipPitch': (2,7), 
                        'KneePitch': (3,8), 
                        'AnklePitch': (4,9),
                        } # Joint pairs that need to be mirrored
        mirror_neg_dict = {'HipYaw': (0,5), 'HipRoll': (1,6), } 
        cartesian_angular_mirror = []
        cartesian_linear_mirror = [(6,9)]
        cartesian_command_mirror = [(9,12)]
        switch_mirror = []
        no_mirror = []
        mirror_weight = 4
        obs_batch = batch_dict['obs']
        actions_batch = batch_dict['prev_actions']
        minibatchsize = obs_batch.shape[0]
        num_acts = actions_batch.shape[1]
        cartesian_mirror_count = 0
        no_mirror_count = 0
        switch_mirror_count = 0
        self.mirror_obs = torch.eye(num_obvs).reshape(1, num_obvs, num_obvs).repeat(minibatchsize, 1, 1).to(device=self.device)
        self.mirror_act = torch.eye(num_acts).reshape(1, num_acts, num_acts).repeat(minibatchsize, 1, 1).to(device=self.device)

        for _, (i,j) in mirror_dict.items():
            self.mirror_act[:, i, i] = 0
            self.mirror_act[:, j, j] = 0
            self.mirror_act[:, i, j] = 1
            self.mirror_act[:, j, i] = 1
        for _, (i, j) in mirror_neg_dict.items():
            self.mirror_act[:, i, i] = 0
            self.mirror_act[:, j, j] = 0
            self.mirror_act[:, i, j] = -1
            self.mirror_act[:, j, i] = -1
                        # Cartesian space mirrors
        for (start, atend) in cartesian_angular_mirror:
                        # cartesian mirrors from range(start, atend)
            if (atend-start)%3==0:
                for i in range(int((atend-start)/3)):
                    self.mirror_obs[:, start + 3*i, start + 3*i] *= -1
                    self.mirror_obs[:, start+2 + 3*i, start+2 + 3*i] *= -1
                    cartesian_mirror_count += 3
            else:
                raise ValueError("SOMETHING WRONG IN CARTESIAN SPACE MIRRORS!!(angular)")
        for (start, atend) in cartesian_linear_mirror:
            if (atend-start)%3==0:
                for i in range(int((atend-start)/3)):
                    self.mirror_obs[:, start+1+ 3*i, start+1+ 3*i] *= -1
                    cartesian_mirror_count += 3
            else:
                raise ValueError("SOMETHING WRONG IN CARTESIAN SPACE MIRRORS!!(linear)")                        
                            
        for (start, atend) in cartesian_command_mirror:
            if (atend-start)%3==0:
                for i in range(int((atend-start)/3)):
                    self.mirror_obs[:, start+1+ 3*i, start+1+ 3*i] *= -1
                    self.mirror_obs[:, start+2+ 3*i, start+2+ 3*i] *= -1
                    cartesian_mirror_count += 3  
            else:
                raise ValueError("SOMETHING WRONG IN CARTESIAN SPACE MIRRORS!!(command)")
        for (start, atend) in switch_mirror:
            if (atend-start)%2==0:
                for i in range(int((atend-start)/2)):
                    self.mirror_obs[:, start+2*i, start+2*i] *= 0
                    self.mirror_obs[:, start+2*i+1, start+2*i+1] *= 0
                    self.mirror_obs[:, start+2*i+1, start+2*i] = 1
                    self.mirror_obs[:, start+2*i, start+2*i+1] = 1
                                
                    switch_mirror_count += 2
            else:
                raise ValueError("SOMETHING WRONG IN SWITCH MIRRORS!!")                    
        for (start, atend) in no_mirror:
            for _ in range(start, atend):
                no_mirror_count += 1
        # if ((num_obvs - cartesian_mirror_count - switch_mirror_count - no_mirror_count) % num_acts) != 0:
        #     raise ValueError("SOMETHING WRONG IN MIRROR TOTAL!!")
        for i in range(int((num_obvs - cartesian_mirror_count - switch_mirror_count - no_mirror_count) / num_acts)):
            self.mirror_obs[:, cartesian_mirror_count + i*num_acts:cartesian_mirror_count + (i+1)*num_acts, cartesian_mirror_count + i*num_acts:cartesian_mirror_count + (i+1)*num_acts] = self.mirror_act

        mirror_obs_batch = torch.zeros(obs_batch.shape, device=self.device)
        for k in range(self.history_len):
            mirror_obs_batch[:, k * num_obvs: (k+1) * num_obvs] = (self.mirror_obs @ (obs_batch[:, k * num_obvs: (k+1) * num_obvs].unsqueeze(2))).squeeze()
        mirrored_batch_dict = batch_dict.copy()
        mirrored_batch_dict['obs'] = mirror_obs_batch
        mirror_loss = torch.mean(torch.square(policy(mirrored_batch_dict)['prev_neglogp'] - policy(batch_dict)['prev_neglogp'])) 
    
        return mirror_loss


