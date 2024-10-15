import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.algorithms import PPO

# This algorithm includes the mirror loss
# https://arxiv.org/pdf/1801.08093.pdf

class PPO_sym(PPO):
    def __init__(self, actor_critic, mirror, mirror_neg = {}, cartesian_angular_mirror=[], cartesian_linear_mirror=[], cartesian_command_mirror=[], switch_mirror = [], no_mirror = [], mirror_weight = 4, num_learning_epochs=1, num_mini_batches=1, clip_param=0.2, gamma=0.998, lam=0.95, value_loss_coef=1, entropy_coef=0, learning_rate=0.001, max_grad_norm=1, use_clipped_value_loss=True, schedule="fixed", desired_kl=0.01, device='cpu'):
        super().__init__(actor_critic, num_learning_epochs, num_mini_batches, clip_param, gamma, lam, value_loss_coef, entropy_coef, learning_rate, max_grad_norm, use_clipped_value_loss, schedule, desired_kl, device)
        self.mirror_dict = mirror
        self.mirror_neg_dict = mirror_neg
        self.cartesian_angular_mirror = cartesian_angular_mirror
        self.cartesian_linear_mirror = cartesian_linear_mirror
        self.cartesian_command_mirror = cartesian_command_mirror
        self.switch_mirror = switch_mirror
        self.no_mirror = no_mirror
        self.mirror_weight = mirror_weight
        self.mirror_init = True
        self.history_len = 1
        print("Sym version of PPO loaded")
        print("Mirror weight: ", mirror_weight)

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_mirror_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
            # The shape of an obs batch is : (minibatchsize, obs_shape)

                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate
                
                # Mirror loss
                # use mirror dict as mirror
                num_obvs = int(obs_batch.shape[1] / self.history_len) # length of each observation : 39
                if self.mirror_init:
                    print("MIRROR TEST")
                    if obs_batch.shape[1] % self.history_len:
                        raise ValueError("Obs batch shape does not match history length.")
                    minibatchsize = obs_batch.shape[0]
                    num_acts = actions_batch.shape[1] 
                    cartesian_mirror_count = 0
                    no_mirror_count = 0
                    switch_mirror_count = 0
                    self.mirror_obs = torch.eye(num_obvs).reshape(1, num_obvs, num_obvs).repeat(minibatchsize, 1, 1).to(device=self.device)
                    self.mirror_act = torch.eye(num_acts).reshape(1, num_acts, num_acts).repeat(minibatchsize, 1, 1).to(device=self.device)
                    # Joint space mirrors

                    for _, (i,j) in self.mirror_dict.items():
                        self.mirror_act[:, i, i] = 0
                        self.mirror_act[:, j, j] = 0
                        self.mirror_act[:, i, j] = 1
                        self.mirror_act[:, j, i] = 1
                    for _, (i, j) in self.mirror_neg_dict.items():
                        self.mirror_act[:, i, i] = 0
                        self.mirror_act[:, j, j] = 0
                        self.mirror_act[:, i, j] = -1
                        self.mirror_act[:, j, i] = -1
                        # Cartesian space mirrors
                    for (start, atend) in self.cartesian_angular_mirror:
                        # cartesian mirrors from range(start, atend)
                        if (atend-start)%3==0:
                            for i in range(int((atend-start)/3)):
                                self.mirror_obs[:, start + 3*i, start + 3*i] *= -1
                                self.mirror_obs[:, start+2 + 3*i, start+2 + 3*i] *= -1
                                cartesian_mirror_count += 3
                        else:
                            raise ValueError("SOMETHING WRONG IN CARTESIAN SPACE MIRRORS!!(angular)")
                    for (start, atend) in self.cartesian_linear_mirror:
                        if (atend-start)%3==0:
                            for i in range(int((atend-start)/3)):
                                self.mirror_obs[:, start+1+ 3*i, start+1+ 3*i] *= -1
                                cartesian_mirror_count += 3
                        else:
                            raise ValueError("SOMETHING WRONG IN CARTESIAN SPACE MIRRORS!!(linear)")                        
                            
                    for (start, atend) in self.cartesian_command_mirror:
                        if (atend-start)%3==0:
                            for i in range(int((atend-start)/3)):
                                self.mirror_obs[:, start+1+ 3*i, start+1+ 3*i] *= -1
                                self.mirror_obs[:, start+2+ 3*i, start+2+ 3*i] *= -1
                                cartesian_mirror_count += 3  
                        else:
                            raise ValueError("SOMETHING WRONG IN CARTESIAN SPACE MIRRORS!!(command)")
                    for (start, atend) in self.switch_mirror:
                        if (atend-start)%2==0:
                            for i in range(int((atend-start)/2)):
                                self.mirror_obs[:, start+2*i, start+2*i] *= 0
                                self.mirror_obs[:, start+2*i+1, start+2*i+1] *= 0
                                self.mirror_obs[:, start+2*i+1, start+2*i] = 1
                                self.mirror_obs[:, start+2*i, start+2*i+1] = 1
                                
                                switch_mirror_count += 2
                        else:
                            raise ValueError("SOMETHING WRONG IN SWITCH MIRRORS!!")                    
                    for (start, atend) in self.no_mirror:
                        for _ in range(start, atend):
                            no_mirror_count += 1
                    # Joint space mirroring
                    if ((num_obvs - cartesian_mirror_count - switch_mirror_count - no_mirror_count) % num_acts) != 0:
                        raise ValueError("SOMETHING WRONG IN MIRROR TOTAL!!")
                    for i in range(int((num_obvs - cartesian_mirror_count - switch_mirror_count - no_mirror_count) / num_acts)):
                        self.mirror_obs[:, cartesian_mirror_count + i*num_acts:cartesian_mirror_count + (i+1)*num_acts, cartesian_mirror_count + i*num_acts:cartesian_mirror_count + (i+1)*num_acts] = self.mirror_act

                    print("------ABOUT MIRROR------")
                    print("Total number of elements of cartesian space mirroring : ", cartesian_mirror_count)
                    print("Total number of elements of no mirroring : ", no_mirror_count)
                    print("Total number of elements of joint space mirroring : ", num_obvs - cartesian_mirror_count - no_mirror_count)                    
                    self.mirror_init = False
                
                
                mirror_obs_batch = torch.zeros(obs_batch.shape, device=self.device)
                for k in range(self.history_len):
                    mirror_obs_batch[:, k * num_obvs: (k+1) * num_obvs] = (self.mirror_obs @ (obs_batch[:, k * num_obvs: (k+1) * num_obvs].unsqueeze(2))).squeeze()

                mirror_loss = torch.mean(torch.square(self.actor_critic.actor(obs_batch) - (self.mirror_act @ self.actor_critic.actor(mirror_obs_batch).unsqueeze(2)).squeeze())) 
            
                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = self.mirror_weight * mirror_loss +surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_mirror_loss += mirror_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_mirror_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_mirror_loss