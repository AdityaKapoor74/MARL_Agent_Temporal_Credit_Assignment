import numpy as np
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from model import Policy, V_network, PopArt
from utils import RolloutBuffer

class PPOAgent:

	def __init__(
		self, 
		env, 
		dictionary,
		comet_ml,
		):

		# Environment Setup
		self.environment = dictionary["environment"]
		self.env = env
		self.env_name = dictionary["env"]
		self.num_agents = dictionary["num_agents"]
		self.num_actions = dictionary["num_actions"]

		# Training setup
		self.max_episodes = dictionary["max_episodes"]
		self.test_num = dictionary["test_num"]
		self.gif = dictionary["gif"]
		self.experiment_type = dictionary["experiment_type"]
		self.n_epochs = dictionary["n_epochs"]
		self.scheduler_need = dictionary["scheduler_need"]
		self.norm_rewards = dictionary["norm_rewards"]
		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"

		self.update_ppo_agent = dictionary["update_ppo_agent"]
		self.max_time_steps = dictionary["max_time_steps"]

		# Critic Setup
		self.use_recurrent_critic = dictionary["use_recurrent_critic"]
		self.temperature_v = dictionary["temperature_v"]
		self.rnn_num_layers_v = dictionary["rnn_num_layers_v"]
		self.rnn_hidden_v = dictionary["rnn_hidden_v"]
		self.v_comp_emb_shape = dictionary["v_comp_emb_shape"]
		self.ally_observation_shape = dictionary["ally_observation_shape"]
		self.num_enemies = dictionary["num_enemies"]
		self.enemy_observation_shape = dictionary["enemy_observation_shape"]


		self.v_value_lr = dictionary["v_value_lr"]
		self.v_weight_decay = dictionary["v_weight_decay"]
		self.critic_weight_entropy_pen = dictionary["critic_weight_entropy_pen"]
		self.critic_weight_entropy_pen_final = dictionary["critic_weight_entropy_pen_final"]
		self.critic_weight_entropy_pen_decay_rate = (dictionary["critic_weight_entropy_pen_final"] - dictionary["critic_weight_entropy_pen"]) / dictionary["critic_weight_entropy_pen_steps"]
		self.critic_score_regularizer = dictionary["critic_score_regularizer"]
		self.target_calc_style = dictionary["target_calc_style"]
		self.n_steps = dictionary["n_steps"]
		self.value_clip = dictionary["value_clip"]
		self.num_heads = dictionary["num_heads"]
		self.enable_hard_attention = dictionary["enable_hard_attention"]
		self.enable_grad_clip_critic_v = dictionary["enable_grad_clip_critic_v"]
		self.grad_clip_critic_v = dictionary["grad_clip_critic_v"]
		self.norm_returns_v = dictionary["norm_returns_v"]
		self.clamp_rewards = dictionary["clamp_rewards"]
		self.clamp_rewards_value_min = dictionary["clamp_rewards_value_min"]
		self.clamp_rewards_value_max = dictionary["clamp_rewards_value_max"]


		# Actor Setup
		self.use_recurrent_policy = dictionary["use_recurrent_policy"]
		self.data_chunk_length = dictionary["data_chunk_length"]
		self.rnn_num_layers_actor = dictionary["rnn_num_layers_actor"]
		self.rnn_hidden_actor = dictionary["rnn_hidden_actor"]
		self.local_observation_shape = dictionary["local_observation_shape"]
		self.policy_lr = dictionary["policy_lr"]
		self.policy_weight_decay = dictionary["policy_weight_decay"]
		self.gamma = dictionary["gamma"]
		self.entropy_pen = dictionary["entropy_pen"]
		self.entropy_pen_decay = (dictionary["entropy_pen"] - dictionary["entropy_pen_final"])/dictionary["entropy_pen_steps"]
		self.entropy_pen_final = dictionary["entropy_pen_final"]
		self.gae_lambda = dictionary["gae_lambda"]
		self.norm_adv = dictionary["norm_adv"]
		self.policy_clip = dictionary["policy_clip"]
		self.enable_grad_clip_actor = dictionary["enable_grad_clip_actor"]
		self.grad_clip_actor = dictionary["grad_clip_actor"]
		
		print("EXPERIMENT TYPE", self.experiment_type)

		# Critic Network
		if self.norm_returns_v:
			self.V_PopArt = PopArt(input_shape=1, num_agents=self.num_agents, device=self.device)
		else:
			self.V_PopArt = None

		self.critic_network_v = V_network(
			use_recurrent_critic=self.use_recurrent_critic,
			ally_obs_input_dim=self.ally_observation_shape, 
			enemy_obs_input_dim=self.enemy_observation_shape, 
			num_heads=self.num_heads, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies, 
			num_actions=self.num_actions, 
			rnn_num_layers=self.rnn_num_layers_v,
			comp_emb_shape=self.v_comp_emb_shape,
			device=self.device, 
			enable_hard_attention=self.enable_hard_attention, 
			attention_dropout_prob=dictionary["attention_dropout_prob_v"], 
			temperature=self.temperature_v,
			norm_returns=self.norm_returns_v,
			environment=self.environment,
			experiment_type=self.experiment_type,
			).to(self.device)
		self.target_critic_network_v = V_network(
			use_recurrent_critic=self.use_recurrent_critic,
			ally_obs_input_dim=self.ally_observation_shape, 
			enemy_obs_input_dim=self.enemy_observation_shape, 
			num_heads=self.num_heads, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies, 
			num_actions=self.num_actions, 
			rnn_num_layers=self.rnn_num_layers_v,
			comp_emb_shape=self.v_comp_emb_shape,
			device=self.device, 
			enable_hard_attention=self.enable_hard_attention, 
			attention_dropout_prob=dictionary["attention_dropout_prob_v"], 
			temperature=self.temperature_v,
			norm_returns=self.norm_returns_v,
			environment=self.environment,
			experiment_type=self.experiment_type,
			).to(self.device)
		# Copy network params
		self.target_critic_network_v.load_state_dict(self.critic_network_v.state_dict())
		# Disable updates for old network
		for param in self.target_critic_network_v.parameters():
			param.requires_grad_(False)
		
		
		# Policy Network
		self.policy_network = Policy(
			use_recurrent_policy=self.use_recurrent_policy,
			obs_input_dim=self.local_observation_shape, 
			num_agents=self.num_agents, 
			num_actions=self.num_actions, 
			rnn_num_layers=self.rnn_num_layers_actor,
			rnn_hidden_actor=self.rnn_hidden_actor,
			device=self.device
			).to(self.device)
		

		self.network_update_interval_v = dictionary["network_update_interval_v"]
		self.soft_update_v = dictionary["soft_update_v"]
		self.tau_v = dictionary["tau_v"]


		self.buffer = RolloutBuffer(
			environment=self.environment,
			experiment_type=self.experiment_type,
			num_episodes=self.update_ppo_agent, 
			max_time_steps=self.max_time_steps, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies,
			ally_state_shape=self.ally_observation_shape, 
			enemy_state_shape=self.enemy_observation_shape, 
			local_obs_shape=self.local_observation_shape, 
			rnn_num_layers_actor=self.rnn_num_layers_actor,
			actor_hidden_state=self.rnn_hidden_actor,
			rnn_num_layers_v=self.rnn_num_layers_v,
			v_hidden_state=self.rnn_hidden_v,
			num_actions=self.num_actions,
			data_chunk_length=self.data_chunk_length,
			norm_returns_v=self.norm_returns_v,
			clamp_rewards=self.clamp_rewards,
			clamp_rewards_value_min=self.clamp_rewards_value_min,
			clamp_rewards_value_max=self.clamp_rewards_value_max,
			norm_rewards=self.norm_rewards,
			target_calc_style=self.target_calc_style,
			gae_lambda=self.gae_lambda,
			n_steps=self.n_steps,
			gamma=self.gamma,
			)

		# Loading models
		if dictionary["load_models"]:
			# For CPU
			if torch.cuda.is_available() is False:
				self.critic_network_v.load_state_dict(torch.load(dictionary["model_path_v_value"], map_location=torch.device('cpu')))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"], map_location=torch.device('cpu')))
			# For GPU
			else:
				self.critic_network_v.load_state_dict(torch.load(dictionary["model_path_v_value"]))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"]))


		self.v_critic_optimizer = optim.AdamW(self.critic_network_v.parameters(), lr=self.v_value_lr, weight_decay=self.v_weight_decay, eps=1e-05)
		self.policy_optimizer = optim.AdamW(self.policy_network.parameters(),lr=self.policy_lr, weight_decay=self.policy_weight_decay, eps=1e-05)

		if self.scheduler_need:
			self.scheduler_policy = optim.lr_scheduler.MultiStepLR(self.policy_optimizer, milestones=[1000, 20000], gamma=0.1)
			self.scheduler_v_critic = optim.lr_scheduler.MultiStepLR(self.v_critic_optimizer, milestones=[1000, 20000], gamma=0.1)
			
		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml

	
	def get_lr(self, it, learning_rate):
		# 1) linear warmup for warmup_iters steps
		warmup_iters = 250
		lr_decay_iters = 20000
		min_lr = 5e-5
		if it < warmup_iters:
			learning_rate = 5e-4
			return learning_rate * it / warmup_iters
		# 2) if it > lr_decay_iters, return min learning rate
		if it > lr_decay_iters:
			return min_lr
		# 3) in between, use cosine decay down to min learning rate
		decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
		assert 0 <= decay_ratio <= 1
		coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
		return min_lr + coeff * (learning_rate - min_lr)

	
	def lr_decay(self, episode, initial_lr):
		"""Decreases the learning rate linearly"""
		lr = initial_lr - (initial_lr * (episode / float(self.max_episodes)))
		# for param_group in optimizer.param_groups:
		# 	param_group['lr'] = lr
		return lr

	
	def get_values(self, local_obs, state_allies, state_enemies, actions, rnn_hidden_state_v, indiv_dones, episode):
		with torch.no_grad():
			indiv_masks = [1-d for d in indiv_dones]
			indiv_masks = torch.FloatTensor(indiv_masks).unsqueeze(0).unsqueeze(0)
			state_allies = torch.FloatTensor(state_allies).unsqueeze(0).unsqueeze(0)
			state_enemies = torch.FloatTensor(state_enemies).unsqueeze(0).unsqueeze(0)
			local_obs = torch.FloatTensor(local_obs).unsqueeze(0).unsqueeze(0)
			actions = torch.FloatTensor(actions).unsqueeze(0).unsqueeze(0)
			rnn_hidden_state_v = torch.FloatTensor(rnn_hidden_state_v)
			
			Value, rnn_hidden_state_v = self.target_critic_network_v(local_obs.to(self.device), state_allies.to(self.device), state_enemies.to(self.device), actions.to(self.device), rnn_hidden_state_v.to(self.device), indiv_masks.to(self.device))
				
			return Value.squeeze(0).cpu().numpy(), rnn_hidden_state_v.cpu().numpy()
	
	
	def get_action(self, state_policy, last_actions, mask_actions, hidden_state, greedy=False):
		with torch.no_grad():
			state_policy = torch.FloatTensor(state_policy).unsqueeze(0).unsqueeze(1).to(self.device)
			last_actions = torch.LongTensor(last_actions).unsqueeze(0).unsqueeze(1).to(self.device)
			mask_actions = torch.BoolTensor(mask_actions).unsqueeze(0).unsqueeze(1).to(self.device)
			hidden_state = torch.FloatTensor(hidden_state).to(self.device)

			dists, hidden_state = self.policy_network(state_policy, last_actions, hidden_state, mask_actions)

			if greedy:
				actions = [dist.argmax().detach().cpu().item() for dist in dists.squeeze(0).squeeze(0)]
				action_logprob = None
			else:
				actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists.squeeze(0).squeeze(0)]

				probs = Categorical(dists)
				action_logprob = probs.log_prob(torch.FloatTensor(actions).to(self.device)).cpu().numpy()

			return actions, action_logprob, hidden_state.cpu().numpy()



	def plot(self, masks, episode):
		
		self.comet_ml.log_metric('Policy_Loss',self.plotting_dict["policy_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_Policy',self.plotting_dict["grad_norm_policy"],episode)
		self.comet_ml.log_metric('Entropy',self.plotting_dict["entropy"],episode)

		self.comet_ml.log_metric('V_Value_Loss',self.plotting_dict["v_value_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_V_Value',self.plotting_dict["grad_norm_value_v"],episode)


	def update_parameters(self):
		if self.entropy_pen - self.entropy_pen_decay > self.entropy_pen_final:
			self.entropy_pen -= self.entropy_pen_decay


	def update(self, episode):
		
		v_value_loss_batch = 0
		policy_loss_batch = 0
		entropy_batch = 0
		grad_norm_value_v_batch = 0
		grad_norm_policy_batch = 0

		self.buffer.calculate_targets(episode, self.V_PopArt)

		# Optimize policy for n epochs
		for pp_epoch in range(self.n_epochs):

			# SAMPLE DATA FROM BUFFER
			ally_states, enemy_states, hidden_state_v, local_obs, hidden_state_actor, logprobs_old, \
			last_actions, actions, action_masks, agent_masks, team_masks, values_old, target_values, advantage  = self.buffer.sample_recurrent_policy()

			
			if self.norm_adv:
				shape = advantage.shape

				advantage_copy = copy.deepcopy(advantage)
				advantage_copy[agent_masks.view(*shape) == 0.0] = float('nan')
				advantage_mean = torch.nanmean(advantage_copy)
				advantage_std = torch.from_numpy(np.array(np.nanstd(advantage_copy.cpu().numpy()))).float()

				advantage = ((advantage - advantage_mean) / (advantage_std + 1e-5))*agent_masks.view(*shape)
			

			values_old *= agent_masks

			target_shape = values_old.shape

			values, h_v = self.critic_network_v(
												local_obs.to(self.device),
												ally_states.to(self.device),
												enemy_states.to(self.device),
												actions.to(self.device),
												hidden_state_v.to(self.device),
												agent_masks.to(self.device),
												)
			
			values = values.reshape(*target_shape)

			values *= agent_masks.to(self.device)
			target_values *= agent_masks

			if self.norm_returns_v:
				targets_shape = target_values.shape
				target_values = (self.V_PopArt(target_values.view(-1), agent_masks.view(-1), train=True).view(targets_shape) * agent_masks.view(targets_shape)).cpu()

			critic_v_loss_1 = F.huber_loss(values, target_values.to(self.device), reduction="sum", delta=10.0) / agent_masks.sum()
			critic_v_loss_2 = F.huber_loss(torch.clamp(values, values_old.to(self.device)-self.value_clip, values_old.to(self.device)+self.value_clip), target_values.to(self.device), reduction="sum", delta=10.0) / agent_masks.sum()

			
			dists, _ = self.policy_network(
					local_obs.to(self.device),
					last_actions.to(self.device),
					hidden_state_actor.to(self.device),
					action_masks.to(self.device),
					)

			probs = Categorical(dists)
			logprobs = probs.log_prob(actions.to(self.device))
		
				
			critic_v_loss = torch.max(critic_v_loss_1, critic_v_loss_2)
			print("Critic V Loss", critic_v_loss.item())
			
			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp((logprobs - logprobs_old.to(self.device)))
			
			# Finding Surrogate Loss
			surr1 = ratios * advantage.to(self.device) * agent_masks.to(self.device)
			surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage.to(self.device) * agent_masks.to(self.device)

			# final loss of clipped objective PPO
			entropy = -torch.sum(torch.sum(dists*agent_masks.unsqueeze(-1).to(self.device) * torch.log(torch.clamp(dists*agent_masks.unsqueeze(-1).to(self.device), 1e-10,1.0)), dim=-1))/ agent_masks.sum() #(masks.sum()*self.num_agents)
			policy_loss_ = (-torch.min(surr1, surr2).sum())/agent_masks.sum()
			policy_loss = policy_loss_ - self.entropy_pen*entropy

			print("Policy Loss", policy_loss_.item(), "Entropy", (-self.entropy_pen*entropy.item()))
			
			self.v_critic_optimizer.zero_grad()
			critic_v_loss.backward()
			if self.enable_grad_clip_critic_v:
				grad_norm_value_v = torch.nn.utils.clip_grad_norm_(self.critic_network_v.parameters(), self.grad_clip_critic_v)
			else:
				total_norm = 0
				for p in self.critic_network_v.parameters():
					if p.grad is None:
						continue
					param_norm = p.grad.detach().data.norm(2)
					total_norm += param_norm.item() ** 2
				grad_norm_value_v = torch.tensor([total_norm ** 0.5])
			self.v_critic_optimizer.step()

			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			if self.enable_grad_clip_actor:
				grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.grad_clip_actor)
			else:
				total_norm = 0
				for p in self.policy_network.parameters():
					if p.grad is None:
						continue
					param_norm = p.grad.detach().data.norm(2)
					total_norm += param_norm.item() ** 2
				grad_norm_policy = torch.tensor([total_norm ** 0.5])
			self.policy_optimizer.step()

			
			policy_loss_batch += policy_loss.item()
			entropy_batch += entropy.item()
			grad_norm_policy_batch += grad_norm_policy.item()
			
			

		# Copy new weights into old critic	
		if self.soft_update_v:
			for target_param, param in zip(self.target_critic_network_v.parameters(), self.critic_network_v.parameters()):
				target_param.data.copy_(target_param.data * (1.0 - self.tau_v) + param.data * self.tau_v)
		elif episode % self.network_update_interval_v == 0:
			self.target_critic_network_v.load_state_dict(self.critic_network_v.state_dict())

		# clear buffer
		self.buffer.clear()
		

		policy_loss_batch /= self.n_epochs
		entropy_batch /= self.n_epochs
		grad_norm_policy_batch /= self.n_epochs
		v_value_loss_batch /= self.n_epochs
		grad_norm_value_v_batch /= self.n_epochs
			

		self.plotting_dict = {
		"v_value_loss": v_value_loss_batch,
		"policy_loss": policy_loss_batch,
		"entropy": entropy_batch,
		"grad_norm_policy": grad_norm_policy_batch,
		"grad_norm_value_v": grad_norm_value_v_batch,
		}
		
		if self.comet_ml is not None:
			self.plot(agent_masks, episode)

		del v_value_loss_batch, policy_loss_batch, entropy_batch, grad_norm_value_v_batch, grad_norm_policy_batch
		torch.cuda.empty_cache()