import numpy as np
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from model import Policy, Q_network
from utils import RolloutBuffer, RewardRolloutBuffer

class PPOAgent:

	def __init__(
		self, 
		env, 
		dictionary,
		comet_ml,
		):

		# Environment Setup
		self.env = env
		self.env_name = dictionary["env"]
		self.num_agents = self.env.n_agents
		self.num_enemies = self.env.n_enemies
		self.num_actions = self.env.action_space[0].n

		# Training setup
		self.max_episodes = dictionary["max_episodes"]
		self.test_num = dictionary["test_num"]
		self.gif = dictionary["gif"]
		self.experiment_type = dictionary["experiment_type"]
		self.n_epochs = dictionary["n_epochs"]
		self.scheduler_need = dictionary["scheduler_need"]
		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"

		self.update_ppo_agent = dictionary["update_ppo_agent"]
		self.max_time_steps = dictionary["max_time_steps"]


		# Reward Model Setup
		self.use_reward_model = dictionary["use_reward_model"]
		self.num_episodes_capacity = dictionary["num_episodes_capacity"]
		self.batch_size = dictionary["batch_size"]
		self.reward_lr = dictionary["reward_lr"]
		self.variance_loss_coeff = dictionary["variance_loss_coeff"]
		self.enable_reward_grad_clip = dictionary["enable_reward_grad_clip"]
		self.reward_grad_clip_value = dictionary["reward_grad_clip_value"]
		self.reward_n_heads = dictionary["reward_n_heads"]
		self.norm_rewards = dictionary["norm_rewards"]

		# Critic Setup
		self.temperature_q = dictionary["temperature_q"]
		self.rnn_num_layers_q = dictionary["rnn_num_layers_q"]
		self.rnn_hidden_q = dictionary["rnn_hidden_q"]
		self.critic_ally_observation = dictionary["ally_observation"]
		self.critic_enemy_observation = dictionary["enemy_observation"]
		self.q_value_lr = dictionary["q_value_lr"]
		self.q_weight_decay = dictionary["q_weight_decay"]
		self.critic_weight_entropy_pen = dictionary["critic_weight_entropy_pen"]
		self.critic_weight_entropy_pen_final = dictionary["critic_weight_entropy_pen_final"]
		self.critic_weight_entropy_pen_decay_rate = (dictionary["critic_weight_entropy_pen_final"] - dictionary["critic_weight_entropy_pen"]) / dictionary["critic_weight_entropy_pen_steps"]
		self.critic_score_regularizer = dictionary["critic_score_regularizer"]
		self.target_calc_style = dictionary["target_calc_style"]
		self.td_lambda = dictionary["td_lambda"] # TD lambda
		self.n_steps = dictionary["n_steps"]
		self.value_clip = dictionary["value_clip"]
		self.num_heads = dictionary["num_heads"]
		self.enable_grad_clip_critic_q = dictionary["enable_grad_clip_critic_q"]
		self.grad_clip_critic_q = dictionary["grad_clip_critic_q"]
		self.norm_returns_q = dictionary["norm_returns_q"]
		self.clamp_rewards = dictionary["clamp_rewards"]
		self.clamp_rewards_value_min = dictionary["clamp_rewards_value_min"]
		self.clamp_rewards_value_max = dictionary["clamp_rewards_value_max"]


		# Actor Setup
		self.data_chunk_length = dictionary["data_chunk_length"]
		self.rnn_num_layers_actor = dictionary["rnn_num_layers_actor"]
		self.rnn_hidden_actor = dictionary["rnn_hidden_actor"]
		self.actor_observation_shape = dictionary["local_observation"]
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
		
		# Q-V Network
		self.critic_network_q = Q_network(
			ally_obs_input_dim=self.critic_ally_observation, 
			enemy_obs_input_dim=self.critic_enemy_observation, 
			num_heads=self.num_heads, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies, 
			num_actions=self.num_actions, 
			rnn_num_layers=self.rnn_num_layers_q,
			device=self.device, 
			attention_dropout_prob=dictionary["attention_dropout_prob_q"], 
			temperature=self.temperature_q
			).to(self.device)
		self.target_critic_network_q = Q_network(
			ally_obs_input_dim=self.critic_ally_observation, 
			enemy_obs_input_dim=self.critic_enemy_observation, 
			num_heads=self.num_heads, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies, 
			num_actions=self.num_actions, 
			rnn_num_layers=self.rnn_num_layers_q,
			device=self.device, 
			attention_dropout_prob=dictionary["attention_dropout_prob_q"], 
			temperature=self.temperature_q
			).to(self.device)

		# Policy Network
		self.policy_network = Policy(
			obs_input_dim=self.actor_observation_shape, 
			num_agents=self.num_agents, 
			num_actions=self.num_actions, 
			rnn_num_layers=self.rnn_num_layers_actor,
			device=self.device
			).to(self.device)
		

		self.network_update_interval_q = dictionary["network_update_interval_q"]
		self.soft_update_q = dictionary["soft_update_q"]
		self.tau_q = dictionary["tau_q"]


		self.buffer = RolloutBuffer(
			num_episodes=self.update_ppo_agent, 
			max_time_steps=self.max_time_steps, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies,
			obs_shape_critic_ally=self.critic_ally_observation, 
			obs_shape_critic_enemy=self.critic_enemy_observation, 
			obs_shape_actor=self.actor_observation_shape, 
			rnn_num_layers_actor=self.rnn_num_layers_actor,
			actor_hidden_state=self.rnn_hidden_actor,
			rnn_num_layers_q=self.rnn_num_layers_q,
			q_hidden_state=self.rnn_hidden_q,
			num_actions=self.num_actions,
			data_chunk_length=self.data_chunk_length,
			norm_returns_q=self.norm_returns_q,
			clamp_rewards=self.clamp_rewards,
			clamp_rewards_value_min=self.clamp_rewards_value_min,
			clamp_rewards_value_max=self.clamp_rewards_value_max,
			target_calc_style=self.target_calc_style,
			td_lambda=self.td_lambda,
			gae_lambda=self.gae_lambda,
			n_steps=self.n_steps,
			gamma=self.gamma,
			Q_PopArt=self.critic_network_q.q_value_layer[-1],
			)

		if self.use_reward_model:

			self.reward_buffer = RewardRolloutBuffer(
				num_new_policy_episodes=self.update_ppo_agent,
				num_episodes_capacity=self.num_episodes_capacity, 
				max_time_steps=self.max_time_steps, 
				num_agents=self.num_agents, 
				num_enemies=self.num_enemies,
				obs_shape=self.critic_ally_observation+(self.critic_enemy_observation)*self.num_enemies,
				num_actions=self.num_actions, 
				batch_size=self.batch_size,
				)

			if self.experiment_type == "AREL":
				from AREL import AREL
				self.reward_model = AREL.Time_Agent_Transformer(
					emb=self.critic_ally_observation+(self.critic_enemy_observation)*self.num_enemies+self.num_actions, 
					heads=dictionary["reward_n_heads"], 
					depth=dictionary["reward_depth"], 
					seq_length=self.max_time_steps, 
					n_agents=self.num_agents, 
					agent=dictionary["reward_agent_attn"], 
					dropout=dictionary["reward_dropout"], 
					wide=dictionary["reward_attn_net_wide"], 
					comp=dictionary["reward_comp"], 
					norm_rewards=dictionary["norm_rewards"],
					device=self.device,
					).to(self.device)

				if self.norm_rewards:
					self.reward_normalizer = self.reward_model.toreward

			elif self.experiment_type == "ATRR":
				from ATRR import ATRR
				self.reward_model = ATRR.Time_Agent_Transformer(
					emb=self.actor_observation_shape+self.num_actions, 
					heads=dictionary["reward_n_heads"], 
					depth=dictionary["reward_depth"], 
					seq_length=self.max_time_steps, 
					n_agents=self.num_agents, 
					agent=dictionary["reward_agent_attn"], 
					dropout=dictionary["reward_dropout"], 
					wide=dictionary["reward_attn_net_wide"], 
					comp=dictionary["reward_comp"], 
					device=self.device,
					).to(self.device)

			self.reward_optimizer = optim.AdamW(self.reward_model.parameters(), lr=self.reward_lr, weight_decay=dictionary["reward_weight_decay"], eps=1e-5)
			
			if self.scheduler_need:
				self.scheduler_reward = optim.lr_scheduler.MultiStepLR(self.reward_optimizer, milestones=[1000, 20000], gamma=0.1)


		self.q_critic_optimizer = optim.AdamW(self.critic_network_q.parameters(), lr=self.q_value_lr, weight_decay=self.q_weight_decay, eps=1e-05)
		self.policy_optimizer = optim.AdamW(self.policy_network.parameters(),lr=self.policy_lr, weight_decay=self.policy_weight_decay, eps=1e-05)

		if self.scheduler_need:
			self.scheduler_policy = optim.lr_scheduler.MultiStepLR(self.policy_optimizer, milestones=[1000, 20000], gamma=0.1)
			self.scheduler_q_critic = optim.lr_scheduler.MultiStepLR(self.q_critic_optimizer, milestones=[1000, 20000], gamma=0.1)

		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml


		# Loading models
		if dictionary["load_models"]:
			# For CPU
			if torch.cuda.is_available() is False:
				self.critic_network_q.load_state_dict(torch.load(dictionary["model_path_value"], map_location=torch.device('cpu')))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"], map_location=torch.device('cpu')))
				self.q_critic_optimizer.load_state_dict(torch.load(dictionary["model_path_q_critic_optimizer"], map_location=torch.device('cpu')))
				self.policy_optimizer.load_state_dict(torch.load(dictionary["model_path_policy_optimizer"], map_location=torch.device('cpu')))
			# For GPU
			else:
				self.critic_network_q.load_state_dict(torch.load(dictionary["model_path_value"]))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"]))
				self.q_critic_optimizer.load_state_dict(torch.load(dictionary["model_path_q_critic_optimizer"]))
				self.policy_optimizer.load_state_dict(torch.load(dictionary["model_path_policy_optimizer"]))

		# Copy network params
		self.target_critic_network_q.load_state_dict(self.critic_network_q.state_dict())
		# Disable updates for old network
		for param in self.target_critic_network_q.parameters():
			param.requires_grad_(False)

	
	def get_q_values(self, state_allies, state_enemies, one_hot_actions, rnn_hidden_state_q, indiv_dones):
		with torch.no_grad():
			indiv_masks = [1-d for d in indiv_dones]
			indiv_masks = torch.FloatTensor(indiv_masks).unsqueeze(0).unsqueeze(0)
			state_allies = torch.FloatTensor(state_allies).unsqueeze(0).unsqueeze(0)
			state_enemies = torch.FloatTensor(state_enemies).unsqueeze(0).unsqueeze(0)
			one_hot_actions = torch.FloatTensor(one_hot_actions).unsqueeze(0).unsqueeze(0)
			rnn_hidden_state_q = torch.FloatTensor(rnn_hidden_state_q)
			Q_value, weights_prd, _, rnn_hidden_state_q = self.target_critic_network_q(state_allies.to(self.device), state_enemies.to(self.device), one_hot_actions.to(self.device), rnn_hidden_state_q.to(self.device), indiv_masks.to(self.device))

			return Q_value.squeeze(0).cpu().numpy(), rnn_hidden_state_q.cpu().numpy()

	
	def get_action(self, state_policy, last_one_hot_actions, mask_actions, hidden_state, greedy=False):
		with torch.no_grad():
			state_policy = torch.FloatTensor(state_policy).unsqueeze(0).unsqueeze(1).to(self.device)
			last_one_hot_actions = torch.FloatTensor(last_one_hot_actions).unsqueeze(0).unsqueeze(1).to(self.device)
			mask_actions = torch.BoolTensor(mask_actions).unsqueeze(0).unsqueeze(1).to(self.device)
			hidden_state = torch.FloatTensor(hidden_state).to(self.device)
			dists, hidden_state = self.policy_network(state_policy, last_one_hot_actions, hidden_state, mask_actions)

			if greedy:
				actions = [dist.argmax().detach().cpu().item() for dist in dists.squeeze(0).squeeze(0)]
				action_logprob = None
			else:
				actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists.squeeze(0).squeeze(0)]

				probs = Categorical(dists)
				action_logprob = probs.log_prob(torch.FloatTensor(actions).to(self.device)).cpu().numpy()

			return actions, action_logprob, hidden_state.cpu().numpy()


	def evaluate_reward_model(self):
		with torch.no_grad():
			state_allies, state_enemies, one_hot_actions, dones = self.buffer.states_critic_allies[self.buffer.episode_num], self.buffer.states_critic_enemies[self.buffer.episode_num], self.buffer.one_hot_actions[self.buffer.episode_num], self.buffer.dones[self.buffer.episode_num]
			state_allies = torch.from_numpy(state_allies).float()
			state_enemies = torch.from_numpy(state_enemies).float().unsqueeze(1).repeat(1, self.num_agents, 1, 1).reshape(state_enemies.shape[0], self.num_agents, -1)
			states = torch.cat([state_allies, state_enemies], dim=-1).unsqueeze(0)
			one_hot_actions = torch.from_numpy(one_hot_actions).float().unsqueeze(0)
			state_actions = torch.cat([states, one_hot_actions], dim=-1)
			masks = 1 - torch.from_numpy(dones[:-1, :]).float()
			team_masks = (masks.sum(dim=-1)[:, ] > 0).float()

			if self.experiment_type == "AREL":
				reward_episode_wise, reward_time_wise = self.reward_model(state_actions.permute(0,2,1,3).to(self.device))
				reward_episode_wise = (reward_time_wise.squeeze(0)*team_masks.to(self.device)).sum()
				if self.norm_rewards:
					shape = reward_episode_wise.shape
					reward_episode_wise = self.reward_normalizer.denormalize(reward_episode_wise.view(-1)).view(shape)*team_masks.to(self.device)
				return reward_episode_wise
			elif self.experiment_type == "ATRR":
				reward_episode_wise, _, _ = self.reward_model(state_actions.permute(0,2,1,3).to(self.device))
				return reward_episode_wise

	def update_reward_model(self, fine_tune=False, episode=None):
		# states, episodic_rewards, one_hot_actions, dones = torch.from_numpy(self.reward_buffer.states).float(), torch.from_numpy(self.reward_buffer.episodic_rewards).float(), torch.from_numpy(self.reward_buffer.one_hot_actions).float(), torch.from_numpy(self.reward_buffer.dones).float()
		# masks = 1 - dones
		if fine_tune:
			states, episodic_rewards, one_hot_actions, masks = self.reward_buffer.sample_new_data()
		else:
			states, episodic_rewards, one_hot_actions, masks = self.reward_buffer.sample()

		team_masks = (masks.sum(dim=-1)[:, ] > 0).float()

		if self.norm_rewards:
			shape = episodic_rewards.shape
			self.reward_normalizer.update(episodic_rewards.view(-1))
			
			episodic_rewards = self.reward_normalizer.normalize(episodic_rewards.view(-1)).view(shape)

		"""
		print(states.shape)
		>>> torch.Size([30, 100, 5, 60])  -> 55 + 5 = 60
		
		print(episodic_rewards.shape)
		>>>	torch.Size([30])
		
		print(dones.shape)
		>>>	torch.Size([30, 100, 5])

		print(self.reward_buffer.episode_length)
		>>> [17. 21. 21. 21. 21. 57. 24. 16. 22. 21. 25. 61. 14. 36. 22. 34. 18. 15.
			21. 15. 20. 21. 41. 63. 30. 17. 39. 18. 53. 22.]

		"""
		# episodic_rewards = rewards.sum(dim=1)


		state_actions = torch.cat([states, one_hot_actions], dim=-1)
		if self.experiment_type == "AREL":
			reward_episode_wise, reward_time_wise = self.reward_model(state_actions.permute(0, 2, 1, 3).to(self.device))

			shape = reward_time_wise.shape
			reward_copy = copy.deepcopy(reward_time_wise.detach())
			reward_copy[team_masks.view(*shape) == 0.0] = 0.0 #float('nan')
			reward_mean = (reward_copy.sum(dim=-1)/team_masks.to(self.device).sum(dim=-1)).unsqueeze(-1) #torch.nanmean(reward_copy, dim=-1).unsqueeze(-1)
			reward_var = (reward_time_wise - reward_mean)**2
			reward_var[team_masks.view(*shape) == 0.0] = 0.0
			reward_var = reward_var.sum() / team_masks.sum()

			# loss = (((reward_time_wise*team_masks.view(*shape).to(self.device)).sum(dim=-1) - episodic_rewards.to(self.device))**2).sum()/team_masks.sum() + self.variance_loss_coeff*reward_var
			loss = F.huber_loss((reward_time_wise*team_masks.view(*shape).to(self.device)).sum(dim=-1), episodic_rewards.to(self.device), reduction='none').sum()/team_masks.sum() + self.variance_loss_coeff*reward_var
			print("LOSS")
			print(loss)
			print((F.huber_loss((reward_time_wise*team_masks.view(*shape).to(self.device)).sum(dim=-1), episodic_rewards.to(self.device), reduction='none').sum()/team_masks.sum()).shape, (self.variance_loss_coeff*reward_var).shape)
		elif self.experiment_type == "ATRR":
			reward_episode_wise, temporal_weights, agent_weights = self.reward_model(state_actions.permute(0, 2, 1, 3).to(self.device))

			# shape = temporal_weights.shape
			# temporal_weights = temporal_weights.reshape(shape[0], self.batch_size, self.num_agents, self.reward_n_heads, self.max_time_steps+1, self.max_time_steps+1)
			# shape = agent_weights.shape
			# agent_weights = agent_weights.reshape(shape[0], self.batch_size, self.num_agents, self.reward_n_heads, self.max_time_steps+1, self.max_time_steps+1)
			# print(temporal_weights.shape, agent_weights.shape)

			loss = F.huber_loss((reward_time_wise*team_masks.view(*shape).to(self.device)).sum(dim=-1), episodic_rewards.to(self.device), reduction='sum')/team_masks.sum()
			


		self.reward_optimizer.zero_grad()
		loss.backward()
		if self.enable_reward_grad_clip:
			grad_norm_value_reward = torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.reward_grad_clip_value)
		else:
			total_norm = 0
			for name, p in self.reward_model.named_parameters():
				if p.requires_grad is False:
					continue
				param_norm = p.grad.detach().data.norm(2)
				total_norm += param_norm.item() ** 2
			grad_norm_value_reward = torch.tensor([total_norm ** 0.5])
		self.reward_optimizer.step()

		if self.experiment_type == "AREL":
			return loss, reward_var, grad_norm_value_reward
		elif self.experiment_type == "ATRR":
			return loss, temporal_weights_entropy, agent_weights_entropy, grad_norm_value_reward

		
	def plot(self, masks, episode):
		self.comet_ml.log_metric('Q_Value_Loss',self.plotting_dict["q_value_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_Q_Value',self.plotting_dict["grad_norm_value_q"],episode)
		self.comet_ml.log_metric('Policy_Loss',self.plotting_dict["policy_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_Policy',self.plotting_dict["grad_norm_policy"],episode)
		self.comet_ml.log_metric('Entropy',self.plotting_dict["entropy"],episode)

		# ENTROPY OF Q WEIGHTS
		for i in range(self.num_heads):
			entropy_weights = -torch.sum(torch.sum((self.plotting_dict["attention_weights_q"][:, i] * torch.log(torch.clamp(self.plotting_dict["attention_weights_q"][:, i], 1e-10, 1.0)) * masks.view(-1, self.num_agents, 1)), dim=-1))/masks.sum()
			self.comet_ml.log_metric('Q_Weight_Entropy_Head_'+str(i+1), entropy_weights.item(), episode)

	
	def update_parameters(self):
		if self.critic_weight_entropy_pen_final + self.critic_weight_entropy_pen_decay_rate > self.critic_weight_entropy_pen:
			self.critic_weight_entropy_pen += self.critic_weight_entropy_pen_decay_rate 

		if self.entropy_pen - self.entropy_pen_decay > self.entropy_pen_final:
			self.entropy_pen -= self.entropy_pen_decay


	def update(self, episode):
		# update reward model is being called from train_agent
		# for i in range(10):
		# 	self.update_reward_model(episode-(10-i))
		
		q_value_loss_batch = 0
		policy_loss_batch = 0
		entropy_batch = 0
		attention_weights_q_batch = None
		grad_norm_value_q_batch = 0
		grad_norm_policy_batch = 0

		if self.experiment_type == "AREL":
			with torch.no_grad():
				state_allies, state_enemies = torch.from_numpy(self.buffer.states_critic_allies).float(), torch.from_numpy(self.buffer.states_critic_enemies).float()
				state_enemies = state_enemies.unsqueeze(2).repeat(1, 1, self.num_agents, 1, 1).reshape(state_enemies.shape[0], state_enemies.shape[1], self.num_agents, -1)
				states = torch.cat([state_allies, state_enemies], dim=-1)
				one_hot_actions = torch.from_numpy(self.buffer.one_hot_actions).float()
				_, reward_time_wise = self.reward_model(torch.cat([states, one_hot_actions], dim=-1).permute(0,2,1,3).to(self.device))
				reward_time_wise = reward_time_wise * ((1-torch.from_numpy(self.buffer.dones[:,:-1,:]).to(self.device)).sum(dim=-1)>0).float()
				
				if self.norm_rewards:
					shape = reward_time_wise.shape
					reward_time_wise = self.reward_normalizer.denormalize(reward_time_wise.view(-1)).view(shape)*((1-torch.from_numpy(self.buffer.dones[:,:-1,:]).to(self.device)).sum(dim=-1)>0).float()

				self.buffer.rewards = reward_time_wise.unsqueeze(-1).repeat(1, 1, self.num_agents).cpu().numpy() #(reward_time_wise.unsqueeze(-1).repeat(1, 1, self.num_agents)*(1-torch.from_numpy(self.buffer.dones[:,:-1,:]).to(self.device))).cpu().numpy()

		self.buffer.calculate_targets(episode)

		# torch.autograd.set_detect_anomaly(True)
		# Optimize policy for n epochs
		for _ in range(self.n_epochs):

			# SAMPLE DATA FROM BUFFER
			states_critic_allies, states_critic_enemies, hidden_state_q, states_actor, hidden_state_actor, logprobs_old, \
			actions, last_one_hot_actions, one_hot_actions, action_masks, masks, q_values_old, target_q_values, advantage  = self.buffer.sample_recurrent_policy()

			q_values_old *= masks

			if self.norm_adv:
				shape = advantage.shape
				advantage_copy = copy.deepcopy(advantage)
				advantage_copy[masks.view(*shape) == 0.0] = float('nan')
				advantage_mean = torch.nanmean(advantage_copy)
				advantage_std = torch.from_numpy(np.array(np.nanstd(advantage_copy.cpu().numpy()))).float()
				
				advantage = ((advantage - advantage_mean) / (advantage_std + 1e-5))*masks.view(*shape)

			target_shape = q_values_old.shape
			q_values, attention_weights_q, score_q, _ = self.critic_network_q(
												states_critic_allies.to(self.device),
												states_critic_enemies.to(self.device),
												one_hot_actions.to(self.device),
												hidden_state_q.to(self.device),
												masks.to(self.device),
												)
			q_values = q_values.reshape(*target_shape)

			dists, _ = self.policy_network(
					states_actor.to(self.device),
					last_one_hot_actions.to(self.device),
					hidden_state_actor.to(self.device),
					action_masks.to(self.device),
					)

			q_values *= masks.to(self.device)	
			target_q_values *= masks

			probs = Categorical(dists)
			logprobs = probs.log_prob(actions.to(self.device))	

			critic_q_loss_1 = F.huber_loss(q_values, target_q_values.to(self.device), reduction="sum", delta=10.0) / masks.sum()
			critic_q_loss_2 = F.huber_loss(torch.clamp(q_values, q_values_old.to(self.device)-self.value_clip, q_values_old.to(self.device)+self.value_clip), target_q_values.to(self.device), reduction="sum", delta=10.0) / masks.sum()

			entropy_weights = 0
			score_q_cum = 0
			for i in range(self.num_heads):
				entropy_weights += -torch.sum(torch.sum((attention_weights_q[:, i] * torch.log(torch.clamp(attention_weights_q[:, i], 1e-10, 1.0)) * masks.view(-1, self.num_agents, 1).to(self.device)), dim=-1))/masks.sum()
				score_q_cum += (score_q[:, i].squeeze(-2)**2 * masks.view(-1, self.num_agents, 1).to(self.device)).sum()/masks.sum()
			
			critic_q_loss = torch.max(critic_q_loss_1, critic_q_loss_2) + self.critic_score_regularizer*score_q_cum + self.critic_weight_entropy_pen*entropy_weights

			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp((logprobs - logprobs_old.to(self.device)))
			
			# Finding Surrogate Loss
			surr1 = ratios * advantage.to(self.device) * masks.to(self.device)
			surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage.to(self.device) * masks.to(self.device)

			# final loss of clipped objective PPO
			entropy = -torch.sum(torch.sum(dists*masks.unsqueeze(-1).to(self.device) * torch.log(torch.clamp(dists*masks.unsqueeze(-1).to(self.device), 1e-10,1.0)), dim=-1))/ masks.sum() #(masks.sum()*self.num_agents)
			policy_loss_ = (-torch.min(surr1, surr2).sum())/masks.sum()
			policy_loss = policy_loss_ - self.entropy_pen*entropy

			print("Policy Loss", policy_loss_.item(), "Entropy", (-self.entropy_pen*entropy.item()))

			self.q_critic_optimizer.zero_grad()
			critic_q_loss.backward()
			if self.enable_grad_clip_critic_q:
				grad_norm_value_q = torch.nn.utils.clip_grad_norm_(self.critic_network_q.parameters(), self.grad_clip_critic_q)
			else:
				total_norm = 0
				for p in self.critic_network_q.parameters():
					param_norm = p.grad.detach().data.norm(2)
					total_norm += param_norm.item() ** 2
				grad_norm_value_q = torch.tensor([total_norm ** 0.5])
			self.q_critic_optimizer.step()

			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			if self.enable_grad_clip_actor:
				grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.grad_clip_actor)
			else:
				total_norm = 0
				for p in self.policy_network.parameters():
					param_norm = p.grad.detach().data.norm(2)
					total_norm += param_norm.item() ** 2
				grad_norm_policy = torch.tensor([total_norm ** 0.5])
			self.policy_optimizer.step()

			q_value_loss_batch += critic_q_loss.item()
			policy_loss_batch += policy_loss.item()
			entropy_batch += entropy.item()
			grad_norm_value_q_batch += grad_norm_value_q
			grad_norm_policy_batch += grad_norm_policy
			if attention_weights_q_batch is None:
				attention_weights_q_batch = attention_weights_q.detach().cpu()
			else:
				attention_weights_q_batch += attention_weights_q.detach().cpu()
			

		# Copy new weights into old critic
		if self.soft_update_q:
			for target_param, param in zip(self.target_critic_network_q.parameters(), self.critic_network_q.parameters()):
				target_param.data.copy_(target_param.data * (1.0 - self.tau_q) + param.data * self.tau_q)
		elif episode % self.network_update_interval_q == 0:
			self.target_critic_network_q.load_state_dict(self.critic_network_q.state_dict())

		# clear buffer
		self.buffer.clear()
		

		q_value_loss_batch /= self.n_epochs
		policy_loss_batch /= self.n_epochs
		entropy_batch /= self.n_epochs
		grad_norm_value_q_batch /= self.n_epochs
		grad_norm_policy_batch /= self.n_epochs
		attention_weights_q_batch /= self.n_epochs


		
		self.plotting_dict = {
		"q_value_loss": q_value_loss_batch,
		"policy_loss": policy_loss_batch,
		"entropy": entropy_batch,
		"grad_norm_value_q": grad_norm_value_q_batch,
		"grad_norm_policy": grad_norm_policy_batch,
		"attention_weights_q": attention_weights_q_batch,
		}

		if self.comet_ml is not None:
			self.plot(masks, episode)

		del q_value_loss_batch, policy_loss_batch, entropy_batch, grad_norm_value_q_batch, grad_norm_policy_batch, attention_weights_q_batch
		torch.cuda.empty_cache()