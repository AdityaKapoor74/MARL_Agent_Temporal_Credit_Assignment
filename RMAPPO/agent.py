import numpy as np
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from model import Policy, Q_network
from utils import RolloutBuffer, RewardReplayMemory

class PPOAgent:

	def __init__(
		self, 
		env, 
		dictionary,
		comet_ml,
		):

		# Environment Setup
		self.env = env
		self.algorithm_type = dictionary["algorithm_type"]
		self.env_name = dictionary["env"]
		self.num_agents = self.env.n_agents
		self.num_enemies = self.env.n_enemies
		self.num_actions = self.env.action_space[0].n
		self.reward_model_obs_shape = dictionary["reward_model_obs_shape"]

		# Training setup
		self.use_reward_model = dictionary["use_reward_model"]
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

		# Reward Model Setup
		self.use_reward_model = dictionary["use_reward_model"]
		self.reward_lr = dictionary["reward_lr"]
		self.variance_loss_coeff = dictionary["variance_loss_coeff"]
		self.enable_reward_grad_clip = dictionary["enable_reward_grad_clip"]
		self.reward_grad_clip_value = dictionary["reward_grad_clip_value"]
		self.reward_n_heads = dictionary["reward_n_heads"]
		self.norm_rewards = dictionary["norm_rewards"]
		self.temporal_score_coefficient = dictionary["temporal_score_coefficient"]
		self.agent_score_coefficient = dictionary["agent_score_coefficient"]

		# Critic Setup
		self.temperature_q = dictionary["temperature_q"]
		self.rnn_num_layers_q = dictionary["rnn_num_layers_q"]
		self.rnn_hidden_q = dictionary["rnn_hidden_q"]
		if self.algorithm_type in ["IPPO", "IAC"]:
			self.critic_observation_shape = dictionary["local_observation"]
		else:
			self.critic_observation_shape = dictionary["global_observation"]
		self.q_value_lr = dictionary["q_value_lr"]
		self.q_weight_decay = dictionary["q_weight_decay"]
		self.target_calc_style = dictionary["target_calc_style"]
		self.td_lambda = dictionary["td_lambda"] # TD lambda
		self.n_steps = dictionary["n_steps"]
		self.value_clip = dictionary["value_clip"]
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

		print("EXPERIMENT TYPE", self.experiment_type)

		# Q Network
		self.critic_network_q = Q_network(
			obs_input_dim=self.critic_observation_shape, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies,
			num_actions=self.num_actions, 
			rnn_num_layers=self.rnn_num_layers_q,
			value_norm=self.norm_returns_q,
			device=self.device, 
			).to(self.device)
		self.target_critic_network_q = Q_network(
			obs_input_dim=self.critic_observation_shape, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies,
			num_actions=self.num_actions, 
			rnn_num_layers=self.rnn_num_layers_q,
			value_norm=self.norm_returns_q,
			device=self.device, 
			).to(self.device)
		# Copy network params
		self.target_critic_network_q.load_state_dict(self.critic_network_q.state_dict())
		# Disable updates for old network
		for param in self.target_critic_network_q.parameters():
			param.requires_grad_(False)

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
			obs_shape_critic=self.critic_observation_shape, 
			obs_shape_actor=self.actor_observation_shape, 
			rnn_num_layers_actor=self.rnn_num_layers_actor,
			actor_hidden_state=self.rnn_hidden_actor,
			rnn_num_layers_q=self.rnn_num_layers_q,
			q_hidden_state=self.rnn_hidden_q,
			obs_shape_reward_model=self.reward_model_obs_shape,
			num_actions=self.num_actions,
			data_chunk_length=self.data_chunk_length,
			norm_returns_q=self.norm_returns_q,
			target_calc_style=self.target_calc_style,
			td_lambda=self.td_lambda,
			gae_lambda=self.gae_lambda,
			n_steps=self.n_steps,
			gamma=self.gamma,
			Q_PopArt=self.critic_network_q.q_value_layer[-1],
			)

		# Loading models
		if dictionary["load_models"]:
			# For CPU
			if torch.cuda.is_available() is False:
				self.critic_network.load_state_dict(torch.load(dictionary["model_path_value"], map_location=torch.device('cpu')))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"], map_location=torch.device('cpu')))
			# For GPU
			else:
				self.critic_network.load_state_dict(torch.load(dictionary["model_path_value"]))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"]))


		self.q_critic_optimizer = optim.AdamW(self.critic_network_q.parameters(), lr=self.q_value_lr, weight_decay=self.q_weight_decay, eps=1e-05)
		self.policy_optimizer = optim.AdamW(self.policy_network.parameters(),lr=self.policy_lr, weight_decay=self.policy_weight_decay, eps=1e-05)

		if self.scheduler_need:
			self.scheduler_policy = optim.lr_scheduler.MultiStepLR(self.policy_optimizer, milestones=[1000, 20000], gamma=0.1)
			self.scheduler_q_critic = optim.lr_scheduler.MultiStepLR(self.q_critic_optimizer, milestones=[1000, 20000], gamma=0.1)


		if self.use_reward_model:
			self.reward_model_buffer = RewardReplayMemory(
				experiment_type = self.experiment_type,
				capacity = dictionary["replay_buffer_size"],
				max_episode_len = self.max_time_steps,
				num_agents = self.num_agents,
				reward_model_obs_shape = dictionary["reward_model_obs_shape"],
				action_shape = self.num_actions,
				device = self.device,
				)

			if "AREL" in self.experiment_type:
				from AREL import AREL
				self.reward_model = AREL.Time_Agent_Transformer(
					emb=dictionary["reward_model_obs_shape"]+self.num_actions,
					heads=dictionary["reward_n_heads"], 
					depth=dictionary["reward_depth"], 
					seq_length=dictionary["max_time_steps"], 
					n_agents=self.num_agents, 
					agent=dictionary["reward_agent_attn"], 
					dropout=dictionary["reward_dropout"], 
					wide=dictionary["reward_attn_net_wide"], 
					version=dictionary["version"], 
					norm_rewards=dictionary["norm_rewards"],
					linear_compression_dim=dictionary["reward_linear_compression_dim"],
					device=self.device,
					).to(self.device)

			elif "ATRR" in self.experiment_type:
				from ATRR import ATRR
				self.reward_model = ATRR.Time_Agent_Transformer(
					obs_shape=dictionary["reward_model_obs_shape"],
					action_shape=self.num_actions, 
					heads=dictionary["reward_n_heads"], 
					depth=dictionary["reward_depth"], 
					seq_length=dictionary["max_time_steps"], 
					n_agents=self.num_agents, 
					agent=dictionary["reward_agent_attn"], 
					dropout=dictionary["reward_dropout"], 
					wide=dictionary["reward_attn_net_wide"], 
					version=dictionary["version"], 
					linear_compression_dim=dictionary["reward_linear_compression_dim"],
					norm_rewards=dictionary["norm_rewards"],
					device=self.device,
					).to(self.device)

			if self.norm_rewards:
				self.reward_normalizer = self.reward_model.rblocks[-1]
			
			if dictionary["load_models"]:
				# For CPU
				if torch.cuda.is_available() is False:
					self.reward_model.load_state_dict(torch.load(dictionary["model_path_reward_net"], map_location=torch.device('cpu')))
				# For GPU
				else:
					self.reward_model.load_state_dict(torch.load(dictionary["model_path_reward_net"]))


			self.reward_optimizer = optim.AdamW(self.reward_model.parameters(), lr=dictionary["reward_lr"], weight_decay=dictionary["reward_weight_decay"], eps=1e-5)
			
			if self.scheduler_need:
				self.scheduler_reward = optim.lr_scheduler.MultiStepLR(self.reward_optimizer, milestones=[10000, 30000], gamma=0.5)

		else:

			self.reward_model = None


		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml

	
	def get_q_values(self, state_critic, rnn_hidden_state_q, indiv_dones):
		with torch.no_grad():
			indiv_masks = [1-d for d in indiv_dones]
			indiv_masks = torch.FloatTensor(indiv_masks).unsqueeze(0).unsqueeze(0)
			state_critic = torch.FloatTensor(state_critic).unsqueeze(0).unsqueeze(0)
			rnn_hidden_state_q = torch.FloatTensor(rnn_hidden_state_q)
			Q_value, rnn_hidden_state_q = self.target_critic_network_q(state_critic.to(self.device), rnn_hidden_state_q.to(self.device), indiv_masks.to(self.device))

			return Q_value.squeeze(0).cpu().numpy(), rnn_hidden_state_q.cpu().numpy()

	
	def get_action(self, state_actor, mask_actions, hidden_state, greedy=False):
		with torch.no_grad():
			state_actor = torch.FloatTensor(state_actor).unsqueeze(0).unsqueeze(1).to(self.device)
			mask_actions = torch.BoolTensor(mask_actions).unsqueeze(0).unsqueeze(1).to(self.device)
			hidden_state = torch.FloatTensor(hidden_state).to(self.device)
			dists, hidden_state = self.policy_network(state_actor, hidden_state, mask_actions)

			if greedy:
				actions = [dist.argmax().detach().cpu().item() for dist in dists.squeeze(0).squeeze(0)]
				action_logprob = None
			else:
				actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists.squeeze(0).squeeze(0)]

				probs = Categorical(dists)
				action_logprob = probs.log_prob(torch.FloatTensor(actions).to(self.device)).cpu().numpy()

			return actions, action_logprob, hidden_state.cpu().numpy()


	def reward_model_output(self, eval_reward_model=False):
		if eval_reward_model:
			latest_sample_index = self.buffer.episode_num
			state_batch = torch.from_numpy(self.buffer.reward_model_obs[latest_sample_index]).float().unsqueeze(0)
			one_hot_actions_batch = torch.from_numpy(self.buffer.one_hot_actions[latest_sample_index]).float().unsqueeze(0)
			team_mask_batch = 1-torch.from_numpy(self.buffer.team_dones[latest_sample_index]).float().unsqueeze(0)
			agent_masks_batch = 1-torch.from_numpy(self.buffer.agent_dones[latest_sample_index, :-1, :]).float().unsqueeze(0)
			episode_len_batch = torch.from_numpy(self.buffer.episode_length[latest_sample_index, :-1]).long().unsqueeze(0)
			episodic_reward_batch = torch.from_numpy(self.buffer.rewards[latest_sample_index, :, 0]).float().sum(dim=-1).unsqueeze(0)
		else:
			state_batch = torch.from_numpy(self.buffer.reward_model_obs).float()
			one_hot_actions_batch = torch.from_numpy(self.buffer.one_hot_actions).float()
			team_mask_batch = 1-torch.from_numpy(self.buffer.team_dones[:, :-1]).float()
			agent_masks_batch = 1-torch.from_numpy(self.buffer.agent_dones[:, :-1, :]).float()
			episode_len_batch = torch.from_numpy(self.buffer.episode_length).long()
			episodic_reward_batch = torch.from_numpy(self.buffer.rewards[:, :, 0]).float().sum(dim=-1)
		
		with torch.no_grad():
			if "AREL" in self.experiment_type:
				state_actions_batch = torch.cat([state_batch, one_hot_actions_batch], dim=-1)

				with torch.no_grad():
					reward_episode_wise, reward_time_wise, _, _, agent_weights, _ = self.reward_model(
						state_actions_batch.permute(0, 2, 1, 3).to(self.device),
						team_masks=team_mask_batch.to(self.device),
						agent_masks=agent_masks_batch.to(self.device)
						)

				reward_batch = reward_time_wise.cpu()

				if self.experiment_type == "AREL_agent":
					reward_batch = reward_batch.unsqueeze(-1) * agent_weights.cpu()
				else:
					reward_batch = reward_batch.unsqueeze(-1).repeat(1, 1, self.num_agents)

				return reward_batch


			elif "ATRR" in self.experiment_type:

				with torch.no_grad():
					rewards, temporal_weights, agent_weights, temporal_weights_final_temporal_block,\
					temporal_scores, agent_scores, temporal_scores_final_temporal_block = self.reward_model(
						state_batch.permute(0, 2, 1, 3).to(self.device), 
						one_hot_actions_batch.permute(0, 2, 1, 3).to(self.device), 
						team_masks=team_mask_batch.to(self.device),
						agent_masks=agent_masks_batch.to(self.device),
						episode_len=episode_len_batch.to(self.device),
						)

					temporal_contribution = F.softmax(rewards.detach().sum(dim=-1), dim=-1).unsqueeze(-1)
					agent_contribution = F.softmax(rewards.detach(), dim=-1)
					rewards = episodic_reward_batch.reshape(-1, 1, 1) * temporal_contribution * agent_contribution

					if self.experiment_type == "ATRR_temporal_attn_weights":
						b, t, n_a, _ = state_batch.shape
						# use last attn block
						# temporal_weights_final = temporal_weights[-1].sum(dim=1)[torch.arange(x.shape[0]), episode_len_batch, :]/(agent_masks_batch.permute(0, 2, 1).sum(dim=1)+1e-5)
						# use attention rollout
						temporal_weights_final = temporal_weights.detach().cpu().sum(dim=2)/(agent_masks_batch.permute(0, 2, 1).sum(dim=1).reshape(1, b, t, 1)+1e-5)
						temporal_weights_final = (temporal_weights_final[0] @ temporal_weights_final[1] @ temporal_weights_final[2, torch.arange(b), episode_len_batch, :].unsqueeze(2)).squeeze(-1)
						temporal_weights_final = F.normalize(temporal_weights_final, dim=-1, p=1.0)
						rewards = (episodic_reward_batch.unsqueeze(-1) * temporal_weights_final).unsqueeze(-1).repeat(1, 1, n_a)
					elif self.experiment_type == "ATRR_agent_temporal_attn_weights":
						b, t, n_a, _ = state_batch.shape
						# use last attn block
						# temporal_weights_final = temporal_weights[-1].sum(dim=1)[torch.arange(x.shape[0]), episode_len_batch, :]/(agent_masks_batch.permute(0, 2, 1).sum(dim=1)+1e-5)
						# use attention rollout
						temporal_weights_final = temporal_weights.detach().cpu().sum(dim=2)/(agent_masks_batch.permute(0, 2, 1).sum(dim=1).reshape(1, b, t, 1)+1e-5)
						temporal_weights_final = (temporal_weights_final[0] @ temporal_weights_final[1] @ temporal_weights_final[2, torch.arange(b), episode_len_batch, :].unsqueeze(2)).squeeze(-1)
						temporal_weights_final = F.normalize(temporal_weights_final, dim=-1, p=1.0)
						rewards = (episodic_reward_batch.unsqueeze(-1) * temporal_weights_final).unsqueeze(-1) * (agent_weights.detach().cpu().mean(dim=0).sum(dim=-2)/(agent_masks_batch.permute(0, 2, 1).sum(dim=1).unsqueeze(-1)+1e-5))
					
					if self.experiment_type == "ATRR_temporal":
						rewards = rewards.unsqueeze(-1).repeat(1, 1, self.num_agents)

					if self.experiment_type == "ATRR_temporal_v2":
						mask_value = torch.tensor(torch.finfo(torch.float).min, dtype=torch.float)
						# void weightage to timesteps after multi-agent system death
						temporal_weightage = F.softmax(torch.where(team_mask_batch.bool().unsqueeze(-1).repeat(1, 1, self.num_agents), rewards.cpu(), mask_value), dim=-2) # batch, timesteps, num_agents
						rewards = episodic_reward_batch.reshape(-1, 1, 1) * temporal_weightage

				# if self.norm_rewards:
				# 	shape = rewards.shape
				# 	rewards = self.reward_normalizer.denormalize(rewards.cpu().view(-1)).view(shape) * agent_masks_batch

				# print("true episodic reward")
				# print(episodic_reward_batch[0])
				# print("rewards post distribution")
				# print(rewards[0])
		
			return rewards.cpu()

		

	def update_reward_model(self, sample):
		# sample episodes from replay buffer
		reward_model_obs_batch, one_hot_actions, reward_batch, team_mask_batch, agent_masks_batch, episode_len_batch = sample
		# convert numpy array to tensor
		reward_model_obs_batch = torch.from_numpy(reward_model_obs_batch).float()
		one_hot_actions = torch.from_numpy(one_hot_actions).float() # same as current one_hot_actions
		reward_batch = torch.from_numpy(reward_batch).float()
		episodic_reward_batch = reward_batch.sum(dim=1)
		team_mask_batch = torch.from_numpy(team_mask_batch).float()
		agent_masks_batch = torch.from_numpy(agent_masks_batch).float()
		episode_len_batch = torch.from_numpy(episode_len_batch).long()

		if self.norm_rewards:
			shape = episodic_reward_batch.shape
			episodic_reward_batch = self.reward_normalizer.normalize(episodic_reward_batch.view(-1)).view(shape)

		# if self.norm_rewards:
		# 	shape = episodic_reward_batch.shape
		# 	self.reward_normalizer.update(episodic_reward_batch.view(-1))
			
		# 	episodic_reward_batch = self.reward_normalizer.normalize(episodic_reward_batch.view(-1)).view(shape)

		if "AREL" in self.experiment_type:
			state_actions_batch = torch.cat([reward_model_obs_batch, one_hot_actions], dim=-1)  # state_actions_batch.size = (b, t, n_agents, e)
			reward_episode_wise, reward_time_wise, _, _, _, _ = self.reward_model(
				state_actions_batch.permute(0, 2, 1, 3).to(self.device),
				team_masks=team_mask_batch.to(self.device),
				agent_masks=agent_masks_batch.to(self.device)
				)


			shape = reward_time_wise.shape
			# reward_copy = copy.deepcopy(reward_time_wise.detach())
			# reward_copy[team_mask_batch.view(*shape) == 0.0] = 0.0 
			# reward_mean = (reward_copy.sum(dim=-1)/team_mask_batch.to(self.device).sum(dim=-1)).unsqueeze(-1)
			# print("Check for NaN")
			# print(torch.isnan(reward_time_wise).any())
			# print(torch.isnan(reward_mean).any())
			# reward_var = (reward_time_wise - reward_mean)**2
			# reward_var[team_mask_batch.view(*shape) == 0.0] = 0.0
			# reward_var = reward_var.sum() / team_mask_batch.sum()

			# print("Huber Loss")
			# print(F.huber_loss((reward_time_wise*team_mask_batch.view(*shape).to(self.device)).sum(dim=-1), episodic_reward_batch.to(self.device)))
			# print("Reward Var")
			# print(reward_var)
			reward_var = torch.tensor([-1])
			reward_loss = F.huber_loss((reward_time_wise*team_mask_batch.view(*shape).to(self.device)).sum(dim=-1), episodic_reward_batch.to(self.device)) #+ self.variance_loss_coeff*reward_var

		elif "ATRR" in self.experiment_type:
			
			rewards, temporal_weights, agent_weights, temporal_weights_final_temporal_block, temporal_scores, agent_scores, temporal_scores_final_temporal_block = self.reward_model(
				reward_model_obs_batch.permute(0, 2, 1, 3).to(self.device), 
				one_hot_actions.permute(0, 2, 1, 3).to(self.device), 
				team_masks=team_mask_batch.to(self.device),
				agent_masks=agent_masks_batch.to(self.device),
				episode_len=episode_len_batch.to(self.device),
				)

			temporal_weights = temporal_weights.cpu().mean(dim=0).sum(dim=1) / (agent_masks_batch.permute(0, 2, 1).sum(dim=1).unsqueeze(-1)+1e-5)
			agent_weights = agent_weights.cpu().mean(dim=0)
			entropy_temporal_weights = (-torch.sum(temporal_weights * torch.log(torch.clamp(temporal_weights, 1e-10, 1.0)))/(team_mask_batch.sum()+1e-5)).item()
			entropy_agent_weights = (-torch.sum(agent_weights.reshape(-1, self.num_agents) * torch.log(torch.clamp(agent_weights.reshape(-1, self.num_agents), 1e-10, 1.0)))/agent_masks_batch.sum()).item() 
			
			if temporal_weights_final_temporal_block is not None:
				entropy_final_temporal_block = (-torch.sum(temporal_weights_final_temporal_block * torch.log(torch.clamp(temporal_weights_final_temporal_block, 1e-10, 1.0)))/team_mask_batch.shape[0]).item()
			else:
				entropy_final_temporal_block = None

			reward_loss = F.huber_loss(rewards.reshape(reward_model_obs_batch.shape[0], -1).sum(dim=-1), episodic_reward_batch.to(self.device)) + self.temporal_score_coefficient * (temporal_scores**2).sum() + self.agent_score_coefficient * (agent_scores**2).sum()
			
			if temporal_scores_final_temporal_block is not None:
				reward_loss += self.temporal_score_coefficient * (temporal_scores_final_temporal_block**2).sum()

		self.reward_optimizer.zero_grad()
		reward_loss.backward()
		if self.enable_reward_grad_clip:
			grad_norm_value_reward = torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.reward_grad_clip_value)
		else:
			total_norm = 0
			for name, p in self.reward_model.named_parameters():
				if p.requires_grad is False or p.grad is None:
					continue
				param_norm = p.grad.detach().data.norm(2)
				total_norm += param_norm.item() ** 2
			grad_norm_value_reward = torch.tensor([total_norm ** 0.5])
		self.reward_optimizer.step()

		if "AREL" in self.experiment_type:
			return reward_loss.item(), reward_var.item(), grad_norm_value_reward.item()
		elif "ATRR" in self.experiment_type:
			return reward_loss.item(), entropy_temporal_weights, entropy_agent_weights, entropy_final_temporal_block, grad_norm_value_reward.item()



	def plot(self, episode):
		self.comet_ml.log_metric('Q_Value_Loss',self.plotting_dict["q_value_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_Q_Value',self.plotting_dict["grad_norm_value_q"],episode)
		self.comet_ml.log_metric('Policy_Loss',self.plotting_dict["policy_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_Policy',self.plotting_dict["grad_norm_policy"],episode)
		self.comet_ml.log_metric('Entropy',self.plotting_dict["entropy"],episode)
		

	def update_parameters(self):
		if self.entropy_pen - self.entropy_pen_decay > self.entropy_pen_final:
			self.entropy_pen -= self.entropy_pen_decay


	def update(self, episode):
		
		q_value_loss_batch = 0
		policy_loss_batch = 0
		entropy_batch = 0
		grad_norm_value_q_batch = 0
		grad_norm_policy_batch = 0

		self.buffer.calculate_targets()

		
		# torch.autograd.set_detect_anomaly(True)
		# Optimize policy for n epochs
		for _ in range(self.n_epochs):

			# SAMPLE DATA FROM BUFFER
			states_critic, hidden_state_q, states_actor, hidden_state_actor, logprobs_old, \
			actions, one_hot_actions, action_masks, team_masks, agent_masks, q_values_old, target_q_values, advantage  = self.buffer.sample_recurrent_policy()

			# print("pre advantage normalization")
			# print(advantage[:, 0])

			if self.norm_adv:
				shape = advantage.shape
				advantage_copy = copy.deepcopy(advantage)
				advantage_copy[agent_masks.view(*shape) == 0.0] = float('nan')
				advantage_mean = torch.nanmean(advantage_copy)
				advantage_std = torch.from_numpy(np.array(np.nanstd(advantage_copy.cpu().numpy()))).float()

				advantage = ((advantage - advantage_mean) / (advantage_std + 1e-5))*agent_masks.view(*shape)


			target_shape = q_values_old.shape
			q_values, _ = self.critic_network_q(
							states_critic.to(self.device),
							hidden_state_q.to(self.device),
							agent_masks.to(self.device),
							)
			q_values = q_values.reshape(*target_shape)

			q_values_old *= agent_masks
			q_values *= agent_masks.to(self.device)	
			target_q_values *= agent_masks
			
			dists, _ = self.policy_network(
					states_actor.to(self.device),
					hidden_state_actor.to(self.device),
					action_masks.to(self.device),
					)

			probs = Categorical(dists)
			logprobs = probs.log_prob(actions.to(self.device))
			
			
			if self.algorithm_type == "IAC" or self.algorithm_type == "MAAC":
				critic_q_loss = F.huber_loss(q_values, target_q_values.to(self.device), reduction="sum", delta=10.0) / (agent_masks.sum()+1e-5)
				policy_loss_ = (logprobs * advantage.to(self.device) * agent_masks.to(self.device)).sum()/(agent_masks.sum()+1e-5)
			else:
				critic_q_loss_1 = F.huber_loss(q_values, target_q_values.to(self.device), reduction="sum", delta=10.0) / (agent_masks.sum()+1e-5)
				critic_q_loss_2 = F.huber_loss(torch.clamp(q_values, q_values_old.to(self.device)-self.value_clip, q_values_old.to(self.device)+self.value_clip), target_q_values.to(self.device), reduction="sum", delta=10.0) / (agent_masks.sum()+1e-5)
				critic_q_loss = torch.max(critic_q_loss_1, critic_q_loss_2)

				# Finding the ratio (pi_theta / pi_theta__old)
				ratios = torch.exp((logprobs - logprobs_old.to(self.device)))
				
				# Finding Surrogate Loss
				surr1 = ratios * advantage.to(self.device) * agent_masks.to(self.device)
				surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage.to(self.device) * agent_masks.to(self.device)

				# final loss of clipped objective PPO
				policy_loss_ = (-torch.min(surr1, surr2).sum())/(agent_masks.sum()+1e-5)

			# calculate entropy
			entropy = -torch.sum(torch.sum(dists*agent_masks.unsqueeze(-1).to(self.device) * torch.log(torch.clamp(dists*agent_masks.unsqueeze(-1).to(self.device), 1e-10,1.0)), dim=-1))/ (agent_masks.sum()+1e-5)
			# add entropy
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


		
		self.plotting_dict = {
		"q_value_loss": q_value_loss_batch,
		"policy_loss": policy_loss_batch,
		"entropy": entropy_batch,
		"grad_norm_value_q": grad_norm_value_q_batch,
		"grad_norm_policy": grad_norm_policy_batch,
		}

		
		if self.comet_ml is not None:
			self.plot(episode)

		del q_value_loss_batch, policy_loss_batch, entropy_batch, grad_norm_value_q_batch, grad_norm_policy_batch
		torch.cuda.empty_cache()