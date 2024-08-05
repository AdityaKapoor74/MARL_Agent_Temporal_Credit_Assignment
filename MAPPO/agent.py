import numpy as np
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from model import Policy, V_network, PopArt
from utils import RolloutBuffer, RewardReplayMemory

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

		# Model Setup
		if "StarCraft" in self.environment:
			self.ally_observation_shape = dictionary["ally_observation_shape"]
			self.num_enemies = dictionary["num_enemies"]
			self.enemy_observation_shape = dictionary["enemy_observation_shape"]

			self.global_observation_shape = None
		elif "Alice_and_Bob" in self.environment:
			self.global_observation_shape = dictionary["global_observation_shape"]

			self.ally_observation_shape = None
			self.num_enemies = None
			self.enemy_observation_shape = None

		self.local_observation_shape = dictionary["local_observation_shape"]

		# Reward Model Setup
		self.use_reward_model = dictionary["use_reward_model"]
		self.version = dictionary["version"]
		self.reward_lr = dictionary["reward_lr"]
		self.variance_loss_coeff = dictionary["variance_loss_coeff"]
		self.enable_reward_grad_clip = dictionary["enable_reward_grad_clip"]
		self.reward_grad_clip_value = dictionary["reward_grad_clip_value"]
		self.reward_n_heads = dictionary["reward_n_heads"]
		self.norm_rewards = dictionary["norm_rewards"]
		self.dynamic_loss_coeffecient = dictionary["dynamic_loss_coeffecient"]
		self.temporal_score_coefficient = dictionary["temporal_score_coefficient"]
		self.agent_score_coefficient = dictionary["agent_score_coefficient"]
		self.reward_depth = dictionary["reward_depth"]

		# Critic
		self.use_recurrent_critic = dictionary["use_recurrent_critic"]
		self.rnn_num_layers_v = dictionary["rnn_num_layers_v"]
		self.rnn_hidden_v = dictionary["rnn_hidden_v"]
		self.v_comp_emb_shape = dictionary["v_comp_emb_shape"]
		self.v_value_lr = dictionary["v_value_lr"]
		self.v_weight_decay = dictionary["v_weight_decay"]
		self.target_calc_style = dictionary["target_calc_style"]
		self.n_steps = dictionary["n_steps"]
		self.value_clip = dictionary["value_clip"]
		self.enable_grad_clip_critic_v = dictionary["enable_grad_clip_critic_v"]
		self.grad_clip_critic_v = dictionary["grad_clip_critic_v"]
		self.norm_returns_v = dictionary["norm_returns_v"]
		self.clamp_rewards = dictionary["clamp_rewards"]
		self.clamp_rewards_value_min = dictionary["clamp_rewards_value_min"]
		self.clamp_rewards_value_max = dictionary["clamp_rewards_value_max"]


		# Actor
		self.use_recurrent_policy = dictionary["use_recurrent_policy"]
		self.data_chunk_length = dictionary["data_chunk_length"]
		self.rnn_num_layers_actor = dictionary["rnn_num_layers_actor"]
		self.rnn_hidden_actor = dictionary["rnn_hidden_actor"]
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

		if dictionary["algorithm_type"] == "MAPPO":
			centralized = True
		else:
			centralized = False

		# Critic Network
		if self.norm_returns_v:
			self.V_PopArt = PopArt(input_shape=1, num_agents=self.num_agents, device=self.device)
		else:
			self.V_PopArt = None

		self.critic_network_v = V_network(
			environment=self.environment,
			use_recurrent_critic=self.use_recurrent_critic,
			centralized=centralized,
			local_observation_input_dim=self.local_observation_shape,
			global_observation_input_dim=self.global_observation_shape,
			ally_obs_input_dim=self.ally_observation_shape, 
			enemy_obs_input_dim=self.enemy_observation_shape,
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies, 
			num_actions=self.num_actions, 
			rnn_num_layers=self.rnn_num_layers_v,
			comp_emb_shape=self.v_comp_emb_shape,
			device=self.device, 
			).to(self.device)
		self.target_critic_network_v = V_network(
			environment=self.environment,
			use_recurrent_critic=self.use_recurrent_critic,
			centralized=centralized,
			local_observation_input_dim=self.local_observation_shape,
			global_observation_input_dim=self.global_observation_shape,
			ally_obs_input_dim=self.ally_observation_shape, 
			enemy_obs_input_dim=self.enemy_observation_shape,
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies, 
			num_actions=self.num_actions, 
			rnn_num_layers=self.rnn_num_layers_v,
			comp_emb_shape=self.v_comp_emb_shape,
			device=self.device, 
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
			global_obs_shape=self.global_observation_shape,
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
			
		
		if self.use_reward_model:
			self.reward_model_buffer = RewardReplayMemory(
				# experiment_type = self.experiment_type,
				capacity = dictionary["replay_buffer_size"],
				max_episode_len = self.max_time_steps,
				num_agents = self.num_agents,
				num_enemies=self.num_enemies,
				ally_obs_shape=self.ally_observation_shape,
				enemy_obs_shape=self.enemy_observation_shape,
				action_shape = self.num_actions,
				device = self.device,
				)

			if "AREL" in self.experiment_type:
				from AREL import AREL
				self.reward_model = AREL.Time_Agent_Transformer(
					ally_obs_shape=self.ally_observation_shape,
					enemy_obs_shape=self.enemy_observation_shape,
					action_shape=self.num_actions, 
					heads=dictionary["reward_n_heads"], 
					depth=dictionary["reward_depth"], 
					seq_length=dictionary["max_time_steps"], 
					n_agents=self.num_agents, 
					n_enemies=self.num_enemies,
					n_actions=self.num_actions,
					agent=dictionary["reward_agent_attn"], 
					dropout=dictionary["reward_dropout"], 
					wide=dictionary["reward_attn_net_wide"], 
					version=dictionary["version"], 
					linear_compression_dim=dictionary["reward_linear_compression_dim"],
					device=self.device,
					).to(self.device)

			elif "ATRR" in self.experiment_type:
				from ATRR import ATRR
				self.reward_model = ATRR.Time_Agent_Transformer(
					ally_obs_shape=self.ally_observation_shape,
					enemy_obs_shape=self.enemy_observation_shape,
					action_shape=self.num_actions, 
					heads=dictionary["reward_n_heads"], 
					depth=dictionary["reward_depth"], 
					seq_length=dictionary["max_time_steps"], 
					n_agents=self.num_agents, 
					n_enemies=self.num_enemies,
					n_actions=self.num_actions,
					agent=dictionary["reward_agent_attn"], 
					dropout=dictionary["reward_dropout"], 
					wide=dictionary["reward_attn_net_wide"], 
					version=dictionary["version"], 
					linear_compression_dim=dictionary["reward_linear_compression_dim"],
					device=self.device,
					).to(self.device)

			elif "STAS" in self.experiment_type:
				from STAS import stas
				self.reward_model = stas.STAS_ML(
					ally_obs_shape=self.ally_observation_shape,
					enemy_obs_shape=self.enemy_observation_shape, 
					n_actions=self.num_actions, 
					emb_dim=dictionary["reward_linear_compression_dim"], 
					n_heads=dictionary["reward_n_heads"], 
					n_layer=dictionary["reward_depth"], 
					seq_length=dictionary["max_time_steps"], 
					n_agents=self.num_agents, 
					n_enemies=self.num_enemies,
					sample_num=5,
					device=self.device, 
					dropout=0.3, 
					emb_dropout=0.0, 
					action_space='discrete'
					).to(self.device)

			if self.norm_rewards:
				self.reward_normalizer = PopArt(input_shape=1, num_agents=self.num_agents, device=self.device)
			
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

			self.classification_loss = nn.CrossEntropyLoss(reduction="none")

		else:
			self.reward_model = None



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

	
	def get_values(self, local_obs, global_obs, state_allies, state_enemies, actions, rnn_hidden_state_v, indiv_dones, episode):
		with torch.no_grad():
			indiv_masks = [1-d for d in indiv_dones]
			indiv_masks = torch.FloatTensor(indiv_masks).unsqueeze(0).unsqueeze(0).to(self.device)
			if "StarCraft" in self.environment:
				state_allies = torch.FloatTensor(state_allies).unsqueeze(0).unsqueeze(0).to(self.device)
				state_enemies = torch.FloatTensor(state_enemies).unsqueeze(0).unsqueeze(0).to(self.device)
			elif "Alice_and_Bob" in self.environment:
				global_obs = torch.FloatTensor(global_obs).unsqueeze(0).unsqueeze(0).to(self.device)
			local_obs = torch.FloatTensor(local_obs).unsqueeze(0).unsqueeze(0).to(self.device)
			actions = torch.FloatTensor(actions).unsqueeze(0).unsqueeze(0).to(self.device)
			rnn_hidden_state_v = torch.FloatTensor(rnn_hidden_state_v).to(self.device)
			
			Value, rnn_hidden_state_v = self.target_critic_network_v(local_obs, global_obs, state_allies, state_enemies, actions, rnn_hidden_state_v, indiv_masks)
				
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



	def reward_model_output(self, eval_reward_model=False):
		if eval_reward_model:
			latest_sample_index = self.buffer.episode_num
			ally_state_batch = torch.from_numpy(self.buffer.ally_states[latest_sample_index]).float().unsqueeze(0)
			enemy_state_batch = torch.from_numpy(self.buffer.enemy_states[latest_sample_index]).float().unsqueeze(0)
			actions_batch = torch.from_numpy(self.buffer.actions[latest_sample_index]).float().unsqueeze(0)
			logprobs_batch = torch.from_numpy(self.buffer.logprobs[latest_sample_index]).float().unsqueeze(0)
			team_mask_batch = 1-torch.from_numpy(self.buffer.team_dones[latest_sample_index]).float().unsqueeze(0)
			agent_masks_batch = 1-torch.from_numpy(self.buffer.indiv_dones[latest_sample_index, :-1, :]).float().unsqueeze(0)
			episode_len_batch = torch.from_numpy(self.buffer.episode_length[latest_sample_index, :-1]).long().unsqueeze(0)
			episodic_reward_batch = torch.from_numpy(self.buffer.rewards[latest_sample_index, :, 0]).float().sum(dim=-1).unsqueeze(0)
		else:
			ally_state_batch = torch.from_numpy(self.buffer.ally_states).float()
			enemy_state_batch = torch.from_numpy(self.buffer.enemy_states).float()
			actions_batch = torch.from_numpy(self.buffer.actions).float()
			logprobs_batch = torch.from_numpy(self.buffer.logprobs).float()
			team_mask_batch = 1-torch.from_numpy(self.buffer.team_dones[:, :-1]).float()
			agent_masks_batch = 1-torch.from_numpy(self.buffer.indiv_dones[:, :-1, :]).float()
			episode_len_batch = torch.from_numpy(self.buffer.episode_length).long()
			episodic_reward_batch = torch.from_numpy(self.buffer.rewards[:, :, 0]).float().sum(dim=-1)
		
		with torch.no_grad():
			if "AREL" in self.experiment_type:
				with torch.no_grad():
					rewards, temporal_weights, agent_weights,\
					temporal_scores, agent_scores = self.reward_model(
						ally_state_batch.permute(0, 2, 1, 3).to(self.device), 
						enemy_state_batch.permute(0, 2, 1, 3).to(self.device), 
						actions_batch.permute(0, 2, 1).to(self.device), 
						episodic_reward_batch.to(self.device),
						team_masks=team_mask_batch.to(self.device),
						agent_masks=agent_masks_batch.to(self.device),
						)


			elif "ATRR" in self.experiment_type:

				with torch.no_grad():
					returns, rewards, temporal_weights, agent_weights,\
					temporal_scores, agent_scores, action_prediction = self.reward_model(
						ally_state_batch.permute(0, 2, 1, 3).to(self.device), 
						enemy_state_batch.permute(0, 2, 1, 3).to(self.device), 
						actions_batch.permute(0, 2, 1).to(self.device), 
						episodic_reward_batch.to(self.device),
						team_masks=team_mask_batch.to(self.device),
						agent_masks=agent_masks_batch.to(self.device),
						logprobs=logprobs_batch.to(self.device),
						)

					print("*"*20)
					print("actions")
					print(actions_batch[0])
					print("rewards")
					print(rewards[0])

					if self.experiment_type == "ATRR_temporal_attn_weights":
						b, t, n_a, _ = state_batch.shape
						
						temporal_weights_final = F.softmax(torch.where(team_mask_batch.bool(), (temporal_scores[-1].mean(dim=1).sum(dim=1)/(agent_masks_batch.sum(dim=-1).reshape(b, t, 1)+1e-5)).diagonal(dim1=-2, dim2=-1), self.mask_value), dim=-1)
						rewards = (episodic_reward_batch.unsqueeze(-1) * temporal_weights_final.detach()).unsqueeze(-1).repeat(1, 1, n_a)
					

					elif self.experiment_type == "ATRR_agent_temporal_attn_weights":
						# b, t, n_a, _ = ally_state_batch.shape
							
						# # use temporal and agent attention networks to distribute rewards
						# indiv_agent_episode_len = (agent_masks_batch.sum(dim=-2)-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, t).long() # subtracting 1 for indexing purposes
						# temporal_weights_final = torch.gather(temporal_weights.mean(dim=0).detach().cpu().reshape(b, n_a, t, t), 2, indiv_agent_episode_len).squeeze(2).transpose(1, 2)

						# agent_weights_final = agent_weights.mean(dim=0).detach().cpu().sum(dim=-2)/(agent_masks_batch.sum(dim=-1, keepdim=True)+1e-5)
						# # # renormalizing
						# agent_weights_final = agent_weights_final / (agent_weights_final.sum(dim=-1, keepdim=True)+1e-5)
						
						# # # multi_agent_temporal_weights = (temporal_weights_final*agent_weights_final).sum(dim=-1)
						# multi_agent_temporal_weights = temporal_weights_final.sum(dim=-1) / (agent_masks_batch.sum(dim=-1)+1e-5)
						# # # renormalizing
						# multi_agent_temporal_weights = multi_agent_temporal_weights / (multi_agent_temporal_weights.sum(dim=-1, keepdim=True) + 1e-5)
						# temporal_rewards = multi_agent_temporal_weights * episodic_reward_batch.unsqueeze(-1)
						# agent_temporal_rewards = temporal_rewards.unsqueeze(-1) * agent_weights_final
						# rewards = agent_temporal_rewards

						# assuming predict episodic reward for each agent
						# indiv_agent_episode_len = (agent_masks_batch.sum(dim=-2)-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, t).long() # subtracting 1 for indexing purposes
						# temporal_weights_final = torch.gather(temporal_weights.mean(dim=0).detach().cpu().reshape(b, n_a, t, t), 2, indiv_agent_episode_len).squeeze(2).transpose(1, 2)
						# rewards = rewards.detach().cpu() * temporal_weights_final

						# use agent attention network to distribute rewards
						# agent_temporal_contribution_weights = torch.gather(temporal_weights.mean(dim=0).detach().cpu().reshape(b, n_a, t, t), 2, (team_mask_batch.sum(dim=-1)-1).reshape(b, 1, 1, 1).repeat(1, n_a, 1, t).long()).squeeze(2).transpose(1, 2)
						# agent_episodic_contribution = (agent_weights.mean(dim=0).detach().cpu().sum(dim=-2)*agent_temporal_contribution_weights).sum(dim=1) / (agent_weights.mean(dim=0).detach().cpu().sum(dim=-2)*agent_temporal_contribution_weights).sum(dim=1).sum(dim=-1, keepdims=True)
						# agent_episodic_rewards = rewards.cpu() # agent_episodic_contribution * episodic_reward_batch.unsqueeze(-1)
						# agent_temporal_contribution = torch.where(agent_masks_batch.bool(), (agent_weights.mean(dim=0).detach().cpu()).sum(dim=-2)*agent_temporal_contribution_weights, 0.0)
						# agent_temporal_contribution = agent_temporal_contribution / (agent_temporal_contribution.sum(dim=1, keepdims=True)+1e-5)
						# agent_temporal_contribution = F.softmax(torch.where(agent_masks_batch.bool(), agent_weights.mean(dim=0).detach().cpu().sum(dim=-2), -1e9), dim=1)
						# agent_temporal_rewards = agent_temporal_contribution * agent_episodic_rewards#.unsqueeze(1)
						# rewards = agent_temporal_rewards

						# print("agent_episodic_contribution")
						# print(agent_episodic_contribution)
						# print("agent_temporal_contribution")
						# print(agent_temporal_contribution)





						# CURRENT WORK 
						# b, t, n_a = logprobs_batch.shape
						# # we don't learn to predict the first action in the sequence so we assume that importance sampling for it is 1
						# gen_policy_probs = Categorical(F.softmax(action_prediction, dim=-1).transpose(1, 2))
						# gen_policy_logprobs = gen_policy_probs.log_prob(actions_batch.to(self.device))
						# importance_sampling = torch.cat([torch.ones(b, 1, n_a).to(self.device), torch.exp((logprobs_batch[:, 1:, :].to(self.device) - gen_policy_logprobs[:, :-1, :].to(self.device)))], dim=1) * agent_masks_batch.to(self.device)

						# print(rewards.shape, importance_sampling.shape)
						# rewards = rewards * importance_sampling

						print("rewards")
						print(rewards[0])
						
						# rewards = rewards.transpose(1, 2).cpu() * temporal_weights_final

					
					if self.experiment_type == "ATRR_temporal":
						rewards = rewards.unsqueeze(-1).repeat(1, 1, self.num_agents)

					if self.experiment_type == "ATRR_temporal_v2":
						mask_value = torch.tensor(torch.finfo(torch.float).min, dtype=torch.float)
						# void weightage to timesteps after multi-agent system death
						temporal_weightage = F.softmax(torch.where(team_mask_batch.bool().unsqueeze(-1).repeat(1, 1, self.num_agents), rewards.cpu(), mask_value), dim=-2) # batch, timesteps, num_agents
						rewards = episodic_reward_batch.reshape(-1, 1, 1) * temporal_weightage

			elif "STAS" in self.experiment_type:

				with torch.no_grad():
					rewards = self.reward_model( 
						ally_state_batch.permute(0, 2, 1, 3).to(self.device), 
						enemy_state_batch.permute(0, 2, 1, 3).to(self.device), 
						actions_batch.long().permute(0, 2, 1).to(self.device), 
						episode_len_batch.long().to(self.device),
						agent_masks_batch.to(self.device),
						)

					rewards = rewards.transpose(1, 2)


			return rewards.cpu()
			

	def update_reward_model(self, sample):
		# sample episodes from replay buffer
		ally_obs_batch, enemy_obs_batch, actions_batch, logprobs_batch, reward_batch, team_mask_batch, agent_masks_batch, episode_len_batch = sample
		# convert numpy array to tensor
		ally_obs_batch = torch.from_numpy(ally_obs_batch).float()
		enemy_obs_batch = torch.from_numpy(enemy_obs_batch).float()
		actions_batch = torch.from_numpy(actions_batch)
		logprobs_batch = torch.from_numpy(logprobs_batch).float() # same as current one_hot_actions
		reward_batch = torch.from_numpy(reward_batch).float()
		episodic_reward_batch = reward_batch.sum(dim=1)
		team_mask_batch = torch.from_numpy(team_mask_batch).float()
		agent_masks_batch = torch.from_numpy(agent_masks_batch).float()
		episode_len_batch = torch.from_numpy(episode_len_batch).long()

		if self.norm_rewards:
			shape = episodic_reward_batch.shape
			episodic_reward_batch = self.reward_normalizer(episodic_reward_batch.view(-1), None).view(shape)

		
		if "AREL" in self.experiment_type:
			rewards, temporal_weights, agent_weights, temporal_scores, agent_scores = self.reward_model(
				ally_obs_batch.permute(0, 2, 1, 3).to(self.device), 
				enemy_obs_batch.permute(0, 2, 1, 3).to(self.device), 
				actions_batch.permute(0, 2, 1).to(self.device), 
				episodic_reward_batch.to(self.device),
				team_masks=team_mask_batch.to(self.device),
				agent_masks=agent_masks_batch.to(self.device),
				)


			# shape = reward_time_wise.shape
			# reward_copy = copy.deepcopy(reward_time_wise.detach())
			# reward_copy[team_mask_batch.view(*shape) == 0.0] = 0.0 
			# reward_mean = (reward_copy.sum(dim=-1)/team_mask_batch.to(self.device).sum(dim=-1)).unsqueeze(-1)
			# print("Check for NaN")
			# print(torch.isnan(reward_time_wise).any())
			# print(torch.isnan(reward_mean).any())
			# reward_var = (reward_time_wise - reward_mean)**2
			# reward_var[team_mask_batch.view(*shape) == 0.0] = 0.0
			# reward_var = reward_var.sum() / team_mask_batch.sum()

			reward_var = torch.tensor([-1])
			reward_loss = F.huber_loss((rewards.reshape(episodic_reward_batch.shape[0], -1)).sum(dim=-1), episodic_reward_batch.to(self.device)) #+ self.variance_loss_coeff*reward_var

		elif "ATRR" in self.experiment_type:
			
			returns, rewards, temporal_weights, agent_weights, temporal_scores, agent_scores, action_prediction = self.reward_model(
				ally_obs_batch.permute(0, 2, 1, 3).to(self.device), 
				enemy_obs_batch.permute(0, 2, 1, 3).to(self.device), 
				actions_batch.permute(0, 2, 1).to(self.device), 
				episodic_reward_batch.to(self.device),
				logprobs=logprobs_batch.to(self.device),
				team_masks=team_mask_batch.to(self.device),
				agent_masks=agent_masks_batch.to(self.device),
				train=True,
				)

			# temporal_weights = temporal_weights.cpu().mean(dim=0).sum(dim=1) / (agent_masks_batch.permute(0, 2, 1).sum(dim=1).unsqueeze(-1)+1e-5)
			# agent_weights = agent_weights.cpu().mean(dim=0)
			entropy_temporal_weights = -torch.sum(temporal_weights * torch.log(torch.clamp(temporal_weights, 1e-10, 1.0)))/((agent_masks_batch.sum()+1e-5)*self.reward_depth)
			entropy_agent_weights = -torch.sum(agent_weights * torch.log(torch.clamp(agent_weights, 1e-10, 1.0)))/((agent_masks_batch.sum()+1e-5)*self.reward_depth)


			if self.version == "agent_temporal_attn_weights":
				b, t, _, e = ally_obs_batch.shape
				# reward_loss = F.mse_loss(rewards.reshape(actions_batch.shape[0], -1).sum(dim=-1), episodic_reward_batch.to(self.device)) #+ 5e-2*(F.mse_loss(state_prediction, state_target, reduction='none') * team_mask_batch.unsqueeze(-1).to(self.device)).sum() / team_mask_batch.sum() #+ 1e-2*(self.classification_loss(action_prediction.reshape(-1, self.num_actions), actions_batch.long().permute(0, 2, 1).reshape(-1).to(self.device)) * agent_masks_batch.reshape(-1).to(self.device)).sum() / (agent_masks_batch.to(self.device).sum() + 1e-5) #+ 1e-4 * entropy_temporal_weights + 1e-4 * entropy_agent_weights
				reward_prediction_loss = F.mse_loss(returns.reshape(actions_batch.shape[0], -1).sum(dim=-1), episodic_reward_batch.to(self.device))
				dynamic_loss = (self.classification_loss(action_prediction.reshape(-1, self.num_actions), actions_batch.long().permute(0, 2, 1).reshape(-1).to(self.device)) * agent_masks_batch.reshape(-1).to(self.device)).sum() / (agent_masks_batch.to(self.device).sum() + 1e-5)
				reward_loss = reward_prediction_loss + self.dynamic_loss_coeffecient * dynamic_loss
				# reward_loss = torch.mean(torch.log(torch.cosh(rewards.squeeze(-1) - episodic_reward_batch.to(self.device))))
				# reward_loss = F.huber_loss(rewards.squeeze(-1).sum(dim=-1), episodic_reward_batch.to(self.device))
			else:
				batch, timesteps = ally_obs_batch.shape[:2]
				reward_loss = F.huber_loss(rewards.reshape(batch, -1).sum(dim=-1), episodic_reward_batch.to(self.device)) #+ self.temporal_score_coefficient * (temporal_scores**2).sum() + self.agent_score_coefficient * (agent_scores**2).sum()
				# weights = rewards / (episodic_reward_batch.reshape(batch, 1, 1).to(self.device)+1e-5)
				# constraint_loss_1 = (((torch.ones((batch, timesteps)).to(self.device) - weights.sum(dim=-1))**2) * team_mask_batch.to(self.device)).sum() / team_mask_batch.to(self.device).sum()
				# reward_loss += constraint_loss_1

		elif "STAS" in self.experiment_type:
			
			rewards = self.reward_model(
				ally_obs_batch.permute(0, 2, 1, 3).to(self.device), 
				enemy_obs_batch.permute(0, 2, 1, 3).to(self.device), 
				actions_batch.long().permute(0, 2, 1).to(self.device), 
				episode_len_batch.long().to(self.device),
				# (agent_masks_batch.sum(dim=1)-1).long().to(self.device),
				agent_masks_batch.to(self.device),
				)

			rewards = rewards.transpose(1, 2) * agent_masks_batch.to(self.device)

			reward_loss = F.mse_loss(rewards.reshape(ally_obs_batch.shape[0], -1).sum(dim=-1), episodic_reward_batch.to(self.device))

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
			return reward_loss.item(), reward_prediction_loss.item(), dynamic_loss.item(), entropy_temporal_weights.item(), entropy_agent_weights.item(), grad_norm_value_reward.item()
		elif "STAS" in self.experiment_type:
			return reward_loss.item(), grad_norm_value_reward.item()


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
			ally_states, enemy_states, hidden_state_v, global_obs, local_obs, hidden_state_actor, logprobs_old, \
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

			if "StarCraft" in self.environment:
				ally_states = ally_states.to(self.device)
				enemy_states = enemy_states.to(self.device)
			elif "Alice_and_Bob" in self.environment:
				global_obs = global_obs.to(self.device)

			values, h_v = self.critic_network_v(
												local_obs.to(self.device),
												global_obs,
												ally_states,
												enemy_states,
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
			v_value_loss_batch += critic_v_loss.item()
			grad_norm_value_v_batch += grad_norm_value_v.item()
			
			

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