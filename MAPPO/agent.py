import numpy as np
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from model import Policy, V_network, PopArt, InverseDynamicsModel
from utils import RolloutBuffer, RewardRolloutBuffer, RolloutBufferShared, RewardRolloutBufferShared

class PPOAgent:

	def __init__(
		self, 
		dictionary,
		comet_ml,
		):

		# Environment Setup
		self.environment = dictionary["environment"]
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
		self.parallel_training = dictionary["parallel_training"]
		self.num_workers = dictionary["num_workers"]

		self.ppo_eps_elapse_update_freq = dictionary["ppo_eps_elapse_update_freq"]
		self.max_time_steps = dictionary["max_time_steps"]

		# Model Setup
		if "StarCraft" in self.environment:
			self.ally_observation_shape = dictionary["ally_observation_shape"]
			self.num_enemies = dictionary["num_enemies"]
			self.enemy_observation_shape = dictionary["enemy_observation_shape"]

			self.global_observation_shape = None
			self.common_information_observation_shape = None
		elif self.environment == "GFootball":
			self.global_observation_shape = dictionary["global_observation_shape"]
			self.ally_observation_shape = dictionary["ally_observation_shape"]
			self.common_information_observation_shape = dictionary["common_information_observation_shape"]

			self.num_enemies = None
			self.enemy_observation_shape = None

		self.local_observation_shape = dictionary["local_observation_shape"]

		# Reward Model Setup
		self.use_reward_model = dictionary["use_reward_model"]
		self.reward_lr = dictionary["reward_lr"]
		self.variance_loss_coeff = dictionary["variance_loss_coeff"]
		self.enable_reward_grad_clip = dictionary["enable_reward_grad_clip"]
		self.reward_grad_clip_value = dictionary["reward_grad_clip_value"]
		self.reward_n_heads = dictionary["reward_n_heads"]
		self.norm_rewards = dictionary["norm_rewards"]
		self.dynamic_loss_coeffecient = dictionary["dynamic_loss_coeffecient"]
		self.expected_logprob_prediction_loss_coeffecient = dictionary["expected_logprob_prediction_loss_coeffecient"]
		self.temporal_score_coefficient = dictionary["temporal_score_coefficient"]
		self.agent_score_coefficient = dictionary["agent_score_coefficient"]
		self.reward_depth = dictionary["reward_depth"]


		self.use_inverse_dynamics = dictionary["use_inverse_dynamics"]
		self.inverse_dynamics_lr = dictionary["inverse_dynamics_lr"]
		self.inverse_dynamics_weight_decay = dictionary["inverse_dynamics_weight_decay"]
		self.enable_grad_clip_inverse_dynamics = dictionary["enable_grad_clip_inverse_dynamics"]
		self.grad_clip_inverse_dynamics = dictionary["grad_clip_inverse_dynamics"]

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



		if self.parallel_training:
			self.buffer = RolloutBufferShared(
				num_workers=self.num_workers,
				environment=self.environment,
				experiment_type=self.experiment_type,
				num_episodes=self.ppo_eps_elapse_update_freq, 
				max_time_steps=self.max_time_steps, 
				num_agents=self.num_agents, 
				num_enemies=self.num_enemies,
				ally_state_shape=self.ally_observation_shape, 
				enemy_state_shape=self.enemy_observation_shape, 
				local_obs_shape=self.local_observation_shape, 
				global_obs_shape=self.global_observation_shape,
				common_information_obs_shape=self.common_information_observation_shape,
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
		else:
			self.buffer = RolloutBuffer(
				environment=self.environment,
				experiment_type=self.experiment_type,
				num_episodes=self.ppo_eps_elapse_update_freq, 
				max_time_steps=self.max_time_steps, 
				num_agents=self.num_agents, 
				num_enemies=self.num_enemies,
				ally_state_shape=self.ally_observation_shape, 
				enemy_state_shape=self.enemy_observation_shape, 
				local_obs_shape=self.local_observation_shape, 
				global_obs_shape=self.global_observation_shape,
				common_information_obs_shape=self.common_information_observation_shape,
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
			if self.parallel_training:
				self.reward_buffer = RewardRolloutBufferShared(
					num_workers=self.num_workers,
					environment = dictionary["environment"],
					capacity = dictionary["replay_buffer_size"],
					max_episode_len = self.max_time_steps,
					num_agents = self.num_agents,
					num_enemies=self.num_enemies,
					ally_obs_shape=self.ally_observation_shape,
					enemy_obs_shape=self.enemy_observation_shape,
					local_obs_shape=self.local_observation_shape,
					common_information_obs_shape=self.common_information_observation_shape,
					rnn_num_layers_actor=self.rnn_num_layers_actor,
					actor_hidden_state=self.rnn_hidden_actor,
					action_shape = self.num_actions,
					device = self.device,
					)
			else:
				self.reward_buffer = RewardRolloutBuffer(
					environment = dictionary["environment"],
					capacity = dictionary["replay_buffer_size"],
					max_episode_len = self.max_time_steps,
					num_agents = self.num_agents,
					num_enemies=self.num_enemies,
					ally_obs_shape=self.ally_observation_shape,
					enemy_obs_shape=self.enemy_observation_shape,
					local_obs_shape=self.local_observation_shape,
					common_information_obs_shape=self.common_information_observation_shape,
					rnn_num_layers_actor=self.rnn_num_layers_actor,
					actor_hidden_state=self.rnn_hidden_actor,
					action_shape = self.num_actions,
					device = self.device,
					)

			if "AREL" in self.experiment_type:
				from AREL import AREL
				self.reward_model = AREL.Time_Agent_Transformer(
					environment=dictionary["environment"],
					ally_obs_shape=self.ally_observation_shape,
					enemy_obs_shape=self.enemy_observation_shape,
					obs_shape=self.common_information_observation_shape,
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

			elif self.experiment_type in ["TAR^2", "TAR^2_v2", "TAR^2_HindSight"]:
				if self.experiment_type == "TAR^2":
					from TARR import TARR
					self.reward_model = TARR.Time_Agent_Transformer(
						environment=dictionary["environment"],
						ally_obs_shape=self.ally_observation_shape,
						enemy_obs_shape=self.enemy_observation_shape,
						obs_shape=self.common_information_observation_shape,
						action_shape=self.num_actions, 
						heads=dictionary["reward_n_heads"], 
						depth=dictionary["reward_depth"], 
						seq_length=dictionary["max_time_steps"], 
						n_agents=self.num_agents, 
						n_enemies=self.num_enemies,
						agent=dictionary["reward_agent_attn"], 
						dropout=dictionary["reward_dropout"], 
						wide=dictionary["reward_attn_net_wide"],
						linear_compression_dim=dictionary["reward_linear_compression_dim"],
						device=self.device,
						).to(self.device)
				elif self.experiment_type == "TAR^2_v2":
					from TARR_v2 import TARR_v2
					self.reward_model = TARR_v2.TARR(
						environment=dictionary["environment"],
						ally_obs_shape=self.ally_observation_shape,
						enemy_obs_shape=self.enemy_observation_shape, 
						obs_shape=self.common_information_observation_shape,
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
				elif self.experiment_type == "TAR^2_HindSight":
					from TARR_HindSight import TARR
					self.reward_model = TARR.Time_Agent_Transformer(
						environment=dictionary["environment"],
						ally_obs_shape=self.ally_observation_shape,
						enemy_obs_shape=self.enemy_observation_shape,
						obs_shape=self.common_information_observation_shape,
						action_shape=self.num_actions, 
						heads=dictionary["reward_n_heads"], 
						depth=dictionary["reward_depth"], 
						seq_length=dictionary["max_time_steps"], 
						n_agents=self.num_agents, 
						n_enemies=self.num_enemies,
						agent=dictionary["reward_agent_attn"], 
						dropout=dictionary["reward_dropout"], 
						wide=dictionary["reward_attn_net_wide"],
						linear_compression_dim=dictionary["reward_linear_compression_dim"],
						device=self.device,
						).to(self.device)


			elif "STAS" in self.experiment_type:
				from STAS import stas
				self.reward_model = stas.STAS_ML(
					environment=dictionary["environment"],
					ally_obs_shape=self.ally_observation_shape,
					enemy_obs_shape=self.enemy_observation_shape, 
					obs_shape=self.common_information_observation_shape,
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


		if self.use_inverse_dynamics:
			# Inverse Dynamics Model
			self.inverse_dynamic_network = InverseDynamicsModel(
				rnn_hidden_actor=self.rnn_hidden_actor, 
				num_actions=self.num_actions, 
				num_agents=self.num_agents, 
				device=self.device,
			).to(self.device)

			self.inverse_dynamic_optimizer = optim.AdamW(self.inverse_dynamic_network.parameters(), lr=dictionary["inverse_dynamics_lr"], weight_decay=dictionary["inverse_dynamics_weight_decay"], eps=1e-5)
			
			if self.scheduler_need:
				self.scheduler_inverse_dynamics = optim.lr_scheduler.MultiStepLR(self.inverse_dynamic_optimizer, milestones=[10000, 30000], gamma=0.5)

			self.inverse_dynamic_cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")



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

	
	def get_values(self, local_obs, global_obs, state_allies, state_enemies, actions, rnn_hidden_state_v, indiv_dones):
		with torch.no_grad():
			indiv_masks = [1-d for d in indiv_dones]
			indiv_masks = torch.FloatTensor(indiv_masks).unsqueeze(0).unsqueeze(0).to(self.device)
			if "StarCraft" in self.environment:
				state_allies = torch.FloatTensor(state_allies).unsqueeze(0).unsqueeze(0).to(self.device)
				state_enemies = torch.FloatTensor(state_enemies).unsqueeze(0).unsqueeze(0).to(self.device)
			elif self.environment == "GFootball":
				global_obs = torch.FloatTensor(global_obs).unsqueeze(0).unsqueeze(0).to(self.device)
			local_obs = torch.FloatTensor(local_obs).unsqueeze(0).unsqueeze(0).to(self.device)
			actions = torch.FloatTensor(actions).unsqueeze(0).unsqueeze(0).to(self.device)
			rnn_hidden_state_v = torch.FloatTensor(rnn_hidden_state_v).to(self.device)
			
			Value, rnn_hidden_state_v = self.target_critic_network_v(local_obs, global_obs, state_allies, state_enemies, actions, rnn_hidden_state_v, indiv_masks)
				
			return Value.squeeze(0).cpu().numpy(), rnn_hidden_state_v.cpu().numpy()


	def get_values_batch(self, local_obs, global_obs, state_allies, state_enemies, actions, rnn_hidden_state_v, indiv_dones):
		with torch.no_grad():
			num_workers, num_layers, num_agents, hidden_dim = rnn_hidden_state_v.shape
			indiv_masks = 1 - indiv_dones
			indiv_masks = torch.FloatTensor(indiv_masks).unsqueeze(1).to(self.device)
			if "StarCraft" in self.environment:
				state_allies = torch.FloatTensor(state_allies).unsqueeze(1).to(self.device)
				state_enemies = torch.FloatTensor(state_enemies).unsqueeze(1).to(self.device)
			elif self.environment == "GFootball":
				global_obs = torch.FloatTensor(global_obs).unsqueeze(1).to(self.device)
			local_obs = torch.FloatTensor(local_obs).unsqueeze(1).to(self.device)
			actions = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
			rnn_hidden_state_v = torch.FloatTensor(rnn_hidden_state_v).permute(1, 0, 2, 3).reshape(num_layers, num_workers*num_agents, hidden_dim).to(self.device)
			Value, rnn_hidden_state_v = self.target_critic_network_v(local_obs, global_obs, state_allies, state_enemies, actions, rnn_hidden_state_v, indiv_masks)
			return Value.cpu().numpy(), rnn_hidden_state_v.reshape(num_layers, num_workers, num_agents, hidden_dim).permute(1, 0, 2, 3).cpu().numpy()
	
	
	def get_action(self, state_policy, last_actions, mask_actions, hidden_state, greedy=False):
		with torch.no_grad():
			state_policy = torch.FloatTensor(state_policy).unsqueeze(0).unsqueeze(1).to(self.device)
			last_actions = torch.LongTensor(last_actions).unsqueeze(0).unsqueeze(1).to(self.device)
			mask_actions = torch.BoolTensor(mask_actions).unsqueeze(0).unsqueeze(1).to(self.device)
			hidden_state = torch.FloatTensor(hidden_state).to(self.device)

			dists, hidden_state, latent_state = self.policy_network(state_policy, last_actions, hidden_state, mask_actions)
			# dists, hidden_state = actor(state_policy, last_actions, hidden_state, mask_actions)

			if greedy:
				actions = [dist.argmax().detach().cpu().item() for dist in dists.squeeze(0).squeeze(0)]
				action_logprob = None
			else:
				actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists.squeeze(0).squeeze(0)]

				probs = Categorical(dists)
				action_logprob = probs.log_prob(torch.FloatTensor(actions).to(self.device)).cpu().numpy()

			return actions, action_logprob, hidden_state.cpu().numpy(), latent_state.cpu().numpy()


	def get_action_batch(self, state_policy, last_actions, mask_actions, hidden_state, greedy=False):
		with torch.no_grad():
			state_policy = torch.FloatTensor(state_policy).unsqueeze(1).to(self.device)
			last_actions = torch.FloatTensor(last_actions).unsqueeze(1).to(self.device)
			mask_actions = torch.BoolTensor(mask_actions).unsqueeze(1).to(self.device)
			num_workers, num_layers, num_agents, hidden_size = hidden_state.shape
			hidden_state = torch.FloatTensor(hidden_state).permute(1,0,2,3).to(self.device)

			dists, hidden_state, latent_state = self.policy_network(state_policy, last_actions, hidden_state, mask_actions)
			if greedy:
				actions = np.array([dist.argmax(dim=-1).detach().cpu().numpy().tolist() for dist in dists.squeeze(1)])
				action_logprob = None
			else:
				actions = np.array([[Categorical(dist).sample().detach().cpu().item() for dist in dists[worker].squeeze(0)] for worker in range(num_workers)])
				probs = Categorical(dists.squeeze(1))
				action_logprob = probs.log_prob(torch.FloatTensor(actions).to(self.device)).cpu().numpy()

			return actions, action_logprob, hidden_state.reshape(num_layers, num_workers, num_agents, hidden_size).permute(1,0,2,3).cpu().numpy(), latent_state.cpu().numpy()
	

	def should_update_agent(self, episode):
		assert self.parallel_training, "Please call this method only while doing parallel training"
		return (True if (self.buffer.episodes_completely_filled >= self.ppo_eps_elapse_update_freq) else False)


	def reward_model_output(self, eval_reward_model=False):

		action_prediction = None

		if eval_reward_model:
			latest_sample_index = self.buffer.episode_num
			if "StarCraft" in self.environment:
				state_batch = None
				ally_state_batch = torch.from_numpy(self.buffer.ally_states[latest_sample_index]).float().unsqueeze(0).permute(0, 2, 1, 3).to(self.device)
				enemy_state_batch = torch.from_numpy(self.buffer.enemy_states[latest_sample_index]).float().unsqueeze(0).permute(0, 2, 1, 3).to(self.device)
			elif "GFootball" in self.environment:
				enemy_state_batch = None
				ally_state_batch = torch.from_numpy(self.buffer.ally_states[latest_sample_index]).float().unsqueeze(0).permute(0, 2, 1, 3).to(self.device)
				state_batch = torch.from_numpy(self.buffer.common_obs[latest_sample_index]).float().unsqueeze(0).to(self.device)
			actions_batch = torch.from_numpy(self.buffer.actions[latest_sample_index]).long().unsqueeze(0).permute(0, 2, 1).to(self.device)
			logprobs_batch = torch.from_numpy(self.buffer.logprobs[latest_sample_index]).float().unsqueeze(0).to(self.device)
			team_mask_batch = 1-torch.from_numpy(self.buffer.team_dones[latest_sample_index]).float().unsqueeze(0).to(self.device)
			agent_masks_batch = 1-torch.from_numpy(self.buffer.indiv_dones[latest_sample_index, :-1, :]).float().unsqueeze(0).to(self.device)
			episode_len_batch = torch.from_numpy(self.buffer.episode_length[latest_sample_index, :-1]).long().unsqueeze(0).to(self.device)
			episodic_reward_batch = torch.from_numpy(self.buffer.rewards[latest_sample_index, :, 0]).float().sum(dim=-1).unsqueeze(0).to(self.device)
		else:
			if "StarCraft" in self.environment:	
				state_batch = None
				ally_state_batch = torch.from_numpy(self.buffer.ally_states).float().permute(0, 2, 1, 3).to(self.device)
				enemy_state_batch = torch.from_numpy(self.buffer.enemy_states).float().permute(0, 2, 1, 3).to(self.device)
			elif "GFootball" in self.environment:
				enemy_state_batch = None
				ally_state_batch = torch.from_numpy(self.buffer.ally_states).float().permute(0, 2, 1, 3).to(self.device)
				state_batch = torch.from_numpy(self.buffer.common_obs).float().to(self.device)
			actions_batch = torch.from_numpy(self.buffer.actions).long().permute(0, 2, 1).to(self.device)
			logprobs_batch = torch.from_numpy(self.buffer.logprobs).float().to(self.device)
			team_mask_batch = 1-torch.from_numpy(self.buffer.team_dones[:, :-1]).float().to(self.device)
			agent_masks_batch = 1-torch.from_numpy(self.buffer.indiv_dones[:, :-1, :]).float().to(self.device)
			episode_len_batch = torch.from_numpy(self.buffer.episode_length).long().to(self.device)
			episodic_reward_batch = torch.from_numpy(self.buffer.rewards[:, :, 0]).float().sum(dim=-1).to(self.device)
		
		with torch.no_grad():
			if "AREL" in self.experiment_type:
				with torch.no_grad():
					rewards, temporal_weights, agent_weights,\
					temporal_scores, agent_scores = self.reward_model(
						ally_state_batch, 
						enemy_state_batch, 
						state_batch,
						actions_batch, 
						episodic_reward_batch,
						team_masks=team_mask_batch,
						agent_masks=agent_masks_batch,
						)


			elif self.experiment_type == "TAR^2" or self.experiment_type == "TAR^2_HindSight":

				with torch.no_grad():
					rewards, temporal_weights, agent_weights, temporal_scores, agent_scores, action_prediction = self.reward_model(
						ally_state_batch, 
						enemy_state_batch, 
						state_batch,
						actions_batch, 
						episodic_reward_batch,
						agent_masks_batch,
						)

					action_prediction = action_prediction.cpu().numpy()

			elif self.experiment_type == "TAR^2_v2":

				with torch.no_grad():
					rewards, temporal_weights, agent_weights, temporal_scores, agent_scores, action_prediction = self.reward_model(
						ally_state_batch, 
						enemy_state_batch, 
						state_batch,
						actions_batch, 
						episode_len_batch,
						episodic_reward_batch,
						agent_masks_batch,
						)

					action_prediction = action_prediction.cpu().numpy()

			elif "STAS" in self.experiment_type:

				with torch.no_grad():
					rewards = self.reward_model( 
						ally_state_batch, 
						enemy_state_batch,
						state_batch, 
						actions_batch, 
						episode_len_batch,
						agent_masks_batch,
						)

					rewards = rewards.transpose(1, 2)

			print("EXPERIMENT TYPE", self.experiment_type, "REWARDS:-")
			print(rewards[0, :, 0])

			# return (rewards*agent_masks_batch).cpu().numpy()


			temporal_weights = F.softmax((rewards*agent_masks_batch).sum(dim=-1, keepdim=True) - 1e9 * (1-(agent_masks_batch.sum(dim=-1, keepdim=True)>0).int()), dim=-2)
			agent_weights = F.softmax((rewards*agent_masks_batch) - 1e9 * (1-agent_masks_batch), dim=-1) * agent_masks_batch
			episodic_rewards = torch.from_numpy(self.buffer.rewards[:, :, 0]).sum(dim=1, keepdim=True).unsqueeze(-1)

			return ((temporal_weights*agent_weights).cpu()*episodic_rewards).numpy()


			

	def update_reward_model(self, sample):
		# sample episodes from replay buffer
		if "StarCraft" in self.environment:
			ally_obs_batch, enemy_obs_batch, local_obs_batch, actions_batch, last_actions_batch, action_masks_batch, hidden_state_actor_batch, logprobs_old_batch, reward_batch, team_mask_batch, agent_masks_batch, episode_len_batch = sample
		elif "GFootball" in self.environment:
			ally_obs_batch, local_obs_batch, common_obs_batch, actions_batch, last_actions_batch, action_masks_batch, hidden_state_actor_batch, logprobs_old_batch, reward_batch, team_mask_batch, agent_masks_batch, episode_len_batch = sample
		
		# convert numpy array to tensor
		if "StarCraft" in self.environment:
			common_obs_batch = None
			ally_obs_batch = torch.from_numpy(ally_obs_batch).float().permute(0, 2, 1, 3).to(self.device)
			enemy_obs_batch = torch.from_numpy(enemy_obs_batch).float().permute(0, 2, 1, 3).to(self.device)
		else:
			enemy_obs_batch = None
			ally_obs_batch = torch.from_numpy(ally_obs_batch).float().permute(0, 2, 1, 3).to(self.device)
			common_obs_batch = torch.from_numpy(common_obs_batch).float().to(self.device)
		local_obs_batch = torch.from_numpy(local_obs_batch).float().to(self.device)
		actions_batch = torch.from_numpy(actions_batch).long().permute(0, 2, 1).to(self.device)
		last_actions_batch = torch.from_numpy(last_actions_batch).long().to(self.device)
		action_masks_batch = torch.from_numpy(action_masks_batch).to(self.device)
		hidden_state_actor_batch = torch.from_numpy(hidden_state_actor_batch).float().to(self.device)
		logprobs_old_batch = torch.from_numpy(logprobs_old_batch).float() .to(self.device)
		reward_batch = torch.from_numpy(reward_batch).float().to(self.device)
		episodic_reward_batch = reward_batch.sum(dim=1).to(self.device)
		team_mask_batch = torch.from_numpy(team_mask_batch).float().to(self.device)
		agent_masks_batch = torch.from_numpy(agent_masks_batch).float().to(self.device)
		episode_len_batch = torch.from_numpy(episode_len_batch).long().to(self.device)


		inverse_dynamic_loss = torch.tensor([0])
		grad_norm_inverse_dynamics = torch.tensor([0])
		if self.use_inverse_dynamics:
			b, _, _, _ = ally_obs_batch.shape
			data_chunks = self.max_time_steps // self.data_chunk_length

			with torch.no_grad():
				_, _, latent_state_actor = self.policy_network(
					local_obs_batch.to(self.device).reshape(b*data_chunks, self.data_chunk_length, self.num_agents, -1),
					last_actions_batch.to(self.device).reshape(b*data_chunks, self.data_chunk_length, self.num_agents),
					hidden_state_actor_batch.to(self.device).reshape(b*data_chunks, self.data_chunk_length, self.rnn_num_layers_actor, self.num_agents, -1)[:, 0, :, :, :].permute(1, 0, 2, 3).reshape(self.rnn_num_layers_actor, b*data_chunks*self.num_agents, -1),
					action_masks_batch.to(self.device).bool().reshape(b*data_chunks, self.data_chunk_length, self.num_agents, -1),
					)

			latent_state_actor = latent_state_actor.reshape(b, data_chunks*self.data_chunk_length, self.num_agents, -1)

			b, t, n_a, _ = latent_state_actor.shape
			upper_triangular_matrix = torch.triu(torch.ones(b*n_a, t, t)).reshape(b, n_a, t, t).permute(0, 2, 3, 1).to(self.device)

			actions = actions_batch.permute(0, 2, 1).unsqueeze(-2).repeat(1, 1, latent_state_actor.shape[1], 1).float() * agent_masks_batch.unsqueeze(1) * upper_triangular_matrix
			action_prediction = self.inverse_dynamic_network(latent_state_actor, latent_state_actor, agent_masks_batch) * (agent_masks_batch.unsqueeze(1) * upper_triangular_matrix).unsqueeze(-1)

			weight = (1.0 / ((agent_masks_batch.unsqueeze(1) * upper_triangular_matrix).sum(dim=-2, keepdim=True) + 1e-5)).repeat(1, 1, t, 1)
			inverse_dynamic_loss = (self.inverse_dynamic_cross_entropy_loss(action_prediction.reshape(-1, self.num_actions), actions.reshape(-1).long()).reshape(b, t, t, n_a) * agent_masks_batch.unsqueeze(1) * upper_triangular_matrix * weight).sum() / agent_masks_batch.sum()

			self.inverse_dynamic_optimizer.zero_grad()
			inverse_dynamic_loss.backward()
			if self.enable_grad_clip_inverse_dynamics:
				grad_norm_inverse_dynamics = torch.nn.utils.clip_grad_norm_(self.inverse_dynamic_network.parameters(), self.grad_clip_inverse_dynamics)
			else:
				total_norm = 0
				for p in self.inverse_dynamic_network.parameters():
					if p.grad is None:
						continue
					param_norm = p.grad.detach().data.norm(2)
					total_norm += param_norm.item() ** 2
				grad_norm_inverse_dynamics = torch.tensor([total_norm ** 0.5])
			self.inverse_dynamic_optimizer.step()


		if self.norm_rewards:
			shape = episodic_reward_batch.shape
			episodic_reward_batch = self.reward_normalizer(episodic_reward_batch.view(-1), None).view(shape)

		
		if "AREL" in self.experiment_type:
			rewards, temporal_weights, agent_weights, temporal_scores, agent_scores = self.reward_model(
				ally_obs_batch, 
				enemy_obs_batch, 
				common_obs_batch,
				actions_batch, 
				episodic_reward_batch,
				team_masks=team_mask_batch,
				agent_masks=agent_masks_batch,
				)


			rewards_mean = rewards.sum(dim=1, keepdims=True) / agent_masks_batch.sum(dim=1, keepdims=True)
			rewards_var = ((rewards - rewards_mean)**2).sum() / agent_masks_batch.sum()
			reward_loss = F.huber_loss((rewards.reshape(episodic_reward_batch.shape[0], -1)).sum(dim=-1), episodic_reward_batch.to(self.device)) - self.variance_loss_coeff*rewards_var

		elif self.experiment_type in ["TAR^2", "TAR^2_v2", "TAR^2_HindSight"]:
			
			import datetime
			start_time = datetime.datetime.now()

			if self.experiment_type == "TAR^2" or self.experiment_type == "TAR^2_HindSight":
				rewards, temporal_weights, agent_weights, temporal_scores, agent_scores, action_prediction = self.reward_model(
					ally_obs_batch, 
					enemy_obs_batch, 
					common_obs_batch,
					actions_batch, 
					episodic_reward_batch,
					agent_masks_batch,
					)
			elif self.experiment_type == "TAR^2_v2":
				rewards, temporal_weights, agent_weights, temporal_scores, agent_scores, action_prediction = self.reward_model(
					ally_obs_batch, 
					enemy_obs_batch, 
					common_obs_batch,
					actions_batch, 
					episode_len_batch,
					episodic_reward_batch,
					agent_masks_batch,
					)

			end_time = datetime.datetime.now()
			print("ELAPSED TIME", end_time-start_time)

			entropy_temporal_weights = -torch.sum(temporal_weights * torch.log(torch.clamp(temporal_weights, 1e-10, 1.0)))/((agent_masks_batch.sum()+1e-5)*self.reward_depth)
			entropy_agent_weights = -torch.sum(agent_weights * torch.log(torch.clamp(agent_weights, 1e-10, 1.0)))/((agent_masks_batch.sum()+1e-5)*self.reward_depth)

			if "StarCraft" in self.environment:
				b, t, _, e = ally_obs_batch.shape
			elif "GFootball" in self.environment:
				b, t, e = common_obs_batch.shape
			# reward_prediction_loss = F.mse_loss(rewards.reshape(actions_batch.shape[0], -1).sum(dim=-1), episodic_reward_batch)
			
			if self.experiment_type == "TAR^2": # or self.experiment_type == "TAR^2_v2":
				reward_prediction_loss = F.mse_loss(rewards.reshape(actions_batch.shape[0], -1).sum(dim=-1), episodic_reward_batch)
				dynamic_loss = self.dynamic_loss_coeffecient * (self.classification_loss(action_prediction.reshape(-1, self.num_actions), actions_batch.long().permute(0, 2, 1).reshape(-1)) * agent_masks_batch.reshape(-1)).sum() / (agent_masks_batch.sum() + 1e-5)
				
				reward_loss = reward_prediction_loss + dynamic_loss
			# elif self.experiment_type == "TAR^2_v2":
			# 	reward_magnitude_prediction_loss = F.mse_loss(reward_magnitude.reshape(actions_batch.shape[0], -1).sum(dim=-1), torch.abs(episodic_reward_batch))
			# 	episodic_reward_sign = (torch.sign(episodic_reward_batch) + 1).long() # -ve -- class label 0 / 0 -- class label 1 / +ve -- class label 2 
			# 	print("TARGET REWARD SIGN", episodic_reward_sign)
			# 	reward_sign_prediction_loss = self.classification_loss(reward_sign.reshape(actions_batch.shape[0], -1), episodic_reward_sign).mean()
			# 	# print(action_prediction.shape, actions_batch.shape, self.classification_loss(action_prediction.reshape(-1, self.num_actions), actions_batch.long().permute(0, 2, 1).reshape(-1)).shape)
			# 	dynamic_loss = self.dynamic_loss_coeffecient * (self.classification_loss(action_prediction.reshape(-1, self.num_actions), actions_batch.long().reshape(-1)) * agent_masks_batch.reshape(-1)).sum() / (agent_masks_batch.sum() + 1e-5)
				
			# 	reward_prediction_loss = reward_magnitude_prediction_loss + reward_sign_prediction_loss
			# 	reward_loss = reward_magnitude_prediction_loss + reward_sign_prediction_loss + dynamic_loss

			# 	print("REWARD MAGNITUDE PREDICTION LOSS", reward_magnitude_prediction_loss.item(), "REWARD SIGN PREDICTION LOSS", reward_sign_prediction_loss.item(), "DYNAMIC LOSS", dynamic_loss.item())
			elif self.experiment_type == "TAR^2_HindSight" or self.experiment_type == "TAR^2_v2":
				reward_prediction_loss = F.mse_loss(rewards.reshape(actions_batch.shape[0], -1).sum(dim=-1), episodic_reward_batch)

				dynamic_loss = self.dynamic_loss_coeffecient * (self.classification_loss(action_prediction.reshape(-1, self.num_actions), actions_batch.long().reshape(-1)) * agent_masks_batch.reshape(-1)).sum() / (agent_masks_batch.sum() + 1e-5)

				# b, t, n_a = rewards.shape
				# upper_triangular_mask = torch.triu(torch.ones(b*n_a, t, t)).reshape(b, n_a, t, t, 1).to(self.device)
				# actions_batch = actions_batch.unsqueeze(-2).repeat(1, 1, t, 1)
				# dynamic_loss = self.dynamic_loss_coeffecient * (self.classification_loss(action_prediction.reshape(-1, self.num_actions), actions_batch.long().reshape(-1)) * upper_triangular_mask.reshape(-1) * agent_masks_batch.unsqueeze(1).repeat(1, t, 1, 1).permute(0, 3, 1, 2).reshape(-1)).sum() / (agent_masks_batch.unsqueeze(1).repeat(1, t, 1, 1).permute(0, 3, 1, 2).reshape(-1).sum() + 1e-5)
				
				# dynamic_loss = torch.tensor([0.0]).to(self.device)
				
				reward_loss = reward_prediction_loss + dynamic_loss

			
		elif "STAS" in self.experiment_type:
			
			rewards = self.reward_model(
				ally_obs_batch, 
				enemy_obs_batch, 
				common_obs_batch,
				actions_batch, 
				episode_len_batch,
				agent_masks_batch,
				)

			rewards = rewards.transpose(1, 2) * agent_masks_batch

			reward_loss = F.mse_loss(rewards.reshape(actions_batch.shape[0], -1).sum(dim=-1), episodic_reward_batch)

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
			return reward_loss.item(), rewards_var.item(), grad_norm_value_reward.item()
		elif "TAR^2" in self.experiment_type:
			return reward_loss.item(), reward_prediction_loss.item(), dynamic_loss.item(), entropy_temporal_weights.item(), entropy_agent_weights.item(), grad_norm_value_reward.item(), inverse_dynamic_loss.item(), grad_norm_inverse_dynamics.item()
		elif "STAS" in self.experiment_type:
			return reward_loss.item(), grad_norm_value_reward.item()


	def plot(self, masks, episode):
		
		self.comet_ml.log_metric('Policy_Loss',self.plotting_dict["policy_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_Policy',self.plotting_dict["grad_norm_policy"],episode)
		self.comet_ml.log_metric('Entropy',self.plotting_dict["entropy"],episode)

		self.comet_ml.log_metric('V_Value_Loss',self.plotting_dict["v_value_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_V_Value',self.plotting_dict["grad_norm_value_v"],episode)

		# self.comet_ml.log_metric("Inverse Dynamic Loss", self.plotting_dict["inverse_dynamic_loss"], episode)
		# self.comet_ml.log_metric("Grad Norm Inverse Dynamic", self.plotting_dict["grad_norm_inverse_dynamics"], episode)	


	def update_parameters(self):
		if self.entropy_pen - self.entropy_pen_decay > self.entropy_pen_final:
			self.entropy_pen -= self.entropy_pen_decay


	def update(self, episode):

		# finetune update inverse dynamics model
		if self.use_inverse_dynamics:
			latent_state_actor = torch.from_numpy(self.buffer.latent_state_actor).float()
		# 	b, t, n_a, _ = latent_state_actor.shape
			agent_masks = 1-torch.from_numpy(self.buffer.indiv_dones[:, :-1, :]).float()
		# 	upper_triangular_matrix = torch.triu(torch.ones(b*n_a, t, t)).reshape(b, n_a, t, t).permute(0, 2, 3, 1)
			

		# 	actions = torch.from_numpy(self.buffer.actions).unsqueeze(-2).repeat(1, 1, latent_state_actor.shape[1], 1).float() * agent_masks.unsqueeze(1) * upper_triangular_matrix
		# 	action_prediction = self.inverse_dynamic_network(latent_state_actor.to(self.device), latent_state_actor.to(self.device), agent_masks.to(self.device)) * (agent_masks.unsqueeze(1) * upper_triangular_matrix).unsqueeze(-1).to(self.device)

		# 	weight = (1.0 / ((agent_masks.unsqueeze(1) * upper_triangular_matrix).sum(dim=-2, keepdim=True) + 1e-5)).repeat(1, 1, t, 1)
		# 	inverse_dynamic_loss = (self.inverse_dynamic_cross_entropy_loss(action_prediction.reshape(-1, self.num_actions), actions.reshape(-1).long().to(self.device)).reshape(b, t, t, n_a) * agent_masks.unsqueeze(1).to(self.device) * upper_triangular_matrix.to(self.device) * weight.to(self.device)).sum() / agent_masks.to(self.device).sum()

		# 	self.inverse_dynamic_optimizer.zero_grad()
		# 	inverse_dynamic_loss.backward()
		# 	if self.enable_grad_clip_inverse_dynamics:
		# 		grad_norm_inverse_dynamics = torch.nn.utils.clip_grad_norm_(self.inverse_dynamic_network.parameters(), self.grad_clip_inverse_dynamics)
		# 	else:
		# 		total_norm = 0
		# 		for p in self.inverse_dynamic_network.parameters():
		# 			if p.grad is None:
		# 				continue
		# 			param_norm = p.grad.detach().data.norm(2)
		# 			total_norm += param_norm.item() ** 2
		# 		grad_norm_inverse_dynamics = torch.tensor([total_norm ** 0.5])
		# 	self.inverse_dynamic_optimizer.step()

			with torch.no_grad():
				self.buffer.action_prediction = self.inverse_dynamic_network(latent_state_actor.to(self.device), latent_state_actor.to(self.device), agent_masks.to(self.device)).cpu().numpy()

		
		v_value_loss_batch = 0
		policy_loss_batch = 0
		entropy_batch = 0
		grad_norm_value_v_batch = 0
		grad_norm_policy_batch = 0

		# if self.experiment_type == "TAR^2_HindSight" or self.experiment_type == "TAR^2_v2":
		# 	self.buffer.calculate_targets_hindsight(episode, self.V_PopArt)
		# else:
		# 	self.buffer.calculate_targets(episode, self.V_PopArt)
		# self.buffer.calculate_targets(episode, self.V_PopArt)
		self.buffer.calculate_targets_hindsight(episode, self.V_PopArt)

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
			elif "GFootball" in self.environment:
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

			
			dists, _, _ = self.policy_network(
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

			torch.cuda.empty_cache()
			
			

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
		# "inverse_dynamic_loss": inverse_dynamic_loss,
		# "grad_norm_inverse_dynamics": grad_norm_inverse_dynamics
		}
		
		if self.comet_ml is not None:
			self.plot(agent_masks, episode)

		del v_value_loss_batch, policy_loss_batch, entropy_batch, grad_norm_value_v_batch, grad_norm_policy_batch
		torch.cuda.empty_cache()