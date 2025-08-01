"""
This file defines the PPOAgent class, which encapsulates the core logic for a
Multi-Agent Proximal Policy Optimization (MAPPO) agent.

The PPOAgent class is responsible for:
- Initializing the actor (policy) and critic (value) networks.
- Initializing the on-policy (RolloutBuffer) and off-policy (RewardRolloutBuffer) buffers.
- Selecting actions based on the current policy.
- Orchestrating the update steps for the policy, critic, and, if applicable, the
  credit assignment model (TAR², AREL, STAS).
- Handling all model-specific logic for training and inference.
"""
import numpy as np
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from model import Policy, Value, PopArt
from utils import RolloutBuffer, RewardRolloutBuffer, torch_nanmin

class PPOAgent:
	"""
	A Multi-Agent Proximal Policy Optimization (MAPPO) agent.
	This class brings together the policy/critic networks, buffers, and credit
	assignment models to implement the full training algorithm.
	"""

	def __init__(self, dictionary, comet_ml):
		"""
		Initializes the agent, including networks, optimizers, and buffers.

		Args:
			dictionary (dict): A dictionary containing all hyperparameters and configuration settings.
			comet_ml: A comet.ml experiment object for logging, or None.
		"""
		# --- Unpack all hyperparameters from the dictionary ---
		# Environment Setup
		self.environment = dictionary["environment"]
		self.experiment_type = dictionary["experiment_type"]
		self.env_name = dictionary["env"]
		self.num_agents = dictionary["num_agents"]
		self.num_actions = dictionary["num_actions"]
		self.max_time_steps = dictionary["max_time_steps"]

		# Training setup
		self.n_epochs = dictionary["n_epochs"]
		self.device = torch.device("cuda" if dictionary["device"] == "gpu" and torch.cuda.is_available() else "cpu")
		self.scheduler_need = dictionary["scheduler_need"]

		# Model Setup
		if "StarCraft" in self.environment:
			self.ally_observation_shape = dictionary["ally_observation_shape"]
			self.num_enemies = dictionary["num_enemies"]
			self.enemy_observation_shape = dictionary["enemy_observation_shape"]
			self.global_observation_shape = None
		elif self.environment == "GFootball":
			self.global_observation_shape = dictionary["global_observation_shape"]
			self.ally_observation_shape = dictionary["ally_observation_shape"]
			self.num_enemies = None
			self.enemy_observation_shape = None
		self.local_observation_shape = dictionary["local_observation_shape"]

		# Critic setup
		self.norm_returns_v = dictionary["norm_returns_v"]
		self.value_clip = dictionary["value_clip"]
		self.grad_clip_critic_v = dictionary["grad_clip_critic_v"]
		self.enable_grad_clip_critic_v = dictionary["enable_grad_clip_critic_v"]
		
		# Actor setup
		self.gamma = dictionary["gamma"]
		self.entropy_pen = dictionary["entropy_pen"]
		self.gae_lambda = dictionary["gae_lambda"]
		self.policy_clip = dictionary["policy_clip"]
		self.norm_adv = dictionary["norm_adv"]
		self.grad_clip_actor = dictionary["grad_clip_actor"]
		self.enable_grad_clip_actor = dictionary["enable_grad_clip_actor"]
		self.entropy_pen_decay = (dictionary["entropy_pen"] - dictionary["entropy_pen_final"])/dictionary["entropy_pen_steps"]

		# Reward setup
		self.norm_rewards = dictionary["norm_rewards"]
		self.enable_reward_grad_clip = dictionary["enable_reward_grad_clip"]
		self.reward_grad_clip_value = dictionary["reward_grad_clip_value"]
		self.dynamic_loss_coeffecient = dictionary["dynamic_loss_coeffecient"]
		self.variance_loss_coeff = dictionary["variance_loss_coeff"]

		# Arguments for the Critic (Value) network
		critic_args = {
			'environment': self.environment,
			'use_recurrent_critic': dictionary['use_recurrent_critic'],
			'global_observation_input_dim': self.global_observation_shape,
			'ally_obs_input_dim': self.ally_observation_shape,
			'enemy_obs_input_dim': self.enemy_observation_shape,
			'num_agents': self.num_agents,
			'num_enemies': self.num_enemies,
			'num_actions': self.num_actions,
			'rnn_num_layers': dictionary['rnn_num_layers_v'],
			'comp_emb_shape': dictionary['v_comp_emb_shape'],
			'device': self.device
		}

		# Arguments for the Actor (Policy) network
		policy_args = {
			'use_recurrent_policy': dictionary['use_recurrent_policy'],
			'obs_input_dim': self.local_observation_shape,
			'num_actions': self.num_actions,
			'num_agents': self.num_agents,
			'rnn_num_layers': dictionary['rnn_num_layers_actor'],
			'rnn_hidden_actor': dictionary['rnn_hidden_actor'],
			'device': self.device
		}
		
		# Arguments for the on-policy RolloutBuffer
		buffer_args = {
			'environment': self.environment,
			'experiment_type': dictionary['experiment_type'],
			'num_episodes': dictionary['ppo_eps_elapse_update_freq'],
			'max_time_steps': self.max_time_steps,
			'num_agents': self.num_agents,
			'num_enemies': self.num_enemies,
			'ally_state_shape': self.ally_observation_shape,
			'enemy_state_shape': self.enemy_observation_shape,
			'local_obs_shape': self.local_observation_shape,
			'global_obs_shape': self.global_observation_shape,
			'rnn_num_layers_actor': dictionary['rnn_num_layers_actor'],
			'actor_hidden_state': dictionary['rnn_hidden_actor'],
			'rnn_num_layers_v': dictionary['rnn_num_layers_v'],
			'v_hidden_state': dictionary['rnn_hidden_v'],
			'num_actions': self.num_actions,
			'data_chunk_length': dictionary['data_chunk_length'],
			'norm_returns_v': self.norm_returns_v,
			'clamp_rewards': dictionary['clamp_rewards'],
			'clamp_rewards_value_min': dictionary['clamp_rewards_value_min'],
			'clamp_rewards_value_max': dictionary['clamp_rewards_value_max'],
			'gae_lambda': self.gae_lambda,
			'gamma': self.gamma
		}

		# --- Initialize Networks ---
		# Critic Network with optional PopArt normalization for value targets
		if self.norm_returns_v:
			self.V_PopArt = PopArt(input_shape=1, num_agents=self.num_agents, device=self.device)
		else:
			self.V_PopArt = None

		self.critic_network_v = Value(**critic_args).to(self.device)
		
		# Policy Network (Actor)
		self.policy_network = Policy(**policy_args).to(self.device)

		# --- Initialize Buffers ---
		self.buffer = RolloutBuffer(**buffer_args)

		# --- Initialize Optimizers ---
		self.v_critic_optimizer = optim.AdamW(self.critic_network_v.parameters(), lr=dictionary["v_value_lr"], weight_decay=dictionary["v_weight_decay"], eps=1e-05)
		self.policy_optimizer = optim.AdamW(self.policy_network.parameters(), lr=dictionary["policy_lr"], weight_decay=dictionary["policy_weight_decay"], eps=1e-05)

		if dictionary["load_models"]:
			# For CPU
			if torch.cuda.is_available() is False:
				self.critic_network_v.load_state_dict(torch.load(dictionary["model_path_v_value"], map_location=torch.device('cpu')))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"], map_location=torch.device('cpu')))
			# For GPU
			else:
				self.critic_network_v.load_state_dict(torch.load(dictionary["model_path_v_value"]))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"]))


		if self.scheduler_need:
			self.scheduler_policy = optim.lr_scheduler.MultiStepLR(self.policy_optimizer, milestones=[1000, 20000], gamma=0.1)
			self.scheduler_v_critic = optim.lr_scheduler.MultiStepLR(self.v_critic_optimizer, milestones=[1000, 20000], gamma=0.1)

		# --- Initialize Credit Assignment Model (if used) ---
		self.use_reward_model = dictionary["use_reward_model"]
		if self.use_reward_model:
			reward_buffer_args = {
				'environment': self.environment,
				'capacity': dictionary['replay_buffer_size'],
				'max_episode_len': self.max_time_steps,
				'num_agents': self.num_agents,
				'num_enemies': self.num_enemies,
				'ally_obs_shape': self.ally_observation_shape,
				'enemy_obs_shape': self.enemy_observation_shape,
				'local_obs_shape': self.local_observation_shape,
				'rnn_num_layers_actor': dictionary['rnn_num_layers_actor'],
				'actor_hidden_state': dictionary['rnn_hidden_actor'],
				'action_shape': self.num_actions,
				'device': self.device
			}
			self.reward_buffer = RewardRolloutBuffer(**reward_buffer_args)
			
			# Dynamically import and initialize the correct reward model based on experiment type
			if "TAR^2" in dictionary["experiment_type"]:
				from TAR2 import TAR2
				reward_model_args = {
                    'environment': self.environment,
                    'ally_obs_shape': self.ally_observation_shape,
                    'enemy_obs_shape': self.enemy_observation_shape,
                    'n_actions': self.num_actions,
                    'emb_dim': dictionary['reward_linear_compression_dim'],
                    'n_heads': dictionary['reward_n_heads'],
                    'n_layer': dictionary['reward_depth'],
                    'seq_length': self.max_time_steps,
                    'n_agents': self.num_agents,
                    'sample_num': 5, # This is hardcoded in your original file
                    'device': self.device,
                    'emb_dropout': 0.0 # This is hardcoded in your original file
                }
				self.reward_model = TAR2.TAR2(**reward_model_args).to(self.device)
			elif "AREL" in dictionary["experiment_type"]:
				from AREL import AREL
				arel_model_args = {
                    'environment': self.environment,
                    'ally_obs_shape': self.ally_observation_shape,
                    'enemy_obs_shape': self.enemy_observation_shape,
                    'action_shape': self.num_actions,
                    'heads': dictionary['reward_n_heads'],
                    'depth': dictionary['reward_depth'],
                    'seq_length': self.max_time_steps,
                    'n_agents': self.num_agents,
                    'n_actions': self.num_actions,
                    'agent': dictionary['reward_agent_attn'],
                    'dropout': dictionary['reward_dropout'],
                    'wide': dictionary['reward_attn_net_wide'],
                    'version': dictionary['version'],
                    'linear_compression_dim': dictionary['reward_linear_compression_dim'],
                    'device': self.device
                }
				self.reward_model = AREL.Time_Agent_Transformer(**arel_model_args).to(self.device)
			elif "STAS" in dictionary["experiment_type"]:
				from STAS import stas
				stas_model_args = {
                    'environment': self.environment,
                    'ally_obs_shape': self.ally_observation_shape,
                    'enemy_obs_shape': self.enemy_observation_shape,
                    'n_actions': self.num_actions,
                    'emb_dim': dictionary['reward_linear_compression_dim'],
                    'n_heads': dictionary['reward_n_heads'],
                    'n_layer': dictionary['reward_depth'],
                    'seq_length': self.max_time_steps,
                    'n_agents': self.num_agents,
                    'sample_num': 5, # This is hardcoded in your original file
                    'device': self.device,
                    'emb_dropout': 0.0 # This is hardcoded in your original file
                }
				self.reward_model = stas.STAS_ML(**stas_model_args).to(self.device)

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

		self.comet_ml = comet_ml

	
	def get_values(self, global_obs, state_allies, state_enemies, actions, rnn_hidden_state_v, indiv_dones):
		"""Gets value estimates from the critic network for the current state."""
		with torch.no_grad():
			indiv_masks = [1-d for d in indiv_dones]
			indiv_masks = torch.FloatTensor(indiv_masks).unsqueeze(0).unsqueeze(0).to(self.device)
			if "StarCraft" in self.environment:
				state_allies = torch.FloatTensor(state_allies).unsqueeze(0).unsqueeze(0).to(self.device)
				state_enemies = torch.FloatTensor(state_enemies).unsqueeze(0).unsqueeze(0).to(self.device)
			elif self.environment == "GFootball":
				global_obs = torch.FloatTensor(global_obs).unsqueeze(0).unsqueeze(0).to(self.device)
			actions = torch.FloatTensor(actions).unsqueeze(0).unsqueeze(0).to(self.device)
			rnn_hidden_state_v = torch.FloatTensor(rnn_hidden_state_v).to(self.device)
			
			Value, rnn_hidden_state_v = self.critic_network_v(global_obs, state_allies, state_enemies, actions, rnn_hidden_state_v)
				
			return Value.squeeze(0).cpu().numpy(), rnn_hidden_state_v.cpu().numpy()


	def get_action(self, state_policy, last_actions, mask_actions, hidden_state, greedy=False):
		"""
		Selects an action for each agent based on the current policy.

		Args:
			state_policy: The local observations for each agent.
			last_actions: The last action taken by each agent.
			mask_actions: A mask of available actions for each agent.
			hidden_state: The recurrent state of the policy network.
			greedy (bool): If True, selects the action with the highest probability (for evaluation).
						   If False, samples from the action distribution (for training).

		Returns:
			tuple: A tuple containing the selected actions, their log probabilities, and the next hidden state.
		"""
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
		"""
		Generates shaped rewards using the credit assignment model.

		Args:
			eval_reward_model (bool): If True, uses only the latest episode for evaluation.
									  If False, uses the full on-policy buffer.

		Returns:
			np.array: An array of shaped rewards for each agent at each timestep.
		"""

		action_prediction = None

		if eval_reward_model:
			latest_sample_index = self.buffer.episode_num
			if "StarCraft" in self.environment:
				ally_state_batch = torch.from_numpy(self.buffer.ally_states[latest_sample_index]).float().unsqueeze(0).permute(0, 2, 1, 3).to(self.device)
				enemy_state_batch = torch.from_numpy(self.buffer.enemy_states[latest_sample_index]).float().unsqueeze(0).permute(0, 2, 1, 3).to(self.device)
			elif "GFootball" in self.environment:
				enemy_state_batch = None
				ally_state_batch = torch.from_numpy(self.buffer.ally_states[latest_sample_index]).float().unsqueeze(0).permute(0, 2, 1, 3).to(self.device)
			actions_batch = torch.from_numpy(self.buffer.actions[latest_sample_index]).long().unsqueeze(0).permute(0, 2, 1).to(self.device)
			team_mask_batch = 1-torch.from_numpy(self.buffer.team_dones[latest_sample_index]).float().unsqueeze(0).to(self.device)
			agent_masks_batch = 1-torch.from_numpy(self.buffer.indiv_dones[latest_sample_index, :-1, :]).float().unsqueeze(0).to(self.device)
			episode_len_batch = torch.from_numpy(self.buffer.episode_length[latest_sample_index, :-1]).long().unsqueeze(0).to(self.device)
			episodic_reward_batch = torch.from_numpy(self.buffer.rewards[latest_sample_index, :, 0]).float().sum(dim=-1).unsqueeze(0).to(self.device)
		else:
			if "StarCraft" in self.environment:	
				ally_state_batch = torch.from_numpy(self.buffer.ally_states).float().permute(0, 2, 1, 3).to(self.device)
				enemy_state_batch = torch.from_numpy(self.buffer.enemy_states).float().permute(0, 2, 1, 3).to(self.device)
			elif "GFootball" in self.environment:
				enemy_state_batch = None
				ally_state_batch = torch.from_numpy(self.buffer.ally_states).float().permute(0, 2, 1, 3).to(self.device)
			actions_batch = torch.from_numpy(self.buffer.actions).long().permute(0, 2, 1).to(self.device)
			team_mask_batch = 1-torch.from_numpy(self.buffer.team_dones[:, :-1]).float().to(self.device)
			agent_masks_batch = 1-torch.from_numpy(self.buffer.indiv_dones[:, :-1, :]).float().to(self.device)
			episode_len_batch = torch.from_numpy(self.buffer.episode_length).long().to(self.device)
			episodic_reward_batch = torch.from_numpy(self.buffer.rewards[:, :, 0]).float().sum(dim=-1).to(self.device)
		
		with torch.no_grad():
			if "AREL" in self.experiment_type:
				with torch.no_grad():
					rewards, temporal_weights, agent_weights,\
					_, _ = self.reward_model(
						ally_state_batch, 
						enemy_state_batch, 
						actions_batch, 
						team_masks=team_mask_batch,
						agent_masks=agent_masks_batch,
						)


			elif "TAR^2" in self.experiment_type:

				with torch.no_grad():
					# Get unnormalized scores from the TAR² model
					rewards, temporal_weights, agent_weights, _, _, _ = self.reward_model(
						ally_state_batch, 
						enemy_state_batch, 
						actions_batch, 
						episode_len_batch,
						agent_masks_batch,
						)

					# USING MIN-MAX NORMALIZATION
					temporal_rewards = (rewards*agent_masks_batch).sum(dim=-1, keepdim=True)
					temporal_rewards_copy = copy.deepcopy(temporal_rewards)
					temporal_rewards_copy[(agent_masks_batch.sum(dim=-1, keepdim=True)>0).int() == 0] = float('nan')
					min_temporal_rewards, _ = torch_nanmin(temporal_rewards_copy, dim=-2, keepdim=True)
					temporal_rewards = (temporal_rewards-min_temporal_rewards) * (agent_masks_batch.sum(dim=-1, keepdim=True)>0).int()
					temporal_weights = temporal_rewards / (temporal_rewards.sum(dim=1, keepdim=True) + 1e-5)

					agent_rewards_copy = copy.deepcopy(rewards)
					agent_rewards_copy[agent_masks_batch.int() == 0] = float('nan')
					min_agent_rewards, _ = torch_nanmin(agent_rewards_copy, dim=-1, keepdim=True)
					agent_rewards = (rewards-min_agent_rewards)*agent_masks_batch
					agent_weights = agent_rewards / (agent_rewards.sum(dim=-1, keepdim=True) + 1e-5)

					# episodic_rewards = torch.from_numpy(self.buffer.rewards[:, :, 0]).sum(dim=1, keepdim=True).unsqueeze(-1)
					
					return ((temporal_weights*agent_weights).cpu()*episodic_reward_batch.unsqueeze(-1)).numpy()

			elif "STAS" in self.experiment_type:

				with torch.no_grad():
					# Get rewards directly from the baseline model
					rewards = self.reward_model( 
						ally_state_batch, 
						enemy_state_batch,
						actions_batch, 
						episode_len_batch,
						agent_masks_batch,
						)

					rewards = rewards.transpose(1, 2)

			return (rewards*agent_masks_batch).cpu().numpy()
			

	def update_reward_model(self, sample):
		"""Performs a single update step for the credit assignment model."""
		# sample episodes from replay buffer
		if "StarCraft" in self.environment:
			ally_obs_batch, enemy_obs_batch, local_obs_batch, actions_batch, last_actions_batch, action_masks_batch, hidden_state_actor_batch, logprobs_old_batch, reward_batch, team_mask_batch, agent_masks_batch, episode_len_batch = sample
		elif "GFootball" in self.environment:
			ally_obs_batch, local_obs_batch, actions_batch, last_actions_batch, action_masks_batch, hidden_state_actor_batch, logprobs_old_batch, reward_batch, team_mask_batch, agent_masks_batch, episode_len_batch = sample
		
		# convert numpy array to tensor
		if "StarCraft" in self.environment:
			ally_obs_batch = torch.from_numpy(ally_obs_batch).float().permute(0, 2, 1, 3).to(self.device)
			enemy_obs_batch = torch.from_numpy(enemy_obs_batch).float().permute(0, 2, 1, 3).to(self.device)
		else:
			enemy_obs_batch = None
			ally_obs_batch = torch.from_numpy(ally_obs_batch).float().permute(0, 2, 1, 3).to(self.device)
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

		if self.norm_rewards:
			shape = episodic_reward_batch.shape
			episodic_reward_batch = self.reward_normalizer(episodic_reward_batch.view(-1), None).view(shape)
		
		if "AREL" in self.experiment_type:
			rewards, temporal_weights, agent_weights, _, _ = self.reward_model(
				ally_obs_batch, 
				enemy_obs_batch, 
				actions_batch, 
				episodic_reward_batch,
				team_masks=team_mask_batch,
				agent_masks=agent_masks_batch,
				)


			rewards_mean = rewards.sum(dim=1, keepdims=True) / agent_masks_batch.sum(dim=1, keepdims=True)
			rewards_var = ((rewards - rewards_mean)**2).sum() / agent_masks_batch.sum()
			reward_loss = F.huber_loss((rewards.reshape(episodic_reward_batch.shape[0], -1)).sum(dim=-1), episodic_reward_batch.to(self.device)) - self.variance_loss_coeff*rewards_var

		elif "TAR^2" in self.experiment_type:
			# Get model outputs
			scores, temporal_weights, agent_weights, _, _, action_prediction = self.reward_model(
				ally_obs_batch, 
				enemy_obs_batch, 
				actions_batch, 
				episode_len_batch,
				agent_masks_batch,
				)

			entropy_temporal_weights = -torch.sum(temporal_weights * torch.log(torch.clamp(temporal_weights, 1e-10, 1.0)))/((agent_masks_batch.sum()+1e-5)*self.reward_depth)
			entropy_agent_weights = -torch.sum(agent_weights * torch.log(torch.clamp(agent_weights, 1e-10, 1.0)))/((agent_masks_batch.sum()+1e-5)*self.reward_depth)
			
			# Calculate the composite loss (reward regression + inverse dynamics)
			total_scores = scores.reshape(actions_batch.shape[0], -1).sum(dim=-1)  # Sum all c_i,t
			reward_prediction_loss = F.huber_loss(total_scores, episodic_reward_batch)
			dynamic_loss = self.dynamic_loss_coeffecient * (self.classification_loss(action_prediction.reshape(-1, self.num_actions), actions_batch.long().reshape(-1)) * agent_masks_batch.reshape(-1)).sum() / (agent_masks_batch.sum() + 1e-5)
			reward_loss = reward_prediction_loss + dynamic_loss
			
		elif "STAS" in self.experiment_type:
			
			rewards = self.reward_model(
				ally_obs_batch, 
				enemy_obs_batch, 
				actions_batch, 
				episode_len_batch,
				agent_masks_batch,
				)

			rewards = rewards.transpose(1, 2) * agent_masks_batch

			reward_loss = F.huber_loss(rewards.reshape(actions_batch.shape[0], -1).sum(dim=-1), episodic_reward_batch)

		self.reward_optimizer.zero_grad()
		reward_loss.backward()
		if self.enable_reward_grad_clip:
			grad_norm_value_reward = torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.reward_grad_clip_value)
		else:
			total_norm = 0
			for _, p in self.reward_model.named_parameters():
				if p.requires_grad is False or p.grad is None:
					continue
				param_norm = p.grad.detach().data.norm(2)
				total_norm += param_norm.item() ** 2
			grad_norm_value_reward = torch.tensor([total_norm ** 0.5])
		self.reward_optimizer.step()

		if "AREL" in self.experiment_type:
			return reward_loss.item(), rewards_var.item(), grad_norm_value_reward.item()
		elif "TAR^2" in self.experiment_type:
			return reward_loss.item(), reward_prediction_loss.item(), dynamic_loss.item(), entropy_temporal_weights.item(), entropy_agent_weights.item(), grad_norm_value_reward.item()#, inverse_dynamic_loss.item(), grad_norm_inverse_dynamics.item()
		elif "STAS" in self.experiment_type:
			return reward_loss.item(), grad_norm_value_reward.item()


	def plot(self, episode):
		
		self.comet_ml.log_metric('Policy_Loss',self.plotting_dict["policy_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_Policy',self.plotting_dict["grad_norm_policy"],episode)
		self.comet_ml.log_metric('Entropy',self.plotting_dict["entropy"],episode)

		self.comet_ml.log_metric('V_Value_Loss',self.plotting_dict["v_value_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_V_Value',self.plotting_dict["grad_norm_value_v"],episode)
		

	def update_parameters(self):
		if self.entropy_pen - self.entropy_pen_decay > self.entropy_pen_final:
			self.entropy_pen -= self.entropy_pen_decay


	def update(self, episode):
		"""
		Performs the main MAPPO update for the actor and critic networks.
		This method is called once per PPO update cycle (e.g., every 10 episodes).
		"""
		v_value_loss_batch = 0
		policy_loss_batch = 0
		entropy_batch = 0
		grad_norm_value_v_batch = 0
		grad_norm_policy_batch = 0

		# First, calculate advantages and value targets from the collected rollouts
		self.buffer.calculate_targets(self.V_PopArt)

		# Perform PPO updates for a fixed number of epochs over the collected data
		# Optimize policy for n epochs
		for _ in range(self.n_epochs):

			# Sample a batch of data chunks for recurrent policy update
			ally_states, enemy_states, hidden_state_v, global_obs, local_obs, hidden_state_actor, logprobs_old, \
			last_actions, actions, action_masks, agent_masks, _, values_old, target_values, advantage  = self.buffer.sample_recurrent_policy()
			
			# Normalize advantages if configured
			if self.norm_adv:
				shape = advantage.shape

				advantage_copy = copy.deepcopy(advantage)
				advantage_copy[agent_masks.view(*shape) == 0.0] = float('nan')
				advantage_mean = torch.nanmean(advantage_copy)
				advantage_std = torch.from_numpy(np.array(np.nanstd(advantage_copy.cpu().numpy()))).float()

				advantage = ((advantage - advantage_mean) / (advantage_std + 1e-5))*agent_masks.view(*shape)

			# print("*"*10, "Normalized Advantages || Mean:", advantage_mean, " || STD:", advantage_std, "*"*10)
			# print(advantage[0, :, 0])

			values_old *= agent_masks

			target_shape = values_old.shape

			if "StarCraft" in self.environment:
				ally_states = ally_states.to(self.device)
				enemy_states = enemy_states.to(self.device)
			elif "GFootball" in self.environment:
				global_obs = global_obs.to(self.device)

			# --- Critic Update ---
			values, _ = self.critic_network_v(
												global_obs,
												ally_states,
												enemy_states,
												actions.to(self.device),
												hidden_state_v.to(self.device),
												)
			
			values = values.reshape(*target_shape)

			values *= agent_masks.to(self.device)
			target_values *= agent_masks

			if self.norm_returns_v:
				targets_shape = target_values.shape
				target_values = (self.V_PopArt(target_values.view(-1), agent_masks.view(-1), train=True).view(targets_shape) * agent_masks.view(targets_shape)).cpu()

			critic_v_loss_1 = F.huber_loss(values, target_values.to(self.device), reduction="sum", delta=10.0) / agent_masks.sum()
			critic_v_loss_2 = F.huber_loss(torch.clamp(values, values_old.to(self.device)-self.value_clip, values_old.to(self.device)+self.value_clip), target_values.to(self.device), reduction="sum", delta=10.0) / agent_masks.sum()
				
			critic_v_loss = torch.max(critic_v_loss_1, critic_v_loss_2)

			print("Agent Mask Sum", agent_masks.sum())
			
			# Perform gradient update for the critic
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


			# --- Actor Update ---
			dists, _ = self.policy_network(
					local_obs.to(self.device),
					last_actions.to(self.device),
					hidden_state_actor.to(self.device),
					action_masks.to(self.device),
					)

			probs = Categorical(dists)
			logprobs = probs.log_prob(actions.to(self.device))
			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp((logprobs - logprobs_old.to(self.device)))
			
			# Finding Surrogate Loss
			surr1 = ratios * advantage.to(self.device) * agent_masks.to(self.device)
			surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage.to(self.device) * agent_masks.to(self.device)

			# final loss of clipped objective PPO
			entropy = -torch.sum(torch.sum(dists*agent_masks.unsqueeze(-1).to(self.device) * torch.log(torch.clamp(dists*agent_masks.unsqueeze(-1).to(self.device), 1e-10,1.0)), dim=-1))/ agent_masks.sum() #(masks.sum()*self.num_agents)
			policy_loss_ = (-torch.min(surr1, surr2).sum())/agent_masks.sum()
			policy_loss = policy_loss_ - self.entropy_pen*entropy

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

		# Clear the on-policy buffer after the update is complete
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
			self.plot(episode)

		del v_value_loss_batch, policy_loss_batch, entropy_batch, grad_norm_value_v_batch, grad_norm_policy_batch
		torch.cuda.empty_cache()