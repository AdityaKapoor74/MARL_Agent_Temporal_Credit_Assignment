"""
This file contains the core implementation of the Temporal-Agent Reward Redistribution (TAR²) model.

The main components are:
1.  ShapelyAttention: A module that uses a multi-head attention mechanism inspired by
	Shapley values to compute agent-specific contributions within a timestep.
2.  TAR2: The main sequence-to-sequence model that processes trajectory data. It uses a
	dual temporal-agent transformer architecture to produce unnormalized credit scores.
	It is regularized by an auxiliary inverse dynamics model and uses final-state
	conditioning to produce a stable learning signal.
"""
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from .modules import EncoderLayer, init_model, MultiAgentAttention

class ShapelyAttention(nn.Module):
	"""
	Approximates Shapley values for agent importance using multi-head attention.

	This module computes the marginal contribution of each agent by attending over
	randomly sampled coalitions of agents. This is used as the agent-axis attention
	mechanism within the main TAR2 transformer body.
	"""
	def __init__(self, emb_dim, n_heads, n_agents, sample_num, device, dropout=0.0):
		super().__init__()
		self.emb_dim = emb_dim
		self.device = device
		self.n_agents = n_agents
		self.sample_num = sample_num
		# The core multi-agent attention mechanism
		self.phi = MultiAgentAttention(emb_dim, n_heads, n_agents, dropout, device)
		self.agent_embedding = nn.Embedding(self.n_agents, emb_dim)

	def get_attn_mask(self, shape):
		"""Generates a random binary mask to represent a coalition of agents."""
		# Create a random mask for sampling a coalition
		mask = torch.bernoulli(torch.full((shape, shape), 0.5))
		# Ensure an agent always attends to itself
		mask = mask - torch.diag(torch.diag(mask)) + torch.eye(shape)
		return mask.to(self.device)

	def _generate_structured_coalitions(self, n_agents, agent_mask):
		"""
		Generates a structured set of coalitions to ensure all active agents are evaluated.

		For each active agent, this function generates pairs of coalitions: one with the
		agent included and one with the agent excluded, while the status of other agents
		is randomized. This provides a more stable estimate of marginal contributions.

		Args:
			n_agents (int): The total number of agents.
			agent_mask (torch.Tensor): A mask indicating which agents are active (1) or
										inactive (0). Shape (batch_size, n_agents).

		Returns:
			torch.Tensor: A tensor of attention masks for the coalitions.
		"""
		b, _ = agent_mask.shape
		coalition_masks = []

		for i in range(n_agents):
			# Only generate coalitions for active agents
			if agent_mask[:, i].any():
				# Sample random coalitions for other agents
				other_agents_mask = torch.bernoulli(torch.full((b, n_agents - 1), 0.5)).to(self.device)
				
				# Create coalition WITH agent i
				coalition_with_i = torch.cat([other_agents_mask[:, :i], torch.ones(b, 1).to(self.device), other_agents_mask[:, i:]], dim=1)
				
				# Create coalition WITHOUT agent i
				coalition_without_i = torch.cat([other_agents_mask[:, :i], torch.zeros(b, 1).to(self.device), other_agents_mask[:, i:]], dim=1)
				
				coalition_masks.extend([coalition_with_i, coalition_without_i])

		# If no agents are active, return a zero mask
		if not coalition_masks:
			return torch.zeros(b, n_agents, n_agents).to(self.device)

		# Stack and create the final attention masks
		stacked_masks = torch.stack(coalition_masks, dim=1) # (b, num_samples*2, n_agents)
		attn_mask = stacked_masks.unsqueeze(-1) * stacked_masks.unsqueeze(-2)

		# Ensure self-attention
		attn_mask = attn_mask + torch.eye(n_agents, device=self.device).unsqueeze(0).unsqueeze(0)
		attn_mask = (attn_mask > 0).float()

		return attn_mask.reshape(b * len(coalition_masks), n_agents, n_agents)

	# def forward(self, input_tensor):
	# 	"""
	# 	Computes the Shapley-inspired rewards for a batch of timesteps.

	# 	Args:
	# 		input_tensor (torch.Tensor): A tensor of input sequences with shape
	# 									(batch, n_agents, seq_len, emb_dim).

	# 	Returns:
	# 		torch.Tensor: The final Shapley-inspired rewards, averaged over samples.
	# 	"""
	# 	b, n_a, t, e = input_tensor.size()
	# 	# Reshape for batch processing of all timesteps
	# 	input_tensor = input_tensor.permute(0, 2, 1, 3).contiguous().reshape(b * t, n_a, -1)

	# 	# Add a randomized agent embedding to break symmetry
	# 	coalition = np.arange(self.n_agents)
	# 	np.random.shuffle(coalition)
	# 	agent_embedding = self.agent_embedding(torch.tensor(coalition).to(self.device))[None, :, :].expand(b * t, n_a, self.emb_dim)
	# 	input_with_embedding = input_tensor + agent_embedding

	# 	shapley_rewards = []
	# 	# Monte Carlo approximation of Shapley values
	# 	for _ in range(self.sample_num):
	# 		attn_mask = self.get_attn_mask(n_a).unsqueeze(0).repeat(b * t, 1, 1)
	# 		marginal_reward, _ = self.phi(input_with_embedding, input_with_embedding, input_with_embedding, attn_mask)
	# 		shapley_rewards.append(marginal_reward)

	# 	# Average the rewards over all sampled coalitions
	# 	avg_shapley_reward = sum(shapley_rewards) / self.sample_num
	# 	# Reshape back to the original batch format
	# 	avg_shapley_reward = avg_shapley_reward.reshape(b, t, n_a, -1).permute(0, 2, 1, 3)

	# 	return avg_shapley_reward

	def forward(self, input_tensor, agent_temporal_mask):
		"""
		Computes the Shapley-inspired rewards for a batch of timesteps.

		Args:
			input_tensor (torch.Tensor): A tensor of input sequences with shape
									 (batch, n_agents, seq_len, emb_dim).
			agent_temporal_mask (torch.Tensor): Mask for inactive agents/timesteps.

		Returns:
			torch.Tensor: The final Shapley-inspired rewards, averaged over samples.
		"""
		b, n_a, t, e = input_tensor.size()
		# Reshape for batch processing of all timesteps
		input_tensor_reshaped = input_tensor.permute(0, 2, 1, 3).contiguous().reshape(b * t, n_a, -1)
		agent_mask_reshaped = agent_temporal_mask.permute(0, 2, 1).contiguous().reshape(b * t, n_a)

		# Add a randomized agent embedding to break symmetry
		coalition = np.arange(self.n_agents)
		np.random.shuffle(coalition)
		agent_embedding = self.agent_embedding(torch.tensor(coalition).to(self.device))[None, :, :].expand(b * t, n_a, self.emb_dim)
		input_with_embedding = input_tensor_reshaped + agent_embedding

		# Generate structured coalitions
		attn_masks = self._generate_structured_coalitions(n_a, agent_mask_reshaped)
		
		# If no agents are active, attn_masks could be empty.
		if attn_masks.shape[0] == 0:
			return torch.zeros_like(input_tensor)

		# Expand input tensor to match the number of coalition samples
		num_samples = attn_masks.shape[0] // (b * t)
		input_expanded = input_with_embedding.unsqueeze(1).repeat(1, num_samples, 1, 1).reshape(-1, n_a, e)

		marginal_rewards, _ = self.phi(input_expanded, input_expanded, input_expanded, attn_masks)
		
		# The attention module now contains weights/scores for all samples.
		# We need to reshape and average them before they are accessed by the TAR2 model.
		# Average agent attention weights
		all_sample_weights = self.phi.agent_weights
		all_sample_weights = all_sample_weights.reshape(b * t, num_samples, n_a, n_a)
		self.phi.agent_weights = all_sample_weights.mean(dim=1) # Overwrite with averaged weights

		# Average agent attention scores
		all_sample_scores = self.phi.agent_scores
		all_sample_scores = all_sample_scores.reshape(b * t, num_samples, self.phi.n_head, n_a, n_a)
		self.phi.agent_scores = all_sample_scores.mean(dim=1) # Overwrite with averaged scores

		# Reshape and average the results for the reward
		marginal_rewards = marginal_rewards.reshape(b * t, num_samples, n_a, e)
		avg_shapley_reward = marginal_rewards.mean(dim=1)
		
		# Reshape back to the original batch format
		avg_shapley_reward = avg_shapley_reward.reshape(b, t, n_a, -1).permute(0, 2, 1, 3)

		return avg_shapley_reward
			


class TAR2(nn.Module):
	"""
	The core Temporal-Agent Reward Redistribution (TAR²) model.

	This model processes entire episode trajectories to produce unnormalized contribution
	scores (c_i,t) for each agent at each timestep. It features a dual-transformer
	architecture to capture both temporal and inter-agent dependencies.

	Key architectural components:
	1.  Final-State Conditioning: The reward prediction is conditioned on an embedding
		of the final trajectory outcome to provide a lower-variance learning target.
	2.  Inverse Dynamics Model: An auxiliary task that regularizes the learned
		representations by forcing them to be predictive of agent actions, thus
		grounding them in causal behavior.
	3.  Deterministic Normalization (not in this class): The scores produced by this
		model are passed to a separate, non-learned function that guarantees
		strict return equivalence.
	"""
	def __init__(self, environment, ally_obs_shape, enemy_obs_shape, n_actions, emb_dim, n_heads, n_layer, seq_length, n_agents, sample_num,
			device, emb_dropout=0.5):
		super().__init__()

		self.environment = environment
		self.emb_dim = emb_dim
		self.n_heads = n_heads
		self.n_layer = n_layer
		self.seq_length = seq_length
		self.device = device
		self.n_agents = n_agents

		# --- Input Embeddings ---
		# Environment-specific observation encoders
		if "StarCraft" in self.environment:
			self.ally_obs_compress_input = nn.Linear(ally_obs_shape, self.emb_dim)
			self.enemy_obs_compress_input = nn.Linear(enemy_obs_shape, self.emb_dim)
		elif "GFootball" in self.environment:
			self.ally_obs_compress_input = nn.Linear(ally_obs_shape, self.emb_dim)

		self.action_emb = nn.Embedding(n_actions + 1, emb_dim)
		self.pos_embedding = nn.Embedding(seq_length, emb_dim)

		# --- Core Transformer Architecture ---
		# A stack of layers, each containing a temporal attention block and an agent attention block
		self.layers = nn.ModuleList([
			nn.ModuleList([
				EncoderLayer(self.emb_dim, self.n_heads, self.emb_dim, emb_dropout),
				ShapelyAttention(emb_dim, n_heads, self.n_agents, sample_num, device, emb_dropout)
			]) for _ in range(self.n_layer)
		])

		# --- Auxiliary Inverse Dynamics Model ---
		# Predicts action from a combination of the current global state and the
		# agent's own historical context.
		# Input: [current_global_state, past_state_action_embedding]
		# Dims:  [emb_dim,            emb_dim*n_layer]
		self.dynamics_model = nn.Sequential(
			nn.Linear(self.emb_dim * (self.n_layer + 2), self.emb_dim),
			nn.GELU(),
			nn.Linear(self.emb_dim, self.emb_dim),
			nn.GELU(),
			nn.Linear(self.emb_dim, n_actions),
		)

		# --- Main Reward Prediction Head ---
		# Predicts the unnormalized scores c_i,t from the post-attention embeddings
		# concatenated with a representation of the final trajectory outcome.
		# Input: [current_context, final_outcome_context]
		# Dims:  [emb_dim*n_layer, emb_dim*n_layer]
		self.reward_prediction = nn.Sequential(
			nn.Linear(2 * emb_dim * self.n_layer, emb_dim),
			nn.GELU(),
			nn.Linear(self.emb_dim, self.emb_dim),
			nn.GELU(),
			nn.Linear(emb_dim, 1),
		)

		init_model(self)

	def get_time_mask(self, episode_length):
		"""Creates a causal attention mask for the temporal transformer."""
		mask = (torch.arange(self.seq_length, device=self.device)[None, :] < episode_length[:, None]).float()
		# Create a triangular mask for causal attention (a timestep can only attend to past timesteps)
		mask = torch.triu(torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))).transpose(-1, -2)
		return mask

	def forward(self, ally_states, enemy_states, actions, episode_length, agent_temporal_mask):
		"""
		Processes a batch of trajectories to produce unnormalized credit scores.

		Args:
			ally_states (torch.Tensor): Ally observations.
			enemy_states (torch.Tensor): Enemy observations.
			actions (torch.Tensor): Actions taken by agents.
			episode_length (torch.Tensor): Length of each episode in the batch.
			agent_temporal_mask (torch.Tensor): Mask for inactive agents/timesteps.

		Returns:
			tuple: A tuple containing:
				- rewards (torch.Tensor): The unnormalized scores c_i,t.
				- temporal_weights (torch.Tensor): Attention weights from the temporal layers.
				- agent_weights (torch.Tensor): Attention weights from the agent layers.
				- temporal_scores (torch.Tensor): Raw attention scores from temporal layers.
				- agent_scores (torch.Tensor): Raw attention scores from agent layers.
				- action_prediction (torch.Tensor): Output of the inverse dynamics model.
		"""
		b, n_a, t, _ = ally_states.size()

		# --- 1. Input Embedding ---
		# Create initial token embeddings from observations, actions, and positions.
		if "StarCraft" in self.environment:
			enemy_obs_embedding = self.enemy_obs_compress_input(enemy_states).mean(dim=1, keepdim=True).repeat(1, n_a, 1, 1)
			ally_obs_embedding = self.ally_obs_compress_input(ally_states)
			x = ally_obs_embedding + enemy_obs_embedding
		elif "GFootball" in self.environment:
			ally_obs_embedding = self.ally_obs_compress_input(ally_states)
			x = ally_obs_embedding

		positions = self.pos_embedding(torch.arange(self.seq_length, device=self.device))[None, None, :, :].expand(b, n_a, t, self.emb_dim)
		actions_embed = self.action_emb(actions.long()).squeeze()

		x = x + actions_embed + positions
		state_action_embedding = x # Keep a copy for later use

		# --- 2. Dual Transformer Processing ---
		# Pass embeddings through the stack of temporal and agent attention layers.
		time_mask = self.get_time_mask(episode_length).repeat(n_a, 1, 1)
		x = x.reshape(b * n_a, t, -1).squeeze()

		x_intermediate = []
		temporal_weights, agent_weights, temporal_scores, agent_scores = [], [], [], []

		for layer in self.layers:
			# Temporal attention
			x, _ = layer[0](x, time_mask)
			temporal_scores.append(layer[0].self_attn.temporal_scores)
			temporal_weights.append(layer[0].self_attn.temporal_weights)
			
			x = x.reshape(b, n_a, t, -1)
			
			# Agent attention
			x = layer[1](x, agent_temporal_mask)
			agent_scores.append(layer[1].phi.agent_scores)
			agent_weights.append(layer[1].phi.agent_weights)
			
			x = x.reshape(b * n_a, t, -1).squeeze()
			x_intermediate.append(x)

		# Apply masks to the collected attention weights and scores
		agent_weights = torch.stack(agent_weights, dim=0).reshape(self.n_layer, b, t, n_a, n_a) * agent_temporal_mask.unsqueeze(0).unsqueeze(-1) * agent_temporal_mask.unsqueeze(0).unsqueeze(-2)
		temporal_weights = torch.stack(temporal_weights, dim=0).reshape(self.n_layer, b, n_a, t, t) * agent_temporal_mask.permute(0, 2, 1).unsqueeze(0).unsqueeze(-1) * agent_temporal_mask.permute(0, 2, 1).unsqueeze(0).unsqueeze(-2)
		agent_scores = torch.stack(agent_scores, dim=0).reshape(self.n_layer, b, self.n_heads, t, n_a, n_a) * agent_temporal_mask.unsqueeze(0).unsqueeze(2).unsqueeze(-1) * agent_temporal_mask.unsqueeze(0).unsqueeze(2).unsqueeze(-2)
		temporal_scores = torch.stack(temporal_scores, dim=0).reshape(self.n_layer, b, self.n_heads, n_a, t, t) * agent_temporal_mask.permute(0, 2, 1).unsqueeze(0).unsqueeze(2).unsqueeze(-1) * agent_temporal_mask.permute(0, 2, 1).unsqueeze(0).unsqueeze(2).unsqueeze(-2)

		# Concatenate the outputs of all layers to get the final rich representation
		x_intermediate = torch.cat(x_intermediate, dim=-1).reshape(b, n_a, t, -1)

		# --- 3. Inverse Dynamics Model ---
		# This implementation predicts the action at time t based on the global state at t
		# and the agent's contextualized history up to t-1.

		# inverse dynamics model
		# 1. Calculate the global state embedding (pre-attention) for each timestep.
		global_state_embeddings = (state_action_embedding.view(b, n_a, t, self.emb_dim) - actions_embed).reshape(b, n_a, t, self.emb_dim).sum(dim=1, keepdim=True).repeat(1, n_a, 1, 1).reshape(b, n_a, t, -1) / (agent_temporal_mask.transpose(1, 2).sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-5)
		# 2. Get the next global state embedding by shifting the tensor.
		# For the last timestep, there is no "next" state, so we pad with zeros.
		next_global_state_embeddings = torch.cat([global_state_embeddings[:, :, 1:, :], torch.zeros(b, n_a, 1, self.emb_dim).to(self.device)], dim=-2)
		# 3. Get the agent-specific state-action context from the previous timestep (post-attention).
		first_past_state_action_embedding = torch.zeros(b, n_a, 1, self.n_layer*self.emb_dim).to(self.device)
		past_state_action_embeddings = torch.cat([first_past_state_action_embedding, x_intermediate[:, :, :-1, :]], dim=-2)
		# 4. Concatenate all three embeddings to form the input.
		dynamics_model_input = torch.cat([global_state_embeddings, next_global_state_embeddings, past_state_action_embeddings], dim=-1)
		# 5. Predict the action.
		action_prediction = self.dynamics_model(dynamics_model_input)

		# --- 4. Final Reward Prediction ---
		# Get the embedding of the final state for each agent in the batch
		indiv_agent_episode_len = (agent_temporal_mask.sum(dim=-2) - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.emb_dim * self.n_layer).long()
		final_x = torch.gather(x_intermediate, 2, indiv_agent_episode_len).squeeze(2)

		# Create the final outcome embedding (Z) by averaging final states across agents.
		# Detach to treat it as a fixed conditioning variable.
		final_outcome_embedding = final_x.mean(dim=1, keepdim=True).detach()

		# Condition the reward prediction on both the current context and the final outcome
		reward_prediction_embeddings = torch.cat([x_intermediate, final_outcome_embedding.unsqueeze(1).repeat(1, n_a, t, 1)], dim=-1)

		rewards = self.reward_prediction(reward_prediction_embeddings).view(b, n_a, t).contiguous().transpose(1, 2) * agent_temporal_mask.to(self.device)

		return rewards, temporal_weights, agent_weights, temporal_scores, agent_scores, action_prediction
