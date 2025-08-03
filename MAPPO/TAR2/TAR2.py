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

# class ShapelyAttention(nn.Module):
# 	"""
# 	Approximates Shapley values for agent importance using multi-head attention.

# 	This module supports three coalition sampling methods:
# 	- 'random': Pure Monte Carlo sampling of random coalitions.
# 	- 'structured': For each agent, samples coalitions with and without it.
# 	- 'stratified': Samples coalitions stratified by size to reduce variance.
# 	"""
# 	def __init__(self, emb_dim, n_heads, n_agents, sample_num, device, dropout=0.0, coalition_method='stratified'):
# 		super().__init__()
# 		self.emb_dim = emb_dim
# 		self.device = device
# 		self.n_agents = n_agents
# 		self.sample_num = sample_num
# 		self.coalition_method = coalition_method
		
# 		self.phi = MultiAgentAttention(emb_dim, n_heads, n_agents, dropout, device)
# 		self.agent_embedding = nn.Embedding(self.n_agents, emb_dim)

# 	def _generate_random_masks(self, batch_shape, n_agents):
# 		"""Generates purely random binary masks for coalitions."""
# 		masks = []
# 		for _ in range(self.sample_num):
# 			mask = torch.bernoulli(torch.full((n_agents, n_agents), 0.5, device=self.device))
# 			mask = mask - torch.diag(torch.diag(mask)) + torch.eye(n_agents, device=self.device)
# 			masks.append(mask)
		
# 		stacked_masks = torch.stack(masks, dim=0).unsqueeze(0).repeat(batch_shape, 1, 1, 1)
# 		return stacked_masks.reshape(batch_shape * self.sample_num, n_agents, n_agents)

# 	def _generate_structured_masks(self, n_agents, agent_mask):
# 		"""Generates structured coalitions (with/without each agent)."""
# 		b, _ = agent_mask.shape
# 		all_coalition_vectors = []

# 		active_agents_indices = [torch.where(m)[0] for m in agent_mask]

# 		for i in range(b):
# 			coalition_vectors = []
# 			active_agents = active_agents_indices[i]
# 			if len(active_agents) == 0:
# 				# If no agents are active, create placeholder zero masks
# 				all_coalition_vectors.append(torch.zeros(2 * n_agents, n_agents, device=self.device))
# 				continue

# 			for agent_idx in active_agents:
# 				other_agents = active_agents[active_agents != agent_idx]
				
# 				# Coalition WITH agent i
# 				coalition_with = torch.zeros(n_agents, device=self.device)
# 				coalition_with[agent_idx] = 1
# 				if len(other_agents) > 0:
# 					random_others = other_agents[torch.bernoulli(torch.full((len(other_agents),), 0.5)).bool()]
# 					coalition_with[random_others] = 1
				
# 				# Coalition WITHOUT agent i
# 				coalition_without = coalition_with.clone()
# 				coalition_without[agent_idx] = 0
				
# 				coalition_vectors.extend([coalition_with, coalition_without])
# 			all_coalition_vectors.append(torch.stack(coalition_vectors))

# 		stacked_vectors = torch.stack(all_coalition_vectors)
# 		attn_mask = stacked_vectors.unsqueeze(-1) * stacked_vectors.unsqueeze(-2)
# 		attn_mask = attn_mask + torch.eye(n_agents, device=self.device).unsqueeze(0).unsqueeze(0)
# 		attn_mask = (attn_mask > 0).float()
		
# 		return attn_mask.reshape(-1, n_agents, n_agents)

# 	def _generate_stratified_masks(self, n_agents, agent_mask):
# 		"""
# 		Generates coalitions using stratified sampling based on coalition size.

# 		This method samples uniformly across coalition sizes to ensure a balanced
# 		evaluation, which can reduce the variance of the Shapley value estimate.
# 		"""
# 		b, _ = agent_mask.shape
# 		all_coalition_vectors = []
		
# 		active_indices_list = [torch.where(m)[0] for m in agent_mask]
# 		num_active_list = agent_mask.sum(dim=1)

# 		for i in range(b):
# 			coalition_vectors = []
# 			num_active = int(num_active_list[i].item())
# 			active_indices = active_indices_list[i]

# 			if num_active == 0:
# 				all_coalition_vectors.append(torch.zeros(self.sample_num, n_agents, device=self.device))
# 				continue

# 			# Sample coalition sizes uniformly from 0 to num_active
# 			sampled_sizes = torch.randint(0, num_active + 1, (self.sample_num,))
			
# 			for k in sampled_sizes:
# 				coalition_mask_vector = torch.zeros(n_agents, device=self.device)
# 				if k > 0:
# 					# Randomly choose k agents from the active set
# 					perm = torch.randperm(num_active)
# 					chosen_indices = active_indices[perm[:k]]
# 					coalition_mask_vector[chosen_indices] = 1.0
# 				coalition_vectors.append(coalition_mask_vector)
			
# 			all_coalition_vectors.append(torch.stack(coalition_vectors))

# 		stacked_vectors = torch.stack(all_coalition_vectors)
# 		attn_mask = stacked_vectors.unsqueeze(-1) * stacked_vectors.unsqueeze(-2)
# 		attn_mask = attn_mask + torch.eye(n_agents, device=self.device).unsqueeze(0).unsqueeze(0)
# 		attn_mask = (attn_mask > 0).float()
		
# 		return attn_mask.reshape(-1, n_agents, n_agents)

# 	def forward(self, input_tensor, agent_temporal_mask):
# 		b, n_a, t, e = input_tensor.size()
# 		input_tensor_reshaped = input_tensor.permute(0, 2, 1, 3).contiguous().reshape(b * t, n_a, -1)
# 		agent_mask_reshaped = agent_temporal_mask.permute(0, 2, 1).contiguous().reshape(b * t, n_a)

# 		coalition = np.arange(self.n_agents)
# 		np.random.shuffle(coalition)
# 		agent_embedding = self.agent_embedding(torch.tensor(coalition, device=self.device))[None, :, :].expand(b * t, n_a, self.emb_dim)
# 		input_with_embedding = input_tensor_reshaped + agent_embedding

# 		# --- Select Coalition Generation Method ---
# 		if self.coalition_method == 'random':
# 			attn_masks = self._generate_random_masks(b * t, n_a)
# 			num_samples = self.sample_num
# 		elif self.coalition_method == 'structured':
# 			attn_masks = self._generate_structured_masks(n_a, agent_mask_reshaped)
# 			num_samples = attn_masks.shape[0] // (b * t) if (b * t) > 0 else 0
# 		elif self.coalition_method == 'stratified':
# 			attn_masks = self._generate_stratified_masks(n_a, agent_mask_reshaped)
# 			num_samples = self.sample_num
# 		else:
# 			raise ValueError(f"Unknown coalition method: {self.coalition_method}")

# 		if num_samples == 0:
# 			return torch.zeros_like(input_tensor)

# 		input_expanded = input_with_embedding.unsqueeze(1).repeat(1, num_samples, 1, 1).reshape(-1, n_a, e)
# 		marginal_rewards, _ = self.phi(input_expanded, input_expanded, input_expanded, attn_masks)
		
# 		# Reshape and average results
# 		marginal_rewards = marginal_rewards.reshape(b * t, num_samples, n_a, e)
# 		avg_shapley_reward = marginal_rewards.mean(dim=1)
		
# 		# Average attention weights and scores for logging/analysis
# 		all_sample_weights = self.phi.agent_weights.reshape(b * t, num_samples, n_a, n_a)
# 		self.phi.agent_weights = all_sample_weights.mean(dim=1)
# 		all_sample_scores = self.phi.agent_scores.reshape(b * t, num_samples, self.phi.n_head, n_a, n_a)
# 		self.phi.agent_scores = all_sample_scores.mean(dim=1)

# 		avg_shapley_reward = avg_shapley_reward.reshape(b, t, n_a, -1).permute(0, 2, 1, 3)
# 		return avg_shapley_reward
	
	
class ShapelyAttention(nn.Module):
	"""
	Approximates Shapley values for agent importance using multi-head attention.

	This module supports three coalition sampling methods:
	- 'random': Pure Monte Carlo sampling of random coalitions.
	- 'structured': For each agent, samples coalitions with and without it.
	- 'stratified': Samples coalitions stratified by size to reduce variance.
	"""
	def __init__(self, emb_dim, n_heads, n_agents, sample_num, device, dropout=0.0, coalition_method='random'):
		super().__init__()
		self.emb_dim = emb_dim
		self.device = device
		self.n_agents = n_agents
		self.sample_num = sample_num
		self.coalition_method = coalition_method
		
		self.phi = MultiAgentAttention(emb_dim, n_heads, n_agents, dropout, device)
		self.agent_embedding = nn.Embedding(self.n_agents, emb_dim)
		
		# Pre-compute agent indices and permutation for efficiency
		self.register_buffer('agent_indices', torch.arange(n_agents, device=device))
		self.register_buffer('eye_mask', torch.eye(n_agents, device=device))
		
		# Pre-allocate coalition size templates for stratified sampling
		if coalition_method == 'stratified':
			self._precompute_stratified_templates()

	def _precompute_stratified_templates(self):
		"""Pre-compute coalition size distributions for stratified sampling."""
		# Pre-compute uniform distribution over coalition sizes
		self.register_buffer('coalition_sizes', 
						   torch.arange(0, self.n_agents + 1, device=self.device))

	def _generate_random_masks_vectorized(self, batch_size):
		"""Vectorized random mask generation - no loops."""
		# Generate all masks at once
		masks = torch.bernoulli(
			torch.full((batch_size, self.sample_num, self.n_agents, self.n_agents), 
					  0.5, device=self.device)
		)
		
		# Remove diagonal and add identity in one operation
		diagonal_indices = torch.arange(self.n_agents, device=self.device)
		masks[:, :, diagonal_indices, diagonal_indices] = 1.0
		
		return masks.reshape(batch_size * self.sample_num, self.n_agents, self.n_agents)

	def _generate_structured_masks_vectorized(self, agent_mask):
		"""Vectorized structured mask generation."""
		b, n_agents = agent_mask.shape
		
		# Find active agents for all batches at once
		active_counts = agent_mask.sum(dim=1, keepdim=True)  # (b, 1)
		max_active = int(active_counts.max().item())
		
		if max_active == 0:
			return torch.zeros(b * 2 * n_agents, n_agents, n_agents, device=self.device)
		
		# Create coalition vectors more efficiently
		all_masks = []
		
		for i in range(b):
			active_agents = torch.where(agent_mask[i])[0]
			n_active = len(active_agents)
			
			if n_active == 0:
				masks = torch.zeros(2 * n_agents, n_agents, n_agents, device=self.device)
			else:
				# Vectorized coalition generation
				coalitions = []
				for agent_idx in active_agents:
					other_agents = active_agents[active_agents != agent_idx]
					
					# Coalition WITH agent (vectorized random selection)
					coalition_with = torch.zeros(n_agents, device=self.device)
					coalition_with[agent_idx] = 1
					if len(other_agents) > 0:
						random_mask = torch.bernoulli(torch.full((len(other_agents),), 0.5, device=self.device))
						coalition_with[other_agents] = random_mask
					
					# Coalition WITHOUT agent
					coalition_without = coalition_with.clone()
					coalition_without[agent_idx] = 0
					
					coalitions.extend([coalition_with, coalition_without])
				
				coalition_vectors = torch.stack(coalitions)
				# Vectorized attention mask computation
				masks = coalition_vectors.unsqueeze(-1) * coalition_vectors.unsqueeze(-2)
				masks = masks + self.eye_mask.unsqueeze(0)
				masks = (masks > 0).float()
			
			all_masks.append(masks)
		
		return torch.cat(all_masks, dim=0)

	def _generate_stratified_masks_vectorized(self, agent_mask):
		"""Highly optimized stratified mask generation."""
		b, n_agents = agent_mask.shape
		
		# Generate all coalition sizes at once
		coalition_sizes = torch.randint(0, self.n_agents + 1, 
									  (b, self.sample_num), device=self.device)
		
		# Vectorized coalition generation
		all_coalition_vectors = torch.zeros(b, self.sample_num, n_agents, device=self.device)
		
		for i in range(b):
			active_indices = torch.where(agent_mask[i])[0]
			n_active = len(active_indices)
			
			if n_active == 0:
				continue
				
			# Clamp coalition sizes to number of active agents
			valid_sizes = torch.clamp(coalition_sizes[i], 0, n_active)
			
			for j, k in enumerate(valid_sizes):
				if k > 0:
					# Efficient random sampling without replacement
					perm = torch.randperm(n_active, device=self.device)[:k]
					chosen_indices = active_indices[perm]
					all_coalition_vectors[i, j, chosen_indices] = 1.0
		
		# Vectorized attention mask computation
		coalition_vectors = all_coalition_vectors.reshape(-1, n_agents)
		attn_mask = coalition_vectors.unsqueeze(-1) * coalition_vectors.unsqueeze(-2)
		attn_mask = attn_mask + self.eye_mask.unsqueeze(0)
		attn_mask = (attn_mask > 0).float()
		
		return attn_mask

	def forward(self, input_tensor, agent_temporal_mask):
		b, n_a, t, e = input_tensor.size()
		
		# More efficient reshaping - combine operations
		input_flat = input_tensor.permute(0, 2, 1, 3).reshape(b * t, n_a, e)
		agent_mask_flat = agent_temporal_mask.permute(0, 2, 1).reshape(b * t, n_a)
		
		# Optimized agent embedding computation
		# Use pre-computed indices instead of numpy operations
		perm_indices = torch.randperm(self.n_agents, device=self.device)
		agent_embedding = self.agent_embedding(perm_indices).unsqueeze(0).expand(b * t, -1, -1)
		input_with_embedding = input_flat + agent_embedding
		
		# Optimized coalition generation
		if self.coalition_method == 'random':
			attn_masks = self._generate_random_masks_vectorized(b * t)
			num_samples = self.sample_num
		elif self.coalition_method == 'structured':
			attn_masks = self._generate_structured_masks_vectorized(agent_mask_flat)
			num_samples = attn_masks.shape[0] // (b * t) if (b * t) > 0 else 0
		elif self.coalition_method == 'stratified':
			attn_masks = self._generate_stratified_masks_vectorized(agent_mask_flat)
			num_samples = self.sample_num
		else:
			raise ValueError(f"Unknown coalition method: {self.coalition_method}")

		if num_samples == 0:
			return torch.zeros_like(input_tensor)

		# More efficient tensor expansion
		input_expanded = input_with_embedding.unsqueeze(1).expand(-1, num_samples, -1, -1).reshape(-1, n_a, e)
		
		# Forward pass through attention
		marginal_rewards, _ = self.phi(input_expanded, input_expanded, input_expanded, attn_masks)
		
		# Efficient averaging and reshaping
		marginal_rewards = marginal_rewards.view(b * t, num_samples, n_a, e).mean(dim=1)
		
		# Update attention weights and scores (if needed for logging)
		if hasattr(self.phi, 'agent_weights'):
			self.phi.agent_weights = self.phi.agent_weights.view(b * t, num_samples, n_a, n_a).mean(dim=1)
		if hasattr(self.phi, 'agent_scores'):
			self.phi.agent_scores = self.phi.agent_scores.view(b * t, num_samples, -1, n_a, n_a).mean(dim=1)
		
		# Final reshape back to original dimensions
		return marginal_rewards.view(b, t, n_a, e).permute(0, 2, 1, 3)


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
	def __init__(self, environment, version, ally_obs_shape, enemy_obs_shape, n_actions, emb_dim, n_heads, n_layer, seq_length, n_agents, sample_num,
			device, emb_dropout=0.5):
		super().__init__()

		self.environment = environment
		self.version = version
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
		if self.version != "no_inverse_dynamics": 
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
		if self.version == "no_final_outcome":
			self.reward_prediction = nn.Sequential(
				nn.Linear(emb_dim * self.n_layer, emb_dim),
				nn.GELU(),
				nn.Linear(self.emb_dim, self.emb_dim),
				nn.GELU(),
				nn.Linear(emb_dim, 1),
			)
		else:
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
		if self.version == "no_inverse_dynamics":
			action_prediction = torch.zeros(b, n_a, t, self.emb_dim).to(self.device)
		else:
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
		if self.version == "no_final_outcome":
			# If no final outcome conditioning, just use the intermediate states.
			reward_prediction_embeddings = x_intermediate
		else:
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
