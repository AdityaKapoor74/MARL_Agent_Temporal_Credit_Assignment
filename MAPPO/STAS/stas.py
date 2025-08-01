"""
This file contains the implementation of the STAS (Spatio-Temporal Attention-based
Shapley value) model, a baseline method for joint agent-temporal credit assignment.

The main components are:
1.  ShapelyAttention: A module that uses a multi-head attention mechanism inspired by
	Shapley values to compute agent-specific contributions within a timestep.
2.  STAS_ML: The main sequence-to-sequence model that processes trajectory data and
	directly predicts the final shaped rewards for each agent at each timestep.
"""
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from .modules import EncoderLayer, MultiAgentAttention

class ShapelyAttention(nn.Module):
	"""
	Approximates Shapley values for agent importance using multi-head attention.

	This module computes the marginal contribution of each agent by attending over
	randomly sampled coalitions of agents. It serves as the agent-axis attention
	mechanism within the main STAS transformer body.
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

	def forward(self, input, agent_temporal_mask):
		"""
		Computes the Shapley-inspired agent representations for a batch of timesteps.

		Args:
			input (torch.Tensor): A tensor of input sequences with shape
								  (batch, n_agents, seq_len, emb_dim).
			agent_temporal_mask (torch.Tensor): Mask for inactive agents/timesteps.

		Returns:
			torch.Tensor: The final Shapley-inspired representations, averaged over samples.
		"""
		b, n_a, t, e = input.size()
		# Reshape for batch processing of all timesteps
		input = input.permute(0, 2, 1, 3).contiguous().reshape(b*t, n_a, -1)
		
		# Add a randomized agent embedding to break symmetry
		coalition = np.arange(self.n_agents)
		np.random.shuffle(coalition)
		agent_embedding = self.agent_embedding(torch.tensor(coalition).to(self.device))[None, :, :].expand(b*t, n_a, self.emb_dim)
		input = input + agent_embedding
		
		shapley_reward = []
		# Monte Carlo approximation of Shapley values
		for i in range(self.sample_num):
			attn_mask = self.get_attn_mask(n_a).unsqueeze(0).repeat(b*t, 1, 1)
			marginal_reward, _ = self.phi(input, input, input, attn_mask)
			shapley_reward.append(marginal_reward)

		# Average the representations over all sampled coalitions
		shapley_reward = sum(shapley_reward)/self.sample_num
		# Reshape back to the original batch format
		shapley_reward = shapley_reward.reshape(b, t, n_a, -1).permute(0, 2, 1, 3)

		return shapley_reward
		
class STAS_ML(nn.Module):
	"""
	The core STAS model for joint agent-temporal credit assignment.

	This model processes entire episode trajectories and directly predicts the final
	shaped reward for each agent at each timestep. It uses a dual-transformer
	architecture similar to TAR², but its training objective is a direct regression
	on the final team reward, which makes its theoretical guarantees conditional
	on the model's accuracy.
	"""
	def __init__(self, environment, ally_obs_shape, enemy_obs_shape, n_actions, emb_dim, n_heads, n_layer, seq_length, n_agents, sample_num,
				device, dropout=0.0, emb_dropout=0.5):
		super().__init__()

		self.environment = environment
		self.emb_dim = emb_dim
		self.n_heads = n_heads
		self.n_layer = n_layer
		self.seq_length = seq_length
		self.device = device
		self.n_agents = n_agents

		# --- Input Embeddings ---
		if "StarCraft" in self.environment:
			self.ally_obs_compress_input = nn.Linear(ally_obs_shape, self.emb_dim)
			self.enemy_obs_compress_input = nn.Linear(enemy_obs_shape, self.emb_dim)
		elif "GFootball" in self.environment:
			self.ally_obs_compress_input = nn.Linear(ally_obs_shape, self.emb_dim)

		self.action_emb = nn.Embedding(n_actions+1, emb_dim)
		self.pos_embedding = nn.Embedding(seq_length, emb_dim)

		# --- Core Transformer Architecture ---
		self.layers = nn.ModuleList([
			nn.ModuleList([
				EncoderLayer(self.emb_dim, self.n_heads, self.emb_dim, emb_dropout),
				ShapelyAttention(emb_dim, n_heads, self.n_agents, sample_num, device, emb_dropout)
			]) for _ in range(self.n_layer)
		])
		
		# --- Final Reward Prediction Head ---
		# This MLP takes the concatenated outputs from all transformer layers and
		# directly regresses the final shaped reward value.
		self.linear = nn.Sequential(
			nn.Linear(emb_dim*self.n_layer, emb_dim),
			nn.GELU(),
			nn.Linear(emb_dim, 1),
			)

	def get_time_mask(self, episode_length):
		"""Creates a causal attention mask for the temporal transformer."""
		mask = (torch.arange(self.seq_length, device=self.device)[None, :] < episode_length[:, None]).float()
		# Create a triangular mask for causal attention
		mask = torch.triu(torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))).transpose(-1, -2)
		return mask

	def forward(self, ally_states, enemy_states, states, actions, episode_length, agent_temporal_mask):
		"""
		Processes a batch of trajectories to directly predict shaped rewards.

		Args:
			ally_states (torch.Tensor): Ally observations.
			enemy_states (torch.Tensor): Enemy observations.
			states: (Not used in this implementation).
			actions (torch.Tensor): Actions taken by agents.
			episode_length (torch.Tensor): Length of each episode in the batch.
			agent_temporal_mask (torch.Tensor): Mask for inactive agents/timesteps.

		Returns:
			torch.Tensor: The predicted shaped rewards (r_i,t) for each agent at each timestep.
		"""
		b, n_a, t, _ = ally_states.size()

		# --- 1. Input Embedding ---
		if "StarCraft" in self.environment:
			enemy_obs_embedding = self.enemy_obs_compress_input(enemy_states).mean(dim=1, keepdim=True).repeat(1, n_a, 1, 1)
			ally_obs_embedding = self.ally_obs_compress_input(ally_states)
			x = (ally_obs_embedding + enemy_obs_embedding)
		elif "GFootball" in self.environment:
			ally_obs_embedding = self.ally_obs_compress_input(ally_states)
			x = ally_obs_embedding

		positions = self.pos_embedding(torch.arange(self.seq_length, device=self.device))[None, None, :, :].expand(b, n_a, self.seq_length, self.emb_dim)
		actions_embed = self.action_emb(actions.long()).squeeze()
		
		x = x + actions_embed + positions

		# --- 2. Dual Transformer Processing ---
		time_mask = self.get_time_mask(episode_length).repeat(n_a, 1, 1)
		x = x.reshape(b*n_a, t, -1).squeeze()
		
		shapley_rewards = []
		for layer in self.layers:
			# Temporal attention
			x, _ = layer[0](x, time_mask)
			x = x.reshape(b, n_a, t, -1)
			# Agent attention (Shapley-based)
			x = layer[1](x, agent_temporal_mask=None) # agent_temporal_mask is not used here
			x = x.reshape(b*n_a, t, -1).squeeze()
			shapley_rewards.append(x)

		# --- 3. Final Reward Prediction ---
		# Concatenate the outputs of all layers and pass through the final MLP head.
		shapley_reward = self.linear(torch.cat(shapley_rewards, dim=-1).reshape(b, n_a, t, -1)).squeeze()
		
		return shapley_reward