"""
This file contains the core implementation of the AREL (Attention-based REward Learning) model,
a baseline method for credit assignment.

The main component is the Time_Agent_Transformer class, which uses a transformer architecture
to process trajectory data and directly predict shaped rewards.
"""
import torch
from torch import nn
import torch.nn.functional as F

from .modules import TransformerBlock, TransformerBlock_Agent
from .util import d
import math
import numpy as np

def init(module, weight_init, bias_init, gain=1):
	"""
	Applies a specified weight and bias initialization to a module.

	Args:
		module (nn.Module): The module to initialize.
		weight_init: The function to use for weight initialization.
		bias_init: The function to use for bias initialization.
		gain (float): A scaling factor for the weight initialization.
	"""
	weight_init(module.weight.data, gain=gain)
	if module.bias is not None:
		bias_init(module.bias.data)
	return module

def init_(m, gain=0.01, activate=False):
	"""
	A helper function for orthogonal initialization of linear layers.

	Args:
		m (nn.Module): The module to initialize.
		gain (float): The gain for the orthogonal initialization.
		activate (bool): If True, calculates gain based on ReLU activation.
	"""
	if activate:
		gain = nn.init.calculate_gain('relu')
	return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class Time_Agent_Transformer(nn.Module):
	"""
	The core AREL model, implemented as a dual-axis transformer.

	This model processes entire episode trajectories to directly predict shaped rewards.
	It can be configured in two modes:
	- "temporal": Performs temporal credit assignment for the team as a whole.
	- "agent_temporal": Performs joint agent-temporal credit assignment.

	The architecture consists of interleaved temporal and agent-wise transformer blocks.
	"""
	def __init__(
		self,
		environment,
		ally_obs_shape, 
		enemy_obs_shape, 
		action_shape,
		heads, 
		depth, 
		seq_length, 
		n_agents, 
		n_actions,
		agent=True, 
		dropout=0.0, 
		wide=True,  
		version="temporal", # Can be "temporal" or "agent_temporal"
		linear_compression_dim=128,
		device=None
		):
		super().__init__()

		self.n_agents = n_agents
		self.version = version
		self.device = device
		self.depth = depth
		self.environment = environment
		self.seq_length = seq_length
		self.comp_emb = linear_compression_dim
		self.heads = heads
		self.agent_attn = agent

		# --- Input Embeddings ---
		if "StarCraft" in self.environment:
			self.ally_obs_compress_input = nn.Sequential(init_(nn.Linear(ally_obs_shape, self.comp_emb)))
			self.enemy_obs_compress_input = nn.Sequential(init_(nn.Linear(enemy_obs_shape, self.comp_emb)))
		elif "GFootball" in self.environment:
			self.ally_obs_compress_input = nn.Sequential(init_(nn.Linear(ally_obs_shape, self.comp_emb)))

		self.action_embedding = nn.Embedding(n_actions, self.comp_emb)
		self.position_embedding = nn.Embedding(seq_length, self.comp_emb)
		self.agent_embedding = nn.Embedding(n_agents, self.comp_emb)

		# --- Core Transformer Architecture ---
		# A sequence of interleaved temporal and agent transformer blocks
		tblocks = []
		for i in range(depth):
			tblocks.append(
				TransformerBlock(emb=self.comp_emb, heads=heads, seq_length=seq_length, mask=True, dropout=dropout, wide=wide))
			if agent:
				tblocks.append(
					TransformerBlock_Agent(emb=self.comp_emb, heads=heads, seq_length=seq_length, n_agents=n_agents,
					mask=False, dropout=dropout, wide=wide))
		self.tblocks = nn.Sequential(*tblocks)

		# --- Final Reward Prediction Head ---
		# A simple linear layer to regress the final reward values from the embeddings.
		self.rblocks = nn.Sequential(init_(nn.Linear(self.comp_emb, 1)))

		self.mask_value = torch.tensor(torch.finfo(torch.float).min, dtype=torch.float).to(self.device)

	def forward(self, ally_obs, enemy_obs, actions, team_masks=None, agent_masks=None):
		"""
		Processes a batch of trajectories to directly predict shaped rewards.

		Args:
			ally_obs (torch.Tensor): Ally observations.
			enemy_obs (torch.Tensor): Enemy observations.
			actions (torch.Tensor): Actions taken by agents.
			team_masks (torch.Tensor): Mask for padding at the team/episode level.
			agent_masks (torch.Tensor): Mask for inactive agents/timesteps.

		Returns:
			tuple: A tuple containing:
				- rewards (torch.Tensor): The predicted shaped rewards.
				- temporal_weights (torch.Tensor): Attention weights from temporal layers.
				- agent_weights (torch.Tensor): Attention weights from agent layers.
				- temporal_scores (torch.Tensor): Raw attention scores from temporal layers.
				- agent_scores (torch.Tensor): Raw attention scores from agent layers.
		"""
		b, n_a, t, _ = ally_obs.size()

		# --- 1. Input Embedding ---
		# Create initial token embeddings from observations, actions, positions, and agent IDs.
		if "StarCraft" in self.environment:
			enemy_obs_embedding = self.enemy_obs_compress_input(enemy_obs).mean(dim=1, keepdim=True)
			ally_obs_embedding = self.ally_obs_compress_input(ally_obs)
			x = ally_obs_embedding + enemy_obs_embedding
		elif "GFootball" in self.environment:
			ally_obs_embedding = self.ally_obs_compress_input(ally_obs)
			x = ally_obs_embedding

		position_embedding = self.position_embedding(torch.arange(t, device=self.device))[None, None, :, :].expand(b, n_a, t, self.comp_emb)
		agent_embedding = self.agent_embedding(torch.arange(self.n_agents, device=self.device))[None, :, None, :].expand(b, n_a, t, self.comp_emb)
		action_embedding = self.action_embedding(actions.long())

		x = (x + action_embedding + agent_embedding + position_embedding).view(b*n_a, t, self.comp_emb)

		# --- 2. Dual Transformer Processing ---
		temporal_weights, agent_weights, temporal_scores, agent_scores = [], [], [], []
		i = 0
		while i < len(self.tblocks):
			# Even-indexed blocks are temporal transformers
			x = self.tblocks[i](x, masks=agent_masks)
			temporal_weights.append(self.tblocks[i].attention.attn_weights)
			temporal_scores.append(self.tblocks[i].attention.attn_scores)
			i += 1

			if self.agent_attn:
				# Odd-indexed blocks are agent transformers
				x = self.tblocks[i](x, masks=agent_masks)
				agent_weights.append(self.tblocks[i].attention.attn_weights)
				agent_scores.append(self.tblocks[i].attention.attn_scores)
				i += 1
			if i == len(self.tblocks):
				break

		# --- 3. Final Reward Prediction ---
		# The final output shape and prediction logic depends on the AREL version.
		if self.version == "temporal":
			# Aggregate agent embeddings and predict a single reward per timestep for the team.
			x = x.reshape(b, n_a, t, -1).permute(0, 2, 1, 3).sum(dim=-2) / (agent_masks.sum(dim=-1, keepdim=True) + 1e-5)
			rewards = (self.rblocks(x).view(b, t).contiguous() * team_masks.to(x.device)).unsqueeze(-1).repeat(1, 1, n_a)
		elif self.version == "agent_temporal":
			# Predict a unique reward for each agent at each timestep.
			x = x.reshape(b, n_a, t, -1).permute(0, 2, 1, 3)
			rewards = self.rblocks(x).view(b, t, n_a).contiguous() * agent_masks.to(x.device)

		# Apply masks to the collected attention weights and scores for analysis
		agent_weights = torch.stack(agent_weights, dim=0).reshape(self.depth, b, t, n_a, n_a) * agent_masks.unsqueeze(0).unsqueeze(-1) * agent_masks.unsqueeze(0).unsqueeze(-2)
		temporal_weights = torch.stack(temporal_weights, dim=0).reshape(self.depth, b, n_a, t, t) * agent_masks.permute(0, 2, 1).unsqueeze(0).unsqueeze(-1) * agent_masks.permute(0, 2, 1).unsqueeze(0).unsqueeze(-2)
		agent_scores = torch.stack(agent_scores, dim=0).reshape(self.depth, b, self.heads, t, n_a, n_a) * agent_masks.unsqueeze(0).unsqueeze(2).unsqueeze(-1) * agent_masks.unsqueeze(0).unsqueeze(2).unsqueeze(-2)
		temporal_scores = torch.stack(temporal_scores, dim=0).reshape(self.depth, b, self.heads, n_a, t, t) * agent_masks.permute(0, 2, 1).unsqueeze(0).unsqueeze(2).unsqueeze(-1) * agent_masks.permute(0, 2, 1).unsqueeze(0).unsqueeze(2).unsqueeze(-2)

		return rewards, temporal_weights, agent_weights, temporal_scores, agent_scores