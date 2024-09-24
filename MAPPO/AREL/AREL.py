import torch
from torch import nn
import torch.nn.functional as F

from .modules import TransformerBlock, TransformerBlock_Agent

from .util import d
import math
import numpy as np


def init(module, weight_init, bias_init, gain=1):
	weight_init(module.weight.data, gain=gain)
	if module.bias is not None:
		bias_init(module.bias.data)
	return module

def init_(m, gain=0.01, activate=False):
	if activate:
		gain = nn.init.calculate_gain('relu')
	return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class Time_Agent_Transformer(nn.Module):
	"""
	Transformer along time steps.
	"""

	def __init__(
		self,
		environment,
		ally_obs_shape, 
		enemy_obs_shape, 
		obs_shape,
		action_shape,
		heads, 
		depth, 
		seq_length, 
		n_agents, 
		n_enemies,
		n_actions,
		agent=True, 
		dropout=0.0, 
		wide=True,  
		version="temporal", # temporal, agent_temporal
		linear_compression_dim=128,
		device=None
		):
		super().__init__()

		self.n_agents = n_agents
		self.version = version
		self.device = device
		self.depth = depth
		self.environment = environment

		self.obs_shape = obs_shape
		self.action_shape = action_shape
		self.seq_length = seq_length
		self.comp_emb = linear_compression_dim

		self.heads = heads
		self.depth = depth
		self.agent_attn = agent

		if "StarCraft" in self.environment:
			self.ally_obs_compress_input = nn.Sequential(
				init_(nn.Linear(ally_obs_shape, self.comp_emb), activate=False),
				)
			self.enemy_obs_compress_input = nn.Sequential(
				init_(nn.Linear(enemy_obs_shape, self.comp_emb), activate=False),
				)
		elif "GFootball" in self.environment:
			self.ally_obs_compress_input = nn.Sequential(
				init_(nn.Linear(ally_obs_shape, self.comp_emb), activate=False),
				)
			self.common_obs_compress_input = nn.Sequential(
				init_(nn.Linear(obs_shape, self.comp_emb), activate=False),
				)

		self.action_embedding = nn.Embedding(n_actions, self.comp_emb)

		self.position_embedding = nn.Embedding(seq_length, self.comp_emb)
		self.agent_embedding = nn.Embedding(n_agents, self.comp_emb)


		tblocks = []
		for i in range(depth):
			tblocks.append(
				TransformerBlock(emb=self.comp_emb, heads=heads, seq_length=seq_length, mask=True, dropout=dropout, wide=wide))

			if agent:
				tblocks.append(
					TransformerBlock_Agent(emb=self.comp_emb, heads=heads, seq_length=seq_length, n_agents=n_agents,
					mask=False, dropout=dropout, wide=wide)
					)

		self.tblocks = nn.Sequential(*tblocks)

		
		self.rblocks = nn.Sequential(
			init_(nn.Linear(self.comp_emb, 1), activate=False),
			)

		self.mask_value = torch.tensor(
				torch.finfo(torch.float).min, dtype=torch.float
			).to(self.device)

			
	def forward(self, ally_obs, enemy_obs, obs, actions, episodic_reward, team_masks=None, agent_masks=None):

		"""
		:param x: A (batch, number of agents, sequence length, state dimension) tensor of state sequences.
		:return: predicted log-probability vectors for each token based on the preceding tokens.
		"""
		# tokens = self.token_embedding(x)
		

		if "StarCraft" in self.environment:
			b, n_a, t, _ = ally_obs.size()
			_, n_e, _, _ = enemy_obs.size()
			enemy_obs_embedding = (self.enemy_obs_compress_input(enemy_obs)).mean(dim=1, keepdim=True)
			ally_obs_embedding = self.ally_obs_compress_input(ally_obs) #+ agent_embedding
		elif "GFootball" in self.environment:
			b, n_a, t, _ = ally_obs.size()
			ally_obs_embedding = self.ally_obs_compress_input(ally_obs)
			common_obs_embedding = self.common_obs_compress_input(obs)
			ally_obs_embedding = ally_obs_embedding + common_obs_embedding.unsqueeze(1)

		position_embedding = self.position_embedding(torch.arange(t, device=self.device))[None, None, :, :].expand(b, n_a, t, self.comp_emb)
		agent_embedding = self.agent_embedding(torch.arange(self.n_agents, device=self.device))[None, :, None, :].expand(b, n_a, t, self.comp_emb)
		action_embedding = self.action_embedding(actions.long())

		if "StarCraft" in self.environment:
			x = (ally_obs_embedding+enemy_obs_embedding+action_embedding+agent_embedding+position_embedding).view(b*n_a, t, self.comp_emb)
		elif "GFootball" in self.environment:
			x = (ally_obs_embedding+action_embedding+agent_embedding+position_embedding).view(b*n_a, t, self.comp_emb)

		temporal_weights, agent_weights, temporal_scores, agent_scores = [], [], [], []
		i = 0

		while i < len(self.tblocks):

			# even numbers have temporal attention
			x = self.tblocks[i](x, masks=agent_masks)
			temporal_weights.append(self.tblocks[i].attention.attn_weights)
			temporal_scores.append(self.tblocks[i].attention.attn_scores)

			i += 1

			if self.agent_attn:
				# odd numbers have agent attention
				x = self.tblocks[i](x, masks=agent_masks)
				agent_weights.append(self.tblocks[i].attention.attn_weights)
				agent_scores.append(self.tblocks[i].attention.attn_scores)

				i += 1

			if i == len(self.tblocks):
				break

		# to ensure masking across rows and columns
		agent_weights = torch.stack(agent_weights, dim=0).reshape(self.depth, b, t, n_a, n_a) * agent_masks.unsqueeze(0).unsqueeze(-1) * agent_masks.unsqueeze(0).unsqueeze(-2)
		temporal_weights = torch.stack(temporal_weights, dim=0).reshape(self.depth, b, n_a, t, t) * agent_masks.permute(0, 2, 1).unsqueeze(0).unsqueeze(-1) * agent_masks.permute(0, 2, 1).unsqueeze(0).unsqueeze(-2)
		agent_scores = torch.stack(agent_scores, dim=0).reshape(self.depth, b, self.heads, t, n_a, n_a) * agent_masks.unsqueeze(0).unsqueeze(2).unsqueeze(-1) * agent_masks.unsqueeze(0).unsqueeze(2).unsqueeze(-2)
		temporal_scores = torch.stack(temporal_scores, dim=0).reshape(self.depth, b, self.heads, n_a, t, t) * agent_masks.permute(0, 2, 1).unsqueeze(0).unsqueeze(2).unsqueeze(-1) * agent_masks.permute(0, 2, 1).unsqueeze(0).unsqueeze(2).unsqueeze(-2)

		if self.version == "temporal":
			x = x.reshape(b, n_a, t, -1).permute(0, 2, 1, 3).sum(dim=-2) / (agent_masks.sum(dim=-1, keepdim=True)+1e-5)
			rewards = (self.rblocks(x).view(b, t).contiguous() * team_masks.to(x.device)).unsqueeze(-1).repeat(1, 1, n_a)
		elif self.version == "agent_temporal":
			x = x.reshape(b, n_a, t, -1).permute(0, 2, 1, 3)
			rewards = self.rblocks(x).view(b, t, n_a).contiguous() * agent_masks.to(x.device)

		return rewards, temporal_weights, agent_weights, temporal_scores, agent_scores
