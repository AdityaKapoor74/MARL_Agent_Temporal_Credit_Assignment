import torch
from torch import nn
import torch.nn.functional as F

import math
import numpy as np

from .modules import TransformerBlock, TransformerBlock_Agent

from .util import d

def init(module, weight_init, bias_init, gain=1):
	if isinstance(module, nn.LayerNorm):
		init.ones_(module.weight)
		if module.bias is not None:
			init.zeros_(module.bias)
	elif isinstance(module, nn.Linear):
		weight_init(module.weight.data, gain=gain)
		# weight_init(module.weight.data)
		if module.bias is not None:
			bias_init(module.bias.data)
	return module

def init_(m, gain=0.01, activate=False):
	if activate:
		gain = nn.init.calculate_gain('relu')
	# return init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0), gain=gain)
	return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class HyperNetwork(nn.Module):
	def __init__(self, num_actions, hidden_dim, final_dim, obs_dim):
		super(HyperNetwork, self).__init__()
		self.num_actions = num_actions
		self.hidden_dim = hidden_dim
		self.final_dim = final_dim

		self.hyper_w1 = nn.Sequential(
			init_(nn.Linear(obs_dim, hidden_dim), activate=True),
			nn.GELU(),
			init_(nn.Linear(hidden_dim, num_actions * hidden_dim), activate=True)
			)
		self.hyper_b1 = nn.Sequential(
			init_(nn.Linear(obs_dim, hidden_dim), activate=True),
			nn.GELU(),
			init_(nn.Linear(hidden_dim, hidden_dim), activate=True)
			)
		self.hyper_w2 = nn.Sequential(
			init_(nn.Linear(obs_dim, hidden_dim), activate=True),
			nn.GELU(),
			init_(nn.Linear(hidden_dim, final_dim*hidden_dim))
			)
		self.hyper_b2 = nn.Sequential(
			init_(nn.Linear(obs_dim, hidden_dim), activate=True),
			nn.GELU(),
			init_(nn.Linear(hidden_dim, hidden_dim))
			)

	def forward(self, one_hot_actions, obs):
		one_hot_actions = one_hot_actions.reshape(-1, 1, self.num_actions)
		w1 = self.hyper_w1(obs)
		b1 = self.hyper_b1(obs)
		w1 = w1.reshape(-1, self.num_actions, self.hidden_dim)
		b1 = b1.reshape(-1, 1, self.hidden_dim)

		x = F.gelu(torch.bmm(one_hot_actions, w1) + b1)

		w2 = self.hyper_w2(obs)
		b2 = self.hyper_b2(obs)
		w2 = w2.reshape(-1, self.hidden_dim, self.final_dim)
		b2 = b2.reshape(-1, 1, self.hidden_dim)

		x = torch.bmm(x, w2) + b2

		return x



class Time_Agent_Transformer(nn.Module):
	"""
	Transformer along time steps.
	"""

	def __init__(
		self,
		# obs_shape, 
		ally_obs_shape, 
		enemy_obs_shape, 
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
		version="temporal", # temporal, agent_temporal, temporal_attn_weights, agent_temporal_attn_weights
		linear_compression_dim=128,
		device=None
		):
		super().__init__()

		self.n_agents = n_agents
		self.version = version
		self.device = device
		self.depth = depth

		# self.obs_shape = obs_shape
		self.action_shape = action_shape
		self.seq_length = seq_length
		self.comp_emb = linear_compression_dim

		self.heads = heads
		self.depth = depth
		self.agent_attn = agent

		self.ally_obs_compress_input = nn.Sequential(
			init_(nn.Linear(ally_obs_shape, 16), activate=False),
			# nn.GELU(),
			)
		self.enemy_obs_compress_input = nn.Sequential(
			init_(nn.Linear(enemy_obs_shape, 16), activate=False),
			# nn.GELU(),
			)
		# self.enemy_layer_norm = nn.LayerNorm(self.comp_emb)

		# self.final_obs_embedding = nn.Sequential(
		# 	init_(nn.Linear(2*self.comp_emb, self.comp_emb), activate=True),
		# 	nn.GELU(),
		# 	init_(nn.Linear(self.comp_emb, self.comp_emb), activate=False),
		# 	)

		# self.agent_one_hot_ids = torch.eye(n_agents)
		# self.enemy_one_hot_ids = torch.eye(n_enemies)
		# self.one_hot_actions = torch.eye(n_agents, n_actions)

		self.action_embedding = nn.Embedding(n_actions, 16)

		self.return_embedding = init_(nn.Linear(1, 16), activate=False)

		self.position_embedding = nn.Embedding(seq_length, 16*4)
		# Create a matrix of shape (max_len, d_model) -- relative position embedding
		# self.position_embedding = torch.zeros(seq_length, self.comp_emb).float()
		# position = torch.arange(0, seq_length).float().unsqueeze(1)
		# div_term = torch.exp(torch.arange(0, self.comp_emb, 2).float() * -(torch.log(torch.tensor(10000.0)) / self.comp_emb))

		# self.position_embedding[:, 0::2] = torch.sin(position * div_term)
		# self.position_embedding[:, 1::2] = torch.cos(position * div_term)
		# self.position_embedding = self.position_embedding.to(self.device)

		self.agent_embedding = nn.Embedding(n_agents, 16*3)
		# self.enemy_embedding = nn.Embedding(n_enemies, self.comp_emb)
		# self.enemy_layer_norm = nn.LayerNorm(self.comp_emb)

		# self.state_embedding_norm = nn.LayerNorm(3*self.comp_emb//2)


		tblocks = []
		for i in range(depth):
			tblocks.append(
				TransformerBlock(emb=16*4, heads=heads, seq_length=seq_length, mask=True, dropout=dropout, wide=wide))
			if agent:
				tblocks.append(
					TransformerBlock_Agent(emb=16*4, heads=heads, seq_length=seq_length, n_agents=n_agents,
					mask=False, dropout=dropout, wide=wide)
					)

		self.tblocks = nn.Sequential(*tblocks)

		self.dynamics_model = nn.Sequential(
			# init_(nn.Linear(2*self.comp_emb, self.comp_emb), activate=True),
			# nn.GELU(),
			# init_(nn.Linear(self.comp_emb, self.comp_emb), activate=True),
			# nn.GELU(),
			# init_(nn.Linear(self.comp_emb, n_actions), activate=False)
			init_(nn.Linear(7*self.comp_emb//2, n_actions), activate=False)
			)
		
		# self.pre_final_norm = nn.LayerNorm(self.comp_emb*depth)

		self.rblocks = nn.Sequential(
			# init_(nn.Linear(self.comp_emb*depth, 1), activate=False),
			init_(nn.Linear(16*4*depth, self.comp_emb), activate=True),
			nn.GELU(),
			init_(nn.Linear(self.comp_emb, self.comp_emb), activate=True),
			nn.GELU(),
			init_(nn.Linear(self.comp_emb, 1)),
			nn.ReLU(),
			)
					   
		self.do = nn.Dropout(dropout)

		self.mask_value = torch.tensor(
				torch.finfo(torch.float).min, dtype=torch.float
			)

			
	def forward(self, ally_obs, enemy_obs, actions, episodic_reward, team_masks=None, agent_masks=None):

		"""
		:param x: A (batch, number of agents, sequence length, state dimension) tensor of state sequences.
		:return: predicted log-probability vectors for each token based on the preceding tokens.
		"""
		# tokens = self.token_embedding(x)
		
		
		b, n_a, t, _ = ally_obs.size()
		_, n_e, _, _ = enemy_obs.size()

		# enemy_ids = self.enemy_one_hot_ids.reshape(1, n_e, 1, n_e).repeat(b, 1, t, 1).to(self.device)
		# enemy_obs = torch.cat([enemy_ids, enemy_obs], dim=-1)#.permute(0, 2, 1, 3).reshape(b, 1, t, -1)

		# enemy_embedding = self.enemy_embedding(torch.arange(n_e).to(self.device))[None, None, :, :].expand(b, t, n_e, self.comp_emb).permute(0, 2, 1, 3)
		# enemy_obs = (self.enemy_obs_compress_input(enemy_obs)+enemy_embedding).sum(dim=1, keepdims=True) # (self.enemy_obs_compress_input(enemy_obs) + enemy_embedding).sum(dim=1).unsqueeze(1)
		enemy_obs = (self.enemy_obs_compress_input(enemy_obs)).mean(dim=1, keepdims=True)


		# ally_ids = self.agent_one_hot_ids.reshape(1, n_a, 1, n_a).repeat(b, 1, t, 1).to(self.device)
		# ally_one_hot_actions = self.one_hot_actions.reshape(1, n_a, 1, self.action_shape).repeat(b, 1, t, 1).to(self.device)
		# ally_obs = torch.cat([ally_ids, ally_obs], dim=-1)
		
		agent_embedding = self.agent_embedding(torch.arange(self.n_agents).to(self.device))[None, None, :, :].expand(b, t, n_a, 16*3).permute(0, 2, 1, 3)
		# ally_obs = (self.ally_obs_compress_input(ally_obs)+agent_embedding) #+ self.action_embedding(actions.long())
		ally_obs = self.ally_obs_compress_input(ally_obs)

		position_embedding = self.position_embedding(torch.arange(self.seq_length).to(self.device))[None, None, :, :].expand(b, n_a, t, 16*4)
		return_embedding = self.return_embedding(episodic_reward.reshape(b, 1)).reshape(b, 1, 1, 16)
		# position_embedding = self.position_embedding[None, None, :, :].expand(b, n_a, t, self.comp_emb)

		# state_embeddings_norm = (self.state_embedding_norm(self.ally_obs_compress_input(ally_obs) + enemy_obs) + agent_embedding + position_embedding).view(b*n_a, t, self.comp_emb) # self.state_embedding_norm(self.ally_obs_compress_input(ally_obs) + enemy_obs + agent_embedding + position_embedding).view(b*n_a, t, self.comp_emb)
		# x = state_embeddings_norm + self.action_embedding(actions.long()).view(b*n_a, t, self.comp_emb)

		# final_obs = torch.cat([ally_obs, enemy_obs.repeat(1, n_a, 1, 1)], dim=-1)
		# x = (self.final_obs_embedding(final_obs) + position_embedding).view(b*n_a, t, self.comp_emb)
		
		# states = (self.state_embedding_norm(ally_obs + enemy_obs) + agent_embedding + position_embedding + return_embedding)
		# x = (states + self.action_embedding(actions.long())).view(b*n_a, t, self.comp_emb)
		states = torch.cat([ally_obs, enemy_obs.repeat(1, self.n_agents, 1, 1), return_embedding.repeat(1, n_a, t, 1)], dim=-1) + agent_embedding
		x = (torch.cat([states, self.action_embedding(actions.long())], dim=-1) + position_embedding).view(b*n_a, t, 16*4)

		temporal_weights, agent_weights, temporal_scores, agent_scores = [], [], [], []
		i = 0

		# x_intermediate_temporal_agent = []
		x_intermediate = []
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


			x_intermediate.append(x)

			if i == len(self.tblocks):
				break

		# zeroth_state_action_embedding = torch.zeros((b, n_a, 1, 16*4)).to(self.device)
		# past_state_action_embeddings = torch.cat([zeroth_state_action_embedding, torch.stack(x_intermediate, dim=0).sum(dim=0).view(b, n_a, t, self.comp_emb*2)[:, :, :-1, :]], dim=2) # first past state-action embedding is 0
		# current_state_embeddings = states.sum(dim=1, keepdim=True)
		# action_prediction = self.dynamics_model(torch.cat([current_state_embeddings.repeat(1, self.n_agents, 1, 1), past_state_action_embeddings], dim=-1))
		action_prediction = None

		# to ensure masking across rows and columns
		agent_weights = torch.stack(agent_weights, dim=0).reshape(self.depth, b, t, n_a, n_a) * agent_masks.unsqueeze(0).unsqueeze(-1) * agent_masks.unsqueeze(0).unsqueeze(-2)
		temporal_weights = torch.stack(temporal_weights, dim=0).reshape(self.depth, b, n_a, t, t) * agent_masks.permute(0, 2, 1).unsqueeze(0).unsqueeze(-1) * agent_masks.permute(0, 2, 1).unsqueeze(0).unsqueeze(-2)
		agent_scores = torch.stack(agent_scores, dim=0).reshape(self.depth, b, self.heads, t, n_a, n_a) * agent_masks.unsqueeze(0).unsqueeze(2).unsqueeze(-1) * agent_masks.unsqueeze(0).unsqueeze(2).unsqueeze(-2)
		temporal_scores = torch.stack(temporal_scores, dim=0).reshape(self.depth, b, self.heads, n_a, t, t) * agent_masks.permute(0, 2, 1).unsqueeze(0).unsqueeze(2).unsqueeze(-1) * agent_masks.permute(0, 2, 1).unsqueeze(0).unsqueeze(2).unsqueeze(-2)

		temporal_weights_final_temporal_block, temporal_scores_final_temporal_block = None, None
		if self.version == "temporal" or self.version == "temporal_v2":
			x = x.reshape(b, n_a, t, -1).permute(0, 2, 1, 3).sum(dim=-2)
			rewards = self.rblocks(x).view(b, t).contiguous() * team_masks.to(x.device)
		elif self.version == "agent_temporal":
			# x = x.reshape(b, n_a, t, -1)
			x = torch.cat(x_intermediate, dim=-1).reshape(b, n_a, t, -1)
			rewards = self.rblocks(x).view(b, n_a, t).permute(0, 2, 1).contiguous() * agent_masks.to(x.device)
		else:
			
			indiv_agent_episode_len = (agent_masks.sum(dim=-2)-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 16*4*self.depth).long() # subtracting 1 for indexing purposes
			x = torch.gather(torch.cat(x_intermediate, dim=-1).reshape(b, n_a, t, -1), 2, indiv_agent_episode_len).squeeze(1)

			# episode_len, final_agent = torch.max((agent_masks.sum(dim=-2)-1), dim=1)
			# x = torch.cat(x_intermediate, dim=-1).reshape(b, n_a, t, -1)[torch.arange(b), final_agent.long()-1][torch.arange(b), episode_len.long()-1]
			

			rewards = self.rblocks(x).view(b, 1, n_a).contiguous()


		return rewards, temporal_weights, agent_weights, temporal_weights_final_temporal_block, temporal_scores, agent_scores, temporal_scores_final_temporal_block, action_prediction



class Time_Transformer(nn.Module):
	"""
	Transformer along time steps only.
	"""

	def __init__(self, emb, heads, depth, seq_length, n_agents, dropout=0.0, wide=True, comp=True, device=None):
		super().__init__()
		self.device = device
		self.comp = comp
		self.n_agents = n_agents
		self.comp_emb = int(1.5*emb//n_agents)
		print(self.comp_emb, '-'*50)
		if not comp:

			self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

			tblocks = []
			
			for i in range(depth):
				tblocks.append(
					TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=True, dropout=dropout, wide=wide))

			self.tblocks = nn.Sequential(*tblocks)

			self.toreward = nn.Linear(emb, 1)

			self.do = nn.Dropout(dropout)
		else:
			self.compress_input = nn.Linear(emb, self.comp_emb)

			self.pos_embedding = nn.Embedding(embedding_dim=self.comp_emb, num_embeddings=seq_length)

			tblocks = []
			
			for i in range(depth):
				tblocks.append(
					TransformerBlock(emb=self.comp_emb, heads=heads, seq_length=seq_length, mask=True, dropout=dropout, wide=wide))

			self.tblocks = nn.Sequential(*tblocks)

			self.toreward = nn.Linear(self.comp_emb, 1)

			self.do = nn.Dropout(dropout)


	def forward(self, x):
		"""
		:param x: A (batch, number of agents, sequence length, state dimension) tensor of state sequences.
		:return: predicted log-probability vectors for each token based on the preceding tokens.
		"""
		# tokens = self.token_embedding(x)
		
		batch_size, t, e = x.size()
		if not self.comp:
			positions = self.pos_embedding(torch.arange(t, device=(self.device if self.device is not None else d())))[None, :, :].expand(batch_size, t, e)
			x = x + positions
		else:
			positions = self.pos_embedding(torch.arange(t, device=(self.device if self.device is not None else d())))[None, :, :].expand(batch_size, t, self.comp_emb)
			x = self.compress_input(x) + positions

		# x = self.do(x)

		x = self.tblocks(x)

		x = self.toreward(x).view(batch_size, t)

		x_time_wise = x

		x_episode_wise = x_time_wise.sum(1)

		return x_episode_wise, x_time_wise
