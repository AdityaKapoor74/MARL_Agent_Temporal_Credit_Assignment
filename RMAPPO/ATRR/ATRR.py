import torch
from torch import nn
import torch.nn.functional as F

import math
import numpy as np

from .modules import TransformerBlock, TransformerBlock_Agent

from .util import d

def init(module, weight_init, bias_init, gain=1):
	# if isinstance(module, nn.LayerNorm):
	# 	init.ones_(module.weight)
	# 	if module.bias is not None:
	# 		init.zeros_(module.bias)
	# elif isinstance(module, nn.Linear):
	# 	# weight_init(module.weight.data, gain=gain)
	# 	weight_init(module.weight.data)
	# 	if module.bias is not None:
	# 		bias_init(module.bias.data)
	return module

def init_(m, gain=0.01, activate=False):
	if activate:
		gain = nn.init.calculate_gain('relu')
	# return init(m, nn.init.kaiming_uniform_, lambda x: nn.init.constant_(x, 0), gain=gain)
	return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class PopArt(torch.nn.Module):
	
	def __init__(self, input_shape, output_shape, norm_axes=1, beta=0.99999, epsilon=1e-5, device=torch.device("cpu")):
		
		super(PopArt, self).__init__()

		self.beta = beta
		self.epsilon = epsilon
		self.norm_axes = norm_axes
		self.tpdv = dict(dtype=torch.float32, device=device)

		self.input_shape = input_shape
		self.output_shape = output_shape

		self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape)).to(**self.tpdv)
		self.bias = nn.Parameter(torch.Tensor(output_shape)).to(**self.tpdv)
		
		self.stddev = nn.Parameter(torch.ones(output_shape), requires_grad=False).to(**self.tpdv)
		self.mean = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
		self.mean_sq = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
		self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

		self.reset_parameters()

	def reset_parameters(self):
		torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		if self.bias is not None:
			fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			torch.nn.init.uniform_(self.bias, -bound, bound)
		self.mean.zero_()
		self.mean_sq.zero_()
		self.debiasing_term.zero_()

	def forward(self, input_vector):
		if type(input_vector) == np.ndarray:
			input_vector = torch.from_numpy(input_vector)
		input_vector = input_vector.to(**self.tpdv)

		return F.linear(input_vector, self.weight, self.bias)
	
	@torch.no_grad()
	def update(self, input_vector):
		if type(input_vector) == np.ndarray:
			input_vector = torch.from_numpy(input_vector)
		input_vector = input_vector.to(**self.tpdv)
		
		old_mean, old_var = self.debiased_mean_var()
		old_stddev = torch.sqrt(old_var)

		batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
		batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))
		
		self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
		self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
		self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

		self.stddev.data = (self.mean_sq - self.mean ** 2).sqrt().clamp(min=1e-4)
		
		new_mean, new_var = self.debiased_mean_var()
		new_stddev = torch.sqrt(new_var)
		
		self.weight.data = self.weight.data * old_stddev / new_stddev
		self.bias.data = (old_stddev * self.bias.data + old_mean - new_mean) / new_stddev

	def debiased_mean_var(self):
		debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
		debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
		debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
		return debiased_mean, debiased_var

	def normalize(self, input_vector):
		if type(input_vector) == np.ndarray:
			input_vector = torch.from_numpy(input_vector)
		input_vector_device = input_vector.device
		input_vector = input_vector.to(**self.tpdv)

		mean, var = self.debiased_mean_var()
		out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
		
		return out.to(input_vector_device)

	def denormalize(self, input_vector):
		if type(input_vector) == np.ndarray:
			input_vector = torch.from_numpy(input_vector)
		input_vector_device = input_vector.device
		input_vector = input_vector.to(**self.tpdv)

		mean, var = self.debiased_mean_var()
		out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
		
		# out = out.cpu().numpy()

		return out.to(input_vector_device)


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
		obs_shape, 
		action_shape,
		heads, 
		depth, 
		seq_length, 
		n_agents, 
		n_actions,
		agent=True, 
		dropout=0.0, 
		wide=True,  
		version="temporal", # temporal, agent_temporal, temporal_attn_weights, agent_temporal_attn_weights
		linear_compression_dim=128,
		norm_rewards=False,
		device=None
		):
		super().__init__()

		self.n_agents = n_agents
		self.version = version
		self.device = device

		self.obs_shape = obs_shape
		self.action_shape = action_shape
		self.comp_emb = linear_compression_dim

		self.heads = heads
		self.depth = depth
		self.agent_attn = agent

		self.obs_compress_input = init_(nn.Linear(obs_shape, self.comp_emb), activate=False)
		# self.action_compress_input = init_(nn.Linear(action_shape, self.comp_emb), activate=False)

		self.action_embedding = nn.Embedding(n_actions, self.comp_emb)

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
		
		# self.pre_final_temporal_block_norm = nn.LayerNorm(self.comp_emb)

		# self.final_temporal_block = []
		# for i in range(depth):
		# 	self.final_temporal_block.append(TransformerBlock(emb=self.comp_emb, heads=heads, seq_length=seq_length, mask=True, dropout=dropout, wide=wide))
		# self.final_temporal_block = nn.Sequential(*self.final_temporal_block)
		self.final_transformer_block = TransformerBlock(emb=self.comp_emb, heads=heads, seq_length=seq_length, mask=True, dropout=dropout, wide=wide)


		if norm_rewards:
			self.rblocks = nn.Sequential(
				init_(nn.Linear(self.comp_emb, 64), activate=True),
				nn.GELU(),
				init_(nn.Linear(64, 64), activate=True),
				nn.GELU(),
				init_(PopArt(64, 1, device=self.device))
				)
		else:
			self.rblocks = nn.Sequential(
				init_(nn.Linear(self.comp_emb, 64), activate=True),
				nn.GELU(),
				init_(nn.Linear(64, 64), activate=True),
				nn.GELU(),
				init_(nn.Linear(64, 1))
				)
					   
		self.do = nn.Dropout(dropout)

		self.mask_value = torch.tensor(
				torch.finfo(torch.float).min, dtype=torch.float
			)

			
	def forward(self, obs, actions, team_masks=None, agent_masks=None, episode_len=None):

		"""
		:param x: A (batch, number of agents, sequence length, state dimension) tensor of state sequences.
		:return: predicted log-probability vectors for each token based on the preceding tokens.
		"""
		# tokens = self.token_embedding(x)
		
		
		b, n_a, t, _ = obs.size()
		# x = torch.cat([obs, one_hot_actions], dim=-1) 
		x = (self.obs_compress_input(obs) + self.action_embedding(actions.long()) + self.agent_embedding(torch.arange(self.n_agents).to(self.device))[None, None, :, :].expand(b, t, n_a, self.comp_emb).permute(0, 2, 1, 3)).view(b*n_a, t, self.comp_emb)

		temporal_weights, agent_weights, temporal_scores, agent_scores = [], [], [], []
		i = 0

		x_intermediate_temporal_agent = []
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

			# x_intermediate_temporal_agent.append(x)

		# x_intermediate_temporal_agent = torch.stack(x_intermediate_temporal_agent, dim=0).sum(dim=0)

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
			x = x.reshape(b, n_a, t, -1)
			rewards = self.rblocks(x).view(b, n_a, t).permute(0, 2, 1).contiguous() * agent_masks.to(x.device)
		else:
			# x = (x+x_intermediate_temporal_agent).reshape(b, n_a, t, -1).permute(0, 2, 1, 3).sum(dim=-2)
			# x = self.pre_final_temporal_block_norm(x[torch.arange(x.shape[0]), episode_len])
			
			# x = (x.reshape(b, n_a, t, -1).permute(0, 2, 1, 3).sum(dim=-2)/(agent_masks.sum(dim=-1).reshape(b, t, 1)+1e-5))[torch.arange(b), episode_len]
			
			# x = self.final_temporal_block(x, masks=team_masks, temporal_only=True)[torch.arange(b), episode_len]
			# temporal_weights_final_temporal_block = self.final_temporal_block.attention.attn_weights
			# temporal_scores_final_temporal_block = self.final_temporal_block.attention.attn_scores

			# temporal_weights_final_temporal_block, temporal_scores_final_temporal_block = [], []
			# for i in range(len(self.final_temporal_block)):
			# 	x = self.final_temporal_block[i](x, masks=team_masks, temporal_only=True)
			# 	temporal_weights_final_temporal_block.append(self.final_temporal_block[i].attention.attn_weights)
			# 	temporal_scores_final_temporal_block.append(self.final_temporal_block[i].attention.attn_scores)

			indiv_agent_episode_len = (agent_masks.sum(dim=-2)-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.comp_emb).long() # subtracting 1 for indexing purposes
			
			# print("Shape of x:", x.shape)
			# print("Shape of x after reshaping:", x.reshape(b, n_a, t, -1).shape)
			# print("Shape of indiv_agent_episode_len:", indiv_agent_episode_len.shape)

			# print("Min index:", indiv_agent_episode_len.min().item())
			# print("Max index:", indiv_agent_episode_len.max().item())
			# assert indiv_agent_episode_len.min() >= 0 and indiv_agent_episode_len.max() < t

			# print(torch.gather(x.reshape(b, n_a, t, -1), 2, indiv_agent_episode_len).shape, torch.gather(x.reshape(b, n_a, t, -1), 2, indiv_agent_episode_len).sum(dim=1).squeeze(1).shape)

			# print((x.reshape(b, n_a, t, -1)[0, 0, indiv_agent_episode_len[0, 0, 0, 0], :]+x.reshape(b, n_a, t, -1)[0, 1, indiv_agent_episode_len[0, 1, 0, 0], :]+x.reshape(b, n_a, t, -1)[0, 2, indiv_agent_episode_len[0, 2, 0, 0], :]+x.reshape(b, n_a, t, -1)[0, 3, indiv_agent_episode_len[0, 3, 0, 0], :]+x.reshape(b, n_a, t, -1)[0, 4, indiv_agent_episode_len[0, 4, 0, 0], :]))

			x = torch.gather(x.reshape(b, n_a, t, -1), 2, indiv_agent_episode_len).sum(dim=1).squeeze(1)

			# print(x[0])

			rewards = self.rblocks(x).view(b, 1).contiguous()
			# temporal_scores_final_temporal_block  = torch.stack(temporal_scores_final_temporal_block, dim=0).reshape(self.depth, b, self.heads, t, t) * team_masks.unsqueeze(0).unsqueeze(2).unsqueeze(-1) * team_masks.unsqueeze(0).unsqueeze(2).unsqueeze(-2)
			# temporal_weights_final_temporal_block = torch.stack(temporal_weights_final_temporal_block, dim=0).reshape(self.depth, b, t, t) * team_masks.unsqueeze(0).unsqueeze(-1) * team_masks.unsqueeze(0).unsqueeze(-2)
			# ATTENTION ROLLOUT
			# temporal_weights_final_temporal_block = (temporal_weights_final_temporal_block[0][torch.arange(x.shape[0]), episode_len].unsqueeze(1) @ temporal_weights_final_temporal_block[1] @ temporal_weights_final_temporal_block[2]).squeeze(dim=-2)
			# temporal_weights_final_temporal_block = temporal_weights_final_temporal_block[-1][torch.arange(x.shape[0]), episode_len]



			if self.version == "temporal_attn_weights":
				# rewards = (rewards * temporal_weights_final_temporal_block).unsqueeze(-1).repeat(1, 1, n_a)
				# dropping final temporal attention block
				# use last attn block
				# temporal_weights_final = temporal_weights[-1].sum(dim=1)[torch.arange(x.shape[0]), episode_len, :]/(agent_masks.permute(0, 2, 1).sum(dim=1)+1e-5)
				# use attention rollout
				# temporal_weights_final = temporal_weights.sum(dim=2)/(agent_masks.permute(0, 2, 1).sum(dim=1).reshape(1, b, t, 1)+1e-5)
				# temporal_weights_final = (temporal_weights_final[0] @ temporal_weights_final[1] @ temporal_weights_final[2, torch.arange(x.shape[0]), episode_len, :].unsqueeze(2)).squeeze(-1)
				# temporal_weights_final = F.normalize(temporal_weights_final, dim=-1, p=1.0)
				
				temporal_weights_final = F.softmax(torch.where(team_masks.bool(), (temporal_scores[-1].mean(dim=1).sum(dim=1)/(agent_masks.sum(dim=-1).reshape(b, t, 1)+1e-5)).diagonal(dim1=-2, dim2=-1), self.mask_value), dim=-1)
				rewards = (rewards * temporal_weights_final.detach()).unsqueeze(-1).repeat(1, 1, n_a)
			
			elif self.version == "agent_temporal_attn_weights":
				# rewards = (rewards * temporal_weights_final_temporal_block.detach()).unsqueeze(-1) * (agent_weights.detach().mean(dim=0).sum(dim=-2)/(agent_masks.permute(0, 2, 1).sum(dim=1).unsqueeze(-1)+1e-5))
				# rewards = (rewards * temporal_weights_final_temporal_block).unsqueeze(-1) * (agent_weights[-1].sum(dim=-2)/(agent_masks.permute(0, 2, 1).sum(dim=1).unsqueeze(-1)+1e-5))

				# use last attn block
				# temporal_weights_final = temporal_weights[-1].sum(dim=1)[torch.arange(x.shape[0]), episode_len, :]/(agent_masks.permute(0, 2, 1).sum(dim=1)+1e-5)
				# use attention rollout
				# temporal_weights_final = temporal_weights.sum(dim=2)/(agent_masks.permute(0, 2, 1).sum(dim=1).reshape(1, b, t, 1)+1e-5)
				# temporal_weights_final = (temporal_weights_final[0] @ temporal_weights_final[1] @ temporal_weights_final[2, torch.arange(x.shape[0]), episode_len, :].unsqueeze(2)).squeeze(-1)
				# temporal_weights_final = F.normalize(temporal_weights_final, dim=-1, p=1.0)
				# rewards = (rewards * temporal_weights_final.detach()).unsqueeze(-1) * (agent_weights.detach().mean(dim=0).sum(dim=-2)/(agent_masks.permute(0, 2, 1).sum(dim=1).unsqueeze(-1)+1e-5))

				# temporal_weights_final = F.softmax(torch.where(team_masks.bool(), (temporal_scores[-1].mean(dim=1).sum(dim=1)/(agent_masks.sum(dim=-1).reshape(b, t, 1)+1e-5)).diagonal(dim1=-2, dim2=-1), self.mask_value), dim=-1)
				# agent_weights_final = F.softmax(torch.where(agent_masks.bool(), (agent_scores[-1].mean(dim=1)).diagonal(dim1=-2, dim2=-1), self.mask_value), dim=-1)
				# agent_weights_final = F.softmax(torch.where(agent_masks.bool(), (agent_scores[-1].mean(dim=1)).sum(dim=-2), self.mask_value), dim=-1)

				# temporal_weights_final = 

				# rewards = (rewards * temporal_weights_final.detach()).unsqueeze(-1) * agent_weights_final.detach()

				pass


		return rewards, temporal_weights, agent_weights, temporal_weights_final_temporal_block, temporal_scores, agent_scores, temporal_scores_final_temporal_block



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
