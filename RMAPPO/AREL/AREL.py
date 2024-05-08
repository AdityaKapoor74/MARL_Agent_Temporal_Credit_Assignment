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



class Time_Agent_Transformer(nn.Module):
	"""
	Transformer along time steps.
	"""

	def __init__(self, emb, heads, depth, seq_length, n_agents, agent=True, 
										dropout=0.0, wide=True, comp=True, norm_rewards=False, linear_compression_dim=128, device=None):
		super().__init__()

		self.comp = comp
		self.n_agents = n_agents
		self.device = device
		self.comp_emb = linear_compression_dim
		self.agent_attn = agent
		self.depth = depth
		self.heads = heads
		# if emb>100:
		# 	self.comp_emb = 100
		# else:
		# 	self.comp_emb = emb
		print(self.comp_emb, '-'*50)

		if not comp:
			self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length).to(self.device)
		
			tblocks = []
			for i in range(depth):
				tblocks.append(
					TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=True, dropout=dropout, wide=wide))
				if agent:
					tblocks.append(
						TransformerBlock_Agent(emb=emb, heads=heads, seq_length=seq_length, n_agents= n_agents,
						mask=False, dropout=dropout, wide=wide))

			self.tblocks = nn.Sequential(*tblocks)

			if norm_rewards:
				self.toreward = init_(PopArt(emb, 1, device=self.device))
			else:
				self.toreward = nn.Linear(emb, 1)

			self.do = nn.Dropout(dropout)
		else:
			self.compress_input = nn.Linear(emb, self.comp_emb)

			self.pos_embedding = nn.Embedding(embedding_dim=self.comp_emb, num_embeddings=seq_length).to(self.device)


			tblocks = []
			for i in range(depth):
				tblocks.append(
					TransformerBlock(emb=self.comp_emb, heads=heads, seq_length=seq_length, mask=True, dropout=dropout, wide=wide))
				if agent:
					tblocks.append(
						TransformerBlock_Agent(emb=self.comp_emb, heads=heads, seq_length=seq_length, n_agents= n_agents,
						mask=False, dropout=dropout, wide=wide))

			self.tblocks = nn.Sequential(*tblocks)
			
			rblocks = []

			rblocks.append(nn.Linear(self.comp_emb, 50))
			
			rblocks.append(nn.Linear(50, 50))
			
			self.rblocks = nn.Sequential(*rblocks)
			
			if norm_rewards:	
				self.toreward = init_(PopArt(50, 1, device=self.device))
			else:
				self.toreward = nn.Linear(50, 1)

			self.do = nn.Dropout(dropout)
			
	def forward(self, x, team_masks=None, agent_masks=None):
		"""
		:param x: A (batch, number of agents, sequence length, state dimension) tensor of state sequences.
		:return: predicted log-probability vectors for each token based on the preceding tokens.
		"""
		# tokens = self.token_embedding(x)
		
		b, n_a, t, e = x.size()
		if not self.comp:
			positions = self.pos_embedding(torch.arange(t, device=(self.device if self.device is not None else d())))[None, :, :].expand(b*n_a, t, e)
			x = x.view(b*n_a, t, e) + positions
		else:
			positions = self.pos_embedding(torch.arange(t, device=(self.device if self.device is not None else d())))[None, :, :].expand(b*n_a, t, self.comp_emb)
			x = self.compress_input(x).view(b*n_a, t, self.comp_emb) + positions

		# x = self.do(x)
		temporal_weights, agent_weights, state_latent_embeddings, dynamics_model_output, temporal_scores, agent_scores = [], [], [], [], [], []
		
		i = 0
		while i < len(self.tblocks):
			x = self.tblocks[i](x, masks=agent_masks)
			temporal_weights.append(self.tblocks[i].attention.attn_weights)
			temporal_scores.append(self.tblocks[i].attention.attn_scores)
			i+=1

			if self.agent_attn:
				x = self.tblocks[i](x, masks=agent_masks)
				agent_weights.append(self.tblocks[i].attention.attn_weights)
				agent_scores.append(self.tblocks[i].attention.attn_scores)
				i += 1

			state_latent_embeddings.append(x)

		x = self.rblocks(x).view(b, n_a, t, 50).contiguous().transpose(1,2).contiguous().sum(2) ###shape (b, t, 50)
				
		x_time_wise = self.toreward(x).view(b, t).contiguous()

		if team_masks is not None:
			x_time_wise = self.toreward(x).view(b, t).contiguous() * team_masks.to(self.device)

		x_episode_wise = x_time_wise.sum(1)

		temporal_weights = (torch.stack(temporal_weights, dim=0).reshape(self.depth, b, n_a, t, t)[-1, :, :, :, :].sum(dim=1)/(agent_masks.permute(0, 2, 1).sum(dim=1).unsqueeze(-1)+1e-5))[:, -1, :] * team_masks

		temporal_scores = torch.stack(temporal_scores, dim=0).reshape(self.depth, b, n_a, self.heads, t, t) * agent_masks.permute(0,2,1).reshape(1, b, n_a, 1, 1, t).to(x.device)
		temporal_scores = temporal_scores * agent_masks.permute(0,2,1).reshape(1, b, n_a, 1, t, 1).to(x.device)

		# taking the last agent-transformer block's weights
		agent_weights = torch.stack(agent_weights, dim=0).reshape(self.depth, b, t, n_a, n_a)[-1, :, :, :, :].sum(dim=-2)/(agent_masks.sum(dim=-1).unsqueeze(-1)+1e-5) * agent_masks

		agent_scores = torch.stack(agent_scores, dim=0).reshape(self.depth, b, t, self.heads, n_a, n_a) * agent_masks.reshape(1, b, t, 1, 1, n_a).to(x.device)
		agent_scores = agent_scores * agent_masks.reshape(1, b, t, 1, n_a, 1).to(x.device)
		return x_episode_wise, x_time_wise, temporal_weights, temporal_scores, agent_weights, agent_scores


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
