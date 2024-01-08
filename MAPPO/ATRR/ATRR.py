import torch
from torch import nn
import torch.nn.functional as F

from .modules import TransformerBlock, TransformerBlock_Agent

from .util import d


class Time_Agent_Transformer(nn.Module):
	"""
	Transformer along time steps.
	"""

	def __init__(self, emb, heads, depth, seq_length, n_agents, agent=True, 
										dropout=0.0, wide=True, comp=True, device=None):
		super().__init__()

		self.comp = comp
		self.n_agents = n_agents
		self.device = device
		if emb>100:
			self.comp_emb = 100
		else:
			self.comp_emb = emb
		print(self.comp_emb, '-'*50)

		seq_length = seq_length + 1 # adding 1 for CLS like token embedding

		self.depth = depth
		self.agent_attn = agent

		if not comp:
			self.summary_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=1).to(self.device)

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

			self.toreward = nn.Linear(emb, 1)

			self.do = nn.Dropout(dropout)
		else:
			self.compress_input = nn.Linear(emb, self.comp_emb)

			self.summary_embedding = nn.Embedding(embedding_dim=self.comp_emb, num_embeddings=1).to(self.device)

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
										   
			self.toreward = nn.Linear(50, 1)

			self.do = nn.Dropout(dropout)
			
	def forward(self, x):
		"""
		:param x: A (batch, number of agents, sequence length, state dimension) tensor of state sequences.
		:return: predicted log-probability vectors for each token based on the preceding tokens.
		"""
		# tokens = self.token_embedding(x)
		
		b, n_a, t, e = x.size()
		if not self.comp:
			positions = self.pos_embedding(torch.arange(t+1, device=(self.device if self.device is not None else d())))[None, :, :].expand(b*n_a, t+1, e)
			x = x.view(b*n_a, t, e)
			x = torch.cat([self.summary_embedding.unsqueeze(0).repeat(b*n_a, 1, 1).to(self.device), x], dim=1) + positions
		else:
			positions = self.pos_embedding(torch.arange(t+1, device=(self.device if self.device is not None else d())))[None, :, :].expand(b*n_a, t+1, self.comp_emb)
			x = self.compress_input(x).view(b*n_a, t, self.comp_emb)
			x = torch.cat([self.summary_embedding.unsqueeze(0).repeat(b*n_a, 1, 1).to(self.device), x], dim=1) + positions

		# x = self.do(x)

		x = self.tblocks(x)

		temporal_weights, agent_weights = [], []

		i = 0
		while i<len(self.tblocks):
			# even numbers have temporal attention
			temporal_weights.append(self.tblocks[i].weights)
			i += 1
			if self.agent_attn:
				# odd numbers have agent attention
				agent_weights.append(self.tblocks[i+1].weights)
				i += 1

				
		x = self.rblocks(x).view(b, n_a, t, 50).contiguous().transpose(1,2).contiguous().sum(2) ###shape (b, t, 50)
				
		x_episode_wise = self.toreward(x[:, 0, :]).view(b, t).contiguous()
		
		# x_episode_wise = x_time_wise.sum(1)

		# return x_episode_wise, x_time_wise

		return x_episode_wise, temporal_weights, agent_weights



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
