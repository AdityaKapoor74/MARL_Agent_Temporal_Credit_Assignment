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
		self.comp_emb = 128
		# if emb>100:
		# 	self.comp_emb = 100
		# else:
		# 	self.comp_emb = emb
		print(self.comp_emb, '-'*50)

		# seq_length = seq_length + 1 # adding 1 for CLS like token embedding

		self.depth = depth
		self.agent_attn = agent

		if not comp:
			# 0: CLS for agent & 1: CLS for temporal
			self.summary_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=2).to(self.device)

			self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length+1).to(self.device)
		
			tblocks = []
			for i in range(depth):
				tblocks.append(
					TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=True, dropout=dropout, wide=wide))
				if agent:
					tblocks.append(
						# adding an extra agent which is analogous to CLS token
						TransformerBlock_Agent(emb=emb, heads=heads, seq_length=seq_length, n_agents= n_agents+1,
						mask=False, dropout=dropout, wide=wide))

			self.tblocks = nn.Sequential(*tblocks)

			self.final_temporal_block = TransformerBlock(emb=emb, heads=heads, seq_length=seq_length+1, mask=True, dropout=dropout, wide=wide)

			self.toreward = nn.Linear(emb, 1)

			self.do = nn.Dropout(dropout)
		else:
			self.compress_input = nn.Linear(emb, self.comp_emb)

			# 0: CLS for agent & 1: CLS for temporal
			self.summary_embedding = nn.Embedding(embedding_dim=self.comp_emb, num_embeddings=2).to(self.device)

			self.pos_embedding = nn.Embedding(embedding_dim=self.comp_emb, num_embeddings=seq_length+1).to(self.device)


			tblocks = []
			for i in range(depth):
				tblocks.append(
					TransformerBlock(emb=self.comp_emb, heads=heads, seq_length=seq_length, mask=True, dropout=dropout, wide=wide))
				if agent:
					tblocks.append(
						# adding an extra agent which is analogous to CLS token
						TransformerBlock_Agent(emb=self.comp_emb, heads=heads, seq_length=seq_length, n_agents= n_agents+1,
						mask=False, dropout=dropout, wide=wide))

			self.tblocks = nn.Sequential(*tblocks)

			self.final_temporal_block = TransformerBlock(emb=self.comp_emb, heads=heads, seq_length=seq_length+1, mask=True, dropout=dropout, wide=wide)
			
			rblocks = []

			rblocks.append(nn.Linear(self.comp_emb, 64))
			
			rblocks.append(nn.Linear(64, 64))
			
			self.rblocks = nn.Sequential(*rblocks)
										   
			self.toreward = nn.Linear(64, 1)

			self.do = nn.Dropout(dropout)
			
	def forward(self, x, team_masks=None, agent_masks=None):

		"""
		:param x: A (batch, number of agents, sequence length, state dimension) tensor of state sequences.
		:return: predicted log-probability vectors for each token based on the preceding tokens.
		"""
		# tokens = self.token_embedding(x)
		
		b, n_a, t, e = x.size()
		if not self.comp:
			positions = self.pos_embedding(torch.arange(t+1, device=(self.device if self.device is not None else d()))[:t])[None, :, :].expand(b*(n_a+1), t, e)
			x = torch.cat([self.summary_embedding(torch.LongTensor([0]).to(self.device)).unsqueeze(0).unsqueeze(0).repeat(b, 1, t, 1).to(self.device), x], dim=1)
			x = x.view(b*(n_a+1), t, e) + positions
		else:
			positions = self.pos_embedding(torch.arange(t+1, device=(self.device if self.device is not None else d()))[:t])[None, :, :].expand(b*(n_a+1), t, self.comp_emb)
			x = self.compress_input(x)
			x = torch.cat([self.summary_embedding(torch.LongTensor([0]).to(self.device)).unsqueeze(0).unsqueeze(0).repeat(b, 1, t, 1).to(self.device), x], dim=1).view(b*(n_a+1), t, self.comp_emb) + positions

		# x = self.do(x)

		temporal_weights, agent_weights, temporal_scores, agent_scores = [], [], [], []

		i = 0

		while i < len(self.tblocks):
			# even numbers have temporal attention
			x = self.tblocks[i](x, masks=agent_masks)
			# temporal_weights.append(self.tblocks[i].attention.attn_weights)
			temporal_scores.append(self.tblocks[i].attention.attn_scores)
			i += 1
			if self.agent_attn:
				# odd numbers have agent attention
				x = self.tblocks[i](x, masks=agent_masks)
				agent_weights.append(self.tblocks[i].attention.attn_weights)
				agent_scores.append(self.tblocks[i].attention.attn_scores)
				i += 1

		x = torch.cat([x.view(b, n_a+1, t, -1)[:, 0, :, :], (self.pos_embedding(torch.LongTensor([t]).to(self.device))+self.summary_embedding(torch.LongTensor([1]).to(self.device)).to(self.device)).to(self.device).unsqueeze(0).repeat(b, 1, 1)], dim=1)
		
		x = self.final_temporal_block(x, team_masks, temporal_only=True)

		x = self.rblocks(x[:, -1, :])
				
		x_episode_wise = self.toreward(x).view(b, 1).contiguous()

		temporal_weights = self.final_temporal_block.attention.attn_weights[:, -1, :-1] * team_masks[: , :-1]
		temporal_scores = torch.stack(temporal_scores, dim=0)
		# print(temporal_scores.shape)

		agent_weights = torch.stack(agent_weights, dim=0).reshape(self.depth, b, t, n_a+1, n_a+1)[:, :, :, 0, 1:].permute(1, 2, 0, 3).mean(dim=-2) * agent_masks[: , :, 1:]
		agent_scores = torch.stack(agent_scores, dim=0)
		# print(agent_scores.shape)

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
