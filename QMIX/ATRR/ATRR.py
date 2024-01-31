import torch
from torch import nn
import torch.nn.functional as F

from .modules import TransformerBlock, TransformerBlock_Agent

from .util import d

def init(module, weight_init, bias_init, gain=1):
	if isinstance(module, nn.LayerNorm):
		init.ones_(module.weight)
		if module.bias is not None:
			init.zeros_(module.bias)
	elif isinstance(module, nn.Linear):
		# weight_init(module.weight.data, gain=gain)
		weight_init(module.weight.data)
		if module.bias is not None:
			bias_init(module.bias.data)
	return module

def init_(m, gain=0.01, activate=False):
	if activate:
		gain = nn.init.calculate_gain('relu')
	return init(m, nn.init.kaiming_uniform_, lambda x: nn.init.constant_(x, 0), gain=gain)


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
		agent=True, 
		dropout=0.0, 
		wide=True, 
		comp="no_compression", 
		hypernet_hidden_dim=128,
		hypernet_final_dim=128,
		linear_compression_dim=128,
		device=None
		):
		super().__init__()

		self.comp = comp
		self.n_agents = n_agents
		self.device = device

		self.obs_shape = obs_shape
		self.action_shape = action_shape
		if comp == "linear_compression":
			self.comp_emb = linear_compression_dim
		else:
			self.comp_emb = hypernet_final_dim

		print(self.comp_emb, '-'*50)

		self.depth = depth
		self.agent_attn = agent

		if comp == "no_compression":
			# one temporal embedding for each agent
			self.temporal_summary_embedding = nn.Embedding(embedding_dim=obs_shape+action_shape, num_embeddings=1).to(self.device)

			self.pos_embedding = nn.Embedding(embedding_dim=obs_shape+action_shape, num_embeddings=seq_length+1).to(self.device)
		
			tblocks = []
			for i in range(depth):
				tblocks.append(
					TransformerBlock(emb=obs_shape+action_shape, heads=heads, seq_length=seq_length, mask=True, dropout=dropout, wide=wide))
				if agent:
					tblocks.append(
						# adding an extra agent which is analogous to CLS token
						# TransformerBlock_Agent(emb=emb, heads=heads, seq_length=seq_length, n_agents= n_agents+1,
						TransformerBlock_Agent(emb=obs_shape+action_shape, heads=heads, seq_length=seq_length, n_agents= n_agents,
						mask=False, dropout=dropout, wide=wide))

			self.tblocks = nn.Sequential(*tblocks)

			self.final_temporal_block = TransformerBlock(emb=emb, heads=heads, seq_length=seq_length+1, mask=True, dropout=dropout, wide=wide)

			self.rblocks = nn.Sequential(
				init_(nn.Linear(emb, 64), activate=True),
				nn.GELU(),
				init_(nn.Linear(64, 64), activate=True),
				nn.GELU(),
				init_(nn.Linear(64, 1))
				)

			self.do = nn.Dropout(dropout)
		elif comp == "linear_compression":
			self.compress_input = nn.Sequential(
					init_(nn.Linear(obs_shape, self.comp_emb), activate=True),
					nn.GELU(),
					# nn.LayerNorm(self.comp_emb),
					)

			# one temporal embedding for each agent
			self.temporal_summary_embedding = nn.Embedding(embedding_dim=self.comp_emb+action_shape, num_embeddings=1).to(self.device)

			self.pos_embedding = nn.Embedding(embedding_dim=self.comp_emb+action_shape, num_embeddings=seq_length+1).to(self.device)


			tblocks = []
			for i in range(depth):
				tblocks.append(
					TransformerBlock(emb=self.comp_emb+action_shape, heads=heads, seq_length=seq_length+1, mask=True, dropout=dropout, wide=wide))
				if agent:
					tblocks.append(
						# adding an extra agent which is analogous to CLS token
						# TransformerBlock_Agent(emb=self.comp_emb, heads=heads, seq_length=seq_length, n_agents= n_agents+1,
						TransformerBlock_Agent(emb=self.comp_emb+action_shape, heads=heads, seq_length=seq_length+1, n_agents=n_agents,
						mask=False, dropout=dropout, wide=wide))

			self.tblocks = nn.Sequential(*tblocks)

			self.final_temporal_block = TransformerBlock(emb=self.comp_emb+action_shape, heads=heads, seq_length=seq_length+1, mask=True, dropout=dropout, wide=wide)
			
			self.rblocks = nn.Sequential(
				init_(nn.Linear(self.comp_emb+action_shape, 64), activate=True),
				nn.GELU(),
				init_(nn.Linear(64, 64), activate=True),
				nn.GELU(),
				init_(nn.Linear(64, 1))
				)
										   
			self.do = nn.Dropout(dropout)

		elif comp == "hypernet_compression":

			self.hypernet = HyperNetwork(action_shape, hypernet_hidden_dim, hypernet_final_dim, obs_shape)

			# one temporal embedding for each agent
			self.temporal_summary_embedding = nn.Embedding(embedding_dim=self.comp_emb, num_embeddings=1).to(self.device)

			self.pos_embedding = nn.Embedding(embedding_dim=self.comp_emb, num_embeddings=seq_length+1).to(self.device)


			tblocks = []
			for i in range(depth):
				tblocks.append(
					TransformerBlock(emb=self.comp_emb, heads=heads, seq_length=seq_length+1, mask=True, dropout=dropout, wide=wide))
				if agent:
					tblocks.append(
						# adding an extra agent which is analogous to CLS token
						# TransformerBlock_Agent(emb=self.comp_emb, heads=heads, seq_length=seq_length, n_agents= n_agents+1,
						TransformerBlock_Agent(emb=self.comp_emb, heads=heads, seq_length=seq_length+1, n_agents=n_agents,
						mask=False, dropout=dropout, wide=wide))

			self.tblocks = nn.Sequential(*tblocks)

			self.final_temporal_block = TransformerBlock(emb=self.comp_emb, heads=heads, seq_length=seq_length+1, mask=True, dropout=dropout, wide=wide)
			
			self.rblocks = nn.Sequential(
				init_(nn.Linear(self.comp_emb, 64), activate=True),
				nn.GELU(),
				init_(nn.Linear(64, 64), activate=True),
				nn.GELU(),
				init_(nn.Linear(64, 1))
				)

			self.do = nn.Dropout(dropout)
			
	def forward(self, obs, one_hot_actions, team_masks=None, agent_masks=None):

		"""
		:param x: A (batch, number of agents, sequence length, state dimension) tensor of state sequences.
		:return: predicted log-probability vectors for each token based on the preceding tokens.
		"""
		# tokens = self.token_embedding(x)
		
		
		if self.comp == "no_compression":
			x = torch.cat([obs, one_hot_actions], dim=-1).to(self.device)
			b, n_a, t, e = x.size()
			positions = self.pos_embedding(torch.arange(t+1, device=(self.device if self.device is not None else d())))[None, :, :].expand(b*n_a, t+1, e)
			# concatenate temporal embedding for each agent
			x = torch.cat([x, self.temporal_summary_embedding(torch.tensor([0]).to(self.device)).unsqueeze(0).unsqueeze(-2).expand(b, n_a, 1, e)], dim=-2)
			x = x.view(b*n_a, t+1, e) + positions
		elif self.comp == "linear_compression":
			b, n_a, t, _ = obs.size()
			positions = self.pos_embedding(torch.arange(t+1, device=(self.device if self.device is not None else d())))[None, :, :].expand(b*n_a, t+1, self.comp_emb+self.action_shape)
			x = self.compress_input(obs)
			x = torch.cat([x, one_hot_actions], dim=-1)
			b, n_a, t, e = x.size()
			# concatenate temporal embedding for each agent
			x = torch.cat([x, self.temporal_summary_embedding(torch.tensor([0]).to(self.device)).unsqueeze(0).unsqueeze(-2).expand(b, n_a, 1, e)], dim=-2).view(b*n_a, t+1, e) + positions
		elif self.comp == "hypernet_compression":
			b, n_a, t, _ = obs.size()
			positions = self.pos_embedding(torch.arange(t+1, device=(self.device if self.device is not None else d())))[None, :, :].expand(b*n_a, t+1, self.comp_emb)
			x = self.hypernet(one_hot_actions, obs).view(b, n_a, t, -1)
			b, n_a, t, e = x.size()
			x = torch.cat([x, self.temporal_summary_embedding(torch.tensor([0]).to(self.device)).unsqueeze(0).unsqueeze(-2).expand(b, n_a, 1, e)], dim=-2).view(b*n_a, t+1, e) + positions
		
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

		# x = torch.cat([x.view(b, n_a+1, t, -1)[:, 0, :, :], (self.pos_embedding(torch.LongTensor([t]).to(self.device))+self.summary_embedding(torch.LongTensor([1]).to(self.device)).to(self.device)).to(self.device).unsqueeze(0).repeat(b, 1, 1)], dim=1)
		# x = torch.cat([x.view(b, n_a, t, -1).sum(dim=1), (self.pos_embedding(torch.LongTensor([t]).to(self.device))+self.summary_embedding(torch.LongTensor([0]).to(self.device)).to(self.device)).to(self.device).unsqueeze(0).repeat(b, 1, 1)], dim=1)
		
		x = x.view(b, n_a, t+1, -1).mean(dim=1)

		x = self.final_temporal_block(x, team_masks, temporal_only=True)

		x_episode_wise = self.rblocks(x[:, -1, :]).view(b, 1).contiguous()

		temporal_weights = self.final_temporal_block.attention.attn_weights[:, -1, :-1] * team_masks[: , :-1]

		temporal_scores = torch.stack(temporal_scores, dim=0)
		# print(temporal_scores.shape)

		# agent_weights = torch.stack(agent_weights, dim=0).reshape(self.depth, b, t, n_a+1, n_a+1)[:, :, :, 0, 1:].permute(1, 2, 0, 3).mean(dim=-2) * agent_masks[: , :, 1:]
		agent_weights = torch.stack(agent_weights, dim=0).reshape(self.depth, b, t+1, n_a, n_a).permute(1, 2, 0, 3, 4).mean(dim=-3).sum(dim=-2)/(agent_masks.sum(dim=-1).unsqueeze(-1)+1e-5) * agent_masks

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
