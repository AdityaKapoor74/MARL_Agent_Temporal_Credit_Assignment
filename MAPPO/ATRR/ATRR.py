import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

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


class ImportanceSamplingHyperNetwork(nn.Module):
	def __init__(self, num_agents, hidden_dim, total_obs_dim):
		super(ImportanceSamplingHyperNetwork, self).__init__()
		self.num_agents = num_agents
		self.hidden_dim = hidden_dim

		self.hyper_w1 = nn.Sequential(
			init_(nn.Linear(total_obs_dim, hidden_dim), activate=True),
			nn.GELU(),
			init_(nn.Linear(hidden_dim, 1))
			# init_(nn.Linear(total_obs_dim, 1), activate=False),
			)

	def forward(self, importance_sampling_ratio, all_agent_state_action, agent_masks):
		b, n_a, t, e = all_agent_state_action.shape
		# importance_sampling_ratio = importance_sampling_ratio.reshape(-1, self.num_agents)
		w1 = self.hyper_w1(all_agent_state_action.transpose(1, 2).reshape(-1, e))

		# scaling the expected importance sampling ratio
		w1 = torch.abs(w1.reshape(-1, self.num_agents)) * agent_masks.reshape(-1, self.num_agents).to(all_agent_state_action.device)
		# x = (w1 * importance_sampling_ratio).prod(dim=-1, keepdim=True)#.clamp(min=1e-1, max=10.0)
		x = torch.bmm(importance_sampling_ratio.reshape(-1, 1, self.num_agents), w1.reshape(-1, self.num_agents, 1))
		x = torch.exp(x)

		return x


class RewardHyperNetwork(nn.Module):
	def __init__(self, num_agents, hidden_dim, total_obs_dim):
		super(RewardHyperNetwork, self).__init__()
		self.num_agents = num_agents
		self.hidden_dim = hidden_dim

		self.hyper_w1 = nn.Sequential(
			init_(nn.Linear(total_obs_dim, hidden_dim), activate=True),
			nn.GELU(),
			init_(nn.Linear(hidden_dim, 1))
			# init_(nn.Linear(total_obs_dim, 1), activate=False),
			)

	def forward(self, expected_rewards, all_agent_state_action, final_multi_agent_state_action, agent_masks):
		b, n_a, t, e = all_agent_state_action.shape
		all_agent_intermediate_final_state_action = (all_agent_state_action + final_multi_agent_state_action.unsqueeze(1)).transpose(1, 2)
		expected_rewards = expected_rewards.reshape(-1, 1, self.num_agents)
		w1 = self.hyper_w1(all_agent_intermediate_final_state_action.reshape(-1, e))

		# scaling the expected reward
		w1 = torch.abs(w1.reshape(-1, self.num_agents, 1) * agent_masks.reshape(-1, self.num_agents, 1).to(all_agent_intermediate_final_state_action.device))
		x = torch.bmm(expected_rewards, w1)

		self.w1 = w1

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
			init_(nn.Linear(ally_obs_shape, self.comp_emb), activate=False),
			)
		self.enemy_obs_compress_input = nn.Sequential(
			init_(nn.Linear(enemy_obs_shape, self.comp_emb), activate=False),
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

		self.dynamics_model = nn.Sequential(
			init_(nn.Linear(self.comp_emb*(depth+1), self.comp_emb), activate=True),
			nn.GELU(),
			init_(nn.Linear(self.comp_emb, n_actions), activate=False)
			)

		self.rblocks = nn.Sequential(
			init_(nn.Linear(self.comp_emb*depth*2, self.comp_emb), activate=True),
			nn.GELU(),
			init_(nn.Linear(self.comp_emb, 1)),
			)

		# self.reward_hyper_net = RewardHyperNetwork(num_agents=self.n_agents, hidden_dim=self.comp_emb, total_obs_dim=self.comp_emb*depth)

		self.importance_sampling_hyper_net = ImportanceSamplingHyperNetwork(num_agents=self.n_agents, hidden_dim=self.comp_emb, total_obs_dim=self.comp_emb*depth)
					   
		self.do = nn.Dropout(dropout)

		self.mask_value = torch.tensor(
				torch.finfo(torch.float).min, dtype=torch.float
			)

			
	def forward(self, ally_obs, enemy_obs, actions, episodic_reward, team_masks=None, agent_masks=None, logprobs=None, train=False):

		"""
		:param x: A (batch, number of agents, sequence length, state dimension) tensor of state sequences.
		:return: predicted log-probability vectors for each token based on the preceding tokens.
		"""
		# tokens = self.token_embedding(x)
		
		
		b, n_a, t, _ = ally_obs.size()
		_, n_e, _, _ = enemy_obs.size()\

		position_embedding = self.position_embedding(torch.arange(t, device=self.device))[None, None, :, :].expand(b, n_a, t, self.comp_emb)
		agent_embedding = self.agent_embedding(torch.arange(self.n_agents, device=self.device))[None, :, None, :].expand(b, n_a, t, self.comp_emb)

		enemy_obs_embedding = (self.enemy_obs_compress_input(enemy_obs)).mean(dim=1, keepdim=True)
		
		ally_obs_embedding = self.ally_obs_compress_input(ally_obs) #+ agent_embedding

		action_embedding = self.action_embedding(actions.long())

		x = (ally_obs_embedding+enemy_obs_embedding+action_embedding+agent_embedding+position_embedding).view(b*n_a, t, self.comp_emb)
		state_action_embedding = x.clone()

		temporal_weights, agent_weights, temporal_scores, agent_scores = [], [], [], []
		i = 0

		x_intermediate = []
		while i < len(self.tblocks):

			# even numbers have temporal attention
			x = self.tblocks[i](x, masks=agent_masks)
			temporal_weights.append(self.tblocks[i].attention.attn_weights)
			temporal_scores.append(self.tblocks[i].attention.attn_scores)

			i += 1

			# keep current context
			x = x + state_action_embedding

			if self.agent_attn:
				# odd numbers have agent attention
				x = self.tblocks[i](x, masks=agent_masks)
				agent_weights.append(self.tblocks[i].attention.attn_weights)
				agent_scores.append(self.tblocks[i].attention.attn_scores)

				i += 1

				# keep current context
				x = x + state_action_embedding


			x_intermediate.append(x)

			if i == len(self.tblocks):
				break

		first_state_embedding = (state_action_embedding.view(b, n_a, t, self.comp_emb) - action_embedding).reshape(b, n_a, t, 1, self.comp_emb)[:, :, 0, :, :].reshape(b, n_a, 1, -1)
		state_embeddings = (state_action_embedding.view(b, n_a, t, self.comp_emb) - action_embedding).reshape(b, n_a, t, self.comp_emb).sum(dim=1, keepdim=True).repeat(1, n_a, 1, 1).reshape(b, n_a, t, -1) / (agent_masks.transpose(1, 2).sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-5)
		state_embeddings = torch.cat([first_state_embedding.to(self.device), state_embeddings[:, :, 1:, :]], dim=2)
		first_past_state_action_embedding = torch.zeros(b, n_a, 1, self.depth*self.comp_emb)
		past_state_action_embeddings = torch.cat([first_past_state_action_embedding.to(self.device), torch.cat(x_intermediate, dim=-1).reshape(b, n_a, t, -1)[:, :, :-1, :]], dim=-2)
		state_past_state_action_embeddings = torch.cat([state_embeddings, past_state_action_embeddings], dim=-1)
		action_prediction = self.dynamics_model(state_past_state_action_embeddings)

		# to ensure masking across rows and columns
		agent_weights = torch.stack(agent_weights, dim=0).reshape(self.depth, b, t, n_a, n_a) * agent_masks.unsqueeze(0).unsqueeze(-1) * agent_masks.unsqueeze(0).unsqueeze(-2)
		temporal_weights = torch.stack(temporal_weights, dim=0).reshape(self.depth, b, n_a, t, t) * agent_masks.permute(0, 2, 1).unsqueeze(0).unsqueeze(-1) * agent_masks.permute(0, 2, 1).unsqueeze(0).unsqueeze(-2)
		agent_scores = torch.stack(agent_scores, dim=0).reshape(self.depth, b, self.heads, t, n_a, n_a) * agent_masks.unsqueeze(0).unsqueeze(2).unsqueeze(-1) * agent_masks.unsqueeze(0).unsqueeze(2).unsqueeze(-2)
		temporal_scores = torch.stack(temporal_scores, dim=0).reshape(self.depth, b, self.heads, n_a, t, t) * agent_masks.permute(0, 2, 1).unsqueeze(0).unsqueeze(2).unsqueeze(-1) * agent_masks.permute(0, 2, 1).unsqueeze(0).unsqueeze(2).unsqueeze(-2)

		
		indiv_agent_episode_len = (agent_masks.sum(dim=-2)-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.comp_emb*self.depth).long() # subtracting 1 for indexing purposes
		all_x = torch.cat(x_intermediate, dim=-1).reshape(b, n_a, t, -1)
		final_x = torch.gather(all_x, 2, indiv_agent_episode_len).squeeze(2)

		# returns = F.relu(self.rblocks(all_x).view(b, n_a, t).contiguous().transpose(1, 2)  * agent_masks.to(self.device) * torch.sign(episodic_reward.to(self.device).reshape(b, 1, 1)))
		# returns = F.relu(self.rblocks(torch.cat([all_x, final_x.mean(dim=1, keepdim=True).unsqueeze(1).repeat(1, n_a, t, 1)], dim=-1)).view(b, n_a, t).contiguous().transpose(1, 2)  * agent_masks.to(self.device) * torch.sign(episodic_reward.to(self.device).reshape(b, 1, 1)))
		# rewards_ = returns.detach()
		# importance_sampling = None
		
		# # expected rewards given a state-action embedding are readjusted using the final multi-agent outcome
		returns = F.relu(self.rblocks(torch.cat([all_x, final_x.mean(dim=1, keepdim=True).unsqueeze(1).repeat(1, n_a, t, 1)], dim=-1)).view(b, n_a, t).contiguous().transpose(1, 2)  * agent_masks.to(self.device) * torch.sign(episodic_reward.to(self.device).reshape(b, 1, 1)))
		gen_policy_probs = Categorical(F.softmax(action_prediction.detach().transpose(1, 2), dim=-1))
		gen_policy_logprobs = gen_policy_probs.log_prob(actions.transpose(1, 2).to(self.device))
		# # use hypernet for importance sampling
		importance_sampling = ((logprobs.to(self.device) - gen_policy_logprobs.to(self.device)) * agent_masks.to(self.device))
		importance_sampling = self.importance_sampling_hyper_net(importance_sampling.detach(), all_x, agent_masks).reshape(b, t)
		rewards_ = returns.detach() * importance_sampling.unsqueeze(-1).detach()

		return returns, rewards_, importance_sampling, temporal_weights, agent_weights, temporal_scores, agent_scores, action_prediction



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
