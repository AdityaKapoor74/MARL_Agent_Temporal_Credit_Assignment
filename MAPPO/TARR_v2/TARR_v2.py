import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from .modules import *

class ShapelyAttention(nn.Module):
	def __init__(self, emb_dim, n_heads, n_agents, sample_num, device, dropout=0.0):
		super().__init__()
		self.emb_dim = emb_dim
		self.device = device
		self.n_agents = n_agents
		self.sample_num = sample_num
		self.phi = MultiAgentAttention(emb_dim, n_heads, n_agents, dropout, device)

		self.agent_embedding = nn.Embedding(self.n_agents, emb_dim)
	
	def get_attn_mask(self, shape):
		# use mask to estimate particular type of coalition
		mask = torch.bernoulli(torch.full((shape, shape), 0.5))
		mask = mask - torch.diag(torch.diag(mask)) + torch.eye(shape)

		return mask.to(self.device)

	def forward(self, input, agent_temporal_mask):
		"""
		:param input: A (batch, number of agents, sequence length, emb dimension) tensor of input sequences.
		:return: deltas, the encoding of time-adjacent states and actions along the agents. (batch, number of agents, sequence length, action dimension).
		"""
		b, n_a, t, e = input.size()
		input = input.permute(0, 2, 1, 3).contiguous().reshape(b*t, n_a, -1)
		coalition = np.arange(self.n_agents)
		np.random.shuffle(coalition)
		agent_embedding = self.agent_embedding(torch.tensor(coalition).to(self.device))[None, :, :].expand(b*t, n_a, self.emb_dim)
		input = input + agent_embedding
		shapley_reward = []

		for i in range(self.sample_num):
			attn_mask = self.get_attn_mask(n_a).unsqueeze(0).repeat(b*t, 1, 1) #* agent_temporal_mask.reshape(b*t, n_a, 1) * agent_temporal_mask.reshape(b*t, 1, n_a)
			marginal_reward, _ = self.phi(input, input, input, attn_mask)
			shapley_reward.append(marginal_reward)

		shapley_reward = sum(shapley_reward)/self.sample_num
		shapley_reward = shapley_reward.reshape(b, t, n_a, -1).permute(0, 2, 1, 3)

		return shapley_reward
		


class TARR(nn.Module):
	def __init__(self, environment, ally_obs_shape, enemy_obs_shape, obs_shape, n_actions, emb_dim, n_heads, n_layer, seq_length, n_agents, n_enemies, sample_num,
				device, dropout=0.0, emb_dropout=0.5, action_space='discrete'):
		super().__init__()

		self.environment = environment
		self.ally_obs_shape = ally_obs_shape
		self.enemy_obs_shape = enemy_obs_shape
		self.obs_shape = obs_shape
		self.emb_dim = emb_dim
		self.n_heads = n_heads
		self.n_layer = n_layer
		self.seq_length = seq_length
		self.sample_num = sample_num
		self.device = device
		self.n_agents = n_agents
		self.emb_dropout = emb_dropout


		if "StarCraft" in self.environment:
			self.ally_obs_compress_input = nn.Sequential(
				nn.Linear(ally_obs_shape, self.emb_dim),
				)
			self.enemy_obs_compress_input = nn.Sequential(
				nn.Linear(enemy_obs_shape, self.emb_dim),
				)
		elif "GFootball" in self.environment:
			self.ally_obs_compress_input = nn.Sequential(
				nn.Linear(ally_obs_shape, self.emb_dim),
				)
			self.common_obs_compress_input = nn.Sequential(
				nn.Linear(obs_shape, self.emb_dim),
				)

		if not action_space == 'discrete':
			self.action_emb = nn.Linear(input_dim, emb_dim)
		else:
			self.action_emb = nn.Embedding(n_actions+1, emb_dim)

		self.pos_embedding = nn.Embedding(seq_length, emb_dim)

		self.layers = nn.ModuleList([nn.ModuleList([EncoderLayer(self.emb_dim, self.n_heads, self.emb_dim, emb_dropout),
								ShapelyAttention(emb_dim, n_heads, self.n_agents, self.sample_num, device, emb_dropout)]) for _ in range(self.n_layer)])

		self.dynamics_model = nn.Sequential(
			nn.Linear(self.emb_dim*(self.n_layer+1), self.emb_dim),
			# nn.Linear(self.emb_dim*3, self.emb_dim),
			nn.GELU(),
			nn.Linear(self.emb_dim, n_actions),
			)


		self.reward_prediction = nn.Sequential(
			nn.Linear(emb_dim*self.n_layer, emb_dim),
			nn.GELU(),
			nn.Linear(emb_dim, 1),
			)
		# self.reward_magnitude = nn.Sequential(
		# 	nn.Linear(emb_dim*self.n_layer*2, emb_dim),
		# 	nn.GELU(),
		# 	nn.Linear(emb_dim, 1),
		# 	nn.ReLU(),
		# 	)

		# self.reward_sign = nn.Sequential(
		# 	nn.Linear(emb_dim*self.n_layer, emb_dim),
		# 	nn.GELU(),
		# 	nn.Linear(emb_dim, 3)
		# 	)

	def get_time_mask(self, episode_length):
		mask = (torch.arange(self.seq_length)[None, :].to(self.device) < episode_length[:, None]).float()
		mask = torch.triu(torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))).transpose(-1, -2)
		# mask = (torch.arange(self.seq_length)[None, None, :].to(self.device) < episode_length[:, :, None]).float()
		# b, n_a, t = mask.shape
		# mask = torch.triu(torch.bmm(mask.reshape(b*n_a, t).unsqueeze(-1), mask.reshape(b*n_a, t).unsqueeze(1)))
		return mask


	def forward(self, ally_states, enemy_states, states, actions, episode_length, episodic_reward, agent_temporal_mask):
		# b, n_a, t, e = ally_states.size()

		if "StarCraft" in self.environment:
			b, n_a, t, _ = ally_states.size()
			_, n_e, _, _ = enemy_states.size()
			enemy_obs_embedding = (self.enemy_obs_compress_input(enemy_states)).mean(dim=1, keepdim=True).repeat(1, n_a, 1, 1)
			ally_obs_embedding = self.ally_obs_compress_input(ally_states)
		elif "GFootball" in self.environment:
			b, n_a, t, _ = ally_states.size()
			ally_obs_embedding = self.ally_obs_compress_input(ally_states)
			common_obs_embedding = self.common_obs_compress_input(states)
			ally_obs_embedding = ally_obs_embedding + common_obs_embedding.unsqueeze(1)

		positions = self.pos_embedding(torch.arange(self.seq_length, device=self.device))[None, None, :, :].expand(b, n_a, self.seq_length, self.emb_dim)
		actions_embed = self.action_emb(actions.long()).squeeze()
		
		if "StarCraft" in self.environment:
			x = (ally_obs_embedding+enemy_obs_embedding+actions_embed+positions)
		elif "GFootball" in self.environment:
			x = (ally_obs_embedding+actions_embed+positions)

		state_action_embedding = x.clone()

		time_mask = self.get_time_mask(episode_length).repeat(n_a, 1, 1)
		# time_mask = self.get_time_mask(episode_length)
		x = x.reshape(b*n_a, t, -1).squeeze()

		x_intermediate = []
		temporal_weights, agent_weights, temporal_scores, agent_scores = [], [], [], []

		for layer in self.layers:
			x, _ = layer[0](x, time_mask)
			temporal_scores.append(layer[0].self_attn.temporal_scores)
			temporal_weights.append(layer[0].self_attn.temporal_weights)
			x = x.reshape(b, n_a, t, -1)
			x = layer[1](x, agent_temporal_mask=None)
			agent_scores.append(layer[1].phi.agent_scores)
			agent_weights.append(layer[1].phi.agent_weights)
			x = x.reshape(b*n_a, t, -1).squeeze()
			x_intermediate.append(x)

		# to ensure masking across rows and columns
		agent_weights = torch.stack(agent_weights, dim=0).reshape(self.n_layer, b, t, n_a, n_a) * agent_temporal_mask.unsqueeze(0).unsqueeze(-1) * agent_temporal_mask.unsqueeze(0).unsqueeze(-2)
		temporal_weights = torch.stack(temporal_weights, dim=0).reshape(self.n_layer, b, n_a, t, t) * agent_temporal_mask.permute(0, 2, 1).unsqueeze(0).unsqueeze(-1) * agent_temporal_mask.permute(0, 2, 1).unsqueeze(0).unsqueeze(-2)
		agent_scores = torch.stack(agent_scores, dim=0).reshape(self.n_layer, b, self.n_heads, t, n_a, n_a) * agent_temporal_mask.unsqueeze(0).unsqueeze(2).unsqueeze(-1) * agent_temporal_mask.unsqueeze(0).unsqueeze(2).unsqueeze(-2)
		temporal_scores = torch.stack(temporal_scores, dim=0).reshape(self.n_layer, b, self.n_heads, n_a, t, t) * agent_temporal_mask.permute(0, 2, 1).unsqueeze(0).unsqueeze(2).unsqueeze(-1) * agent_temporal_mask.permute(0, 2, 1).unsqueeze(0).unsqueeze(2).unsqueeze(-2)

		x_intermediate = torch.cat(x_intermediate, dim=-1).reshape(b, n_a, t, -1)

		# normal inverse dynamics model
		first_state_embedding = (state_action_embedding.view(b, n_a, t, self.emb_dim) - actions_embed).reshape(b, n_a, t, 1, self.emb_dim)[:, :, 0, :, :].reshape(b, n_a, 1, -1)
		state_embeddings = (state_action_embedding.view(b, n_a, t, self.emb_dim) - actions_embed).reshape(b, n_a, t, self.emb_dim).sum(dim=1, keepdim=True).repeat(1, n_a, 1, 1).reshape(b, n_a, t, -1) / (agent_temporal_mask.transpose(1, 2).sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-5)
		state_embeddings = torch.cat([first_state_embedding.to(self.device), state_embeddings[:, :, 1:, :]], dim=2)
		# using agent specific embeddings only
		# first_past_state_action_embedding = torch.zeros(b, n_a, 1, self.emb_dim)
		# past_state_action_embeddings = torch.cat([first_past_state_action_embedding.to(self.device), state_action_embedding[:, :, :-1, :]], dim=-2)
		# state_past_state_action_embeddings = torch.cat([state_embeddings, past_state_action_embeddings], dim=-1)
		# using intermediate embeddings
		first_past_state_action_embedding = torch.zeros(b, n_a, 1, self.n_layer*self.emb_dim)
		past_state_action_embeddings = torch.cat([first_past_state_action_embedding.to(self.device), x_intermediate[:, :, :-1, :]], dim=-2)
		state_past_state_action_embeddings = torch.cat([state_embeddings, past_state_action_embeddings], dim=-1)
		action_prediction = self.dynamics_model(state_past_state_action_embeddings)

		# Hindsight inverse dynamics model
		# first_state_embedding = (state_action_embedding.view(b, n_a, t, self.emb_dim) - actions_embed).reshape(b, n_a, t, 1, self.emb_dim)[:, :, 0, :, :].reshape(b, n_a, 1, -1)
		# state_embeddings = (state_action_embedding.view(b, n_a, t, self.emb_dim) - actions_embed).reshape(b, n_a, t, self.emb_dim).sum(dim=1, keepdim=True).repeat(1, n_a, 1, 1).reshape(b, n_a, t, -1) / (agent_temporal_mask.transpose(1, 2).sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-5)
		# state_embeddings = torch.cat([first_state_embedding.to(self.device), state_embeddings[:, :, 1:, :]], dim=2)
		# upper_triangular_mask = torch.triu(torch.ones(b*n_a, t, t)).reshape(b, n_a, t, t, 1).to(self.device)
		# using agent specific embeddings only
		# first_past_state_action_embedding = torch.zeros(b, n_a, 1, self.emb_dim)
		# past_state_action_embeddings = torch.cat([first_past_state_action_embedding.to(self.device), state_action_embedding[:, :, :-1, :]], dim=-2)
		# state_past_state_action_embeddings = torch.cat([state_embeddings, past_state_action_embeddings], dim=-1)
		# x_goal_states = (state_embeddings*agent_temporal_mask.transpose(1, 2).unsqueeze(-1)).unsqueeze(-3).repeat(1, 1, t, 1, 1)
		# state_past_state_action_embeddings = (state_past_state_action_embeddings*agent_temporal_mask.transpose(1, 2).unsqueeze(-1)).unsqueeze(-2).repeat(1, 1, 1, t, 1)
		# current_context_goal = torch.cat([state_past_state_action_embeddings, x_goal_states], dim=-1) * upper_triangular_mask # b, n_a, t, t, -1
		# using intermediate embeddings
		# first_past_state_action_embedding = torch.zeros(b, n_a, 1, self.n_layer*self.emb_dim)
		# past_state_action_embeddings = torch.cat([first_past_state_action_embedding.to(self.device), x_intermediate[:, :, :-1, :]], dim=-2)
		# state_past_state_action_embeddings = torch.cat([state_embeddings, past_state_action_embeddings], dim=-1) # b, n_a, t, -1
		# x_goal_states = (x_intermediate*agent_temporal_mask.transpose(1, 2).unsqueeze(-1)).unsqueeze(-3).repeat(1, 1, t, 1, 1)
		# state_past_state_action_embeddings = (state_past_state_action_embeddings*agent_temporal_mask.transpose(1, 2).unsqueeze(-1)).unsqueeze(-2).repeat(1, 1, 1, t, 1)
		# current_context_goal = torch.cat([state_past_state_action_embeddings, x_goal_states], dim=-1) * upper_triangular_mask # b, n_a, t, t, -1
		
		# action_prediction = self.dynamics_model(current_context_goal)

		# no inverse dynamics model
		# action_prediction = None


		# indiv_agent_episode_len = (agent_temporal_mask.sum(dim=-2)-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.emb_dim*self.n_layer).long() # subtracting 1 for indexing purposes
		# final_x = torch.gather(x_intermediate, 2, indiv_agent_episode_len).squeeze(2)

		# reward_prediction_embeddings = torch.cat([x_intermediate, final_x.mean(dim=1, keepdim=True).unsqueeze(1).repeat(1, n_a, t, 1)], dim=-1)
		# rewards = F.relu(self.linear(reward_prediction_embeddings).view(b, n_a, t).contiguous().transpose(1, 2) * agent_temporal_mask.to(self.device) * torch.sign(episodic_reward.to(self.device).reshape(b, 1, 1))) * torch.sign(episodic_reward.to(self.device).reshape(b, 1, 1))
		# rewards = F.relu(self.linear(reward_prediction_embeddings).view(b, n_a, t).contiguous().transpose(1, 2) * agent_temporal_mask.to(self.device))
		
		# rewards = F.relu(self.linear(x_intermediate).view(b, n_a, t).contiguous().transpose(1, 2) * agent_temporal_mask.to(self.device) * torch.sign(episodic_reward.to(self.device).reshape(b, 1, 1))) * torch.sign(episodic_reward.to(self.device).reshape(b, 1, 1))

		# reward_prediction_embeddings = torch.cat([x_intermediate, final_x.mean(dim=1, keepdim=True).unsqueeze(1).repeat(1, n_a, t, 1)], dim=-1)
		# reward_magnitude = self.reward_magnitude(reward_prediction_embeddings).reshape(b, n_a, t).permute(0, 2, 1)
		# reward_sign = self.reward_sign(final_x.mean(dim=1))

		# sign = torch.argmax(reward_sign.clone(), dim=-1) # -ve -- class label 0 / 0 -- class label 1 / +ve -- class label 2 
		# sign[sign==0] = -1.0
		# sign[sign==1] = 0.0
		# sign[sign==2] = 1.0
		# print("SIGN", sign)
		# rewards = reward_magnitude.clone() * sign.reshape(b, 1, 1)

		rewards = self.reward_prediction(x_intermediate).view(b, n_a, t).contiguous().transpose(1, 2) * agent_temporal_mask.to(self.device)

		return rewards, temporal_weights, agent_weights, temporal_scores, agent_scores, action_prediction



# reward_predictor = STAS_ML(input_dim=10, n_actions=12, emb_dim=64, n_heads=4, n_layer=3, seq_length=50, n_agents=5, sample_num=5,
# 				device=torch.device('cpu'), dropout=0.0, emb_dropout=0.3)