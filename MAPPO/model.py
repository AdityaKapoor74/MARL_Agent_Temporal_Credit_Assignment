import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy

class PopArt(nn.Module):
	""" Normalize a vector of observations - across the first norm_axes dimensions"""

	def __init__(self, input_shape, num_agents, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5, device=torch.device("cpu")):
		super(PopArt, self).__init__()

		self.input_shape = input_shape
		self.num_agents = num_agents
		self.norm_axes = norm_axes
		self.epsilon = epsilon
		self.beta = beta
		self.per_element_update = per_element_update
		self.tpdv = dict(dtype=torch.float32, device=device)

		self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
		self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
		self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

	def reset_parameters(self):
		self.running_mean.zero_()
		self.running_mean_sq.zero_()
		self.debiasing_term.zero_()

	def running_mean_var(self):
		debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
		debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
		debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
		return debiased_mean, debiased_var

	def forward(self, input_vector, mask, train=True):
		# Make sure input is float32
		input_vector_device = input_vector.device
		if type(input_vector) == np.ndarray:
			input_vector = torch.from_numpy(input_vector)
		input_vector = input_vector.to(**self.tpdv)

		if train:
			# Detach input before adding it to running means to avoid backpropping through it on
			# subsequent batches.
			detached_input = input_vector.detach()
			# batch_mean = detached_input.mean(dim=tuple(range(self.norm_axes)))
			# batch_sq_mean = (detached_input ** 2).mean(dim=tuple(range(self.norm_axes)))
			batch_mean = detached_input.sum(dim=tuple(range(self.norm_axes)))/mask.sum(dim=tuple(range(self.norm_axes)))
			batch_sq_mean = (detached_input ** 2).sum(dim=tuple(range(self.norm_axes)))/mask.sum(dim=tuple(range(self.norm_axes)))

			if self.per_element_update:
				# batch_size = np.prod(detached_input.size()[:self.norm_axes])
				batch_size = (mask.reshape(-1, self.num_agents).sum(dim=-1)>0.0).sum()
				weight = self.beta ** batch_size
			else:
				weight = self.beta

			self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
			self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
			self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

		mean, var = self.running_mean_var()
		out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
		
		return out.to(input_vector_device)

	def denormalize(self, input_vector):
		""" Transform normalized data back into original distribution """
		input_vector_device = input_vector.device
		if type(input_vector) == np.ndarray:
			input_vector = torch.from_numpy(input_vector)
		input_vector = input_vector.to(**self.tpdv)

		mean, var = self.running_mean_var()
		out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
		
		# out = out.cpu().numpy()
		
		# return out
		return out.to(input_vector_device)


def init(module, weight_init, bias_init, gain=1):
	weight_init(module.weight.data, gain=gain)
	if module.bias is not None:
		bias_init(module.bias.data)
	return module

def init_(m, gain=0.01, activate=False):
	if activate:
		gain = nn.init.calculate_gain('tanh')
	return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class Policy(nn.Module):
	def __init__(
		self, 
		use_recurrent_policy,
		obs_input_dim, 
		num_actions, 
		num_agents, 
		rnn_num_layers, 
		rnn_hidden_actor,
		device
		):
		super(Policy, self).__init__()

		self.use_recurrent_policy = use_recurrent_policy
		self.rnn_num_layers = rnn_num_layers
		self.rnn_hidden_actor = rnn_hidden_actor

		self.mask_value = torch.tensor(
				torch.finfo(torch.float).min, dtype=torch.float
			)
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device

		self.agent_embedding = nn.Embedding(self.num_agents, self.rnn_hidden_actor)
		self.action_embedding = nn.Embedding(self.num_actions+1, self.rnn_hidden_actor) # we assume the first "last action" to be NON-EXISTENT so one of the embedding represents that

		if self.use_recurrent_policy:
			
			self.obs_embedding = nn.Sequential(
				init_(nn.Linear(obs_input_dim, rnn_hidden_actor), activate=True),
				nn.GELU(),
				)

			# self.obs_embed_layer_norm = nn.LayerNorm(self.rnn_hidden_actor)
			
			self.RNN = nn.GRU(input_size=rnn_hidden_actor, hidden_size=rnn_hidden_actor, num_layers=rnn_num_layers, batch_first=True)
			for name, param in self.RNN.named_parameters():
				if 'bias' in name:
					nn.init.constant_(param, 0)
				elif 'weight' in name:
					nn.init.orthogonal_(param)

			self.Layer_2 = nn.Sequential(
				nn.LayerNorm(rnn_hidden_actor),
				init_(nn.Linear(rnn_hidden_actor, num_actions), gain=0.01)
				)

		else:
			self.obs_embedding = nn.Sequential(
				init_(nn.Linear(obs_input_dim, rnn_hidden_actor), activate=True),
				nn.GELU(),
				)

			self.obs_embed_layer_norm = nn.LayerNorm(self.rnn_hidden_actor)

			self.final_layer = nn.Sequential(
				init_(nn.Linear(rnn_hidden_actor, num_actions), gain=0.01)
				)


	def forward(self, local_observations, last_actions, hidden_state, mask_actions):

		batch, timesteps, _, _ = local_observations.shape
		agent_embedding = self.agent_embedding(torch.arange(self.num_agents).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_agents, self.rnn_hidden_actor)
		last_action_embedding = self.action_embedding(last_actions.long())
		obs_embedding = self.obs_embedding(local_observations)
		final_obs_embedding = (obs_embedding + last_action_embedding + agent_embedding).permute(0, 2, 1, 3).reshape(batch*self.num_agents, timesteps, -1) # self.obs_embed_layer_norm(obs_embedding + last_action_embedding + agent_embedding).permute(0, 2, 1, 3).reshape(batch*self.num_agents, timesteps, -1)

		if self.use_recurrent_policy:
			hidden_state = hidden_state.reshape(self.rnn_num_layers, batch*self.num_agents, -1)
			output, h = self.RNN(final_obs_embedding, hidden_state)
			output = output.reshape(batch, self.num_agents, timesteps, -1).permute(0, 2, 1, 3)
			logits = self.Layer_2(output)
		else:
			logits = self.final_layer(local_observations)

		logits = torch.where(mask_actions, logits, self.mask_value)
		return F.softmax(logits, dim=-1), h



class Q_network(nn.Module):
	def __init__(
		self, 
		use_recurrent_critic,
		centralized,
		local_observation_input_dim,
		ally_obs_input_dim, 
		enemy_obs_input_dim,
		num_agents, 
		num_enemies,
		num_actions, 
		rnn_num_layers,
		comp_emb_shape,
		device, 
		):
		super(Q_network, self).__init__()
		
		self.use_recurrent_critic = use_recurrent_critic
		self.centralized = centralized
		self.num_agents = num_agents
		self.num_enemies = num_enemies
		self.num_actions = num_actions
		self.rnn_num_layers = rnn_num_layers
		self.comp_emb_shape = comp_emb_shape
		self.device = device

		self.agent_embedding = nn.Embedding(self.num_agents, self.comp_emb_shape)
		self.action_embedding = nn.Embedding(self.num_actions, self.comp_emb_shape)

		if self.centralized:
			self.enemy_embedding = nn.Embedding(self.num_enemies, self.comp_emb_shape)

			self.ally_obs_embedding = nn.Sequential(
				# nn.LayerNorm(ally_obs_input_dim),
				init_(nn.Linear(ally_obs_input_dim, comp_emb_shape, bias=True), activate=True),
				nn.GELU()
				)
			self.enemy_obs_embedding = nn.Sequential(
				# nn.LayerNorm(enemy_obs_input_dim),
				init_(nn.Linear(enemy_obs_input_dim, comp_emb_shape, bias=True), activate=True),
				nn.GELU()
				)

			# self.state_action_embedding_layer_norm = nn.LayerNorm(comp_emb_shape)

		else:
			self.obs_embedding = nn.Sequential(
				# nn.LayerNorm(local_observation_input_dim),
				init_(nn.Linear(local_observation_input_dim, comp_emb_shape, bias=True), activate=True),
				nn.GELU()
				)

			# self.state_action_embedding_layer_norm = nn.LayerNorm(comp_emb_shape)
				
			self.intermediate_embedding = nn.Sequential(
				init_(nn.Linear(comp_emb_shape, comp_emb_shape, bias=True), activate=True),
				nn.GELU(),
				)

		if self.use_recurrent_critic:
			self.RNN = nn.GRU(input_size=comp_emb_shape, hidden_size=comp_emb_shape, num_layers=self.rnn_num_layers, batch_first=True)
			for name, param in self.RNN.named_parameters():
				if 'bias' in name:
					nn.init.constant_(param, 0)
				elif 'weight' in name:
					nn.init.orthogonal_(param)

		self.q_value_layer = nn.Sequential(
			nn.LayerNorm(comp_emb_shape),
			init_(nn.Linear(comp_emb_shape, 1), activate=False)
			)
		

		self.mask_value = torch.tensor(
			torch.finfo(torch.float).min, dtype=torch.float
			)


	def forward(self, local_observations, ally_states, enemy_states, actions, rnn_hidden_state):
		if self.centralized:
			batch, timesteps, _, _ = ally_states.shape
			agent_embedding = self.agent_embedding(torch.arange(self.num_agents).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_agents, self.comp_emb_shape)
			enemy_embedding = self.enemy_embedding(torch.arange(self.num_enemies).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_enemies, self.comp_emb_shape)
			ally_state_embedding = (self.ally_obs_embedding(ally_states) + agent_embedding + self.action_embedding(actions.long())).sum(dim=-2)
			enemy_state_embedding = (self.enemy_obs_embedding(enemy_states) + enemy_embedding).sum(dim=-2)

			final_state_embedding = ally_state_embedding+enemy_state_embedding # self.state_action_embedding_layer_norm(ally_state_embedding+enemy_state_embedding)

			if self.use_recurrent_critic:
				final_state_embedding, h = self.RNN(final_state_embedding, rnn_hidden_state)

			Q_value = self.q_value_layer(final_state_embedding)

		else:
			batch, timesteps, num_agents, _ = local_observations.shape
			agent_embedding = self.agent_embedding(torch.arange(self.num_agents).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_agents, self.comp_emb_shape)
			obs_embedding = self.obs_embedding(local_observations) + agent_embedding + self.action_embedding(actions.long()) # self.state_action_embedding_layer_norm(self.obs_embedding(local_observations) + agent_embedding + self.action_embedding(actions.long()))
			intermediate_embedding = self.intermediate_embedding(obs_embedding)
			
			if self.use_recurrent_critic:
				intermediate_embedding, h = self.RNN(intermediate_embedding.permute(0, 2, 1, 3).reshape(batch*num_agents, timesteps, -1), rnn_hidden_state)
				intermediate_embedding = intermediate_embedding.reshape(batch, num_agents, timesteps, -1).permute(0, 2, 1, 3).reshape(batch*timesteps, num_agents, -1)
			
			Q_value = self.q_value_layer(intermediate_embedding)

		return Q_value.squeeze(-1), h


class Q_network(nn.Module):
	def __init__(
		self, 
		use_recurrent_critic,
		centralized,
		local_observation_input_dim,
		ally_obs_input_dim, 
		enemy_obs_input_dim,
		num_agents, 
		num_enemies,
		num_actions, 
		rnn_num_layers,
		comp_emb_shape,
		device, 
		):
		super(Q_network, self).__init__()
		
		self.use_recurrent_critic = use_recurrent_critic
		self.centralized = centralized
		self.num_agents = num_agents
		self.num_enemies = num_enemies
		self.num_actions = num_actions
		self.rnn_num_layers = rnn_num_layers
		self.comp_emb_shape = comp_emb_shape
		self.device = device

		self.agent_embedding = nn.Embedding(self.num_agents, self.comp_emb_shape)
		self.action_embedding = nn.Embedding(self.num_actions, self.comp_emb_shape)

		if self.centralized:
			# self.enemy_embedding = nn.Embedding(self.num_enemies, self.comp_emb_shape)

			# self.ally_obs_embedding = nn.Sequential(
			# 	# nn.LayerNorm(ally_obs_input_dim),
			# 	init_(nn.Linear(ally_obs_input_dim, comp_emb_shape, bias=True), activate=True),
			# 	nn.GELU()
			# 	)
			# self.enemy_obs_embedding = nn.Sequential(
			# 	# nn.LayerNorm(enemy_obs_input_dim),
			# 	init_(nn.Linear(enemy_obs_input_dim, comp_emb_shape, bias=True), activate=True),
			# 	nn.GELU()
			# 	)

			# self.state_action_embedding_layer_norm = nn.LayerNorm(comp_emb_shape*2)

			# self.intermediate_embedding = nn.Sequential(
			# 	init_(nn.Linear(comp_emb_shape*2, comp_emb_shape, bias=True), activate=True),
			# 	nn.GELU(),
			# 	)

			self.embedding = nn.Sequential(
				init_(nn.Linear((self.num_agents+ally_obs_input_dim+self.num_actions)*self.num_agents + enemy_obs_input_dim*self.num_enemies, comp_emb_shape, bias=True), activate=True),
				nn.GELU(),
				nn.LayerNorm(comp_emb_shape),
				init_(nn.Linear(comp_emb_shape, comp_emb_shape, bias=True), activate=True),
				nn.GELU(),
				nn.LayerNorm(comp_emb_shape),
				)

		else:
			self.obs_embedding = nn.Sequential(
				# nn.LayerNorm(local_observation_input_dim),
				init_(nn.Linear(local_observation_input_dim, comp_emb_shape, bias=True), activate=True),
				nn.GELU()
				)

			self.state_action_embedding_layer_norm = nn.LayerNorm(comp_emb_shape)
				
			self.intermediate_embedding = nn.Sequential(
				init_(nn.Linear(comp_emb_shape, comp_emb_shape, bias=True), activate=True),
				nn.GELU(),
				)

		if self.use_recurrent_critic:
			self.RNN = nn.GRU(input_size=comp_emb_shape, hidden_size=comp_emb_shape, num_layers=self.rnn_num_layers, batch_first=True)
			for name, param in self.RNN.named_parameters():
				if 'bias' in name:
					nn.init.constant_(param, 0)
				elif 'weight' in name:
					nn.init.orthogonal_(param)

		self.q_value_layer = nn.Sequential(
			nn.LayerNorm(comp_emb_shape),
			init_(nn.Linear(comp_emb_shape, 1), activate=False)
			)
		

		self.mask_value = torch.tensor(
			torch.finfo(torch.float).min, dtype=torch.float
			)

		self.one_hot_actions = torch.eye(self.num_actions).to(self.device)
		self.agent_ids = torch.eye(self.num_agents).to(self.device)


	def forward(self, local_observations, ally_states, enemy_states, actions, rnn_hidden_state):
		if self.centralized:
			batch, timesteps, _, _ = ally_states.shape
			# agent_embedding = self.agent_embedding(torch.arange(self.num_agents).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_agents, self.comp_emb_shape)
			# enemy_embedding = self.enemy_embedding(torch.arange(self.num_enemies).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_enemies, self.comp_emb_shape)
			# ally_state_embedding = (self.ally_obs_embedding(ally_states) + agent_embedding + self.action_embedding(actions.long())).sum(dim=-2) #/ self.num_agents
			# enemy_state_embedding = (self.enemy_obs_embedding(enemy_states) + enemy_embedding).sum(dim=-2) #/ self.num_enemies

			# # final_state_embedding = ally_state_embedding+enemy_state_embedding # self.state_action_embedding_layer_norm(ally_state_embedding+enemy_state_embedding)
			# final_state_embedding = self.state_action_embedding_layer_norm(torch.cat([ally_state_embedding, enemy_state_embedding], dim=-1))

			# final_state_embedding = self.intermediate_embedding(final_state_embedding)

			one_hot_actions = self.one_hot_actions[actions.long()]
			agent_ids = self.agent_ids.reshape(1, 1, self.num_agents, self.num_agents).repeat(batch, timesteps, 1, 1)
			ally_observations = torch.cat([agent_ids, ally_states, one_hot_actions], dim=-1)
			ally_observations = torch.stack([torch.roll(ally_observations, shifts=-i, dims=2) for i in range(self.num_agents)], dim=0).to(self.device)
			ally_observations = ally_observations.permute(1, 2, 0, 3, 4).reshape(batch, timesteps, self.num_agents, -1)
			enemy_observations = enemy_states.reshape(batch, timesteps, 1, -1).repeat(1, 1, self.num_agents, 1)
			final_state_embedding = self.embedding(torch.cat([ally_observations, enemy_observations], dim=-1)).permute(0, 2, 1, 3).reshape(batch*self.num_agents, timesteps, -1)

			if self.use_recurrent_critic:
				final_state_embedding, h = self.RNN(final_state_embedding, rnn_hidden_state)
				final_state_embedding = final_state_embedding.reshape(batch, self.num_agents, timesteps, -1).permute(0, 2, 1, 3).reshape(batch*timesteps, self.num_agents, -1)

			Q_value = self.q_value_layer(final_state_embedding)

		else:
			batch, timesteps, num_agents, _ = local_observations.shape
			agent_embedding = self.agent_embedding(torch.arange(self.num_agents).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_agents, self.comp_emb_shape)
			obs_embedding = self.obs_embedding(local_observations) + agent_embedding + self.action_embedding(actions.long()) # self.state_action_embedding_layer_norm(self.obs_embedding(local_observations) + agent_embedding + self.action_embedding(actions.long()))
			intermediate_embedding = self.intermediate_embedding(obs_embedding)
			
			if self.use_recurrent_critic:
				intermediate_embedding, h = self.RNN(intermediate_embedding.permute(0, 2, 1, 3).reshape(batch*num_agents, timesteps, -1), rnn_hidden_state)
				intermediate_embedding = intermediate_embedding.reshape(batch, num_agents, timesteps, -1).permute(0, 2, 1, 3).reshape(batch*timesteps, num_agents, -1)
			
			Q_value = self.q_value_layer(intermediate_embedding)

		return Q_value.squeeze(-1), h


class Q_network(nn.Module):
	def __init__(
		self, 
		use_recurrent_critic,
		ally_obs_input_dim, 
		enemy_obs_input_dim,
		num_agents, 
		num_enemies,
		num_actions, 
		rnn_num_layers,
		comp_emb_shape,
		device, 
		num_heads=1,
		centralized=True,
		num_teams=0,
		enable_hard_attention=False, 
		attention_dropout_prob=0.0, 
		temperature=1.0,
		norm_returns=True,
		environment="StarCraft",
		local_observation_input_dim=None,
		):
		super(Q_network, self).__init__()
		
		self.use_recurrent_critic = use_recurrent_critic
		self.num_heads = num_heads
		self.num_agents = num_agents
		self.num_enemies = num_enemies
		self.num_teams = num_teams
		self.num_actions = num_actions
		self.rnn_num_layers = rnn_num_layers
		self.comp_emb_shape = comp_emb_shape
		self.device = device
		self.enable_hard_attention = enable_hard_attention
		self.environment = environment
		self.attention_dropout_prob = attention_dropout_prob

		# self.attention_dropout = AttentionDropout(dropout_prob=attention_dropout_prob)

		self.temperature = temperature

		# positional, agent, enemy and team embeddings
		self.agent_embedding = nn.Embedding(self.num_agents, self.comp_emb_shape)
		self.action_embedding = nn.Embedding(self.num_actions, self.comp_emb_shape)
		if "MPE" in self.environment:
			self.team_embedding = nn.Embedding(self.num_teams, self.comp_emb_shape)
		if "StarCraft" in self.environment:
			self.enemy_embedding = nn.Embedding(self.num_enemies, self.comp_emb_shape)
			self.enemy_state_embed = nn.Sequential(
				# nn.LayerNorm(enemy_obs_input_dim),
				init_(nn.Linear(enemy_obs_input_dim, self.comp_emb_shape, bias=True), activate=True),
				nn.GELU(),
				)

		# Embedding Networks
		self.ally_state_embed = nn.Sequential(
			# nn.LayerNorm(ally_obs_input_dim),
			init_(nn.Linear(ally_obs_input_dim, self.comp_emb_shape, bias=True), activate=True),
			nn.GELU(),
			)

		self.state_embed_layer_norm = nn.LayerNorm(self.comp_emb_shape)
			
		# Key, Query, Attention Value, Hard Attention Networks
		assert 64%self.num_heads == 0
		self.key = init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape), activate=False)
		self.query = init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape), activate=False)
		self.attention_value = init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape), activate=False)

		self.attention_value_dropout = nn.Dropout(0.2)
		self.attention_value_layer_norm = nn.LayerNorm(64)

		self.attention_value_linear = nn.Sequential(
			init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape, bias=True), activate=True),
			nn.GELU(),
			init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape, bias=True)),
			)
		self.attention_value_linear_dropout = nn.Dropout(0.2)

		self.attention_value_linear_layer_norm = nn.LayerNorm(self.comp_emb_shape)

		if self.enable_hard_attention:
			self.hard_attention = nn.Sequential(
				init_(nn.Linear(self.comp_emb_shape+self.comp_emb_shape, 2, bias=True))
				)

		# dimesion of key
		self.d_k_agents = self.comp_emb_shape

		self.common_layer = nn.Sequential(
			init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape, bias=True), activate=True),
			nn.GELU()
			)

		if self.use_recurrent_critic:
			self.RNN = nn.GRU(input_size=self.comp_emb_shape, hidden_size=self.comp_emb_shape, num_layers=self.rnn_num_layers, batch_first=True)
			for name, param in self.RNN.named_parameters():
				if 'bias' in name:
					nn.init.constant_(param, 0)
				elif 'weight' in name:
					nn.init.orthogonal_(param)

		

		self.q_value_layer = nn.Sequential(
			nn.LayerNorm(self.comp_emb_shape),
			init_(nn.Linear(self.comp_emb_shape, 1, bias=True))
			)

		self.mask_value = torch.tensor(
			torch.finfo(torch.float).min, dtype=torch.float
			)


	def get_attention_masks(self, agent_masks):
		# since we add the attention masks to the score we want to have 0s where the agent is alive and -inf when agent is dead
		attention_masks = copy.deepcopy(1-agent_masks).unsqueeze(-2).repeat(1, 1, self.num_agents, 1)
		# choose columns in each row where the agent is dead and make it -inf
		attention_masks[agent_masks.unsqueeze(-2).repeat(1, 1, self.num_agents, 1)[:, :, :, :] == 0.0] = self.mask_value
		# choose rows of the agent which is dead and make it -inf
		attention_masks[agent_masks.unsqueeze(-2).repeat(1, 1, self.num_agents, 1).transpose(-1,-2)[:, :, :, :] == 0.0] = self.mask_value

		for i in range(self.num_agents):
			attention_masks[:, :, i, i] = self.mask_value
		return attention_masks


	def forward(self, local_observations, states, enemy_states, actions, rnn_hidden_state, agent_masks):
		batch, timesteps, num_agents, _ = states.shape
		states = states.reshape(batch*timesteps, self.num_agents, -1)
		actions = actions.reshape(batch*timesteps, num_agents)

		# extract agent embedding
		agent_embedding = self.agent_embedding(torch.arange(self.num_agents).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_agents, self.comp_emb_shape).reshape(batch*timesteps, self.num_agents, -1)

		states_embed = self.ally_state_embed(states) + agent_embedding

		if "MPE" in self.environment:
			team_embedding = self.team_embedding(torch.arange(self.num_teams).to(self.device))[None, None, :, None, :].expand(batch, timesteps, self.num_teams, self.num_agents//self.num_teams, self.comp_emb_shape).reshape(batch*timesteps, self.num_agents, self.comp_emb_shape)
			states_embed = states_embed + team_embedding

		if "StarCraft" in self.environment:
			enemy_embedding = self.enemy_embedding(torch.arange(self.num_enemies).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_enemies, self.comp_emb_shape)
			enemy_state_embed = (self.enemy_state_embed(enemy_states) + enemy_embedding).sum(dim=2).unsqueeze(2).reshape(batch*timesteps, 1, self.comp_emb_shape)
			states_embed = states_embed + enemy_state_embed
		
		states_embed = self.state_embed_layer_norm(states_embed)

		# if self.use_recurrent_critic:
		# 	states_embed = states_embed.reshape(batch, timesteps, num_agents, -1).permute(0, 2, 1, 3).reshape(batch*num_agents, timesteps, -1)
		# 	states_embed, h = self.RNN(states_embed, rnn_hidden_state)
		# 	states_embed = states_embed.reshape(batch, num_agents, timesteps, -1).permute(0, 2, 1, 3).reshape(batch*timesteps, num_agents, -1)

		# KEYS
		key_obs = self.key(states_embed).reshape(batch*timesteps, num_agents, self.num_heads, -1).permute(0, 2, 1, 3) # Batch_size, Num Heads, Num agents, dim
		# QUERIES
		query_obs = self.query(states_embed).reshape(batch*timesteps, num_agents, self.num_heads, -1).permute(0, 2, 1, 3) # Batch_size, Num Heads, Num agents, dim

		# HARD ATTENTION
		if self.enable_hard_attention:
			query_key_concat = torch.cat([query_obs.unsqueeze(3).repeat(1,1,1,self.num_agents,1), key_obs.unsqueeze(2).repeat(1,1,self.num_agents,1,1)], dim=-1) # Batch_size, Num Heads, Num agents, Num Agents, dim
			query_key_concat_intermediate = self.hard_attention(query_key_concat) # Batch_size, Num Heads, Num agents, Num agents-1, dim
			hard_attention_weights = F.gumbel_softmax(query_key_concat_intermediate, hard=True, tau=1.0)[:,:,:,:,1] # Batch_size, Num Heads, Num agents, Num Agents, 1			
			for i in range(self.num_agents):
				hard_attention_weights[:,:,i,i] = 1.0
		else:
			hard_attention_weights = torch.ones(states.shape[0], self.num_heads, self.num_agents, self.num_agents).float().to(self.device)
			

		# SOFT ATTENTION
		score = torch.matmul(query_obs,(key_obs).transpose(-2,-1))/(self.d_k_agents//self.num_heads)**(1/2) # Batch_size, Num Heads, Num agents, Num Agents
		
		attention_masks = self.get_attention_masks(agent_masks).reshape(batch*timesteps, num_agents, num_agents).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
		attention_masks = attention_masks + (1-hard_attention_weights)*self.mask_value
		weights = F.softmax((score/(torch.max(score*(attention_masks[:, :, :, :]!=self.mask_value).float(), dim=-1).values-torch.min(score*(attention_masks[:, :, :, :]!=self.mask_value).float(), dim=-1).values+1e-5).detach().unsqueeze(-1)) + attention_masks.reshape(*score.shape).to(score.device), dim=-1) # Batch_size, Num Heads, Num agents, Num Agents
		
		# if self.attention_dropout_prob > 0.0:
		# 	for head in range(self.num_heads):
		# 		weights[:, head, :, :] = self.attention_dropout(weights[:, head, :, :])

		final_weights = weights.clone()
		prd_weights = F.softmax((score/(torch.max(score*(attention_masks[:, :, :, :]!=self.mask_value).float(), dim=-2).values-torch.min(score*(attention_masks[:, :, :, :]!=self.mask_value).float(), dim=-2).values+1e-5).detach().unsqueeze(-1)) + attention_masks.reshape(*score.shape).to(score.device), dim=-2) # Batch_size, Num Heads, Num agents, Num Agents
		for i in range(self.num_agents):
			final_weights[:, :, i, i] = 1.0 # since weights[:, :, i, i] = 0.0
			prd_weights[:, :, i, i] = 1.0

		final_weights = final_weights * agent_masks.reshape(batch*timesteps, 1, self.num_agents, 1).repeat(1, self.num_heads, 1, self.num_agents)
		final_weights = final_weights * agent_masks.reshape(batch*timesteps, 1, 1, self.num_agents).repeat(1, self.num_heads, self.num_agents, 1)
		prd_weights = prd_weights * agent_masks.reshape(batch*timesteps, 1, self.num_agents, 1).repeat(1, self.num_heads, 1, self.num_agents)
		prd_weights = prd_weights * agent_masks.reshape(batch*timesteps, 1, 1, self.num_agents).repeat(1, self.num_heads, self.num_agents, 1)

		
		# EMBED STATE ACTION
		obs_actions_embed = states_embed + self.action_embedding(actions.long())

		attention_values = self.attention_value(obs_actions_embed).reshape(batch*timesteps, num_agents, self.num_heads, -1).permute(0, 2, 1, 3) #torch.stack([self.attention_value[i](obs_actions_embed) for i in range(self.num_heads)], dim=0).permute(1,0,2,3,4) # Batch_size, Num heads, Num agents, Num agents - 1, dim//num_heads
		
		aggregated_node_features = torch.matmul(final_weights, attention_values) # Batch_size, Num heads, Num agents, dim//num_heads
		aggregated_node_features = self.attention_value_dropout(aggregated_node_features)
		aggregated_node_features = aggregated_node_features.permute(0,2,1,3).reshape(batch*timesteps, self.num_agents, -1) # Batch_size, Num agents, dim
		aggregated_node_features_ = self.attention_value_layer_norm(obs_actions_embed+aggregated_node_features) # Batch_size, Num agents, dim
		aggregated_node_features = self.attention_value_linear(aggregated_node_features_) # Batch_size, Num agents, dim
		aggregated_node_features = self.attention_value_linear_dropout(aggregated_node_features)
		aggregated_node_features = self.attention_value_linear_layer_norm(aggregated_node_features_+aggregated_node_features) # Batch_size, Num agents, dim
		
		curr_agent_node_features = self.common_layer(aggregated_node_features) # Batch_size, Num agents, dim
		
		if self.use_recurrent_critic:
			curr_agent_node_features = curr_agent_node_features.reshape(batch, timesteps, num_agents, -1).permute(0, 2, 1, 3).reshape(batch*num_agents, timesteps, -1)
			rnn_output, h = self.RNN(curr_agent_node_features, rnn_hidden_state)
			rnn_output = rnn_output.reshape(batch, num_agents, timesteps, -1).permute(0, 2, 1, 3).reshape(batch*timesteps, num_agents, -1)
			Q_value = self.q_value_layer(rnn_output) # Batch_size, Num agents, 1
		else:
			Q_value = self.q_value_layer(curr_agent_node_features) # Batch_size, Num agents, 1

		# Q_value = self.q_value_layer(curr_agent_node_features)

		return Q_value.squeeze(-1), h