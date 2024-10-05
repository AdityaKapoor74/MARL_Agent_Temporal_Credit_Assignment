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
			if mask is not None:
				batch_mean = detached_input.sum(dim=tuple(range(self.norm_axes)))/mask.sum(dim=tuple(range(self.norm_axes)))
				batch_sq_mean = (detached_input ** 2).sum(dim=tuple(range(self.norm_axes)))/mask.sum(dim=tuple(range(self.norm_axes)))
			else:
				batch_mean = detached_input.mean(dim=tuple(range(self.norm_axes)))
				batch_sq_mean = (detached_input ** 2).mean(dim=tuple(range(self.norm_axes)))

			if self.per_element_update:
				if mask is not None:
					batch_size = (mask.reshape(-1, self.num_agents).sum(dim=-1)>0.0).sum()
				else:
					batch_size = detached_input.size()
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
		
		return out.to(input_vector_device)



def init(module, weight_init, bias_init, gain=1):
	weight_init(module.weight.data, gain=gain)
	if module.bias is not None:
		bias_init(module.bias.data)
	return module

def init_(m, gain=0.01, activate=False):
	if activate:
		gain = nn.init.calculate_gain('relu')
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

		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device

		self.mask_value = torch.tensor(
				torch.finfo(torch.float).min, dtype=torch.float
			).to(self.device)

		self.agent_embedding = nn.Embedding(self.num_agents, self.rnn_hidden_actor)
		self.action_embedding = nn.Embedding(self.num_actions+1, self.rnn_hidden_actor) # we assume the first "last action" to be NON-EXISTENT so one of the embedding represents that

		if self.use_recurrent_policy:
			
			self.obs_embedding = nn.Sequential(
				# nn.LayerNorm(obs_input_dim),
				init_(nn.Linear(obs_input_dim, rnn_hidden_actor), activate=True),
				nn.GELU(),
				)

			self.obs_embed_layer_norm = nn.LayerNorm(self.rnn_hidden_actor)
			
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
				nn.LayerNorm(rnn_hidden_actor),
				init_(nn.Linear(rnn_hidden_actor, num_actions), gain=0.01)
				)


	def forward(self, local_observations, last_actions, hidden_state, mask_actions):

		batch, timesteps, _, _ = local_observations.shape
		agent_embedding = self.agent_embedding(torch.arange(self.num_agents).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_agents, self.rnn_hidden_actor)
		last_action_embedding = self.action_embedding(last_actions.long())
		obs_embedding = self.obs_embedding(local_observations)
		final_obs_embedding = self.obs_embed_layer_norm(obs_embedding + last_action_embedding + agent_embedding).permute(0, 2, 1, 3).reshape(batch*self.num_agents, timesteps, -1) # self.obs_embed_layer_norm(obs_embedding + last_action_embedding + agent_embedding).permute(0, 2, 1, 3).reshape(batch*self.num_agents, timesteps, -1)

		if self.use_recurrent_policy:
			hidden_state = hidden_state.reshape(self.rnn_num_layers, batch*self.num_agents, -1)
			output, h = self.RNN(final_obs_embedding, hidden_state)
			output = output.reshape(batch, self.num_agents, timesteps, -1).permute(0, 2, 1, 3)
			logits = self.Layer_2(output)
		else:
			logits = self.final_layer(local_observations)

		logits = torch.where(mask_actions, logits, self.mask_value)

		if self.use_recurrent_policy:
			return F.softmax(logits, dim=-1), h, output
		else:
			return F.softmax(logits, dim=-1), h, local_observations



class V_network(nn.Module):
	def __init__(
		self, 
		environment,
		use_recurrent_critic,
		centralized,
		local_observation_input_dim,
		global_observation_input_dim,
		ally_obs_input_dim, 
		enemy_obs_input_dim,
		num_agents, 
		num_enemies,
		num_actions, 
		rnn_num_layers,
		comp_emb_shape,
		device,
		):
		
		super(V_network, self).__init__()
		
		self.environment = environment
		self.use_recurrent_critic = use_recurrent_critic
		self.centralized = centralized
		self.num_agents = num_agents
		self.num_enemies = num_enemies
		self.num_actions = num_actions
		self.rnn_num_layers = rnn_num_layers
		self.comp_emb_shape = comp_emb_shape
		self.device = device

		if self.centralized:
			if "StarCraft" in self.environment:
				input_dim = (self.num_agents+ally_obs_input_dim+self.num_actions)*self.num_agents + enemy_obs_input_dim*self.num_enemies
			elif self.environment in ["Alice_and_Bob", "GFootball"]:
				input_dim = global_observation_input_dim+self.num_agents+self.num_actions*self.num_agents
		else:
			input_dim = local_observation_input_dim

		self.embedding = nn.Sequential(
			init_(nn.Linear(input_dim, comp_emb_shape*2, bias=True), activate=True),
			nn.GELU(),
			nn.LayerNorm(comp_emb_shape*2),
			init_(nn.Linear(comp_emb_shape*2, comp_emb_shape, bias=True), activate=True),
			nn.GELU(),
			nn.LayerNorm(comp_emb_shape),
			)


		if self.use_recurrent_critic:
			self.RNN = nn.GRU(input_size=comp_emb_shape, hidden_size=comp_emb_shape, num_layers=self.rnn_num_layers, batch_first=True)
			for name, param in self.RNN.named_parameters():
				if 'bias' in name:
					nn.init.constant_(param, 0)
				elif 'weight' in name:
					nn.init.orthogonal_(param)

		self.value_layer = nn.Sequential(
			nn.LayerNorm(comp_emb_shape),
			init_(nn.Linear(comp_emb_shape, 1), activate=False)
			)
		

		self.mask_value = torch.tensor(
			torch.finfo(torch.float).min, dtype=torch.float
			).to(self.device)

		self.one_hot_actions = torch.eye(self.num_actions).to(self.device)
		self.agent_ids = torch.eye(self.num_agents).to(self.device)


	def forward(self, local_observations, global_observations, ally_states, enemy_states, actions, rnn_hidden_state, agent_masks):
		if self.centralized:
			if "StarCraft" in self.environment:
				batch, timesteps, _, _ = ally_states.shape
				one_hot_actions = self.one_hot_actions[actions.long()]
				agent_ids = self.agent_ids.reshape(1, 1, self.num_agents, self.num_agents).repeat(batch, timesteps, 1, 1)
				ally_states = torch.cat([agent_ids, ally_states, one_hot_actions], dim=-1)
				ally_states = torch.stack([torch.roll(ally_states, shifts=-i, dims=2) for i in range(self.num_agents)], dim=0).to(self.device)
				ally_states = ally_states.permute(1, 2, 0, 3, 4)
				ally_states[:, :, :, 0, -self.num_actions:] = torch.zeros(batch, timesteps, self.num_agents, self.num_actions).to(self.device)
				ally_states = ally_states.reshape(batch, timesteps, self.num_agents, -1)
				enemy_states = enemy_states.reshape(batch, timesteps, 1, -1).repeat(1, 1, self.num_agents, 1)
				final_state_embedding = self.embedding(torch.cat([ally_states, enemy_states], dim=-1)).permute(0, 2, 1, 3).reshape(batch*self.num_agents, timesteps, -1)
			elif "Alice_and_Bob" in self.environment:
				batch, timesteps,  _ = global_observations.shape
				global_observations = global_observations.unsqueeze(-2).repeat(1, 1, self.num_agents, 1)
				one_hot_actions = self.one_hot_actions[actions.long()].unsqueeze(-3)
				one_hot_actions = one_hot_actions.repeat(1, 1, self.num_agents, 1, 1)
				for i in range(self.num_agents):
					one_hot_actions[:, :, i, i, :] = torch.zeros(batch, timesteps, self.num_actions).to(self.device)
				one_hot_actions = one_hot_actions.reshape(batch, timesteps, self.num_agents, self.num_agents*self.num_actions)
				agent_ids = self.agent_ids.reshape(1, 1, self.num_agents, self.num_agents).repeat(batch, timesteps, 1, 1)
				global_observations = torch.cat([agent_ids, global_observations, one_hot_actions], dim=-1)
				final_state_embedding = self.embedding(global_observations).permute(0, 2, 1, 3).reshape(batch*self.num_agents, timesteps, -1)
			elif "GFootball" in self.environment:
				batch, timesteps, _, _ = global_observations.shape
				one_hot_actions = self.one_hot_actions[actions.long()].unsqueeze(-3)
				one_hot_actions = one_hot_actions.repeat(1, 1, self.num_agents, 1, 1)
				for i in range(self.num_agents):
					one_hot_actions[:, :, i, i, :] = torch.zeros(batch, timesteps, self.num_actions).to(self.device)
				one_hot_actions = one_hot_actions.reshape(batch, timesteps, self.num_agents, self.num_agents*self.num_actions)
				agent_ids = self.agent_ids.reshape(1, 1, self.num_agents, self.num_agents).repeat(batch, timesteps, 1, 1)
				global_observations = torch.cat([agent_ids, global_observations, one_hot_actions], dim=-1)
				final_state_embedding = self.embedding(global_observations).permute(0, 2, 1, 3).reshape(batch*self.num_agents, timesteps, -1)
		else:
			batch, timesteps, _, _ = local_observations.shape
			final_state_embedding = self.embedding(local_observations).permute(0, 2, 1, 3).reshape(batch*self.num_agents, timesteps, -1)
		
		if self.use_recurrent_critic:
			final_state_embedding, h = self.RNN(final_state_embedding, rnn_hidden_state)
			final_state_embedding = final_state_embedding.reshape(batch, self.num_agents, timesteps, -1).permute(0, 2, 1, 3).reshape(batch*timesteps, self.num_agents, -1)

		Value = self.value_layer(final_state_embedding)

		return Value.squeeze(-1), h



class InverseDynamicsModel(nn.Module):
	def __init__(
		self, 
		rnn_hidden_actor, 
		num_actions, 
		num_agents, 
		device
		):
		super(InverseDynamicsModel, self).__init__()

		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device

		self.action_prediction = nn.Sequential(
			nn.LayerNorm(2*rnn_hidden_actor),
			init_(nn.Linear(rnn_hidden_actor*2, rnn_hidden_actor)),
			nn.GELU(),
			init_(nn.Linear(rnn_hidden_actor, num_actions))
			)


	def forward(self, current_latent_state, goal_latent_state, agent_masks):

		batch, timesteps, _, _ = current_latent_state.shape
		upper_triangular_matrix = torch.triu(torch.ones(batch*self.num_agents, timesteps, timesteps)).reshape(batch, self.num_agents, timesteps, timesteps, 1).permute(0, 2, 3, 1, 4)
		current_latent_state = current_latent_state.unsqueeze(2).repeat(1, 1, timesteps, 1, 1)
		goal_latent_state = goal_latent_state.unsqueeze(1).repeat(1, timesteps, 1, 1, 1)
		current_goal_latent_state = torch.cat([current_latent_state, goal_latent_state], dim=-1) * agent_masks.unsqueeze(1).unsqueeze(-1) * upper_triangular_matrix.to(self.device)
		logits = self.action_prediction(current_goal_latent_state)

		return logits