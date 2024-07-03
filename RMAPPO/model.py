import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
# from utils import gumbel_sigmoid


class RunningMeanStd(object):
	def __init__(self, epsilon: float = 1e-4, shape = (1), device="cpu"):
		"""
		https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
		"""
		self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
		self.var = torch.ones(shape, dtype=torch.float32, device=device)
		self.count = epsilon

	def update(self, arr, mask):
		# arr = arr.reshape(-1, arr.size(-1))
		# batch_mean = torch.mean(arr, dim=0)
		# batch_var = torch.var(arr, dim=0)
		batch_mean = torch.sum(arr, dim=0) / mask.sum(dim=0)
		batch_var = torch.sum((arr - batch_mean)**2, dim=0) / mask.sum(dim=0)
		batch_count = mask.sum() #arr.shape[0]
		self.update_from_moments(batch_mean, batch_var, batch_count)

	def update_from_moments(self, batch_mean, batch_var, batch_count: int):
		delta = batch_mean - self.mean
		tot_count = self.count + batch_count

		new_mean = self.mean + delta * batch_count / tot_count
		m_a = self.var * self.count
		m_b = batch_var * batch_count
		m_2 = (
			m_a
			+ m_b
			+ torch.square(delta)
			* self.count
			* batch_count
			/ (self.count + batch_count)
		)
		new_var = m_2 / (self.count + batch_count)

		new_count = batch_count + self.count

		self.mean = new_mean
		self.var = new_var
		self.count = new_count


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

		self.one_hot_actions = torch.eye(self.num_actions+1).to(self.device)

		if self.use_recurrent_policy:
			
			self.obs_embedding = nn.Sequential(
				init_(nn.Linear(obs_input_dim+self.num_actions+1, rnn_hidden_actor), activate=True),
				nn.GELU(),
				nn.LayerNorm(rnn_hidden_actor)
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

			# self.obs_embed_layer_norm = nn.LayerNorm(self.rnn_hidden_actor)

			self.final_layer = nn.Sequential(
				init_(nn.Linear(rnn_hidden_actor, num_actions), gain=0.01)
				)


	def forward(self, local_observations, last_actions, hidden_state, mask_actions):

		batch, timesteps, _, _ = local_observations.shape
		# agent_embedding = self.agent_embedding(torch.arange(self.num_agents).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_agents, self.rnn_hidden_actor)
		# last_action_embedding = self.action_embedding(last_actions.long())
		# obs_embedding = self.obs_embedding(local_observations)
		# final_obs_embedding = self.obs_embed_layer_norm(obs_embedding + last_action_embedding + agent_embedding).permute(0, 2, 1, 3).reshape(batch*self.num_agents, timesteps, -1) # (obs_embedding + last_action_embedding + agent_embedding).permute(0, 2, 1, 3).reshape(batch*self.num_agents, timesteps, -1) 

		one_hot_actions = self.one_hot_actions[last_actions.long()]
		final_obs_embedding = self.obs_embedding(torch.cat([local_observations, one_hot_actions], dim=-1)).permute(0, 2, 1, 3).reshape(batch*self.num_agents, timesteps, -1)


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
				init_(nn.Linear((ally_obs_input_dim+self.num_actions)*self.num_agents + enemy_obs_input_dim*self.num_enemies, comp_emb_shape*2, bias=True), activate=True),
				nn.GELU(),
				nn.LayerNorm(comp_emb_shape*2)
				init_(nn.Linear(comp_emb_shape*2, comp_emb_shape, bias=True), activate=True),
				nn.GELU(),
				nn.LayerNorm(comp_emb_shape)
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
			ally_observations = torch.cat([ally_states, one_hot_actions], dim=-1).reshape(batch, timesteps, -1)
			enemy_observations = enemy_states.reshape(batch, timesteps, -1)
			final_state_embedding = self.embedding(torch.cat([ally_observations, enemy_observations], dim=-1))


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