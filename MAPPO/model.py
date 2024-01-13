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
	def update(self, input_vector, mask):
		if type(input_vector) == np.ndarray:
			input_vector = torch.from_numpy(input_vector)
		input_vector = input_vector.to(**self.tpdv)
		
		old_mean, old_var = self.debiased_mean_var()
		old_stddev = torch.sqrt(old_var)

		# batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
		# batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))
		batch_mean = input_vector.sum(dim=tuple(range(self.norm_axes)))/mask.sum(dim=tuple(range(self.norm_axes)))
		batch_sq_mean = (input_vector ** 2).sum(dim=tuple(range(self.norm_axes)))/mask.sum(dim=tuple(range(self.norm_axes)))

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


class ValueNorm(nn.Module):
	""" Normalize a vector of observations - across the first norm_axes dimensions"""

	def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5, device=torch.device("cpu")):
		super(ValueNorm, self).__init__()

		self.input_shape = input_shape
		self.norm_axes = norm_axes
		self.epsilon = epsilon
		self.beta = beta
		self.per_element_update = per_element_update
		self.tpdv = dict(dtype=torch.float32, device=device)

		self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
		self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
		self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)
		
		self.reset_parameters()

	def reset_parameters(self):
		self.running_mean.zero_()
		self.running_mean_sq.zero_()
		self.debiasing_term.zero_()

	def running_mean_var(self):
		debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
		debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
		debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
		return debiased_mean, debiased_var

	@torch.no_grad()
	def update(self, input_vector, mask):
		if type(input_vector) == np.ndarray:
			input_vector = torch.from_numpy(input_vector)
		input_vector = input_vector.to(**self.tpdv)

		# batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
		# batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))
		batch_mean = input_vector.sum(dim=tuple(range(self.norm_axes)))/mask.sum(dim=tuple(range(self.norm_axes)))
		batch_sq_mean = (input_vector ** 2).sum(dim=tuple(range(self.norm_axes)))/mask.sum(dim=tuple(range(self.norm_axes)))

		if self.per_element_update:
			batch_size = mask.sum() #np.prod(input_vector.size()[:self.norm_axes])
			weight = self.beta ** batch_size
		else:
			weight = self.beta

		self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
		self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
		self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

	def normalize(self, input_vector):
		# Make sure input is float64
		if type(input_vector) == np.ndarray:
			input_vector = torch.from_numpy(input_vector)
		input_vector = input_vector.to(**self.tpdv)

		mean, var = self.running_mean_var()
		out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
		
		return out

	def denormalize(self, input_vector):
		""" Transform normalized data back into original distribution """
		if type(input_vector) == np.ndarray:
			input_vector = torch.from_numpy(input_vector)
		input_vector = input_vector.to(**self.tpdv)

		mean, var = self.running_mean_var()
		out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
		
		return out
		

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
		obs_input_dim, 
		num_actions, 
		num_agents, 
		rnn_num_layers, 
		device
		):
		super(Policy, self).__init__()

		self.rnn_num_layers = rnn_num_layers

		self.mask_value = torch.tensor(
				torch.finfo(torch.float).min, dtype=torch.float
			)
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device
		self.Layer_1 = nn.Sequential(
			init_(nn.Linear(obs_input_dim, 64), activate=True),
			nn.GELU(),
			nn.LayerNorm(64),
			)
		self.RNN = nn.GRU(input_size=64, hidden_size=64, num_layers=rnn_num_layers, batch_first=True)
		self.Layer_2 = nn.Sequential(
			nn.LayerNorm(64),
			init_(nn.Linear(64, num_actions), gain=0.01)
			)

		for name, param in self.RNN.named_parameters():
			if 'bias' in name:
				nn.init.constant_(param, 0)
			elif 'weight' in name:
				nn.init.orthogonal_(param)


	def forward(self, local_observations, one_hot_actions, hidden_state, mask_actions=None):
		batch, timesteps, num_agents, _ = local_observations.shape
		intermediate = self.Layer_1(local_observations)
		intermediate = intermediate.permute(0, 2, 1, 3).reshape(batch*num_agents, timesteps, -1)
		hidden_state = hidden_state.reshape(self.rnn_num_layers, batch*num_agents, -1)
		output, h = self.RNN(intermediate, hidden_state)
		output = output.reshape(batch, num_agents, timesteps, -1).permute(0, 2, 1, 3)
		logits = self.Layer_2(output)
		
		logits = torch.where(mask_actions, logits, self.mask_value)
		return F.softmax(logits, dim=-1), h


class Q_network(nn.Module):
	def __init__(
		self, 
		ally_obs_input_dim, 
		enemy_obs_input_dim,
		num_heads, 
		num_agents, 
		num_enemies,
		num_actions, 
		rnn_num_layers,
		device, 
		attention_dropout_prob, 
		temperature
		):
		super(Q_network, self).__init__()
		
		self.num_heads = num_heads
		self.num_agents = num_agents
		self.num_enemies = num_enemies
		self.num_actions = num_actions
		self.rnn_num_layers = rnn_num_layers
		self.device = device

		self.temperature = temperature

		# Embedding Networks
		self.ally_state_embed_1 = nn.Sequential(
			init_(nn.Linear(ally_obs_input_dim, 64, bias=True), activate=True),
			nn.GELU(),
			nn.LayerNorm(64),
			)

		self.enemy_state_embed = nn.Sequential(
			init_(nn.Linear(enemy_obs_input_dim, 64, bias=True), activate=True),
			nn.GELU(),
			nn.LayerNorm(64),
			)

		self.ally_state_act_embed = nn.Sequential(
			init_(nn.Linear(ally_obs_input_dim+self.num_actions, 64), activate=True), 
			nn.GELU(),
			nn.LayerNorm(64),
			)

		# Key, Query, Attention Value, Hard Attention Networks
		assert 64%self.num_heads == 0
		self.key = init_(nn.Linear(64, 64))
		self.query = init_(nn.Linear(64, 64))
		self.attention_value = init_(nn.Linear(64, 64))

		self.attention_value_dropout = nn.Dropout(0.2)
		self.attention_value_layer_norm = nn.LayerNorm(64)

		self.attention_value_linear = nn.Sequential(
			init_(nn.Linear(64, 64), activate=True),
			nn.Dropout(0.2),
			nn.GELU(),
			nn.LayerNorm(64),
			init_(nn.Linear(64, 64))
			)
		self.attention_value_linear_dropout = nn.Dropout(0.2)

		self.attention_value_linear_layer_norm = nn.LayerNorm(64)

		# dimesion of key
		self.d_k_agents = 64

		# Attention for agents to enemies
		# Key, Query, Attention Value Networks
		assert 64%self.num_heads == 0
		self.key_enemies = init_(nn.Linear(64, 64))
		self.query_enemies = init_(nn.Linear(64, 64))
		self.attention_value_enemies = init_(nn.Linear(64, 64))

		self.attention_value_enemies_dropout = nn.Dropout(0.2)
		self.attention_value_enemies_layer_norm = nn.LayerNorm(64)

		self.attention_value_linear_enemies = nn.Sequential(
			init_(nn.Linear(64, 64), activate=True),
			nn.Dropout(0.2),
			nn.GELU(),
			nn.LayerNorm(64),
			init_(nn.Linear(64, 64))
			)
		self.attention_value_linear_enemies_dropout = nn.Dropout(0.2)

		self.attention_value_linear_enemies_layer_norm = nn.LayerNorm(64)

		# dimesion of key
		self.d_k_enemies = 64


		# FCN FINAL LAYER TO GET Q-VALUES
		self.common_layer = nn.Sequential(
			init_(nn.Linear(64+64, 64), activate=True),
			nn.GELU()
			)

		self.RNN = nn.GRU(input_size=64, hidden_size=64, num_layers=self.rnn_num_layers, batch_first=True)
		for name, param in self.RNN.named_parameters():
			if 'bias' in name:
				nn.init.constant_(param, 0)
			elif 'weight' in name:
				nn.init.orthogonal_(param)

		self.q_value_layer = nn.Sequential(
			nn.LayerNorm(64),
			init_(PopArt(64, 1, device=self.device))
			)
		
		self.mask_value = torch.tensor(
				torch.finfo(torch.float).min, dtype=torch.float
			)

	def get_attention_masks(self, agent_masks):
		attention_masks = copy.deepcopy(agent_masks).unsqueeze(-2).repeat(1, 1, self.num_agents, 1)
		attention_masks[attention_masks[:, :, :, :] == 0.0] = self.mask_value
		attention_masks[attention_masks[:, :, :, :] == 1.0] = 0.0
		for i in range(self.num_agents):
			attention_masks[:, :, i, i] = self.mask_value
		return attention_masks


	def forward(self, states, enemy_states, actions, rnn_hidden_state, agent_masks):
		batch, timesteps, num_agents, _ = states.shape
		_, _, num_enemies, _ = enemy_states.shape
		states = states.reshape(batch*timesteps, num_agents, -1)
		enemy_states = enemy_states.reshape(batch*timesteps, num_enemies, -1)
		actions = actions.reshape(batch*timesteps, num_agents, -1)

		# EMBED STATES KEY & QUERY
		states_embed = self.ally_state_embed_1(states)
		
		# KEYS
		key_obs = self.key(states_embed).reshape(batch*timesteps, num_agents, self.num_heads, -1).permute(0, 2, 1, 3) # Batch_size, Num Heads, Num agents, dim
		# QUERIES
		query_obs = self.query(states_embed).reshape(batch*timesteps, num_agents, self.num_heads, -1).permute(0, 2, 1, 3) # Batch_size, Num Heads, Num agents, dim

		# SOFT ATTENTION
		score = torch.matmul(query_obs,(key_obs).transpose(-2,-1))/(self.d_k_agents//self.num_heads)**(1/2) # Batch_size, Num Heads, Num agents, Num Agents
		score_ret = score.clone()
		attention_masks = self.get_attention_masks(agent_masks)
		score = score + attention_masks.reshape(*score.shape).to(score.device)
		weights = F.softmax(score, dim=-1) #* hard_attention_weights.unsqueeze(1).permute(0, 1, 2, 4, 3) # Batch_size, Num Heads, Num agents, 1, Num Agents - 1

		final_weights = weights.clone()
		for i in range(self.num_agents):
			final_weights[:, :, i, i] = 1.0 # since weights[:, :, i, i] = 0.0

		# EMBED STATE ACTION
		obs_actions = torch.cat([states, actions], dim=-1).to(self.device) # Batch_size, Num agents, dim
		obs_actions_embed = self.ally_state_act_embed(obs_actions) #+ self.positional_embedding.unsqueeze(0) # Batch_size, Num agents, dim
		attention_values = self.attention_value(obs_actions_embed).reshape(batch*timesteps, num_agents, self.num_heads, -1).permute(0, 2, 1, 3) #torch.stack([self.attention_value[i](obs_actions_embed) for i in range(self.num_heads)], dim=0).permute(1,0,2,3,4) # Batch_size, Num heads, Num agents, Num agents - 1, dim//num_heads
		
		aggregated_node_features = torch.matmul(final_weights, attention_values).squeeze(-2) # Batch_size, Num heads, Num agents, dim//num_heads
		aggregated_node_features = self.attention_value_dropout(aggregated_node_features)
		aggregated_node_features = aggregated_node_features.permute(0,2,1,3).reshape(states.shape[0], self.num_agents, -1) # Batch_size, Num agents, dim
		aggregated_node_features_ = self.attention_value_layer_norm(obs_actions_embed+aggregated_node_features) # Batch_size, Num agents, dim
		aggregated_node_features = self.attention_value_linear(aggregated_node_features_) # Batch_size, Num agents, dim
		aggregated_node_features = self.attention_value_linear_dropout(aggregated_node_features)
		aggregated_node_features = self.attention_value_linear_layer_norm(aggregated_node_features_+aggregated_node_features) # Batch_size, Num agents, dim
		
		# ATTENTION AGENTS TO ENEMIES
		enemy_state_embed = self.enemy_state_embed(enemy_states.view(batch*timesteps, self.num_enemies, -1))
		query_enemies = self.query_enemies(states_embed) # Batch, num_agents, dim
		key_enemies = self.key_enemies(enemy_state_embed) # Batch, num_enemies, dim
		attention_values_enemies = self.attention_value_enemies(enemy_state_embed) # Batch, num_enemies, dim
		# SOFT ATTENTION
		score_enemies = torch.matmul(query_enemies,(key_enemies).transpose(-2,-1))/math.sqrt((self.d_k_enemies)) # Batch_size, Num agents, Num_enemies, dim
		weight_enemies = F.softmax(score_enemies, dim=-1)
		aggregated_attention_value_enemies = torch.matmul(weight_enemies, attention_values_enemies) # Batch, num agents, dim
		aggregated_attention_value_enemies = self.attention_value_enemies_dropout(aggregated_attention_value_enemies)
		aggregated_attention_value_enemies = self.attention_value_enemies_layer_norm(enemy_state_embed.mean(dim=-2).unsqueeze(-2)+states_embed+aggregated_attention_value_enemies)
		aggregated_attention_value_enemies_ = self.attention_value_linear_enemies(aggregated_attention_value_enemies)
		aggregated_attention_value_enemies_ = self.attention_value_linear_enemies_dropout(aggregated_attention_value_enemies_)
		aggregated_attention_value_enemies = self.attention_value_linear_enemies_layer_norm(aggregated_attention_value_enemies+aggregated_attention_value_enemies_)

		curr_agent_node_features = torch.cat([aggregated_attention_value_enemies, aggregated_node_features], dim=-1) # Batch_size, Num agents, dim
		curr_agent_node_features = self.common_layer(curr_agent_node_features) # Batch_size, Num agents, dim
		curr_agent_node_features = curr_agent_node_features.reshape(batch, timesteps, num_agents, -1).permute(0, 2, 1, 3).reshape(batch*num_agents, timesteps, -1)
		rnn_output, h = self.RNN(curr_agent_node_features, rnn_hidden_state)
		rnn_output = rnn_output.reshape(batch, num_agents, timesteps, -1).permute(0, 2, 1, 3).reshape(batch*timesteps, num_agents, -1)
		Q_value = self.q_value_layer(rnn_output) # Batch_size, Num agents, num_actions

		return Q_value.squeeze(-1), weights, score_ret, h