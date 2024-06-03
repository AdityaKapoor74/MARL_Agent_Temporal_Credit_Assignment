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

		# print("mean")
		# print(self.mean)
		# print("var")
		# print(self.var)
		# print("count")
		# print(self.count)


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
			nn.Tanh(),
			nn.LayerNorm(64)
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


	def forward(self, local_observations, hidden_state, mask_actions=None):
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
		obs_input_dim, 
		num_agents, 
		num_enemies,
		num_actions, 
		rnn_num_layers,
		value_norm,
		device, 
		):
		super(Q_network, self).__init__()
		
		self.num_agents = num_agents
		self.num_enemies = num_enemies
		self.num_actions = num_actions
		self.rnn_num_layers = rnn_num_layers
		self.device = device

		# Embedding Networks
		self.mlp_layer = nn.Sequential(
			init_(nn.Linear(obs_input_dim, 64, bias=True), activate=True),
			nn.Tanh(),
			nn.LayerNorm(64),
			init_(nn.Linear(64, 64, bias=True), activate=True),
			nn.Tanh(),
			nn.LayerNorm(64),
			)

		self.RNN = nn.GRU(input_size=64, hidden_size=64, num_layers=self.rnn_num_layers, batch_first=True)
		for name, param in self.RNN.named_parameters():
			if 'bias' in name:
				nn.init.constant_(param, 0)
			elif 'weight' in name:
				nn.init.orthogonal_(param)

		if value_norm:
			self.q_value_layer = nn.Sequential(
				nn.LayerNorm(64),
				init_(PopArt(64, 1, device=self.device), activate=False)
				)
		else:
			self.q_value_layer = nn.Sequential(
				nn.LayerNorm(64),
				init_(Linear(64, 1), activate=False)
				)
		

		self.mask_value = torch.tensor(
				torch.finfo(torch.float).min, dtype=torch.float
			)


	def forward(self, states, rnn_hidden_state, agent_masks):
		batch, timesteps, num_agents, _ = states.shape
		intermediate_embedding = self.mlp_layer(states).reshape(batch, timesteps, num_agents, -1).permute(0, 2, 1, 3).reshape(batch*num_agents, timesteps, -1)
		rnn_output, h = self.RNN(intermediate_embedding, rnn_hidden_state)
		rnn_output = rnn_output.reshape(batch, num_agents, timesteps, -1).permute(0, 2, 1, 3).reshape(batch*timesteps, num_agents, -1)
		Q_value = self.q_value_layer(rnn_output)

		return Q_value.squeeze(-1), h