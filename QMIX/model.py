import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize weights
def weights_init(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight, gain=1)
		torch.nn.init.constant_(m.bias, 0)


class RNNQNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_dim):
		super(RNNQNetwork, self).__init__()

		self.rnn_hidden_state = None
		self.Layer1 = nn.Linear(num_inputs + num_actions, hidden_dim)
		self.RNN = nn.GRUCell(hidden_dim, hidden_dim)
		self.Layer2 = nn.Linear(hidden_dim, num_actions)

		self.apply(weights_init)

	def forward(self, states_actions):
		x = F.gelu(self.Layer1(states_actions))
		self.rnn_hidden_state = self.RNN(x, self.rnn_hidden_state)
		Q_a_values = self.Layer2(self.rnn_hidden_state)
		return Q_a_values


class AgentQNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions):
		super(AgentQNetwork, self).__init__()

		self.QNet = nn.Sequential(
			nn.Linear(num_inputs+num_actions, 128),
			nn.GELU(),
			nn.Linear(128, 64),
			nn.GELU(),
			nn.Linear(64, num_actions)
			)

		self.reset_parameters()

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""

		# EMBEDDINGS
		nn.init.xavier_uniform_(self.QNet[0].weight)
		nn.init.xavier_uniform_(self.QNet[2].weight)
		nn.init.xavier_uniform_(self.QNet[4].weight)

	def forward(self, states_actions):
		return self.QNet(states_actions)


class QMIXNetwork(nn.Module):
	def __init__(self, num_agents, hidden_dim, total_obs_dim):
		super(QMIXNetwork, self).__init__()
		self.num_agents = num_agents
		self.hidden_dim = hidden_dim

		self.hyper_w1 = nn.Sequential(
			nn.Linear(total_obs_dim, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, num_agents * hidden_dim)
			)
		self.hyper_b1 = nn.Sequential(
			nn.Linear(total_obs_dim, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, hidden_dim)
			)
		self.hyper_w2 = nn.Sequential(
			nn.Linear(total_obs_dim, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, hidden_dim)
			)
		self.hyper_b2 = nn.Sequential(
			nn.Linear(total_obs_dim, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, 1)
			)

		# self.apply(weights_init)

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""

		# EMBEDDINGS
		nn.init.xavier_uniform_(self.hyper_w1[0].weight)
		nn.init.xavier_uniform_(self.hyper_w1[2].weight)

		nn.init.xavier_uniform_(self.hyper_b1[0].weight)
		nn.init.xavier_uniform_(self.hyper_b1[2].weight)

		nn.init.xavier_uniform_(self.hyper_w2[0].weight)
		nn.init.xavier_uniform_(self.hyper_w2[2].weight)

		nn.init.xavier_uniform_(self.hyper_b2[0].weight)
		nn.init.xavier_uniform_(self.hyper_b2[2].weight)

	def forward(self, q_values, total_obs):
		q_values = q_values.reshape(-1, 1, self.num_agents)
		w1 = torch.abs(self.hyper_w1(total_obs))
		b1 = self.hyper_b1(total_obs)
		w1 = w1.reshape(-1, self.num_agents, self.hidden_dim)
		b1 = b1.reshape(-1, 1, self.hidden_dim)

		x = F.elu(torch.bmm(q_values, w1) + b1)

		w2 = torch.abs(self.hyper_w2(total_obs))
		b2 = self.hyper_b2(total_obs)
		w2 = w2.reshape(-1, self.hidden_dim, 1)
		b2 = b2.reshape(-1, 1, 1)

		x = torch.bmm(x, w2) + b2

		return x