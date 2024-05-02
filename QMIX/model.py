import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize weights
def init(module, weight_init, bias_init, gain=1):
	weight_init(module.weight.data, gain=gain)
	if module.bias is not None:
		bias_init(module.bias.data)
	return module

def init_(m, gain=0.01, activate=False):
	if activate:
		gain = nn.init.calculate_gain('relu')
	return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class RNNQNetwork(nn.Module):
	def __init__(self, input_dim, num_actions, rnn_hidden_dim, rnn_num_layers):
		super(RNNQNetwork, self).__init__()

		self.mask_value = torch.tensor(
				torch.finfo(torch.float).min, dtype=torch.float
			)
		self.rnn_num_layers = rnn_num_layers

		# self.rnn_hidden_state = None
		# self.Layer1 = nn.Linear(input_dim + num_actions, hidden_dim)
		self.Layer_1 = nn.Sequential(
			init_(nn.Linear(input_dim, rnn_hidden_dim*2), activate=True),
			nn.GELU(),
			nn.LayerNorm(rnn_hidden_dim*2),
			init_(nn.Linear(rnn_hidden_dim*2, rnn_hidden_dim), activate=True),
			nn.GELU(),
			nn.LayerNorm(rnn_hidden_dim),
			)
		# self.RNN = nn.GRUCell(hidden_dim, hidden_dim)
		self.RNN = nn.GRU(input_size=rnn_hidden_dim, hidden_size=rnn_hidden_dim, num_layers=rnn_num_layers, batch_first=True)
		# self.Layer2 = nn.Linear(hidden_dim, num_actions)
		self.Layer_2 = nn.Sequential(
			nn.LayerNorm(rnn_hidden_dim),
			init_(nn.Linear(rnn_hidden_dim, num_actions), gain=0.01)
			)

		# self.apply(weights_init)

	def forward(self, states_actions, hidden_state, action_masks):
		batch, timesteps, num_agents, _ = states_actions.shape
		# x = F.gelu(self.Layer1(states_actions))
		# self.rnn_hidden_state = self.RNN(x, self.rnn_hidden_state)

		intermediate = self.Layer_1(states_actions)
		intermediate = intermediate.permute(0, 2, 1, 3).reshape(batch*num_agents, timesteps, -1)
		hidden_state = hidden_state.reshape(self.rnn_num_layers, batch*num_agents, -1)
		output, h = self.RNN(intermediate, hidden_state)
		output = output.reshape(batch, num_agents, timesteps, -1).permute(0, 2, 1, 3)
		Q_a_values = self.Layer_2(output)

		# Q_a_values = self.Layer2(self.rnn_hidden_state)
		Q_a_values = torch.where(action_masks, Q_a_values, self.mask_value)
		
		return Q_a_values, h


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