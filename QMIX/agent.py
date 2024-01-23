import numpy as np 
import copy
import random
import torch
import torch.nn as nn
from functools import reduce
from torch.optim import Adam, AdamW, RMSprop
import torch.nn.functional as F
from model import QMIXNetwork, RNNQNetwork, AgentQNetwork
from utils import hard_update


EPS = 1e-2

class QMIXAgent:

	def __init__(
		self, 
		env, 
		dictionary,
		):		

		# Environment Setup
		self.env = env
		self.experiment_type = dictionary["experiment_type"]
		self.env_name = dictionary["env"]
		self.num_agents = self.env.n_agents
		self.num_actions = self.env.action_space[0].n
		self.obs_input_dim = dictionary["observation_shape"]
		self.scheduler_need = dictionary["scheduler_need"]

		# Training setup
		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"

		self.batch_size = dictionary["batch_size"]
		self.gamma = dictionary["gamma"]

		# Reward Model Setup
		self.use_reward_model = dictionary["use_reward_model"]
		# self.num_episodes_capacity = dictionary["num_episodes_capacity"]
		# self.batch_size = dictionary["batch_size"]
		self.reward_lr = dictionary["reward_lr"]
		self.variance_loss_coeff = dictionary["variance_loss_coeff"]
		self.enable_reward_grad_clip = dictionary["enable_reward_grad_clip"]
		self.reward_grad_clip_value = dictionary["reward_grad_clip_value"]
		self.reward_n_heads = dictionary["reward_n_heads"]
		self.norm_rewards = dictionary["norm_rewards"]
	
		# Model Setup
		self.learning_rate = dictionary["learning_rate"]
		self.enable_grad_clip = dictionary["enable_grad_clip"]
		self.grad_clip = dictionary["grad_clip"]
		self.tau = dictionary["tau"] # target network smoothing coefficient
		self.rnn_hidden_dim = dictionary["rnn_hidden_dim"]
		self.hidden_dim = dictionary["hidden_dim"]

		self.lambda_ = dictionary["lambda"]
		self.norm_returns = dictionary["norm_returns"]


		# Q Network
		self.Q_network = RNNQNetwork(self.obs_input_dim, self.num_actions, self.rnn_hidden_dim).to(self.device)
		self.target_Q_network = RNNQNetwork(self.obs_input_dim, self.num_actions, self.rnn_hidden_dim).to(self.device)
		# self.Q_network = AgentQNetwork(self.obs_input_dim, self.num_actions).to(self.device)
		# self.target_Q_network = AgentQNetwork(self.obs_input_dim, self.num_actions).to(self.device)
		self.QMix_network = QMIXNetwork(self.num_agents, self.hidden_dim, self.obs_input_dim * self.num_agents).to(self.device)
		self.target_QMix_network = QMIXNetwork(self.num_agents, self.hidden_dim, self.obs_input_dim * self.num_agents).to(self.device)

		self.loss_fn = nn.HuberLoss(reduction="sum")

		self.model_parameters = list(self.Q_network.parameters()) + list(self.QMix_network.parameters())
		self.optimizer = AdamW(self.model_parameters, lr=self.learning_rate, eps=1e-5)
		# self.optimizer = RMSprop(self.model_parameters, lr=self.learning_rate, alpha=0.99, eps=1e-5)

		# Loading models
		if dictionary["load_models"]:
			# For CPU
			if torch.cuda.is_available() is False:
				self.Q_network.load_state_dict(torch.load(dictionary["model_path_q_net"], map_location=torch.device('cpu')))
				self.QMix_network.load_state_dict(torch.load(dictionary["model_path_qmix_net"], map_location=torch.device('cpu')))
			# For GPU
			else:
				self.Q_network.load_state_dict(torch.load(dictionary["model_path_q_net"]))
				self.QMix_network.load_state_dict(torch.load(dictionary["model_path_qmix_net"]))

		# Copy network params
		hard_update(self.target_Q_network, self.Q_network)
		hard_update(self.target_QMix_network, self.QMix_network)
		# Disable updates for old network
		for param in self.target_Q_network.parameters():
			param.requires_grad_(False)
		for param in self.target_QMix_network.parameters():
			param.requires_grad_(False)
				

		if self.scheduler_need:
			self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1000, 20000], gamma=0.1)
		
		if self.use_reward_model:

			if self.experiment_type == "AREL":
				from AREL import AREL
				self.reward_model = AREL.Time_Agent_Transformer(
					emb=self.obs_input_dim+self.num_actions, 
					heads=dictionary["reward_n_heads"], 
					depth=dictionary["reward_depth"], 
					seq_length=dictionary["max_time_steps"], 
					n_agents=self.num_agents, 
					agent=dictionary["reward_agent_attn"], 
					dropout=dictionary["reward_dropout"], 
					wide=dictionary["reward_attn_net_wide"], 
					comp=dictionary["reward_comp"], 
					norm_rewards=dictionary["norm_rewards"],
					device=self.device,
					).to(self.device)

			elif self.experiment_type == "ATRR":
				from ATRR import ATRR
				self.reward_model = ATRR.Time_Agent_Transformer(
					obs_shape=self.obs_input_dim,
					action_shape=self.num_actions, 
					heads=dictionary["reward_n_heads"], 
					depth=dictionary["reward_depth"], 
					seq_length=dictionary["max_time_steps"], 
					n_agents=self.num_agents, 
					agent=dictionary["reward_agent_attn"], 
					dropout=dictionary["reward_dropout"], 
					wide=dictionary["reward_attn_net_wide"], 
					comp=dictionary["reward_comp"], 
					linear_compression_dim=dictionary["reward_linear_compression_dim"],
					hypernet_hidden_dim=dictionary["reward_hypernet_hidden_dim"],
					hypernet_final_dim=dictionary["reward_hypernet_final_dim"],
					device=self.device,
					).to(self.device)

			if self.norm_rewards:
				self.reward_normalizer = self.reward_model.toreward

			self.reward_optimizer = AdamW(self.reward_model.parameters(), lr=self.reward_lr, weight_decay=dictionary["reward_weight_decay"], eps=1e-5)
			
			if self.scheduler_need:
				self.scheduler_reward = optim.lr_scheduler.MultiStepLR(self.reward_optimizer, milestones=[1000, 20000], gamma=0.1)

	def get_action(self, state, last_one_hot_action, epsilon_greedy, mask_actions, actions_available):
		if np.random.uniform() < epsilon_greedy:
			actions = []
			for info in range(self.env.n_agents):
				avail_indices = [i for i, x in enumerate(actions_available[info]) if x]
				actions.append(int(np.random.choice(avail_indices)))
			# actions = [np.random.choice(self.num_actions) for _ in range(self.num_agents)]
		else:
			with torch.no_grad():
				state = torch.FloatTensor(state)
				last_one_hot_action = torch.FloatTensor(last_one_hot_action)
				mask_actions = torch.FloatTensor(mask_actions)
				final_state = torch.cat([state, last_one_hot_action], dim=-1)
				Q_values = self.Q_network(final_state.to(self.device), mask_actions.to(self.device))
				actions = Q_values.argmax(dim=-1).cpu().tolist()
				# actions = [Categorical(dist).sample().detach().cpu().item() for dist in Q_values]
		
		return actions


	def calculate_returns(self, rewards):
		returns = []
		R = 0
		
		for r in reversed(rewards):
			R = r + R * self.gamma
			returns.insert(0, R)
		
		returns_tensor = torch.stack(returns)
		
		if self.norm_returns:
			returns_tensor = (returns_tensor - returns_tensor.mean()) / returns_tensor.std()
			
		return returns_tensor

	
	def build_td_lambda_targets(self, rewards, terminated, mask, target_qs):
		# Assumes  <target_qs > in B*T*A and <reward >, <terminated >  in B*T*A, <mask > in (at least) B*T-1*1
		# Initialise  last  lambda -return  for  not  terminated  episodes
		# print(rewards.shape, terminated.shape, mask.shape, target_qs.shape)
		ret = target_qs.new_zeros(*target_qs.shape)
		ret = target_qs * (1-terminated)
		next_Q = torch.zeros(target_qs[:, -1].shape).to(self.device) # assume the Q value next to last is 0
		# ret[:, -1] = target_qs[:, -1] * (1 - (torch.sum(terminated, dim=1)>0).int())
		# Backwards  recursive  update  of the "forward  view"
		for t in range(ret.shape[1] - 1, -1,  -1):
			ret[:, t] = self.lambda_ * self.gamma * next_Q + mask[:, t] \
						* (rewards[:, t] + (1 - self.lambda_) * self.gamma * target_qs[:, t] * (1 - terminated[:, t]))
			next_Q = ret[:, t]
		# Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
		# return ret[:, 0:-1]
		return ret


	def update(self, sample, episode):
		# # sample episodes from replay buffer
		state_batch, actions_batch, last_one_hot_actions_batch, next_state_batch, next_last_one_hot_actions_batch, next_mask_actions_batch, reward_batch, done_batch, mask_batch, agent_masks_batch, max_episode_len = sample
		# # convert list to tensor
		state_batch = torch.FloatTensor(state_batch)
		actions_batch = torch.FloatTensor(actions_batch)
		last_one_hot_actions_batch = torch.FloatTensor(last_one_hot_actions_batch)
		next_state_batch = torch.FloatTensor(next_state_batch)
		next_last_one_hot_actions_batch = torch.FloatTensor(next_last_one_hot_actions_batch) # same as current one_hot_actions
		next_mask_actions_batch = torch.FloatTensor(next_mask_actions_batch)
		reward_batch = torch.FloatTensor(reward_batch)
		episodic_reward_batch = reward_batch.sum(dim=1)
		done_batch = torch.FloatTensor(done_batch)
		mask_batch = torch.FloatTensor(mask_batch)
		agent_masks_batch = torch.FloatTensor(agent_masks_batch)

		if self.norm_rewards:
			shape = episodic_rewards.shape
			self.reward_normalizer.update(episodic_reward_batch.view(-1))
			
			episodic_reward_batch = self.reward_normalizer.normalize(episodic_reward_batch.view(-1)).view(shape)

		if self.experiment_type == "AREL":
			state_actions_batch = torch.cat([state_batch, next_last_one_hot_actions_batch], dim=-1)

			reward_episode_wise, reward_time_wise = self.reward_model(
				state_actions_batch.permute(0, 2, 1, 3).to(self.device),
				team_masks=mask_batch.to(self.device),
				agent_masks=agent_masks_batch.to(self.device)
				)

			shape = reward_time_wise.shape
			reward_copy = copy.deepcopy(reward_time_wise.detach())
			reward_copy[mask_batch.view(*shape) == 0.0] = 0.0 
			reward_mean = (reward_copy.sum(dim=-1)/mask_batch.to(self.device).sum(dim=-1)).unsqueeze(-1)
			reward_var = (reward_time_wise - reward_mean)**2
			reward_var[mask_batch.view(*shape) == 0.0] = 0.0
			reward_var = reward_var.sum() / mask_batch.sum()

			reward_loss = F.huber_loss((reward_time_wise*mask_batch.view(*shape).to(self.device)).sum(dim=-1), episodic_reward_batch.to(self.device)) + self.variance_loss_coeff*reward_var


		elif self.experiment_type == "ATRR":
			agent_masks = torch.cat([agent_masks_batch, torch.ones(agent_masks_batch.shape[0], 1, agent_masks_batch.shape[2])], dim=1)

			reward_episode_wise, temporal_weights, agent_weights = self.reward_model(
				state_batch.permute(0, 2, 1, 3).to(self.device), 
				next_last_one_hot_actions_batch.permute(0, 2, 1, 3).to(self.device), 
				team_masks=torch.cat([mask_batch, torch.tensor([1]).unsqueeze(0).repeat(mask_batch.shape[0], 1)], dim=-1).to(self.device),
				# agent_masks=torch.cat([masks, torch.ones(masks.shape[0], masks.shape[1], 1)], dim=-1).to(self.device)
				agent_masks=agent_masks.to(self.device)
				)
			
			entropy_temporal_weights = -torch.sum(torch.sum((temporal_weights * torch.log(torch.clamp(temporal_weights, 1e-10, 1.0)) * mask_batch.to(self.device)), dim=-1))/mask_batch.sum()
			entropy_agent_weights = -torch.sum(torch.sum((agent_weights.reshape(-1, self.num_agents) * torch.log(torch.clamp(agent_weights.reshape(-1, self.num_agents), 1e-10, 1.0)) * agent_masks.reshape(-1, self.num_agents).to(self.device)), dim=-1))/agent_masks.sum()
			
			reward_loss = F.huber_loss(reward_episode_wise.reshape(-1), episodic_reward_batch.to(self.device))

		self.reward_optimizer.zero_grad()
		reward_loss.backward()
		if self.enable_reward_grad_clip:
			grad_norm_value_reward = torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.reward_grad_clip_value)
		else:
			total_norm = 0
			for name, p in self.reward_model.named_parameters():
				if p.requires_grad is False:
					continue
				param_norm = p.grad.detach().data.norm(2)
				total_norm += param_norm.item() ** 2
			grad_norm_value_reward = torch.tensor([total_norm ** 0.5])
		self.reward_optimizer.step()

		self.Q_network.rnn_hidden_state = None
		self.target_Q_network.rnn_hidden_state = None

		Q_mix_values = []
		target_Q_mix_values = []

		for t in range(max_episode_len):
			# train in time order
			states_slice = state_batch[:,t].reshape(-1, self.obs_input_dim)
			last_one_hot_actions_slice = last_one_hot_actions_batch[:,t].reshape(-1, self.num_actions)
			actions_slice = actions_batch[:, t].reshape(-1)
			next_states_slice = next_state_batch[:,t].reshape(-1, self.obs_input_dim)
			next_last_one_hot_actions_slice = next_last_one_hot_actions_batch[:,t].reshape(-1, self.num_actions)
			next_mask_actions_slice = next_mask_actions_batch[:,t].reshape(-1, self.num_actions)
			reward_slice = reward_batch[:, t].reshape(-1)
			done_slice = done_batch[:, t].reshape(-1)
			mask_slice = mask_batch[:, t].reshape(-1)

			final_state_slice = torch.cat([states_slice, last_one_hot_actions_slice], dim=-1)
			Q_values = self.Q_network(final_state_slice.to(self.device))
			Q_evals = torch.gather(Q_values, dim=-1, index=actions_slice.long().unsqueeze(-1).to(self.device)).squeeze(-1)
			Q_mix = self.QMix_network(Q_evals, state_batch[:,t].reshape(-1, self.num_agents*self.obs_input_dim).to(self.device)).squeeze(-1).squeeze(-1) * mask_slice.to(self.device)

			with torch.no_grad():
				next_final_state_slice = torch.cat([next_states_slice, next_last_one_hot_actions_slice], dim=-1)
				Q_evals_next = self.Q_network(next_final_state_slice.to(self.device))
				Q_targets = self.target_Q_network(next_final_state_slice.to(self.device), next_mask_actions_slice.to(self.device))
				a_argmax = torch.argmax(Q_evals_next, dim=-1, keepdim=True)
				Q_targets = torch.gather(Q_targets, dim=-1, index=a_argmax.to(self.device)).squeeze(-1)
				Q_mix_target = self.target_QMix_network(Q_targets, next_state_batch[:, t].reshape(-1, self.num_agents*self.obs_input_dim).to(self.device)).squeeze(-1).squeeze(-1)
				
			Q_mix_values.append(Q_mix)
			target_Q_mix_values.append(Q_mix_target)


		Q_mix_values = torch.stack(Q_mix_values, dim=1).to(self.device)
		target_Q_mix_values = torch.stack(target_Q_mix_values, dim=1).to(self.device)

		target_Q_mix_values = self.build_td_lambda_targets(reward_batch[:, :max_episode_len].to(self.device), done_batch[:, :max_episode_len].to(self.device), mask_batch[:, :max_episode_len].to(self.device), target_Q_mix_values)

		Q_loss = self.loss_fn(Q_mix_values, target_Q_mix_values.detach()) / mask_batch.to(self.device).sum()

		self.optimizer.zero_grad()
		Q_loss.backward()
		if self.enable_grad_clip:
			grad_norm = torch.nn.utils.clip_grad_norm_(self.model_parameters, self.grad_clip).item()
		else:
			grad_norm = 0
			for p in self.model_parameters:
				param_norm = p.grad.detach().data.norm(2)
				grad_norm += param_norm.item() ** 2
			grad_norm = grad_norm ** 0.5
		self.optimizer.step()

		torch.cuda.empty_cache()

		if self.experiment_type == "AREL":
			return Q_loss.item(), grad_norm, reward_loss.item(), reward_var.item(), grad_norm_value_reward.item()
		elif self.experiment_type == "ATRR":
			return Q_loss.item(), grad_norm, reward_loss.item(), entropy_temporal_weights.item(), entropy_agent_weights.item(), grad_norm_value_reward.item()