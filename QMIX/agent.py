import numpy as np 
import copy
import random
import torch
import torch.nn as nn
from functools import reduce
from torch.optim import Adam, AdamW, RMSprop
import torch.nn.functional as F
from model import QMIXNetwork, RNNQNetwork
from utils import hard_update
import torch.optim as optim


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
		self.num_agents = dictionary["num_agents"]
		self.num_actions = self.env.action_space[0].n
		self.q_obs_input_dim = dictionary["q_observation_shape"]
		self.q_mix_obs_input_dim = dictionary["q_mix_observation_shape"]
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
		self.temporal_score_coefficient = dictionary["temporal_score_efficient"]
		self.agent_score_coefficient = dictionary["agent_score_efficient"]
		self.variance_loss_coeff = dictionary["variance_loss_coeff"]
		self.enable_reward_grad_clip = dictionary["enable_reward_grad_clip"]
		self.reward_grad_clip_value = dictionary["reward_grad_clip_value"]
		self.reward_n_heads = dictionary["reward_n_heads"]
		self.norm_rewards = dictionary["norm_rewards"]
	
		# Model Setup
		self.algorithm_type = dictionary["algorithm_type"]
		self.learning_rate = dictionary["learning_rate"]
		self.enable_grad_clip = dictionary["enable_grad_clip"]
		self.grad_clip = dictionary["grad_clip"]
		self.tau = dictionary["tau"] # target network smoothing coefficient
		self.rnn_num_layers = dictionary["rnn_num_layers"]
		self.rnn_hidden_dim = dictionary["rnn_hidden_dim"]
		self.hidden_dim = dictionary["hidden_dim"]

		self.lambda_ = dictionary["lambda"]
		self.norm_returns = dictionary["norm_returns"]


		# Q Network
		self.Q_network = RNNQNetwork(self.q_obs_input_dim+self.num_actions, self.num_actions, self.rnn_hidden_dim, self.rnn_num_layers).to(self.device)
		self.target_Q_network = RNNQNetwork(self.q_obs_input_dim+self.num_actions, self.num_actions, self.rnn_hidden_dim, self.rnn_num_layers).to(self.device)
		
		self.QMix_network = QMIXNetwork(self.num_agents, self.hidden_dim, self.q_mix_obs_input_dim).to(self.device)
		self.target_QMix_network = QMIXNetwork(self.num_agents, self.hidden_dim, self.q_mix_obs_input_dim).to(self.device)

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


		# Load models


		# Disable updates for old network
		for param in self.target_Q_network.parameters():
			param.requires_grad_(False)
		for param in self.target_QMix_network.parameters():
			param.requires_grad_(False)
				

		if self.scheduler_need:
			self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10000, 30000], gamma=1.0)
		
		if self.use_reward_model:

			if "AREL" in self.experiment_type:
				from AREL import AREL
				self.reward_model = AREL.Time_Agent_Transformer(
					emb=dictionary["reward_model_obs_shape"]+self.num_actions, #self.q_obs_input_dim+self.num_actions
					heads=dictionary["reward_n_heads"], 
					depth=dictionary["reward_depth"], 
					seq_length=dictionary["max_time_steps"], 
					n_agents=self.num_agents, 
					agent=dictionary["reward_agent_attn"], 
					dropout=dictionary["reward_dropout"], 
					wide=dictionary["reward_attn_net_wide"], 
					comp=dictionary["reward_comp"], 
					norm_rewards=dictionary["norm_rewards"],
					linear_compression_dim=dictionary["reward_linear_compression_dim"],
					device=self.device,
					).to(self.device)

			elif "ATRR" in self.experiment_type:
				from ATRR import ATRR
				self.reward_model = ATRR.Time_Agent_Transformer(
					obs_shape=dictionary["reward_model_obs_shape"],
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
					norm_rewards=dictionary["norm_rewards"],
					device=self.device,
					).to(self.device)

			if self.norm_rewards:
				self.reward_normalizer = self.reward_model.rblocks[-1]
				print(self.reward_normalizer)

			
			if dictionary["load_models"]:
				# For CPU
				if torch.cuda.is_available() is False:
					self.reward_model.load_state_dict(torch.load(dictionary["model_path_reward_net"], map_location=torch.device('cpu')))
				# For GPU
				else:
					self.reward_model.load_state_dict(torch.load(dictionary["model_path_reward_net"]))


			self.reward_optimizer = AdamW(self.reward_model.parameters(), lr=self.reward_lr, weight_decay=dictionary["reward_weight_decay"], eps=1e-5)
			
			if self.scheduler_need:
				self.scheduler_reward = optim.lr_scheduler.MultiStepLR(self.reward_optimizer, milestones=[10000, 30000], gamma=0.5)

		else:

			self.reward_model = None


	def get_action(self, state, last_one_hot_action, rnn_hidden_state, epsilon_greedy, action_masks):
		if np.random.uniform() < epsilon_greedy:
			actions = []
			for info in range(self.num_agents):
				avail_indices = [i for i, x in enumerate(action_masks[info]) if x]
				actions.append(int(np.random.choice(avail_indices)))
			# actions = [np.random.choice(self.num_actions) for _ in range(self.num_agents)]
		else:
			with torch.no_grad():
				state = torch.FloatTensor(state)
				last_one_hot_action = torch.FloatTensor(last_one_hot_action)
				rnn_hidden_state = torch.FloatTensor(rnn_hidden_state)
				action_masks = torch.BoolTensor(action_masks)
				final_state = torch.cat([state, last_one_hot_action], dim=-1).unsqueeze(0).unsqueeze(0)
				Q_values, rnn_hidden_state = self.Q_network(final_state.to(self.device), rnn_hidden_state.to(self.device), action_masks.to(self.device)) #+ mask_actions
				actions = Q_values.reshape(self.num_agents, self.num_actions).argmax(dim=-1).cpu().tolist()
				rnn_hidden_state = rnn_hidden_state.cpu().numpy()
				# actions = [Categorical(dist).sample().detach().cpu().item() for dist in Q_values]
		
		return actions, rnn_hidden_state


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

	
	def build_td_lambda_targets(self, rewards, mask, target_qs):
		# Assumes  <target_qs > in B*T*A and <reward >, <terminated >  in B*T*A, <mask > in (at least) B*T-1*1
		# Initialise  last  lambda -return  for  not  terminated  episodes
		# print(rewards.shape, terminated.shape, mask.shape, target_qs.shape)
		ret = target_qs.new_zeros(*target_qs.shape)
		ret = target_qs * mask
		next_Q = torch.zeros(target_qs[:, -1].shape).to(self.device) # assume the Q value next to last is 0
		# ret[:, -1] = target_qs[:, -1] * (1 - (torch.sum(terminated, dim=1)>0).int())
		# Backwards  recursive  update  of the "forward  view"
		for t in range(ret.shape[1] - 1, -1,  -1):
			ret[:, t] = self.lambda_ * self.gamma * next_Q + mask[:, t] \
						* (rewards[:, t] + (1 - self.lambda_) * self.gamma * target_qs[:, t] * mask[:, t])
			next_Q = ret[:, t]
		# Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
		# return ret[:, 0:-1]
		return ret

	def reward_model_output(self, buffer):
		latest_sample_index = buffer.episode
		state_batch = torch.from_numpy(buffer.buffer['state'][latest_sample_index]).float().unsqueeze(0)
		next_last_one_hot_actions_batch = torch.from_numpy(buffer.buffer['next_last_one_hot_actions'][latest_sample_index]).float().unsqueeze(0)
		mask_batch = 1-torch.from_numpy(buffer.buffer['done'][latest_sample_index]).float().unsqueeze(0)
		agent_masks_batch = 1-torch.from_numpy(buffer.buffer['indiv_dones'][latest_sample_index]).float().unsqueeze(0)
		
		with torch.no_grad():
			if "AREL" in self.experiment_type:
				state_actions_batch = torch.cat([state_batch, next_last_one_hot_actions_batch], dim=-1)

				reward_episode_wise, reward_time_wise, temporal_weights, temporal_scores, agent_weights, agent_scores = self.reward_model(
					state_actions_batch.permute(0, 2, 1, 3).to(self.device),
					team_masks=mask_batch.to(self.device),
					agent_masks=agent_masks_batch.to(self.device)
					)
				
				if self.experiment_type == "AREL_agent":
					reward_agent_wise = reward_time_wise.unsqueeze(-1) * agent_weights.cpu()
					return reward_agent_wise

			elif "ATRR" in self.experiment_type:
				# agent_masks = torch.cat([agent_masks_batch, torch.ones(agent_masks_batch.shape[0], 1, agent_masks_batch.shape[2])], dim=1)

				reward_episode_wise, temporal_weights, agent_weights, temporal_scores, agent_scores, _, _ = self.reward_model(
					state_batch.permute(0, 2, 1, 3).to(self.device), 
					next_last_one_hot_actions_batch.permute(0, 2, 1, 3).to(self.device), 
					# team_masks=torch.cat([mask_batch, torch.tensor([1]).unsqueeze(0).repeat(mask_batch.shape[0], 1)], dim=-1).to(self.device),
					# agent_masks=torch.cat([masks, torch.ones(masks.shape[0], masks.shape[1], 1)], dim=-1).to(self.device)
					team_masks=mask_batch.to(self.device),
					agent_masks=agent_masks_batch.to(self.device)
					)

				reward_time_wise = (reward_episode_wise * temporal_weights).cpu()

				if self.experiment_type == "ATRR_agent":
					reward_agent_wise = reward_time_wise.unsqueeze(-1) * agent_weights[:, :-1, :].cpu()

					return reward_agent_wise

			return reward_time_wise

		

	def update_reward_model(self, sample):
		# sample episodes from replay buffer
		reward_model_obs_batch, next_last_one_hot_actions_batch, reward_batch, mask_batch, agent_masks_batch, episode_len_batch = sample
		# convert numpy array to tensor
		reward_model_obs_batch = torch.from_numpy(reward_model_obs_batch).float()
		next_last_one_hot_actions_batch = torch.from_numpy(next_last_one_hot_actions_batch).float() # same as current one_hot_actions
		reward_batch = torch.from_numpy(reward_batch).float()
		episodic_reward_batch = reward_batch.sum(dim=1)
		mask_batch = torch.from_numpy(mask_batch).float()
		agent_masks_batch = torch.from_numpy(agent_masks_batch).float()
		episode_len_batch = torch.from_numpy(episode_len_batch).long()

		# if self.norm_rewards:
		# 	shape = episodic_reward_batch.shape
		# 	self.reward_normalizer.update(episodic_reward_batch.view(-1))
			
		# 	episodic_reward_batch = self.reward_normalizer.normalize(episodic_reward_batch.view(-1)).view(shape)

		if "AREL" in self.experiment_type:
			state_actions_batch = torch.cat([reward_model_obs_batch, next_last_one_hot_actions_batch], dim=-1)  # state_actions_batch.size = (b, t, n_agents, e)
			reward_episode_wise, reward_time_wise, _, _, _, _ = self.reward_model(
				state_actions_batch.permute(0, 2, 1, 3).to(self.device),
				team_masks=mask_batch.to(self.device),
				agent_masks=agent_masks_batch.to(self.device)
				)


			shape = reward_time_wise.shape
			# reward_copy = copy.deepcopy(reward_time_wise.detach())
			# reward_copy[mask_batch.view(*shape) == 0.0] = 0.0 
			# reward_mean = (reward_copy.sum(dim=-1)/mask_batch.to(self.device).sum(dim=-1)).unsqueeze(-1)
			# print("Check for NaN")
			# print(torch.isnan(reward_time_wise).any())
			# print(torch.isnan(reward_mean).any())
			# reward_var = (reward_time_wise - reward_mean)**2
			# reward_var[mask_batch.view(*shape) == 0.0] = 0.0
			# reward_var = reward_var.sum() / mask_batch.sum()

			# print("Huber Loss")
			# print(F.huber_loss((reward_time_wise*mask_batch.view(*shape).to(self.device)).sum(dim=-1), episodic_reward_batch.to(self.device)))
			# print("Reward Var")
			# print(reward_var)
			reward_var = torch.tensor([-1])
			reward_loss = F.huber_loss((reward_time_wise*mask_batch.view(*shape).to(self.device)).sum(dim=-1), episodic_reward_batch.to(self.device)) #+ self.variance_loss_coeff*reward_var

		elif "ATRR" in self.experiment_type:
			# agent_masks = torch.cat([agent_masks_batch, torch.ones(agent_masks_batch.shape[0], 1, agent_masks_batch.shape[2])], dim=1)

			# reward_episode_wise, temporal_weights, agent_weights, temporal_scores, agent_scores, state_latent_embeddings, dynamics_model_output = self.reward_model(
			# 	state_batch.permute(0, 2, 1, 3).to(self.device), 
			# 	next_last_one_hot_actions_batch.permute(0, 2, 1, 3).to(self.device), 
			# 	# team_masks=torch.cat([mask_batch, torch.tensor([1]).unsqueeze(0).repeat(mask_batch.shape[0], 1)], dim=-1).to(self.device),
			# 	# agent_masks=torch.cat([masks, torch.ones(masks.shape[0], masks.shape[1], 1)], dim=-1).to(self.device)
			# 	team_masks=mask_batch.to(self.device),
			# 	agent_masks=agent_masks_batch.to(self.device),
			# 	episode_len=episode_len_batch.to(self.device),
			# 	)

			reward_agent_time, temporal_weights, agent_weights, temporal_scores, agent_scores, state_latent_embeddings, dynamics_model_output = self.reward_model(
				reward_model_obs_batch.permute(0, 2, 1, 3).to(self.device), 
				next_last_one_hot_actions_batch.permute(0, 2, 1, 3).to(self.device), 
				# team_masks=torch.cat([mask_batch, torch.tensor([1]).unsqueeze(0).repeat(mask_batch.shape[0], 1)], dim=-1).to(self.device),
				# agent_masks=torch.cat([masks, torch.ones(masks.shape[0], masks.shape[1], 1)], dim=-1).to(self.device)
				team_masks=mask_batch.to(self.device),
				agent_masks=agent_masks_batch.to(self.device),
				episode_len=episode_len_batch.to(self.device),
				)

			entropy_temporal_weights = -torch.sum(torch.sum((temporal_weights * torch.log(torch.clamp(temporal_weights, 1e-10, 1.0)) * mask_batch.to(self.device)), dim=-1))/mask_batch.shape[0]
			entropy_agent_weights = -torch.sum(torch.sum((agent_weights.reshape(-1, self.num_agents) * torch.log(torch.clamp(agent_weights.reshape(-1, self.num_agents), 1e-10, 1.0)) * agent_masks_batch.reshape(-1, self.num_agents).to(self.device)), dim=-1))/mask_batch.sum() #agent_masks.reshape(-1, self.num_agents).shape[0]
			
			# div = torch.mean(torch.distributions.kl.kl_divergence(dynamics_model_output, state_latent_embeddings.detach()))
			# self.free_nats = 3
			# div = torch.max(div, div.new_full(div.size(), self.free_nats))
			reward_loss = F.huber_loss(reward_agent_time.reshape(reward_model_obs_batch.shape[0], -1).sum(dim=-1), episodic_reward_batch.to(self.device)) + self.temporal_score_coefficient * (temporal_scores**2).sum() + self.agent_score_coefficient * (agent_scores**2).sum()
			# reward_loss = F.huber_loss(reward_episode_wise.reshape(-1), episodic_reward_batch.to(self.device)) + self.temporal_score_coefficient * (temporal_scores**2).sum() + self.agent_score_coefficient * (agent_scores**2).sum()
			# reward_loss = F.huber_loss(reward_episode_wise.reshape(-1), episodic_reward_batch.to(self.device)) + div + self.temporal_score_coefficient * (temporal_scores**2).sum() + self.agent_score_coefficient * (agent_scores**2).sum()
			# reward_loss = F.huber_loss(reward_episode_wise.reshape(-1), episodic_reward_batch.to(self.device)) + F.huber_loss(state_latent_embeddings.detach(), dynamics_model_output) + self.temporal_score_coefficient * (temporal_scores**2).sum() + self.agent_score_coefficient * (agent_scores**2).sum()

		self.reward_optimizer.zero_grad()
		reward_loss.backward()
		if self.enable_reward_grad_clip:
			grad_norm_value_reward = torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.reward_grad_clip_value)
		else:
			total_norm = 0
			for name, p in self.reward_model.named_parameters():
				if p.requires_grad is False or p.grad is None:
					continue
				param_norm = p.grad.detach().data.norm(2)
				total_norm += param_norm.item() ** 2
			grad_norm_value_reward = torch.tensor([total_norm ** 0.5])
		self.reward_optimizer.step()

		if "AREL" in self.experiment_type:
			return reward_loss.item(), reward_var.item(), grad_norm_value_reward.item()
		elif "ATRR" in self.experiment_type:
			return reward_loss.item(), entropy_temporal_weights.item(), entropy_agent_weights.item(), grad_norm_value_reward.item()


	def update(self, sample):
		# # sample episodes from replay buffer
		# state_batch, global_state_batch, actions_batch, last_one_hot_actions_batch, next_state_batch, next_global_state_batch, next_last_one_hot_actions_batch, next_mask_actions_batch, reward_batch, done_batch, mask_batch, agent_masks_batch, episode_len_batch, max_episode_len = sample
		# # # convert list to tensor
		# state_batch = torch.FloatTensor(state_batch)
		# global_state_batch = torch.FloatTensor(global_state_batch)
		# actions_batch = torch.FloatTensor(actions_batch)
		# last_one_hot_actions_batch = torch.FloatTensor(last_one_hot_actions_batch)
		# next_state_batch = torch.FloatTensor(next_state_batch)
		# next_global_state_batch = torch.FloatTensor(next_global_state_batch)
		# next_last_one_hot_actions_batch = torch.FloatTensor(next_last_one_hot_actions_batch) # same as current one_hot_actions
		# next_mask_actions_batch = torch.FloatTensor(next_mask_actions_batch)
		# reward_batch = torch.FloatTensor(reward_batch)
		# done_batch = torch.FloatTensor(done_batch)
		# mask_batch = torch.FloatTensor(mask_batch)
		# agent_masks_batch = torch.FloatTensor(agent_masks_batch)
		# episode_len_batch = torch.LongTensor(episode_len_batch)

		# if "AREL" in self.experiment_type:
		# 	state_actions_batch = torch.cat([state_batch, next_last_one_hot_actions_batch], dim=-1)

		# 	with torch.no_grad():
		# 		reward_episode_wise, reward_time_wise, _, _, agent_weights, _ = self.reward_model(
		# 			state_actions_batch.permute(0, 2, 1, 3).to(self.device),
		# 			team_masks=mask_batch.to(self.device),
		# 			agent_masks=agent_masks_batch.to(self.device)
		# 			)

		# 	reward_batch = reward_time_wise.cpu()

		# 	if self.experiment_type == "AREL_agent":
		# 		reward_batch = reward_batch.unsqueeze(-1) * agent_weights.cpu()


		# elif "ATRR" in self.experiment_type:
		# 	# agent_masks = torch.cat([agent_masks_batch, torch.ones(agent_masks_batch.shape[0], 1, agent_masks_batch.shape[2])], dim=1)

		# 	with torch.no_grad():
		# 		reward_episode_wise, temporal_weights, agent_weights, _, _, _, _ = self.reward_model(
		# 			state_batch.permute(0, 2, 1, 3).to(self.device), 
		# 			next_last_one_hot_actions_batch.permute(0, 2, 1, 3).to(self.device), 
		# 			# team_masks=torch.cat([mask_batch, torch.tensor([1]).unsqueeze(0).repeat(mask_batch.shape[0], 1)], dim=-1).to(self.device),
		# 			# agent_masks=torch.cat([masks, torch.ones(masks.shape[0], masks.shape[1], 1)], dim=-1).to(self.device)
		# 			team_masks=mask_batch.to(self.device),
		# 			agent_masks=agent_masks_batch.to(self.device),
		# 			episode_len=episode_len_batch.to(self.device),
		# 			)

		# 	if self.norm_rewards:
		# 		shape = reward_episode_wise.shape
		# 		reward_episode_wise = self.reward_normalizer.denormalize(reward_episode_wise.view(-1)).view(shape)

		# 	reward_batch = (reward_episode_wise * temporal_weights).cpu()

		# 	if self.experiment_type == "ATRR_agent":
		# 		reward_batch = reward_batch.unsqueeze(-1) * agent_weights[:, :-1, :].cpu()



		# self.Q_network.rnn_hidden_state = None
		# self.target_Q_network.rnn_hidden_state = None

		# if self.experiment_type != "ATRR_agent" and self.experiment_type != "AREL_agent":
		# 	Q_mix_values = []
		# 	target_Q_mix_values = []
		# else:
		# 	Q_vals_ = []
		# 	Q_targets_ = []

		# for t in range(max_episode_len):
		# 	# train in time order
		# 	states_slice = state_batch[:,t].reshape(-1, self.q_obs_input_dim)
		# 	global_states_slice = global_state_batch[:,t].reshape(-1, self.q_mix_obs_input_dim)
		# 	last_one_hot_actions_slice = last_one_hot_actions_batch[:,t].reshape(-1, self.num_actions)
		# 	actions_slice = actions_batch[:, t].reshape(-1)
		# 	next_states_slice = next_state_batch[:,t].reshape(-1, self.q_obs_input_dim)
		# 	next_global_states_slice = next_global_state_batch[:,t].reshape(-1, self.q_mix_obs_input_dim)
		# 	next_last_one_hot_actions_slice = next_last_one_hot_actions_batch[:,t].reshape(-1, self.num_actions)
		# 	next_mask_actions_slice = next_mask_actions_batch[:,t].reshape(-1, self.num_actions)
		# 	reward_slice = reward_batch[:, t].reshape(-1)
		# 	done_slice = done_batch[:, t].reshape(-1)
		# 	mask_slice = mask_batch[:, t].reshape(-1)

		# 	final_state_slice = torch.cat([states_slice, last_one_hot_actions_slice], dim=-1)
		# 	Q_values = self.Q_network(final_state_slice.to(self.device))
		# 	Q_evals = torch.gather(Q_values, dim=-1, index=actions_slice.long().unsqueeze(-1).to(self.device)).squeeze(-1)

		# 	if self.experiment_type != "ATRR_agent" and self.experiment_type != "AREL_agent":
		# 		Q_mix = self.QMix_network(Q_evals, global_states_slice.to(self.device)).squeeze(-1).squeeze(-1) * mask_slice.to(self.device)

		# 	with torch.no_grad():
		# 		next_final_state_slice = torch.cat([next_states_slice, next_last_one_hot_actions_slice], dim=-1)
		# 		Q_evals_next = self.Q_network(next_final_state_slice.to(self.device))
		# 		Q_targets = self.target_Q_network(next_final_state_slice.to(self.device))
		# 		a_argmax = torch.argmax(Q_evals_next+(1-next_mask_actions_slice.to(self.device)*(-1e9)), dim=-1, keepdim=True)
		# 		Q_targets = torch.gather(Q_targets, dim=-1, index=a_argmax.to(self.device)).squeeze(-1)
				
		# 		if self.experiment_type != "ATRR_agent" and self.experiment_type != "AREL_agent":
		# 			Q_mix_target = self.target_QMix_network(Q_targets, next_global_states_slice.to(self.device)).squeeze(-1).squeeze(-1)
				
		# 	if self.experiment_type != "ATRR_agent" and self.experiment_type != "AREL_agent":
		# 		Q_mix_values.append(Q_mix)
		# 		target_Q_mix_values.append(Q_mix_target)
		# 	else:
		# 		Q_vals_.append(Q_evals.reshape(-1, self.num_agents))
		# 		Q_targets_.append(Q_targets.reshape(-1, self.num_agents))

		# if self.experiment_type != "ATRR_agent" and self.experiment_type != "AREL_agent":
		# 	Q_mix_values = torch.stack(Q_mix_values, dim=1).to(self.device)
		# 	target_Q_mix_values = torch.stack(target_Q_mix_values, dim=1).to(self.device)

		# 	target_Q_mix_values = self.build_td_lambda_targets(reward_batch[:, :max_episode_len].to(self.device), mask_batch[:, :max_episode_len].to(self.device), target_Q_mix_values)

		# 	Q_loss = self.loss_fn(Q_mix_values, target_Q_mix_values.detach()) / mask_batch.to(self.device).sum()
		# else:
		# 	Q_vals_ = torch.stack(Q_vals_, dim=1).to(self.device)
		# 	Q_targets_ = torch.stack(Q_targets_, dim=1).to(self.device)

		# 	Q_targets_ = self.build_td_lambda_targets(reward_batch[:, :max_episode_len].to(self.device), agent_masks_batch[:, :max_episode_len].to(self.device), Q_targets_)
		
		# 	Q_loss = self.loss_fn(Q_vals_, Q_targets_.detach()) / mask_batch.to(self.device).sum()


		state_batch, rnn_hidden_state_batch, full_state_batch, actions_batch, last_one_hot_actions_batch, mask_actions_batch, next_state_batch, next_rnn_hidden_state_batch, next_full_state_batch, \
		next_last_one_hot_actions_batch, next_mask_actions_batch, reward_batch, done_batch, indiv_dones_batch, next_indiv_dones_batch, team_mask_batch, max_episode_len, target_Q_values = sample

		final_state_batch = torch.cat([state_batch, last_one_hot_actions_batch], dim=-1)
		Q_values, _ = self.Q_network(final_state_batch.to(self.device), rnn_hidden_state_batch.to(self.device), torch.ones_like(next_mask_actions_batch).bool().to(self.device))
		Q_evals = (torch.gather(Q_values, dim=-1, index=actions_batch.to(self.device)) * (1-indiv_dones_batch).to(self.device)).squeeze(-1)
		
		if self.algorithm_type == "IDQN":
			Q_loss = self.loss_fn(Q_evals.reshape(-1), target_Q_values.reshape(-1).to(self.device)) / (1-indiv_dones_batch).to(self.device).sum()
		else:
			Q_mix = self.QMix_network(Q_evals, next_full_state_batch.to(self.device)).reshape(-1) #* team_mask_batch.reshape(-1).to(self.device)

			Q_mix *= (1-done_batch).reshape(-1).to(self.device)

			target_Q_mix_values *= (1-done_batch).squeeze(-1)

			Q_loss = self.loss_fn(Q_mix, target_Q_values.reshape(-1).to(self.device)) / team_mask_batch.to(self.device).sum()
 
		self.optimizer.zero_grad()
		Q_loss.backward()
		if self.enable_grad_clip:
			grad_norm = torch.nn.utils.clip_grad_norm_(self.model_parameters, self.grad_clip).item()
		else:
			grad_norm = 0
			for p in self.model_parameters:
				if p.grad is not None:
					param_norm = p.grad.detach().data.norm(2)
					grad_norm += param_norm.item() ** 2
			grad_norm = grad_norm ** 0.5
		self.optimizer.step()

		torch.cuda.empty_cache()

		return Q_loss.item(), grad_norm