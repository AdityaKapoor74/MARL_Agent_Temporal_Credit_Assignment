"""
This file provides utility classes and functions for the MAPPO training process.

It includes:
1.  RewardRolloutBuffer: An off-policy replay buffer designed to store full episode
	trajectories for training the credit assignment models (e.g., TAR², AREL, STAS).
2.  RolloutBuffer: An on-policy buffer that stores the most recent batch of trajectories
	for the MAPPO actor and critic updates. It includes logic for calculating
	Generalized Advantage Estimation (GAE) targets.
3.  Helper functions for tensor operations that are robust to NaN values.
"""
import torch
import numpy as np

def torch_nanmax(tensor, dim=None, keepdim=False):
	"""PyTorch equivalent of np.nanmax"""
	min_value = torch.finfo(tensor.dtype).min
	output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)
	return output

def torch_nanmin(tensor, dim=None, keepdim=False):
	"""PyTorch equivalent of np.nanmin"""
	max_value = torch.finfo(tensor.dtype).max
	output = tensor.nan_to_num(max_value).min(dim=dim, keepdim=keepdim)
	return output

class RewardRolloutBuffer:
	"""
	An off-policy replay buffer for storing full episode trajectories.

	This buffer is used to train the credit assignment models, which are learned
	off-policy from a larger history of past experiences. It stores all the
	necessary information for the sequence-to-sequence reward models.
	"""
	def __init__(self, environment, capacity, max_episode_len, num_agents, num_enemies,
				 ally_obs_shape, enemy_obs_shape, local_obs_shape, rnn_num_layers_actor,
				 actor_hidden_state, action_shape, device):
		self.environment = environment
		self.capacity = capacity
		self.length = 0
		self.episode = 0
		self.t = 0
		self.max_episode_len = max_episode_len
		self.num_agents = num_agents
		self.action_shape = action_shape

		# Initialize numpy arrays to store trajectory data
		self.buffer = dict()
		if "StarCraft" in self.environment:
			self.buffer['ally_obs'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents, ally_obs_shape), dtype=np.float32)
			self.buffer['enemy_obs'] = np.zeros((self.capacity, self.max_episode_len, num_enemies, enemy_obs_shape), dtype=np.float32)
		elif "GFootball" in self.environment:
			self.buffer['ally_obs'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents, ally_obs_shape), dtype=np.float32)
			
		self.buffer['actions'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents), dtype=np.float32)
		self.buffer['reward'] = np.zeros((self.capacity, self.max_episode_len), dtype=np.float32)
		self.buffer['done'] = np.ones((self.capacity, self.max_episode_len), dtype=np.float32)
		self.buffer['indiv_dones'] = np.ones((self.capacity, self.max_episode_len, self.num_agents), dtype=np.float32)
		self.buffer['local_obs'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents, local_obs_shape))
		self.buffer['hidden_state_actor'] = np.zeros((self.capacity, self.max_episode_len, rnn_num_layers_actor, self.num_agents, actor_hidden_state))
		self.buffer['action_masks'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents, self.action_shape))
		self.episode_len = np.zeros(self.capacity)

	def push(self, ally_obs, enemy_obs, local_obs, actions, action_masks, hidden_state_actor, logprobs, reward, done, indiv_dones):
		"""Adds a single timestep of experience to the buffer."""
		if "StarCraft" in self.environment:
			self.buffer['ally_obs'][self.episode][self.t] = ally_obs
			self.buffer['enemy_obs'][self.episode][self.t] = enemy_obs
		elif "GFootball" in self.environment:
			self.buffer['ally_obs'][self.episode][self.t] = ally_obs
		self.buffer['local_obs'][self.episode][self.t] = local_obs
		self.buffer['actions'][self.episode][self.t] = actions
		self.buffer['action_masks'][self.episode][self.t] = action_masks
		self.buffer['hidden_state_actor'][self.episode][self.t] = hidden_state_actor
		self.buffer['reward'][self.episode][self.t] = reward
		self.buffer['done'][self.episode][self.t] = done
		self.buffer['indiv_dones'][self.episode][self.t] = indiv_dones
		self.t += 1

	def end_episode(self):
		"""Marks the end of an episode and updates buffer pointers."""
		self.episode_len[self.episode] = self.t
		if self.length < self.capacity:
			self.length += 1
		self.episode = (self.episode + 1) % self.capacity
		self.t = 0
	
	def sample_reward_model(self, num_episodes):
		"""
		Samples a batch of full episodes for training the reward model.

		Args:
			num_episodes (int): The number of full episodes to sample.

		Returns:
			tuple: A tuple of numpy arrays containing the sampled trajectory data.
		"""
		assert num_episodes <= self.length
		batch_indices = np.random.choice(self.length, size=num_episodes, replace=False)
		
		# Extract data for the sampled episodes
		if "StarCraft" in self.environment:
			ally_obs_batch = np.take(self.buffer['ally_obs'], batch_indices, axis=0)
			enemy_obs_batch = np.take(self.buffer['enemy_obs'], batch_indices, axis=0)
		elif "GFootball" in self.environment:
			ally_obs_batch = np.take(self.buffer['ally_obs'], batch_indices, axis=0)
			enemy_obs_batch = None # GFootball doesn't use separate enemy obs here
			
		local_obs_batch = np.take(self.buffer['local_obs'], batch_indices, axis=0)
		actions_batch = np.take(self.buffer['actions'], batch_indices, axis=0)
		action_masks_batch = np.take(self.buffer['action_masks'], batch_indices, axis=0)
		hidden_state_actor_batch = np.take(self.buffer['hidden_state_actor'], batch_indices, axis=0)
		reward_batch = np.take(self.buffer['reward'], batch_indices, axis=0)
		mask_batch = 1 - np.take(self.buffer['done'], batch_indices, axis=0)
		agent_masks_batch = 1 - np.take(self.buffer['indiv_dones'], batch_indices, axis=0)
		episode_len_batch = np.take(self.episode_len, batch_indices, axis=0)

		# Construct the "last_actions" tensor for the policy/reward models
		first_last_actions = np.zeros((num_episodes, 1, self.num_agents), dtype=int) + self.action_shape
		last_actions_batch = np.concatenate((first_last_actions, actions_batch[:, :-1, :]), axis=1)

		if "StarCraft" in self.environment:
			return ally_obs_batch, enemy_obs_batch, local_obs_batch, actions_batch, last_actions_batch, action_masks_batch, hidden_state_actor_batch, None, reward_batch, mask_batch, agent_masks_batch, episode_len_batch
		elif "GFootball" in self.environment:
			return ally_obs_batch, local_obs_batch, actions_batch, last_actions_batch, action_masks_batch, hidden_state_actor_batch, None, reward_batch, mask_batch, agent_masks_batch, episode_len_batch

	def __len__(self):
		return self.length

class RolloutBuffer:
	"""
	An on-policy buffer for storing trajectories for a single MAPPO update.

	This buffer collects a fixed number of episodes and then calculates the advantages
	and returns (value targets) required for the PPO loss. After the update, the
	buffer is cleared.
	"""
	def __init__(self, environment, experiment_type, num_episodes, max_time_steps, num_agents, num_enemies,
				 ally_state_shape, enemy_state_shape, local_obs_shape, global_obs_shape,
				 rnn_num_layers_actor, actor_hidden_state, rnn_num_layers_v, v_hidden_state,
				 num_actions, data_chunk_length, norm_returns_v, clamp_rewards,
				 clamp_rewards_value_min, clamp_rewards_value_max, gae_lambda, gamma):
		
		# Store all configuration parameters
		self.environment = environment
		self.experiment_type = experiment_type
		self.num_episodes = num_episodes
		self.max_time_steps = max_time_steps
		self.num_agents = num_agents
		self.num_enemies = num_enemies if "StarCraft" in self.environment else None
		self.ally_state_shape = ally_state_shape if "StarCraft" in self.environment else None
		self.enemy_state_shape = enemy_state_shape if "StarCraft" in self.environment else None
		self.local_obs_shape = local_obs_shape
		self.global_obs_shape = global_obs_shape if "GFootball" in self.environment else None
		self.rnn_num_layers_actor = rnn_num_layers_actor
		self.actor_hidden_state = actor_hidden_state
		self.rnn_num_layers_v = rnn_num_layers_v
		self.v_hidden_state = v_hidden_state
		self.num_actions = num_actions
		self.data_chunk_length = data_chunk_length
		self.norm_returns_v = norm_returns_v
		self.clamp_rewards = clamp_rewards
		self.clamp_rewards_value_min = clamp_rewards_value_min
		self.clamp_rewards_value_max = clamp_rewards_value_max
		self.gae_lambda = gae_lambda
		self.gamma = gamma
		self.episode_num = 0
		self.time_step = 0

		# Initialize numpy arrays for storing on-policy data
		if "StarCraft" in self.environment:
			self.ally_states = np.zeros((num_episodes, max_time_steps, num_agents, ally_state_shape))
			self.enemy_states = np.zeros((num_episodes, max_time_steps, num_enemies, enemy_state_shape))
		elif "GFootball" in self.environment:
			self.global_obs = np.zeros((num_episodes, max_time_steps, num_agents, global_obs_shape))
		
		self.hidden_state_v = np.zeros((num_episodes, max_time_steps, rnn_num_layers_v, num_agents, v_hidden_state))
		self.V_values = np.zeros((num_episodes, max_time_steps + 1, num_agents))
		self.local_obs = np.zeros((num_episodes, max_time_steps, num_agents, local_obs_shape))
		self.hidden_state_actor = np.zeros((num_episodes, max_time_steps, rnn_num_layers_actor, num_agents, actor_hidden_state))
		self.logprobs = np.zeros((num_episodes, max_time_steps, num_agents))
		self.actions = np.zeros((num_episodes, max_time_steps, num_agents), dtype=int)
		self.action_masks = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.rewards = np.zeros((num_episodes, max_time_steps, num_agents))
		self.indiv_dones = np.ones((num_episodes, max_time_steps + 1, num_agents))
		self.team_dones = np.ones((num_episodes, max_time_steps + 1))
		self.episode_length = np.zeros(num_episodes)

	def clear(self):
		"""Resets the buffer to be empty."""
		self.__init__(self.environment, self.experiment_type, self.num_episodes, self.max_time_steps, self.num_agents, 
					  self.num_enemies if "StarCraft" in self.environment else None,
					  self.ally_state_shape if "StarCraft" in self.environment else None,
					  self.enemy_state_shape if "StarCraft" in self.environment else None,
					  self.local_obs_shape,
					  self.global_obs_shape if "GFootball" in self.environment else None,
					  self.rnn_num_layers_actor, self.actor_hidden_state, self.rnn_num_layers_v, self.v_hidden_state,
					  self.num_actions, self.data_chunk_length, self.norm_returns_v, self.clamp_rewards,
					  self.clamp_rewards_value_min, self.clamp_rewards_value_max, self.gae_lambda, self.gamma)

	def push(self, ally_states, enemy_states, value, hidden_state_v, global_obs, local_obs, 
			 hidden_state_actor, logprobs, actions, action_masks, rewards, indiv_dones, team_dones):
		"""Adds a single timestep of experience to the on-policy buffer."""
		if "StarCraft" in self.environment:
			self.ally_states[self.episode_num][self.time_step] = ally_states
			self.enemy_states[self.episode_num][self.time_step] = enemy_states
		elif self.environment == "GFootball":
			self.global_obs[self.episode_num][self.time_step] = global_obs

		self.V_values[self.episode_num][self.time_step] = value
		self.hidden_state_v[self.episode_num][self.time_step] = hidden_state_v
		self.local_obs[self.episode_num][self.time_step] = local_obs
		self.hidden_state_actor[self.episode_num][self.time_step] = hidden_state_actor
		self.logprobs[self.episode_num][self.time_step] = logprobs
		self.actions[self.episode_num][self.time_step] = actions
		self.action_masks[self.episode_num][self.time_step] = action_masks
		self.rewards[self.episode_num][self.time_step] = rewards
		self.indiv_dones[self.episode_num][self.time_step] = indiv_dones
		self.team_dones[self.episode_num][self.time_step] = team_dones
		self.time_step = (self.time_step + 1) % self.max_time_steps

	def end_episode(self, t, value, indiv_dones, team_dones):
		"""Marks the end of an episode and stores the final value estimate."""
		self.V_values[self.episode_num][self.time_step] = value
		self.indiv_dones[self.episode_num][self.time_step] = indiv_dones
		self.team_dones[self.episode_num][self.time_step] = team_dones
		self.episode_length[self.episode_num] = t
		self.episode_num += 1
		self.time_step = 0

	def sample_recurrent_policy(self):
		"""
		Samples and prepares data for recurrent policy updates.

		This method reshapes the stored trajectories into overlapping chunks suitable
		for training recurrent neural networks (RNNs).
		"""
		data_chunks = self.max_time_steps // self.data_chunk_length
		# Randomly shuffle the order of episodes and chunks to decorrelate the data
		rand_batch = np.random.permutation(self.num_episodes)
		rand_time = np.random.permutation(data_chunks)

		first_last_actions = np.zeros((self.num_episodes, 1, self.num_agents), dtype=int) + self.num_actions

		if "StarCraft" in self.environment:
			ally_states = torch.from_numpy(self.ally_states).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.ally_state_shape)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.ally_state_shape)
			enemy_states = torch.from_numpy(self.enemy_states).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_enemies, self.enemy_state_shape)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_enemies, self.enemy_state_shape)
			global_obs = None
		elif "GFootball" in self.environment:
			global_obs = torch.from_numpy(self.global_obs).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.global_obs_shape)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.global_obs_shape)
			ally_states, enemy_states = None, None
		hidden_state_v = torch.from_numpy(self.hidden_state_v).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers_v, self.num_agents, self.v_hidden_state)[:, rand_time][rand_batch, :][:, :, 0, :, :, :].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers_v, -1, self.v_hidden_state)
		local_obs = torch.from_numpy(self.local_obs).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.local_obs_shape)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.local_obs_shape).reshape(-1, self.data_chunk_length, self.num_agents, self.local_obs_shape)
		hidden_state_actor = torch.from_numpy(self.hidden_state_actor).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers_actor, self.num_agents, self.actor_hidden_state)[:, rand_time][rand_batch, :][:, :, 0, :, :, :].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers_actor, -1, self.actor_hidden_state)
		logprobs = torch.from_numpy(self.logprobs).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		last_actions = torch.from_numpy(np.concatenate((first_last_actions, self.actions[:, :-1, :]), axis=1)).long().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		actions = torch.from_numpy(self.actions).long().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		action_masks = torch.from_numpy(self.action_masks).bool().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.num_actions)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.num_actions)
		agent_masks = 1-torch.from_numpy(self.indiv_dones[:, :-1]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		team_masks = 1-torch.from_numpy(self.team_dones[:, :-1]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length)

		values = torch.from_numpy(self.V_values[:, :-1, :]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		target_values = self.target_values.float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		advantage = self.advantage.float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)

		return ally_states, enemy_states, hidden_state_v, global_obs, local_obs, hidden_state_actor, logprobs, \
		last_actions, actions, action_masks, agent_masks, team_masks, values, target_values, advantage


	def calculate_targets(self, episode, v_value_norm=None):
		"""
		Calculates the advantage and value targets using GAE.

		This method should be called after a batch of episodes has been collected
		and before the update starts.
		"""
		masks = 1 - torch.from_numpy(self.indiv_dones[:, :-1, :])
		next_mask = 1 - torch.from_numpy(self.indiv_dones[:, -1, :])
		rewards = torch.from_numpy(self.rewards)
		values = torch.from_numpy(self.V_values[:, :-1, :]) * masks
		next_values = torch.from_numpy(self.V_values[:, -1, :]) * next_mask

		# De-normalize values if PopArt is used
		if self.norm_returns_v:
			values = v_value_norm.denormalize(values) * masks
			next_values = v_value_norm.denormalize(next_values) * next_mask

		if self.clamp_rewards:
			rewards = torch.clamp(rewards, min=self.clamp_rewards_value_min, max=self.clamp_rewards_value_max)

		# Calculate GAE targets
		self.target_values = self.gae_targets(rewards, values, next_values, masks, next_mask)
		self.advantage = (self.target_values - values).detach()

	def gae_targets(self, rewards, values, next_value, masks, next_mask):
		"""
		Computes Generalized Advantage Estimation (GAE).

		Reference: Schulman et al., 2015, "High-Dimensional Continuous Control Using Trust Region Policy Optimization"
		"""
		target_values = torch.zeros_like(rewards)
		advantage = 0

		# Iterate backwards through time to calculate advantages
		for t in reversed(range(rewards.shape[1])):
			td_error = rewards[:, t, :] + (self.gamma * next_value * next_mask) - values[:, t, :] * masks[:, t, :]
			advantage = td_error + self.gamma * self.gae_lambda * advantage * next_mask
			target_values[:, t, :] = advantage + values[:, t, :] * masks[:, t, :]
			next_value = values[:, t, :]
			next_mask = masks[:, t, :]
		return target_values * masks
