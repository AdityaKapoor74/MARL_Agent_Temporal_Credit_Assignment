import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np


def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
	"""
	Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
	The discretization converts the values greater than `threshold` to 1 and the rest to 0.
	The code is adapted from the official PyTorch implementation of gumbel_softmax:
	https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
	Args:
	  logits: `[..., num_features]` unnormalized log probabilities
	  tau: non-negative scalar temperature
	  hard: if ``True``, the returned samples will be discretized,
			but will be differentiated as if it is the soft sample in autograd
	 threshold: threshold for the discretization,
				values greater than this will be set to 1 and the rest to 0
	Returns:
	  Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
	  If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
	  be probability distributions.
	"""
	gumbels = (
		-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
	)  # ~Gumbel(0, 1)
	gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
	y_soft = gumbels.sigmoid()

	if hard:
		# Straight through.
		indices = (y_soft > threshold).nonzero(as_tuple=True)
		y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
		y_hard[indices[0], indices[1], indices[2], indices[3]] = 1.0
		ret = y_hard - y_soft.detach() + y_soft
	else:
		# Reparametrization trick.
		ret = y_soft
	return ret



class RewardRolloutBuffer:
	def __init__(
		self, 
		environment,
		capacity, 
		max_episode_len, 
		num_agents, 
		num_enemies,
		ally_obs_shape,
		enemy_obs_shape,
		local_obs_shape,
		common_information_obs_shape,
		rnn_num_layers_actor,
		actor_hidden_state,
		action_shape,
		device,
		):

		self.environment = environment
		self.capacity = capacity
		self.length = 0
		self.episode = 0
		self.t = 0
		self.max_episode_len = max_episode_len
		self.num_agents = num_agents
		self.num_enemies = num_enemies
		self.ally_obs_shape = ally_obs_shape
		self.enemy_obs_shape = enemy_obs_shape
		self.local_obs_shape = local_obs_shape
		self.common_information_obs_shape = common_information_obs_shape
		self.rnn_num_layers_actor = rnn_num_layers_actor
		self.actor_hidden_state = actor_hidden_state
		self.action_shape = action_shape
		self.device = device

		self.buffer = dict()
		if "StarCraft" in self.environment:
			self.buffer['ally_obs'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents, self.ally_obs_shape), dtype=np.float32)
			self.buffer['enemy_obs'] = np.zeros((self.capacity, self.max_episode_len, self.num_enemies, self.enemy_obs_shape), dtype=np.float32)
		elif "GFootball" in self.environment:
			self.buffer['ally_obs'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents, self.ally_obs_shape), dtype=np.float32)
			self.buffer['common_obs'] = np.zeros((self.capacity, self.max_episode_len, self.common_information_obs_shape))

		self.buffer['actions'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents), dtype=np.float32)
		self.buffer['logprobs'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents), dtype=np.float32)
		self.buffer['reward'] = np.zeros((self.capacity, self.max_episode_len), dtype=np.float32)
		self.buffer['done'] = np.ones((self.capacity, self.max_episode_len), dtype=np.float32)
		self.buffer['indiv_dones'] = np.ones((self.capacity, self.max_episode_len, self.num_agents), dtype=np.float32)

		self.buffer['local_obs'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents, self.local_obs_shape))
		self.buffer['hidden_state_actor'] = np.zeros((self.capacity, self.max_episode_len, self.rnn_num_layers_actor, self.num_agents, self.actor_hidden_state))
		self.buffer['action_masks'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents, self.action_shape))

		self.episode_len = np.zeros(self.capacity)

	# push once per step
	def push(self, ally_obs, enemy_obs, local_obs, common_obs, actions, action_masks, hidden_state_actor, logprobs, reward, done, indiv_dones):
		if "StarCraft" in self.environment:
			self.buffer['ally_obs'][self.episode][self.t] = ally_obs
			self.buffer['enemy_obs'][self.episode][self.t] = enemy_obs
		elif "GFootball" in self.environment:
			self.buffer['ally_obs'][self.episode][self.t] = ally_obs
			self.buffer['common_obs'][self.episode][self.t] = common_obs
		self.buffer['local_obs'][self.episode][self.t] = local_obs
		self.buffer['actions'][self.episode][self.t] = actions
		self.buffer['action_masks'][self.episode][self.t] = action_masks
		self.buffer['hidden_state_actor'][self.episode][self.t] = hidden_state_actor
		self.buffer['logprobs'][self.episode][self.t] = logprobs
		self.buffer['reward'][self.episode][self.t] = reward
		self.buffer['done'][self.episode][self.t] = done
		self.buffer['indiv_dones'][self.episode][self.t] = indiv_dones
		self.t += 1

	def end_episode(self):
		self.episode_len[self.episode] = self.t - 1
		if self.length < self.capacity:
			self.length += 1
		self.episode = (self.episode + 1) % self.capacity
		self.t = 0
	
	def sample_reward_model(self, num_episodes):
		assert num_episodes <= self.length
		batch_indices = np.random.choice(self.length, size=num_episodes, replace=False)
		if "StarCraft" in self.environment:
			ally_obs_batch = np.take(self.buffer['ally_obs'], batch_indices, axis=0)
			enemy_obs_batch = np.take(self.buffer['enemy_obs'], batch_indices, axis=0)
		elif "GFootball" in self.environment:
			ally_obs_batch = np.take(self.buffer['ally_obs'], batch_indices, axis=0)
			common_obs_batch = np.take(self.buffer['common_obs'], batch_indices, axis=0)
		local_obs_batch = np.take(self.buffer['local_obs'], batch_indices, axis=0)
		actions_batch = np.take(self.buffer['actions'], batch_indices, axis=0)
		action_masks_batch = np.take(self.buffer['action_masks'], batch_indices, axis=0)
		hidden_state_actor_batch = np.take(self.buffer['hidden_state_actor'], batch_indices, axis=0)
		logprobs_batch = np.take(self.buffer['logprobs'], batch_indices, axis=0)
		reward_batch = np.take(self.buffer['reward'], batch_indices, axis=0)
		mask_batch = 1 - np.take(self.buffer['done'], batch_indices, axis=0)
		agent_masks_batch = 1 - np.take(self.buffer['indiv_dones'], batch_indices, axis=0)
		episode_len_batch = np.take(self.episode_len, batch_indices, axis=0)

		first_last_actions = np.zeros((num_episodes, 1, self.num_agents), dtype=int) + self.action_shape
		last_actions_batch = np.concatenate((first_last_actions, actions_batch[:, :-1, :]), axis=1)

		if "StarCraft" in self.environment:
			return ally_obs_batch, enemy_obs_batch, local_obs_batch, actions_batch, last_actions_batch, action_masks_batch, hidden_state_actor_batch, logprobs_batch, reward_batch, mask_batch, agent_masks_batch, episode_len_batch
		elif "GFootball" in self.environment:
			return ally_obs_batch, local_obs_batch, common_obs_batch, actions_batch, last_actions_batch, action_masks_batch, hidden_state_actor_batch, logprobs_batch, reward_batch, mask_batch, agent_masks_batch, episode_len_batch


	def __len__(self):
		return self.length



class RewardRolloutBufferShared(RewardRolloutBuffer):
	def __init__(
		self, 
		num_workers, 
		environment,
		capacity, 
		max_episode_len, 
		num_agents, 
		num_enemies,
		ally_obs_shape,
		enemy_obs_shape,
		local_obs_shape,
		common_information_obs_shape,
		rnn_num_layers_actor,
		actor_hidden_state,
		action_shape,
		device,
		):

		super(RewardRolloutBufferShared, self).__init__(
			environment,
			capacity, 
			max_episode_len, 
			num_agents, 
			num_enemies,
			ally_obs_shape,
			enemy_obs_shape,
			local_obs_shape,
			common_information_obs_shape,
			rnn_num_layers_actor,
			actor_hidden_state,
			action_shape,
			device,
			)

		self.num_workers = num_workers
		self.episodes_completely_filled = np.zeros(self.capacity, dtype=int)  # this will be used to identify the episodes in the buffer that are completely filled. 
		self.worker_episode_counter = np.arange(self.num_workers)
		self.time_steps = np.zeros(self.num_workers, dtype=int)

		assert self.num_workers < self.capacity

	@property
	def next_episode_index_to_fill(self):
		return np.max(self.worker_episode_counter) + 1
	
	@property
	def episodes_filled(self):
		return min(np.min(self.worker_episode_counter), self.capacity)


	def clear(self):
		super().clear()
		self.episodes_completely_filled = np.zeros(self.capacity, dtype=int)
		self.worker_episode_counter = np.arange(self.num_workers)
		self.time_steps = np.zeros(self.num_workers, dtype=int)


	def push(
		self,
		ally_obs, 
		enemy_obs, 
		local_obs, 
		common_obs,
		actions, 
		action_masks, 
		hidden_state_actor, 
		logprobs, 
		reward, 
		done, 
		indiv_dones,
		masks=None,
		):

		# print("From reward push buffer")
		# print(f"Episode_counter: {self.worker_episode_counter}")
		# print(f"Timesteps: {self.time_steps}")

		if self.environment == "StarCraft":
			assert ally_obs.shape[0] == self.num_workers
			assert enemy_obs.shape[0] == self.num_workers
		elif self.environment == "GFootball":
			assert ally_obs.shape[0] == self.num_workers
			assert common_obs.shape[0] == self.num_workers
		assert local_obs.shape[0] == self.num_workers
		assert actions.shape[0] == self.num_workers
		assert action_masks.shape[0] == self.num_workers
		assert hidden_state_actor.shape[0] == self.num_workers
		assert logprobs.shape[0] == self.num_workers
		assert reward.shape[0] == self.num_workers
		assert done.shape[0] == self.num_workers
		assert indiv_dones.shape[0] == self.num_workers

		for worker_index in range(self.num_workers):
			if type(masks) == np.ndarray:
				if masks[worker_index]:
					continue
			episode_num = self.worker_episode_counter[worker_index] % self.capacity
			self.episodes_completely_filled[episode_num] = 0
			time_step = self.time_steps[worker_index]

			if "StarCraft" in self.environment:
				self.buffer['ally_obs'][episode_num][time_step] = ally_obs[worker_index]
				self.buffer['enemy_obs'][episode_num][time_step] = enemy_obs[worker_index]
			elif "GFootball" in self.environment:
				self.buffer['ally_obs'][episode_num][time_step] = ally_obs[worker_index]
				self.buffer['common_obs'][episode_num][time_step] = common_obs[worker_index]
			self.buffer['local_obs'][episode_num][time_step] = local_obs[worker_index]
			self.buffer['actions'][episode_num][time_step] = actions[worker_index]
			self.buffer['action_masks'][episode_num][time_step] = action_masks[worker_index]
			self.buffer['hidden_state_actor'][episode_num][time_step] = hidden_state_actor[worker_index]
			self.buffer['logprobs'][episode_num][time_step] = logprobs[worker_index]
			self.buffer['reward'][episode_num][time_step] = reward[worker_index]
			self.buffer['done'][episode_num][time_step] = done[worker_index]
			self.buffer['indiv_dones'][episode_num][time_step] = indiv_dones[worker_index]

			if self.time_steps[worker_index] < self.max_episode_len-1:
				self.time_steps[worker_index] += 1
		# print("")


	def end_episode(self, worker_indices):
		for i, worker_index in enumerate(worker_indices):
			episode_num = self.worker_episode_counter[worker_index] % self.capacity
			timestep = self.time_steps[worker_index]

			self.episodes_completely_filled[episode_num] = 1

			self.worker_episode_counter[worker_index] = self.next_episode_index_to_fill

			self.episode_len[episode_num] = timestep

			self.time_steps[worker_index] = 0

			# print("From episode end reward buffer")
			# print(f"Ending episode for worker {worker_index}")
			# print(f"Episodes {self.worker_episode_counter}")
			# print(f"Time Steps: {self.time_steps}")
			# print("")


	def sample_reward_model(self, num_episodes):
		indices = np.where(self.episodes_completely_filled == 1)[0]
		assert indices.shape[0] >= num_episodes
		batch_indices = np.random.choice(indices, size=num_episodes, replace=False)
		
		if "StarCraft" in self.environment:
			ally_obs_batch = np.take(self.buffer['ally_obs'], batch_indices, axis=0)
			enemy_obs_batch = np.take(self.buffer['enemy_obs'], batch_indices, axis=0)
		elif "GFootball" in self.environment:
			ally_obs_batch = np.take(self.buffer['ally_obs'], batch_indices, axis=0)
			common_obs_batch = np.take(self.buffer['common_obs'], batch_indices, axis=0)
		local_obs_batch = np.take(self.buffer['local_obs'], batch_indices, axis=0)
		actions_batch = np.take(self.buffer['actions'], batch_indices, axis=0)
		action_masks_batch = np.take(self.buffer['action_masks'], batch_indices, axis=0)
		hidden_state_actor_batch = np.take(self.buffer['hidden_state_actor'], batch_indices, axis=0)
		logprobs_batch = np.take(self.buffer['logprobs'], batch_indices, axis=0)
		reward_batch = np.take(self.buffer['reward'], batch_indices, axis=0)
		mask_batch = 1 - np.take(self.buffer['done'], batch_indices, axis=0)
		agent_masks_batch = 1 - np.take(self.buffer['indiv_dones'], batch_indices, axis=0)
		episode_len_batch = np.take(self.episode_len, batch_indices, axis=0)

		# print(np.sum(agent_masks_batch, axis=-2))

		first_last_actions = np.zeros((num_episodes, 1, self.num_agents), dtype=int) + self.action_shape
		last_actions_batch = np.concatenate((first_last_actions, actions_batch[:, :-1, :]), axis=1)

		if "StarCraft" in self.environment:
			assert ally_obs_batch.shape == (num_episodes, self.max_episode_len, self.num_agents, self.ally_obs_shape)
			assert enemy_obs_batch.shape == (num_episodes, self.max_episode_len, self.num_enemies, self.enemy_obs_shape)
		elif "GFootball" in self.environment:
			assert ally_obs_batch.shape == (num_episodes, self.max_episode_len, self.num_agents, self.ally_obs_shape)
			assert common_obs_batch.shape == (num_episodes, self.max_episode_len, self.common_information_obs_shape)
		assert local_obs_batch.shape == (num_episodes, self.max_episode_len, self.num_agents, self.local_obs_shape)
		assert actions_batch.shape == (num_episodes, self.max_episode_len, self.num_agents)
		assert logprobs_batch.shape == (num_episodes, self.max_episode_len, self.num_agents)
		assert reward_batch.shape == (num_episodes, self.max_episode_len)
		assert mask_batch.shape == (num_episodes, self.max_episode_len)
		assert agent_masks_batch.shape == (num_episodes, self.max_episode_len, self.num_agents)
		assert hidden_state_actor_batch.shape == (num_episodes, self.max_episode_len, self.rnn_num_layers_actor, self.num_agents, self.actor_hidden_state)
		assert action_masks_batch.shape == (num_episodes, self.max_episode_len, self.num_agents, self.action_shape)
		assert episode_len_batch.shape == (num_episodes,)
		assert last_actions_batch.shape == (num_episodes, self.max_episode_len, self.num_agents)


		if "StarCraft" in self.environment:
			return ally_obs_batch, enemy_obs_batch, local_obs_batch, actions_batch, last_actions_batch, action_masks_batch, hidden_state_actor_batch, logprobs_batch, reward_batch, mask_batch, agent_masks_batch, episode_len_batch
		elif "GFootball" in self.environment:
			return ally_obs_batch, local_obs_batch, common_obs_batch, actions_batch, last_actions_batch, action_masks_batch, hidden_state_actor_batch, logprobs_batch, reward_batch, mask_batch, agent_masks_batch, episode_len_batch



class RolloutBuffer:
	def __init__(
		self, 
		environment,
		experiment_type,
		num_episodes, 
		max_time_steps, 
		num_agents, 
		num_enemies,
		ally_state_shape, 
		enemy_state_shape, 
		local_obs_shape, 
		global_obs_shape,
		common_information_obs_shape,
		rnn_num_layers_actor,
		actor_hidden_state,
		rnn_num_layers_v,
		v_hidden_state,
		num_actions, 
		data_chunk_length,
		norm_returns_v,
		clamp_rewards,
		clamp_rewards_value_min,
		clamp_rewards_value_max,
		target_calc_style,
		gae_lambda,
		n_steps,
		gamma,
		):
		self.environment = environment
		self.experiment_type = experiment_type
		self.num_episodes = num_episodes
		self.max_time_steps = max_time_steps
		self.num_agents = num_agents

		if "StarCraft" in self.environment:
			self.num_enemies = num_enemies
			self.ally_state_shape = ally_state_shape
			self.enemy_state_shape = enemy_state_shape
		elif self.environment == "GFootball":
			self.ally_state_shape = ally_state_shape
			self.global_obs_shape = global_obs_shape
			self.common_information_obs_shape = common_information_obs_shape

		self.local_obs_shape = local_obs_shape
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

		self.target_calc_style = target_calc_style
		self.gae_lambda = gae_lambda
		self.gamma = gamma
		self.n_steps = n_steps

		self.episode_num = 0
		self.time_step = 0

		if "StarCraft" in self.environment:
			self.ally_states = np.zeros((num_episodes, max_time_steps, num_agents, ally_state_shape))
			self.enemy_states = np.zeros((num_episodes, max_time_steps, num_enemies, enemy_state_shape))
		elif "GFootball" in self.environment:
			self.ally_states = np.zeros((num_episodes, max_time_steps, num_agents, ally_state_shape))
			self.global_obs = np.zeros((num_episodes, max_time_steps, num_agents, global_obs_shape))
			self.common_obs = np.zeros((num_episodes, max_time_steps, common_information_obs_shape))
		self.hidden_state_v = np.zeros((num_episodes, max_time_steps, rnn_num_layers_v, num_agents, v_hidden_state))
		self.V_values = np.zeros((num_episodes, max_time_steps+1, num_agents))
		self.local_obs = np.zeros((num_episodes, max_time_steps, num_agents, local_obs_shape))
		self.hidden_state_actor = np.zeros((num_episodes, max_time_steps, rnn_num_layers_actor, num_agents, actor_hidden_state))
		self.latent_state_actor = np.zeros((num_episodes, max_time_steps, num_agents, actor_hidden_state))
		self.logprobs = np.zeros((num_episodes, max_time_steps, num_agents))
		self.actions = np.zeros((num_episodes, max_time_steps, num_agents), dtype=int)
		self.one_hot_actions = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.action_masks = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.rewards = np.zeros((num_episodes, max_time_steps, num_agents))
		self.indiv_dones = np.ones((num_episodes, max_time_steps+1, num_agents))
		self.team_dones = np.ones((num_episodes, max_time_steps+1))

		self.episode_length = np.zeros(num_episodes)

		self.action_prediction = None
	

	def clear(self):

		if "StarCraft" in self.environment:
			self.ally_states = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.ally_state_shape))
			self.enemy_states = np.zeros((self.num_episodes, self.max_time_steps, self.num_enemies, self.enemy_state_shape))
		elif "GFootball" in self.environment:
			self.ally_states = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.ally_state_shape))
			self.global_obs = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.global_obs_shape))
			self.common_obs = np.zeros((self.num_episodes, self.max_time_steps, self.common_information_obs_shape))
		self.hidden_state_v = np.zeros((self.num_episodes, self.max_time_steps, self.rnn_num_layers_v, self.num_agents, self.v_hidden_state))
		self.V_values = np.zeros((self.num_episodes, self.max_time_steps+1, self.num_agents))
		self.local_obs = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.local_obs_shape))
		self.hidden_state_actor = np.zeros((self.num_episodes, self.max_time_steps, self.rnn_num_layers_actor, self.num_agents, self.actor_hidden_state))
		self.latent_state_actor = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.actor_hidden_state))
		self.logprobs = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents))
		self.actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents), dtype=int)
		self.one_hot_actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.action_masks = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.rewards = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents))
		self.indiv_dones = np.ones((self.num_episodes, self.max_time_steps+1, self.num_agents))
		self.team_dones = np.ones((self.num_episodes, self.max_time_steps+1))

		self.episode_length = np.zeros(self.num_episodes)

		self.time_step = 0
		self.episode_num = 0

		self.action_prediction = None


	def push(
		self, 
		ally_states, 
		enemy_states, 
		value, 
		hidden_state_v,
		global_obs,
		local_obs, 
		common_obs,
		latent_state_actor,
		hidden_state_actor, 
		logprobs, 
		actions, 
		one_hot_actions,
		action_masks, 
		rewards, 
		indiv_dones,
		team_dones,
		):

		
		if "StarCraft" in self.environment:
			self.ally_states[self.episode_num][self.time_step] = ally_states
			self.enemy_states[self.episode_num][self.time_step] = enemy_states
		elif self.environment == "GFootball":
			self.ally_states[self.episode_num][self.time_step] = ally_states
			self.global_obs[self.episode_num][self.time_step] = global_obs
			self.common_obs[self.episode_num][self.time_step] = common_obs

		self.V_values[self.episode_num][self.time_step] = value
		self.hidden_state_v[self.episode_num][self.time_step] = hidden_state_v
		
		self.local_obs[self.episode_num][self.time_step] = local_obs
		self.latent_state_actor[self.episode_num][self.time_step] = latent_state_actor
		self.hidden_state_actor[self.episode_num][self.time_step] = hidden_state_actor
		self.logprobs[self.episode_num][self.time_step] = logprobs
		self.actions[self.episode_num][self.time_step] = actions
		self.one_hot_actions[self.episode_num][self.time_step] = one_hot_actions
		self.action_masks[self.episode_num][self.time_step] = action_masks
		self.rewards[self.episode_num][self.time_step] = rewards
		self.indiv_dones[self.episode_num][self.time_step] = indiv_dones
		self.team_dones[self.episode_num][self.time_step] = team_dones

		if self.time_step < self.max_time_steps-1:
			self.time_step += 1


	def end_episode(
		self, 
		t, 
		value, 
		indiv_dones,
		team_dones
		):
		self.V_values[self.episode_num][self.time_step] = value
		self.indiv_dones[self.episode_num][self.time_step] = indiv_dones
		self.team_dones[self.episode_num][self.time_step] = team_dones

		self.episode_length[self.episode_num] = t
		self.episode_num += 1
		self.time_step = 0


	def sample_recurrent_policy(self):

		data_chunks = self.max_time_steps // self.data_chunk_length
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


	def sample_finetune_reward_model(self):
		if "StarCraft" in self.environment:
			return self.ally_states, self.enemy_states, self.local_obs, self.actions, np.concatenate((np.zeros((self.num_episodes, 1, self.num_agents), dtype=int) + self.num_actions, self.actions[:, :-1, :]), axis=1), \
			self.action_masks, self.hidden_state_actor, self.logprobs, self.rewards[:, 0], 1-self.team_dones[:, :-1], \
			1-self.indiv_dones[:, :-1, :], self.episode_length
		elif "GFootball" in self.environment:
			return self.ally_states, self.local_obs, self.common_obs, self.actions, np.concatenate((np.zeros((self.num_episodes, 1, self.num_agents), dtype=int) + self.num_actions, self.actions[:, :-1, :]), axis=1), \
			self.action_masks, self.hidden_state_actor, self.logprobs, self.rewards[:, 0], 1-self.team_dones[:, :-1], \
			1-self.indiv_dones[:, :-1, :], self.episode_length


	def calculate_targets(self, episode, v_value_norm=None):
		
		masks = 1 - torch.from_numpy(self.indiv_dones[:, :-1, :])
		next_mask = 1 - torch.from_numpy(self.indiv_dones[:, -1, :])

		rewards = torch.from_numpy(self.rewards)

		values = torch.from_numpy(self.V_values[:, :-1, :]) * masks
		next_values = torch.from_numpy(self.V_values[:, -1, :]) * next_mask

		if self.norm_returns_v:
			values_shape = values.shape
			values = v_value_norm.denormalize(values.view(-1)).view(values_shape) * masks.view(values_shape)

			next_values_shape = next_values.shape
			next_values = v_value_norm.denormalize(next_values.view(-1)).view(next_values_shape) * next_mask.view(next_values_shape)

		if self.clamp_rewards:
			rewards = torch.clamp(rewards, min=self.clamp_rewards_value_min, max=self.clamp_rewards_value_max)

		# TARGET CALC
		if self.target_calc_style == "GAE":
			target_values = self.gae_targets(rewards, values, next_values, masks, next_mask)
		elif self.target_calc_style == "N_steps":
			target_values = self.nstep_returns(rewards, values, next_values, masks, next_mask)			

		self.advantage = (target_values - values).detach()
		self.target_values = target_values


	def gae_targets(self, rewards, values, next_value, masks, next_mask):
		
		target_values = rewards.new_zeros(*rewards.shape)
		advantage = 0

		for t in reversed(range(0, rewards.shape[1])):

			td_error = rewards[:,t,:] + (self.gamma * next_value * next_mask) - values.data[:,t,:] * masks[:, t, :]
			advantage = td_error + self.gamma * self.gae_lambda * advantage * next_mask
			
			target_values[:, t, :] = advantage + values.data[:, t, :] * masks[:, t, :]
			
			next_value = values.data[:, t, :]
			next_mask = masks[:, t, :]

		return target_values * masks


	def nstep_returns(self, rewards, values, next_value, masks, next_mask):
		
		nstep_values = torch.zeros_like(values)
		for t_start in range(rewards.size(1)):
			nstep_return_t = torch.zeros_like(values[:, 0])
			for step in range(self.n_steps + 1):
				t = t_start + step
				if t >= rewards.size(1):
					break
				elif step == self.n_steps:
					nstep_return_t += self.gamma ** (step) * values[:, t] * masks[:, t]
				elif t == rewards.size(1) - 1: # and self.args.add_value_last_step:
					nstep_return_t += self.gamma ** (step) * rewards[:, t] * masks[:, t]
					nstep_return_t += self.gamma ** (step + 1) * next_value * next_mask 
				else:
					nstep_return_t += self.gamma ** (step) * rewards[:, t] * masks[:, t]
			nstep_values[:, t_start, :] = nstep_return_t
		
		return nstep_values


	def calculate_targets_hindsight(self, episode, v_value_norm=None):

		masks = 1 - torch.from_numpy(self.indiv_dones[:, :-1, :])
		next_mask = 1 - torch.from_numpy(self.indiv_dones[:, -1, :])

		rewards = torch.from_numpy(self.rewards)

		values = torch.from_numpy(self.V_values[:, :-1, :]) * masks
		next_values = torch.from_numpy(self.V_values[:, -1, :]) * next_mask

		b, t, n_a = self.actions.shape
		# print("action_prediction_dist_output")
		# print(torch.from_numpy(self.action_prediction).permute(0, 2, 3, 1, 4)[0, 0, :, 0])
		action_prediction_dist = F.softmax(torch.from_numpy(self.action_prediction).permute(0, 2, 3, 1, 4), dim=-1) # b x t x t x n_a x n_actions
		# print("action_prediction_dist", action_prediction_dist.shape)
		action_prediction_probs = Categorical(action_prediction_dist)
		actions_batch = torch.from_numpy(self.actions).unsqueeze(-2).repeat(1, 1, t, 1) # b x t x t x n_a
		# print("actions_batch", actions_batch.shape)
		# print(actions_batch[0, 0, :, 0])
		# print(action_prediction_dist[0, 0, :, 0, :])
		# print(actions_batch[0, 0, :, 0])
		action_prediction_logprobs = action_prediction_inverse_dynamic_probs.log_prob(actions_batch) # b x t x t x n_a
		# print("action_prediction_logprobs", action_prediction_logprobs.shape)
		# print(action_prediction_logprobs[0, 0, :, 0])
		action_logprobs = torch.from_numpy(self.logprobs).unsqueeze(-2).repeat(1, 1, t, 1) # b x t x t x n_a
		# print("action_logprobs", action_logprobs.shape)
		# print(action_logprobs[0, 0, :, 0])
		upper_triangular_mask = torch.triu(torch.ones(b*n_a, t, t)).reshape(b, n_a, t, t).permute(0, 2, 3, 1)
		action_importance_sampling = torch.exp(action_prediction_logprobs-action_logprobs).clamp(0, 2.0) # b x t x t x n_a
		# print("action_importance_sampling", action_importance_sampling.shape)
		# make diagonal elements 1
		action_importance_sampling = action_importance_sampling.permute(0, 3, 1, 2).reshape(b*n_a, t, t)
		action_probs = torch.exp(torch.from_numpy(self.logprobs))
		for i in range(t):
			action_importance_sampling[:, i, i] = action_probs[:, i, :].reshape(-1) # torch.ones(b*n_a)
		action_importance_sampling = action_importance_sampling.reshape(b, n_a, t, t).permute(0, 2, 3, 1) * upper_triangular_mask * masks.unsqueeze(1)
		# action_importance_sampling = torch.prod(action_importance_sampling, dim=-1, keepdim=True)

		print("masks")
		print(masks[0, :, 0])
		print("actions")
		print(actions_batch[0, :, 0, 0])
		print("action_prediction_logprobs")
		print(action_prediction_logprobs[0, 0, :, 0])
		print("action_logprobs")
		print(action_logprobs[0, :, 0, 0])
		print("action_importance_sampling")
		print(action_importance_sampling[0, 0, :, 0])
		print(action_importance_sampling[0, 1, :, 0])


		if self.norm_returns_v:
			values_shape = values.shape
			values = v_value_norm.denormalize(values.view(-1)).view(values_shape) * masks.view(values_shape)

			next_values_shape = next_values.shape
			next_values = v_value_norm.denormalize(next_values.view(-1)).view(next_values_shape) * next_mask.view(next_values_shape)

		if self.clamp_rewards:
			rewards = torch.clamp(rewards, min=self.clamp_rewards_value_min, max=self.clamp_rewards_value_max)

		# TARGET CALC
		if self.target_calc_style == "GAE":
			target_values = self.gae_targets(rewards, values, next_values, masks, next_mask)
			advantage = self.gae_targets_hindsight(rewards.unsqueeze(1), values.unsqueeze(1), next_values.unsqueeze(1), masks.unsqueeze(1), next_mask.unsqueeze(1), action_importance_sampling)
		
		self.advantage = advantage.detach()
		self.target_values = target_values


	def gae_targets_hindsight(self, rewards, values, next_value, masks, next_mask, action_importance_sampling):

		# print(action_importance_sampling)

		b, _, t_, n_a = rewards.shape
		# target_values = rewards.new_zeros(*rewards.shape).repeat(1, t_, 1, 1)
		advantages = rewards.new_zeros(*rewards.shape).repeat(1, t_, 1, 1)
		advantage = 0
		next_action_importance_sampling = torch.ones(b, t_, n_a)
		upper_triangular_mask = torch.triu(torch.ones(b*n_a, t_, t_)).reshape(b, n_a, t_, t_).permute(0, 2, 3, 1)

		# print("-------------------------------------- ACTION IMPORTANCE SAMPLING --------------------------------------")
		# print(action_importance_sampling[0,:,:,0])

		for t in reversed(range(0, rewards.shape[2])):

			td_error = action_importance_sampling[:,:,t,:] * rewards[:,:,t,:] + (self.gamma * next_action_importance_sampling * next_value * next_mask) - values.data[:,:,t,:] * masks[:,:,t,:]
			# td_error = rewards[:,:,t,:] + (self.gamma * next_value * next_mask) - values.data[:,:,t,:] * masks[:,:,t,:]
			advantage = td_error + self.gamma * self.gae_lambda * advantage * next_mask
			
			# target_values[:,:,t,:] = (advantage + action_importance_sampling[:,:,t,:] * values.data[:,:,t,:] * masks[:,:,t,:]) * upper_triangular_mask[:,:,t,:]
			advantages[:,:,t,:] = advantage

			next_value = values.data[:,:,t,:]
			next_mask = masks[:,:,t,:]
			next_action_importance_sampling = action_importance_sampling[:,:,t,:]

		# upper_triangular_mask = torch.triu(torch.ones(b*n_a, t, t)).reshape(b, n_a, t, t).permute(0, 2, 3, 1)
		# return advantages * masks * upper_triangular_mask

		# extract advantages
		# advantages = (target_values - upper_triangular_mask*values).detach()
		advantages = torch.diagonal(advantages.permute(0, 3, 1, 2).reshape(-1, t_, t_), offset=0, dim1=-2, dim2=-1).reshape(b, n_a, t_).permute(0, 2, 1)

		print("-------------------------------------- ADVANTAGES --------------------------------------")
		print(advantages[0, :, 0])

		return advantages * masks.squeeze(1)
		



class RolloutBufferShared(RolloutBuffer):
	def __init__(
		self, 
		num_workers,
		environment,
		experiment_type,
		num_episodes, 
		max_time_steps, 
		num_agents, 
		num_enemies,
		ally_state_shape, 
		enemy_state_shape, 
		local_obs_shape, 
		global_obs_shape,
		common_information_obs_shape,
		rnn_num_layers_actor,
		actor_hidden_state,
		rnn_num_layers_v,
		v_hidden_state,
		num_actions, 
		data_chunk_length,
		norm_returns_v,
		clamp_rewards,
		clamp_rewards_value_min,
		clamp_rewards_value_max,
		target_calc_style,
		gae_lambda,
		n_steps,
		gamma,
		):

		super(RolloutBufferShared, self).__init__(
			environment,
			experiment_type,
			num_episodes, 
			max_time_steps, 
			num_agents, 
			num_enemies,
			ally_state_shape, 
			enemy_state_shape, 
			local_obs_shape, 
			global_obs_shape,
			common_information_obs_shape,
			rnn_num_layers_actor,
			actor_hidden_state,
			rnn_num_layers_v,
			v_hidden_state,
			num_actions, 
			data_chunk_length,
			norm_returns_v,
			clamp_rewards,
			clamp_rewards_value_min,
			clamp_rewards_value_max,
			target_calc_style,
			gae_lambda,
			n_steps,
			gamma,
			)

		self.environment = environment
		self.num_workers = num_workers
		# counter for each rollout thread
		self.worker_episode_counter = np.arange(self.num_workers)
		self.time_steps = np.zeros(self.num_workers, dtype=int)

	@property
	def episodes_completely_filled(self):
		return np.min(self.worker_episode_counter)

	@property
	def next_episode_index_to_fill(self):
		return np.max(self.worker_episode_counter) + 1


	def clear(self):
		super().clear()
		self.worker_episode_counter = np.arange(self.num_workers)
		self.time_steps = np.zeros(self.num_workers, dtype=int)
	

	def push(
		self, 
		ally_states, 
		enemy_states, 
		value, 
		hidden_state_v,
		global_obs,
		local_obs, 
		common_obs,
		latent_state_actor,
		hidden_state_actor, 
		logprobs, 
		actions, 
		one_hot_actions,
		action_masks, 
		rewards, 
		indiv_dones,
		team_dones,
		worker_step_counter,
		masks=None,
		):

		if "StarCraft" in self.environment:
			assert ally_states.shape[0] == self.num_workers
			assert enemy_states.shape[0] == self.num_workers
		if self.environment == "GFootball":
			assert ally_states.shape[0] == self.num_workers
			assert global_obs.shape[0] == self.num_workers
			assert common_obs.shape[0] == self.num_workers
		assert value.shape[0] == self.num_workers
		assert hidden_state_v.shape[0] == self.num_workers
		assert local_obs.shape[0] == self.num_workers
		assert latent_state_actor.shape[0] == self.num_workers
		assert hidden_state_actor.shape[0] == self.num_workers
		assert logprobs.shape[0] == self.num_workers
		assert actions.shape[0] == self.num_workers
		assert one_hot_actions.shape[0] == self.num_workers
		assert action_masks.shape[0] == self.num_workers
		assert rewards.shape[0] == self.num_workers
		assert indiv_dones.shape[0] == self.num_workers 
		assert team_dones.shape[0] == self.num_workers 
		
		# print("From push buffer")
		# print(f"Episode_counter: {self.worker_episode_counter}")
		# print(f"Timesteps: {self.time_steps}")
		for worker_index in range(self.num_workers):
			if type(masks) == np.ndarray:  # the masks array indicates whether the current worker's data should be ignored
				if masks[worker_index]:
					# print(f"Skipping worker {worker_index} since it is masked.")
					continue
			episode_num = self.worker_episode_counter[worker_index]
			time_step = self.time_steps[worker_index]

			if episode_num >= self.num_episodes:
				# print(f"skipping worker {worker_index} since it has collected more than needed")
				# the workers that have collected all required episodes for this update should not store anything more
				continue

			# the below condition might hold only when running train_parallel_agent_async.py 
			if time_step == 0 and worker_step_counter[worker_index] != 1:
				assert masks == None
				# because of the above skip, after updation completes, it might be the case that the workers are somewhere in the middle of an ongoing episode
				# so we will just do nothing till that episode completes. After it completes, storing would resume.
				# print(f"skipping worker {worker_index} till it resets")
				continue
		
			if "StarCraft" in self.environment:
				self.ally_states[episode_num][time_step] = ally_states[worker_index]
				self.enemy_states[episode_num][time_step] = enemy_states[worker_index]
			elif self.environment == "GFootball":
				self.ally_states[episode_num][time_step] = ally_states[worker_index]
				self.common_obs[episode_num][time_step] = common_obs[worker_index]
				self.global_obs[episode_num][time_step] = global_obs[worker_index]

			self.V_values[episode_num][time_step] = value[worker_index]
			self.hidden_state_v[episode_num][time_step] = hidden_state_v[worker_index]
			
			self.local_obs[episode_num][time_step] = local_obs[worker_index]
			self.latent_state_actor[episode_num][time_step] = latent_state_actor[worker_index]
			self.hidden_state_actor[episode_num][time_step] = hidden_state_actor[worker_index]
			self.logprobs[episode_num][time_step] = logprobs[worker_index]
			self.actions[episode_num][time_step] = actions[worker_index]
			self.one_hot_actions[episode_num][time_step] = one_hot_actions[worker_index]
			self.action_masks[episode_num][time_step] = action_masks[worker_index]
			self.rewards[episode_num][time_step] = rewards[worker_index]
			self.indiv_dones[episode_num][time_step] = indiv_dones[worker_index]
			self.team_dones[episode_num][time_step] = team_dones[worker_index]

			# print(f"Filled for {worker_index}")
			if time_step < self.max_time_steps-1:
				self.time_steps[worker_index] += 1

		# print("")


	def end_episode(
		self, 
		t, 
		value, 
		indiv_dones,
		team_dones,
		worker_indices,
		):

		assert t.shape[0] == len(worker_indices)
		assert value.shape[0] == len(worker_indices)
		assert indiv_dones.shape[0] == len(worker_indices)
		assert team_dones.shape[0] == len(worker_indices)


		for i, worker_index in enumerate(worker_indices):
			episode_num = self.worker_episode_counter[worker_index]
			time_step = self.time_steps[worker_index]
			if time_step == 0:
				# Do nothing in case the worker has not stored anything
				continue

			self.V_values[episode_num][time_step] = value[i]
			self.team_dones[episode_num][time_step] = team_dones[i]
			self.indiv_dones[episode_num][time_step] = indiv_dones[i]

			self.episode_length[episode_num] = t[i]
			self.episode_num += 1
			self.time_steps[worker_index] = 0
			self.worker_episode_counter[worker_index] = self.next_episode_index_to_fill
			# print("From episode end buffer")
			# print(f"Ending episode for worker {worker_index}")
			# print(f"Episodes {self.worker_episode_counter}")
			# print(f"Time Steps: {self.time_steps}")
			# print("")