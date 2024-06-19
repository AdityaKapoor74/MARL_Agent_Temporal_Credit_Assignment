import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from model import RunningMeanStd


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
	# print(y_soft)

	if hard:
		# Straight through.
		indices = (y_soft > threshold).nonzero(as_tuple=True)
		y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
		y_hard[indices[0], indices[1], indices[2], indices[3]] = 1.0
		ret = y_hard - y_soft.detach() + y_soft
	else:
		# Reparametrization trick.
		ret = y_soft

	# print("GUMBEL SIGMOID")
	# print(ret)
	
	return ret



class RewardReplayMemory:
	def __init__(
		self, 
		experiment_type,
		capacity, 
		max_episode_len, 
		num_agents, 
		num_enemies,
		# reward_model_obs_shape,
		ally_obs_shape,
		enemy_obs_shape,
		action_shape,
		device,
		):

		self.experiment_type = experiment_type
		self.capacity = capacity
		self.length = 0
		self.episode = 0
		self.t = 0
		self.max_episode_len = max_episode_len
		self.num_agents = num_agents
		self.num_enemies = num_enemies
		# self.reward_model_obs_shape = reward_model_obs_shape
		self.ally_obs_shape = ally_obs_shape
		self.enemy_obs_shape = enemy_obs_shape
		self.action_shape = action_shape
		self.device = device

		self.buffer = dict()
		# self.buffer['reward_model_obs'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents, self.reward_model_obs_shape), dtype=np.float32)
		self.buffer['ally_obs'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents, self.ally_obs_shape), dtype=np.float32)
		self.buffer['enemy_obs'] = np.zeros((self.capacity, self.max_episode_len, self.num_enemies, self.enemy_obs_shape), dtype=np.float32)
		self.buffer['actions'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents), dtype=np.float32)
		self.buffer['one_hot_actions'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents, self.action_shape), dtype=np.float32)
		self.buffer['reward'] = np.zeros((self.capacity, self.max_episode_len), dtype=np.float32)
		self.buffer['done'] = np.ones((self.capacity, self.max_episode_len), dtype=np.float32)
		self.buffer['indiv_dones'] = np.ones((self.capacity, self.max_episode_len, self.num_agents), dtype=np.float32)

		self.episode_len = np.zeros(self.capacity)

	# push once per step
	def push(self, ally_obs, enemy_obs, actions, one_hot_actions, reward, done, indiv_dones):
		# self.buffer['reward_model_obs'][self.episode][self.t] = reward_model_obs
		self.buffer['ally_obs'][self.episode][self.t] = ally_obs
		self.buffer['enemy_obs'][self.episode][self.t] = enemy_obs
		self.buffer['actions'][self.episode][self.t] = actions
		self.buffer['one_hot_actions'][self.episode][self.t] = one_hot_actions
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
		# reward_model_obs_batch = np.take(self.buffer['reward_model_obs'], batch_indices, axis=0)
		ally_obs_batch = np.take(self.buffer['ally_obs'], batch_indices, axis=0)
		enemy_obs_batch = np.take(self.buffer['enemy_obs'], batch_indices, axis=0)
		actions_batch = np.take(self.buffer['actions'], batch_indices, axis=0)
		one_hot_actions_batch = np.take(self.buffer['one_hot_actions'], batch_indices, axis=0)
		reward_batch = np.take(self.buffer['reward'], batch_indices, axis=0)
		mask_batch = 1 - np.take(self.buffer['done'], batch_indices, axis=0)
		agent_masks_batch = 1 - np.take(self.buffer['indiv_dones'], batch_indices, axis=0)
		episode_len_batch = np.take(self.episode_len, batch_indices, axis=0)

		return ally_obs_batch, enemy_obs_batch, actions_batch, one_hot_actions_batch, reward_batch, mask_batch, agent_masks_batch, episode_len_batch


	def __len__(self):
		return self.length



class RolloutBuffer:
	def __init__(
		self, 
		num_episodes, 
		max_time_steps, 
		num_agents, 
		num_enemies,
		obs_shape_critic, 
		obs_shape_actor, 
		rnn_num_layers_actor,
		actor_hidden_state,
		rnn_num_layers_q,
		q_hidden_state,
		# obs_shape_reward_model,
		ally_obs_shape,
		enemy_obs_shape,
		num_actions, 
		data_chunk_length,
		norm_returns_q,
		target_calc_style,
		td_lambda,
		gae_lambda,
		n_steps,
		gamma,
		# Q_PopArt,
		):
		self.num_episodes = num_episodes
		self.max_time_steps = max_time_steps
		self.num_agents = num_agents
		self.num_enemies = num_enemies
		self.obs_shape_critic = obs_shape_critic
		self.obs_shape_actor = obs_shape_actor
		self.rnn_num_layers_actor = rnn_num_layers_actor
		self.actor_hidden_state = actor_hidden_state
		self.rnn_num_layers_q = rnn_num_layers_q
		self.q_hidden_state = q_hidden_state
		# self.obs_shape_reward_model = obs_shape_reward_model
		self.ally_obs_shape = ally_obs_shape
		self.enemy_obs_shape = enemy_obs_shape
		self.num_actions = num_actions

		self.data_chunk_length = data_chunk_length
		self.norm_returns_q = norm_returns_q

		self.target_calc_style = target_calc_style
		self.td_lambda = td_lambda
		self.gae_lambda = gae_lambda
		self.gamma = gamma
		self.n_steps = n_steps
			
		# if self.norm_returns_q:
		# 	self.q_value_norm = Q_PopArt

		self.episode_num = 0
		self.time_step = 0

		self.states_critic = np.zeros((num_episodes, max_time_steps, num_agents, obs_shape_critic))
		self.hidden_state_q = np.zeros((num_episodes, max_time_steps, rnn_num_layers_q, num_agents, q_hidden_state))
		self.Q_values = np.zeros((num_episodes, max_time_steps+1, num_agents))
		self.states_actor = np.zeros((num_episodes, max_time_steps, num_agents, obs_shape_actor))
		self.hidden_state_actor = np.zeros((num_episodes, max_time_steps, rnn_num_layers_actor, num_agents, actor_hidden_state))
		# self.reward_model_obs = np.zeros((num_episodes, max_time_steps, num_agents, obs_shape_reward_model))
		self.ally_obs = np.zeros((num_episodes, max_time_steps, num_agents, ally_obs_shape))
		self.enemy_obs = np.zeros((num_episodes, max_time_steps, num_enemies, enemy_obs_shape))
		self.logprobs = np.zeros((num_episodes, max_time_steps, num_agents))
		self.actions = np.zeros((num_episodes, max_time_steps, num_agents), dtype=int)
		self.one_hot_actions = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.action_masks = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.rewards = np.zeros((num_episodes, max_time_steps, num_agents))
		self.agent_dones = np.ones((num_episodes, max_time_steps+1, num_agents))
		self.team_dones = np.ones((num_episodes, max_time_steps+1))

		self.episode_length = np.zeros(num_episodes)
	

	def clear(self):

		self.states_critic = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.obs_shape_critic))
		self.hidden_state_q = np.zeros((self.num_episodes, self.max_time_steps, self.rnn_num_layers_q, self.num_agents, self.q_hidden_state))
		self.Q_values = np.zeros((self.num_episodes, self.max_time_steps+1, self.num_agents))
		self.states_actor = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.obs_shape_actor))
		self.hidden_state_actor = np.zeros((self.num_episodes, self.max_time_steps, self.rnn_num_layers_actor, self.num_agents, self.actor_hidden_state))
		# self.reward_model_obs = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.obs_shape_reward_model))
		self.ally_obs = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.ally_obs_shape))
		self.enemy_obs = np.zeros((self.num_episodes, self.max_time_steps, self.num_enemies, self.enemy_obs_shape))
		self.logprobs = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents))
		self.actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents), dtype=int)
		self.one_hot_actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.action_masks = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.rewards = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents))
		self.agent_dones = np.ones((self.num_episodes, self.max_time_steps+1, self.num_agents))
		self.team_dones = np.ones((self.num_episodes, self.max_time_steps+1))

		self.episode_length = np.zeros(self.num_episodes)

		self.time_step = 0
		self.episode_num = 0


	def push(
		self, 
		state_critic, 
		q_value,
		hidden_state_q,
		state_actor, 
		hidden_state_actor, 
		logprobs, 
		actions,  
		one_hot_actions, 
		action_masks, 
		# reward_model_obs,
		ally_obs,
		enemy_obs,
		rewards, 
		agent_dones,
		team_dones,
		):

		self.states_critic[self.episode_num][self.time_step] = state_critic
		self.hidden_state_q[self.episode_num][self.time_step] = hidden_state_q
		self.Q_values[self.episode_num][self.time_step] = q_value
		self.states_actor[self.episode_num][self.time_step] = state_actor
		self.hidden_state_actor[self.episode_num][self.time_step] = hidden_state_actor
		self.logprobs[self.episode_num][self.time_step] = logprobs
		self.actions[self.episode_num][self.time_step] = actions
		self.one_hot_actions[self.episode_num][self.time_step] = one_hot_actions
		self.action_masks[self.episode_num][self.time_step] = action_masks
		# self.reward_model_obs[self.episode_num][self.time_step] = reward_model_obs
		self.ally_obs[self.episode_num][self.time_step] = ally_obs
		self.enemy_obs[self.episode_num][self.time_step] = enemy_obs
		self.rewards[self.episode_num][self.time_step] = rewards
		self.agent_dones[self.episode_num][self.time_step] = agent_dones
		self.team_dones[self.episode_num][self.time_step] = team_dones

		if self.time_step < self.max_time_steps-1:
			self.time_step += 1


	def end_episode(
		self, 
		t, 
		q_value, 
		agent_dones,
		team_dones
		):
		self.Q_values[self.episode_num][self.time_step+1] = q_value
		self.agent_dones[self.episode_num][self.time_step+1] = agent_dones
		self.team_dones[self.episode_num][self.time_step+1] = team_dones

		self.episode_length[self.episode_num] = t-1
		self.episode_num += 1
		self.time_step = 0


	def sample_recurrent_policy(self):

		data_chunks = self.max_time_steps // self.data_chunk_length
		rand_batch = np.random.permutation(self.num_episodes)
		rand_time = np.random.permutation(data_chunks)

		states_critic = torch.from_numpy(self.states_critic).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.obs_shape_critic)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.obs_shape_critic)
		hidden_state_q = torch.from_numpy(self.hidden_state_q).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers_q, self.num_agents, self.q_hidden_state)[:, rand_time][rand_batch, :][:, :, 0, :, :, :].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers_q, -1, self.q_hidden_state)
		states_actor = torch.from_numpy(self.states_actor).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.obs_shape_actor)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.obs_shape_actor).reshape(-1, self.data_chunk_length, self.num_agents, self.obs_shape_actor)
		hidden_state_actor = torch.from_numpy(self.hidden_state_actor).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers_actor, self.num_agents, self.actor_hidden_state)[:, rand_time][rand_batch, :][:, :, 0, :, :, :].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers_actor, -1, self.actor_hidden_state)
		logprobs = torch.from_numpy(self.logprobs).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		actions = torch.from_numpy(self.actions).long().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		one_hot_actions = torch.from_numpy(self.one_hot_actions).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.num_actions)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.num_actions)
		action_masks = torch.from_numpy(self.action_masks).bool().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.num_actions)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.num_actions)
		team_masks = 1-torch.from_numpy(self.team_dones[:, :-1]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length)
		agent_masks = 1-torch.from_numpy(self.agent_dones[:, :-1]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		
		# target_values, target_q_values, advantage = self.calculate_targets(advantage_type, episode, select_above_threshold)
		q_values = torch.from_numpy(self.Q_values[:, :-1, :]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		target_q_values = self.target_q_values.float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		advantage = self.advantage.float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		
		return states_critic, hidden_state_q, states_actor, hidden_state_actor, logprobs, \
		actions, one_hot_actions, action_masks, team_masks, agent_masks, q_values, target_q_values, advantage

	def calculate_targets(self, q_value_norm):
		
		agent_masks = 1 - torch.from_numpy(self.agent_dones[:, :-1, :])
		agent_next_mask = 1 - torch.from_numpy(self.agent_dones[:, -1, :])

		rewards = torch.from_numpy(self.rewards)

		q_values = torch.from_numpy(self.Q_values[:, :-1, :]) * agent_masks
		next_q_values = torch.from_numpy(self.Q_values[:, -1, :]) * agent_next_mask

		if self.norm_returns_q:
			q_values_shape = q_values.shape
			q_values = q_value_norm.denormalize(q_values.view(-1)).view(q_values_shape) * agent_masks.view(q_values_shape)

			next_q_values_shape = next_q_values.shape
			next_q_values = q_value_norm.denormalize(next_q_values.view(-1)).view(next_q_values_shape) * agent_next_mask.view(next_q_values_shape)

		if self.target_calc_style == "GAE":
			target_q_values = self.gae_targets(rewards, q_values, next_q_values, agent_masks, agent_next_mask)
		elif self.target_calc_style == "N_steps":
			target_q_values = self.nstep_returns(rewards, q_values, next_q_values, agent_masks, agent_next_mask)

		self.advantage = (target_q_values - q_values).detach()

		# if self.norm_returns_q:
		# 	targets_shape = target_q_values.shape
		# 	q_value_norm.update(target_q_values.view(-1), agent_masks.view(-1))
			
		# 	target_q_values = self.q_value_norm.normalize(target_q_values.view(-1)).view(targets_shape) * agent_masks.view(targets_shape)

		self.target_q_values = target_q_values.detach().cpu()


	def gae_targets(self, rewards, values, next_value, masks, next_mask):
		
		# advantages = rewards.new_zeros(*rewards.shape)
		target_values = rewards.new_zeros(*rewards.shape)
		advantage = 0

		for t in reversed(range(0, rewards.shape[1])):

			td_error = rewards[:,t,:] + (self.gamma * next_value * next_mask) - values.data[:,t,:] * masks[:, t, :]
			advantage = td_error + self.gamma * self.gae_lambda * advantage * next_mask
			
			target_values[:, t, :] = advantage + values.data[:, t, :] * masks[:, t, :]
			
			next_value = values.data[:, t, :]
			next_mask = masks[:, t, :]

			# advantages[:,t,:] = advantage

		# target_values_ = advantages + values

		# print(torch.sum(target_values - target_values_))

		return target_values*masks

	
	def build_td_lambda_targets(self, rewards, values, next_value, masks, next_mask):
		# Assumes  <target_qs > in B*T*A and <reward >, <terminated >  in B*T*A, <mask > in (at least) B*T-1*1
		# Initialise  last  lambda -return  for  not  terminated  episodes
		ret = target_qs.new_zeros(*target_qs.shape)
		ret = target_qs * (1-terminated[:, :-1]) # some episodes end early so we can't assume that by copying the last target_qs in ret would be good enough
		# ret[:, -1] = target_qs[:, -1] * (1 - (torch.sum(terminated, dim=1)>0).int())
		# Backwards  recursive  update  of the "forward  view"
		for t in range(ret.shape[1] - 2, -1, -1):
			ret[:, t] = self.td_lambda * self.gamma * ret[:, t + 1] + mask[:, t].unsqueeze(-1) \
						* (rewards[:, t] + (1 - self.td_lambda) * self.gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
		# Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
		# return ret[:, 0:-1]
		return ret


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