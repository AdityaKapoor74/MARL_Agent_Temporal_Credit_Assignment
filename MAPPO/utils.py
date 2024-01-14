import torch
from torch import Tensor
import numpy as np
import random

from model import ValueNorm, RunningMeanStd


class RolloutBuffer:
	def __init__(
		self, 
		num_episodes, 
		max_time_steps, 
		num_agents, 
		num_enemies,
		obs_shape_critic_ally, 
		obs_shape_critic_enemy, 
		obs_shape_actor, 
		rnn_num_layers_actor,
		actor_hidden_state,
		rnn_num_layers_q,
		q_hidden_state,
		num_actions, 
		data_chunk_length,
		norm_returns_q,
		clamp_rewards,
		clamp_rewards_value_min,
		clamp_rewards_value_max,
		target_calc_style,
		td_lambda,
		gae_lambda,
		n_steps,
		gamma,
		Q_PopArt,
		):
		self.num_episodes = num_episodes
		self.max_time_steps = max_time_steps
		self.num_agents = num_agents
		self.num_enemies = num_enemies
		self.obs_shape_critic_ally = obs_shape_critic_ally
		self.obs_shape_critic_enemy = obs_shape_critic_enemy
		self.obs_shape_actor = obs_shape_actor
		self.rnn_num_layers_actor = rnn_num_layers_actor
		self.actor_hidden_state = actor_hidden_state
		self.rnn_num_layers_q = rnn_num_layers_q
		self.q_hidden_state = q_hidden_state
		self.num_actions = num_actions

		self.data_chunk_length = data_chunk_length
		self.norm_returns_q = norm_returns_q
		self.clamp_rewards = clamp_rewards
		self.clamp_rewards_value_min = clamp_rewards_value_min
		self.clamp_rewards_value_max = clamp_rewards_value_max

		self.target_calc_style = target_calc_style
		self.td_lambda = td_lambda
		self.gae_lambda = gae_lambda
		self.gamma = gamma
		self.n_steps = n_steps
			
		if self.norm_returns_q:
			self.q_value_norm = Q_PopArt

		self.episode_num = 0
		self.time_step = 0

		self.states_critic_allies = np.zeros((num_episodes, max_time_steps, num_agents, obs_shape_critic_ally))
		self.states_critic_enemies = np.zeros((num_episodes, max_time_steps, num_enemies, obs_shape_critic_enemy))
		self.hidden_state_q = np.zeros((num_episodes, max_time_steps, rnn_num_layers_q, num_agents, q_hidden_state))
		self.Q_values = np.zeros((num_episodes, max_time_steps+1, num_agents))
		self.states_actor = np.zeros((num_episodes, max_time_steps, num_agents, obs_shape_actor))
		self.hidden_state_actor = np.zeros((num_episodes, max_time_steps, rnn_num_layers_actor, num_agents, actor_hidden_state))
		self.logprobs = np.zeros((num_episodes, max_time_steps, num_agents))
		self.actions = np.zeros((num_episodes, max_time_steps, num_agents), dtype=int)
		self.last_one_hot_actions = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.one_hot_actions = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.action_masks = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.rewards = np.zeros((num_episodes, max_time_steps, num_agents))
		self.dones = np.ones((num_episodes, max_time_steps+1, num_agents))

		self.episode_length = np.zeros(num_episodes)
	

	def clear(self):

		self.states_critic_allies = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.obs_shape_critic_ally))
		self.states_critic_enemies = np.zeros((self.num_episodes, self.max_time_steps, self.num_enemies, self.obs_shape_critic_enemy))
		self.hidden_state_q = np.zeros((self.num_episodes, self.max_time_steps, self.rnn_num_layers_q, self.num_agents, self.q_hidden_state))
		self.Q_values = np.zeros((self.num_episodes, self.max_time_steps+1, self.num_agents))
		self.states_actor = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.obs_shape_actor))
		self.hidden_state_actor = np.zeros((self.num_episodes, self.max_time_steps, self.rnn_num_layers_actor, self.num_agents, self.actor_hidden_state))
		self.logprobs = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents))
		self.actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents), dtype=int)
		self.last_one_hot_actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.one_hot_actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.action_masks = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.rewards = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents))
		self.dones = np.ones((self.num_episodes, self.max_time_steps+1, self.num_agents))

		self.episode_length = np.zeros(self.num_episodes)

		self.time_step = 0
		self.episode_num = 0

	def push(
		self, 
		state_critic_allies, 
		state_critic_enemies, 
		q_value,
		hidden_state_q,
		state_actor, 
		hidden_state_actor, 
		logprobs, 
		actions, 
		last_one_hot_actions, 
		one_hot_actions, 
		action_masks, 
		rewards, 
		dones
		):

		self.states_critic_allies[self.episode_num][self.time_step] = state_critic_allies
		self.states_critic_enemies[self.episode_num][self.time_step] = state_critic_enemies
		self.hidden_state_q[self.episode_num][self.time_step] = hidden_state_q
		self.Q_values[self.episode_num][self.time_step] = q_value
		self.states_actor[self.episode_num][self.time_step] = state_actor
		self.hidden_state_actor[self.episode_num][self.time_step] = hidden_state_actor
		self.logprobs[self.episode_num][self.time_step] = logprobs
		self.actions[self.episode_num][self.time_step] = actions
		self.last_one_hot_actions[self.episode_num][self.time_step] = last_one_hot_actions
		self.one_hot_actions[self.episode_num][self.time_step] = one_hot_actions
		self.action_masks[self.episode_num][self.time_step] = action_masks
		self.rewards[self.episode_num][self.time_step] = rewards
		self.dones[self.episode_num][self.time_step] = dones

		if self.time_step < self.max_time_steps-1:
			self.time_step += 1


	def end_episode(
		self, 
		t, 
		q_value, 
		dones
		):
		self.Q_values[self.episode_num][self.time_step+1] = q_value
		self.dones[self.episode_num][self.time_step+1] = dones

		self.episode_length[self.episode_num] = t
		self.episode_num += 1
		self.time_step = 0


	def sample_recurrent_policy(self):

		data_chunks = self.max_time_steps // self.data_chunk_length
		rand_batch = np.random.permutation(self.num_episodes)
		rand_time = np.random.permutation(data_chunks)

		states_critic_allies = torch.from_numpy(self.states_critic_allies).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.obs_shape_critic_ally)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.obs_shape_critic_ally)
		states_critic_enemies = torch.from_numpy(self.states_critic_enemies).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_enemies, self.obs_shape_critic_enemy)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_enemies, self.obs_shape_critic_enemy)
		hidden_state_q = torch.from_numpy(self.hidden_state_q).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers_q, self.num_agents, self.q_hidden_state)[:, rand_time][rand_batch, :][:, :, 0, :, :, :].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers_q, -1, self.q_hidden_state)
		states_actor = torch.from_numpy(self.states_actor).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.obs_shape_actor)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.obs_shape_actor).reshape(-1, self.data_chunk_length, self.num_agents, self.obs_shape_actor)
		hidden_state_actor = torch.from_numpy(self.hidden_state_actor).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers_actor, self.num_agents, self.actor_hidden_state)[:, rand_time][rand_batch, :][:, :, 0, :, :, :].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers_actor, -1, self.actor_hidden_state)
		logprobs = torch.from_numpy(self.logprobs).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		actions = torch.from_numpy(self.actions).long().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		last_one_hot_actions = torch.from_numpy(self.last_one_hot_actions).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.num_actions)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.num_actions)
		one_hot_actions = torch.from_numpy(self.one_hot_actions).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.num_actions)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.num_actions)
		action_masks = torch.from_numpy(self.action_masks).bool().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.num_actions)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.num_actions)
		masks = 1-torch.from_numpy(self.dones[:, :-1]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		
		# target_values, target_q_values, advantage = self.calculate_targets(advantage_type, episode, select_above_threshold)
		q_values = torch.from_numpy(self.Q_values[:, :-1, :]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		target_q_values = self.target_q_values.float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		advantage = self.advantage.float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		
		return states_critic_allies, states_critic_enemies, hidden_state_q, states_actor, hidden_state_actor, logprobs, \
		actions, last_one_hot_actions, one_hot_actions, action_masks, masks, q_values, target_q_values, advantage

	def calculate_targets(self, episode):
		
		masks = 1 - torch.from_numpy(self.dones[:, :-1, :])
		next_mask = 1 - torch.from_numpy(self.dones[:, -1, :])

		rewards = torch.from_numpy(self.rewards)

		if self.clamp_rewards:
			rewards = torch.clamp(rewards, min=self.clamp_rewards_value_min, max=self.clamp_rewards_value_max)
		
		# TARGET CALC
		q_values = torch.from_numpy(self.Q_values[:, :-1, :]) * masks
		next_q_values = torch.from_numpy(self.Q_values[:, -1, :]) * next_mask

		if self.norm_returns_q:
			values_shape = q_values.shape
			q_values = self.q_value_norm.denormalize(q_values.view(-1)).view(values_shape) * masks.view(values_shape)

			next_values_shape = next_q_values.shape
			next_q_values = self.q_value_norm.denormalize(next_q_values.view(-1)).view(next_values_shape) * next_mask.view(next_values_shape)

		if self.target_calc_style == "GAE":
			target_q_values = self.gae_targets(rewards, q_values, next_q_values, masks, next_mask)
		elif self.target_calc_style == "N_steps":
			target_q_values = self.nstep_returns(rewards, q_values, next_q_values, masks, next_mask)

		self.advantage = (target_q_values - q_values).detach()

		if self.norm_returns_q:
			targets_shape = target_q_values.shape
			self.q_value_norm.update(target_q_values.view(-1), masks.view(-1))
			
			target_q_values = self.q_value_norm.normalize(target_q_values.view(-1)).view(targets_shape) * masks.view(targets_shape)
		
		self.target_q_values = target_q_values.cpu()


	def gae_targets(self, rewards, values, next_value, masks, next_mask):
		
		target_values = rewards.new_zeros(*rewards.shape)
		advantage = 0

		for t in reversed(range(0, rewards.shape[1])):

			td_error = rewards[:,t,:] + (self.gamma * next_value * next_mask) - values.data[:,t,:] * masks[:, t, :]
			advantage = td_error + self.gamma * self.gae_lambda * advantage * next_mask
			
			target_values[:, t, :] = advantage + values.data[:, t, :] * masks[:, t, :]
			
			next_value = values.data[:, t, :]
			next_mask = masks[:, t, :]

		return target_values*masks

	
	def build_td_lambda_targets(self, rewards, values, next_value, masks, next_mask):
		# Assumes  <target_qs > in B*T*A and <reward >, <terminated >  in B*T*A, <mask > in (at least) B*T-1*1
		# Initialise  last  lambda -return  for  not  terminated  episodes
		# ret = target_qs.new_zeros(*target_qs.shape)
		# ret = target_qs * (1-terminated[:, :-1]) # some episodes end early so we can't assume that by copying the last target_qs in ret would be good enough
		# ret[:, -1] = target_qs[:, -1] * (1 - (torch.sum(terminated, dim=1)>0).int())
		# Backwards  recursive  update  of the "forward  view"
		# for t in range(ret.shape[1] - 2, -1, -1):
		# 	ret[:, t] = self.td_lambda * self.gamma * ret[:, t + 1] + mask[:, t].unsqueeze(-1) \
		# 				* (rewards[:, t] + (1 - self.td_lambda) * self.gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
		# Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
		# return ret[:, 0:-1]
		# return ret
		...


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



class RewardRolloutBuffer:
	def __init__(
		self, 
		num_new_policy_episodes,
		num_episodes_capacity, 
		max_time_steps, 
		num_agents, 
		num_enemies,
		obs_shape,
		num_actions, 
		batch_size,
		):

		self.num_new_policy_episodes = num_new_policy_episodes
		self.num_episodes_capacity = num_episodes_capacity
		self.max_time_steps = max_time_steps
		self.num_agents = num_agents
		self.num_enemies = num_enemies
		self.obs_shape = obs_shape
		self.num_actions = num_actions

		self.episode_num = 0
		self.time_step = 0
		self.batch_size = batch_size

		self.states = np.zeros((num_episodes_capacity, max_time_steps, num_agents, obs_shape))
		self.one_hot_actions = np.zeros((num_episodes_capacity, max_time_steps, num_agents, num_actions))
		self.episodic_rewards = np.zeros((num_episodes_capacity))
		self.dones = np.ones((num_episodes_capacity, max_time_steps, num_agents))

		self.episode_length = np.zeros(num_episodes_capacity)
	

	def clear(self):
		episode_num = self.episode_num % self.num_episodes_capacity

		self.states[episode_num] = np.zeros((self.max_time_steps, self.num_agents, self.obs_shape))
		self.one_hot_actions[episode_num] = np.zeros((self.max_time_steps, self.num_agents, self.num_actions))
		self.episodic_rewards[episode_num] = 0.0
		self.dones[episode_num] = np.ones((self.max_time_steps, self.num_agents))

		self.episode_length[episode_num] = 0

	def end_episode(self, episodic_reward, t):
		episode_num = self.episode_num % self.num_episodes_capacity

		self.episodic_rewards[episode_num] = episodic_reward
		
		self.episode_length[episode_num] = t

		self.episode_num += 1

		# self.clear()

		self.time_step = 0

	def push(
		self, 
		states,
		one_hot_actions, 
		dones
		):

		episode_num = self.episode_num % self.num_episodes_capacity

		self.states[episode_num][self.time_step] = states
		self.one_hot_actions[episode_num][self.time_step] = one_hot_actions
		self.dones[episode_num][self.time_step] = dones

		if self.time_step < self.max_time_steps-1:
			self.time_step += 1


	def sample(self):

		# uniform sampling
		if self.episode_num > self.num_episodes_capacity:
			episode_num = self.num_episodes_capacity
		else:
			episode_num = self.episode_num
		
		print(episode_num, self.batch_size)
		rand_batch = random.sample(range(0, episode_num), self.batch_size)

		states = torch.from_numpy(self.states[rand_batch]).float()
		episodic_rewards = torch.from_numpy(self.episodic_rewards[rand_batch]).float()
		one_hot_actions = torch.from_numpy(self.one_hot_actions[rand_batch]).float()
		masks = 1-torch.from_numpy(self.dones[rand_batch]).float()
		
		return states, episodic_rewards, one_hot_actions, masks

	def sample_new_data(self):
		# find current index in the buffer
		episode_num = self.episode_num % self.num_episodes_capacity
		if episode_num-self.num_new_policy_episodes < 0:
			new_states = torch.from_numpy(np.append(self.states[episode_num-self.num_new_policy_episodes], self.states[:episode_num], axis=-1)).float()
			new_episodic_rewards = torch.from_numpy(np.append(self.episodic_rewards[episode_num-self.num_new_policy_episodes], self.episodic_rewards[:episode_num], axis=-1)).float()
			new_one_hot_actions = torch.from_numpy(np.append(self.one_hot_actions[episode_num-self.num_new_policy_episodes], self.one_hot_actions[:episode_num], axis=-1)).float()
			new_masks = 1-torch.from_numpy(np.append(self.dones[episode_num-self.num_new_policy_episodes], self.dones[:episode_num], axis=-1)).float()
		else:
			new_states = torch.from_numpy(self.states[episode_num-self.num_new_policy_episodes: episode_num]).float()
			new_episodic_rewards = torch.from_numpy(self.episodic_rewards[episode_num-self.num_new_policy_episodes: episode_num]).float()
			new_one_hot_actions = torch.from_numpy(self.one_hot_actions[episode_num-self.num_new_policy_episodes: episode_num]).float()
			new_masks = 1-torch.from_numpy(self.dones[episode_num-self.num_new_policy_episodes: episode_num]).float()

		return new_states, new_episodic_rewards, new_one_hot_actions, new_masks