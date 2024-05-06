import random
import numpy as np
import torch

class ReplayMemory:
	def __init__(
		self, 
		algorithm_type,
		experiment_type,
		capacity, 
		max_episode_len, 
		num_agents, 
		q_obs_shape, 
		q_mix_obs_shape, 
		rnn_num_layers, 
		rnn_hidden_state_shape, 
		reward_model_obs_shape,
		data_chunk_length, 
		action_shape,
		gamma,
		lambda_,
		device,
		):

		self.algorithm_type = algorithm_type
		self.experiment_type = experiment_type
		self.capacity = capacity
		self.length = 0
		self.episode = 0
		self.t = 0
		self.max_episode_len = max_episode_len
		self.num_agents = num_agents
		self.q_obs_shape = q_obs_shape
		self.q_mix_obs_shape = q_mix_obs_shape
		self.rnn_num_layers = rnn_num_layers
		self.rnn_hidden_state_shape = rnn_hidden_state_shape
		self.reward_model_obs_shape = reward_model_obs_shape
		self.data_chunk_length = data_chunk_length
		self.action_shape = action_shape
		self.gamma = gamma
		self.lambda_ = lambda_
		self.device = device

		self.buffer = dict()
		self.buffer['state'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents, self.q_obs_shape), dtype=np.float32)
		self.buffer['rnn_hidden_state'] = np.zeros((self.capacity, self.max_episode_len, self.rnn_num_layers, self.num_agents, self.rnn_hidden_state_shape), dtype=np.float32)
		self.buffer['mask_actions'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents, self.action_shape), dtype=np.float32)
		self.buffer['next_state'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents, self.q_obs_shape), dtype=np.float32)
		self.buffer['next_rnn_hidden_state'] = np.zeros((self.capacity, self.max_episode_len, self.rnn_num_layers, self.num_agents, self.rnn_hidden_state_shape), dtype=np.float32)
		self.buffer['full_state'] = np.zeros((self.capacity, self.max_episode_len, 1, self.q_mix_obs_shape), dtype=np.float32)
		self.buffer['next_full_state'] = np.zeros((self.capacity, self.max_episode_len, 1, self.q_mix_obs_shape), dtype=np.float32)
		self.buffer['reward_model_obs'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents, self.reward_model_obs_shape), dtype=np.float32)
		self.buffer['actions'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents), dtype=np.float32)
		self.buffer['last_one_hot_actions'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents, self.action_shape), dtype=np.float32)
		self.buffer['next_last_one_hot_actions'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents, self.action_shape), dtype=np.float32)
		self.buffer['next_mask_actions'] = np.zeros((self.capacity, self.max_episode_len, self.num_agents, self.action_shape), dtype=np.float32)
		self.buffer['reward'] = np.zeros((self.capacity, self.max_episode_len), dtype=np.float32)
		self.buffer['indiv_dones'] = np.ones((self.capacity, self.max_episode_len, self.num_agents), dtype=np.float32)
		self.buffer['next_indiv_dones'] = np.ones((self.capacity, self.max_episode_len, self.num_agents), dtype=np.float32)
		self.buffer['done'] = np.ones((self.capacity, self.max_episode_len), dtype=np.float32)
		self.buffer['mask'] = np.zeros((self.capacity, self.max_episode_len), dtype=np.float32)

		self.episode_len = np.zeros(self.capacity)

	# push once per step
	def push(self, state, rnn_hidden_state, full_state, reward_model_obs, actions, last_one_hot_actions, mask_actions, next_state, next_rnn_hidden_state, next_full_state, next_last_one_hot_actions, next_mask_actions, reward, done, indiv_dones, next_indiv_dones):
		self.buffer['state'][self.episode][self.t] = state
		self.buffer['rnn_hidden_state'][self.episode][self.t] = rnn_hidden_state
		self.buffer['full_state'][self.episode][self.t] = full_state
		self.buffer['reward_model_obs'][self.episode][self.t] = reward_model_obs
		self.buffer['actions'][self.episode][self.t] = actions
		self.buffer['last_one_hot_actions'][self.episode][self.t] = last_one_hot_actions
		self.buffer['mask_actions'][self.episode][self.t] = mask_actions
		self.buffer['next_state'][self.episode][self.t] = next_state
		self.buffer['next_rnn_hidden_state'][self.episode][self.t] = next_rnn_hidden_state
		self.buffer['next_full_state'][self.episode][self.t] = next_full_state
		self.buffer['next_last_one_hot_actions'][self.episode][self.t] = next_last_one_hot_actions
		self.buffer['next_mask_actions'][self.episode][self.t] = next_mask_actions
		self.buffer['reward'][self.episode][self.t] = reward
		self.buffer['done'][self.episode][self.t] = done
		self.buffer['indiv_dones'][self.episode][self.t] = indiv_dones
		self.buffer['next_indiv_dones'][self.episode][self.t] = next_indiv_dones
		self.buffer['mask'][self.episode][self.t] = 1.
		self.t += 1

	def end_episode(self):
		self.episode_len[self.episode] = self.t - 1
		if self.length < self.capacity:
			self.length += 1
		self.episode = (self.episode + 1) % self.capacity
		self.t = 0
		# clear previous data
		self.buffer['state'][self.episode] = np.zeros((self.max_episode_len, self.num_agents, self.q_obs_shape), dtype=np.float32)
		self.buffer['rnn_hidden_state'][self.episode] = np.zeros((self.max_episode_len, self.rnn_num_layers, self.num_agents, self.rnn_hidden_state_shape), dtype=np.float32)
		self.buffer['full_state'][self.episode] = np.zeros((self.max_episode_len, 1, self.q_mix_obs_shape), dtype=np.float32)
		self.buffer['actions'][self.episode] = np.zeros((self.max_episode_len, self.num_agents), dtype=np.float32)
		self.buffer['last_one_hot_actions'][self.episode] = np.zeros((self.max_episode_len, self.num_agents, self.action_shape), dtype=np.float32)
		self.buffer['mask_actions'][self.episode] = np.zeros((self.max_episode_len, self.num_agents, self.action_shape), dtype=np.float32)
		self.buffer['next_state'][self.episode] = np.zeros((self.max_episode_len, self.num_agents, self.q_obs_shape), dtype=np.float32)
		self.buffer['next_rnn_hidden_state'][self.episode] = np.zeros((self.max_episode_len, self.rnn_num_layers, self.num_agents, self.rnn_hidden_state_shape), dtype=np.float32)
		self.buffer['next_full_state'][self.episode] = np.zeros((self.max_episode_len, 1, self.q_mix_obs_shape), dtype=np.float32)
		self.buffer['reward_model_obs'][self.episode] = np.zeros((self.max_episode_len, self.num_agents, self.reward_model_obs_shape), dtype=np.float32)
		self.buffer['next_last_one_hot_actions'][self.episode] = np.zeros((self.max_episode_len, self.num_agents, self.action_shape), dtype=np.float32)
		self.buffer['next_mask_actions'][self.episode] = np.zeros((self.max_episode_len, self.num_agents, self.action_shape), dtype=np.float32)
		self.buffer['reward'][self.episode] = np.zeros((self.max_episode_len,), dtype=np.float32)
		self.buffer['done'][self.episode] = np.ones((self.max_episode_len,), dtype=np.float32)
		self.buffer['indiv_dones'][self.episode] = np.ones((self.max_episode_len, self.num_agents), dtype=np.float32)
		self.buffer['next_indiv_dones'][self.episode] = np.ones((self.max_episode_len, self.num_agents), dtype=np.float32)
		self.buffer['mask'][self.episode] = np.zeros((self.max_episode_len,), dtype=np.float32)



	def reward_model_output(self, reward_model, reward_model_obs_batch, next_last_one_hot_actions_batch, mask_batch, agent_masks_batch, episode_len_batch):
		
		if "AREL" in self.experiment_type:
			state_actions_batch = torch.cat([reward_model_obs_batch, next_last_one_hot_actions_batch], dim=-1)

			with torch.no_grad():
				reward_episode_wise, reward_time_wise, _, _, agent_weights, _ = reward_model(
					state_actions_batch.permute(0, 2, 1, 3).to(self.device),
					team_masks=mask_batch.to(self.device),
					agent_masks=agent_masks_batch.to(self.device)
					)

			reward_batch = reward_time_wise.cpu()

			if self.experiment_type == "AREL_agent":
				reward_batch = reward_batch.unsqueeze(-1) * agent_weights.cpu()


		elif "ATRR" in self.experiment_type:

			with torch.no_grad():
				# reward_episode_wise, temporal_weights, agent_weights, _, _, _, _ = reward_model(
				# 	state_batch.permute(0, 2, 1, 3).to(self.device), 
				# 	next_last_one_hot_actions_batch.permute(0, 2, 1, 3).to(self.device), 
				# 	team_masks=mask_batch.to(self.device),
				# 	agent_masks=agent_masks_batch.to(self.device),
				# 	episode_len=episode_len_batch.to(self.device),
				# 	)
				reward_agent_temporal, temporal_weights, agent_weights, _, _, _, _ = reward_model(
					reward_model_obs_batch.permute(0, 2, 1, 3).to(self.device), 
					next_last_one_hot_actions_batch.permute(0, 2, 1, 3).to(self.device), 
					team_masks=mask_batch.to(self.device),
					agent_masks=agent_masks_batch.to(self.device),
					episode_len=episode_len_batch.to(self.device),
					)

			# if self.norm_rewards:
			# 	shape = reward_episode_wise.shape
			# 	reward_episode_wise = self.reward_normalizer.denormalize(reward_episode_wise.view(-1)).view(shape)

			# reward_batch = (reward_episode_wise * temporal_weights).cpu()

			# if self.experiment_type == "ATRR_agent":
			# 	reward_batch = reward_batch.unsqueeze(-1) * agent_weights[:, :-1, :].cpu()

		# return reward_batch
		return reward_agent_temporal.cpu().permute(0, 2, 1)


	def build_td_lambda_targets(self, rewards, terminated, target_qs):
		# Assumes  <target_qs > in B*T*A and <reward >, <terminated >  in B*T*A
		# Initialise  last  lambda -return  for  not  terminated  episodes
		# print(rewards.shape, terminated.shape, mask.shape, target_qs.shape)
		ret = target_qs.new_zeros(*target_qs.shape)
		ret = target_qs * (1-terminated)
		# ret[:, -1] = target_qs[:, -1] * (1 - (torch.sum(terminated, dim=1)>0).int())
		# Backwards  recursive  update  of the "forward  view"
		for t in range(ret.shape[1] - 2, -1,  -1):
			ret[:, t] = self.lambda_ * self.gamma * ret[:, t+1] + (1-terminated[:, t]) \
						* (rewards[:, t] + (1 - self.lambda_) * self.gamma * target_qs[:, t] * (1 - terminated[:, t]))
		# Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
		# return ret[:, 0:-1]
		return ret


	def sample(self, num_episodes, Q_network, target_Q_network, target_QMix_network, reward_model):
		assert num_episodes <= self.length

		data_chunks = self.max_episode_len // self.data_chunk_length
		batch_indices = np.random.choice(self.length, size=num_episodes, replace=False)
		rand_time = np.random.permutation(data_chunks)

		state_batch = torch.from_numpy(np.take(self.buffer['state'], batch_indices, axis=0))
		last_one_hot_actions_batch = torch.from_numpy(np.take(self.buffer['last_one_hot_actions'], batch_indices, axis=0))
		rnn_hidden_state_batch = torch.from_numpy(np.take(self.buffer['rnn_hidden_state'], batch_indices, axis=0))
		mask_actions_batch = torch.from_numpy(np.take(self.buffer['mask_actions'], batch_indices, axis=0)).bool()
		full_state_batch = torch.from_numpy(np.take(self.buffer['full_state'], batch_indices, axis=0))
		reward_model_obs_batch = torch.from_numpy(np.take(self.buffer['reward_model_obs'], batch_indices, axis=0))
		next_state_batch = torch.from_numpy(np.take(self.buffer['next_state'], batch_indices, axis=0))
		next_last_one_hot_actions_batch = torch.from_numpy(np.take(self.buffer['next_last_one_hot_actions'], batch_indices, axis=0))
		next_rnn_hidden_state_batch = torch.from_numpy(np.take(self.buffer['next_rnn_hidden_state'], batch_indices, axis=0))
		next_mask_actions_batch = torch.from_numpy(np.take(self.buffer['next_mask_actions'], batch_indices, axis=0)).bool()
		next_full_state_batch = torch.from_numpy(np.take(self.buffer['next_full_state'], batch_indices, axis=0))
		done_batch = torch.from_numpy(np.take(self.buffer['done'], batch_indices, axis=0))
		indiv_done_batch = torch.from_numpy(np.take(self.buffer['indiv_dones'], batch_indices, axis=0))
		episode_len_batch = torch.from_numpy(np.take(self.episode_len, batch_indices, axis=0)).long()

		if reward_model is not None:
			reward_batch = self.reward_model_output(reward_model, reward_model_obs_batch, last_one_hot_actions_batch, 1-done_batch, 1-indiv_done_batch, episode_len_batch)
		else:
			reward_batch = torch.from_numpy(np.take(self.buffer['reward'], batch_indices, axis=0))
		

		with torch.no_grad():
			# Calculating next Q values of MAS using target network
			next_final_state_batch = torch.cat([next_state_batch, next_last_one_hot_actions_batch], dim=-1)
			next_Q_evals, _ = Q_network(
				next_final_state_batch.reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1).to(self.device), 
				next_rnn_hidden_state_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers, self.num_agents, -1)[:, :, 0].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers, num_episodes*data_chunks*self.num_agents, -1).to(self.device),
				next_mask_actions_batch.reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1).to(self.device)
				)
			next_Q_target, _ = target_Q_network(
				next_final_state_batch.reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1).to(self.device), 
				next_rnn_hidden_state_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers, self.num_agents, -1)[:, :, 0].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers, num_episodes*data_chunks*self.num_agents, -1).to(self.device),
				next_mask_actions_batch.reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1).to(self.device)
				)
			next_a_argmax = torch.argmax(next_Q_evals, dim=-1, keepdim=True)
			next_Q_target = torch.gather(next_Q_target, dim=-1, index=next_a_argmax.to(self.device)).squeeze(-1)
			
			if self.experiment_type == "ATRR_agent" or self.experiment_type == "AREL_agent":
				# Finally using TD-lambda equation to generate targets
				target_Q_values = self.build_td_lambda_targets(reward_batch.permute(0, 2, 1).reshape(-1, self.max_episode_len), indiv_done_batch.permute(0, 2, 1).reshape(-1, self.max_episode_len), next_Q_target.reshape(-1, self.max_episode_len, self.num_agents).permute(0, 2, 1).reshape(-1, self.max_episode_len).cpu())
			else:
				next_Q_mix_target = target_QMix_network(
				next_Q_target, 
				next_full_state_batch.reshape(num_episodes*data_chunks, self.data_chunk_length, -1).to(self.device), 
				).reshape(-1) #* team_mask_batch.reshape(-1).to(self.device)

				# Finally using TD-lambda equation to generate targets
				target_Q_values = self.build_td_lambda_targets(reward_batch.reshape(-1, self.max_episode_len), done_batch.reshape(-1, self.max_episode_len), next_Q_mix_target.reshape(-1, self.max_episode_len).cpu())

		
		
		
		state_batch = state_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		rnn_hidden_state_batch = rnn_hidden_state_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers, self.num_agents, -1)[:, rand_time][:, :, 0].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers, num_episodes*data_chunks*self.num_agents, -1)
		full_state_batch = full_state_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, 1, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, -1)
		actions_batch = torch.from_numpy(np.take(self.buffer['actions'], batch_indices, axis=0)).long().reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		last_one_hot_actions_batch = last_one_hot_actions_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		mask_actions_batch = mask_actions_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		next_state_batch = next_state_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		next_rnn_hidden_state_batch = next_rnn_hidden_state_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers, self.num_agents, -1)[:, rand_time][:, :, 0].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers, num_episodes*data_chunks*self.num_agents, -1)
		next_full_state_batch = next_full_state_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, 1, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, -1)
		next_last_one_hot_actions_batch = next_last_one_hot_actions_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		next_mask_actions_batch = next_mask_actions_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		reward_batch = reward_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, -1)
		done_batch = done_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, -1)
		indiv_dones_batch = torch.from_numpy(np.take(self.buffer['indiv_dones'], batch_indices, axis=0)).reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		next_indiv_dones_batch = torch.from_numpy(np.take(self.buffer['next_indiv_dones'], batch_indices, axis=0)).reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		team_mask_batch = torch.from_numpy(np.take(self.buffer['mask'], batch_indices, axis=0)).reshape(num_episodes, data_chunks, self.data_chunk_length, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, -1)

		if self.experiment_type == "AREL_agent" or self.experiment_type == "ATRR_agent":
			target_Q_values = target_Q_values.reshape(num_episodes, self.num_agents, data_chunks, self.data_chunk_length)[:, :, rand_time].reshape(num_episodes*self.num_agents*data_chunks, self.data_chunk_length)
		else:
			target_Q_values = target_Q_values.reshape(num_episodes, data_chunks, self.data_chunk_length)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length)

		max_episode_len = int(np.max(self.episode_len[batch_indices]))

		return state_batch, rnn_hidden_state_batch, full_state_batch, actions_batch, last_one_hot_actions_batch, mask_actions_batch, next_state_batch, next_rnn_hidden_state_batch, next_full_state_batch, \
		next_last_one_hot_actions_batch, next_mask_actions_batch, reward_batch, done_batch, indiv_dones_batch, next_indiv_dones_batch, team_mask_batch, max_episode_len, target_Q_values

	
	def sample_reward_model(self, num_episodes):
		assert num_episodes <= self.length
		batch_indices = np.random.choice(self.length, size=num_episodes, replace=False)
		reward_model_obs_batch = np.take(self.buffer['reward_model_obs'], batch_indices, axis=0)
		next_last_one_hot_actions_batch = np.take(self.buffer['next_last_one_hot_actions'], batch_indices, axis=0)
		reward_batch = np.take(self.buffer['reward'], batch_indices, axis=0)
		mask_batch = 1 - np.take(self.buffer['done'], batch_indices, axis=0)
		agent_masks_batch = 1 - np.take(self.buffer['indiv_dones'], batch_indices, axis=0)
		episode_len_batch = np.take(self.episode_len, batch_indices, axis=0)

		return reward_model_obs_batch, next_last_one_hot_actions_batch, reward_batch, mask_batch, agent_masks_batch, episode_len_batch



	def __len__(self):
		return self.length