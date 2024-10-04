import os
import sys
import time
import numpy as np
import datetime
import random
import copy
import multiprocessing as mp

from comet_ml import Experiment

from agent import PPOAgent

import torch
import torch.nn.functional as F

from env_wrappers import ShareSubprocVecEnv
import atexit



class MAPPO:

	def __init__(self, env, dictionary):

		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"
		self.env = ShareSubprocVecEnv(env, truncation_steps=dictionary["max_time_steps"], environment_name=dictionary["environment"])
		self.environment = dictionary["environment"]
		self.gif = dictionary["gif"]
		self.save_model = dictionary["save_model"]
		self.save_model_checkpoint = dictionary["save_model_checkpoint"]
		self.save_comet_ml_plot = dictionary["save_comet_ml_plot"]
		self.learn = dictionary["learn"]
		self.gif_checkpoint = dictionary["gif_checkpoint"]
		self.eval_policy = dictionary["eval_policy"]
		self.parallel_training = dictionary["parallel_training"]
		self.num_workers = dictionary["num_workers"]
		assert len(env) == self.num_workers, f"number of rollout threads ({self.num_workers}) doesn't match the number of env functions ({len(env)})"
		atexit.register(self.close)


		self.num_agents = dictionary["num_agents"]

		self.algorithm_type = dictionary["algorithm_type"]
		self.use_reward_model = dictionary["use_reward_model"]
		self.warm_up_period = dictionary["warm_up_period"]
		
		if "StarCraft" in self.environment:
			self.num_enemies = dictionary["num_enemies"]

			self.enemy_ids = []
			for i in range(self.num_enemies):
				enemy_id = np.array([0 for i in range(self.num_enemies)])
				enemy_id[i] = 1
				self.enemy_ids.append(enemy_id)
			self.enemy_ids = np.array(self.enemy_ids)
		else:
			self.num_enemies = 1

		self.num_actions = dictionary["num_actions"]
		self.date_time = f"{datetime.datetime.now():%d-%m-%Y}"
		self.env_name = dictionary["env"]
		self.test_num = dictionary["test_num"]
		self.max_episodes = dictionary["max_episodes"]
		self.max_time_steps = dictionary["max_time_steps"]
		self.experiment_type = dictionary["experiment_type"]
		self.ppo_eps_elapse_update_freq = dictionary["ppo_eps_elapse_update_freq"]

		# RNN HIDDEN
		self.rnn_num_layers_v = dictionary["rnn_num_layers_v"]
		self.rnn_hidden_v = dictionary["rnn_hidden_v"]
		self.rnn_num_layers_actor = dictionary["rnn_num_layers_actor"]
		self.rnn_hidden_actor = dictionary["rnn_hidden_actor"]


		if self.use_reward_model:
			self.reward_batch_size = dictionary["reward_batch_size"]
			self.update_reward_model_freq = dictionary["update_reward_model_freq"]
			self.reward_model_update_epochs = dictionary["reward_model_update_epochs"]

		self.comet_ml = None
		if self.save_comet_ml_plot:
			self.comet_ml = Experiment("im5zK8gFkz6j07uflhc3hXk8I", project_name=dictionary["test_num"])
			self.comet_ml.set_name(dictionary["experiment_name"])
			self.comet_ml.log_parameters(dictionary)


		self.agents = PPOAgent(dictionary, self.comet_ml)
		# self.init_critic_hidden_state(np.zeros((1, self.num_agents, 256)))

		if self.save_model:
			critic_dir = dictionary["critic_dir"]
			try: 
				os.makedirs(critic_dir, exist_ok = True) 
				print("Critic Directory created successfully") 
			except OSError as error: 
				print("Critic Directory can not be created") 
			actor_dir = dictionary["actor_dir"]
			try: 
				os.makedirs(actor_dir, exist_ok = True) 
				print("Actor Directory created successfully") 
			except OSError as error: 
				print("Actor Directory can not be created")
			optim_dir = dictionary["optim_dir"]
			try: 
				os.makedirs(optim_dir, exist_ok = True) 
				print("Optim Directory created successfully") 
			except OSError as error: 
				print("Optim Directory can not be created") 
			reward_dir = dictionary["reward_dir"]
			try: 
				os.makedirs(reward_dir, exist_ok = True) 
				print("Reward Directory created successfully") 
			except OSError as error: 
				print("Reward Directory can not be created") 

			
			self.critic_model_path = critic_dir+"critic"
			self.actor_model_path = actor_dir+"actor"
			self.optim_model_path = optim_dir+"optim"
			self.reward_model_path = reward_dir+"reward"
			

		if self.gif:
			gif_dir = dictionary["gif_dir"]
			try: 
				os.makedirs(gif_dir, exist_ok = True) 
				print("Gif Directory created successfully") 
			except OSError as error: 
				print("Gif Directory can not be created")
			self.gif_path = gif_dir+self.env_name+'.gif'


		if self.eval_policy:
			self.policy_eval_dir = dictionary["policy_eval_dir"]
			try: 
				os.makedirs(self.policy_eval_dir, exist_ok = True) 
				print("Policy Eval Directory created successfully") 
			except OSError as error: 
				print("Policy Eval Directory can not be created")

			self.rewards = []
			self.rewards_mean_per_1000_eps = []
			self.timesteps = []
			self.timesteps_mean_per_1000_eps = []


	def make_gif(self,images,fname,fps=10, scale=1.0):
		from moviepy.editor import ImageSequenceClip
		"""Creates a gif given a stack of images using moviepy
		Notes
		-----
		works with current Github version of moviepy (not the pip version)
		https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
		Usage
		-----
		>>> X = randn(100, 64, 64)
		>>> gif('test.gif', X)
		Parameters
		----------
		filename : string
			The filename of the gif to write to
		array : array_like
			A numpy array that contains a sequence of images
		fps : int
			frames per second (default: 10)
		scale : float
			how much to rescale each image by (default: 1.0)
		"""

		# copy into the color dimension if the images are black and white
		if images.ndim == 3:
			images = images[..., np.newaxis] * np.ones(3)

		# make the moviepy clip
		clip = ImageSequenceClip(list(images), fps=fps).resize(scale)
		clip.write_gif(fname, fps=fps)


	def close(self):
		print("Cleaning up")
		self.env.close()


	def run(self):  

		def _combine_indiv_dones(info):
			final_indiv_dones = []
			for worker_index in range(self.num_workers):
				if "_indiv_dones" not in info.keys():
					assert info["did_reset"][worker_index]
					final_indiv_dones.append(info["last_info"]["indiv_dones"][worker_index])
				else:
					if info["_indiv_dones"][worker_index]:
						final_indiv_dones.append(info["indiv_dones"][worker_index])
					else:
						assert info["did_reset"][worker_index]
						final_indiv_dones.append(info["last_info"]["indiv_dones"][worker_index])

			return np.array(final_indiv_dones, dtype=np.int64)


		self.reward_plot_counter = 0
		self.num_episodes_done = 0
		self.worker_step_counter = np.zeros(self.num_workers, dtype=np.int64)


		if "StarCraft" in self.environment:
			local_obs, info = self.env.reset(return_info=True)
			mask_actions = np.array(info["avail_actions"], dtype=int)
			ally_states = np.array(info["ally_states"])
			enemy_states = np.array(info["enemy_states"])
			global_obs, common_obs = None, None
		elif "GFootball" in self.environment:
			global_obs, info = local_obs, info = self.env.reset(return_info=True)
			global_obs = np.array(global_obs)
			mask_actions = np.ones((self.num_workers, self.num_agents, self.num_actions))
			ally_states = np.array(info["player_observations"])
			common_obs = np.array(info["common_information"])
			enemy_states = None
			
		

		local_obs = np.array(local_obs)
		last_actions = np.zeros((self.num_workers, self.num_agents)) + self.num_actions
		indiv_dones = np.zeros((self.num_workers, self.num_agents), dtype=np.int64)
		dones = np.zeros((self.num_workers), dtype=np.int64)
		
		images = []

		episodic_team_reward = [0 for _ in range(self.num_workers)]
		episodic_agent_reward = [[0]*self.num_agents for _ in range(self.num_workers)]
		

		episode_reward = [0 for _ in range(self.num_workers)]
		episode_indiv_rewards = [[0]*self.num_agents for _ in range(self.num_workers)]

		rnn_hidden_state_v = np.zeros((self.num_workers, self.rnn_num_layers_v, self.num_agents, self.rnn_hidden_v))
		rnn_hidden_state_actor = np.zeros((self.num_workers, self.rnn_num_layers_actor, self.num_agents, self.rnn_hidden_actor))

		# we don't want the models to be updated by each thread -- first thread that finishes performs the update
		self.update_agent = False
		self.update_reward_model = False

			
		while self.num_episodes_done < self.max_episodes:

			if self.gif:
				# At each step, append an image to list
				# import time
				# time.sleep(0.1)
				self.env.render()
				# Advance a step and render a new image
				with torch.no_grad():
					actions, action_logprob, next_rnn_hidden_state_actor, latent_state_actor = self.agents.get_action_batch(local_obs, last_actions, mask_actions, rnn_hidden_state_actor, greedy=False)
			else:
				actions, action_logprob, next_rnn_hidden_state_actor, latent_state_actor = self.agents.get_action_batch(local_obs, last_actions, mask_actions, rnn_hidden_state_actor)

			one_hot_actions = F.one_hot(torch.tensor(actions), num_classes=self.num_actions).cpu().numpy()

			value, next_rnn_hidden_state_v = self.agents.get_values_batch(local_obs, global_obs, ally_states, enemy_states, actions, rnn_hidden_state_v, indiv_dones)
			
			# Using '__' as an identifier here for ease of control in _add_info function in step()
			additional_info = {
				"__last_actions": last_actions,
				"__rnn_hidden_state_actor": rnn_hidden_state_actor,
				"__mask_actions": mask_actions,
				"__rnn_hidden_state_v": rnn_hidden_state_v,
				"__global_obs": global_obs,
				}


			next_local_obs, rewards, next_dones, info = self.env.step(actions, additional_info)

			next_local_obs = np.array(next_local_obs)
			for i, d in enumerate(dones):
				if not d: #and self.worker_step_counter[i]<self.max_time_steps-1:
					self.worker_step_counter[i] += 1


			if "StarCraft" in self.environment:
				next_ally_states = np.array(info["ally_states"])
				next_enemy_states = np.array(info["enemy_states"])
				next_mask_actions = np.array(info["avail_actions"], dtype=int)
				next_indiv_dones = info["indiv_dones"]
				indiv_rewards = info["indiv_rewards"]
				
				next_global_obs, next_common_obs = None, None

			elif "GFootball" in self.environment:
				next_ally_states = np.array(info["player_observations"])
				next_common_obs = np.array(info["common_information"])
				next_global_obs = next_local_obs
				next_indiv_dones = info["indiv_dones"]
				indiv_rewards = info["indiv_rewards"]
				next_mask_actions = np.ones((self.num_workers, self.num_agents, self.num_actions)) # np.array(info["mask_actions"], dtype=int)

				next_enemy_states = None
			
			
			if self.experiment_type == "temporal_team":
				rewards_to_send = []
				for worker_index in range(self.num_workers):
					rewards_to_send.append(np.array([rewards[worker_index]] * self.num_agents))
			
			elif self.experiment_type == "temporal_agent":
				# rewards_to_send = []
				# for worker_index in range(self.num_workers):
				# 	if "indiv_rewards" in info.keys():
				# 		if info["indiv_rewards"][worker_index]:
				# 			rewards_to_send.append(info["indiv_rewards"][worker_index])
				# 		else:  # this is only possible if the environment resets
				# 			assert info["did_reset"]
				# 			last_info = info["last_info"]
				# 			rewards_to_send.append(last_info["indiv_rewards"][worker_index])
				# 	else:
				# 		assert info["did_reset"]
				# 		last_info = info["last_info"]
				# 		rewards_to_send.append(last_info["indiv_rewards"][worker_index])
				rewards_to_send = indiv_rewards
		
			elif self.experiment_type in ["episodic_team", "uniform_team_redistribution", "AREL", "TAR^2", "STAS", "TAR^2_v2", "TAR^2_HindSight"]:
				rewards_to_send = []
				for worker_index in range(self.num_workers):
					episodic_team_reward[worker_index] = rewards[worker_index]+episodic_team_reward[worker_index] # [r+rewards[worker_index] for r in episodic_team_reward[worker_index]]
					if next_dones[worker_index] or self.worker_step_counter[worker_index] == self.max_time_steps:
						rewards_to_send.append(episodic_team_reward[worker_index])
					else:
						rewards_to_send.append(0)
			
			elif self.experiment_type == "episodic_agent":
				rewards_to_send = []
				# for worker_index in range(self.num_workers):
				# 	if "_indiv_rewards" in info.keys:
				# 		individual_rewards = info["indiv_rewards"][worker_index] if info["_indiv_rewards"][worker_index] else info["last_info"]["indiv_rewards"][worker_index]  # the last info is inside info["last_info"]
				# 	else:
				# 		assert info["did_reset"][worker_index]
				# 		individual_rewards = info["last_info"]["indiv_rewards"][worker_index]
				# 	episodic_agent_reward[worker_index] = [a_r+r for a_r, r in zip(episodic_agent_reward[worker_index], individual_rewards)]
				# 	if (not next_dones[worker_index]) and (self.worker_step_counter[worker_index] == self.max_time_steps):
				# 		rewards_to_send.append(np.array(episodic_agent_reward[worker_index]))
				# 	else:
				# 		rewards_to_send_worker = []
				# 		if "_indiv_dones" in info.keys():
				# 			next_individual_dones = info["indiv_dones"][worker_index] if info["_indiv_dones"][worker_index] else info["last_info"]["indiv_rewards"][worker_index]
				# 		else:
				# 			assert info["did_reset"][worker_index]
				# 			next_individual_dones = info["last_info"]["indiv_rewards"][worker_index]
				# 		for i, d in enumerate(next_individual_dones):
				# 			if d:
				# 				rewards_to_send_worker.append(episodic_agent_reward[worker_index][i])
				# 				# once reward is allocated to agent, make future rewards 0
				# 				episodic_agent_reward[worker_index][i] = 0
				# 			else:
				# 				rewards_to_send_worker.append(0.0)
				# 		rewards_to_send.append(np.array(rewards_to_send_worker))
				for worker_index in range(self.num_workers):
					episodic_agent_reward[worker_index] = np.array(episodic_agent_reward[worker_index]) + np.array(indiv_rewards[worker_index])
					if next_dones[worker_index] or self.worker_step_counter[worker_index] == self.max_time_steps:
						rewards_to_send.append(episodic_agent_reward[worker_index])
					else:
						rewards_to_send.append(np.zeros(self.num_agents))

					episodic_agent_reward[worker_index] = np.array(episodic_agent_reward[worker_index]) * (1-np.array(next_indiv_dones[worker_index])) # reset only agent rewards that are done to avoid appending episodic agent reward in the next iteration

			rewards_to_send = np.array(rewards_to_send)


			if self.learn:
				self.agents.buffer.push(
					ally_states, enemy_states, value, rnn_hidden_state_v, \
					global_obs, local_obs, common_obs, latent_state_actor, rnn_hidden_state_actor, action_logprob, actions, one_hot_actions, mask_actions, \
					rewards_to_send, indiv_dones, dones, self.worker_step_counter
					)

				if self.use_reward_model:
					self.agents.reward_buffer.push(
						ally_states, enemy_states, local_obs, common_obs, actions, mask_actions, rnn_hidden_state_actor, action_logprob, rewards_to_send, dones, indiv_dones
						)

			ally_states, enemy_states = next_ally_states, next_enemy_states
			common_obs = next_common_obs
			global_obs, local_obs, last_actions, mask_actions = next_global_obs, next_local_obs, actions, next_mask_actions
			indiv_dones, dones = _combine_indiv_dones(info), next_dones
			rnn_hidden_state_v, rnn_hidden_state_actor = next_rnn_hidden_state_v, next_rnn_hidden_state_actor

			
			for worker_index in range(self.num_workers):  # TODO for later: vectorize this loop if possible
				episode_reward[worker_index] += rewards[worker_index]
				if "_indiv_rewards" in info.keys():
					individual_rewards = info["indiv_rewards"][worker_index] if info["_indiv_rewards"][worker_index] else info["last_info"]["indiv_rewards"][worker_index]
				else:  # if "_indiv_rewards" does not exist in the info dict. very rare case (when all the parallel environments reset simultaneously) 
					assert info["did_reset"][worker_index]
					individual_rewards = info["last_info"]["indiv_rewards"][worker_index]
				episode_indiv_rewards[worker_index] = [r+individual_rewards[i] for i, r in enumerate(episode_indiv_rewards[worker_index])]

				if dones[worker_index] or self.worker_step_counter[worker_index] == self.max_time_steps:
					assert info["did_reset"][worker_index]

					flag = True
					episode_num = self.agents.buffer.worker_episode_counter[worker_index]
					time_step = self.agents.buffer.time_steps[worker_index]

					if episode_num >= self.agents.buffer.num_episodes:
						# print(f"skipping worker {worker_index} since it has collected more than needed")
						# the workers that have collected all required episodes for this update should not store anything more
						flag = False

					# the below condition might hold only when running train_parallel_agent_async.py 
					if time_step == 0 and self.worker_step_counter[worker_index] != 1:
						# assert masks == None
						# because of the above skip, after updation completes, it might be the case that the workers are somewhere in the middle of an ongoing episode
						# so we will just do nothing till that episode completes. After it completes, storing would resume.
						# print(f"skipping worker {worker_index} till it resets")
						flag = False
					
					if flag:
						self.num_episodes_done += 1

						step = self.worker_step_counter[worker_index]
						# print(f"Worker {worker_index} done!")
						if "_num_allies" in info: assert info["_num_allies"][worker_index] == False 
						if "_num_enemies" in info: assert info["_num_enemies"][worker_index] == False 
						print("*"*100)
						print(
							"EPISODE: {} | BUFFER EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} | INDIV REWARD STREAMS: {} \n".format(
								self.num_episodes_done,
								np.max(self.agents.buffer.worker_episode_counter)+1,
								np.round(episode_reward[worker_index], decimals=4),
								step,
								self.max_time_steps,
								episode_indiv_rewards[worker_index],
							)
						)
						if "StarCraft" in self.environment:
							print("Num Allies Alive: {} | Num Enemies Alive: {} \n".format(info["last_info"]["num_allies"][worker_index], info["last_info"]["num_enemies"][worker_index]))

						# if self.use_reward_model:
						# 	predicted_episode_reward = self.agents.reward_model_output()
							# print(action_list)

						if self.save_comet_ml_plot:
							self.comet_ml.log_metric('Episode_Length', step, self.num_episodes_done)
							self.comet_ml.log_metric('Reward', episode_reward[worker_index], self.num_episodes_done)
							if "StarCraft" in self.environment:
								self.comet_ml.log_metric('Num Enemies', info["last_info"]["num_enemies"][worker_index], self.num_episodes_done)
								self.comet_ml.log_metric('Num Allies', info["last_info"]["num_allies"][worker_index], self.num_episodes_done)
								self.comet_ml.log_metric('All Enemies Dead', info["last_info"]["all_enemies_dead"][worker_index], self.num_episodes_done)
								self.comet_ml.log_metric('All Allies Dead', info["last_info"]["all_allies_dead"][worker_index], self.num_episodes_done)
							elif self.environment == "GFootball":
								self.comet_ml.log_metric('Agents Done', dones[worker_index], self.num_episodes_done)

							# if self.use_reward_model:
							# 	self.comet_ml.log_metric('Predicted Reward', predicted_episode_reward, self.num_episodes_done)

						# update entropy params
						self.agents.update_parameters()

					if self.learn:
						local_obs_ = np.expand_dims(info["last_obs"][worker_index], axis=0)
						last_actions_ = np.expand_dims(info["__last_actions"][worker_index], axis=0)
						rnn_hidden_state_actor_ = np.expand_dims(info["__rnn_hidden_state_actor"][worker_index], axis=0)
						rnn_hidden_state_v_ = np.expand_dims(info["__rnn_hidden_state_v"][worker_index], axis=0)
						indiv_dones_ = np.expand_dims(info["last_info"]["indiv_dones"], axis=0)
						if "StarCraft" in self.environment:
							ally_states_ = np.expand_dims(info["last_info"]["ally_states"][worker_index], axis=0)
							enemy_states_ = np.expand_dims(info["last_info"]["enemy_states"][worker_index], axis=0)
							global_obs_ = None
							mask_actions_ = np.expand_dims(info["last_info"]["avail_actions"][worker_index], axis=0)
						elif self.environment == "GFootball":
							ally_states_, enemy_states_ = None, None
							global_obs_ = np.expand_dims(info["__global_obs"][worker_index], axis=0)
							mask_actions_ = np.expand_dims(info["__mask_actions"][worker_index], axis=0)
						


						# add final time to buffer
						actions, action_logprob, next_rnn_hidden_state_actor, latent_state_actor = self.agents.get_action_batch(local_obs_, last_actions_, mask_actions_, rnn_hidden_state_actor_)
						
						assert np.array(actions).shape == (1, self.num_agents)
						assert action_logprob.shape == (1, self.num_agents)
						assert next_rnn_hidden_state_actor.shape == (1, self.rnn_num_layers_actor, self.num_agents, self.rnn_hidden_actor)
						# actions -> list; np.array(actions).shape = (1, 5)
						# action_logprob.shape = (1, 5)
						# next_rnn_hidden_state_actor.shape = (1, rnn_num_layers_actor, 5, 64)

						value, _ = self.agents.get_values_batch(local_obs_, global_obs_, ally_states_, enemy_states_, actions, rnn_hidden_state_v_, indiv_dones_)
						
						assert value.shape == (1, self.num_agents)

						self.agents.buffer.end_episode(np.array([self.worker_step_counter[worker_index]]), value, latent_state_actor, np.array([indiv_dones[worker_index]]), np.array([dones[worker_index]]), np.array([worker_index]))

						if self.use_reward_model:
							self.agents.reward_buffer.end_episode(np.array([worker_index]))

					if flag:
						if self.agents.scheduler_need:
							self.agents.scheduler_policy.step()
							self.agents.scheduler_v_critic.step()
							if self.use_reward_model:
								self.agents.scheduler_reward.step()
							if self.agents.use_inverse_dynamics:
								self.agents.scheduler_inverse_dynamics.step()

						if self.eval_policy:
							self.rewards.append(episode_reward[worker_index])
							self.timesteps.append(step)

						if self.num_episodes_done > self.save_model_checkpoint and self.eval_policy:
							self.rewards_mean_per_1000_eps.append(sum(self.rewards[self.num_episodes_done-self.save_model_checkpoint:self.num_episodes_done])/self.save_model_checkpoint)
							self.timesteps_mean_per_1000_eps.append(sum(self.timesteps[self.num_episodes_done-self.save_model_checkpoint:self.num_episodes_done])/self.save_model_checkpoint)


						if not(self.num_episodes_done%self.save_model_checkpoint) and self.num_episodes_done!=0 and self.save_model:	
							# save actor, critic, reward and optims
							torch.save(self.agents.critic_network_v.state_dict(), self.critic_model_path+'_V_epsiode'+str(self.num_episodes_done)+'.pt')
							torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(self.num_episodes_done)+'.pt')  
							torch.save(self.agents.v_critic_optimizer.state_dict(), self.optim_model_path+'_critic_epsiode_'+str(self.num_episodes_done)+'.pt')
							torch.save(self.agents.policy_optimizer.state_dict(), self.optim_model_path+'_policy_epsiode_'+str(self.num_episodes_done)+'.pt')  

							if self.use_reward_model:
								torch.save(self.agents.reward_model.state_dict(), self.reward_model_path+'_epsiode_'+str(self.num_episodes_done)+'.pt')
								torch.save(self.agents.reward_optimizer.state_dict(), self.optim_model_path+'_reward_optim_epsiode_'+str(self.num_episodes_done)+'.pt')
				

						# update agent
						if self.learn and self.agents.should_update_agent(self.num_episodes_done) and not(self.update_agent):

							self.update_agent = True
							
							if self.experiment_type == "uniform_team_redistribution":
								b, t, n_a = self.agents.buffer.rewards.shape
								episodic_avg_reward = np.sum(self.agents.buffer.rewards[:, :, 0], axis=1)/self.agents.buffer.episode_length
								self.agents.buffer.rewards[:, :, :] = np.repeat(np.expand_dims(np.repeat(np.expand_dims(episodic_avg_reward, axis=-1), repeats=t, axis=-1), axis=-1), repeats=n_a, axis=-1)
								self.agents.buffer.rewards *= (1-self.agents.buffer.indiv_dones[:, :-1, :])
								self.agents.update(self.num_episodes_done)
							elif self.use_reward_model is False:
								self.agents.update(self.num_episodes_done)
							elif self.use_reward_model:
								if self.num_episodes_done >= self.warm_up_period:
									# finetune
									# sample = self.agents.buffer.sample_finetune_reward_model()
									# self.agents.update_reward_model(sample)
									
									self.agents.buffer.rewards, self.agents.buffer.action_prediction = self.agents.reward_model_output()
									self.agents.update(self.num_episodes_done)
								else:
									self.agents.buffer.clear()

						# update reward model
						if self.learn and self.use_reward_model and not(self.update_reward_model):

							if self.reward_batch_size <= self.agents.reward_buffer.episodes_filled and self.num_episodes_done != 0 and self.num_episodes_done % self.update_reward_model_freq == 0:
								
								self.update_reward_model = True

								reward_loss_batch, grad_norm_reward_batch = 0.0, 0.0
								if "AREL" in self.experiment_type:
									reward_var_batch = 0.0
								elif "TAR^2" in self.experiment_type:
									entropy_temporal_weights_batch, entropy_agent_weights_batch = 0.0, 0.0
									reward_prediction_loss_batch, dynamic_loss_batch = 0.0, 0.0
								
								for i in range(self.reward_model_update_epochs):

									print("reward model update", i)
									sample = self.agents.reward_buffer.sample_reward_model(num_episodes=self.reward_batch_size)
									if "AREL" in self.experiment_type:
										reward_loss, reward_var, grad_norm_value_reward = self.agents.update_reward_model(sample)
										reward_var_batch += (reward_var/self.reward_model_update_epochs)
									elif "TAR^2" in self.experiment_type:
										reward_loss, reward_prediction_loss, dynamic_loss, entropy_temporal_weights, entropy_agent_weights, grad_norm_value_reward = self.agents.update_reward_model(sample)
										entropy_temporal_weights_batch += (entropy_temporal_weights/self.reward_model_update_epochs)
										entropy_agent_weights_batch += (entropy_agent_weights/self.reward_model_update_epochs)
										reward_prediction_loss_batch += (reward_prediction_loss/self.reward_model_update_epochs)
										dynamic_loss_batch += (dynamic_loss/self.reward_model_update_epochs)
									elif "STAS" in self.experiment_type:
										reward_loss, grad_norm_value_reward = self.agents.update_reward_model(sample)

									reward_loss_batch += (reward_loss/self.reward_model_update_epochs)
									grad_norm_reward_batch += (grad_norm_value_reward/self.reward_model_update_epochs)

									if self.agents.scheduler_need:
										self.agents.scheduler_reward.step()

									torch.cuda.empty_cache()

								if self.comet_ml is not None:
									self.comet_ml.log_metric('Reward_Loss', reward_loss_batch, self.num_episodes_done)
									self.comet_ml.log_metric('Reward_Grad_Norm', grad_norm_reward_batch, self.num_episodes_done)

									if "AREL" in self.experiment_type:
										self.comet_ml.log_metric('Reward_Var', reward_var_batch, self.num_episodes_done)
									elif "TAR^2" in self.experiment_type:
										self.comet_ml.log_metric('Entropy_Temporal_Weights', entropy_temporal_weights_batch, self.num_episodes_done)
										self.comet_ml.log_metric('Entropy_Agent_Weights', entropy_agent_weights_batch, self.num_episodes_done)

										self.comet_ml.log_metric('Reward Prediction Loss', reward_prediction_loss_batch, self.num_episodes_done)
										self.comet_ml.log_metric('Reward Dynamic Loss', dynamic_loss_batch, self.num_episodes_done)
									
						

						if self.eval_policy and not(self.num_episodes_done%self.save_model_checkpoint) and self.num_episodes_done!=0:
							np.save(os.path.join(self.policy_eval_dir,self.test_num+"reward_list"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
							np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_rewards_per_1000_eps"), np.array(self.rewards_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
							np.save(os.path.join(self.policy_eval_dir,self.test_num+"timestep_list"), np.array(self.timesteps), allow_pickle=True, fix_imports=True)
							np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_timestep_per_1000_eps"), np.array(self.timesteps_mean_per_1000_eps), allow_pickle=True, fix_imports=True)


					episodic_team_reward[worker_index] = 0
					episodic_agent_reward[worker_index] = [0]*self.num_agents
					
					episode_reward[worker_index] = 0
					episode_indiv_rewards[worker_index] = [0]*self.num_agents

					last_actions[worker_index] = np.zeros((self.num_agents)) + self.num_actions
					indiv_dones[worker_index] = np.zeros((self.num_agents), dtype=np.int64)
					dones[worker_index] = 0

					rnn_hidden_state_v[worker_index] = np.zeros((self.rnn_num_layers_v, self.num_agents, self.rnn_hidden_v))
					rnn_hidden_state_actor[worker_index] = np.zeros((self.rnn_num_layers_actor, self.num_agents, self.rnn_hidden_actor))

					self.worker_step_counter[worker_index] = 0

					if self.update_reward_model:
						self.update_reward_model = False

					if self.update_agent:
						self.update_agent = False



if __name__ == '__main__':

	RENDER = False
	USE_CPP_RVO2 = False

	torch.set_printoptions(profile="full")
	torch.autograd.set_detect_anomaly(True)

	for i in range(1, 6):
		extension = "MAPPO_"+str(i)
		test_num = "Learning_Reward_Func_for_Credit_Assignment"
		environment = "GFootball" # StarCraft/ GFootball
		env_name = "academy_3_vs_1_with_keeper" # 5m_vs_6m, 10m_vs_11m, 3s5z/ academy_3_vs_1_with_keeper, academy_counterattack_easy, academy_pass_and_shoot_with_keeper, academy_counterattack_hard, academy_cornery, academy_run_and_pass_with_keeper, academy_run_pass_and_shoot_with_keeper
		experiment_type = "TAR^2_v2" # episodic_team, episodic_agent, temporal_team, temporal_agent, uniform_team_redistribution, AREL, STAS, TAR^2, TAR^2_v2, TAR^2_HindSight
		experiment_name = "MAPPO_TAR^2_v2_normal_inverse_dynamics_model_w_curr_future_state_past_intermediat_emb_w_final_state_reward_pred" # default setting: reward prediction loss + dynamic loss
		algorithm_type = "MAPPO"

		dictionary = {
				# TRAINING
				"iteration": i,
				"device": "gpu",
				"critic_dir": '../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
				"actor_dir": '../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
				"optim_dir": '../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/optimizers/',
				"reward_dir": '../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/reward_network/',
				"gif_dir": '../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_eval_dir":'../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"n_epochs": 15,
				"ppo_eps_elapse_update_freq": 30, # update ppo agent after every ppo_eps_elapse_update_freq episodes; 10 (StarCraft/MPE/PressurePlate/LBF)/ 5 (PettingZoo)
				"environment": environment,
				"experiment_name": experiment_name,
				"test_num": test_num,
				"extension": extension,
				"gamma": 0.99,
				"gif": False,
				"gif_checkpoint":1,
				"load_models": False,
				"model_path_v_value": "../../tests/RLC_2024/relevant_set_visualization/crossing_team_greedy/prd_soft_advantage/models/crossing_team_greedy_prd_soft_advantage_MAPPO_1/critic_networks/critic_V_epsiode10000.pt",
				"model_path_policy": "../../tests/RLC_2024/relevant_set_visualization/crossing_team_greedy/prd_soft_advantage/models/crossing_team_greedy_prd_soft_advantage_MAPPO_1/actor_networks/actor_epsiode10000.pt",
				"eval_policy": False,
				"save_model": False,
				"save_model_checkpoint": 1000,
				"save_comet_ml_plot": True,
				"learn":True,
				"max_episodes": 30000, # 30000 (StarCraft environments)/ 50000 (GFootball)
				"max_time_steps": 100, # 50 (StarCraft environments -- 100 for 3s5z)/ 100 (GFootball -- entropy: 4e-3 3v1/ 1e-2 pass_&_shoot/ 2e-3 ca_easy)
				"experiment_type": experiment_type,
				"parallel_training": True,
				"num_workers": 5,
				"scheduler_need": False,
				"norm_rewards": False,
				"clamp_rewards": False,
				"clamp_rewards_value_min": 0.0,
				"clamp_rewards_value_max": 2.0,
				"warm_up_period": 200, # 200


				# REWARD MODEL
				"use_reward_model": True,
				"reward_n_heads": 4, # 3
				"reward_depth": 3, # 3
				"reward_agent_attn": True,
				"reward_dropout": 0.0,
				"reward_attn_net_wide": True,
				"version": "agent_temporal", # temporal, agent_temporal ---- For AREL
				"reward_linear_compression_dim": 64, # 16 for TAR^2_agent_temporal
				"reward_batch_size": 128, # 64
				"reward_lr": 5e-4,
				"reward_weight_decay": 0.0,
				"dynamic_loss_coeffecient": 5e-1,
				"expected_logprob_prediction_loss_coeffecient": 5e-2,
				"temporal_score_coefficient": 0.0,
				"agent_score_coefficient": 0.0,
				"variance_loss_coeff": 0.0,
				"enable_reward_grad_clip": True,
				"reward_grad_clip_value": 10.0,
				"replay_buffer_size": 5000,
				"update_reward_model_freq": 200, # 100
				"reward_model_update_epochs": 200, # 200
				"norm_rewards": False,

				"use_inverse_dynamics": True,
				"inverse_dynamics_lr": 5e-4,
				"inverse_dynamics_weight_decay": 1e-5,
				"enable_grad_clip_inverse_dynamics": True,
				"grad_clip_inverse_dynamics": 0.5,


				"algorithm_type": algorithm_type,


				# ENVIRONMENT
				"env": env_name,

				# CRITIC
				"use_recurrent_critic": True,
				"rnn_num_layers_v": 1,
				"rnn_hidden_v": 64,
				"v_value_lr": 5e-4, #1e-3
				"v_weight_decay": 0.0,
				"v_comp_emb_shape": 64,
				"enable_grad_clip_critic_v": True,
				"grad_clip_critic_v": 0.5,
				"value_clip": 0.2,
				"enable_hard_attention": False,
				"target_calc_style": "GAE", # GAE, N_steps
				"n_steps": 5,
				"norm_returns_v": True,
				"soft_update_v": False,
				"tau_v": 0.05,
				"network_update_interval_v": 1,
				

				# ACTOR
				"use_recurrent_policy": True,
				"data_chunk_length": 10,
				"rnn_num_layers_actor": 1,
				"rnn_hidden_actor": 64,
				"enable_grad_clip_actor": True,
				"grad_clip_actor": 0.5,
				"policy_clip": 0.2,
				"policy_lr": 5e-4, # prd 1e-4
				"policy_weight_decay": 0.0,
				"entropy_pen": 4e-3, # 8e-3
				"entropy_pen_final": 4e-3,
				"entropy_pen_steps": 20000,
				"gae_lambda": 0.95,
				"norm_adv": True,
			}

		seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(seeds[dictionary["iteration"]-1])
		
		if "StarCraft" in dictionary["environment"]:
			import gym
			import smaclite  # noqa
			
			# env = gym.make(f"smaclite/{env_name}-v0", use_cpp_rvo2=USE_CPP_RVO2)
			env = [lambda: gym.make(f"smaclite/{env_name}-v0", seed=seeds[i-1] + 1000*j, use_cpp_rvo2=USE_CPP_RVO2) for j in range(dictionary["num_workers"])]
			obs, info = env[0]().reset(return_info=True)
			dictionary["ally_observation_shape"] = info["ally_states"][0].shape[0]
			dictionary["enemy_observation_shape"] = info["enemy_states"][0].shape[0]
			dictionary["local_observation_shape"] = obs[0].shape[0]
			dictionary["num_agents"] = env[0]().n_agents
			dictionary["num_enemies"] = env[0]().n_enemies
			dictionary["num_actions"] = env[0]().action_space[0].n
		
		elif "GFootball" in dictionary["environment"]:
			import random

			import gfootball.env as football_env
			from gym import spaces
			import numpy as np


			class FootballEnv(object):
				'''Wrapper to make Google Research Football environment compatible'''

				def __init__(self, seed):

					self.seed(seed)

					self.scenario_name = env_name

					if self.scenario_name == "academy_pass_and_shoot_with_keeper":
						'''
						num_env_steps=25000000
						episode_length=200
						'''
						self.num_agents = 2

					elif self.scenario_name == "academy_3_vs_1_with_keeper":
						'''
						num_env_steps=25000000
						episode_length=200
						'''
						self.num_agents = 3
					elif self.scenario_name in ["academy_counterattack_easy", "academy_counterattack_hard"]:
						'''
						num_env_steps=25000000
						episode_length=200
						'''
						self.num_agents = 4
					elif self.scenario_name == "academy_corner":
						'''
						num_env_steps=50000000
						episode_length=1000
						'''
						self.num_agents = 10
					elif self.scenario_name in ["academy_run_and_pass_with_keeper", "academy_run_pass_and_shoot_with_keeper"]:
						'''
						num_env_steps=25000000
						episode_length=200
						'''
						self.num_agents = 2

					self.env = football_env.create_environment(
					  env_name=self.scenario_name,
					  stacked=False,
					  representation="simple115v2",
					  rewards="scoring,checkpoints",
					  number_of_left_players_agent_controls=self.num_agents,
					  number_of_right_players_agent_controls=0,
					  channel_dimensions=(96, 72),
					  render=(False and False)
					)
						
					self.max_steps = self.env.unwrapped.observation()[0]["steps_left"]
					self.remove_redundancy = False
					self.zero_feature = False
					self.share_reward = True
					self.action_space = []
					self.observation_space = []
					self.share_observation_space = []

					if self.num_agents == 1:
						self.action_space.append(self.env.action_space)
						self.observation_space.append(self.env.observation_space)
						self.share_observation_space.append(self.env.observation_space)
					else:
						for idx in range(self.num_agents):
							self.action_space.append(spaces.Discrete(
								n=self.env.action_space[idx].n
							))
							self.observation_space.append(spaces.Box(
								low=self.env.observation_space.low[idx],
								high=self.env.observation_space.high[idx],
								shape=self.env.observation_space.shape[1:],
								dtype=self.env.observation_space.dtype
							))
							self.share_observation_space.append(spaces.Box(
								low=self.env.observation_space.low[idx],
								high=self.env.observation_space.high[idx],
								shape=self.env.observation_space.shape[1:],
								dtype=self.env.observation_space.dtype
							))

						self.indiv_agent_observation_shape = 4 # pose + direction
						self.common_information_observation_shape = 115 - self.num_agents*self.indiv_agent_observation_shape - 11 # 11 -- one hot for active players


				def process_indiv_agent_obs(self, obs):
					'''
					https://github.com/google-research/football/blob/master/gfootball/doc/observation.md

					simple115_v2:-
					Same as simple115, but with the bug fixed.

					    22 - (x,y) coordinates of left team players (COMMON)
					    22 - (x,y) direction of left team players (COMMON)
					    22 - (x,y) coordinates of right team players (COMMON)
					    22 - (x, y) direction of right team players (COMMON)
					    3 - (x, y and z) - ball position (COMMON)
					    3 - ball direction (COMMON)
					    3 - one hot encoding of ball ownership (noone, left, right) (COMMON)
					    11 - one hot encoding of which player is active
					    7 - one hot encoding of game_mode (COMMON)

					Entries for players that are not active (either due to red cards or if number of player is less than 11) are set to -1.
					'''

					obs = np.array(obs)

					player_positions = np.reshape(obs[0, :22], (11, 2))
					player_directions = np.reshape(obs[0, 22:44], (11, 2))
					opponent_positions = obs[0, 44:66]
					opponent_directions = obs[0, 66:88]
					ball_position = np.reshape(obs[0, 88:91], (3))
					ball_direction = np.reshape(obs[0, 91:94], (3))
					ball_ownership = np.reshape(obs[0, 94:97], (3))
					active_players = np.reshape(obs[:, 97:108], (self.num_agents, 11))
					players_active_inactive = np.sum(active_players, axis=1)
					active_players = np.argmax(active_players, axis=1).reshape((self.num_agents))
					game_mode = np.reshape(obs[0, 108:], (7))

					player_observations = []
					indiv_dones = []
					for player_id, player_active in zip(active_players, players_active_inactive):
						if player_active:
							player_obs = [player_positions[player_id][0], player_positions[player_id][1], player_directions[player_id][0], player_directions[player_id][1]]
							indiv_dones.append(0)
						else:
							player_obs = [0, 0, 0, 0]
							indiv_dones.append(1)

						player_observations.append(player_obs)

					common_information = []
					for player_id in range(11):
						if player_id in active_players:
							continue

						common_information.extend([player_positions[player_id][0], player_positions[player_id][1], player_directions[player_id][0], player_directions[player_id][1]])

					common_information.extend(opponent_positions)
					common_information.extend(opponent_directions)
					common_information.extend(ball_position)
					common_information.extend(ball_direction)
					common_information.extend(ball_ownership)
					common_information.extend(game_mode)

					player_observations, common_information, indiv_dones = np.array(player_observations), np.array(common_information), np.array(indiv_dones)

					return player_observations, common_information, indiv_dones


				def reset(self):
					obs = self.env.reset()
					obs, _, _, info = self.env.step([0]*self.num_agents) # we want to mine information from info dict -- wrong approach but can do for now
					obs = self._obs_wrapper(obs)

					# info = {}
					player_observations, common_information, indiv_dones = self.process_indiv_agent_obs(obs)
					info["player_observations"] = player_observations
					info["common_information"] = common_information
					info["indiv_dones"] = indiv_dones

					info = self._info_wrapper(info)

					# ['ball_direction', 'ball', 'left_team_roles', 'right_team', 'left_team_direction', 'score', 'right_team_active', 'right_team_direction', 'right_team_yellow_card', 'right_team_tired_factor', 'left_team_tired_factor', 'left_team_active', 'ball_rotation', 'right_team_roles', 'left_team', 'designated', 'active', 'sticky_actions', 'max_steps']
					# print(info.keys())
					# print("BALL OWNED TEAM")
					# print(info["ball_owned_team"])
					# print("BALL OWNED PLAYER")
					# print(info["ball_owned_player"])
					# print("BALL DIRECTION")
					# print(info["ball_direction"])
					# print("LEFT TEAM YELLOW")
					# print(info["left_team_yellow_card"])



					return obs, info

				def step(self, action):
					obs, reward, done, info = self.env.step(action)
					obs = self._obs_wrapper(obs)
					reward = reward.reshape(self.num_agents)
					info["indiv_rewards"] = np.array(reward)
					if self.share_reward:
						# global_reward = np.sum(reward)
						# reward = [[global_reward]] * self.num_agents
						reward = np.sum(reward)

					# done = np.array([done] * self.num_agents)
					# info["indiv_dones"] = np.array([done] * self.num_agents)
					info = self._info_wrapper(info)

					player_observations, common_information, indiv_dones = self.process_indiv_agent_obs(obs)
					info["player_observations"] = player_observations
					info["common_information"] = common_information
					info["indiv_dones"] = indiv_dones # the indiv dones only capture red card
					if done:
						info["indiv_dones"] = np.array([done] * self.num_agents)

					# print("*"*20)
					# print(info.keys())

					return obs, reward, done, info

				def seed(self, seed=None):
					if seed is None:
						random.seed(1)
					else:
						random.seed(seed)

				def close(self):
					self.env.close()

				def _obs_wrapper(self, obs):
					if self.num_agents == 1:
						return obs[np.newaxis, :]
					else:
						return obs

				def _info_wrapper(self, info):
					state = self.env.unwrapped.observation()
					info.update(state[0])
					info["max_steps"] = self.max_steps
					info["active"] = np.array([state[i]["active"] for i in range(self.num_agents)])
					info["designated"] = np.array([state[i]["designated"] for i in range(self.num_agents)])
					info["sticky_actions"] = np.stack([state[i]["sticky_actions"] for i in range(self.num_agents)])

					return info


			env = [lambda: FootballEnv(seeds[i-1] + j*1000) for j in range(dictionary["num_workers"])]

			dictionary["num_agents"] = env[0]().num_agents
			dictionary["local_observation_shape"] = env[0]().observation_space[0].shape[0]
			dictionary["global_observation_shape"] = env[0]().observation_space[0].shape[0]
			dictionary["ally_observation_shape"] = env[0]().indiv_agent_observation_shape
			dictionary["common_information_observation_shape"] = env[0]().common_information_observation_shape
			dictionary["num_actions"] = env[0]().action_space[0].n
			

		# torch.set_num_threads(16)
		ma_controller = MAPPO(env, dictionary)
		ma_controller.run()





# visa@manchester.ac.uk --- For VISA || international@manchester.ac.uk