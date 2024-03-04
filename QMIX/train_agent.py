import os
import time
from comet_ml import Experiment
import numpy as np
from agent import QMIXAgent
from buffer import ReplayMemory
from utils import soft_update, hard_update

import torch
import datetime

import gym
import smaclite  # noqa

torch.autograd.set_detect_anomaly(True)


class QMIX:

	def __init__(self, env, dictionary):

		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"
		self.env = env
		self.experiment_type = dictionary["experiment_type"]
		self.gif = dictionary["gif"]
		self.save_model = dictionary["save_model"]
		self.save_model_checkpoint = dictionary["save_model_checkpoint"]
		self.save_comet_ml_plot = dictionary["save_comet_ml_plot"]
		self.learn = dictionary["learn"]
		self.gif_checkpoint = dictionary["gif_checkpoint"]
		self.eval_policy = dictionary["eval_policy"]
		self.num_agents = dictionary["num_agents"]
		self.num_actions = self.env.action_space[0].n
		self.date_time = f"{datetime.datetime.now():%d-%m-%Y}"
		self.env_name = dictionary["env"]
		self.test_num = dictionary["test_num"]
		self.max_episodes = dictionary["max_episodes"]
		self.max_time_steps = dictionary["max_time_steps"]


		self.update_episode_interval = dictionary["update_episode_interval"]
		self.num_updates = dictionary["num_updates"]
		self.scheduler_need = dictionary["scheduler_need"]
		self.soft_update = dictionary["soft_update"]
		self.target_update_interval = dictionary["target_update_interval"]
		self.use_reward_model = dictionary["use_reward_model"]

		self.q_observation_shape = dictionary["q_observation_shape"]
		self.q_mix_observation_shape = dictionary["q_mix_observation_shape"]
		self.replay_buffer_size = dictionary["replay_buffer_size"]
		self.batch_size = dictionary["batch_size"] # number of datapoints to sample
		self.buffer = ReplayMemory(
			capacity = self.replay_buffer_size,
			max_episode_len = self.max_time_steps,
			num_agents = self.num_agents,
			q_obs_shape = self.q_observation_shape,
			q_mix_obs_shape=self.q_mix_observation_shape,
			action_shape = self.num_actions
			)

		self.epsilon_greedy = dictionary["epsilon_greedy"]
		self.epsilon_greedy_min = dictionary["epsilon_greedy_min"]
		self.epsilon_decay_rate = (self.epsilon_greedy - self.epsilon_greedy_min) / dictionary["epsilon_greedy_decay_episodes"]

		self.reward_batch_size = dictionary["reward_batch_size"]
		self.update_reward_model_freq = dictionary["update_reward_model_freq"]
		self.reward_model_update_epochs = dictionary["reward_model_update_epochs"]

		self.comet_ml = None
		if self.save_comet_ml_plot:
			self.comet_ml = Experiment("im5zK8gFkz6j07uflhc3hXk8I",project_name=dictionary["test_num"])
			self.comet_ml.log_parameters(dictionary)


		self.agent_ids = []
		for i in range(self.num_agents):
			agent_id = np.array([0 for i in range(self.num_agents)])
			agent_id[i] = 1
			self.agent_ids.append(agent_id)
		self.agent_ids = np.array(self.agent_ids)

		self.agents = QMIXAgent(self.env, dictionary)

		if self.save_model:
			model_dir = dictionary["model_dir"]
			try: 
				os.makedirs(model_dir, exist_ok = True) 
				print("Model Directory created successfully") 
			except OSError as error: 
				print("Model Directory cannot be created") 
			

			
			self.model_path = model_dir+"model"
			

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


	def run(self):  
		if self.eval_policy:
			self.rewards = []
			self.rewards_mean_per_1000_eps = []
			self.timesteps = []
			self.timesteps_mean_per_1000_eps = []

		for episode in range(1, self.max_episodes+1):

			states, info = self.env.reset(return_info=True)
			states = np.array(states)
			states = np.concatenate((self.agent_ids, states), axis=-1)
			globa_states_allies = np.concatenate((self.agent_ids, info["ally_states"]), axis=-1).reshape(-1)
			global_states_enemies = info["enemy_states"].reshape(-1)
			global_states = np.concatenate((globa_states_allies, global_states_enemies), axis=-1)
			mask_actions = np.array(info["avail_actions"], dtype=int)
			last_one_hot_action = np.zeros((self.num_agents, self.num_actions))
			indiv_dones = [0]*self.num_agents
			indiv_dones = np.array(indiv_dones)

			images = []

			episodic_team_reward = 0

			episode_reward = 0
			final_timestep = self.max_time_steps

			self.agents.Q_network.rnn_hidden_state = None
			self.agents.target_Q_network.rnn_hidden_state = None

			for step in range(1, self.max_time_steps+1):

				if self.gif:
					# At each step, append an image to list
					# if not(episode%self.gif_checkpoint):
						# images.append(np.squeeze(self.env.render(mode='rgb_array')))
					self.env.render()
					# import time
					# time.sleep(0.1)
					# Advance a step and render a new image
					with torch.no_grad():
						actions = self.agents.get_action(states, last_one_hot_action, self.epsilon_greedy, mask_actions)
				else:
					actions = self.agents.get_action(states, last_one_hot_action, self.epsilon_greedy, mask_actions)

				next_last_one_hot_action = np.zeros((self.num_agents,self.num_actions))
				for i,act in enumerate(actions):
					next_last_one_hot_action[i][act] = 1

				next_states, rewards, dones, info = self.env.step(actions)
				next_states = np.array(next_states)
				next_states = np.concatenate((self.agent_ids, next_states), axis=-1)
				next_globa_states_allies = np.concatenate((self.agent_ids, info["ally_states"]), axis=-1).reshape(-1)
				next_global_states_enemies = info["enemy_states"].reshape(-1)
				next_global_states = np.concatenate((next_globa_states_allies, next_global_states_enemies), axis=-1)
				next_mask_actions = np.array(info["avail_actions"], dtype=int)
				next_indiv_dones = info["indiv_dones"]

				if self.learn:
					# can't give agent level rewards since the algorithm cannot make use of it
					if self.experiment_type == "temporal_team":
						rewards_to_send = rewards
						# rewards_to_send = [rewards if indiv_dones[i]==0 else 0 for i in range(self.num_agents)]
					elif self.experiment_type == "episodic_team":
						episodic_team_reward = episodic_team_reward+rewards
						if all(next_indiv_dones) or step == self.max_time_steps:
							rewards_to_send = episodic_team_reward
						else:
							rewards_to_send = 0
					elif self.experiment_type == "AREL" or "ATRR" in self.experiment_type:
						episodic_team_reward = episodic_team_reward+rewards
						if all(next_indiv_dones) or step == self.max_time_steps:
							rewards_to_send = episodic_team_reward
						else:
							rewards_to_send = 0

					self.buffer.push(states, global_states, actions, last_one_hot_action, next_states, next_global_states, next_last_one_hot_action, next_mask_actions, rewards_to_send, dones, indiv_dones)

				states, global_states, mask_actions, last_one_hot_action, indiv_dones = next_states, next_global_states, next_mask_actions, next_last_one_hot_action, next_indiv_dones

				episode_reward += np.sum(rewards)

				if dones or step == self.max_time_steps:

					print("*"*100)
					print("EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} | Num Allies Alive: {} | Num Enemies Alive: {} \n".format(episode, np.round(episode_reward,decimals=4), step, self.max_time_steps, info["num_allies"], info["num_enemies"]))
					print("*"*100)

					final_timestep = step

					if self.save_comet_ml_plot:
						self.comet_ml.log_metric('Episode_Length', step, episode)
						self.comet_ml.log_metric('Reward', episode_reward, episode)
						self.comet_ml.log_metric('Num Enemies', info["num_enemies"], episode)
						self.comet_ml.log_metric('Num Allies', info["num_allies"], episode)
						self.comet_ml.log_metric('All Enemies Dead', info["all_enemies_dead"], episode)
						self.comet_ml.log_metric('All Allies Dead', info["all_allies_dead"], episode)

					break

			self.epsilon_greedy = self.epsilon_greedy - self.epsilon_decay_rate if self.epsilon_greedy - self.epsilon_decay_rate > self.epsilon_greedy_min else self.epsilon_greedy_min
			self.buffer.end_episode()

			if self.learn and self.use_reward_model and self.reward_batch_size <= self.buffer.length and episode != 0 and episode % self.update_reward_model_freq == 0:
				reward_loss_batch, grad_norm_reward_batch = 0.0, 0.0
				if self.experiment_type == "AREL":
					reward_var_batch = 0.0
				elif "ATRR" in self.experiment_type:
					entropy_temporal_weights_batch, entropy_agent_weights_batch = 0.0, 0.0
				for i in range(self.reward_model_update_epochs):
					sample = self.buffer.sample(num_episodes=self.reward_batch_size)
					if self.experiment_type == "AREL":
						reward_loss, reward_var, grad_norm_value_reward = self.agents.update_reward_model(sample)
						reward_var_batch += (reward_var/self.reward_model_update_epochs)
					elif "ATRR" in self.experiment_type:
						reward_loss, entropy_temporal_weights, entropy_agent_weights, grad_norm_value_reward = self.agents.update_reward_model(sample)
						entropy_temporal_weights_batch += (entropy_temporal_weights/self.reward_model_update_epochs)
						entropy_agent_weights_batch += (entropy_agent_weights/self.reward_model_update_epochs)

					reward_loss_batch += (reward_loss/self.reward_model_update_epochs)
					grad_norm_reward_batch += (grad_norm_value_reward/self.reward_model_update_epochs)

					if self.scheduler_need:
						self.scheduler_reward.step()

				if self.comet_ml is not None:
					self.comet_ml.log_metric('Reward_Loss', reward_loss_batch, episode)
					self.comet_ml.log_metric('Reward_Grad_Norm', grad_norm_reward_batch, episode)

					if self.experiment_type == "AREL":
						self.comet_ml.log_metric('Reward_Var', reward_var_batch, episode)
					elif "ATRR" in self.experiment_type:
						self.comet_ml.log_metric('Entropy_Temporal_Weights', entropy_temporal_weights_batch, episode)
						self.comet_ml.log_metric('Entropy_Agent_Weights', entropy_agent_weights_batch, episode)


			if self.learn and self.batch_size <= self.buffer.length and episode != 0 and episode%self.update_episode_interval == 0:
				Q_loss_batch, grad_norm_Q_batch = 0.0, 0.0
				
				for i in range(self.num_updates):
					sample = self.buffer.sample(num_episodes=self.batch_size)
					
					Q_loss, grad_norm_Q = self.agents.update(sample)
					
					Q_loss_batch += (Q_loss/self.num_updates)
					grad_norm_Q_batch += (grad_norm_Q/self.num_updates)

				if self.scheduler_need:
					self.agents.scheduler.step()

				if self.soft_update:
					soft_update(self.agents.target_Q_network, self.agents.Q_network, self.tau)
					soft_update(self.agents.target_QMix_network, self.agents.QMix_network, self.tau)
				else:
					if episode % self.target_update_interval == 0:
						hard_update(self.agents.target_Q_network, self.agents.Q_network)
						hard_update(self.agents.target_QMix_network, self.agents.QMix_network)

				if self.comet_ml is not None:
					self.comet_ml.log_metric('Q_Loss', Q_loss_batch, episode)
					self.comet_ml.log_metric('Q_Grad_Norm', grad_norm_Q_batch, episode)


			if self.eval_policy:
				self.rewards.append(episode_reward)
				self.timesteps.append(final_timestep)

			if episode > self.save_model_checkpoint and self.eval_policy:
				self.rewards_mean_per_1000_eps.append(sum(self.rewards[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.timesteps_mean_per_1000_eps.append(sum(self.timesteps[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)

			if not(episode%self.save_model_checkpoint) and episode!=0 and self.save_model:	
				torch.save(self.agents.Q_network.state_dict(), self.model_path+'_Q_epsiode_'+str(episode)+'.pt')
				torch.save(self.agents.QMix_network.state_dict(), self.model_path+'_QMix_epsiode_'+str(episode)+'.pt')

			
			# elif self.gif and not(episode%self.gif_checkpoint):
			# 	print("GENERATING GIF")
			# 	self.make_gif(np.array(images),self.gif_path)

			if self.eval_policy and not(episode%self.save_model_checkpoint) and episode!=0:
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"reward_list"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_rewards_per_1000_eps"), np.array(self.rewards_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"timestep_list"), np.array(self.timesteps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_timestep_per_1000_eps"), np.array(self.timesteps_mean_per_1000_eps), allow_pickle=True, fix_imports=True)

if __name__ == '__main__':

	RENDER = False
	USE_CPP_RVO2 = False

	torch.autograd.set_detect_anomaly(True)

	for i in range(1,6):
		extension = "QMix_"+str(i)
		test_num = "Learning_Reward_Func_for_Credit_Assignment"
		env_name = "5m_vs_6m"
		experiment_type = "ATRR_temporal" # episodic_team, episodic_agent, temporal_team, temporal_agent, AREL, ATRR_temporal, ATRR_agent, SeqModel, RUDDER

		dictionary = {
				# TRAINING
				"iteration": i,
				"device": "gpu",
				"model_dir": '../../tests/'+test_num+'/models/'+env_name+'_'+extension+'/models/',
				"gif_dir": '../../tests/'+test_num+'/gifs/'+env_name+'_'+'_'+extension+'/',
				"policy_eval_dir":'../../tests/'+test_num+'/policy_eval/'+env_name+'_'+extension+'/',
				"test_num":test_num,
				"extension":extension,
				"experiment_type": experiment_type,
				"gif": False,
				"gif_checkpoint":1,
				"load_models": False,
				"model_path": "../../tests/PRD_2_MPE/models/crossing_team_greedy_prd_above_threshold_MAPPO_Q_run_2/critic_networks/critic_epsiode100000.pt",
				"eval_policy": True,
				"save_model": True,
				"save_model_checkpoint": 1000,
				"save_comet_ml_plot": True,
				"norm_returns": False,
				"learn":True,
				"parallel_training": False,
				"scheduler_need": False,
				"max_episodes": 50000,
				"max_time_steps": 50,
				"gamma": 0.99,
				"replay_buffer_size": 5000,
				"batch_size": 32,
				"update_episode_interval": 1,
				"num_updates": 1,
				"epsilon_greedy": 0.8,
				"epsilon_greedy_min": 0.05,
				"epsilon_greedy_decay_episodes": 50000,
				"lambda": 0.8,

				# REWARD MODEL
				"use_reward_model": True,
				"reward_n_heads": 3, # 3
				"reward_depth": 3, # 3
				"reward_agent_attn": True,
				"reward_dropout": 0.0,
				"reward_attn_net_wide": True,
				"reward_comp": "linear_compression", # no_compression, linear_compression, hypernet_compression
				"reward_linear_compression_dim": 64,
				"reward_hypernet_hidden_dim": 64,
				"reward_hypernet_final_dim": 64,
				# "num_episodes_capacity": 2000, # 40000
				"reward_batch_size": 64, # 128
				"reward_lr": 1e-4,
				"reward_weight_decay": 5e-4,
				"temporal_score_efficient": 0.0,
				"agent_score_efficient": 0.0,
				"variance_loss_coeff": 0.0,
				"enable_reward_grad_clip": True,
				"reward_grad_clip_value": 5.0,
				# "reward_warmup": 5000, # 1000
				"update_reward_model_freq": 200, # 200
				"reward_model_update_epochs": 400, # 400
				"norm_rewards": False,
				"clamp_rewards": False,
				"clamp_rewards_value_min": 0.0,
				"clamp_rewards_value_max": 2.0,

				# ENVIRONMENT
				"env": env_name,

				# MODEL
				"learning_rate": 5e-4, #1e-3
				"enable_grad_clip": True,
				"grad_clip": 0.5,
				"rnn_hidden_dim": 64,
				"hidden_dim": 64,
				"norm_returns": False,
				"soft_update": False,
				"tau": 0.001,
				"target_update_interval": 200,
			}

		seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(seeds[dictionary["iteration"]-1])
		env = gym.make(f"smaclite/{env_name}-v0", use_cpp_rvo2=USE_CPP_RVO2)
		obs, info = env.reset(return_info=True)
		dictionary["num_agents"] = env.n_agents
		dictionary["q_observation_shape"] = env.n_agents+obs[0].shape[0]
		dictionary["q_mix_observation_shape"] = env.n_agents*(env.n_agents+info["ally_states"].shape[1])+env.n_enemies*info["enemy_states"].shape[1]
		ma_controller = QMIX(env, dictionary)
		ma_controller.run()