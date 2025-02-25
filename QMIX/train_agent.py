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

		self.algorithm_type = dictionary["algorithm_type"]
		self.q_observation_shape = dictionary["q_observation_shape"]
		self.q_mix_observation_shape = dictionary["q_mix_observation_shape"]
		self.data_chunk_length = dictionary["data_chunk_length"]
		self.rnn_hidden_dim = dictionary["rnn_hidden_dim"]
		self.rnn_num_layers = dictionary["rnn_num_layers"]
		self.replay_buffer_size = dictionary["replay_buffer_size"]
		self.batch_size = dictionary["batch_size"] # number of datapoints to sample

		self.buffer = ReplayMemory(
			algorithm_type = self.algorithm_type,
			experiment_type = self.experiment_type,
			capacity = self.replay_buffer_size,
			max_episode_len = self.max_time_steps,
			num_agents = self.num_agents,
			q_obs_shape = self.q_observation_shape,
			q_mix_obs_shape = self.q_mix_observation_shape,
			rnn_num_layers = self.rnn_num_layers,
			rnn_hidden_state_shape = self.rnn_hidden_dim,
			reward_model_obs_shape = dictionary["reward_model_obs_shape"],
			data_chunk_length = self.data_chunk_length,
			action_shape = self.num_actions,
			gamma = dictionary["gamma"],
			lambda_ = dictionary["lambda"],
			device = self.device,
			)

		self.epsilon_greedy = dictionary["epsilon_greedy"]
		self.epsilon_greedy_min = dictionary["epsilon_greedy_min"]
		self.epsilon_greedy_decay_episodes = dictionary["epsilon_greedy_decay_episodes"]
		self.epsilon_decay_rate = (self.epsilon_greedy - self.epsilon_greedy_min) / self.epsilon_greedy_decay_episodes

		self.reward_batch_size = dictionary["reward_batch_size"]
		self.update_reward_model_freq = dictionary["update_reward_model_freq"]
		self.reward_model_update_epochs = dictionary["reward_model_update_epochs"]

		self.comet_ml = None
		if self.save_comet_ml_plot:
			self.comet_ml = Experiment("im5zK8gFkz6j07uflhc3hXk8I",project_name=dictionary["test_num"])
			self.comet_ml.set_name(dictionary["experiment_name"])
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


	def run(self):  
		if self.eval_policy:
			self.rewards = []
			self.rewards_mean_per_1000_eps = []
			self.timesteps = []
			self.timesteps_mean_per_1000_eps = []

		for episode in range(1, self.max_episodes+1):

			states, info = self.env.reset(return_info=True)
			mask_actions = np.array(info["avail_actions"]) #(np.array(info["avail_actions"]) - 1) * 1e5
			last_one_hot_action = np.zeros((self.num_agents, self.num_actions))
			states = np.array(states)
			states = np.concatenate((self.agent_ids, states), axis=-1)
			states_allies = np.concatenate((self.agent_ids, info["ally_states"]), axis=-1).reshape(-1)
			states_enemies = np.array(info["enemy_states"]).reshape(-1)
			full_state = np.concatenate((states_allies, states_enemies), axis=-1)
			ally_states = np.concatenate((self.agent_ids, info["ally_states"]), axis=-1)
			enemy_states = np.repeat(np.expand_dims(states_enemies, axis=0), repeats=self.num_agents, axis=0)
			reward_model_obs = np.concatenate((ally_states, enemy_states), axis=-1)
			indiv_dones = [0]*self.num_agents
			indiv_dones = np.array(indiv_dones)
			dones = all(indiv_dones)


			# for recording data with gif
			images = []
			action_list = []
			enemy_action_list = []
			num_allies_alive = []
			num_enemies_alive = []
			true_rewards = []

			episodic_team_reward = 0

			episode_reward = 0
			final_timestep = self.max_time_steps

			rnn_hidden_state = np.zeros((self.rnn_num_layers, self.num_agents, self.rnn_hidden_dim))

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
						actions, next_rnn_hidden_state = self.agents.get_action(states, last_one_hot_action, rnn_hidden_state, 0.05, mask_actions)
						action_list.append(actions)
				else:
					actions, next_rnn_hidden_state = self.agents.get_action(states, last_one_hot_action, rnn_hidden_state, self.epsilon_greedy, mask_actions)

				next_last_one_hot_action = np.zeros((self.num_agents,self.num_actions))
				for i,act in enumerate(actions):
					next_last_one_hot_action[i][act] = 1

				next_states, rewards, next_dones, info = self.env.step(actions)
				next_states = np.array(next_states)
				next_states = np.concatenate((self.agent_ids, next_states), axis=-1)
				next_states_allies = np.concatenate((self.agent_ids, info["ally_states"]), axis=-1).reshape(-1)
				next_states_enemies = np.array(info["enemy_states"]).reshape(-1)
				next_full_state = np.concatenate((next_states_allies, next_states_enemies), axis=-1)
				ally_states = np.concatenate((self.agent_ids, info["ally_states"]), axis=-1)
				enemy_states = np.repeat(np.expand_dims(states_enemies, axis=0), repeats=self.num_agents, axis=0)
				next_reward_model_obs = np.concatenate((ally_states, enemy_states), axis=-1)
				next_mask_actions = np.array(info["avail_actions"]) # (np.array(info["avail_actions"]) - 1) * 1e5
				next_indiv_dones = info["indiv_dones"]

				# record data for gif
				enemy_action_list.append(info["enemy_action_list"])
				num_allies_alive.append(info["num_allies"])
				num_enemies_alive.append(info["num_enemies"])
				true_rewards.append(rewards)

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
				elif self.experiment_type == "uniform_team_redistribution":
					episodic_team_reward = episodic_team_reward+rewards
					if all(next_indiv_dones) or step == self.max_time_steps:
						rewards_to_send = episodic_team_reward
					else:
						rewards_to_send = 0
				elif "AREL" in self.experiment_type or "ATRR" in self.experiment_type:
					episodic_team_reward = episodic_team_reward+rewards
					if all(next_indiv_dones) or step == self.max_time_steps:
						rewards_to_send = episodic_team_reward
					else:
						rewards_to_send = 0

				# need buffer to get reward model output while generating gif
				self.buffer.push(states, rnn_hidden_state, full_state, reward_model_obs, actions, last_one_hot_action, mask_actions, next_states, next_rnn_hidden_state, next_full_state, next_last_one_hot_action, next_mask_actions, rewards_to_send, dones, indiv_dones, next_indiv_dones)

				states, full_state, reward_model_obs, mask_actions, last_one_hot_action, rnn_hidden_state = next_states, next_full_state, next_reward_model_obs, next_mask_actions, next_last_one_hot_action, next_rnn_hidden_state
				dones, indiv_dones = next_dones, next_indiv_dones

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

			if self.gif:
				reward_model_output = self.agents.reward_model_output(self.buffer)
				ally_attack, enemy_attack = [], []
				for i in range(final_timestep):
					print("ALLY ACTIONS:", action_list[i], "ENEMY ACTIONS:", enemy_action_list[i])
					print("REWARD AT TIMESTEP ", i, "is", reward_model_output[0, i].item())

					aa, ea = 0, 0
					for action in action_list[i]:
						if action > 5:
							aa += 1
					for action in enemy_action_list[i]:
						if action < 0: # negative numbers indicate attack on an ally
							ea += 1

					ally_attack.append(aa)
					enemy_attack.append(ea)

				import matplotlib.pyplot as plt

				# PLOTTING JUST REWARDS
				# # Create a figure and an axes.
				# fig, ax = plt.subplots()

				# # Plot data
				# ax.plot(reward_model_output[0, :final_timestep+1].numpy())

				# # Set a title and labels for the axes.
				# ax.set_title('Reward Redistribution vs Timestep')
				# ax.set_xlabel('Timesteps')
				# ax.set_ylabel('Rewards')

				# Create a figure and a set of subplots for plotting redistributed rewards, true rewards, ally attacks, enemy attacks, num allies alive, num enemies alive
				# fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # 3 Rows, 1 Column
				fig, axs = plt.subplots(6, 1, figsize=(10, 18))  # 6 Rows, 1 Column

				# Plot on the first subplot
				axs[0].plot(reward_model_output[0, :final_timestep+1].numpy(), marker='o', color='red')
				axs[0].set_title('Redistributed Rewards vs Timesteps')
				axs[0].set_xlabel('Timesteps')
				axs[0].set_ylabel('Redistributed Rewards')

				# Plot on the second subplot
				axs[1].plot(true_rewards, marker='o', color='blue')
				axs[1].set_title('True Rewards vs Timesteps')
				axs[1].set_xlabel('Timesteps')
				axs[1].set_ylabel('True Rewards')

				# Plot on the third subplot
				axs[2].plot(ally_attack, marker='s', color='magenta')
				axs[2].set_title('Ally Attacks vs Timesteps')
				axs[2].set_xlabel('Timesteps')
				axs[2].set_ylabel('Ally Attacks')

				# Plot on the fourth subplot
				axs[3].plot(enemy_attack, marker='s', color='purple')
				axs[3].set_title('Enemy Attacks vs Timesteps')
				axs[3].set_xlabel('Timesteps')
				axs[3].set_ylabel('Enemy Attacks')

				# Plot on the fifth subplot
				axs[4].plot(num_allies_alive, marker='s', color='orange')
				axs[4].set_title('Num Allies Alive vs Timesteps')
				axs[4].set_xlabel('Timesteps')
				axs[4].set_ylabel('Num Allies')

				# Plot on the sixth subplot
				axs[5].plot(num_enemies_alive, marker='s', color='yellow')
				axs[5].set_title('Num Enemies Alive vs Timesteps')
				axs[5].set_xlabel('Timesteps')
				axs[5].set_ylabel('Num Enemies')

				# Adjust layout to prevent overlapping
				plt.tight_layout()

				# Save the figure
				plt.savefig('Redistributed_Rewards_Analysis.png', format='png', dpi=300)

				plt.show()

			self.epsilon_greedy = self.epsilon_greedy - self.epsilon_decay_rate if self.epsilon_greedy - self.epsilon_decay_rate > self.epsilon_greedy_min else self.epsilon_greedy_min
			self.buffer.end_episode()

			if self.learn:
				if self.use_reward_model and self.reward_batch_size <= self.buffer.length and episode != 0 and episode % self.update_reward_model_freq == 0:
					reward_loss_batch, grad_norm_reward_batch = 0.0, 0.0
					if "AREL" in self.experiment_type:
						reward_var_batch = 0.0
					elif "ATRR" in self.experiment_type:
						entropy_temporal_weights_batch, entropy_agent_weights_batch = 0.0, 0.0
					
					for i in range(self.reward_model_update_epochs):
						sample = self.buffer.sample_reward_model(num_episodes=self.reward_batch_size)
						if "AREL" in self.experiment_type:
							reward_loss, reward_var, grad_norm_value_reward = self.agents.update_reward_model(sample)
							reward_var_batch += (reward_var/self.reward_model_update_epochs)
						elif "ATRR" in self.experiment_type:
							reward_loss, entropy_temporal_weights, entropy_agent_weights, grad_norm_value_reward = self.agents.update_reward_model(sample)
							entropy_temporal_weights_batch += (entropy_temporal_weights/self.reward_model_update_epochs)
							entropy_agent_weights_batch += (entropy_agent_weights/self.reward_model_update_epochs)

						reward_loss_batch += (reward_loss/self.reward_model_update_epochs)
						grad_norm_reward_batch += (grad_norm_value_reward/self.reward_model_update_epochs)

						if self.scheduler_need:
							self.agents.scheduler_reward.step()

					if self.comet_ml is not None:
						self.comet_ml.log_metric('Reward_Loss', reward_loss_batch, episode)
						self.comet_ml.log_metric('Reward_Grad_Norm', grad_norm_reward_batch, episode)

						if "AREL" in self.experiment_type:
							self.comet_ml.log_metric('Reward_Var', reward_var_batch, episode)
						elif "ATRR" in self.experiment_type:
							self.comet_ml.log_metric('Entropy_Temporal_Weights', entropy_temporal_weights_batch, episode)
							self.comet_ml.log_metric('Entropy_Agent_Weights', entropy_agent_weights_batch, episode)


				if self.batch_size <= self.buffer.length and episode != 0 and episode%self.update_episode_interval == 0:
					Q_loss_batch = 0.0
					grad_norm_batch = 0.0
					for _ in range(self.num_updates):
						sample = self.buffer.sample(
							num_episodes=self.batch_size, 
							Q_network=self.agents.Q_network, 
							target_Q_network=self.agents.target_Q_network, 
							target_QMix_network=self.agents.target_QMix_network,
							reward_model=self.agents.reward_model,
							)
						Q_loss, grad_norm = self.agents.update(sample)
						Q_loss_batch += Q_loss
						grad_norm_batch += grad_norm
					Q_loss_batch /= self.num_updates
					grad_norm_batch /= self.num_updates

					self.plotting_dict = {
					"loss": Q_loss_batch,
					"grad_norm": grad_norm_batch,
					}

					if self.comet_ml is not None:
						self.comet_ml.log_metric('Loss', Q_loss_batch, episode)
						self.comet_ml.log_metric('Grad_Norm', grad_norm_batch, episode)

					if self.scheduler_need:
						self.agents.scheduler.step()

				if self.soft_update:
					soft_update(self.agents.target_Q_network, self.agents.Q_network, self.tau)
					if self.algorithm_type != "IDQN":
						soft_update(self.agents.target_QMix_network, self.agents.QMix_network, self.tau)
				else:
					if episode % self.target_update_interval == 0:
						hard_update(self.agents.target_Q_network, self.agents.Q_network)
						if self.algorithm_type != "IDQN":
							hard_update(self.agents.target_QMix_network, self.agents.QMix_network)


			if self.eval_policy:
				self.rewards.append(episode_reward)
				self.timesteps.append(final_timestep)

			if episode > self.save_model_checkpoint and self.eval_policy:
				self.rewards_mean_per_1000_eps.append(sum(self.rewards[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.timesteps_mean_per_1000_eps.append(sum(self.timesteps[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)

			if not(episode%self.save_model_checkpoint) and episode!=0 and self.save_model:	
				torch.save(self.agents.Q_network.state_dict(), self.model_path+'_Q_epsiode_'+str(episode)+'.pt')
				if self.algorithm_type != "IDQN":
					torch.save(self.agents.QMix_network.state_dict(), self.model_path+'_QMix_epsiode_'+str(episode)+'.pt')

				if self.use_reward_model:
					torch.save(self.agents.reward_model.state_dict(), self.model_path+'_'+self.experiment_type+'_'+str(episode)+'.pt')

			
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

	for i in range(1, 6):
		extension = "QMix_"+str(i)
		test_num = "Learning_Reward_Func_for_Credit_Assignment"
		env_name = "5m_vs_6m"
		experiment_type = "ATRR_agent" # episodic_team, episodic_agent, temporal_team, temporal_agent, uniform_team_redistribution, AREL, ATRR_temporal, ATRR_agent, SeqModel, RUDDER, AREL_agent
		experiment_name = "IDQN_ATRR_agent"
		dictionary = {
				# TRAINING
				"iteration": i,
				"device": "gpu",
				"model_dir": '../../tests/'+test_num+'/'+experiment_type+'/models/'+env_name+'_'+extension+'/models/',
				"gif_dir": '../../tests/'+test_num+'/'+experiment_type+'/gifs/'+env_name+'_'+'_'+extension+'/',
				"policy_eval_dir":'../../tests/'+test_num+'/'+experiment_type+'/policy_eval/'+env_name+'_'+extension+'/',
				"test_num":test_num,
				"extension":extension,
				"experiment_type": experiment_type,
				"gif": False,
				"gif_checkpoint":1,
				"load_models": False,
				"model_path_q_net": "../../../tests/AREL_temporal/models/5m_vs_6m_QMix_2/models/model_Q_epsiode_50000.pt",
				"model_path_qmix_net": "../../../tests/AREL_temporal/models/5m_vs_6m_QMix_2/models/model_QMix_epsiode_50000.pt",
				"model_path_reward_net": "../../../tests/AREL_temporal/models/5m_vs_6m_QMix_2/models/model_AREL_temporal_50000.pt",
				# "model_path": "../../tests/PRD_2_MPE/models/crossing_team_greedy_prd_above_threshold_MAPPO_Q_run_2/critic_networks/critic_epsiode100000.pt",
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
				"batch_size": 128,
				"update_episode_interval": 10,
				"num_updates": 10,
				"epsilon_greedy": 1.0,
				"epsilon_greedy_min": 0.05,
				"epsilon_greedy_decay_episodes": 4000,
				"lambda": 0.8,
				"experiment_name": experiment_name,

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
				"reward_batch_size": 128, # 128
				"reward_lr": 1e-4,
				"reward_weight_decay": 0.0,
				"temporal_score_efficient": 0.0,
				"agent_score_efficient": 0.0,
				"variance_loss_coeff": 0.0,
				"enable_reward_grad_clip": True,
				"reward_grad_clip_value": 10.0,
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
				"algorithm_type": "IDQN", # IDQN, QMIX
				"learning_rate": 1e-4, #1e-3
				"enable_grad_clip": False,
				"grad_clip": 0.5,
				"data_chunk_length": 10,
				"rnn_num_layers": 1,
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
		dictionary["reward_model_obs_shape"] = env.n_agents+info["ally_states"].shape[1]+env.n_enemies*info["enemy_states"].shape[1]
		ma_controller = QMIX(env, dictionary)
		ma_controller.run()