import os
import time
from comet_ml import Experiment
import numpy as np
from agent import PPOAgent
import torch
import datetime

import gym
import smaclite

torch.autograd.set_detect_anomaly(True)



class MAPPO:

	def __init__(self, env, dictionary):

		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"
		self.env = env
		self.algorithm_type = dictionary["algorithm_type"]
		self.gif = dictionary["gif"]
		self.save_model = dictionary["save_model"]
		self.save_model_checkpoint = dictionary["save_model_checkpoint"]
		self.save_comet_ml_plot = dictionary["save_comet_ml_plot"]
		self.learn = dictionary["learn"]
		self.gif_checkpoint = dictionary["gif_checkpoint"]
		self.eval_policy = dictionary["eval_policy"]
		self.num_agents = self.env.n_agents
		self.num_enemies = self.env.n_enemies
		self.num_actions = self.env.action_space[0].n
		self.date_time = f"{datetime.datetime.now():%d-%m-%Y}"
		self.env_name = dictionary["env"]
		self.test_num = dictionary["test_num"]
		self.max_episodes = dictionary["max_episodes"]
		self.max_time_steps = dictionary["max_time_steps"]
		self.update_ppo_agent = dictionary["update_ppo_agent"]
		self.warm_up_period = dictionary["warm_up_period"]

		# RNN HIDDEN
		self.rnn_num_layers_q = dictionary["rnn_num_layers_q"]
		self.rnn_hidden_q = dictionary["rnn_hidden_q"]
		self.rnn_num_layers_actor = dictionary["rnn_num_layers_actor"]
		self.rnn_hidden_actor = dictionary["rnn_hidden_actor"]

		if self.algorithm_type in ["MAPPO", "MAAC"]:
			self.centralized = True
		else:
			self.centralized = False

		self.agent_ids = []
		for i in range(self.num_agents):
			agent_id = np.array([0 for i in range(self.num_agents)])
			agent_id[i] = 1
			self.agent_ids.append(agent_id)
		self.agent_ids = np.array(self.agent_ids)

		self.enemy_ids = []
		for i in range(self.num_enemies):
			enemy_id = np.array([0 for i in range(self.num_enemies)])
			enemy_id[i] = 1
			self.enemy_ids.append(enemy_id)
		self.enemy_ids = np.array(self.enemy_ids)


		self.comet_ml = None
		if self.save_comet_ml_plot:
			self.comet_ml = Experiment("im5zK8gFkz6j07uflhc3hXk8I",project_name=dictionary["test_num"])
			self.comet_ml.set_name(dictionary["experiment_name"])
			self.comet_ml.log_parameters(dictionary)


		self.agents = PPOAgent(self.env, dictionary, self.comet_ml)
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

			
			self.critic_model_path = critic_dir+"critic"
			self.actor_model_path = actor_dir+"actor"
			

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

		for episode in range(1,self.max_episodes+1):

			local_states, info = self.env.reset(return_info=True)
			mask_actions = np.array(info["avail_actions"], dtype=int)
			last_actions = np.zeros((self.num_agents)) + self.num_actions
			ally_states = info["ally_states"]
			enemy_states = info["enemy_states"]
			indiv_dones = [0]*self.num_agents
			indiv_dones = np.array(indiv_dones)
			dones = int(all(indiv_dones))
			

			images = []

			episodic_team_reward = 0

			episode_reward = 0
			action_frequency = np.array([0.0]*self.num_actions)
			episode_indiv_rewards = [0 for i in range(self.num_agents)]
			final_timestep = self.max_time_steps

			if self.centralized:
				rnn_hidden_state_q = np.zeros((self.rnn_num_layers_q, 1, self.rnn_hidden_q))
			else:
				rnn_hidden_state_q = np.zeros((self.rnn_num_layers_q, self.num_agents, self.rnn_hidden_q))
			rnn_hidden_state_actor = np.zeros((self.rnn_num_layers_actor, self.num_agents, self.rnn_hidden_actor))

			for step in range(1, self.max_time_steps+1):

				if self.gif:
					# import time
					# time.sleep(0.1)
					self.env.render()
					# Advance a step and render a new image
					with torch.no_grad():
						actions, action_logprob, next_rnn_hidden_state_actor = self.agents.get_action(local_states, last_actions, mask_actions, rnn_hidden_state_actor, greedy=False)
				else:
					actions, action_logprob, next_rnn_hidden_state_actor = self.agents.get_action(local_states, last_actions, mask_actions, rnn_hidden_state_actor)

				one_hot_actions = np.zeros((self.num_agents, self.num_actions))
				for i, act in enumerate(actions):
					one_hot_actions[i][act] = 1

				
				q_value, next_rnn_hidden_state_q = self.agents.get_q_values(local_states, info["ally_states"], info["enemy_states"], actions, rnn_hidden_state_q)

				next_local_states, rewards, next_dones, info = self.env.step(actions)
				next_ally_states = info["ally_states"]
				next_enemy_states = info["enemy_states"]
				next_mask_actions = np.array(info["avail_actions"], dtype=int)
				next_indiv_dones = info["indiv_dones"]

				if self.learn:
					self.agents.buffer.push(
						q_value, rnn_hidden_state_q, \
						local_states, rnn_hidden_state_actor, action_logprob, actions, one_hot_actions, mask_actions, \
						ally_states, enemy_states, rewards, indiv_dones, dones,
						)

				episode_reward += np.sum(rewards)
				episode_indiv_rewards = [r+info["indiv_rewards"][i] for i, r in enumerate(episode_indiv_rewards)]
				action_frequency += np.sum(one_hot_actions, axis=0)

				local_states, last_actions, ally_states, enemy_states, mask_actions, indiv_dones, dones = next_local_states, actions, next_ally_states, next_enemy_states, next_mask_actions, next_indiv_dones, next_dones
				rnn_hidden_state_q, rnn_hidden_state_actor = next_rnn_hidden_state_q, next_rnn_hidden_state_actor

				if all(indiv_dones) or step == self.max_time_steps:

					print("*"*100)
					print("EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} | Num Allies Alive: {} | Num Enemies Alive: {} \n".format(episode, np.round(episode_reward,decimals=4), step, self.max_time_steps, info["num_allies"], info["num_enemies"]))
					print("INDIV REWARD STREAMS", episode_indiv_rewards, "AGENTS DEAD", info["indiv_dones"])
					print("ACTION FREQUENCY", action_frequency)
					print("*"*100)

					final_timestep = step

					if self.save_comet_ml_plot:
						self.comet_ml.log_metric('Episode_Length', step, episode)
						self.comet_ml.log_metric('Reward', episode_reward, episode)
						self.comet_ml.log_metric('Num Enemies', info["num_enemies"], episode)
						self.comet_ml.log_metric('Num Allies', info["num_allies"], episode)
						self.comet_ml.log_metric('All Enemies Dead', info["all_enemies_dead"], episode)
						self.comet_ml.log_metric('All Allies Dead', info["all_allies_dead"], episode)


					if self.learn:
						# add final time to buffer
						actions, action_logprob, next_rnn_hidden_state_actor = self.agents.get_action(local_states, last_actions, mask_actions, rnn_hidden_state_actor)

						q_value, _ = self.agents.get_q_values(local_states, info["ally_states"], info["enemy_states"], actions, rnn_hidden_state_q)

						self.agents.buffer.end_episode(final_timestep, q_value, indiv_dones, dones)

					break


			if self.agents.scheduler_need:
				self.agents.scheduler_policy.step()
				self.agents.scheduler_q_critic.step()

			if self.eval_policy:
				self.rewards.append(episode_reward)
				self.timesteps.append(final_timestep)

			if episode > self.save_model_checkpoint and self.eval_policy:
				self.rewards_mean_per_1000_eps.append(sum(self.rewards[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.timesteps_mean_per_1000_eps.append(sum(self.timesteps[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)

			if not(episode%self.save_model_checkpoint) and episode!=0 and self.save_model:	
				torch.save(self.agents.critic_network_q.state_dict(), self.critic_model_path+'_Q_epsiode'+str(episode)+'.pt')
				torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'.pt')  

			if self.learn and not(episode%self.update_ppo_agent) and episode != 0:
				self.agents.update(episode)

			if self.eval_policy and not(episode%self.save_model_checkpoint) and episode!=0:
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"reward_list"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_rewards_per_1000_eps"), np.array(self.rewards_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"timestep_list"), np.array(self.timesteps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_timestep_per_1000_eps"), np.array(self.timesteps_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
				

if __name__ == '__main__':

	RENDER = False
	USE_CPP_RVO2 = False

	import os
	os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

	for i in range(1, 6):
		extension = "MAPPO_"+str(i)
		test_num = "Learning_Reward_Func_for_Credit_Assignment"
		env_name = "5m_vs_6m"
		algorithm_type = "MAPPO" # IPPO/ MAPPO
		experiment_name = algorithm_type + "_" + env_name

		dictionary = {
				# TRAINING
				"iteration": i,
				"device": "gpu",
				"critic_dir": '../../tests/'+test_num+'/models/'+env_name+'_'+extension+'/critic_networks/',
				"actor_dir": '../../tests/'+test_num+'/models/'+env_name+'_'+extension+'/actor_networks/',
				"gif_dir": '../../tests/'+test_num+'/gifs/'+env_name+'_'+extension+'/',
				"policy_eval_dir":'../../tests/'+test_num+'/policy_eval/'+env_name+'_'+extension+'/',
				"n_epochs": 5,
				"update_ppo_agent": 10, # update ppo agent after every update_ppo_agent episodes
				"test_num": test_num,
				"extension": extension,
				"experiment_name": experiment_name,
				"gamma": 0.99,
				"gif": False,
				"gif_checkpoint":1,
				"load_models": False,
				"model_path_value": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_prd_above_threshold_MAPPO_Q_run_2/critic_networks/critic_epsiode100000.pt",
				"model_path_policy": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_prd_above_threshold_MAPPO_Q_run_2/actor_networks/actor_epsiode100000.pt",
				"eval_policy": True,
				"save_model": True,
				"save_model_checkpoint": 1000,
				"save_comet_ml_plot": True,
				"learn":True,
				"max_episodes": 50000,
				"max_time_steps": 50,
				"scheduler_need": False,
				"norm_rewards": False,
				"clamp_rewards": False,
				"clamp_rewards_value_min": 0.0,
				"clamp_rewards_value_max": 2.0,
				"warm_up_period": 2000, # 2000


				# ENVIRONMENT
				"env": env_name,

				# ALGORITHM TYPE
				"algorithm_type": algorithm_type,

				# CRITIC
				"use_recurrent_critic": True,
				"rnn_num_layers_q": 1,
				"rnn_hidden_q": 64,
				"q_value_lr": 5e-4, #1e-3
				"temperature_q": 1.0,
				"q_weight_decay": 0.0,
				"enable_grad_clip_critic_q": True,
				"grad_clip_critic_q": 10.0,
				"value_clip": 0.2,
				"target_calc_style": "GAE", # GAE, TD_Lambda, N_steps
				"td_lambda": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
				"n_steps": 5,
				"norm_returns_q": True,
				"soft_update_q": False,
				"tau_q": 0.05,
				"network_update_interval_q": 1,
				

				# ACTOR
				"use_recurrent_policy": True,
				"data_chunk_length": 10,
				"rnn_num_layers_actor": 1,
				"rnn_hidden_actor": 64,
				"enable_grad_clip_actor": True,
				"grad_clip_actor": 10.0,
				"policy_clip": 0.2,
				"policy_lr": 5e-4, #prd 1e-4
				"policy_weight_decay": 0.0,
				"entropy_pen": 1e-2, #8e-3
				"entropy_pen_final": 1e-2,
				"entropy_pen_steps": 20000,
				"gae_lambda": 0.95,
				"norm_adv": True,
			}

		seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(seeds[dictionary["iteration"]-1])
		env = gym.make(f"smaclite/{env_name}-v0", use_cpp_rvo2=USE_CPP_RVO2)
		obs, info = env.reset(return_info=True)
		dictionary["local_observation_shape"] = obs[0].shape[0]
		dictionary["ally_obs_shape"] = info["ally_states"].shape[1]
		dictionary["enemy_obs_shape"] = info["enemy_states"].shape[1]
		ma_controller = MAPPO(env,dictionary)
		ma_controller.run()