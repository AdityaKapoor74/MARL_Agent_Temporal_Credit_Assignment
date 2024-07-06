import os
import time
from comet_ml import Experiment
import numpy as np
from agent import PPOAgent
import torch
import datetime



class MAPPO:

	def __init__(self, env, dictionary):

		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"
		self.env = env
		self.environment = dictionary["environment"]
		self.gif = dictionary["gif"]
		self.save_model = dictionary["save_model"]
		self.save_model_checkpoint = dictionary["save_model_checkpoint"]
		self.save_comet_ml_plot = dictionary["save_comet_ml_plot"]
		self.learn = dictionary["learn"]
		self.gif_checkpoint = dictionary["gif_checkpoint"]
		self.eval_policy = dictionary["eval_policy"]
		self.num_agents = dictionary["num_agents"]
		
		if "StarCraft" in self.environment:
			self.num_enemies = self.env.n_enemies

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
		self.update_ppo_agent = dictionary["update_ppo_agent"]

		# RNN HIDDEN
		self.rnn_num_layers_v = dictionary["rnn_num_layers_v"]
		self.rnn_hidden_v = dictionary["rnn_hidden_v"]
		self.rnn_num_layers_actor = dictionary["rnn_num_layers_actor"]
		self.rnn_hidden_actor = dictionary["rnn_hidden_actor"]


		self.comet_ml = None
		if self.save_comet_ml_plot:
			self.comet_ml = Experiment("im5zK8gFkz6j07uflhc3hXk8I", project_name=dictionary["test_num"])
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


		for episode in range(1,self.max_episodes+1):

			local_obs, info = self.env.reset(return_info=True)
			mask_actions = np.array(info["avail_actions"], dtype=int)
			last_actions = np.zeros((self.num_agents)) + self.num_actions
			ally_states = np.array(info["ally_states"])
			enemy_states = np.array(info["enemy_states"])
			local_obs = np.array(local_obs)
			indiv_dones = [0]*self.num_agents
			indiv_dones = np.array(indiv_dones)
			dones = all(indiv_dones)
			
			images = []

			episode_reward = 0
			episode_indiv_rewards = [0 for i in range(self.num_agents)]
			final_timestep = self.max_time_steps

			
			rnn_hidden_state_v = np.zeros((self.rnn_num_layers_v, self.num_agents, self.rnn_hidden_v))
			rnn_hidden_state_actor = np.zeros((self.rnn_num_layers_actor, self.num_agents, self.rnn_hidden_actor))

			for step in range(1, self.max_time_steps+1):

				if self.gif:
					# At each step, append an image to list
					# import time
					# time.sleep(0.1)
					self.env.render()
					# Advance a step and render a new image
					with torch.no_grad():
						actions, action_logprob, next_rnn_hidden_state_actor = self.agents.get_action(local_obs, last_actions, mask_actions, rnn_hidden_state_actor, greedy=False)
				else:
					actions, action_logprob, next_rnn_hidden_state_actor = self.agents.get_action(local_obs, last_actions, mask_actions, rnn_hidden_state_actor)

				one_hot_actions = np.zeros((self.num_agents, self.num_actions))
				for i, act in enumerate(actions):
					one_hot_actions[i][act] = 1

				value, next_rnn_hidden_state_v = self.agents.get_values(local_obs, ally_states, enemy_states, actions, rnn_hidden_state_v, indiv_dones, episode)
				
				next_local_obs, rewards, next_dones, info = self.env.step(actions)
				next_local_obs = np.array(next_local_obs)
				next_ally_states = np.array(info["ally_states"])
				next_enemy_states = np.array(info["enemy_states"])
				next_mask_actions = np.array(info["avail_actions"], dtype=int)
				next_indiv_dones = info["indiv_dones"]
				indiv_rewards = info["indiv_rewards"]
				
				episode_reward += np.sum(rewards)
				episode_indiv_rewards = [r+indiv_rewards[i] for i, r in enumerate(episode_indiv_rewards)]

				if self.learn:

					if self.experiment_type == "temporal_team":
						rewards_to_send = [rewards]*self.num_agents
					elif self.experiment_type == "episodic_team" or self.experiment_type == "uniform_team_redistribution" or "AREL" in self.experiment_type or "ATRR" in self.experiment_type:
						episodic_team_reward = episodic_team_reward+rewards
						if all(next_indiv_dones) or step == self.max_time_steps:
							rewards_to_send = episodic_team_reward
						else:
							rewards_to_send = 0

					self.agents.buffer.push(
						ally_states, enemy_states, value, rnn_hidden_state_v, \
						local_obs, rnn_hidden_state_actor, action_logprob, actions, one_hot_actions, mask_actions, \
						rewards_to_send, indiv_dones, dones
						)

					ally_states, enemy_states = next_ally_states, next_enemy_states

					
				local_obs, last_actions, mask_actions, indiv_dones, dones = next_local_obs, actions, next_mask_actions, next_indiv_dones, next_dones
				rnn_hidden_state_v, rnn_hidden_state_actor = next_rnn_hidden_state_v, next_rnn_hidden_state_actor

				if all(indiv_dones) or step == self.max_time_steps:

					final_timestep = step

					if self.learn:
						# add final time to buffer
						actions, action_logprob, next_rnn_hidden_state_actor = self.agents.get_action(local_obs, last_actions, mask_actions, rnn_hidden_state_actor)
					

						value, _ = self.agents.get_values(local_obs, ally_states, enemy_states, actions, rnn_hidden_state_v, indiv_dones, episode)
						
						self.agents.buffer.end_episode(final_timestep, value, indiv_dones, dones)

					print("*"*100)
					print("EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} | INDIV REWARD STREAMS: {} \n".format(episode, np.round(episode_reward,decimals=4), step, self.max_time_steps, episode_indiv_rewards))
					print("Num Allies Alive: {} | Num Enemies Alive: {} | AGENTS DEAD: {} \n".format(info["num_allies"], info["num_enemies"], info["indiv_dones"]))
					print("*"*100)

					if self.save_comet_ml_plot:
						self.comet_ml.log_metric('Episode_Length', step, episode)
						self.comet_ml.log_metric('Reward', episode_reward, episode)

						self.comet_ml.log_metric('Num Enemies', info["num_enemies"], episode)
						self.comet_ml.log_metric('Num Allies', info["num_allies"], episode)
						self.comet_ml.log_metric('All Enemies Dead', info["all_enemies_dead"], episode)
						self.comet_ml.log_metric('All Allies Dead', info["all_allies_dead"], episode)
						
					break

			if self.agents.scheduler_need:
				self.agents.scheduler_policy.step()
				self.agents.scheduler_v_critic.step()

			if self.eval_policy:
				self.rewards.append(episode_reward)
				self.timesteps.append(final_timestep)

			if episode > self.save_model_checkpoint and self.eval_policy:
				self.rewards_mean_per_1000_eps.append(sum(self.rewards[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.timesteps_mean_per_1000_eps.append(sum(self.timesteps[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)

				
			if not(episode%self.save_model_checkpoint) and episode!=0 and self.save_model:	
				torch.save(self.agents.critic_network_v.state_dict(), self.critic_model_path+'_V_epsiode'+str(episode)+'.pt')
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

	torch.set_printoptions(profile="full")
	torch.autograd.set_detect_anomaly(True)

	for i in range(1, 2):
		extension = "MAPPO_"+str(i)
		test_num = "Learning_Reward_Func_for_Credit_Assignment"
		environment = "StarCraft" # StarCraft/ MPE/ PressurePlate/ PettingZoo/ LBForaging
		env_name = "5m_vs_6m" # 5m_vs_6m/ 10m_vs_11m/ 3s5z/ crossing_team_greedy/ pressureplate-linear-6p-v0/ pursuit_v4/ "Foraging-{0}x{0}-{1}p-{2}f{3}-v2".format(grid_size, num_players, num_food, "-coop" if fully_coop else "")
		experiment_type = "temporal_team" # episodic_team, episodic_agent, temporal_team, temporal_agent, uniform_team_redistribution, ATRR_temporal ~ AREL, ATRR_temporal_v2, ATRR_temporal_attn_weights, ATRR_agent, ATRR_agent_temporal_attn_weights
		experiment_name = "MAPPO_temporal_team"
		algorithm_type = "MAPPO"

		dictionary = {
				# TRAINING
				"iteration": i,
				"device": "gpu",
				"update_learning_rate_with_prd": False,
				"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
				"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
				"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"n_epochs": 5,
				"update_ppo_agent": 10, # update ppo agent after every update_ppo_agent episodes; 10 (StarCraft/MPE/PressurePlate/LBF)/ 5 (PettingZoo)
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
				"eval_policy": True,
				"save_model": True,
				"save_model_checkpoint": 1000,
				"save_comet_ml_plot": True,
				"learn":True,
				"max_episodes": 30000, # 20000 (StarCraft environments)/ 30000 (MPE/PressurePlate)/ 5000 (PettingZoo)/ 15000 (LBForaging)
				"max_time_steps": 50, # 100 (StarCraft environments & MPE)/ 70 (PressurePlate & LBForaging)/ 500 (PettingZoo)
				"experiment_type": experiment_type,
				"parallel_training": False,
				"scheduler_need": False,
				"norm_rewards": False,
				"clamp_rewards": False,
				"clamp_rewards_value_min": 0.0,
				"clamp_rewards_value_max": 2.0,


				# REWARD MODEL
				"use_reward_model": False,
				"reward_n_heads": 3, # 3
				"reward_depth": 3, # 3
				"reward_agent_attn": True,
				"reward_dropout": 0.0,
				"reward_attn_net_wide": True,
				"version": "agent_temporal_attn_weights", # temporal, temporal_v2, agent_temporal, temporal_attn_weights, agent_temporal_attn_weights
				"reward_linear_compression_dim": 64,
				"reward_batch_size": 128, # 128
				"reward_lr": 5e-4,
				"reward_weight_decay": 0.0,
				"temporal_score_coefficient": 0.0,
				"agent_score_coefficient": 0.0,
				"variance_loss_coeff": 0.0,
				"enable_reward_grad_clip": True,
				"reward_grad_clip_value": 10.0,
				"replay_buffer_size": 5000,
				"update_reward_model_freq": 200, # 200
				"reward_model_update_epochs": 400, # 400
				"norm_rewards": False,


				"algorithm_type": algorithm_type,


				# ENVIRONMENT
				"env": env_name,

				# CRITIC
				"use_recurrent_critic": True,
				"rnn_num_layers_v": 1,
				"rnn_hidden_v": 64,
				"v_value_lr": 5e-4, #1e-3
				"temperature_v": 1.0,
				"attention_dropout_prob_v": 0.0,
				"v_comp_emb_shape": 64,
				"v_weight_decay": 0.0,
				"enable_grad_clip_critic_v": True,
				"grad_clip_critic_v": 0.5,
				"value_clip": 0.2,
				"enable_hard_attention": False,
				"num_heads": 4,
				"critic_weight_entropy_pen": 0.0,
				"critic_weight_entropy_pen_final": 0.0,
				"critic_weight_entropy_pen_steps": 100, # number of updates
				"critic_score_regularizer": 0.0,
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
				"policy_lr": 5e-4, #prd 1e-4
				"policy_weight_decay": 0.0,
				"entropy_pen": 1e-3, #8e-3
				"entropy_pen_final": 1e-3,
				"entropy_pen_steps": 20000,
				"gae_lambda": 0.95,
				"norm_adv": True,
			}

		seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(seeds[dictionary["iteration"]-1])
			
		import gym
		import smaclite  # noqa
		
		env = gym.make(f"smaclite/{env_name}-v0", use_cpp_rvo2=USE_CPP_RVO2)
		obs, info = env.reset(return_info=True)
		dictionary["ally_observation_shape"] = info["ally_states"][0].shape[0]
		dictionary["enemy_observation_shape"] = info["enemy_states"][0].shape[0]
		dictionary["local_observation_shape"] = obs[0].shape[0]
		dictionary["num_agents"] = env.n_agents
		dictionary["num_enemies"] = env.n_enemies
		dictionary["num_actions"] = env.action_space[0].n

		ma_controller = MAPPO(env,dictionary)
		ma_controller.run()





# visa@manchester.ac.uk --- For VISA || international@manchester.ac.uk