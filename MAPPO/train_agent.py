"""
This script serves as the main entry point for training the MAPPO agent with various
credit assignment methods, including TAR², AREL, STAS, and other baselines.

It defines the MAPPO class, which orchestrates the entire training process:
- Initializes the environment, agent, and necessary buffers.
- Manages the main training loop, including episode rollouts and data collection.
- Coordinates the updates for the policy networks (MAPPO) and, if applicable, the
  credit assignment model (e.g., TAR²).
- Handles logging, model saving, and evaluation.

The script is configured via a dictionary and can be run for different environments
like StarCraft II (via SMACLite) and Google Research Football.
"""
import os
import argparse
from comet_ml import Experiment
import numpy as np
from agent import PPOAgent
import torch

class MAPPO:
	"""
	Main controller for the Multi-Agent Proximal Policy Optimization (MAPPO) training process.
	"""
	def __init__(self, env, dictionary):
		"""
		Initializes the training environment, agent, and all necessary components.

		Args:
			env: The multi-agent environment instance.
			dictionary (dict): A dictionary containing all hyperparameters and configuration settings.
		"""
		# --- Basic Setup ---
		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"
		self.env = env
		self.environment = dictionary["environment"]
		self.env_name = dictionary["env"]
		self.num_agents = dictionary["num_agents"]
		self.num_actions = dictionary["num_actions"]
		
		# --- Experiment Configuration ---
		self.learn = dictionary["learn"]
		self.max_episodes = dictionary["max_episodes"]
		self.max_time_steps = dictionary["max_time_steps"]
		self.experiment_type = dictionary["experiment_type"]
		self.ppo_eps_elapse_update_freq = dictionary["ppo_eps_elapse_update_freq"]

		# --- Logging and Saving ---
		self.save_model = dictionary["save_model"]
		self.save_model_checkpoint = dictionary["save_model_checkpoint"]
		self.eval_policy = dictionary["eval_policy"]
		
		# --- Credit Assignment Model Setup ---
		self.use_reward_model = dictionary["use_reward_model"]
		self.warm_up_period = dictionary["warm_up_period"]
		if self.use_reward_model:
			self.reward_batch_size = dictionary["reward_batch_size"]
			self.update_reward_model_freq = dictionary["update_reward_model_freq"]
			self.reward_model_update_epochs = dictionary["reward_model_update_epochs"]

		# --- Environment-Specific Setup ---
		if "StarCraft" in self.environment:
			self.num_enemies = self.env.n_enemies
		else:
			self.num_enemies = 1

		# --- RNN Hidden State Dimensions ---
		self.rnn_num_layers_v = dictionary["rnn_num_layers_v"]
		self.rnn_hidden_v = dictionary["rnn_hidden_v"]
		self.rnn_num_layers_actor = dictionary["rnn_num_layers_actor"]
		self.rnn_hidden_actor = dictionary["rnn_hidden_actor"]

		# --- Comet.ml Logging ---
		self.save_comet_ml_plot = dictionary["save_comet_ml_plot"]
		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = Experiment("im5zK8gFkz6j07uflhc3hXk8I", project_name=dictionary["test_num"])
			self.comet_ml.set_name(dictionary["experiment_name"])
			self.comet_ml.log_parameters(dictionary)

		# --- Agent Initialization ---
		self.agents = PPOAgent(dictionary, self.comet_ml)

		# --- Model Saving Directories ---
		if self.save_model:
			critic_dir = dictionary["critic_dir"]
			actor_dir = dictionary["actor_dir"]
			os.makedirs(critic_dir, exist_ok=True)
			os.makedirs(actor_dir, exist_ok=True)
			self.critic_model_path = os.path.join(critic_dir, "critic")
			self.actor_model_path = os.path.join(actor_dir, "actor")

		if self.eval_policy:
			self.policy_eval_dir = dictionary["policy_eval_dir"]
			os.makedirs(self.policy_eval_dir, exist_ok=True)

	def run(self):
		"""
		Executes the main training loop for the specified number of episodes.
		"""
		if self.eval_policy:
			self.rewards = []
			self.timesteps = []

		# --- Main Training Loop ---
		for episode in range(1, self.max_episodes + 1):
			# --- Episode Initialization ---
			if "StarCraft" in self.environment:
				local_obs, info = self.env.reset(return_info=True)
				mask_actions = np.array(info["avail_actions"], dtype=int)
				ally_states = np.array(info["ally_states"])
				enemy_states = np.array(info["enemy_states"])
				global_obs = None
			elif "GFootball" in self.environment:
				global_obs = local_obs = self.env.reset()
				global_obs = np.array(global_obs)
				mask_actions = np.ones([self.num_agents, self.num_actions])
				ally_states, enemy_states = None, None
				info = {}
			
			last_actions = np.zeros((self.num_agents)) + self.num_actions
			local_obs = np.array(local_obs)
			indiv_dones = np.zeros(self.num_agents)
			
			episode_reward = 0
			episodic_team_reward = 0
			
			rnn_hidden_state_v = np.zeros((self.rnn_num_layers_v, self.num_agents, self.rnn_hidden_v))
			rnn_hidden_state_actor = np.zeros((self.rnn_num_layers_actor, self.num_agents, self.rnn_hidden_actor))

			# --- Episode Rollout Loop ---
			for step in range(1, self.max_time_steps + 1):
				# Get actions and value estimates from the agent
				actions, action_logprob, next_rnn_hidden_state_actor = self.agents.get_action(local_obs, last_actions, mask_actions, rnn_hidden_state_actor)
				value, next_rnn_hidden_state_v = self.agents.get_values(global_obs, ally_states, enemy_states, actions, rnn_hidden_state_v, indiv_dones)
				
				# Step the environment
				next_local_obs, rewards, next_dones, next_info = self.env.step(actions)
				next_local_obs = np.array(next_local_obs)

				# Process environment-specific outputs
				if "StarCraft" in self.environment:
					next_global_obs = None
					next_ally_states = np.array(next_info["ally_states"])
					next_enemy_states = np.array(next_info["enemy_states"])
					next_mask_actions = np.array(next_info["avail_actions"], dtype=int)
					next_indiv_dones = next_info["indiv_dones"]
				elif "GFootball" in self.environment:
					next_ally_states, next_enemy_states = None, None
					next_global_obs = next_local_obs
					next_indiv_dones = next_dones
					rewards = rewards[0] * self.num_agents
					next_mask_actions = np.ones([self.num_agents, self.num_actions])
				
				episode_reward += np.sum(rewards)

				# --- Reward Shaping for Episodic Tasks ---
				# Accumulate the dense rewards to form a single episodic signal.
				rewards_to_send = 0
				if "episodic_team" in self.experiment_type or "Uniform" in self.experiment_type or "AREL" in self.experiment_type or "TAR" in self.experiment_type or "STAS" in self.experiment_type:
					episodic_team_reward += np.sum(rewards)
					if all(next_indiv_dones) or step == self.max_time_steps:
						rewards_to_send = episodic_team_reward
				else: # For dense reward settings
					rewards_to_send = rewards

				# --- Store data in buffers ---
				if self.learn:
					print("I am here")
					self.agents.buffer.push(
						ally_states, enemy_states, value, rnn_hidden_state_v,
						global_obs, local_obs, rnn_hidden_state_actor, action_logprob, actions, mask_actions,
						rewards_to_send, indiv_dones, all(indiv_dones)
					)
				if self.use_reward_model:
					self.agents.reward_buffer.push(
						ally_states, enemy_states, local_obs, actions, mask_actions, rnn_hidden_state_actor, action_logprob, rewards_to_send, all(indiv_dones), indiv_dones
					)

				# Update states for the next iteration
				global_obs, local_obs, last_actions, mask_actions, indiv_dones = next_global_obs, next_local_obs, actions, next_mask_actions, next_indiv_dones
				rnn_hidden_state_v, rnn_hidden_state_actor = next_rnn_hidden_state_v, next_rnn_hidden_state_actor
				if "StarCraft" in self.environment:
					ally_states, enemy_states = next_ally_states, next_enemy_states
				info = next_info

				# --- End of Episode Handling ---
				if all(indiv_dones) or step == self.max_time_steps:
					if self.learn:
						# Get final value estimate for GAE calculation
						final_value_v, _ = self.agents.get_values(global_obs, ally_states, enemy_states, actions, rnn_hidden_state_v, indiv_dones)
						self.agents.buffer.end_episode(step, final_value_v, indiv_dones, all(indiv_dones))

					# During warmup, clear the on-policy buffer to only use shaped rewards later
					if self.use_reward_model and episode <= self.warm_up_period:
						self.agents.buffer.clear()

					print(f"Episode: {episode} | Reward: {episode_reward:.2f} | Timesteps: {step}/{self.max_time_steps}")
					
					if self.save_comet_ml_plot:
						self.comet_ml.log_metric('Reward', episode_reward, episode)
					break

			if self.use_reward_model:
				self.agents.reward_buffer.end_episode()

			# --- Agent and Model Updates ---
			# Update the main MAPPO agent
			if self.learn and (episode % self.ppo_eps_elapse_update_freq == 0) and episode > 0:
				if self.use_reward_model and episode > self.warm_up_period:
					# Replace buffer rewards with the output of the credit assignment model
					self.agents.buffer.rewards = self.agents.reward_model_output()
				self.agents.update(episode)

			# Update the credit assignment model
			if self.learn and self.use_reward_model and self.reward_batch_size <= self.agents.reward_buffer.length and (episode % self.update_reward_model_freq == 0) and episode > 0:
				for _ in range(self.reward_model_update_epochs):
					sample = self.agents.reward_buffer.sample_reward_model(num_episodes=self.reward_batch_size)
					# The update_reward_model function handles the specific loss for each model type
					self.agents.update_reward_model(sample)

			# --- Saving and Logging ---
			if self.eval_policy:
				self.rewards.append(episode_reward)
			if (episode % self.save_model_checkpoint == 0) and self.save_model:
				torch.save(self.agents.critic_network_v.state_dict(), f'{self.critic_model_path}_V_episode{episode}.pt')
				torch.save(self.agents.policy_network.state_dict(), f'{self.actor_model_path}_episode{episode}.pt')
				

def parse_args():
	"""
	Parses command-line arguments for the training script.
	"""
	parser = argparse.ArgumentParser(description="Train MAPPO with various credit assignment methods.")
	
	# --- General Training Arguments ---
	parser.add_argument("--iteration", type=int, default=1, help="Seed and iteration number for the run.")
	parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"], help="Device to use for training.")
	parser.add_argument("--n_epochs", type=int, default=5, help="Number of PPO update epochs.")
	parser.add_argument("--ppo_eps_elapse_update_freq", type=int, default=10, help="Update PPO agent after this many episodes.")
	parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
	parser.add_argument("--learn", action="store_true", default=True, help="Flag to enable learning.")
	parser.add_argument("--max_episodes", type=int, default=30000, help="Maximum number of training episodes.")
	parser.add_argument("--warm_up_period", type=int, default=200, help="Number of episodes to warm up the reward buffer.")
	parser.add_argument("--scheduler_need", action="store_true", default=False, help="Flag to use a learning rate scheduler.")

	# --- Environment Arguments ---
	parser.add_argument("--environment", type=str, default="StarCraft", choices=["StarCraft", "GFootball"], help="Environment to use.")
	parser.add_argument("--env", type=str, default="3s5z", help="Specific map or scenario name.")
	parser.add_argument("--max_time_steps", type=int, default=100, help="Maximum timesteps per episode.")

	# --- Credit Assignment Arguments ---
	parser.add_argument("--experiment_type", type=str, default="TAR^2", choices=["episodic_team", "Uniform", "AREL", "STAS", "TAR^2"], help="Credit assignment method to use.")
	parser.add_argument("--use_reward_model", action="store_true", default=True, help="Flag to use a credit assignment model.")
	parser.add_argument("--reward_n_heads", type=int, default=4, help="Number of attention heads in the reward model.")
	parser.add_argument("--reward_depth", type=int, default=3, help="Number of layers in the reward model.")
	parser.add_argument("--reward_linear_compression_dim", type=int, default=64, help="Embedding dimension in the reward model.")
	parser.add_argument("--reward_batch_size", type=int, default=64, help="Batch size for reward model updates.")
	parser.add_argument("--reward_lr", type=float, default=1e-4, help="Learning rate for the reward model.")
	parser.add_argument("--reward_weight_decay", type=float, default=0.0, help="Weight decay for the reward model optimizer.")
	parser.add_argument("--dynamic_loss_coeffecient", type=float, default=5e-2, help="Coefficient for the inverse dynamics loss.")
	parser.add_argument("--variance_loss_coeff", type=float, default=0.0, help="Coefficient for the variance loss (AREL specific).")
	parser.add_argument("--replay_buffer_size", type=int, default=5000, help="Capacity of the off-policy reward buffer.")
	parser.add_argument("--update_reward_model_freq", type=int, default=100, help="Frequency of reward model updates (in episodes).")
	parser.add_argument("--reward_model_update_epochs", type=int, default=200, help="Number of gradient steps per reward model update.")
	parser.add_argument("--reward_agent_attn", action="store_true", default=True, help="Flag to use agent attention in AREL.")
	parser.add_argument("--reward_dropout", type=float, default=0.0, help="Dropout in AREL.")
	parser.add_argument("--reward_attn_net_wide", action="store_true", default=True, help="Flag to use wide attention in AREL.")
	parser.add_argument("--version", type=str, default="temporal", choices=["temporal", "agent_temporal"], help="Version of AREL to use.")
	parser.add_argument("--norm_rewards", action="store_true", default=False, help="Flag to normalize rewards.")
	parser.add_argument("--clamp_rewards", action="store_true", default=False, help="Flag to clamp rewards.")
	parser.add_argument("--clamp_rewards_value_min", type=float, default=0.0, help="Min value for reward clamping.")
	parser.add_argument("--clamp_rewards_value_max", type=float, default=2.0, help="Max value for reward clamping.")
	parser.add_argument("--enable_reward_grad_clip", action="store_true", default=True, help="Flag to enable gradient clipping for the reward model.")
	parser.add_argument("--reward_grad_clip_value", type=float, default=0.5, help="Gradient clipping value for the reward model.")

	# --- Actor Arguments ---
	parser.add_argument("--use_recurrent_policy", action="store_true", default=True, help="Flag to use a recurrent policy.")
	parser.add_argument("--data_chunk_length", type=int, default=10, help="Length of chunks for recurrent policy training.")
	parser.add_argument("--rnn_num_layers_actor", type=int, default=1, help="Number of RNN layers in the actor.")
	parser.add_argument("--rnn_hidden_actor", type=int, default=64, help="Hidden dimension of the actor's RNN.")
	parser.add_argument("--policy_lr", type=float, default=5e-4, help="Actor learning rate.")
	parser.add_argument("--policy_weight_decay", type=float, default=0.0, help="Weight decay for the actor optimizer.")
	parser.add_argument("--entropy_pen", type=float, default=6e-3, help="Entropy bonus coefficient.")
	parser.add_argument("--entropy_pen_final", type=float, default=6e-3, help="Final entropy bonus coefficient.")
	parser.add_argument("--entropy_pen_steps", type=int, default=20000, help="Steps over which to decay the entropy bonus.")
	parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter.")
	parser.add_argument("--norm_adv", action="store_true", default=True, help="Flag to normalize advantages.")
	parser.add_argument("--policy_clip", type=float, default=0.2, help="PPO clipping parameter.")
	parser.add_argument("--enable_grad_clip_actor", action="store_true", default=True, help="Flag to enable gradient clipping for the actor.")
	parser.add_argument("--grad_clip_actor", type=float, default=0.5, help="Gradient clipping value for the actor.")

	# --- Critic Arguments ---
	parser.add_argument("--use_recurrent_critic", action="store_true", default=True, help="Flag to use a recurrent critic.")
	parser.add_argument("--rnn_num_layers_v", type=int, default=1, help="Number of RNN layers in the critic.")
	parser.add_argument("--rnn_hidden_v", type=int, default=64, help="Hidden dimension of the critic's RNN.")
	parser.add_argument("--v_comp_emb_shape", type=int, default=64, help="Embedding dimension for the critic's input.")
	parser.add_argument("--v_value_lr", type=float, default=5e-4, help="Critic learning rate.")
	parser.add_argument("--v_weight_decay", type=float, default=0.0, help="Weight decay for the critic optimizer.")
	parser.add_argument("--value_clip", type=float, default=0.2, help="PPO value clipping parameter.")
	parser.add_argument("--enable_grad_clip_critic_v", action="store_true", default=True, help="Flag to enable gradient clipping for the critic.")
	parser.add_argument("--grad_clip_critic_v", type=float, default=0.5, help="Gradient clipping value for the critic.")
	parser.add_argument("--norm_returns_v", action="store_true", default=True, help="Flag to use PopArt normalization for returns.")
	
	# --- Logging and Saving Arguments ---
	parser.add_argument("--load_models", action="store_true", default=False, help="Flag to save model checkpoints.")
	parser.add_argument("--save_model", action="store_true", default=True, help="Flag to save model checkpoints.")
	parser.add_argument("--save_model_checkpoint", type=int, default=1000, help="Frequency of model saving (in episodes).")
	parser.add_argument("--save_comet_ml_plot", action="store_true", default=True, help="Flag to enable Comet.ml logging.")
	parser.add_argument("--eval_policy", action="store_true", default=True, help="Flag to enable evaluation data saving.")
	parser.add_argument("--test_num", type=str, default="Learning_Reward_Func_for_Credit_Assignment", help="Test name for logging.")
	
	args = parser.parse_args()
	return vars(args)


if __name__ == '__main__':
	# This block sets up the configuration dictionary and launches the training run.
	# It allows for easy configuration of different environments, baselines, and hyperparameters.
	RENDER = False
	USE_CPP_RVO2 = False

	# --- Parse Arguments and Set Up ---
	args = parse_args()
	
	# Set seed for reproducibility
	seeds = [42, 142, 242, 342, 442]
	torch.manual_seed(seeds[args["iteration"] - 1])
	np.random.seed(seeds[args["iteration"] - 1])

	# --- Create Dynamic Configuration Dictionary ---
	extension = f"MAPPO_{args['iteration']}"
	args["experiment_name"] = f"MAPPO_{args['experiment_type']}"
	args["critic_dir"] = f"../../../tests/{args['test_num']}/models/{args['env']}_{args['experiment_type']}_{extension}/critic_networks/"
	args["actor_dir"] = f"../../../tests/{args['test_num']}/models/{args['env']}_{args['experiment_type']}_{extension}/actor_networks/"
	args["policy_eval_dir"] = f"../../../tests/{args['test_num']}/policy_eval/{args['env']}_{args['experiment_type']}_{extension}/"
	args["model_path_v_value"] = f"../../../tests/{args['test_num']}/models/{args['env']}_{args['experiment_type']}_{extension}/critic_networks/critic_V_episode10000.pt"
	args["model_path_policy"] = f"../../../tests/{args['test_num']}/models/{args['env']}_{args['experiment_type']}_{extension}/actor_networks/actor_episode10000.pt"


	torch.set_printoptions(profile="full")
	torch.autograd.set_detect_anomaly(True)

	# for i in range(1, 6):
		# extension = "MAPPO_"+str(i)
		# test_num = "Learning_Reward_Func_for_Credit_Assignment"
		# environment = "StarCraft" # StarCraft/ GFootball
		# env_name = "3s5z" # 5m_vs_6m, 10m_vs_11m, 3s5z/ academy_3_vs_1_with_keeper, academy_counterattack_easy, academy_run_pass_and_shoot_with_keeper 
		# experiment_type = "TAR^2" # episodic_team, episodic_agent, temporal_team, temporal_agent, Uniform, AREL, STAS, TAR^2
		# experiment_name = "MAPPO_TAR^2" # MAPPO_TAR^2, MAPPO_AREL, MAPPO_STAS, MAPPO_Uniform, MAPPO_temporal, MAPPO_agent_temporal, MAPPO_episodic_agent, MAPPO_episodic_team

		# dictionary = {
		# 		# TRAINING
		# 		"iteration": i,
		# 		"device": "gpu",
		# 		"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
		# 		"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
		# 		"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
		# 		"n_epochs": 5,
		# 		"ppo_eps_elapse_update_freq": 10, # update ppo agent after every ppo_eps_elapse_update_freq episodes; 10 (StarCraft/MPE/PressurePlate/LBF)/ 5 (PettingZoo)
		# 		"environment": environment,
		# 		"experiment_name": experiment_name,
		# 		"test_num": test_num,
		# 		"extension": extension,
		# 		"gamma": 0.99,
		# 		"load_models": False,
		# 		"model_path_v_value": "../../tests/RLC_2024/relevant_set_visualization/crossing_team_greedy/prd_soft_advantage/models/crossing_team_greedy_prd_soft_advantage_MAPPO_1/critic_networks/critic_V_epsiode10000.pt",
		# 		"model_path_policy": "../../tests/RLC_2024/relevant_set_visualization/crossing_team_greedy/prd_soft_advantage/models/crossing_team_greedy_prd_soft_advantage_MAPPO_1/actor_networks/actor_epsiode10000.pt",
		# 		"eval_policy": True,
		# 		"save_model": True,
		# 		"save_model_checkpoint": 1000,
		# 		"save_comet_ml_plot": True,
		# 		"learn":True,
		# 		"max_episodes": 30000, # 30000 (StarCraft environments)/ 120000 (GFootball)
		# 		"max_time_steps": 100, # 50 (StarCraft environments)/ 200 (GFootball)
		# 		"experiment_type": experiment_type,
		# 		"scheduler_need": False,
		# 		"norm_rewards": False,
		# 		"clamp_rewards": False,
		# 		"clamp_rewards_value_min": 0.0,
		# 		"clamp_rewards_value_max": 2.0,
		# 		"warm_up_period": 200, # 200


		# 		# REWARD MODEL
		# 		"use_reward_model": True,
		# 		"reward_n_heads": 4, # 3
		# 		"reward_depth": 3, # 3
		# 		"reward_agent_attn": True,
		# 		"reward_dropout": 0.0,
		# 		"reward_attn_net_wide": True,
		# 		"version": "temporal", # temporal, agent_temporal ---- For AREL
		# 		"reward_linear_compression_dim": 64, # 16 for TAR^2_agent_temporal
		# 		"reward_batch_size": 64, # 128
		# 		"reward_lr": 1e-4,
		# 		"reward_weight_decay": 0.0,
		# 		"dynamic_loss_coeffecient": 5e-2,
		# 		"variance_loss_coeff": 0.0,
		# 		"enable_reward_grad_clip": True,
		# 		"reward_grad_clip_value": 0.5,
		# 		"replay_buffer_size": 5000,
		# 		"update_reward_model_freq": 100, # 100
		# 		"reward_model_update_epochs": 200, # 200
		# 		"norm_rewards": False,


		# 		# ENVIRONMENT
		# 		"env": env_name,

		# 		# CRITIC
		# 		"use_recurrent_critic": True,
		# 		"rnn_num_layers_v": 1,
		# 		"rnn_hidden_v": 64,
		# 		"v_value_lr": 5e-4, #1e-3
		# 		"v_weight_decay": 0.0,
		# 		"v_comp_emb_shape": 64,
		# 		"enable_grad_clip_critic_v": True,
		# 		"grad_clip_critic_v": 0.5,
		# 		"value_clip": 0.2,
		# 		"norm_returns_v": True,
				

		# 		# ACTOR
		# 		"use_recurrent_policy": True,
		# 		"data_chunk_length": 10,
		# 		"rnn_num_layers_actor": 1,
		# 		"rnn_hidden_actor": 64,
		# 		"enable_grad_clip_actor": True,
		# 		"grad_clip_actor": 0.5,
		# 		"policy_clip": 0.2,
		# 		"policy_lr": 5e-4, 
		# 		"policy_weight_decay": 0.0,
		# 		"entropy_pen": 6e-3, #8e-3
		# 		"entropy_pen_final": 6e-3,
		# 		"entropy_pen_steps": 20000,
		# 		"gae_lambda": 0.95,
		# 		"norm_adv": True,
		# 	}

	# 	seeds = [42, 142, 242, 342, 442]
	# 	torch.manual_seed(seeds[dictionary["iteration"]-1])
		
	if "StarCraft" in args["environment"]:
		import gym
		import smaclite  # noqa
		
		env = gym.make(f"smaclite/{args['env']}-v0", use_cpp_rvo2=USE_CPP_RVO2)
		obs, info = env.reset(return_info=True)
		args["ally_observation_shape"] = info["ally_states"][0].shape[0]
		args["enemy_observation_shape"] = info["enemy_states"][0].shape[0]
		args["local_observation_shape"] = obs[0].shape[0]
		args["num_agents"] = env.n_agents
		args["num_enemies"] = env.n_enemies
		args["num_actions"] = env.action_space[0].n
	elif "GFootball" in args["environment"]:
		import random

		import gfootball.env as football_env
		from gym import spaces
		import numpy as np


		class FootballEnv(object):
			'''Wrapper to make Google Research Football environment compatible'''

			def __init__(self, env_name):
				self.scenario_name = env_name

				if self.scenario_name == "academy_3_vs_1_with_keeper":
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


			def reset(self):
				obs = self.env.reset()
				obs = self._obs_wrapper(obs)
				return obs

			def step(self, action):
				obs, reward, done, info = self.env.step(action)
				obs = self._obs_wrapper(obs)
				reward = reward.reshape(self.num_agents, 1)
				if self.share_reward:
					global_reward = np.sum(reward)
					reward = [[global_reward]] * self.num_agents

				done = np.array([done] * self.num_agents)
				info = self._info_wrapper(info)
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


		env = FootballEnv(args["env"])

		args["num_agents"] = env.num_agents
		args["local_observation_shape"] = env.observation_space[0].shape[0]
		args["global_observation_shape"] = env.observation_space[0].shape[0]
		args["num_actions"] = env.action_space[0].n
		

	ma_controller = MAPPO(env, args)
	ma_controller.run()