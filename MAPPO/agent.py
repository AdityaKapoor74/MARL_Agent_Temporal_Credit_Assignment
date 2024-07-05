import numpy as np
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from model import Policy, Q_network, PopArt
from utils import RolloutBuffer

class PPOAgent:

	def __init__(
		self, 
		env, 
		dictionary,
		comet_ml,
		):

		# Environment Setup
		self.env = env
		self.algorithm_type = dictionary["algorithm_type"]
		self.env_name = dictionary["env"]
		self.num_agents = self.env.n_agents
		self.num_enemies = self.env.n_enemies
		self.num_actions = self.env.action_space[0].n

		# Training setup
		self.max_episodes = dictionary["max_episodes"]
		self.test_num = dictionary["test_num"]
		self.gif = dictionary["gif"]
		self.n_epochs = dictionary["n_epochs"]
		self.scheduler_need = dictionary["scheduler_need"]
		self.norm_rewards = dictionary["norm_rewards"]
		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"

		self.update_ppo_agent = dictionary["update_ppo_agent"]
		self.max_time_steps = dictionary["max_time_steps"]

		# Critic Setup
		self.temperature_q = dictionary["temperature_q"]
		self.rnn_num_layers_q = dictionary["rnn_num_layers_q"]
		self.rnn_hidden_q = dictionary["rnn_hidden_q"]
		self.q_value_lr = dictionary["q_value_lr"]
		self.q_weight_decay = dictionary["q_weight_decay"]
		self.target_calc_style = dictionary["target_calc_style"]
		self.td_lambda = dictionary["td_lambda"] # TD lambda
		self.n_steps = dictionary["n_steps"]
		self.value_clip = dictionary["value_clip"]
		self.enable_grad_clip_critic_q = dictionary["enable_grad_clip_critic_q"]
		self.grad_clip_critic_q = dictionary["grad_clip_critic_q"]
		self.norm_returns_q = dictionary["norm_returns_q"]
		self.ally_obs_shape = dictionary["ally_obs_shape"]
		self.enemy_obs_shape = dictionary["enemy_obs_shape"]


		self.clamp_rewards = dictionary["clamp_rewards"]
		self.clamp_rewards_value_min = dictionary["clamp_rewards_value_min"]
		self.clamp_rewards_value_max = dictionary["clamp_rewards_value_max"]


		# Actor Setup
		self.data_chunk_length = dictionary["data_chunk_length"]
		self.rnn_num_layers_actor = dictionary["rnn_num_layers_actor"]
		self.rnn_hidden_actor = dictionary["rnn_hidden_actor"]
		self.policy_lr = dictionary["policy_lr"]
		self.policy_weight_decay = dictionary["policy_weight_decay"]
		self.gamma = dictionary["gamma"]
		self.entropy_pen = dictionary["entropy_pen"]
		self.entropy_pen_decay = (dictionary["entropy_pen"] - dictionary["entropy_pen_final"])/dictionary["entropy_pen_steps"]
		self.entropy_pen_final = dictionary["entropy_pen_final"]
		self.gae_lambda = dictionary["gae_lambda"]
		self.norm_adv = dictionary["norm_adv"]
		self.policy_clip = dictionary["policy_clip"]
		self.enable_grad_clip_actor = dictionary["enable_grad_clip_actor"]
		self.grad_clip_actor = dictionary["grad_clip_actor"]

		# Q Network
		if self.algorithm_type in ["MAPPO", "MAAC"]:
			self.centralized = True
		else:
			self.centralized = False


		self.critic_network_q = Q_network(
			use_recurrent_critic=dictionary["use_recurrent_critic"],
			centralized=self.centralized,
			local_observation_input_dim=dictionary["local_observation_shape"], 
			ally_obs_input_dim=dictionary["ally_obs_shape"],
			enemy_obs_input_dim=dictionary["enemy_obs_shape"],
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies,
			num_actions=self.num_actions, 
			rnn_num_layers=self.rnn_num_layers_q,
			comp_emb_shape=self.rnn_hidden_q,
			device=self.device, 
			).to(self.device)
		self.target_critic_network_q = Q_network(
			use_recurrent_critic=dictionary["use_recurrent_critic"],
			centralized=self.centralized,
			local_observation_input_dim=dictionary["local_observation_shape"], 
			ally_obs_input_dim=dictionary["ally_obs_shape"],
			enemy_obs_input_dim=dictionary["enemy_obs_shape"],
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies,
			num_actions=self.num_actions, 
			rnn_num_layers=self.rnn_num_layers_q,
			comp_emb_shape=self.rnn_hidden_q,
			device=self.device, 
			).to(self.device)
		# Copy network params
		self.target_critic_network_q.load_state_dict(self.critic_network_q.state_dict())
		# Disable updates for old network
		for param in self.target_critic_network_q.parameters():
			param.requires_grad_(False)

		if self.norm_returns_q:
			self.Q_PopArt = PopArt(input_shape=1, num_agents=self.num_agents, device=torch.device('cpu'))
		else:
			self.Q_PopArt = None

		# Policy Network
		self.policy_network = Policy(
			use_recurrent_policy=dictionary["use_recurrent_policy"],
			obs_input_dim=dictionary["local_observation_shape"], 
			num_agents=self.num_agents, 
			num_actions=self.num_actions, 
			rnn_num_layers=self.rnn_num_layers_actor,
			rnn_hidden_actor=self.rnn_hidden_actor,
			device=self.device
			).to(self.device)
		

		self.network_update_interval_q = dictionary["network_update_interval_q"]
		self.soft_update_q = dictionary["soft_update_q"]
		self.tau_q = dictionary["tau_q"]


		self.buffer = RolloutBuffer(
			centralized_critic=self.centralized,
			num_episodes=self.update_ppo_agent, 
			max_time_steps=self.max_time_steps, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies,
			local_obs_shape=dictionary["local_observation_shape"], 
			rnn_num_layers_actor=self.rnn_num_layers_actor,
			actor_hidden_state=self.rnn_hidden_actor,
			rnn_num_layers_q=self.rnn_num_layers_q,
			q_hidden_state=self.rnn_hidden_q,
			ally_obs_shape=self.ally_obs_shape,
			enemy_obs_shape=self.enemy_obs_shape,
			num_actions=self.num_actions,
			data_chunk_length=self.data_chunk_length,
			norm_returns_q=self.norm_returns_q,
			target_calc_style=self.target_calc_style,
			td_lambda=self.td_lambda,
			gae_lambda=self.gae_lambda,
			n_steps=self.n_steps,
			gamma=self.gamma,
			)

		# Loading models
		if dictionary["load_models"]:
			# For CPU
			if torch.cuda.is_available() is False:
				self.critic_network.load_state_dict(torch.load(dictionary["model_path_value"], map_location=torch.device('cpu')))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"], map_location=torch.device('cpu')))
			# For GPU
			else:
				self.critic_network.load_state_dict(torch.load(dictionary["model_path_value"]))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"]))


		self.q_critic_optimizer = optim.AdamW(self.critic_network_q.parameters(), lr=self.q_value_lr, weight_decay=self.q_weight_decay, eps=1e-05)
		self.policy_optimizer = optim.AdamW(self.policy_network.parameters(),lr=self.policy_lr, weight_decay=self.policy_weight_decay, eps=1e-05)

		if self.scheduler_need:
			self.scheduler_policy = optim.lr_scheduler.MultiStepLR(self.policy_optimizer, milestones=[1000, 20000], gamma=0.1)
			self.scheduler_q_critic = optim.lr_scheduler.MultiStepLR(self.q_critic_optimizer, milestones=[1000, 20000], gamma=0.1)

		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml

		self.mask_value = torch.tensor(
				torch.finfo(torch.float).min, dtype=torch.float
			)

	
	def get_q_values(self, local_states, ally_states, enemy_states, actions, rnn_hidden_state_q):
		with torch.no_grad():
			local_states = torch.FloatTensor(local_states).unsqueeze(0).unsqueeze(0)
			ally_states = torch.FloatTensor(ally_states).unsqueeze(0).unsqueeze(0)
			enemy_states = torch.FloatTensor(enemy_states).unsqueeze(0).unsqueeze(0)
			actions = torch.LongTensor(actions).unsqueeze(0).unsqueeze(0)
			rnn_hidden_state_q = torch.FloatTensor(rnn_hidden_state_q)
			Q_value, rnn_hidden_state_q = self.target_critic_network_q(local_states.to(self.device), ally_states.to(self.device), enemy_states.to(self.device), actions.to(self.device), rnn_hidden_state_q.to(self.device))

			return Q_value.squeeze(0).cpu().numpy(), rnn_hidden_state_q.cpu().numpy()

	
	def get_action(self, state_actor, last_actions, mask_actions, hidden_state, greedy=False):
		with torch.no_grad():
			state_actor = torch.FloatTensor(state_actor).unsqueeze(0).unsqueeze(1).to(self.device)
			last_actions = torch.LongTensor(last_actions).unsqueeze(0).unsqueeze(1).to(self.device)
			mask_actions = torch.BoolTensor(mask_actions).unsqueeze(0).unsqueeze(1).to(self.device)
			hidden_state = torch.FloatTensor(hidden_state).to(self.device)
			dists, hidden_state = self.policy_network(state_actor, last_actions, hidden_state, mask_actions)

			if greedy:
				actions = [dist.argmax().detach().cpu().item() for dist in dists.squeeze(0).squeeze(0)]
				action_logprob = None
			else:
				actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists.squeeze(0).squeeze(0)]

				probs = Categorical(dists)
				action_logprob = probs.log_prob(torch.FloatTensor(actions).to(self.device)).cpu().numpy()

			return actions, action_logprob, hidden_state.cpu().numpy()

	def plot(self, episode):
		self.comet_ml.log_metric('Q_Value_Loss',self.plotting_dict["q_value_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_Q_Value',self.plotting_dict["grad_norm_value_q"],episode)
		self.comet_ml.log_metric('Policy_Loss',self.plotting_dict["policy_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_Policy',self.plotting_dict["grad_norm_policy"],episode)
		self.comet_ml.log_metric('Entropy',self.plotting_dict["entropy"],episode)
		

	def update_parameters(self):
		if self.entropy_pen - self.entropy_pen_decay > self.entropy_pen_final:
			self.entropy_pen -= self.entropy_pen_decay


	def update(self, episode):
		
		q_value_loss_batch = 0
		policy_loss_batch = 0
		entropy_batch = 0
		grad_norm_value_q_batch = 0
		grad_norm_policy_batch = 0

		self.buffer.calculate_targets(self.Q_PopArt)
		
		for ppo_epoch in range(self.n_epochs):

			# SAMPLE DATA FROM BUFFER
			local_obs, ally_obs, enemy_obs, hidden_state_q, hidden_state_actor, logprobs_old, \
			last_actions, actions, one_hot_actions, action_masks, team_masks, agent_masks, q_values_old, target_q_values, advantage  = self.buffer.sample_recurrent_policy()

			if self.centralized:
				masks = team_masks.unsqueeze(-1)
			else:
				masks = agent_masks

			if self.norm_adv:
				shape = advantage.shape
				advantage_copy = copy.deepcopy(advantage)
				advantage_copy[masks.view(*shape) == 0.0] = float('nan')
				advantage_mean = torch.nanmean(advantage_copy)
				advantage_std = torch.from_numpy(np.array(np.nanstd(advantage_copy.cpu().numpy()))).float()

				advantage = ((advantage - advantage_mean) / (advantage_std + 1e-5))*masks.view(*shape)


			target_shape = q_values_old.shape
			q_values, _ = self.critic_network_q(
							local_obs.to(self.device),
							ally_obs.to(self.device),
							enemy_obs.to(self.device),
							actions.to(self.device),
							hidden_state_q.to(self.device),
							)
			q_values = q_values.reshape(*target_shape)

			q_values_old *= masks
			q_values *= masks.to(self.device)	
			target_q_values *= masks

			if self.norm_returns_q:
				targets_shape = target_q_values.shape
				target_q_values = self.Q_PopArt(target_q_values.view(-1), masks.view(-1)).view(targets_shape) * masks.view(targets_shape)

			dists, _ = self.policy_network(
					local_obs.to(self.device),
					last_actions.to(self.device),
					hidden_state_actor.to(self.device),
					action_masks.to(self.device),
					)

			probs = Categorical(dists)
			logprobs = probs.log_prob(actions.to(self.device))
			
			
			if self.algorithm_type == "IAC" or self.algorithm_type == "MAAC":
				critic_q_loss = F.huber_loss(q_values, target_q_values.to(self.device), reduction="sum", delta=10.0) / (masks.sum()+1e-5)
				policy_loss_ = (logprobs * advantage.to(self.device) * masks.to(self.device)).sum()/(masks.sum()+1e-5)
			else:
				critic_q_loss_1 = F.huber_loss(q_values, target_q_values.to(self.device), reduction="sum", delta=10.0) / (masks.sum()+1e-5)
				critic_q_loss_2 = F.huber_loss(torch.clamp(q_values, q_values_old.to(self.device)-self.value_clip, q_values_old.to(self.device)+self.value_clip), target_q_values.to(self.device), reduction="sum", delta=10.0) / (masks.sum()+1e-5)
				critic_q_loss = torch.max(critic_q_loss_1, critic_q_loss_2)

				# Finding the ratio (pi_theta / pi_theta__old)
				ratios = torch.exp((logprobs - logprobs_old.to(self.device)))
				
				# Finding Surrogate Loss
				surr1 = ratios * advantage.to(self.device) * masks.to(self.device)
				surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage.to(self.device) * masks.to(self.device)

				# final loss of clipped objective PPO
				policy_loss_ = (-torch.min(surr1, surr2).sum())/(masks.sum()+1e-5)

			# calculate entropy
			entropy = -torch.sum(torch.sum(dists*agent_masks.unsqueeze(-1).to(self.device) * torch.log(torch.clamp(dists*agent_masks.unsqueeze(-1).to(self.device), 1e-10,1.0)), dim=-1))/ (agent_masks.sum()+1e-5)
			# add entropy
			policy_loss = policy_loss_ - self.entropy_pen*entropy

			print("Policy Loss", policy_loss_.item(), "Entropy", (-self.entropy_pen*entropy.item()))

			self.q_critic_optimizer.zero_grad()
			critic_q_loss.backward()
			if self.enable_grad_clip_critic_q:
				grad_norm_value_q = torch.nn.utils.clip_grad_norm_(self.critic_network_q.parameters(), self.grad_clip_critic_q)
			else:
				total_norm = 0
				for p in self.critic_network_q.parameters():
					param_norm = p.grad.detach().data.norm(2)
					total_norm += param_norm.item() ** 2
				grad_norm_value_q = torch.tensor([total_norm ** 0.5])
			self.q_critic_optimizer.step()


			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			if self.enable_grad_clip_actor:
				grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.grad_clip_actor)
			else:
				total_norm = 0
				for p in self.policy_network.parameters():
					param_norm = p.grad.detach().data.norm(2)
					total_norm += param_norm.item() ** 2
				grad_norm_policy = torch.tensor([total_norm ** 0.5])
			self.policy_optimizer.step()

			q_value_loss_batch += critic_q_loss.item()
			policy_loss_batch += policy_loss.item()
			entropy_batch += entropy.item()
			grad_norm_value_q_batch += grad_norm_value_q
			grad_norm_policy_batch += grad_norm_policy
			

		# Copy new weights into old critic
		if self.soft_update_q:
			for target_param, param in zip(self.target_critic_network_q.parameters(), self.critic_network_q.parameters()):
				target_param.data.copy_(target_param.data * (1.0 - self.tau_q) + param.data * self.tau_q)
		elif episode % self.network_update_interval_q == 0:
			self.target_critic_network_q.load_state_dict(self.critic_network_q.state_dict())
	
		
		# clear buffer
		self.buffer.clear()
		

		q_value_loss_batch /= self.n_epochs
		policy_loss_batch /= self.n_epochs
		entropy_batch /= self.n_epochs
		grad_norm_value_q_batch /= self.n_epochs
		grad_norm_policy_batch /= self.n_epochs


		
		self.plotting_dict = {
		"q_value_loss": q_value_loss_batch,
		"policy_loss": policy_loss_batch,
		"entropy": entropy_batch,
		"grad_norm_value_q": grad_norm_value_q_batch,
		"grad_norm_policy": grad_norm_policy_batch,
		}

		
		if self.comet_ml is not None:
			self.plot(episode)

		del q_value_loss_batch, policy_loss_batch, entropy_batch, grad_norm_value_q_batch, grad_norm_policy_batch
		torch.cuda.empty_cache()