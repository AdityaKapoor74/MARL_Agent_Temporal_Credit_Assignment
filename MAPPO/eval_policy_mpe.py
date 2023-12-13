from model import MLP_Policy, Q_network, V_network
from utils import RolloutBuffer

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

import numpy as np
import os
import sys
sys.path.append("../../../")
sys.path.append("./")

import torch
from torch.distributions import Categorical

def make_env(scenario_name, benchmark=False):
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	world = scenario.make_world()
	if benchmark:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.isFinished)
	return env, scenario.observation_shape, scenario.transformer_observation_shape


env_name = "crossing_team_greedy"
env, local_observation_size, global_observation_size = make_env(scenario_name=env_name, benchmark=False)
num_agents = env.n
num_actions = env.action_space[0].n

experiment_type = "shared"
update_ppo_agent = 1
n_epochs = 1
policy_clip = 0.05
entropy_pen = 1e-3
grad_clip_actor = 0.5
select_above_threshold = 0.12
steps_to_take = 0
enable_hard_attention = False
max_episodes = 100
max_time_steps = 100
lambda_ = 0.95
gae_lambda = 0.95
gamma = 0.99

if experiment_type == "shared":
	model_root_dir = "./../../../tests/TEAM COLLISION AVOIDANCE/models/crossing_team_greedy_shared_MAPPO_1/"
else:
	model_root_dir = "./../../../tests/TEAM COLLISION AVOIDANCE/models/crossing_team_greedy_prd_above_threshold_ascend_MAPPO_1/"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_states(states):
	states_critic = []
	states_actor = []
	for i in range(num_agents):
		states_critic.append(states[i][0])
		states_actor.append(states[i][1])

	states_critic = np.asarray(states_critic)
	states_actor = np.asarray(states_actor)

	return states_critic, states_actor


critic_network_q = Q_network(obs_input_dim=global_observation_size, num_heads=4, num_agents=num_agents, num_actions=num_actions, device=device, enable_hard_attention=enable_hard_attention).to(device)
# critic_network_q.load_state_dict(torch.load(dictionary["model_path_value"], map_location=torch.device('cpu')))

critic_network_v = V_network(obs_input_dim=global_observation_size, num_heads=4, num_agents=num_agents, num_actions=num_actions, device=device, enable_hard_attention=enable_hard_attention).to(device)
# critic_network_v.load_state_dict(torch.load(dictionary["model_path_value"], map_location=torch.device('cpu')))

policy_network = MLP_Policy(obs_input_dim=local_observation_size, num_agents=num_agents, num_actions=num_actions, device=device).to(device)
# policy_network.load_state_dict(torch.load(dictionary["model_path_value"], map_location=torch.device('cpu')))


buffer = RolloutBuffer(
	num_episodes=update_ppo_agent, 
	max_time_steps=max_time_steps, 
	num_agents=num_agents, 
	obs_shape_critic=global_observation_size, 
	obs_shape_actor=local_observation_size, 
	num_actions=num_actions
	)


def get_action(state_policy):
	with torch.no_grad():
		state_policy = torch.FloatTensor(state_policy).to(device)
		dists = policy_network(state_policy)
		
		actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists]

		probs = Categorical(dists)
		action_logprob = probs.log_prob(torch.FloatTensor(actions).to(device))

		return actions, action_logprob.cpu().numpy()

def build_td_lambda_targets(rewards, terminated, mask, target_qs):
	# Assumes  <target_qs > in B*T*A and <reward >, <terminated >  in B*T*A, <mask > in (at least) B*T-1*1
	# Initialise  last  lambda -return  for  not  terminated  episodes
	ret = target_qs.new_zeros(*target_qs.shape)
	ret = target_qs * (1-terminated)
	# ret[:, -1] = target_qs[:, -1] * (1 - (torch.sum(terminated, dim=1)>0).int())
	# Backwards  recursive  update  of the "forward  view"
	for t in range(ret.shape[1] - 2, -1,  -1):
		ret[:, t] = lambda_ * gamma * ret[:, t + 1] + mask[:, t].unsqueeze(-1) \
					* (rewards[:, t] + (1 - lambda_) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
	# Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
	# return ret[:, 0:-1]
	return ret


def calculate_advantages(values, values_old, rewards, dones, masks_):
	advantages = []
	next_value = 0
	advantage = 0
	masks = 1 - dones
	for t in reversed(range(0, len(rewards))):
		td_error = rewards[t] + (gamma * next_value * masks[t]) - values.data[t]
		next_value = values_old.data[t]
		advantage = (td_error + (gamma * gae_lambda * advantage * masks[t]))*masks_[t]
		advantages.insert(0, advantage)

	advantages = torch.stack(advantages)
	
	return advantages


def calculate_advantages_based_on_exp(V_values, V_values_old, rewards, dones, weights_prd, masks, episode):
	advantage = None
	masking_rewards = None
	mean_min_weight_value = -1
	if "shared" in experiment_type:
		rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, num_agents, 1), dim=-1)
		advantage = calculate_advantages(V_values, V_values_old, rewards_, dones, masks)
	elif "prd_above_threshold_ascend" in experiment_type or "prd_above_threshold_decay" in experiment_type:
		masking_rewards = (weights_prd>select_above_threshold).int()
		rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, num_agents, 1) * torch.transpose(masking_rewards,-1,-2), dim=-1)
		advantage = calculate_advantages(V_values, V_values_old, rewards_, dones, masks)
	elif "prd_above_threshold" in experiment_type:
		if episode > steps_to_take:
			masking_rewards = (weights_prd>select_above_threshold).int()
			rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, num_agents, 1) * torch.transpose(masking_rewards,-1,-2), dim=-1)
		else:
			masking_rewards = torch.ones(weights_prd.shape).to(device)
			rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, num_agents, 1), dim=-1)
		advantage = calculate_advantages(V_values, V_values_old, rewards_, dones, masks)
	elif "top" in experiment_type:
		if episode > steps_to_take:
			rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, num_agents, 1), dim=-1)
			advantage = calculate_advantages(V_values, V_values_old, rewards_, dones, masks)
			masking_rewards = torch.ones(weights_prd.shape).to(device)
			min_weight_values, _ = torch.min(weights_prd, dim=-1)
			mean_min_weight_value = torch.mean(min_weight_values)
		else:
			values, indices = torch.topk(weights_prd,k=top_k,dim=-1)
			min_weight_values, _ = torch.min(values, dim=-1)
			mean_min_weight_value = torch.mean(min_weight_values)
			masking_rewards = torch.sum(F.one_hot(indices, num_classes=num_agents), dim=-2)
			rewards_ = torch.sum(masking_rewards * torch.transpose(masking_rewards,-1,-2), dim=-1)
			advantage = calculate_advantages(V_values, V_values_old, rewards_, dones, masks)
	elif "prd_soft_advantage" in experiment_type:
		masking_rewards = (weights_prd>1e-4).int()
		rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, num_agents, 1) * torch.transpose(weights_prd * num_agents,-1,-2), dim=-1)
		advantage = calculate_advantages(V_values, V_values_old, rewards_, dones, masks)

	return advantage.detach(), masking_rewards, mean_min_weight_value

policy_grads = []
for iterator in range(1, 31):
	critic_network_q.load_state_dict(torch.load(os.path.join(model_root_dir, "critic_networks", "critic_Q_epsiode"+str(1000*iterator)+".pt"), map_location=device))

	critic_network_v.load_state_dict(torch.load(os.path.join(model_root_dir, "critic_networks", "critic_V_epsiode"+str(1000*iterator)+".pt"), map_location=device))
	
	policy_network.load_state_dict(torch.load(os.path.join(model_root_dir, "actor_networks", "actor_epsiode"+str(1000*iterator)+".pt"), map_location=device))
	
	policy_grads_runs = []
	for run in range(1, 11):
		policy_grads_episodic = []
		for episode in range(1, max_episodes+1):

			states = env.reset()

			states_critic, states_actor = split_states(states)

			episode_reward = 0
			for step in range(1, max_time_steps+1):

				actions, action_logprob = get_action(states_actor)
				one_hot_actions = np.zeros((num_agents, num_actions))
				for act_num, act in enumerate(actions):
					one_hot_actions[act_num][act] = 1
				

				next_states, rewards, dones, info = env.step(actions)
				rewards = [value[0] for value in rewards]
				next_states_critic, next_states_actor = split_states(next_states)

				buffer.push(states_critic, states_actor, action_logprob, actions, one_hot_actions, rewards, dones)

				episode_reward += np.sum(rewards)

				states_critic, states_actor = next_states_critic, next_states_actor
				states = next_states

				if all(dones) or step == max_time_steps:

					print("*"*100)
					print("ITERATION: {} | RUN: {} | EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} \n".format(iterator, run, episode,np.round(episode_reward,decimals=4), step, max_time_steps))
					print("*"*100)


					# convert list to tensor
					old_states_critic = torch.FloatTensor(np.array(buffer.states_critic)).reshape(-1, num_agents, global_observation_size)
					old_states_actor = torch.FloatTensor(np.array(buffer.states_actor)).reshape(-1, num_agents, local_observation_size)
					old_actions = torch.FloatTensor(np.array(buffer.actions)).reshape(-1, num_agents)
					old_one_hot_actions = torch.FloatTensor(np.array(buffer.one_hot_actions)).reshape(-1, num_agents, num_actions)
					old_logprobs = torch.FloatTensor(buffer.logprobs).reshape(-1, num_agents)
					rewards = torch.FloatTensor(np.array(buffer.rewards))
					dones = torch.FloatTensor(np.array(buffer.dones)).long()
					masks = torch.FloatTensor(np.array(buffer.masks)).long()


					with torch.no_grad():
						# Q_values_old, weights_prd_old, _ = critic_network_q(
						# 									old_states_critic.to(device),
						# 									old_one_hot_actions.to(device)
						# 									)

						Values_old, _, _ = critic_network_v(
											old_states_critic.to(device),
											old_one_hot_actions.to(device)
											)

					# if "threshold" in experiment_type or "top" in experiment_type:
					# 	mask_rewards = (torch.mean(weights_prd_old, dim=1)>select_above_threshold).int()
					# 	target_V_rewards = torch.sum(rewards.reshape(-1, num_agents).unsqueeze(-2).repeat(1, num_agents, 1) * torch.transpose(mask_rewards.detach().cpu(),-1,-2), dim=-1)
					# else:
					# 	target_V_rewards = torch.sum(rewards.reshape(-1, num_agents).unsqueeze(-2).repeat(1, num_agents, 1), dim=-1)

					# target_Q_values = build_td_lambda_targets(rewards.to(device), dones.to(device), masks.to(device), Q_values_old.reshape(update_ppo_agent, max_time_steps, num_agents)).reshape(-1, num_agents)
					# target_V_values = build_td_lambda_targets(target_V_rewards.reshape(update_ppo_agent, max_time_steps, num_agents).to(device), dones.to(device), masks.to(device), Values_old.reshape(update_ppo_agent, max_time_steps, num_agents)).reshape(-1, num_agents)
					
					rewards = rewards.reshape(-1, num_agents)
					dones = dones.reshape(-1, num_agents)
					masks = masks.reshape(-1, 1)

					total_norm = 0.0

					# Optimize policy for n epochs
					for _ in range(n_epochs):

						
						Q_value, weights_prd, score_q = critic_network_q(
							old_states_critic.to(device), 
							old_one_hot_actions.to(device)
							)
						Value, weight_v, score_v = critic_network_v(
							old_states_critic.to(device),  
							old_one_hot_actions.to(device)
							)

						advantage, masking_rewards, mean_min_weight_value = calculate_advantages_based_on_exp(Value, Values_old, rewards.to(device), dones.to(device), torch.mean(weights_prd.detach(), dim=1), masks.to(device), episode)

						dists = policy_network(old_states_actor.to(device))
						probs = Categorical(dists.squeeze(0))
						logprobs = probs.log_prob(old_actions.to(device))

						# if "threshold" in experiment_type or "top" in experiment_type:
						# 	agent_groups_over_episode = torch.sum(torch.sum(masking_rewards.float(), dim=-2),dim=0)/masking_rewards.shape[0]
						# 	avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)
						# 	agent_groups_over_episode_batch += agent_groups_over_episode
						# 	avg_agent_group_over_episode_batch += avg_agent_group_over_episode
							
						
						# critic_v_loss_1 = F.mse_loss(Value*masks.to(device), target_V_values*masks.to(device), reduction="sum") / masks.sum()
						# critic_v_loss_2 = F.mse_loss(torch.clamp(Value, Values_old.to(device)-value_clip, Values_old.to(device)+value_clip)*masks.to(device), target_V_values*masks.to(device), reduction="sum") / masks.sum()

						# critic_q_loss_1 = F.mse_loss(Q_value*masks.to(device), target_Q_values*masks.to(device), reduction="sum") / masks.sum()
						# critic_q_loss_2 = F.mse_loss(torch.clamp(Q_value, Q_values_old.to(device)-value_clip, Q_values_old.to(device)+value_clip)*masks.to(device), target_Q_values*masks.to(device), reduction="sum") / masks.sum()

						# Finding the ratio (pi_theta / pi_theta__old)
						ratios = torch.exp(logprobs - old_logprobs.to(device))
						# Finding Surrogate Loss
						surr1 = ratios * advantage*masks.unsqueeze(-1).to(device)
						surr2 = torch.clamp(ratios, 1-policy_clip, 1+policy_clip) * advantage * masks.unsqueeze(-1).to(device)

						# final loss of clipped objective PPO
						entropy = -torch.mean(torch.sum(dists*masks.unsqueeze(-1).to(device) * torch.log(torch.clamp(dists*masks.unsqueeze(-1).to(device), 1e-10,1.0)), dim=2))
						policy_loss = (-torch.min(surr1, surr2).mean() - entropy_pen*entropy)
						
						# entropy_weights = 0
						# entropy_weights_v = 0
						# for i in range(num_heads):
						# 	entropy_weights += -torch.mean(torch.sum(weights_prd[:, i] * torch.log(torch.clamp(weights_prd[:, i], 1e-10,1.0)), dim=2))
						# 	entropy_weights_v += -torch.mean(torch.sum(weight_v[:, i] * torch.log(torch.clamp(weight_v[:, i], 1e-10,1.0)), dim=2))

						# critic_q_loss = torch.max(critic_q_loss_1, critic_q_loss_2) + self.critic_score_regularizer*(score_q**2).mean() + self.critic_weight_entropy_pen*entropy_weights
						# critic_v_loss = torch.max(critic_v_loss_1, critic_v_loss_2) + self.critic_score_regularizer*(score_v**2).mean() + self.critic_weight_entropy_pen*entropy_weights_v

						
						# q_critic_optimizer.zero_grad()
						# critic_q_loss.backward()
						# grad_norm_value_q = torch.nn.utils.clip_grad_norm_(critic_network_q.parameters(), grad_clip_critic)
						# q_critic_optimizer.step()
						
						# v_critic_optimizer.zero_grad()
						# critic_v_loss.backward()
						# grad_norm_value_v = torch.nn.utils.clip_grad_norm_(critic_network_v.parameters(), grad_clip_critic)
						# v_critic_optimizer.step()
						

						# policy_optimizer.zero_grad()
						policy_loss.backward()
						# for p in policy_network.parameters():
						# 	total_norm += torch.sum(p.grad.detach().data)
						total_grads = None
						for p in policy_network.parameters():
							param_grad = p.grad.detach().data.reshape(-1)
							if total_grads is not None:
								total_grads = torch.cat([total_grads, param_grad], dim=-1)
							else:
								total_grads = param_grad

						# grad_norm_policy = torch.nn.utils.clip_grad_norm_(policy_network.parameters(), grad_clip_actor)
						# policy_optimizer.step()

						# self.history_states_critic_q = new_history_states_critic_q.detach().cpu()
						# self.history_states_critic_v = new_history_states_critic_v.detach().cpu()

					# self.scheduler.step()
					# print("learning rate of policy", self.scheduler.get_lr())

					# clear buffer
					buffer.clear()

					policy_grads_episodic.append(total_grads.tolist())

					break

		policy_grads_runs.append(policy_grads_episodic)

	policy_grads.append(policy_grads_runs)


	np.save("../../../tests/TEAM COLLISION AVOIDANCE/eval_policy/"+str(experiment_type), np.array(policy_grads), allow_pickle=True, fix_imports=True)
