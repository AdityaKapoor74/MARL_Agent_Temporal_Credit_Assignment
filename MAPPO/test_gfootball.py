import random
import multiprocessing as mp
import numpy as np
import time
import gfootball.env as football_env
from gym import spaces
import numpy as np


class FootballEnv(object):
		'''Wrapper to make Google Research Football environment compatible'''

		def __init__(self, env_config):
				self.num_agents = env_config["number_of_left_players_agent_controls"]
				self.scenario_name = env_config["env_name"]

				self.env = football_env.create_environment(**env_config)
						
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



# env = FootballEnv()
# obs = env.reset()
# actions = [0, 1, 2, 3]
# obs, reward, done, info = env.step(actions)
# print(obs.shape, reward, done, info)

# Define the worker function for each environment
def worker_process(env_config, queue, process_id, seed, max_steps):
		# Initialize the football environment for each agent
		env = FootballEnv(env_config)
		env.seed(seed)
		done = False
		states = env.reset()
		
		for i in range(max_steps):
				# Assume you have some agent that takes action based on current state
				# This is where your MAPPO policy or any policy interacts with the environment
				actions = [np.random.choice(env.action_space[agent_num].n) for agent_num in range(env.num_agents)]  # Random action for placeholder
				next_states, rewards, dones, info = env.step(actions)
				
				# Store transition (state, action, reward, next_state, done) in the queue
				queue.put((process_id, states, actions, rewards, next_states, dones))
				
				# Update state for the next iteration
				states = next_states

				print("PROCESS ID", process_id, "STEPS", i, "DONES", all(dones))
				
				if all(dones):
						# Reset environment when the episode ends
						# states = env.reset()
						return

def data_collector(num_processes, env_config, num_episodes=10, max_steps=1000):
		"""
		Spawns multiple environments in parallel and collects data.
		
		Args:
				num_processes: Number of parallel environments (or processes).
				env_config: Configuration for each football environment.
				num_episodes: Number of episodes to collect.
				max_steps: Max steps per episode.
				
		Returns:
				List of collected transitions.
		"""
		
		
		# Start each process
		for i in range(1, num_episodes, num_processes):

				processes = []
				queue = mp.Queue()
				collected_data = [[] for i in range(num_processes)]
				done_dict = {}
				for i in range(num_processes):
						done_dict[i] = False
				
				for process_id in range(num_processes):
						print("STARTED PROCESS ID", process_id)
						seed = np.random.randint(1000) # Random seed for each agent
						p = mp.Process(target=worker_process, args=(env_config, queue, process_id, seed, max_steps))
						processes.append(p)
						p.start()
		
				# Collect data from the queue
				steps = 0
				while steps < max_steps:
						while not queue.empty():
								transition = queue.get()
								done_dict[transition[0]] = all(transition[-1])
								collected_data[transition[0]].append(transition)
								steps += 1

								if all(transition[-1]):
										processes[transition[0]].join()

						if all(done_dict.values()):
								break

				# Wait for all processes to complete
				# for p in processes:
				# 		p.join()

		return collected_data

if __name__ == "__main__":
		# Define the environment configuration
		# Parameters
		num_agents = 4  # Number of parallel environments (workers)
		num_processes = 10
		num_episodes = 10
		max_steps_per_episode = 1000

		env_config = {
				'env_name': 'academy_counterattack_easy',  # Google Football environment
				'stacked': False,
				'rewards': 'scoring,checkpoints',
				'number_of_left_players_agent_controls': num_agents,
				'number_of_right_players_agent_controls': 0,
				'representation': 'simple115v2',
				'channel_dimensions': (96, 72),
				'render': (False and False),
		}
		
		
		# Start data collection
		start_time = time.time()
		collected_data = data_collector(num_processes, env_config, num_episodes, max_steps_per_episode)
		end_time = time.time()

		# print(np.array(collected_data[0]).shape, np.array(collected_data[1]).shape)
		print(f"Collected {len(collected_data)} transitions in {end_time - start_time} seconds.")
