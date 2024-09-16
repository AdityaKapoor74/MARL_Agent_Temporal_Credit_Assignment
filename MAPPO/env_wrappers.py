"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
import torch
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
from enum import Enum
from gym.error import AlreadyPendingCallError, ClosedEnvironmentError, CustomSpaceError, NoAsyncCallError
from typing import Any, Dict
import warnings


def tile_images(img_nhwc):
	"""
	Tile N images into one big PxQ image
	(P,Q) are chosen to be as close as possible, and if N
	is square, then P=Q.
	input: img_nhwc, list or array of images, ndim=4 once turned into array
		n = batch index, h = height, w = width, c = channel
	returns:
		bigim_HWc, ndarray with ndim=3
	"""
	img_nhwc = np.asarray(img_nhwc)
	N, h, w, c = img_nhwc.shape
	H = int(np.ceil(np.sqrt(N)))
	W = int(np.ceil(float(N)/H))
	img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
	img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
	img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
	img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
	return img_Hh_Ww_c

class AsyncState(Enum):
	DEFAULT = "default"
	WAITING_RESET = "reset"
	WAITING_STEP = "step"

class CloudpickleWrapper(object):
	"""
	Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
	"""

	def __init__(self, x):
		self.x = x

	def __getstate__(self):
		import cloudpickle
		return cloudpickle.dumps(self.x)

	def __setstate__(self, ob):
		import pickle
		self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
	"""
	An abstract asynchronous, vectorized environment.
	Used to batch data from multiple copies of an environment, so that
	each observation becomes an batch of observations, and expected action is a batch of actions to
	be applied per-environment.
	"""
	closed = False
	viewer = None

	metadata = {
		'render.modes': ['human', 'rgb_array']
	}

	def __init__(self, num_envs, observation_space, action_space):
		self.num_envs = num_envs
		self.observation_space = observation_space  # observation space of the individual env
		self.action_space = action_space  # action space of the individual env
		self._state = AsyncState.DEFAULT

	@abstractmethod
	def reset_async(self, seed=None, return_info=False, options=None):
		"""
		Reset all the environments and return an array of
		observations, or a dict of observation arrays.

		If step_async is still doing work, that work will
		be cancelled and step_wait() should not be called
		until step_async() is invoked again.
		"""
		pass
	
	@abstractmethod
	def reset_wait():
		pass

	@abstractmethod
	def step_async(self, actions, additional_info=None):
		"""
		Tell all the environments to start taking a step
		with the given actions.
		Call step_wait() to get the results of the step.

		You should not call this if a step_async run is
		already pending.
		"""
		pass

	@abstractmethod
	def step_wait(self):
		"""
		Wait for the step taken with step_async().

		Returns (obs, rews, dones, infos):
		 - obs: an array of observations, or a dict of
				arrays of observations.
		 - rews: an array of rewards
		 - dones: an array of "episode done" booleans
		 - infos: a sequence of info objects
		"""
		pass

	def close_extras(self):
		"""
		Clean up the  extra resources, beyond what's in this base class.
		Only runs when not self.closed.
		"""
		pass

	def close(self):
		if self.closed:
			return
		if self.viewer is not None:
			self.viewer.close()
		self.close_extras()
		self.closed = True

	def step(self, actions, additional_info):
		"""
		Step the environments synchronously.

		This is available for backwards compatibility.
		"""
		self.step_async(actions, additional_info)
		return self.step_wait()
	
	def reset(self, seed=None, return_info=False, options=None):
		self.reset_async(seed=seed, return_info=return_info, options=options)
		return self.reset_wait()

	def render(self, mode='human'):
		imgs = self.get_images()
		bigimg = tile_images(imgs)
		if mode == 'human':
			self.get_viewer().imshow(bigimg)
			return self.get_viewer().isopen
		elif mode == 'rgb_array':
			return bigimg
		else:
			raise NotImplementedError

	def get_images(self):
		"""
		Return RGB images from each environment
		"""
		raise NotImplementedError

	@property
	def unwrapped(self):
		return self

	def get_viewer(self):
		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.SimpleImageViewer()
		return self.viewer


def shareworker(remote, parent_remote, env_fn_wrapper, environment_name):
	with warnings.catch_warnings():  # ignoring Gym API deprecation warnings
		warnings.simplefilter("ignore")
		parent_remote.close()
		env = env_fn_wrapper.x()
		while True:
			cmd, data = remote.recv()
			if cmd == 'step':
				data, should_truncate, additional_info = data
				ob, reward, done, info = env.step(data)
				info["did_reset"] = False
				if done or should_truncate:
					if environment_name == "GFootball":
						next_ob = env.reset()
						# storing this info to generate state-value function
						next_info = {
							"__global_obs": additional_info["__global_obs"],
							"__last_actions": additional_info["__last_actions"],
							"__rnn_hidden_state_actor": additional_info["__rnn_hidden_state_actor"],
							"__mask_actions": additional_info["__mask_actions"],
							"__rnn_hidden_state_v": additional_info["__rnn_hidden_state_v"],
							}
					elif environment_name == "StarCraft":
						next_ob, next_info = env.reset(return_info=True)
						# storing this info to generate state-value function
						next_info["__last_actions"] = additional_info["__last_actions"]
						next_info["__rnn_hidden_state_actor"] = additional_info["__rnn_hidden_state_actor"]
						next_info["__rnn_hidden_state_v"] = additional_info["__rnn_hidden_state_v"]
					

					next_info["last_obs"] = ob
					next_info["last_info"] = info
					next_info["did_reset"] = True
					ob = next_ob
					info = next_info
				remote.send((ob, reward, done, info))
			elif cmd == 'reset':
				assert type(data) == dict
				seed = data["seed"] if "seed" in data.keys() else None
				return_info = data["return_info"] if "return_info" in data.keys() else False
				options = data["options"] if "options" in data.keys() else None
				if return_info:
					if environment_name == "GFootball":
						ob = env.reset()
						info = {}
						info["did_reset"] = True
					elif environment_name == "StarCraft":
						ob, info = env.reset(seed=seed, return_info=return_info, options=options)
						info["did_reset"] = True
				else:
					if environment_name == "GFootball":
						ob = env.reset()
					elif environment_name == "StarCraft":
						ob = env.reset(seed=seed, return_info=return_info, options=options)
					info = None
				remote.send((ob, info))
			elif cmd == 'render':
				if data == "rgb_array":
					fr = env.render(mode=data)
					remote.send(fr)
				elif data == "human":
					env.render(mode=data)
			elif cmd == 'close':
				env.close()
				remote.close()
				break
			elif cmd == 'get_spaces':
				remote.send(
					(env.observation_space, env.action_space))
			elif cmd == 'render_vulnerability':
				fr = env.render_vulnerability(data)
				remote.send((fr))
			else:
				raise NotImplementedError


class ShareSubprocVecEnv(ShareVecEnv):
	def __init__(self, env_fns, spaces=None, truncation_steps=None, environment_name=None):
		"""
		envs: list of gym environments to run in subprocesses
		"""
		self.waiting = False
		self.closed = False
		nenvs = len(env_fns)
		self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
		self.ps = [Process(target=shareworker, args=(work_remote, remote, CloudpickleWrapper(env_fn), environment_name))
				   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
		for p in self.ps:
			p.daemon = True  # if the main process crashes, we should not cause things to hang
			p.start()
		for remote in self.work_remotes:
			remote.close()
		self.remotes[0].send(('get_spaces', None))
		observation_space, action_space = self.remotes[0].recv()
		self._state: AsyncState = AsyncState.DEFAULT
		self.truncation_steps = truncation_steps
		self.num_steps = np.zeros(nenvs, dtype=int)
		self.lock = 

		ShareVecEnv.__init__(self, len(env_fns), observation_space, action_space)
	
	def _assert_is_running(self):
		if self.closed:
			raise ClosedEnvironmentError(
				f"Trying to operate on `{type(self).__name__}`, after a call to `close()`."
			)
		
	def _should_truncate(self, process_index):
		if self.truncation_steps == None:
			return False
		if self.num_steps[process_index] == self.truncation_steps:
			return True
		else:
			return False

	def step_async(self, actions, additional_info=None):
		self._assert_is_running()
		self.num_steps += 1
		if self._state != AsyncState.DEFAULT:
			raise AlreadyPendingCallError(
				f"Calling `step_async` while waiting for a pending call to `{self._state.value}` to complete.",
				self._state.value,
			)
		
		for process_index, (remote, action) in enumerate(zip(self.remotes, actions)):
			should_truncate = self._should_truncate(process_index)
			if should_truncate:
				self.num_steps[process_index] = 0
			remote.send(('step', (action, should_truncate, additional_info)))
		# self.num_steps += 1
		self.waiting = True
		self._state = AsyncState.WAITING_STEP

	def step_wait(self):
		self._assert_is_running()
		if self._state != AsyncState.WAITING_STEP:
			raise NoAsyncCallError(
				"Calling `step_wait` without any prior call " "to `step_async`.",
				AsyncState.WAITING_STEP.value,
			)
		
		results = [remote.recv() for remote in self.remotes]
		self._state = AsyncState.DEFAULT
		self.waiting = False
		obs, rews, dones, infos = zip(*results)
		done_mask = 1 - np.array(dones, dtype=int)
		self.num_steps *= done_mask  # making num_steps = 0 for processes whose episodes are terminated.
		combined_info = {}
		for i, info in enumerate(infos):
			combined_info = self._add_info(combined_info, info, i)
		return np.stack(obs), np.stack(rews), np.stack(dones), combined_info

	def reset_async(self, seed=None, return_info=False, options=None):
		self._assert_is_running()
		if seed is None:
			seed = [None for _ in range(self.num_envs)]

		if isinstance(seed, int):
			seed = [seed + i for i in range(self.num_envs)]

		assert len(seed) == self.num_envs

		if self._state != AsyncState.DEFAULT:
			raise AlreadyPendingCallError(
				f"Calling `reset_async` while waiting for a pending call to `{self._state.value}` to complete",
				self._state.value,
			)
		for remote, single_seed in zip(self.remotes, seed):
			data_dict = {
				"seed": single_seed,
				"return_info": return_info,
				"options": options
			}
			remote.send(('reset', data_dict))
		self._state = AsyncState.WAITING_RESET        
	
	def reset_wait(self):
		self._assert_is_running()
		if self._state != AsyncState.WAITING_RESET:
			raise NoAsyncCallError(
				"Calling `reset_wait` without any prior " "call to `reset_async`.",
				AsyncState.WAITING_RESET.value,
			)
		
		observations, infos = zip(*[pipe.recv() for pipe in self.remotes])
		self._state = AsyncState.DEFAULT

		self.num_steps = np.zeros(self.num_envs, dtype=int)
		
		if infos[0] is None:
			return np.stack(observations)
		else:
			combined_info = {}
			for i, info in enumerate(infos):
				combined_info = self._add_info(combined_info, info, i)
			return np.stack(observations), combined_info
		
	def _combine_infos(self, infos):
		assert len(infos) == self.num_envs
		combined_dict = {}
		separate_infos = [{} for _ in range(self.num_envs)]
		for i, info in enumerate(infos):
			for k, v in info.items():
				if k in ["next_obs", "next_info"]:
					separate_infos[i][k] = v
				else:
					if k in combined_dict:
						combined_dict[k].append(np.array(v) if type(v) != np.ndarray else v)
					else:
						combined_dict[k] = [np.array(v) if type(v) != np.ndarray else v]

		return {"combined": combined_dict, "separate": separate_infos}
	
	def _add_info(
		self, vector_infos: Dict[str, Any], env_info: Dict[str, Any], env_num: int
	) -> Dict[str, Any]:
		"""Add env info to the info dictionary of the vectorized environment.

		Given the `info` of a single environment add it to the `infos` dictionary
		which represents all the infos of the vectorized environment.
		Every `key` of `info` is paired with a boolean mask `_key` representing
		whether or not the i-indexed environment has this `info`.

		Args:
			vector_infos (dict): the infos of the vectorized environment
			env_info (dict): the info coming from the single environment
			env_num (int): the index of the single environment

		Returns:
			infos (dict): the (updated) infos of the vectorized environment
		"""
		for key, value in env_info.items():
			# If value is a dictionary, then we apply the `_add_info` recursively.
			if isinstance(value, dict):
				array = self._add_info(vector_infos.get(key, {}), value, env_num)
			# Otherwise, we are a base case to group the data
			else:
				# If the key doesn't exist in the vector infos, then we can create an array of that batch type
				if key not in vector_infos:
					if type(value) in [int, float, bool] or issubclass(
						type(value), np.number
					):
						array = np.zeros(self.num_envs, dtype=type(value))
					elif isinstance(value, np.ndarray):
						# We assume that all instances of the np.array info are of the same shape
						if "__" in key: # we are saving the entire batch (self.num_workers, data dim) in additional_info so after every info dict update capture data neatly
							array = np.zeros(
								(self.num_envs, *value.shape[1:]), dtype=value.dtype
							)
						else:
							array = np.zeros(
								(self.num_envs, *value.shape), dtype=value.dtype
							)
					elif isinstance(value, list):
						value = np.array(value)
						if "__" in key: # we are saving the entire batch (self.num_workers, data dim) in additional_info so after every info dict update capture data neatly
							array = np.zeros(
								(self.num_envs, *value.shape[1:])
							)
						else:
							array = np.zeros(
								(self.num_envs, *value.shape)
							)
					else:
						# For unknown objects, we use a Numpy object array
						array = np.full(self.num_envs, fill_value=None, dtype=object)
				# Otherwise, just use the array that already exists
				else:
					array = vector_infos[key]

				# Assign the data in the `env_num` position
				#   We only want to run this for the base-case data (not recursive data forcing the ugly function structure)
				if "__" in key: # we are saving the entire batch (self.num_workers, data dim) in additional_info so after every info dict update capture data neatly
					array[env_num] = value[env_num]
				else:
					array[env_num] = value

			# Get the array mask and if it doesn't already exist then create a zero bool array
			array_mask = vector_infos.get(
				f"_{key}", np.zeros(self.num_envs, dtype=np.bool_)
			)
			array_mask[env_num] = True

			# Update the vector info with the updated data and mask information
			vector_infos[key], vector_infos[f"_{key}"] = array, array_mask

		return vector_infos

	def close(self):
		if self.closed:
			return
		if self.waiting:
			for remote in self.remotes:
				remote.recv()
		for remote in self.remotes:
			remote.send(('close', None))
		for p in self.ps:
			p.join()
		self.closed = True
