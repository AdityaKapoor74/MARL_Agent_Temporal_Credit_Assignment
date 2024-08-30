import random

import gfootball.env as football_env
from gym import spaces
import numpy as np


class FootballEnv(object):
    '''Wrapper to make Google Research Football environment compatible'''

    def __init__(self):
        self.num_agents = 4 # args.num_agents
        self.scenario_name = "academy_counterattack_easy" # args.scenario_name
        
        # make env
        # if not (args.use_render and args.save_videos):
        #     self.env = football_env.create_environment(
        #         env_name=args.scenario_name,
        #         stacked=args.use_stacked_frames,
        #         representation=args.representation,
        #         rewards=args.rewards,
        #         number_of_left_players_agent_controls=args.num_agents,
        #         number_of_right_players_agent_controls=0,
        #         channel_dimensions=(args.smm_width, args.smm_height),
        #         render=(args.use_render and args.save_gifs)
        #     )
        # else:
        #     # render env and save videos
        #     self.env = football_env.create_environment(
        #         env_name=args.scenario_name,
        #         stacked=args.use_stacked_frames,
        #         representation=args.representation,
        #         rewards=args.rewards,
        #         number_of_left_players_agent_controls=args.num_agents,
        #         number_of_right_players_agent_controls=0,
        #         channel_dimensions=(args.smm_width, args.smm_height),
        #         # video related params
        #         write_full_episode_dumps=True,
        #         render=True,
        #         write_video=True,
        #         dump_frequency=1,
        #         logdir=args.video_dir
        #     )

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



env = FootballEnv()
obs = env.reset()
actions = [0, 1, 2, 3]
obs, reward, done, info = env.step(actions)
print(obs.shape, reward, done, info)


# env="Football"
# scenario="academy_counterattack_easy"
# algo="rmappo" # "mappo" "ippo"
# exp="check"
# seed=1

# # football param
# num_agents=4

# # train param
# num_env_steps=25000000
# episode_length=200

# echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

# CUDA_VISIBLE_DEVICES=0 python ../train/train_football.py \
# --env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
# --num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
# --representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 50 --ppo_epoch 15 --num_mini_batch 2 \
# --save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
# --user_name "yuchao" --wandb_name "xxx" 