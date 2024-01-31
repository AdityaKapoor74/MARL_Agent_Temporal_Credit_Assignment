import os
import time
from comet_ml import Experiment
import numpy as np
from agent import PPOAgent
import torch
import torch.nn.functional as F
import datetime

import gym
import smaclite  # noqa
from env_wrappers import ShareSubprocVecEnv
import atexit
torch.autograd.set_detect_anomaly(True)

class MAPPO:

    def __init__(self, env_fns, dictionary):

        if dictionary["device"] == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.envs = ShareSubprocVecEnv(env_fns, truncation_steps=dictionary["max_time_steps"], is_sync=True)
        dummy_env = env_fns[0]()
        self.gif = dictionary["gif"]
        self.save_model = dictionary["save_model"]
        self.save_model_checkpoint = dictionary["save_model_checkpoint"]
        self.save_comet_ml_plot = dictionary["save_comet_ml_plot"]
        self.learn = dictionary["learn"]
        self.gif_checkpoint = dictionary["gif_checkpoint"]
        self.eval_policy = dictionary["eval_policy"]
        self.num_agents = dummy_env.n_agents
        self.num_enemies = dummy_env.n_enemies
        self.num_actions = dummy_env.action_space[0].n
        self.date_time = f"{datetime.datetime.now():%d-%m-%Y}"
        self.env_name = dictionary["env"]
        self.test_num = dictionary["test_num"]
        self.max_episodes = dictionary["max_episodes"]
        self.max_time_steps = dictionary["max_time_steps"]
        self.experiment_type = dictionary["experiment_type"]
        self.update_ppo_agent = dictionary["update_ppo_agent"]
        self.use_reward_model = dictionary["use_reward_model"]
        self.reward_warmup = dictionary["reward_warmup"]
        self.update_reward_model_freq = dictionary["update_reward_model_freq"]
        self.reward_model_update_epochs = dictionary["reward_model_update_epochs"]
        self.fine_tune_epochs = dictionary["fine_tune_epochs"]
        self.batch_size = dictionary["batch_size"]
        self.num_workers = dictionary["num_workers"]
        assert len(env_fns) == self.num_workers, f"number of rollout threads ({self.num_workers}) doesn't match the number of env functions ({len(env_fns)})"
        atexit.register(self.close)

        # RNN HIDDEN
        self.rnn_num_layers_q = dictionary["rnn_num_layers_q"]
        self.rnn_hidden_q = dictionary["rnn_hidden_q"]
        self.rnn_num_layers_actor = dictionary["rnn_num_layers_actor"]
        self.rnn_hidden_actor = dictionary["rnn_hidden_actor"]

        self.agent_ids = np.stack([np.eye(self.num_agents) for _ in range(self.num_workers)])
        self.enemy_ids = np.stack([np.eye(self.num_enemies) for _ in range(self.num_workers)])

        self.comet_ml = None
        if self.save_comet_ml_plot:
            self.comet_ml = Experiment("im5zK8gFkz6j07uflhc3hXk8I", project_name=dictionary["test_num"])
            self.comet_ml.log_parameters(dictionary)


        self.agents = PPOAgent(dummy_env, dictionary, self.comet_ml)  # passing dummy_env here since dummy_env is enough to set the required parameters

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

            optim_dir = dictionary["optim_dir"]
            try: 
                os.makedirs(optim_dir, exist_ok = True) 
                print("Optim Directory created successfully") 
            except OSError as error: 
                print("Optim Directory can not be created")	
            
            self.critic_model_path = dictionary["critic_dir"]
            self.actor_model_path = dictionary["actor_dir"]
            self.optim_model_path = dictionary["optim_dir"]

            if self.use_reward_model:
                reward_dir = dictionary["reward_dir"]
                try: 
                    os.makedirs(reward_dir, exist_ok = True) 
                    print("Reward Directory created successfully") 
                except OSError as error: 
                    print("Reward Directory can not be created")

                self.reward_model_path = dictionary["reward_dir"] 
            

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

    def close(self):
        print("Cleaning up")
        self.envs.close()


    def run(self):
        if self.eval_policy:
            self.rewards = []
            self.rewards_mean_per_1000_eps = []
            self.timesteps = []
            self.timesteps_mean_per_1000_eps = []

        self.reward_plot_counter = 0
        self.num_episodes_done = 0
        self.worker_step_counter = np.zeros(self.num_workers, dtype=np.int64)

        
        while self.num_episodes_done < self.max_episodes:
            print("Resetting the environment")
            states_actor, infos = self.envs.reset(return_info=True)
            self.worker_dones = np.array([False] * self.num_workers, dtype=bool)
            self.worker_step_counter = np.zeros(self.num_workers, dtype=np.int64)


            ## num_agents = 5, num_enemies = 6, num_actions = 12
            mask_actions = infos["avail_actions"].astype(int)  # mask_actions.shape = (num_workers, 5, 12)
            last_one_hot_actions = np.zeros((self.num_workers, self.num_agents, self.num_actions))
            states_allies_critic = np.concatenate((self.agent_ids, infos["ally_states"]), axis=-1)  # states_allies_critic.shape = (num_workers, 5, 9)
            states_enemies_critic = np.concatenate((self.enemy_ids, infos["enemy_states"]), axis=-1)  # states_enemies_critic.shape = (num_workers, 6, 9)
            states_actor = np.concatenate((self.agent_ids, states_actor), axis=-1)  # states_actor.shape = (num_workers, 5, 60)
            indiv_dones = np.zeros((self.num_workers, self.num_agents), dtype=np.int64)  # indiv_dones.shape = (num_workers, 5)

            action_list = []

            episodic_team_reward = [[0]*self.num_agents for _ in range(self.num_workers)]
            episodic_agent_reward = [[0]*self.num_agents for _ in range(self.num_workers)]
            

            images = []

            episode_reward = [0 for _ in range(self.num_workers)]
            episode_indiv_rewards = [[0 for i in range(self.num_agents)] for _ in range(self.num_workers)]

            rnn_hidden_state_q = np.zeros((self.num_workers, self.rnn_num_layers_q, self.num_agents, self.rnn_hidden_q))
            rnn_hidden_state_actor = np.zeros((self.num_workers, self.rnn_num_layers_actor, self.num_agents, self.rnn_hidden_actor))


            while not np.all(self.worker_dones):
                actions, action_logprob, next_rnn_hidden_state_actor = self.agents.get_action_batch(states_actor, last_one_hot_actions, mask_actions, rnn_hidden_state_actor)
                # actions -> list; np.array(actions).shape = (num_workers, 5)
                # action_logprob.shape = (num_workers, 5)
                # next_rnn_hidden_state_actor.shape = (num_workers, rnn_num_layers_actor, 5, 64)

                action_list.append(actions)
                one_hot_actions = F.one_hot(torch.tensor(actions), num_classes=self.num_actions).cpu().numpy()  # one_hot_actions.shape = (3, 5, 12)
            
                q_value, next_rnn_hidden_state_q = self.agents.get_q_values_batch(states_allies_critic, states_enemies_critic, one_hot_actions, rnn_hidden_state_q, indiv_dones)
                # q_value.shape (num_workers, 5)
                # next_rnn_hidden_state_q.shape (num_workers, rnn_num_layers_q, 5, 64)

                next_states_actor, rewards, next_dones, info = self.envs.step(actions)
                print(next_dones)
                print("")
                self.worker_step_counter += 1  # increments for all the workers

            
                next_states_actor = np.array(next_states_actor)
                next_states_actor = np.concatenate((self.agent_ids, next_states_actor), axis=-1)
                if "ally_states" not in info:
                    print(info.keys())
                next_states_allies_critic = np.concatenate((self.agent_ids, info["ally_states"]), axis=-1)
                next_states_enemies_critic = np.concatenate((self.enemy_ids, info["enemy_states"]), axis=-1)
                next_mask_actions = np.array(info["avail_actions"], dtype=int)
                next_indiv_dones = info["indiv_dones"]
                assert(states_actor.shape == next_states_actor.shape)
                assert(states_allies_critic.shape == next_states_allies_critic.shape)
                assert(states_enemies_critic.shape == next_states_enemies_critic.shape)
                assert(mask_actions.shape == next_mask_actions.shape)
                assert(indiv_dones.shape == next_indiv_dones.shape)
                # next_states_allies_critic.shape = (num_workers, 5, 9)
                # next_states_enemies_critic.shape = (num_workers, 6, 9)
                # next_states_actor.shape = (num_workers, 5, 60)
                # next_mask_actions.shape = (num_workers, 5, 12)
                # next_indiv_dones.shape = (num_workers, 5)

                if self.learn:
                    if self.experiment_type == "temporal_team":
                        rewards_to_send = []
                        for worker_index in range(self.num_workers):
                            rewards_to_send.append(np.array([rewards[worker_index]] * self.num_agents))
                    
                    elif self.experiment_type == "temporal_agent":
                        rewards_to_send = []
                        for worker_index in range(self.num_workers):
                            if self.worker_dones[i]:
                                rewards_to_send.append(np.zeros(self.num_agents))  # rewards_to_send for a worker whose episode has finished will have dummy data; it won't be stored either.
                            else:
                                rewards_to_send.append(info["indiv_rewards"][worker_index])
                    
                    elif self.experiment_type == "episodic_team":
                        rewards_to_send = []
                        for worker_index in range(self.num_workers):
                            if self.worker_dones[worker_index]:
                                rewards_to_send.append(np.zeros(self.num_agents))
                                continue
                            episodic_team_reward[worker_index] = [r+rewards[worker_index] for r in episodic_team_reward[worker_index]]
                            if next_dones[worker_index] or self.worker_step_counter[worker_index] == self.max_time_steps:
                                rewards_to_send.append(np.array(episodic_team_reward[worker_index]))
                            else:
                                rewards_to_send.append(np.zeros(self.num_agents))
                    
                    elif self.experiment_type == "episodic_agent":
                        rewards_to_send = []
                        for worker_index in range(self.num_workers):
                            if self.worker_dones[worker_index]:
                                rewards_to_send.append(np.zeros(self.num_agents))
                                continue
                            individual_rewards = info["indiv_rewards"][worker_index]
                            episodic_agent_reward[worker_index] = [a_r+r for a_r, r in zip(episodic_agent_reward[worker_index], individual_rewards)]
                            if (not next_dones[worker_index]) and (self.worker_step_counter[worker_index] == self.max_time_steps):
                                rewards_to_send.append(np.array(episodic_agent_reward[worker_index]))
                            else:
                                rewards_to_send_worker = []
                                next_individual_dones = info["indiv_dones"][worker_index]
                                for i, d in enumerate(next_individual_dones):
                                    if d:
                                        rewards_to_send_worker.append(episodic_agent_reward[worker_index][i])
                                        # once reward is allocated to agent, make future rewards 0
                                        episodic_agent_reward[worker_index][i] = 0
                                    else:
                                        rewards_to_send_worker.append(0.0)
                                rewards_to_send.append(np.array(rewards_to_send_worker))
                    
                    elif self.experiment_type == "AREL" or self.experiment_type == "ATRR":
                        rewards_to_send = []
                        for worker_index in range(self.num_workers):
                            if self.worker_dones[worker_index]:
                                rewards_to_send.append(np.zeros(self.num_agents))
                                continue
                            episodic_team_reward[worker_index] = [r+rewards[worker_index] for r in episodic_team_reward[worker_index]]
                            if next_dones[worker_index] or self.worker_step_counter[worker_index] == self.max_time_steps:
                                rewards_to_send.append(np.array(episodic_team_reward[worker_index]))
                            else:
                                rewards_to_send.append(np.zeros(self.num_agents))
                    rewards_to_send = np.array(rewards_to_send)
                    self.agents.buffer.push(
                        states_allies_critic, states_enemies_critic, q_value, rnn_hidden_state_q, \
                        states_actor, rnn_hidden_state_actor, action_logprob, np.array(actions), last_one_hot_actions, one_hot_actions, mask_actions, \
                        rewards_to_send, indiv_dones, self.worker_step_counter, masks=self.worker_dones
                    )

                    if self.use_reward_model:
                        self.agents.reward_buffer.push(
                            states_actor, one_hot_actions, indiv_dones, masks=self.worker_dones
                        )

                states_actor, last_one_hot_actions, states_allies_critic, states_enemies_critic, mask_actions, indiv_dones = next_states_actor, one_hot_actions, next_states_allies_critic, next_states_enemies_critic, next_mask_actions, info["indiv_dones"]
                rnn_hidden_state_q, rnn_hidden_state_actor = next_rnn_hidden_state_q, next_rnn_hidden_state_actor
            
                for worker_index in range(self.num_workers):
                    if self.worker_dones[worker_index]: continue
                    episode_reward[worker_index] += rewards[worker_index]
                    individual_rewards = info["indiv_rewards"][worker_index]
                    episode_indiv_rewards[worker_index] = [r+individual_rewards[i] for i, r in enumerate(episode_indiv_rewards[worker_index])]


                    if all(indiv_dones[worker_index]) or self.worker_step_counter[worker_index] == self.max_time_steps:
                        self.num_episodes_done += 1
                        self.worker_dones[worker_index] = True
                        step = self.worker_step_counter[worker_index]
                        print(f"Worker {worker_index} done!")
                        print("*"*100)
                        print(
                            "EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} | Num Allies Alive: {} | Num Enemies Alive: {} \n".format(
                                self.num_episodes_done,
                                np.round(episode_reward[worker_index], decimals=4),
                                step,
                                self.max_time_steps,
                                info["num_allies"][worker_index],
                                info["num_enemies"][worker_index],
                            )
                        )
                        
                        print("INDIV REWARD STREAMS", episode_indiv_rewards[worker_index], "AGENTS DEAD", info["indiv_dones"][worker_index])
                        print("*"*100)

                        if self.use_reward_model:
                            predicted_episode_reward = self.agents.evaluate_reward_model()
                            # print(action_list)

                        if self.save_comet_ml_plot:
                            self.comet_ml.log_metric('Episode_Length', step, self.num_episodes_done)
                            self.comet_ml.log_metric('Reward', episode_reward[worker_index], self.num_episodes_done)
                            self.comet_ml.log_metric('Num Enemies', info["num_enemies"][worker_index], self.num_episodes_done)
                            self.comet_ml.log_metric('Num Allies',  info["num_allies"][worker_index], self.num_episodes_done)
                            self.comet_ml.log_metric('All Enemies Dead', info["all_enemies_dead"][worker_index], self.num_episodes_done)
                            self.comet_ml.log_metric('All Allies Dead', info["all_allies_dead"][worker_index], self.num_episodes_done)

                            if self.use_reward_model:
                                self.comet_ml.log_metric('Predicted Reward', predicted_episode_reward, self.num_episodes_done)

                        # update entropy params
                        self.agents.update_parameters()

                        if self.learn:

                            last_states_actor = np.expand_dims(states_actor[worker_index], axis=0)
                            last_states_allies_critic = np.expand_dims(np.concatenate((self.agent_ids[worker_index], info["ally_states"][worker_index]), axis=-1), axis=0)
                            last_states_enemies_critic = np.expand_dims(np.concatenate((self.enemy_ids[worker_index], info["enemy_states"][worker_index]), axis=-1), axis=0)
                            last_mask_actions = np.expand_dims(np.array(info["avail_actions"][worker_index], dtype=int), axis=0)
                            last_indiv_dones = np.expand_dims(info["indiv_dones"][worker_index], axis=0)


                            # add final time to buffer
                            actions, action_logprob, next_rnn_hidden_state_actor = self.agents.get_action_batch(last_states_actor, 
                                                                                                                np.expand_dims(last_one_hot_actions[worker_index], axis=0), 
                                                                                                                last_mask_actions, 
                                                                                                                np.expand_dims(rnn_hidden_state_actor[worker_index], axis=0))
                            assert np.array(actions).shape == (1, self.num_agents)
                            assert action_logprob.shape == (1, self.num_agents)
                            assert next_rnn_hidden_state_actor.shape == (1, self.rnn_num_layers_actor, self.num_agents, self.rnn_hidden_actor)
                            # actions -> list; np.array(actions).shape = (1, 5)
                            # action_logprob.shape = (1, 5)
                            # next_rnn_hidden_state_actor.shape = (1, rnn_num_layers_actor, 5, 64)
                        
                            one_hot_actions = F.one_hot(torch.tensor(actions), num_classes=self.num_actions).cpu().numpy()  # one_hot_actions.shape = (1, 5, 12)

                            q_value, _ = self.agents.get_q_values_batch(last_states_allies_critic, 
                                                                        last_states_enemies_critic, 
                                                                        one_hot_actions, 
                                                                        np.expand_dims(rnn_hidden_state_q[worker_index], axis=0), 
                                                                        last_indiv_dones)
                            
                            assert q_value.shape == (1, self.num_agents)

                            self.agents.buffer.end_episode(np.array([step]), q_value, last_indiv_dones, [worker_index])

                            if self.use_reward_model:
                                self.agents.reward_buffer.end_episode(np.array([episode_reward[worker_index]]), np.array([step]), [worker_index])

                        if self.agents.scheduler_need:
                            self.agents.scheduler_policy.step()
                            self.agents.scheduler_q_critic.step()
                            self.agents.scheduler_reward.step()

                        if self.eval_policy:
                            self.rewards.append(episode_reward[worker_index])
                            self.timesteps.append(step)

                        if self.num_episodes_done > self.save_model_checkpoint and self.eval_policy:
                            self.rewards_mean_per_1000_eps.append(sum(self.rewards[self.num_episodes_done-self.save_model_checkpoint:self.num_episodes_done])/self.save_model_checkpoint)
                            self.timesteps_mean_per_1000_eps.append(sum(self.timesteps[self.num_episodes_done-self.save_model_checkpoint:self.num_episodes_done])/self.save_model_checkpoint)


                        if not(self.num_episodes_done%self.save_model_checkpoint) and self.num_episodes_done!=0 and self.save_model:	
                            # save actor, critic, reward and optims
                            torch.save(self.agents.critic_network_q.state_dict(), self.critic_model_path+'critic_Q_epsiode_'+str(self.num_episodes_done)+'.pt')
                            torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'actor_epsiode_'+str(self.num_episodes_done)+'.pt')
                            torch.save(self.agents.q_critic_optimizer.state_dict(), self.optim_model_path+'critic_optim_epsiode_'+str(self.num_episodes_done)+'.pt')
                            torch.save(self.agents.policy_optimizer.state_dict(), self.optim_model_path+'policy_optim_epsiode_'+str(self.num_episodes_done)+'.pt')  

                            if self.use_reward_model:
                                torch.save(self.agents.reward_model.state_dict(), self.reward_model_path+'reward_model_epsiode_'+str(self.num_episodes_done)+'.pt')
                                torch.save(self.agents.reward_optimizer.state_dict(), self.reward_model_path+'reward_optim_epsiode_'+str(self.num_episodes_done)+'.pt')
                
                        if self.use_reward_model:
                            if self.num_episodes_done % self.update_reward_model_freq == 0 and self.agents.reward_buffer.episodes_filled >= self.batch_size:
                                for i in range(self.reward_model_update_epochs):
                                    if i < self.reward_model_update_epochs - self.fine_tune_epochs:
                                        if self.experiment_type == "AREL":
                                            loss, reward_var, grad_norm_value_reward = self.agents.update_reward_model(fine_tune=False)
                                        elif self.experiment_type == "ATRR":
                                            loss, temporal_weights_entropy, agent_weights_entropy, grad_norm_value_reward = self.agents.update_reward_model(fine_tune=False)
                                    else:
                                        if self.experiment_type == "AREL":
                                            loss, reward_var, grad_norm_value_reward = self.agents.update_reward_model(fine_tune=True)
                                        elif self.experiment_type == "ATRR":
                                            loss, temporal_weights_entropy, agent_weights_entropy, grad_norm_value_reward = self.agents.update_reward_model(fine_tune=True)
                                    
                                    if self.save_comet_ml_plot:
                                        self.comet_ml.log_metric("Reward_Loss", loss.item(), step=self.reward_plot_counter)
                                        if self.experiment_type == "AREL":
                                            self.comet_ml.log_metric("Reward_Var", reward_var.item(), step=self.reward_plot_counter)
                                        elif self.experiment_type == "ATRR":
                                            self.comet_ml.log_metric("Temporal Weights Entropy", temporal_weights_entropy.item(), step=self.reward_plot_counter)
                                            self.comet_ml.log_metric("Agent Weights Entropy", agent_weights_entropy.item(), step=self.reward_plot_counter)
                                        
                                        self.comet_ml.log_metric("Reward_Grad_Norm", grad_norm_value_reward.item(), step=self.reward_plot_counter)

                                        self.reward_plot_counter += 1

                        if self.learn and self.agents.should_update_agent():
                            if self.use_reward_model is False:
                                self.agents.update(self.num_episodes_done)
                            else:
                                if self.num_episodes_done >= self.reward_warmup:
                                    self.agents.update(self.num_episodes_done)
                                else:
                                    self.agents.buffer.clear()


if __name__ == '__main__':

    RENDER = False
    USE_CPP_RVO2 = False

    for i in range(1,6):
        extension = "MAPPO_"+str(i)
        test_num = "Learning_Reward_Func_for_Credit_Assignment"
        env_name = "5m_vs_6m"
        experiment_type = "ATRR" # episodic_team, episodic_agent, temporal_team, temporal_agent, AREL, SeqModel, RUDDER

        dictionary = {
                # TRAINING
                "iteration": i,
                "device": "gpu",
                "critic_dir": 'tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
                "actor_dir": 'tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
                "reward_dir": 'tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/reward_networks/',
                "gif_dir": 'tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
                "policy_eval_dir":'tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
                "optim_dir": 'tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/optimizer_models/',
                "n_epochs": 5,
                "update_ppo_agent": 10, # update ppo agent after every update_ppo_agent episodes
                "test_num": test_num,
                "extension": extension,
                "gamma": 0.99,
                "gif": False,
                "gif_checkpoint":1,
                "load_models": False,
                "model_path_value": "tests/MultiAgentCreditAssignment/models/5m_vs_6m_temporal_team_MAPPO_1/critic_networks/critic_epsiode_100000.pt",
                "model_path_policy": "tests/MultiAgentCreditAssignment/models/5m_vs_6m_temporal_team_MAPPO_1/actor_networks/actor_epsiode_100000.pt",
                "model_path_policy_optimizer": "tests/MultiAgentCreditAssignment/models/5m_vs_6m_temporal_team_MAPPO_1/critic_networks/critic_optim_epsiode_100000.pt",
                "model_path_q_critic_optimizer": "tests/MultiAgentCreditAssignment/models/5m_vs_6m_temporal_team_MAPPO_1/critic_networks/policy_optim_epsiode_100000.pt",
                "eval_policy": True,
                "save_model": True,
                "save_model_checkpoint": 1000,
                "save_comet_ml_plot": False,
                "learn":True,
                "max_episodes": 50000,
                "max_time_steps": 100,
                "experiment_type": experiment_type,
                "parallel_training": True,
                "num_workers": 10,
                "scheduler_need": False,


                # ENVIRONMENT
                "env": env_name,

                # REWARD MODEL
                "use_reward_model": True,
				"reward_n_heads": 3, # 3
				"reward_depth": 3, # 3
				"reward_agent_attn": True,
				"reward_dropout": 0.0,
				"reward_attn_net_wide": True,
				"reward_comp": "hypernet_compression", # no_compression, linear_compression, hypernet_compression
				"reward_linear_compression_dim": 128,
				"reward_hypernet_hidden_dim": 64,
				"reward_hypernet_final_dim": 64,
				"num_episodes_capacity": 2000, # 40000
				"batch_size": 32, # 128
				"reward_lr": 1e-4,
				"reward_weight_decay": 1e-5,
				"variance_loss_coeff": 0.0,
				"enable_reward_grad_clip": False,
				"reward_grad_clip_value": 0.5,
				"reward_warmup": 5000, # 1000
				"update_reward_model_freq": 100, # 200
				"reward_model_update_epochs": 100, # 100
				"fine_tune_epochs": 1, # 10
				"fine_tune_reward_lr": 1e-4,
				"fine_tune_batch_size": 10,
				"norm_rewards": False,
				"clamp_rewards": False,
				"clamp_rewards_value_min": 0.0,
				"clamp_rewards_value_max": 2.0,

                # CRITIC
                "rnn_num_layers_q": 1,
                "rnn_hidden_q": 64,
                "q_value_lr": 5e-4, #1e-3
                "temperature_q": 1.0,
                "attention_dropout_prob_q": 0.0,
                "q_weight_decay": 0.0,
                "enable_grad_clip_critic_q": True,
                "grad_clip_critic_q": 0.5,
                "value_clip": 0.2,
                "num_heads": 1,
                "critic_weight_entropy_pen": 0.0,
                "critic_weight_entropy_pen_final": 0.0,
                "critic_weight_entropy_pen_steps": 100, # number of updates
                "critic_score_regularizer": 0.0,
                "target_calc_style": "GAE", # GAE, TD_Lambda, N_steps
                "td_lambda": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
                "n_steps": 5,
                "norm_returns_q": True,
                "soft_update_q": False,
                "tau_q": 0.05,
                "network_update_interval_q": 1,
                

                # ACTOR
                "data_chunk_length": 10,
                "rnn_num_layers_actor": 1,
                "rnn_hidden_actor": 64,
                "enable_grad_clip_actor": True,
                "grad_clip_actor": 0.5,
                "policy_clip": 0.2,
                "policy_lr": 5e-4, #prd 1e-4
                "policy_weight_decay": 0.0,
                "entropy_pen": 1e-2, #8e-3
                "entropy_pen_final": 1e-2,
                "entropy_pen_steps": 2000,
                "gae_lambda": 0.95,
                "norm_adv": True,
            }
        assert dictionary["update_ppo_agent"] % dictionary["num_workers"] == 0
        seeds = [42, 142, 242, 342, 442]
        torch.manual_seed(seeds[dictionary["iteration"]-1])
        env = gym.make(f"smaclite/{env_name}-v0", use_cpp_rvo2=USE_CPP_RVO2)
        obs, info = env.reset(return_info=True)
        dictionary["ally_observation"] = info["ally_states"][0].shape[0]+env.n_agents #+env.action_space[0].n #4+env.action_space[0].n+env.n_agents
        dictionary["enemy_observation"] = info["enemy_states"][0].shape[0]+env.n_enemies
        dictionary["local_observation"] = obs[0].shape[0]+env.n_agents
        env_fns = [lambda: gym.make(f"smaclite/{env_name}-v0", use_cpp_rvo2=USE_CPP_RVO2) for _ in range(dictionary["num_workers"])]
        ma_controller = MAPPO(env_fns,dictionary)
        ma_controller.run()


# TRAIN EPISODIC_TEAM with full episode length