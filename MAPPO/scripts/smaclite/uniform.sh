#!/bin/bash

# Uniform baseline - simple uniform reward redistribution
# No reward model needed, just splits episodic reward equally

python train_agent.py \
    --iteration 1 \
    --learn \
    --device gpu \
    --environment StarCraft \
    --env 3s5z \
    --experiment_type "Uniform" \
    --max_episodes 30000 \
    --max_time_steps 100 \
    --ppo_eps_elapse_update_freq 10 \
    --policy_lr 5e-4 \
    --v_value_lr 5e-4 \
    --entropy_pen 6e-3 \
    --save_model \
    --save_comet_ml_plot \
    --test_num "Learning_Reward_Func_for_Credit_Assignment"