#!/bin/bash

# Uniform baseline for Google Research Football
# Simple uniform reward redistribution

python train_agent.py \
    --iteration 1 \
    --learn \
    --device gpu \
    --environment GFootball \
    --env academy_3_vs_1_with_keeper \
    --experiment_type "Uniform" \
    --max_episodes 30000 \
    --max_time_steps 200 \
    --ppo_eps_elapse_update_freq 5 \
    --policy_lr 5e-4 \
    --v_value_lr 5e-4 \
    --entropy_pen 8e-3 \
    --save_model \
    --save_comet_ml_plot \
    --test_num "Learning_Reward_Func_for_Credit_Assignment_GRF"