#!/bin/bash

python train_agent.py \
    --iteration 1 \
    --learn \
    --use_reward_model \
    --device gpu \
    --environment GFootball \
    --env academy_3_vs_1_with_keeper \
    --experiment_type "TAR^2" \
    --version "original" \
    --max_episodes 30000 \
    --max_time_steps 200 \
    --ppo_eps_elapse_update_freq 5 \
    --reward_lr 1e-4 \
    --policy_lr 5e-4 \
    --v_value_lr 5e-4 \
    --reward_depth 3 \
    --reward_n_heads 4 \
    --reward_linear_compression_dim 64 \
    --dynamic_loss_coeffecient 5e-2 \
    --entropy_pen 8e-3 \
    --save_model \
    --save_comet_ml_plot \
    --test_num "Learning_Reward_Func_for_Credit_Assignment_GRF"