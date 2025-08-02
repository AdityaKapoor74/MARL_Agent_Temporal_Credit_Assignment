#!/bin/bash

# STAS (Spatio-Temporal Attention-based Shapley) baseline
# This method directly predicts shaped rewards using attention mechanisms

python train_agent.py \
    --iteration 1 \
    --learn \
    --use_reward_model \
    --device gpu \
    --environment StarCraft \
    --env 3s5z \
    --experiment_type "STAS" \
    --max_episodes 30000 \
    --max_time_steps 100 \
    --ppo_eps_elapse_update_freq 10 \
    --reward_lr 1e-4 \
    --policy_lr 5e-4 \
    --v_value_lr 5e-4 \
    --reward_depth 3 \
    --reward_n_heads 4 \
    --reward_linear_compression_dim 64 \
    --reward_batch_size 64 \
    --reward_weight_decay 0.0 \
    --enable_reward_grad_clip \
    --reward_grad_clip_value 0.5 \
    --update_reward_model_freq 100 \
    --reward_model_update_epochs 200 \
    --entropy_pen 6e-3 \
    --save_model \
    --save_comet_ml_plot \
    --test_num "Learning_Reward_Func_for_Credit_Assignment"