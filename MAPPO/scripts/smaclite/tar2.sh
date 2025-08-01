#!/bin/bash

# This is an example script to launch a TAR² training run.
# You can create similar scripts for each of your baselines (STAS, AREL, etc.)
# by changing the --experiment_type argument.

# To run 5 seeds, you could use a loop:
# for seed in 1 2 3 4 5
# do
#   python train_agent.py --iteration $seed ...
# done

python train_agent.py \
    --iteration 1 \
    --device gpu \
    --environment StarCraft \
    --env 3s5z \
    --experiment_type "TAR^2" \
    --max_episodes 30000 \
    --max_time_steps 100 \
    --ppo_eps_elapse_update_freq 10 \
    --reward_lr 1e-4 \
    --policy_lr 5e-4 \
    --v_value_lr 5e-4 \
    --reward_depth 3 \
    --reward_n_heads 4 \
    --reward_linear_compression_dim 64 \
    --dynamic_loss_coeffecient 5e-2 \
    --entropy_pen 6e-3 \
    --save_model \
    --save_comet_ml_plot \
    --test_num "Learning_Reward_Func_for_Credit_Assignment"