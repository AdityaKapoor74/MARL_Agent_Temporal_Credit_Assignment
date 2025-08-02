# Temporal-Agent Reward Redistribution (TAR²): Multi-Agent Credit Assignment

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/arXiv-2502.04864-b31b1b.svg)](https://arxiv.org/pdf/2502.04864)

This repository contains the official implementation of **TAR² (Temporal-Agent Reward Redistribution)**, a novel approach for joint agent-temporal credit assignment in episodic multi-agent reinforcement learning (MARL). TAR² provides **structural robustness** and **formal policy preservation guarantees** through its unique decoupled architecture.

📄 **Paper**: [TAR2: Temporal-Agent Reward Redistribution for Optimal Policy Preservation in Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2502.04864)

## 🎯 Key Contributions

- **Structurally Robust Design**: Decouples credit modeling from constraint satisfaction, overcoming theoretical fragility of prior methods
- **Formal Policy Preservation**: Proven equivalence to valid Potential-Based Reward Shaping (PBRS) guarantees optimal policy preservation
- **State-of-the-Art Performance**: Superior sample efficiency and final performance on challenging SMACLite and Google Research Football benchmarks
- **Variance Reduction**: Novel final-state conditioning and episodic potential constraints reduce gradient variance

## 🏗️ Architecture Overview

TAR² processes trajectory data through four main stages:

1. **Input Embedding**: Converts observations and actions into rich representations with positional encoding
2. **Dual Transformer**: Sequential temporal and agent attention blocks with inverse dynamics regularization
3. **Score Prediction**: Computes unnormalized contribution scores conditioned on final outcome
4. **Deterministic Normalization**: Converts scores into policy-preserving shaped rewards with strict return equivalence

## 📦 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (recommended for GPU acceleration)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/AdityaKapoor74/MARL_Agent_Temporal_Credit_Assignment.git
cd MARL_Agent_Temporal_Credit_Assignment

# Create conda environment
conda create -n tar2 python=3.8
conda activate tar2

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install core dependencies
pip install -r requirements.txt
```

### Multi-Agent Environment Setup

#### SMACLite Environment
SMACLite is a lightweight implementation of StarCraft Multi-Agent Challenge.

```bash
# Install SMACLite (custom fork with episodic rewards)
pip install git+https://github.com/AdityaKapoor74/smaclite.git
```

For detailed setup instructions and troubleshooting, please refer to the [SMACLite README](https://github.com/AdityaKapoor74/smaclite).

#### Google Research Football Environment
Google Research Football is a physics-based football simulation environment.

```bash
# Install Google Research Football
pip install gfootball

# For visualization support (optional)
pip install pygame
```

For comprehensive installation instructions, system requirements, and troubleshooting, please refer to the [Google Research Football README](https://github.com/google-research/football).

## 📂 Repository Structure

```
MARL_Agent_Temporal_Credit_Assignment/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── MAPPO/                          # Main implementation directory
│   ├── train_agent.py             # Main training script
│   ├── agent.py                   # PPO agent implementation
│   ├── model.py                   # Actor-critic networks and PopArt
│   ├── utils.py                   # Buffers and utility functions
│   │
│   ├── TAR2/                      # TAR² implementation
│   │   ├── TAR2.py               # Main TAR² model with ShapelyAttention
│   │   └── modules.py            # Transformer components and attention layers
│   │
│   ├── AREL/                     # AREL baseline implementation
│   │   ├── AREL.py               # Time-Agent Transformer model
│   │   ├── modules.py            # Transformer blocks for AREL
│   │   └── util.py               # Utility functions
│   │
│   ├── STAS/                     # STAS baseline implementation
│   │   ├── stas.py               # STAS model with ShapelyAttention
│   │   └── modules.py            # Encoder layers and attention mechanisms
│   │
│   └── scripts/                  # Training scripts
│       ├── smaclite/            # SMACLite environment scripts
│       │   ├── tar2.sh          # TAR² training script
│       │   ├── stas.sh          # STAS baseline script
│       │   ├── arel_temporal.sh # AREL temporal-only script
│       │   ├── arel_agent_temporal.sh # AREL agent-temporal script
│       │   └── uniform.sh       # Uniform baseline script
│       └── grf/                 # Google Research Football scripts
│           ├── tar2_grf.sh      # TAR² for GRF
│           ├── stas_grf.sh      # STAS for GRF
│           ├── arel_temporal_grf.sh # AREL temporal for GRF
│           ├── arel_agent_temporal_grf.sh # AREL agent-temporal for GRF
│           └── uniform_grf.sh   # Uniform baseline for GRF
│
├── Plot/                         # Experimental results and plotting utilities
│   └── 5m_vs_6m/
│       └── Uniform_All_Enemies_Dead.json
└── .gitignore                   # Git ignore file
```

## 🚀 Quick Start

### Training TAR² on SMACLite

Navigate to the MAPPO directory and run the provided shell scripts:

```bash
cd MAPPO

# Make scripts executable
chmod +x scripts/smaclite/*.sh

# Train TAR² on 3s5z scenario
./scripts/smaclite/tar2.sh

# Train on different scenarios
python train_agent.py \
    --iteration 1 \
    --learn \
    --use_reward_model \
    --device gpu \
    --environment StarCraft \
    --env 5m_vs_6m \
    --experiment_type "TAR^2" \
    --max_episodes 30000 \
    --save_model \
    --test_num "TAR2_5m_vs_6m"
```

Available SMACLite scenarios:
- `3s5z` - 3 Stalkers vs 5 Zealots (heterogeneous)
- `5m_vs_6m` - 5 Marines vs 6 Marines (homogeneous)
- `10m_vs_11m` - 10 Marines vs 11 Marines (large scale)

### Training on Google Research Football

```bash
# Make scripts executable
chmod +x scripts/grf/*.sh

# Train TAR² on basic scenario
./scripts/grf/tar2_grf.sh

# Train on different scenarios
python train_agent.py \
    --iteration 1 \
    --environment GFootball \
    --env academy_counterattack_easy \
    --experiment_type "TAR^2" \
    --max_episodes 30000 \
    --max_time_steps 200
```

Available GRF scenarios:
- `academy_3_vs_1_with_keeper` - Basic offensive scenario
- `academy_counterattack_easy` - Transition from defense to offense
- `academy_pass_and_shoot_with_keeper` - Coordination and passing

### Running Baseline Methods

```bash
# STAS baseline
./scripts/smaclite/stas.sh

# AREL temporal-only baseline
./scripts/smaclite/arel_temporal.sh

# AREL agent-temporal baseline
./scripts/smaclite/arel_agent_temporal.sh

# Uniform baseline (no reward model)
./scripts/smaclite/uniform.sh
```

### Multi-Seed Experiments

For statistical significance, run experiments across multiple seeds:

```bash
# Run TAR² with 5 different seeds
for seed in 1 2 3 4 5; do
    python train_agent.py \
        --iteration $seed \
        --experiment_type "TAR^2" \
        --env 3s5z \
        --test_num "TAR2_3s5z_seed_${seed}" \
        --save_model
done
```

## 🧪 Ablation Studies

TAR² includes several architectural components that can be ablated by modifying the `--version` flag:

```bash
# Remove final outcome conditioning
python train_agent.py \
    --experiment_type "TAR^2" \
    --version "no_final_outcome" \
    [other arguments...]

# Remove inverse dynamics regularization
python train_agent.py \
    --experiment_type "TAR^2" \
    --version "no_inverse_dynamics" \
    [other arguments...]

# Remove deterministic normalization (breaks policy preservation guarantee)
python train_agent.py \
    --experiment_type "TAR^2" \
    --version "no_normalization" \
    [other arguments...]
```

## 📊 Monitoring and Logging

The implementation supports comprehensive logging through Comet.ml:

```bash
# Enable Comet.ml logging
python train_agent.py \
    --save_comet_ml_plot \
    --test_num "Your_Experiment_Name" \
    [other arguments...]
```

Key metrics logged:
- Episode rewards and length  
- Model losses (policy, critic, reward)
- Attention weights and entropy
- Gradient norms
- Agent and temporal credit distributions

## 🚦 Troubleshooting

### Environment installation issues: 
   - For SMACLite: Check the [SMACLite repository](https://github.com/AdityaKapoor74/smaclite) for detailed setup
   - For GRF: Refer to [Google Research Football documentation](https://github.com/google-research/football)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built upon the excellent [MAPPO implementation](https://github.com/marlbenchmark/on-policy)
- SMACLite environment by [Michalski et al.](https://github.com/uoe-agents/smaclite)
- Google Research Football by [Kurach et al.](https://github.com/google-research/football)
- Baseline implementations adapted from STAS and AREL papers