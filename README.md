# JaxMARL-HFT: GPU-Accelerated Multi-Agent Reinforcement Learning for High-Frequency Trading

A JAX-based framework for multi-agent reinforcement learning for high-frequency trading, based on the [JAX-LOB simulator](https://github.com/KangOxford/jax-lob) and an extension of [JaxMARL](https://github.com/FLAIROx/JaxMARL) to the financial trading domain.

## Key Features

- **GPU-Accelerated**: Built on JAX for high-performance parallel computation with JIT compilation
- **Two levels of Parallelization**: Parallel processing across episodes and agent types using `vmap`
- **Multi-Agent RL**: Supports market making, execution, and directional trading agents
- **LOBSTER Data Integration**: Real market data support with efficient GPU memory usage
- **Scalable**: Handles thousands of parallel environments
- **Heterogeneous Agents**: Supports different observation/action spaces

## Quick Start

### Docker Setup

```bash
# Set up data directory
mkdir -p ~/data
```

**Note**: Configure the Makefile for your specific environment (GPU device, data directory path, etc.)

```bash
# Build and run with Docker
make build
make run
```

### Training

```bash
# Run IPPO training (from the repo root)
python3 gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py
```

## Setup

### Paths to configure

Before running, you need to set several paths in the configuration files:

**Environment config** (`config/env_configs/*.json`) — in the `world_config` section:
- **`alphatradePath`**: Path to the repo root. Used for caching preprocessed data (`pre_reset_states/`) and saving checkpoints (`checkpoints/`).
- **`dataPath`**: Path to your data directory (see Data section below).
- **`stock`**: Stock ticker (e.g. `"GOOG"`, `"AMZN"`)
- **`timePeriod`**: Subdirectory name for the time period (must match the folder name under `rawLOBSTER/<stock>/`)

**Training config** (`config/rl_configs/*.yaml`):
- **`TimePeriod`**: Time period for training data (must match `timePeriod` in the env config)
- **`EvalTimePeriod`**: Time period for evaluation data
- **`ENV_CONFIG`**: Path to the environment config JSON file (e.g. `"config/env_configs/2_player_fq_fqc.json"`)

### Data

The framework expects [LOBSTER](https://lobsterdata.com/) market data organised as:
```
<dataPath>/rawLOBSTER/<stock>/<timePeriod>/
├── <stock>_<date>_<...>_message_<...>.csv
└── <stock>_<date>_<...>_orderbook_<...>.csv
```

### WandB

By default, training uses [Weights & Biases](https://wandb.ai/) sweeps for hyperparameter search. Configure in the RL YAML config (`config/rl_configs/`):

```yaml
ENTITY: "your-wandb-entity"
PROJECT: "your-project-name"
WANDB_MODE: "online"
```

To run **without WandB**, set `WANDB_MODE: "disabled"` and leave `SWEEP_PARAMETERS` empty in your YAML config. This runs a single training directly.

## Agent Types

### Market Making Agents
- **Purpose**: Provide liquidity by posting bid/ask orders
- **Action Spaces**: Multiple discrete action spaces (spread_skew, fixed_quants, AvSt, directional_trading, simple)
- **Reward Functions**: Various PnL-based rewards with configurable inventory penalties

### Execution Agents
- **Purpose**: Execute large orders with minimal market impact
- **Action Spaces**: Discrete quantity selection at reference prices (fixed_quants, fixed_prices, complex variants)
- **Reward Functions**: Slippage-based with configurable end-of-episode penalties

### Directional Trading
- **Purpose**: Simple directional trading strategy
- **Action Spaces**: Bid/ask at best prices or no action
- **Reward Function**: Portfolio value
- **Note:** Uses the same class as the market making agent

## Repository Structure

```
config/
├── env_configs/          # Environment JSON configurations
└── rl_configs/           # Training YAML configurations
gymnax_exchange/
├── jaxen/                # Environment implementations
│   ├── marl_env.py       # Multi-agent RL environment
│   ├── mm_env.py         # Market making (and directional trading) environment
│   ├── exec_env.py       # Execution environment
│   └── from_JAXMARL/     # Multi-agent base classes and spaces
├── jaxrl/                # Reinforcement learning algorithms
│   └── MARL/             # IPPO implementation and baseline evaluation
├── jaxob/                # Order book implementation
├── jaxlobster/           # LOBSTER data integration
└── utils/                # Shared utilities
```

## Configuration

The framework uses a comprehensive configuration system with dataclasses for different components:

### Core Configuration Classes

- **`MultiAgentConfig`**: Main configuration combining world and agent settings
- **`World_EnvironmentConfig`**: Global environment parameters (data paths, episode settings, market hours)
- **`MarketMaking_EnvironmentConfig`**: Market making and directional trading agent configuration (action spaces, reward functions, observation spaces)
- **`Execution_EnvironmentConfig`**: Execution agent configuration (task types, action spaces, reward parameters)

### Training Configuration

Edit YAML files in `config/rl_configs/` to customize:
- Number of parallel environments (default: 4096)
- Training parameters (steps, learning rates, etc.)
- Agent configurations (action spaces, reward functions)
- Market data settings (resolution, episode length)

Environment configurations are in `config/env_configs/`.

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- JAX, Flax, and related dependencies (see `requirements.txt`)

## Citation

If you use JaxMARL-HFT in your research, please cite:

```bibtex
@inproceedings{mohl2025jaxmarlhft,
  title={JaxMARL-HFT: GPU-Accelerated Large-Scale Multi-Agent Reinforcement Learning for High-Frequency Trading},
  author={Mohl, Valentin and Frey, Sascha and Leyland, Reuben and Li, Kang and Nigmatulin, George and Cucuringu, Mihai and Zohren, Stefan and Foerster, Jakob and Calinescu, Anisoara},
  booktitle={Proceedings of the 6th ACM International Conference on AI in Finance (ICAIF)},
  pages={18--26},
  year={2025},
  doi={10.1145/3768292.3770416}
}
```

## Acknowledgements

JaxMARL-HFT builds on:
- [JaxMARL](https://github.com/FLAIROx/JaxMARL) — Multi-agent RL environments and algorithms in JAX
- [JAX-LOB](https://github.com/KangOxford/jax-lob) — GPU-accelerated limit order book simulator

## Disclaimer

This software is provided for **research and educational purposes only**. It is not intended for live trading, financial decision-making, or any form of real-money deployment. The authors and contributors make no warranties regarding the accuracy, reliability, or suitability of this software for any particular purpose.

**The authors assume no responsibility or liability for any financial losses, damages, or other consequences arising from the use of this software.** Use at your own risk.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
