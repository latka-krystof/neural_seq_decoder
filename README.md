## Pytorch implementation of [Neural Sequence Decoder](https://github.com/fwillett/speechBCI/tree/main/NeuralDecoder)

## Requirements
- python >= 3.9

## Installation

```bash
pip install -e .
```

## Data Preparation

1. Convert the speech BCI dataset using [formatCompetitionData.ipynb](./notebooks/formatCompetitionData.ipynb)
   - This will create a preprocessed dataset file (typically `processedData`)

## Running Experiments

This repository uses a configuration-based system for running experiments. All experiment configurations are located in `config/experiments/`.

### Basic Usage

**List all available experiments:**
```bash
python scripts/test_model_configuration.py --list
```

**Run a specific experiment:**
```bash
python scripts/test_model_configuration.py baseline
python scripts/test_model_configuration.py test_adamw_eps_005
python scripts/test_model_configuration.py test_post_gru_stack_layers2
```

**Run multiple experiments:**
```bash
python scripts/test_model_configuration.py baseline test_adamw_eps_005 test_post_gru_stack_layers2
```

**Run all experiments:**
```bash
python scripts/test_model_configuration.py --all
```

### Environment-Specific Paths

If running on different environments (Kaggle, Google Cloud, GCP instance), use the `--env` flag to automatically adjust paths:

```bash
# For Kaggle
python scripts/test_model_configuration.py baseline --env kaggle

# For Google Cloud
python scripts/test_model_configuration.py baseline --env google_cloud

# For GCP instance
python scripts/test_model_configuration.py baseline --env gcp_instance
```

### Creating New Experiments

To create a new experiment, add a Python file in `config/experiments/` that defines an `args` dictionary with your configuration. See existing config files for examples.

### Analyzing Results

Training statistics are saved in the experiment's output directory. Use the analysis script to visualize results:

```bash
python scripts/analyze_training_stats.py <path_to_training_stats>
```

