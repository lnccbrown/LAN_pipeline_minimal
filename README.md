# LAN_pipeline_minimal
Minimal version of the LAN pipeline for internal purposes

## Installation Instructions

We recommend using this pipeline with `uv`, and you can find installation instructions for `uv` [here](https://docs.astral.sh/uv/getting-started/installation/). 

After `uv` has been downloaded, you can ensure you have the proper environment setup by:
1. Cloning this repo
2. Ensuring this repo is your working directory
3. Running `uv sync` to create a `.venv` with the correct environment setup to run `LAN_pipeline_minimal`. 
4. After `.venv` has been created, you can run `source .venv/bin/activate` to activate the environment if desired. 

## Usage

The pipeline works as a two-step process.

1. Data generation (to generate training data appropriate for specific, or multiple network types)
2. Network training

To create your own `.sh` scripts for data generation and network training, you can run the `gen_sbatch.py` script, which creates a SBATCH script based on user configurations to either generate data or train a network (using either jax or torch as backends) on this data.

Get started by viewing the help for each of the available commands in `gen_sbatch.py`

```
uv run sbatch_scipts/gen_sbatch.py --help
uv run sbatch_scipts/gen_sbatch.py generate --help
uv run sbatch_scipts/gen_sbatch.py jaxtrain --help
uv run sbatch_scripts/gen_sbatch.py torchtrain --help
```

In the `user_configs_examples` folder, you will find example `.yaml` config files used for data generation and network training (you pass the configs mentioned below as arguments to the scripts), as well as sample sbatch `.sh` scripts for `generate` and `jaxtrain`. 

### `config` Logic

The basic logic for configs is this.

For *data generation* we have one `.yaml` config (`config_data_generation.yaml`), which provides a bunch of hyperparameters concerning training data generation.

For *network training* likewise we have one `.yaml` config for a given network type:

1. One example for **cpn** networks (`config_network_training_cpn.yaml`) 
2. One example  for **lan** networks (`config_network_training_lan.yaml`).
