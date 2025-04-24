# LAN_pipeline_minimal
Minimal version of the LAN pipeline for internal purposes

## Installation Instructions

To install an environment that will be sufficient for running the whole pipeline, you can use the following steps:

(This assumes you have the `mamba` package manager installed in the base environment of your `conda` package manager,
if you don't have `mamba` installed create the new environment with conda instead)

```
mamba create -n lan_pipe python=3.10
conda activate lan_pipe
pip install --upgrade pip
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install lanfactory
pip install ssm-simulators
pip install jupyter
pip install frozendict
pip install pyyaml
```

If you want to make your new environment available in e.g. Visual Studio Code, run also,

```
python -m ipykernel install --user --name=lan_pipe
```

### Oscar specifics

In your oscar home directory, create a `.modules` file (if you don't already have it, in which case just add the lines below to it), and enter the following content,

```
module load git/2.29.2
module load gcc/10.2
module load cuda/11.8.0
module load cudnn/8.6.0
```

## Usage

The pipeline works as a two-step process.

1. Data generation (to generate training data appropriate for specific, or multiple network types)
2. Network training

You want to respectively run the `bash_data_generation.sh` script for data generation, and the `bash_network_training.sh` script for network training.

Find example call to these scripts in the `bash_run.sh` file, which sits in the `user_configs_examples` folder (you pass the configs mentioned below as arguments to the `bash` scripts).

### `user_configs_examples` folder

There are now a few config files that can be changed outside of the main code (no jupyter notebook need to be run anymore for this :)).
I will run through the examples. Just copy and rename to create your own. They are only ever passed to the `bash_data_generation.sh` and / or `bash_network_training.sh` scripts.
All scripts should be generic now so one doesn't have to make changes to main code for running anything (situation previously).


### `config`-logic

The basic logic for configs is this. We have one config file that concern *basic account settings*: `config_acct.sh`.

For *data generation* we have one `.sh` config (`config_data_generation.sh`) and one `.yaml` config (`config_data_generation.yaml`):

1. The `.sh` config is very simple and just specifies the amount of array jobs you want to run. 
2. The `.yaml` config provides a bunch of hyperparameters concerning a tranining data run.

For *network training* likewise we have one `.sh` config (`config_network_training.sh`) and one `.yaml` config for a given network type:

1. I provide one example for **cpn** networks (`config_network_training_cpn.yaml`) 
2. One example  for **lan** networks (`config_network_training_lan.yaml`).

Again,
1.  The `.sh` config provides some very basic settings
3.  The `.yaml` configs provide detailed instruction to the specific training run.

### workflow

The repo is now organized so that there is a common body of code (which can be improved via PRs) and personal config files.
I left example configs in the `user_configs_examples` folder, but you can put your configs wherever you like. I was thinking about a workflow where you can create your own branch, add 
your own configs and otherwise keep pulling changes / improvements from the main branch as they come in.

There are other ways and please leave suggestions if you have a better idea.