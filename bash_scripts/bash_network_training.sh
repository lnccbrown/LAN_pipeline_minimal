#!/bin/bash


while [ ! $# -eq 0 ]
    do
        case "$1" in
             --bash_config_acct | -a)
                bash_config_acct=$2
             ;;
             --bash_config_network | -n)
                bash_config_network=$2
            ;;
             --yaml_config_network | -y)
                yaml_config_network=$2
            ;;
        esac
        shift 2
    done

# Run configs (parameters to script) ----------------
. $bash_config_acct
# acct_name="my-condo" # your oscar account name  / "carney-frankmj-condo"
# bashrc_path=/path/to/your/.bashrc # path to your bashrc file which will be used to initialize worker (scripts expect that this initializes conda!)
# conda_env_name=your-conda-env-name # name of the your conda environment to use for network training (e.g. lan_pipe)
# project_folder='/users/afengler/data/proj_lan_pipeline/LAN_scripts/' # base folder of the project


# Data generating process ------
# model="ddm" # the model string (used to identify training data folder) #"ddm_deadline" #'ds_conflict_drift'

# # Network training configs -----------
# network_type="cpn" # cpn or lan
# optimizer="adam" # which optimizer to choose
# backend="jax" # jax, torch
# partition="gpu" # gpu, cpu
# dl_workers=2 # number of processes to use for data-loading
# Network training configs
. $bash_config_network
let n_networks-- 

#network_n_epochs=20 # How many epochs to train the network for ?
# data_gen_n_samples_per_sim=20000 #200000 # How many simulated trials per call to the simulator ?
# data_gen_n_parameter_sets=1000 #5000 # How many parameter sets do we request ? 
# data_gen_n_training_examples_per_parameter_set=2000 # How many training examples do we harvest from a given parameter set

# network_training_yaml=$project_folder'/data/config_files/network/'\
# $network_type'/'$model'/train_config_opt_'$optimizer'_n_'$data_gen_n_samples_per_sim'_dt_0.001_nps_'\
# $data_gen_n_parameter_sets'_npts_'$data_gen_n_training_examples_per_parameter_set'_architecture_search.pickle'

networks_path_base=$project_folder'/data/networks/'$backend

echo 'Config file passed to sbatch_network_training.sh'
echo $yaml_config_network
echo $networks_path_base

# Train networks ----
if [ "$partition" == "gpu" ]; then
    sbatch -p gpu --gres=gpu:1 \
                  --account=$oscar_acct \
                  --array=0-$n_networks ../sbatch_scripts/sbatch_network_training.sh \
                  --backend $backend \
                  --config-path $yaml_config_network \
                  --networks_path_base $networks_path_base \
                  --dl_workers $dl_workers \
                  --conda_env_name $conda_env_name \
                  --bashrc_path $bashrc_path | cut -f 4 -d' '
elif [ "$partition" == "cpu" ]; then
    sbatch -p batch --account=$oscar_acct  \
                    --array=0-$n_networks ../sbatch_scripts/sbatch_network_training.sh \
                    --backend $backend \
                    --config-path $yaml_config_network \
                    --networks_path_base $networks_path_base \
                    --dl_workers $dl_workers \
                    --conda_env_name $conda_env_name \
                    --bashrc_path $bashrc_path | cut -f 4 -d' '       
fi