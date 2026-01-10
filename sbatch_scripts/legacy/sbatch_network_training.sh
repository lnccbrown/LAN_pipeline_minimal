#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J model_trainer

# priority
##SBATCH --account=your-condo

# output file
#SBATCH --output ../slurm/slurm_model_trainer_%A_%a.out

# Request runtime, memory, cores
#SBATCH --time=32:00:00
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -N 1

##SBATCH -p gpu --gres=gpu:1
##SBATCH --array=0-8 # should be 89

# --------------------------------------------------------------------------------------

# Read in arguments:
# These are supposed to be overwritten by arguments passed to the script
# they serve as reasonable defaults though
network_id=None
config_path=None
networks_path_base="/users/afengler/data/proj_lan_pipeline_minimal/LAN_pipeline_minimal/data/"
dl_workers=4
n_networks=2
backend="jax"
conda_env_name=lan_pipe
bashrc_path=/users/afengler/.bashrc

echo "arguments passed to sbatch_network_training.sh $#"

while [ ! $# -eq 0 ]
    do
        case "$1" in
            --config_path | -p)
                echo "passing config path $2"
                config_path=$2
                ;;
            --networks_path_base | -o)
                echo "passing output_folder $2"
                networks_path_base=$2
                ;;
            --n_networks | -n)
                echo "passing number of networks $2"
                n_networks=$2
                ;;
            --backend | -b)
                echo "passing deep learning backend specification: $2"
                backend=$2
                ;;
            --dl_workers | -d)
                echo "passing number of dataloader workers $2"
                dl_workers=$2
                ;;
            --conda_env_name | -ce)
                echo "passing conda environment: name $2"
                conda_env_name=$2
                ;;
            --bashrc_path | -bp)
                echo "passing bashrc path: $2"
                bashrc_path=$2
        esac
        shift 2
    done

# Setup
source $bashrc_path
# TODO: This double conda deactivate can be simplified further --> key is understanding how to handle .bashrc / .bash_profile correctly
conda deactivate
conda deactivate
conda activate $conda_env_name

echo "The config file supplied is: $config_path"
echo "The config dictionary key supplied is: $network_id"
echo "Output folder is: $output_folder"

x='teststr' # defined only for the check below (testing whether SLURM_ARRAY_TASK_ID is set)
if [ -z ${SLURM_ARRAY_TASK_ID} ];
then
    for ((i = 1; i <= $n_networks; i++))
        do
            echo "NOW TRAINING NETWORK: $i of $n_networks"
            echo "No array ID"
            
            if [ "$backend" == "jax" ]; then
                python -u ../scripts/jax_training_script.py --config_path $config_path \
                                                         --network_id 0 \
                                                         --networks_path_base $networks_path_base \
                                                         --dl_workers $dl_workers
            elif [ "$backend" == "torch" ]; then
                python -u ../scripts/torch_training_script.py --config_path $config_path \
                                                           --network_id 0 \
                                                           --networks_path_base $networks_path_base \
                                                           --dl_workers $dl_workers
                                                       
            fi
        done
else
    for ((i = 1; i <= $n_networks; i++))
        do
            echo "NOW TRAINING NETWORK: $i of $n_networks"
            echo "Array ID is $SLURM_ARRAY_TASK_ID" 
            
            if [ "$backend" == "jax" ]; then
                python -u ../scripts/jax_training_script.py --config_path $config_path \
                                                         --network_id $SLURM_ARRAY_TASK_ID \
                                                         --networks_path_base $networks_path_base \
                                                         --dl_workers $dl_workers
            elif [ "$backend" == "torch" ]; then
                python -u ../scripts/torch_training_script.py --config_path $config_path \
                                                           --network_id $SLURM_ARRAY_TASK_ID \
                                                           --networks_path_base $networks_path_base \
                                                           --dl_workers $dl_workers
                                                       
            fi
        done
fi