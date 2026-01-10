#!/bin/bash

# Running data generation
# bash bash_data_generation.sh --bash_config_acct user_configs/config_acct.sh \
#                              --bash_config_data_gen user_configs/config_data_generation.sh \
#                              --yaml_config_data_gen user_configs/config_data_generation.yaml

# Running network training
bash bash_network_training.sh --bash_config_acct user_configs/config_acct.sh \
                           --bash_config_network user_configs/config_network_training.sh \
                           --yaml_config_network user_configs/config_network_training_cpn.yaml