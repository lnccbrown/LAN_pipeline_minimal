#!/bin/bash

# TODO: Maybe instead of having two network configs (one yaml, one sh), we want only one of them!
# Then have a meta python script that runs the correct training script, based on configurations

# Network training configs -----------
backend="torch" # jax, torch
partition="gpu" # gpu, cpu
dl_workers=2 # number of processes to use for data-loading
n_networks=6