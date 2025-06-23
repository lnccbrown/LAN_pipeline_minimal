# Test shell script for running gen_sbatch.py

# Generate
uv run LAN_pipeline_minimal/sbatch_scripts/gen_sbatch.py generate --config-path LAN_pipeline_minimal/user_configs_examples/config_data_generation.yaml --output-path my_generated_data --time 00:10:00 --make-env --log-level INFO

# Jaxtrain
uv run LAN_pipeline_minimal/sbatch_scripts/gen_sbatch.py jaxtrain --config-path LAN_pipeline_minimal/user_configs_examples/config_network_training_lan.yaml --output-path my_trained_network --training-data-folder ./my_generated_data/data/training_data/lan/training_data_n_samples_20000_dt_0.001/ddm --network-id 0 --dl-workers 1 --time 00:10:00 --make-env --log-level INFO