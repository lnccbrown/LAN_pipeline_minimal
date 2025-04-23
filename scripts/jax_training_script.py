import psutil
import argparse
import lanfactory
from copy import deepcopy
import os
import pickle
import torch
import random
import numpy as np
import uuid
import yaml
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from config import *

import jax
from lanfactory.utils import try_gen_folder


def none_or_str(value):
    print("none_or_str")
    print(value)
    print(type(value))
    if value == "None":
        return None
    return value


def none_or_int(value):
    print("none_or_int")
    print(value)
    print(type(value))
    # print('printing type of value supplied to non_or_int ', type(int(value)))
    if value == "None":
        return None
    return int(value)


if __name__ == "__main__":
    # Interface ----
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--config-path", type=none_or_str, default=None)
    CLI.add_argument("--network_id", type=none_or_int, default=None)
    CLI.add_argument("--networks_path_base", type=none_or_str, default=None)
    CLI.add_argument("--dl_workers", type=int, default=0)

    args = CLI.parse_args()
    print("Arguments passed: ", args)

    if args.dl_workers == 0:
        n_workers = min(12, psutil.cpu_count(logical=False) - 2)
    else:
        n_workers = args.dl_workers
    print("Number of workers we assign to the DataLoader: ", n_workers)

    # Load config dict (new)
    if args.network_id is None:
        config_dict = get_train_network_config(
            yaml_config_path=args.config_path, net_index=0
        )
    else:
        config_dict = get_train_network_config(
            yaml_config_path=args.config_path, net_index=args.network_id
        )

    # # Load config dict
    # if args.network_id is None:
    #     config_dict = pickle.load(open(args.config_path, 'rb'))[0]
    # else:
    #     config_dict = pickle.load(open(args.config_path, 'rb'))[args.network_id]

    print("config dict keys: ")
    print(config_dict.keys())
    train_config = config_dict["config_dict"]["train_config"]
    network_config = config_dict["config_dict"]["network_config"]
    extra_config = config_dict["extra_fields"]

    print("TRAIN CONFIG")
    print(train_config)

    print("NETWORK CONFIG")
    print(network_config)

    # Get training and validation data files
    file_list = os.listdir(config_dict["config_dict"]["training_data_folder"])

    print("TRAINING DATA FILES")
    print(file_list)

    valid_file_list = np.array(
        [
            config_dict["config_dict"]["training_data_folder"] + "/" + file_
            for file_ in file_list
        ]
    )

    print("VALID FILE LIST")
    print(valid_file_list)

    random.shuffle(valid_file_list)
    n_training_files = min(len(valid_file_list), train_config["n_training_files"])
    val_idx_cutoff = int(
        config_dict["config_dict"]["train_val_split"] * n_training_files
    )

    print("NUMBER OF TRAINING FILES FOUND: ")
    print(len(valid_file_list))

    print("NUMBER OF TRAINING FILES USED: ")
    print(n_training_files)

    # Check if gpu is available
    if jax.default_backend() == "gpu":
        batch_size = train_config["gpu_batch_size"]
        train_config["train_batch_size"] = batch_size
    else:
        batch_size = train_config["cpu_batch_size"]
        train_config["train_batch_size"] = batch_size

    print("CUDA devices: ")
    print(jax.devices())

    print("BATCH SIZE CHOSEN: ")
    print(batch_size)

    # Make the dataloaders
    train_dataset = lanfactory.trainers.DatasetTorch(
        file_ids=valid_file_list[:val_idx_cutoff],
        batch_size=batch_size,
        label_lower_bound=train_config["label_lower_bound"],
        features_key=train_config["features_key"],
        label_key=train_config["label_key"],
    )

    dataloader_train = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=train_config["shuffle_files"],
        batch_size=None,
        num_workers=n_workers,
        pin_memory=True,
    )

    val_dataset = lanfactory.trainers.DatasetTorch(
        file_ids=valid_file_list[val_idx_cutoff:],
        batch_size=batch_size,
        label_lower_bound=train_config["label_lower_bound"],
        features_key=train_config["features_key"],
        label_key=train_config["label_key"],
    )

    dataloader_val = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=train_config["shuffle_files"],
        batch_size=None,
        num_workers=n_workers,
        pin_memory=True,
    )

    # Load network
    net = lanfactory.trainers.MLPJaxFactory(
        network_config=deepcopy(network_config),
        train=True,
    )

    # Load model trainer
    model_trainer = lanfactory.trainers.ModelTrainerJaxMLP(
        train_config=deepcopy(train_config),
        train_dl=dataloader_train,
        valid_dl=dataloader_val,
        model=net,
        allow_abs_path_folder_generation=True,
    )

    # run_id
    run_id = uuid.uuid1().hex

    # wandb_project_id
    wandb_project_id = extra_config["model"] + "_" + network_config["network_type"]

    # save network config for this run
    networks_path = (
        args.networks_path_base
        + "/"
        + network_config["network_type"]
        + "/"
        + extra_config["model"]
    )

    try_gen_folder(folder=networks_path, allow_abs_path_folder_generation=True)
    pickle.dump(
        network_config,
        open(
            networks_path
            + "/"
            + run_id
            + "_"
            + network_config["network_type"]
            + "_"
            + extra_config["model"]
            + "_"
            + "network_config.pickle",
            "wb",
        ),
    )

    # Train model
    model_trainer.train_and_evaluate(
        save_history=train_config["save_history"],
        output_folder=networks_path,
        output_file_id=extra_config["model"],
        run_id=run_id,
        wandb_on=True,
        wandb_project_id=wandb_project_id,
        save_all=True,
        verbose=1,
    )
