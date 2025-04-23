import pickle
import numpy as np
import pandas as pd
import ssms
import argparse
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from config import *


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
    CLI.add_argument("--data-gen-base-path", type=none_or_str, default=None)

    args = CLI.parse_args()
    print(args)

    config_dict = get_data_generator_config(
        yaml_config_path=args.config_path, base_path=args.data_gen_base_path
    )["config_dict"]

    assert args.config_path is not None, (
        "You need to supply a config file path to the script"
    )

    print("Printing config specs: ")
    print("GENERATOR CONFIG")
    print(config_dict["data_config"])

    print("MODEL CONFIG")
    print(config_dict["model_config"])

    # Make the generator
    print("Now generating data")
    my_dataset_generator = ssms.dataset_generators.lan_mlp.data_generator(
        generator_config=config_dict["data_config"],
        model_config=config_dict["model_config"],
    )
    if "cpn_only" in config_dict["data_config"].keys():
        if config_dict["data_config"]["cpn_only"]:
            x = my_dataset_generator.generate_data_training_uniform(
                save=True, cpn_only=True
            )
        else:
            x = my_dataset_generator.generate_data_training_uniform(
                save=True, cpn_only=False
            )
    print("Data generation finished")
