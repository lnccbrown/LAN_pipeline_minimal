import lanfactory
import ssms
import numpy as np
import os
import pickle
from copy import deepcopy
import pandas as pd


def get_model_df(
    filter_="training_data_angle",
    path_="/users/afengler/data/proj_lan_pipeline/LAN_scripts/data/torch_models",
):
    model_wise_pds = []
    pds = {}

    print(filter_)
    print(path_)
    print(os.listdir(path_))
    # Extract the training history files and use info to
    # fill in some columns
    for file_ in os.listdir(path_):
        if filter_ in file_:
            if "training_history" in file_:
                pd_tmp = pd.read_csv(path_ + "/" + file_)
                pd_tmp["model_id"] = file_[: file_.find("_")]
                pd_tmp["filename"] = file_
                pd_tmp["model_filename"] = (
                    file_[: file_.find("_training_history")] + "_state_dict.pt"
                )
                pd_tmp["path"] = path_ + "/" + file_
                pd_tmp["model_path"] = (
                    path_
                    + "/"
                    + file_[: file_.find("_training_history")]
                    + "_state_dict.pt"
                )
                pd_tmp["model_type"] = filter_

                # pds stores the pd_tmp subdictionaries
                # in a single dictionary that has model indices as keys
                pds[file_[: file_.find("_")]] = pd_tmp

                # print(pds)

    # Add extra info to pds dictionary
    # concerning specifics about the network configuration
    for m_id in pds.keys():
        for file_ in os.listdir(path_):
            if m_id in file_ and "network_config" in file_:
                network_config_tmp = pickle.load(open(path_ + "/" + file_, "rb"))
                pds[m_id]["n_hidden_layers"] = (
                    len(network_config_tmp["layer_sizes"]) - 1
                )
                pds[m_id]["size_hidden_layers"] = network_config_tmp["layer_sizes"][0]
                pds[m_id]["network_config_path"] = path_ + "/" + file_

    training_dat = pd.concat([pds[m_id] for m_id in pds.keys()]).reset_index(drop=True)
    training_dat.dropna(inplace=True)

    print(training_dat)
    print("MAX EPOCHS")
    # print(training_dat['epoch'].max())

    best_models_pds = []
    for n_h_l in training_dat["n_hidden_layers"].unique():
        for s_h_l in training_dat["size_hidden_layers"].unique():
            training_dat_sub = training_dat.loc[
                (training_dat["n_hidden_layers"] == n_h_l)
                & (training_dat["size_hidden_layers"] == s_h_l)
                & (
                    training_dat["epoch"]
                    == training_dat.loc[
                        (training_dat["n_hidden_layers"] == n_h_l)
                        & (training_dat["size_hidden_layers"] == s_h_l),
                        "epoch",
                    ].max()
                ),
                :,
            ]
            # print(training_dat_sub)
            val_loss_min = training_dat_sub["val_loss"].min()
            best_models_pds.append(
                training_dat_sub.loc[training_dat_sub["val_loss"] == val_loss_min, :]
            )

    best_models_dat = (
        pd.concat(best_models_pds).reset_index(drop=True).drop(["Unnamed: 0"], axis=1)
    )

    return best_models_dat


def get_model_performance_summary_df(
    filter_="training_data_angle",
    path="/users/afengler/data/proj_lan_pipeline/LAN_scripts/data/torch_models",
    show_top_n=10,
):
    data = get_model_df(filter_=filter_, path_=path)
    print(data)

    val_loss = data["val_loss"]
    data = data.sort_values("val_loss")

    print("Overall val loss statistics: \n")
    print("val loss mean: ", data["val_loss"].mean())
    print("val loss min: ", data["val_loss"].min())
    print("val loss max: ", data["val_loss"].max())

    print("\n")

    print("Top five network architectures \n")
    print(
        data.iloc[:show_top_n, :][["val_loss", "n_hidden_layers", "size_hidden_layers"]]
    )

    print("\n")
    print("Top five networks, paths \n")
    if show_top_n > data.shape[0]:
        return data
    else:
        for i in range(show_top_n):
            if data.shape[0] > i:
                print(data.iloc[i, :]["model_filename"])
                print(data.iloc[i, :]["model_path"])
    return data.iloc[:show_top_n, :]
