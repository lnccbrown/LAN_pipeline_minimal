# Define and run an experiment

An *experiment* is a directory that holds everything one tracked run of the
pipeline needs: which model, the generation and training configs, how the
network is validated, and one **lineage id** that every stage stamps on its
MLflow runs so data, network and fits can be joined later. `lan-sbatch init`
creates it; the job commands read it with `--experiment`.

## 1. Scaffold

```bash
uv run lan-sbatch init ddm-pilot --model ddm
```

For a small smoke-test scale, add `--quick`; for a choice-probability network,
`--network-type cpn`; for the PyTorch trainer, `--trainer torchtrain`. The
command prints one JSON line:

```json
{"experiment_dir": "/repo/experiments/ddm-pilot",
 "lineage_id": "96d9bb5f055b4881bf166336eecd4b6d",
 "experiments": {"data_generation": "ddm-data-generation",
                 "training": "ddm-training", "inference": "ddm-inference"}}
```

and writes:

```text
experiments/ddm-pilot/
  experiment.yaml          # model, lineage id, network, MLflow names, stage settings
  data_generation.yaml     # copied from configs/examples (or quick_test), MODEL set
  network_training.yaml    # copied likewise, MODEL and NETWORK_TYPE set
```

Commit these three files with your project. Outputs (`data/`, `networks/`,
`recovery/`, `runs/`, `state.json`) land beside them and are gitignored.

## 2. Edit the configs

Open `data_generation.yaml` and `network_training.yaml` and set the scale you
want; they are the upstream packages' own formats (see the
[configuration reference](../reference/configuration.md)). `experiment.yaml`
is validated on every use: the `MODEL` in both configs must equal the
experiment's `model`, and `NETWORK_TYPE` must equal `network.type`, so a
copy-paste slip fails before a job is written.

Under `stages.recover` choose which recovery designs and how many datasets
per design the fit stage runs; the defaults are one small design with one
dataset, enough to prove the chain works.

## 3. Submit the stages

Every stage takes `--experiment DIR`; explicit flags still override.

```bash
# data: one Slurm array, every task shares the lineage id
uv run lan-sbatch generate --experiment experiments/ddm-pilot \
  --cluster-config configs/cluster/oscar.yaml --n-jobs-in-array 50 --n-files 20

# training: the datagen experiment id comes from the experiment's state.json
# (written by the submission above); LANfactory derives the data folder from it
uv run lan-sbatch jaxtrain --experiment experiments/ddm-pilot \
  --cluster-config configs/cluster/oscar.yaml

# recovery fits: one array per design x likelihood, tracked with hssm.track
uv run lan-sbatch recover --experiment experiments/ddm-pilot \
  --cluster-config configs/cluster/oscar.yaml
```

Rehearse any of them with `--script-only` first; that renders the script and
prints the JSON line without touching Slurm or MLflow.

The training command needs a data source. With `--experiment` it uses the
`training_data_folder` recorded in `state.json` when present, and otherwise
passes only `--data-generation-experiment-id`, letting LANfactory read the
folder from the generation runs. Pass `--training-data-folder` to override.

## 4. Find the runs

All three stages log to the experiments named in `experiment.yaml`, by default
`<model>-data-generation`, `<model>-training` and `<model>-inference`, and every
run carries the tag `lineage_id`. In the MLflow UI, select the three
experiments and filter:

```text
tags.lineage_id = '96d9bb5f055b4881bf166336eecd4b6d'
```

Or from Python:

```python
mlflow.search_runs(search_all_experiments=True,
                   filter_string="tags.lineage_id = '<id>'")
```

## What the lineage id does

ssm-simulators writes the id into every training pickle
(`generator_config["lineage_id"]`), LANfactory reads it back and stores it in
the network's config pickles and the HuggingFace `manifest.json`, and
`hssm.track` puts it on each fit. So a fit made months later against the
published network still resolves to the data it was trained on. Two datasets
must never share an id: `init` mints a fresh one, and a rerun that is meant to
be a *new* dataset should be a new experiment (or set `lineage_id: null` and
let the next submission mint one).

## Tracking server

Set `MLFLOW_TRACKING_URI` in the submitting shell; the generated jobs inherit
it. Against an http(s) server, `MLFLOW_ARTIFACT_LOCATION` is ignored with a
warning, because the server owns artifact storage. See
[Track runs with MLflow](track-with-mlflow.md).
