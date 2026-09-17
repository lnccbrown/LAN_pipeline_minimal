# Run a tracked experiment

Generate simulated data, train a likelihood network on it, test parameter
recovery, and have every step recorded in MLflow under one lineage id.

## Setup

```bash
git clone https://github.com/lnccbrown/LAN_pipeline_minimal.git
cd LAN_pipeline_minimal
uv sync --locked --group validate
```

!!! warning "Interim: install the tracking-capable tools"

    Until the next releases of ssm-simulators, LANfactory and HSSM, the
    lockfile pins versions that cannot record lineage. After the sync, install
    the branch checkouts into this environment and run every command below
    with `uv run --no-sync` (a plain `uv run` re-syncs and removes them):

    ```bash
    uv pip install -e ~/ssm-simulators -e ~/LANfactory -e ~/HSSM
    ```

    with the checkouts on `cpaniaguam/feat/mlflow-schema-v2`,
    `feat/mlflow-schema-v2` (with PR #140 merged in) and
    `feat/mlflow-infer-tracking` respectively. The runner checks this and
    stops with a clear message if a tool is the wrong version. Note that
    `uv sync` and `./scripts/docs.sh` both reset the environment; repeat the
    install afterwards.

Point jobs at the MLflow server (in your shell profile):

```bash
export MLFLOW_TRACKING_URI="http://<mlflow-host>:5000"
```

Without it, records go to a SQLite file inside the experiment directory.

On a cluster, describe your allocation once (writes a personal, uncommitted file):

```bash
uv run python scripts/discover_cluster.py --ssh-host oscar
```


## 1. Create an experiment

```bash
uv run lan-sbatch init ddm-baseline --model ddm
```

Options: `--quick` for a minutes-long configuration, `--network-type cpn`,
`--trainer torchtrain`.

Creates:

```text
experiments/ddm-baseline/
  experiment.yaml          # model, lineage id, MLflow experiment names, stage settings
  data_generation.yaml     # simulation settings
  network_training.yaml    # architecture and training schedule
```

The printed `lineage_id` is stamped on every job of this experiment.

## 2. Set the scale

Edit the two YAML files.

| File | Key | Meaning |
| --- | --- | --- |
| `data_generation.yaml` | `PIPELINE.N_PARAMETER_SETS` | parameter settings per data file |
| | `SIMULATOR.N_SAMPLES` | trials per parameter setting |
| `network_training.yaml` | `LAYER_SIZES`, `ACTIVATIONS` | architecture (first entry used by default) |
| | `N_EPOCHS` | training length |
| | `N_TRAINING_FILES` | data files used for training |

Do not change `MODEL` or `NETWORK_TYPE`; they must match `experiment.yaml`.

## Local: run everything on this machine

```bash
uv run --no-sync lan-sbatch run --experiment experiments/ddm-baseline --local
```

Runs generate → train → validate → recover, then prints every MLflow run of the
lineage and checks the stages link. With a `--quick` experiment this takes
about three minutes. `--stages generate,train` runs a subset; `--dry-run`
prints the commands only. Then continue at step 6.

The cluster path is steps 3–5.

## 3. Generate data

```bash
uv run lan-sbatch generate --experiment experiments/ddm-baseline \
  --cluster-config configs/cluster/oscar.yaml \
  --n-jobs-in-array 50 --n-files 20
```

One array of 50 workers × 20 files. Add `--script-only` to render the job
script without submitting. Output: `experiments/ddm-baseline/data/`.

## 4. Train

After the array finishes:

```bash
uv run lan-sbatch jaxtrain --experiment experiments/ddm-baseline \
  --cluster-config configs/cluster/oscar.yaml
```

Use `torchtrain` if the experiment was created with `--trainer torchtrain`.
Output: `experiments/ddm-baseline/networks/`.

## 5. Test recovery

```bash
uv run lan-sbatch recover --experiment experiments/ddm-baseline \
  --cluster-config configs/cluster/oscar.yaml
```

Fits simulated datasets with the trained network and, where the model has an
exact likelihood, with that too. Designs, dataset count and sampler settings:
`stages.recover` in `experiment.yaml`. Output: `experiments/ddm-baseline/recovery/`.

## 6. Find the records

Open the MLflow UI. With a server, that is its address. Without one (local
run), start the UI on the experiment's own store:

```bash
uv run --no-sync mlflow ui --backend-store-uri sqlite:///$PWD/experiments/ddm-baseline/mlflow.db --port 5000
```

and open <http://127.0.0.1:5000>. Select the experiments `ddm-data-generation`,
`ddm-training` and `ddm-inference`, and filter:

```text
tags.lineage_id = '<lineage id from experiment.yaml>'
```

Use **Columns** (slider icon, right of the search box) to show `user`,
`val_loss`, or recovery metrics such as `covered_v` and `abs_error_a`.

Do not click **Generate demo data**; it can stall the server.

## Repeating

- More data, same lineage: run step 3 again.
- Another architecture on the same data: edit `network_training.yaml`, run step 4.
- New dataset: `init` a new experiment.

## Errors

| Message | Fix |
| --- | --- |
| `console script 'lan-recover' not found` | `uv sync --locked --group validate` |
| `validate: exit 1: {"passed": false, ... "density": "failed"}` | the network is too small to pass the density gate; use `--quick` (skips it) or `--continue-on-gate-failure` |
| `MODEL 'x' disagrees with experiment model` | change `model` in `experiment.yaml`, not the stage config |
| `Training needs --training-data-folder` | run step 3 from this experiment first, or pass the folder |
| `No ONNX network found for the network arm` | wait for step 4, or pass `--onnx-path` |
| `Invalid Host header` | open the server at the address configured for it |

Job logs: `experiments/<name>/<data|networks|recovery>/runs/*.out|.err`.

## Reference

- [Define and run an experiment](../how-to/define-an-experiment.md): every
  `experiment.yaml` field and stage option.
- [Local workflow](local-workflow.md): the individual commands the local run wraps.
- [Validate](../how-to/validate-network.md) and
  [publish](../how-to/stage-and-publish.md) a network.
