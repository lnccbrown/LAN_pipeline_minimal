# Pipeline architecture

LAN_pipeline_minimal is a control plane around three kinds of work: simulation,
network training, and artifact promotion. It keeps the scientific computation
in the ecosystem packages that own it and adds scheduling, identity, lineage,
and safety at their boundaries.

## Responsibilities by component

| Component | Responsibility | Does not own |
| --- | --- | --- |
| `ssm-simulators` | Simulate trials and generate likelihood-training files | Slurm resource policy or promotion |
| `LANfactory` | Train LAN/CPN/OPN networks and export artifacts | Cross-run staging or production approval |
| `sbatch_scripts/gen_sbatch.py` | Resolve resources, create MLflow identity, render and submit jobs | Simulation or training algorithms |
| `validation/validate_network.py` | Test an ONNX candidate against the ecosystem contract | Deciding which training run to select |
| `publish/publish_network.py` | Resolve one run, isolate its artifacts, gate, upload to staging, and record | Direct production publication |
| HSSM | Load the resulting likelihood in an inference model | Training or publishing the network |

This separation lets the pipeline remain a small orchestration repository. It
does not copy simulator or trainer APIs, and its documentation points to those
projects for scientific configuration details.

## Control flow

```text
operator
  |
  |  pipeline YAML + cluster YAML + environment
  v
lan-sbatch
  |-- resolve resources
  |-- create/reuse MLflow experiment (and a training run when applicable)
  |-- render timestamped script and log paths
  `-- call sbatch
          |
          |  generated script enters this checkout and runs through uv
          v
   generate | jaxtrain | torchtrain
          |
          |  data files or run-identified network artifacts
          v
   validate -> isolated stage -> non-production Hugging Face repository
```

The generated script records the checkout that rendered it and changes into
that directory before invoking `uv run`. Slurm's submission directory therefore
does not decide which project or lockfile the compute node uses.

## Four planes

### Compute plane

ssm-simulators performs CPU-oriented, embarrassingly parallel data generation.
LANfactory performs a single JAX or Torch training job, usually on one GPU.
The generated script calls their installed console commands directly.

### Scheduling plane

Cluster-wide defaults and a gitignored personal overlay resolve to one concrete
resource request. Data-generation arrays can fan out across independent CPU
lanes. Training stays on one selected GPU lane because a single process cannot
combine separate Slurm allocations.

### Evidence plane

MLflow holds generation experiments, worker runs, training runs, and publication
runs. JSON written to standard output gives external drivers the corresponding
job IDs, run IDs, paths, and outcomes without scraping logs.

### Promotion plane

The validator gathers mechanical, integration, and statistical evidence. The
publisher isolates a single run by `run_uuid`, requires the promotion-critical
gates to have actually run, and uploads only to a non-production target. The
normal CLI cannot write to the production repository.

## Reproducibility boundary

`uv.lock` pins the complete project environment, including the git revisions of
ssm-simulators and LANfactory. Job scripts call `uv run` from the repository,
so moving a config without its checkout and lockfile does not reproduce a run.

Documentation uses a different boundary: `scripts/docs.sh` syncs only the
pinned `docs` dependency group with `--no-install-project`. Building this site
does not install CUDA wheels, HSSM, or either git-sourced project dependency.
