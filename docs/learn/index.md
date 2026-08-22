# Learning path

This path is for someone who can run Python commands but has not operated the
LAN training pipeline before. Work through it in order: each step introduces an
interface that the next step relies on.

## Before you begin

You need:

- Python 3.12 and [uv](https://docs.astral.sh/uv/);
- a checkout of this repository with its tracked `uv.lock`;
- GSL and a working C compiler, because ssm-simulators builds native extensions;
- access to a Slurm cluster only when you move beyond the local tutorial.

The repository installs ssm-simulators and LANfactory from the git revisions in
`uv.lock`. Treat that lockfile as part of the workflow, not generated clutter.

## 1. Establish a local baseline

[Run the CI-backed local workflow](local-workflow.md). It generates a tiny DDM
dataset, trains a small JAX LAN, and records the handoff in MLflow. You should be
able to identify the generated data, the trained artifacts, and the two MLflow
experiments before continuing.

## 2. Learn the operator controls

Next, use the how-to guides to:

1. [discover the cluster lanes](../how-to/configure-cluster.md) attached to your
   own account;
2. turn the quick-test commands into
   [inspectable Slurm scripts](../how-to/submit-slurm-jobs.md);
3. move [MLflow metadata and artifacts](../how-to/track-with-mlflow.md) onto
   shared storage;
4. [validate and inspect](../how-to/validate-network.md) a candidate network;
5. [dry-run the staging and publication plan](../how-to/stage-and-publish.md).

The local workflow proves the package integration. A generated Slurm script
proves the orchestration plan. Neither result proves a network is scientifically
fit to publish; the validation and promotion steps provide that boundary.

## 3. Read the contracts before automating

Read the [architecture](../explanations/architecture.md),
[data-lineage](../explanations/data-lineage.md), and
[promotion-safety](../explanations/promotion-safety.md) explanations before
building a driver. Then use the [CLI](../reference/cli.md),
[configuration](../reference/configuration.md),
[environment](../reference/environment.md), and
[JSON](../reference/json-output.md) references as the stable interface map.

When you finish, you should be able to trace one network backward from its
publication record to its training run, its generation experiment, its configs,
and the exact dependency revisions in `uv.lock`.
