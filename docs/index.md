# LAN pipeline

LAN_pipeline_minimal coordinates the operational path from simulated training
data to a likelihood network that is ready for HSSM. It gives operators one
place to prepare reproducible jobs, carry MLflow lineage through data generation
and training, validate candidate ONNX artifacts, and publish an approved network.

!!! warning "Active infrastructure"

    This project is operational infrastructure under active development. Its
    lockfile currently pins unreleased `main` revisions of ssm-simulators and
    LANfactory, and its cluster defaults describe a particular Brown Oscar
    allocation. Start with the local quick workflow, inspect every generated
    script, and validate a candidate before publishing it.

## Install and verify

With Python 3.12, uv, GSL, and a compiler available:

```bash
uv sync --locked
bash local_test_run.sh
```

The [local workflow tutorial](learn/local-workflow.md) explains the generated
data, trained artifacts, and MLflow lineage, plus platform-specific prerequisites.

## Where the pipeline fits

| Stage | Owned by | What this repository adds |
| --- | --- | --- |
| Simulate | [ssm-simulators](https://lnccbrown.github.io/ssm-simulators/) | Cluster submission, resource selection, and run identity |
| Train and export | [LANfactory](https://lnccbrown.github.io/LANfactory/) | Training submission and a link back to the source data experiment |
| Validate | LAN_pipeline_minimal | Structural, trainer-parity, HSSM-load, and density gates |
| Consume | [HSSM](https://lnccbrown.github.io/HSSM/) | A differentiable or black-box likelihood for Bayesian inference |

The pipeline orchestrates those packages; it does not replace their scientific
or API documentation. It is intentionally small, config-driven, and centered
on operator-visible files and machine-readable command results.

## Choose a path

- **New to the repository?** Follow the [learning path](learn/index.md), beginning
  with the same local workflow exercised by CI.
- **Preparing or running cluster work?** Use the [operator guides](how-to/index.md)
  for cluster discovery, Slurm scripts, MLflow, validation, and publishing.
- **Reviewing a design or a failure mode?** Start with the
  [conceptual explanations](explanations/index.md).
- **Automating the pipeline?** Use the [interface reference](reference/index.md)
  for commands, configuration, environment variables, and JSON output.

## The safe operating sequence

1. Run the quick local workflow against the tracked lockfile.
2. Discover personal cluster lanes and review the merged resource plan.
3. Generate data, preserving the reported MLflow experiment ID.
4. Train a network linked to that data-generation experiment.
5. Stage one training run and run every applicable validation gate.
6. Dry-run publication, then publish only the validated staged artifact.

The rest of this site makes each handoff and safety boundary explicit.
