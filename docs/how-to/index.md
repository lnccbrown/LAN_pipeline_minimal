# Choose an operator task

These task-oriented guides cover the work that surrounds simulation and network
training: discovering resources, submitting jobs, preserving lineage, checking
candidate artifacts, and promoting a network safely.

Use the quick-test configs while learning. The files under `configs/examples/`
are templates for larger runs, not universal production settings.

## Prepare and submit work

- [Configure your cluster resources](configure-cluster.md) to discover personal
  lanes without committing personal allocations.
- [Generate and submit Slurm jobs](submit-slurm-jobs.md) to rehearse scripts,
  select resources, fan out generation, and preserve submission results.
- [Track runs with MLflow](track-with-mlflow.md) to keep generation, training,
  and promotion linked through one authoritative store.

## Promote an artifact

- [Validate and inspect a candidate network](validate-network.md) with the four
  automated gates and the optional visual inspector.
- [Stage and publish a validated network](stage-and-publish.md) to isolate one
  run, inspect a dry-run plan, and upload only to a staging repository.

The promotion guides deliberately stop short of the production Hugging Face
repository. The normal `lan-publish` CLI refuses that destination so production
promotion remains an explicit, separately governed action.
