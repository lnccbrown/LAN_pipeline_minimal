# MLflow guidance has moved

The documentation site is now canonical:

- [Track runs and preserve lineage](https://lnccbrown.github.io/LAN_pipeline_minimal/how-to/track-with-mlflow/)
- [Data lineage and run identity](https://lnccbrown.github.io/LAN_pipeline_minimal/explanations/data-lineage/)
- [Environment variables](https://lnccbrown.github.io/LAN_pipeline_minimal/reference/environment/)
- [JSON output contracts](https://lnccbrown.github.io/LAN_pipeline_minimal/reference/json-output/)

The essential handoff is unchanged: data-generation workers share a
`{model}-data-generation` experiment, and training receives that experiment's
ID through `--data-generation-experiment-id`. The linked site owns tracking
backend, shared-path, run-identity, and publication-record details.

This root file remains only as a pointer for existing links. Put future MLflow
operator guidance in the site pages above.
