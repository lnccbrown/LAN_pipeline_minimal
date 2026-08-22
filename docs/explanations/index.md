# Concepts

The explanation pages describe the pipeline as a set of boundaries and records:
which package owns each computation, how artifacts move between stages, and why
validation and staging stand between training and publication.

Read these pages when reviewing a proposed workflow, diagnosing a handoff, or
deciding which identifier and artifact should be authoritative.

- [Pipeline architecture](architecture.md) separates computation, scheduling,
  evidence, and promotion responsibilities.
- [Data lineage and run identity](data-lineage.md) traces a network back through
  MLflow experiments, runs, artifact identifiers, configs, and the lockfile.
- [Promotion safety](promotion-safety.md) explains staging, gate semantics,
  destination refusal, and the decisions that remain human-governed.
