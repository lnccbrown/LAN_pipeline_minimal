# Promotion safety

Publishing changes an artifact under a name that HSSM users resolve at runtime.
A syntactically valid but wrong network can therefore affect downstream
inference immediately. The promotion path is designed to fail closed before
that boundary.

For the operator procedure, see
[Stage and publish a validated network](../how-to/stage-and-publish.md).

## The safety layers

### 1. Completed-run selection

The publisher accepts an explicit MLflow run ID or resolves the newest run for a
model/network-type pair. In either case, the run must carry `run_uuid`. MLflow
status alone is rejected as evidence of completed training.

### 2. Isolated staging

Only files whose names contain the selected `run_uuid` are copied. The staging
directory must contain exactly one ONNX, and it may not contain unrelated files.
Copying rather than symlinking keeps LANfactory's folder checks and upload
manifest aligned with the isolated artifact set.

### 3. Ordered validation

The gates run from cheapest to most informative:

1. **Structure** enforces the concrete-dimension, single-input, scalar-output
   ONNX contract.
2. **Parity** compares against the trainer state when that evidence exists.
3. **HSSM load** checks the actual downstream integration and a finite initial
   log probability.
4. **Density** checks mass and simulator-relative distribution quality.

Structure, HSSM load, and density must be present, run, and pass. A missing or
skipped required gate is a refusal even if the report's aggregate `passed` field
is true. Parity may skip because Torch artifacts legitimately lack JAX state.

### 4. Dry-run review

`--dry-run` reads MLflow, stages files, runs validation, and prints the complete
plan, but writes neither Hugging Face nor MLflow publication state. A persistent
staging directory lets the operator inspect the exact files and report before
repeating the command.

### 5. Production is reachable only deliberately

The CLI refuses `franklab/HSSM` by default, normalized for capitalization,
whitespace, and trailing slashes. That repository is the production source for
released HSSM versions, so the ordinary path uploads elsewhere, where humans and
downstream checks can review the candidate.

Promotion requires `--allow-production` *and* retyping the repo id at an
interactive prompt. The second half is the load-bearing one: a flag survives
shell history, a copied runbook line, and a re-run of the wrong command, so a
flag alone would be the ordinary invocation this check exists to prevent.
`--dry-run` is not prompted, because it writes nothing.

### 6. Explicit replacement and verifiable records

Replacing an existing canonical root filename requires `--overwrite-root`.
After an upload, the publisher reads repository head and records a trusted
`hf_commit` only when the commit message matches this operation. Otherwise it
records an explicitly uncertain candidate SHA.

## What the guardrails do not decide

The pipeline can demonstrate that an artifact is structurally compatible,
faithful to an available trainer state, loadable by HSSM, and plausible against
simulation under the implemented density checks. It does not decide:

- whether the training design covers the intended scientific domain;
- whether thresholds are appropriate for a new model family;
- whether staging review is sufficient for a production release;
- who is authorized to promote into the production repository.

Those are review and governance decisions. The production refusal exists so
they cannot be collapsed into an ordinary CLI invocation.

## Interpreting failure

| Failure | Meaningful next action |
| --- | --- |
| No `run_uuid` | Confirm that training completed and wrote artifacts; do not select by status |
| Mixed staging directory | Choose a clean destination or remove unrelated files yourself |
| Structure failure | Fix the exporter or wrong-model/network-type selection |
| Parity failure | Compare the ONNX with the exact trusted trainer state |
| HSSM-load failure | Diagnose the consumer contract before evaluating density |
| Density failure | Inspect KDE/manifold plots and revisit training data or model quality |
| Existing root artifact | Review the target and use `--overwrite-root` only for an intentional replacement |
| Production-repository refusal | Complete the separate staging review and governed promotion process |
