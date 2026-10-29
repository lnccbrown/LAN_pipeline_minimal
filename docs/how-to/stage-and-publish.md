# Stage and publish a validated network

`lan-publish` resolves one completed training run, copies only that run's
artifacts into an isolated directory, runs the validation gate, uploads to a
non-production Hugging Face repository, and records the result in MLflow.

The [Promotion safety](../explanations/promotion-safety.md) explanation describes
why the staging, validation, and destination guardrails fail closed.

The command is intended to run on an operator machine, not a compute node. Keep
the Hugging Face credential there and point MLflow at the authoritative store.

## Select the training run explicitly

Prefer an MLflow run ID when promoting a known candidate:

```bash
export MLFLOW_TRACKING_URI="sqlite:////shared/path/mlflow/tracking.db"

uv run lan-publish \
  --hf-repo your-org/HSSM_staging \
  --run-id "$TRAINING_RUN_ID" \
  --artifact-dir /local/path/to/training/artifacts \
  --staging-dir /local/path/to/staged-candidate \
  --dry-run
```

You may instead pass `--model` and `--network-type`; the command takes the most
recent matching run with a `run_uuid`. That convenience is useful for
exploration, but an explicit run ID makes an audited promotion unambiguous.

If the training output path recorded in MLflow exists locally,
`--artifact-dir` can be omitted. Cluster paths commonly do not, so fetch the
artifacts first and name the local source explicitly.

## Why staging is mandatory

LANfactory writes multiple training runs into a flat model directory. The
publisher uses the selected run's `run_uuid` to copy its files into an isolated
staging directory, then requires exactly one ONNX there. This prevents three
silent errors:

- uploading files from a different run;
- matching the wrong trainer state during parity validation;
- writing publication-generated files back into the training output.

The entire staging directory is uploaded. If a persistent `--staging-dir`
contains unrelated files, the command refuses to proceed. It preserves files
from its own earlier dry run or failed gate so you can inspect the report.

## Read the dry-run plan

A successful dry run prints one JSON object with `dry_run: true` and
`published: false`, and exits 0. A dry run a gate refused prints the plan with
an `error` and exits 1, as a real attempt does. Review:

- `training_run_id` and `run_uuid`;
- `model`, `network_type`, and target `hf_repo`;
- the complete `staged` filename list;
- the canonical `root_filename` HSSM will request;
- `gate`, which must say all required gates ran and passed;
- `provenance`, empty for a LAN and the derive-aux keys for a cpn or opn.

`--skip-density` can make a dry rehearsal faster, but its gate result cannot
authorize a real upload.

## Publish to staging

After reviewing the dry run, repeat the command without `--dry-run`:

```bash
uv run lan-publish \
  --hf-repo your-org/HSSM_staging \
  --run-id "$TRAINING_RUN_ID" \
  --artifact-dir /local/path/to/training/artifacts \
  --staging-dir /local/path/to/staged-candidate
```

If the target already has the canonical root artifact, replacement requires an
explicit `--overwrite-root`. That flag is consequential: HSSM consumers of the
target resolve the root name.

Non-production destinations are the default. The CLI refuses `franklab/HSSM`
unless told otherwise, including capitalization and trailing slash variants:
that repository is the production source used by released HSSM versions.
Promoting to it requires `--allow-production` and retyping the repo id at an
interactive terminal -- piped input is refused outright, so an ordinary script
or copied runbook line cannot answer the prompt.

## Publish an auxiliary network (cpn, opn)

A choice-probability network (`cpn`) or omission-probability network (`opn`)
publishes through the same command. The root filename is derived from the
base model and the type -- `ddm_sdv_cpn.onnx`, `ddm_sdv_opn.onnx` -- because
HSSM builds `{model}{suffix}.onnx` from the model name it is given. Three
things differ from a LAN publish.

**The training run must carry its provenance.** An auxiliary network is
integrated from a LAN, and the publisher refuses to ship one whose source LAN
it cannot name. LANfactory logs these params on a training run started from a
`derive-aux` corpus:

| Param | Value |
| --- | --- |
| `derivation_method` | `derived-from-lan`; the contract also names `trained-from-simulation`, which no writer emits yet and the publisher refuses, since the generated card would assert a LAN lineage |
| `aux_category` | `choice` for a cpn, `omission` for an opn |
| `source_lan_run_uuid`, `source_lan_sha256`, `source_lan_hf_commit` | The LAN the corpus was integrated from; `source_lan_run_uuid` may be empty for a legacy Hub LAN (the 2023 artifacts predate run uuids), the other two never |
| `integration_grid`, `integration_max_t` | The quadrature the corpus was built on |

A run started from a simulated corpus carries none of them and is refused,
naming the first missing key, before anything is staged. An empty value, or
the string `None` MLflow stores for a param logged as `None`, counts as
missing. Relabel it with the source LAN's identity, or retrain from a derived
corpus. A run whose
`aux_category` does not match its type (a `cpn` labelled `omission`) is
refused too: it was derived for something other than what its root filename
would promise.

**The auxiliary gate set applies.** `structure`, `hssm_missing_load`, and
`accuracy` must be present, run, and pass; `parity` may skip as for a LAN.
`hssm_missing_load` pairs the candidate with the base LAN, which HSSM
downloads by name -- pass `--lan-onnx` to use a local LAN, and for a model
outside HSSM's registry you must. `--skip-accuracy` shortens a dry run the
way `--skip-density` does for a LAN; neither can publish. Under HSSM < 0.6.0
the cpn `hssm_missing_load` gate skips itself, and a skipped required gate is
a refusal, so a cpn cannot be published until the locked HSSM moves.

**A model card is generated unless you staged one.** With no
`model_card.yaml` beside the artifacts, the publisher writes one from the
provenance and the gate report: the input contract with the trailing column
named, the source LAN's sha256, `run_uuid`, and Hub commit, the integration
grid, the accuracy numbers against the Monte-Carlo truth's standard error, the
tail-policy caveat, and a `hssm.HSSM(..., missing_data=True)` usage example
(`deadline=True` with a `deadline` column for an opn). It is round-tripped
through LANfactory's card loader before staging, so a card that would crash
the upload stops the publish on the laptop. An operator card in the artifact
folder is staged instead and never overwritten.

```bash
uv run lan-publish \
  --hf-repo your-org/HSSM_staging \
  --run-id "$CPN_TRAINING_RUN_ID" \
  --artifact-dir /local/path/to/trained/cpn/ddm_sdv \
  --staging-dir /local/path/to/staged-cpn \
  --lan-onnx /local/path/to/ddm_sdv.onnx \
  --dry-run
```

`--artifact-dir` is the trainer's output folder for the run
(`<output_path>/cpn/<model>`), not the `derive-aux` corpus the training config
points at; the publisher looks for artifacts carrying the run's `run_uuid`
there.

The dry-run plan gains a `provenance` block, shows under `forwarded_tags` the
training-run tags the publish run would carry, and lists the generated
`model_card.yaml` under `staged`; open it before repeating without
`--dry-run`. The publish run records the provenance params beside the source
run identity, forwards the training run's tail-policy record
(`derive_total_mass_{mean,min,max}`, `derive_fallback_frac`,
`derive_sim_past_max_t_max`, `derive_leak_below_onset_p99`) and `data_origin`
tags, and logs the accuracy and missing-load scores as metrics. Each tag is
forwarded only when the training run carries it (a tag stored as `"None"` or
`""` is dropped as absent). The card quotes the mass stats, the fallback
fraction and the onset leak and states the absence of each it lacks -- an
older training run has no fallback fraction or onset leak, and the card says
so instead of inventing one; `derive_sim_past_max_t_max` is quoted only when
present, and `data_origin` is forwarded but not quoted.

Two refusals are absolute, and both fire before provenance is read or anything
is staged. Only `lan`, `cpn` and `opn` publish: any other network type (a
`gonogo`, say, which nothing in HSSM consumes) is refused, because a root
filename on the Hub is permanent, and a run without provenance gets this
refusal, not an instruction to relabel it. A `_deadline`
model name is refused: the deadline variant is derived internally wherever a
simulation needs it, and HSSM never asks for `ddm_sdv_deadline_opn.onnx`.
Publish under the base model.

## Verify the records

On success, the JSON result contains the Hugging Face URL, the publication
MLflow run ID, and a commit SHA only when the uploader can verify that the
repository head matches this upload. An unverified read-back is labeled
`hf_commit_candidate`, never `hf_commit`.

The publication experiment records the source training run, source `run_uuid`,
validation report and scores, target repository, and upload result. The training
run receives back-reference tags when the tracking store allows writes.
