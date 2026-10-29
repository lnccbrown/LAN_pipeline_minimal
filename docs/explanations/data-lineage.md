# Data lineage and run identity

A network is meaningful only together with the simulation, configuration, code,
and training run that produced it. The pipeline carries several identifiers
because each one answers a different question.

For the operator procedure, see
[Track runs and preserve lineage with MLflow](../how-to/track-with-mlflow.md).

## The lineage chain

```text
uv.lock + generation config
            |
            v
data-generation experiment
  |-- worker run 1 -> generated file inventory
  |-- worker run 2 -> generated file inventory
  `-- worker run N -> generated file inventory
            |
            | data_generation_experiment_id
            v
training run -> run_uuid -> run-named artifact set
            |
            v
validation_report.json
            |
            v
publication run -> Hugging Face URL + verified commit (when available)
```

## The derived-from-LAN chain

An auxiliary network (cpn, opn) has no data-generation experiment. Its corpus
is integrated from a published LAN by LANfactory's `derive-aux`, so its lineage
points at another network rather than at simulator workers:

```text
published LAN (root file on the Hub)
  |-- source_lan_sha256      the exact bytes integrated
  |-- source_lan_hf_commit   the Hub revision they were downloaded at
  `-- source_lan_run_uuid    the LAN's own training run
            |
            | derivation_method, aux_category,
            | integration_grid, integration_max_t
            v
derived corpus -> training run (params above; tags derive_total_mass_*,
                                derive_fallback_frac, derive_sim_past_max_t_max,
                                derive_leak_below_onset_p99)
            |
            v
validation_report.json (structure, parity, hssm_missing_load, accuracy)
            |
            v
publication run -> Hugging Face URL + the same provenance params
```

The keys are one contract shared by three writers: the derived corpus's
`generator_config["source"]`, the training run's MLflow params, and the
publication run's params. `derivation_method` is `derived-from-lan` or
`trained-from-simulation`, though only the former has a publish path today --
a simulation-trained auxiliary network has no source LAN for this chain to
point at; `aux_category` is `choice` for a cpn and `omission` for an opn.

The training run also carries the derived corpus's tail-policy record as
tags, logged by LANfactory from the corpus manifest. The labels of a derived
corpus are renormalised by the source LAN's own total on the integration
grid, and the record says how much that renormalisation had to do and where
it was not trusted:

| Tag | What it records |
| --- | --- |
| `derive_total_mass_{mean,min,max}` | The per-file total mass the LAN put on the grid, before renormalisation -- how far from normalised the source LAN was |
| `derive_fallback_frac` | The share of the parameter box labelled by simulation instead, because the LAN's total was outside (0.98, 1.03) |
| `derive_sim_past_max_t_max` | The most simulated mass any file placed past `max_t` |
| `derive_leak_below_onset_p99` | The p99, over files, of the mass the LAN leaked below the onset (non-decision time) |

The numbers are kept so that inheritance can be read rather than hidden: a
LAN whose total drifts at the box edge shows up as a fallback fraction on
every network derived from it. The publisher forwards each tag to the
publication run when the training run carries it (a tag MLflow stored as
`"None"` or `""` counts as absent). The generated model card quotes the mass
stats, the fallback fraction and the onset leak, and states the absence of
each a training run lacks; `derive_sim_past_max_t_max` is quoted only when
present, and `data_origin` travels as a tag without being quoted.

The publisher refuses an auxiliary run that lacks any required key, so a
network on the Hub can always be traced back to the LAN it was integrated
from -- which is what makes a later problem in that LAN actionable.

## Why training links to an experiment

One data-generation submission can create many worker runs, and multi-lane
fan-out can create several Slurm arrays. The training dataset is their union.
A single worker run ID would therefore name only a fragment of the input.

`data_generation_experiment_id` names the collection. LANfactory can aggregate
worker inventories from that experiment and compare them with the files in the
training folder. Preserve the ID printed by `lan-sbatch generate` and pass it to
the training command.

## Why artifacts link to `run_uuid`

LANfactory stores multiple runs for a model in one flat directory. File names
carry a `run_uuid`, but JAX and Torch exporters place it in different positions.
The publisher matches it anywhere in the filename and copies that set into an
isolated directory.

The MLflow status is not a safe completion signal. The submission process ends
its handle after `sbatch` returns, before the compute job may have started. A
training run without `run_uuid` has not reached the artifact-producing stage and
is ineligible for publication.

## Which record is authoritative

| Question | Record to trust |
| --- | --- |
| What dependency revisions ran? | The checkout's tracked `uv.lock` |
| Which Slurm submission landed? | Each submission JSON object's `job_id`, account, partition, and script path |
| Which workers formed the data source? | Runs and file inventories in the generation experiment |
| Which generation collection trained the network? | `data_generation_experiment_id` on the training run |
| Which files belong to the training run? | The run's `run_uuid` and matching artifact names |
| Which LAN was an auxiliary network integrated from? | `source_lan_sha256` and `source_lan_hf_commit` on the training and publication runs |
| Which checks ran on the candidate? | `validation_report.json`, including each gate's skipped state |
| What was uploaded? | The publication run and Hugging Face URL |
| Which remote revision is confirmed? | `hf_commit` only when `hf_commit_verified` is true |

An `hf_commit_candidate` is deliberately weaker than `hf_commit`: another
writer may have moved repository head between upload and read-back.

## Preserve the chain

- Keep every JSON line from a multi-lane submission; partial success is still
  work running on the cluster.
- Point all stages at one authoritative MLflow store rather than a disposable
  mirror.
- Treat copied artifact folders as inputs: preserve names containing the
  `run_uuid`.
- Keep the validation report beside the staged artifacts that produced it.
- Do not infer lineage from timestamps or "latest" when a stable identifier is
  available.
