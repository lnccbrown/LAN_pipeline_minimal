# Run a parameter-recovery sweep

Recovery asks whether a network supports inference, not whether it is a
well-formed density. For why the harness is shaped the way it is — coverage
versus contraction, the reference arms, the ladder — see
[Parameter recovery](../explanations/parameter-recovery.md).

## Install the validation profile

```bash
uv sync --locked --group validate
```

## Fit one dataset

One invocation is one fit is one shard, which is also one Slurm array task.

```bash
uv run python validation/recover_parameters.py \
  --model ddm_sdv --design L1_n500 --dataset-index 7 \
  --likelihood approx_differentiable --onnx-path /path/to/ddm_sdv.onnx \
  --out-dir results/
```

Adding a model needs no code change: the parameter list, bounds and available
likelihood kinds are read from HSSM's own model config.

## Run the reference arm

Same data, same priors, no network — available only for models that declare an
analytical likelihood.

```bash
uv run python validation/recover_parameters.py \
  --model ddm_sdv --design L1_n500 --dataset-index 7 \
  --likelihood analytical --out-dir results/
```

`--bounds-from` decides whose bounds become the priors and defaults to the
network's box. Keep it identical across arms or the comparison is not paired.

## Vary a different parameter

Any parameter may be the one that varies across conditions. Each choice is a
separate design, written to its own shards and scored separately.

```bash
uv run python validation/recover_parameters.py \
  --model ddm_sdv --design L1_n500 --condition-param sv --dataset-index 7 \
  --likelihood approx_differentiable --onnx-path /path/to/ddm_sdv.onnx \
  --out-dir results/
```

## Aggregate into a verdict

```bash
uv run python validation/aggregate_recovery.py --shard-dir results/
```

Writes `recovery_report.json`, prints one JSON line, and exits non-zero when
the verdict is not a pass. The line carries `n_usable_fits` and
`n_errored_shards` alongside `n_shards`, so a driver can tell a clean sweep
from one where most tasks died.

## Set `--p-outlier` deliberately

It defaults to `None` here, **not** to HSSM's `0.05`. HSSM's default fits
`0.95·f(rt|θ) + 0.05·Uniform(0, 20)` while the simulator generates no lapse
process at all. That mismatch is a fixed misspecification, so the bias in the
posterior mean stays constant while the posterior sd shrinks as 1/√n —
coverage gets *worse as the dataset grows*, which reads exactly like a broken
likelihood. Measured on plain `ddm` with its exact analytical likelihood over
twelve datasets:

| n | `p_outlier` | coverage | mean \|z\| |
| --- | --- | --- | --- |
| 500 | 0.05 | 0.88 | 0.96 |
| 500 | none | **0.96** | 0.62 |
| 2000 | 0.05 | 0.77 | 1.24 |
| 2000 | none | **0.90** | 0.91 |

Set it to a real value only when the data really were generated with lapses.
Either way it is recorded in every shard.

## Cluster notes

Use one core per task. `cores` is a no-op under `sampler="numpyro"` — pymc's
`_sample_external_nuts` never forwards it — and chains fall back to sequential
on a single JAX device, so parallelise across fits rather than across chains.
That also puts recovery on the `batch` partition instead of the GPU QOS.

Pass `--onnx-path` as a local path. HSSM otherwise downloads the ONNX from
`franklab/HSSM` at construction time, where a candidate network does not exist,
and compute nodes may have no egress.
