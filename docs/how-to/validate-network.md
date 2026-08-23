# Validate and inspect a candidate network

Validation asks a narrow operational question: will HSSM load this ONNX
likelihood and obtain numerically and scientifically plausible values? Run it
on artifacts you trust before any publication attempt.

## Install the validation profile

```bash
uv sync --locked --group validate
```

This adds HSSM and ONNX Runtime to the normal pipeline environment. The docs
environment is intentionally separate and cannot run these gates.

## Run all four gates

```bash
uv run python validation/validate_network.py \
  --onnx-path /path/to/run_uuid_lan_ddm__network.onnx \
  --model-name ddm \
  --network-type lan
```

The command writes `validation_report.json` next to the ONNX unless
`--report-path` is given. It prints one compact JSON line and exits non-zero
when a gate fails.

| Gate | Evidence |
| --- | --- |
| G1 `structure` | ONNX loads; all input dimensions are concrete; there is one input, one scalar output, and the input width matches the model |
| G2 `parity` | The exported ONNX matches the JAX trainer state when the required siblings exist |
| G3 `hssm_load` | HSSM accepts the likelihood and obtains a finite initial log probability |
| G4 `density` | Integrated mass is near one and Hellinger error is acceptable relative to a measured simulator sampling floor |

G2 legitimately skips for Torch artifacts and for a bare ONNX without the JAX
state/config pair. G1, G3, and G4 are required for publication. A skipped gate
may carry `passed: true` in the detailed report to mean it did not itself fail;
inspect the compact `gates` states rather than treating that as evidence the
check ran.

!!! danger "Only validate trusted artifact folders"

    The parity gate unpickles the sibling `*_network_config.pickle`. Unpickling
    can execute code. Validate folders you produced or fetched from a repository
    you control; do not point the command at arbitrary downloads.

## Use skips only for diagnosis

`--skip-hssm` and `--skip-density` shorten a diagnostic run, but do not create
publishable evidence. In particular, the publisher refuses any report where a
required gate is skipped or missing.

If the density comparison needs a reviewed tolerance change, pass
`--hellinger-ratio-max` and preserve the resulting threshold in the report.
Do not loosen it merely to turn one candidate green.

## Inspect the result visually

The marimo inspector compares the network's implied likelihood with simulator
KDEs and shows a parameter manifold. It explains a gate result; it does not
replace one.

```bash
export INSPECT_ONNX="/path/to/run_uuid_lan_ddm__network.onnx"
export INSPECT_MODEL="ddm"
uv run --group inspect marimo edit validation/inspect_network.py
```

To produce a static local report without opening an editor:

```bash
uv run --group inspect marimo export html validation/inspect_network.py \
  -o inspection.html
```

If `validation_report.json` is present, the inspector displays it only when its
recorded ONNX filename matches the artifact being viewed.
