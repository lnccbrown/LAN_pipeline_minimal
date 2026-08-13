#!/usr/bin/env python3
"""Look at a candidate network before trusting the gate's verdict.

`validate_network.py` returns numbers. This is the human counterpart: it puts
the network's implied density next to kernel density estimates from the
simulator it is meant to approximate, so "reasonable" is something you can see
rather than infer from a Hellinger ratio.

Run it against any artifact:

    export INSPECT_ONNX=/path/to/..._model.onnx
    export INSPECT_MODEL=ddm_sdv
    uv run --group inspect marimo edit validation/inspect_network.py

or render it without a browser:

    uv run --group inspect marimo export html validation/inspect_network.py \\
        -o inspection.html

The predictor adapter is the only non-obvious part. lanfactory's inspectors
were written against the torch backend and call the network with a whole
(n_grid, n_params + 2) batch; ecosystem ONNX artifacts take exactly one trial
per call by contract, so the adapter loops. It is the same asymmetry the
parity gate has to live with.
"""

import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Does this network look like the simulator?

        Two views, both from `lanfactory.network_inspectors`:

        1. **KDE vs LAN** — the network's likelihood over an RT grid, drawn on
           top of repeated kernel density estimates from actual simulations at
           the same parameters. A usable network traces the KDE cloud. An
           untrained one wanders off it, and you can see exactly where.
        2. **Manifold** — the same likelihood swept across one parameter. A
           trained network deforms smoothly; an untrained one is lumpy or flat.

        Neither replaces the gate. They explain *what* the gate objected to.
        """
    )
    return


@app.cell
def _(mo):
    import json
    import os
    from pathlib import Path

    # The notebook's entire input is this one variable, so say so rather than
    # letting a bare KeyError land in the cell output of a human-facing tool.
    _onnx_env = os.environ.get("INSPECT_ONNX")
    if not _onnx_env:
        raise RuntimeError(
            "Set INSPECT_ONNX to the artifact to inspect, then reopen:\n"
            "    export INSPECT_ONNX=/path/to/..._model.onnx\n"
            "    export INSPECT_MODEL=ddm_sdv   # optional, defaults to ddm"
        )
    onnx_path = Path(_onnx_env).expanduser()
    if not onnx_path.is_file():
        # Caught here rather than in the onnxruntime cell below, so the cells
        # in between do not render a confident header for a file that is absent.
        raise FileNotFoundError(f"INSPECT_ONNX does not exist: {onnx_path}")
    model_name = os.environ.get("INSPECT_MODEL", "ddm")

    # The gate writes this next to the artifact when it runs. Optional: the
    # notebook is useful on a network that was never gated.
    report_path = onnx_path.parent / "validation_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None

    mo.md(f"""
    **Artifact** `{onnx_path.name}`
    **Model** `{model_name}`
    **Gate report** {"found" if report else "not found — showing plots only"}
    """)
    return json, model_name, onnx_path, report


@app.cell
def _(mo, report):
    def _verdict_table(report):
        if not report:
            return mo.md("_No gate report alongside this artifact._")
        rows = []
        for gate in report["gates"]:
            scores = {
                k: v
                for k, v in gate.items()
                if k
                in (
                    "max_abs_error",
                    "initial_logp",
                    "worst_ratio",
                    "worst_total_mass",
                    "input_shape",
                )
            }
            mark = (
                "skipped"
                if gate.get("skipped")
                else ("pass" if gate["passed"] else "FAIL")
            )
            detail = ", ".join(f"`{k}`={v}" for k, v in scores.items())
            rows.append(f"| {gate['gate']} | {mark} | {detail} |")
        return mo.md("| gate | verdict | scores |\n|---|---|---|\n" + "\n".join(rows))

    _verdict_table(report)
    return


@app.cell
def _(model_name, onnx_path):
    import numpy as np
    import onnxruntime as ort

    _session = ort.InferenceSession(str(onnx_path))
    _input_name = _session.get_inputs()[0].name

    def predict_on_batch(batch):
        """Adapt a single-trial ONNX to the inspectors' batch predictor.

        The ecosystem contract fixes the graph's batch dimension at 1 — a
        stacked feed is rejected by onnxruntime outright — so the batch is
        evaluated row by row and restacked as (n, 1), which is what
        `evaluate_network` indexes with `[:, 0]`.
        """
        rows = np.asarray(batch, dtype=np.float32)
        return np.stack(
            [
                np.asarray(
                    _session.run(None, {_input_name: rows[i : i + 1]})[0]
                ).reshape(-1)
                for i in range(rows.shape[0])
            ]
        )

    import ssms

    _config = ssms.config.model_config[model_name]
    param_names = list(_config["params"])
    lower, upper = (np.asarray(b, dtype=float) for b in _config["param_bounds"])
    return lower, np, param_names, predict_on_batch, upper


@app.cell
def _(lower, mo, np, param_names, upper):
    import pandas as pd

    # Away from the edges: a network is not expected to be accurate at the very
    # boundary of the space it was trained on, and testing there produces
    # failures that say nothing about ordinary use. Same 10% the gate uses.
    _shrink = 0.1
    _span = upper - lower
    # Not underscore-prefixed like the rest: marimo keeps `_`-names cell-local,
    # and the manifold sweep below has to land inside the same box these draws
    # come from.
    lo_shrunk, hi_shrunk = lower + _shrink * _span, upper - _shrink * _span

    _rng = np.random.default_rng(0)
    n_parameter_sets = 3
    parameter_df = pd.DataFrame(
        lo_shrunk
        + (hi_shrunk - lo_shrunk)
        * _rng.uniform(size=(n_parameter_sets, len(param_names))),
        columns=param_names,
    )
    mo.ui.table(parameter_df.round(3), label="Parameter vectors under inspection")
    return hi_shrunk, lo_shrunk, parameter_df, pd


@app.cell
def _(mo):
    mo.md("""## 1. Network likelihood vs simulated KDEs""")
    return


@app.cell
def _(model_name, parameter_df, predict_on_batch):
    from lanfactory.network_inspectors import kde_vs_lan_likelihoods

    kde_vs_lan_likelihoods(
        parameter_df=parameter_df,
        model=model_name,
        torch_mlp_predict=predict_on_batch,
        n_samples=2000,
        n_reps=5,
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. Likelihood manifold across a swept parameter

        Drift is the parameter whose effect is easiest to read: raising it
        should move mass toward one choice and sharpen the RT peak.
        """
    )
    return


@app.cell
def _(
    hi_shrunk,
    lo_shrunk,
    model_name,
    np,
    param_names,
    parameter_df,
    predict_on_batch,
):
    from lanfactory.network_inspectors import lan_manifold

    # Sweep drift where the model has it, otherwise the first parameter, so
    # this cell does not assume a particular model's parameterisation — and
    # sweep it across its own trained range rather than a fixed window. The
    # models without a `v` lead with a strictly positive parameter (race `v0`,
    # lba `A`, poisson_race `r1`), where a hardcoded -1.5..1.5 is mostly
    # domain the network was never shown.
    _sweep = "v" if "v" in param_names else param_names[0]
    _i = param_names.index(_sweep)

    lan_manifold(
        parameter_df=parameter_df,
        vary_dict={_sweep: list(np.linspace(lo_shrunk[_i], hi_shrunk[_i], 9))},
        model=model_name,
        torch_mlp_predict=predict_on_batch,
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ---
        **Reading these.** A network worth publishing sits inside the spread of
        the KDE replicates — the replicates themselves disagree, and that
        disagreement is the noise floor the gate calibrates against. A network
        that is above, below, or the wrong shape is not close; no amount of
        threshold tuning fixes it, and the gate's `worst_ratio` is measuring
        exactly this gap.
        """
    )
    return


if __name__ == "__main__":
    app.run()
