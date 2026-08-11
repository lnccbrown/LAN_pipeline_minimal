#!/usr/bin/env python3
"""Validate a trained likelihood network before it is published.

Publishing puts a file under a name every released HSSM downloads, so the
question this answers is narrow: *would HSSM load this and get sane numbers?*

Four gates, cheapest first, each independently reported so a failure says
which property broke:

    G1 structure  — ONNX loads, onnxruntime builds a session, every input dim
                    is concrete, exactly one tensor goes in and one log-density
                    comes out, and the input width matches the parameter space
                    (the ecosystem's single-trial contract).
    G2 parity     — ONNX output equals the trainer's own jax forward pass,
                    when the *_train_state.jax sibling is present.
    G3 hssm-load  — hssm.HSSM accepts it as a likelihood and produces a finite
                    initial logp on simulated data.
    G4 density    — the network's implied density integrates to ~1 and matches
                    simulation (Hellinger) at several in-bounds parameter
                    draws. This is the only gate that catches a network that
                    loads perfectly and has learned nothing.

G1-G3 are mechanical. G4 is statistical, and it calibrates itself against the
sampling noise of the simulator rather than against a fixed number — see
DEFAULT_HELLINGER_RATIO_MAX.

Trust: G2 unpickles the ``*_network_config.pickle`` sitting next to the ONNX,
because that is the only format lanfactory writes it in, and unpickling runs
whatever the file says. Point this at artifact folders you produced or fetched
from a repository you control. G2 skips itself when the flax sibling is absent,
so validating a bare downloaded ``{model}.onnx`` reads no pickle at all.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import typer

logger = logging.getLogger("validate_network")

app = typer.Typer(add_completion=False)

DEFAULT_PARITY_ATOL = 1e-4
DEFAULT_MASS_RANGE = (0.9, 1.1)

# G4 compares the network's density to simulation via Hellinger distance, but
# an *absolute* Hellinger bound is meaningless on its own: with finite samples
# two draws from the very same distribution already differ. Measured on ddm at
# n_sim=20000, n_grid=200, that sampling floor is 0.053-0.076 — larger than the
# error of the production network itself, so a fixed 0.10 bound would have
# failed a network that demonstrably works, and any fixed number silently
# depends on n_sim and n_grid.
#
# So the gate measures its own floor per parameter draw (one extra simulation,
# compared against the first) and judges the ratio. A perfect network scores
# about 1/sqrt(2) — it is smooth, so only one side of the comparison carries
# sampling noise. Measured ratios for the production ddm.onnx across 8 draws:
# 0.71, 0.71, 0.76, 0.83, 0.84, 0.92, 1.10, 1.90. The bound below leaves room
# above that worst case while still catching a network several times noisier
# than sampling error.
DEFAULT_HELLINGER_RATIO_MAX = 3.0

# LANs take [*params, rt, response]; cpn/opn take the parameters only.
EXTRA_INPUTS_BY_NETWORK_TYPE = {"lan": 2, "cpn": 0, "opn": 0, "gonogo": 0}


def _result(name: str, passed: bool, **details: Any) -> dict:
    return {"gate": name, "passed": bool(passed), **details}


def gate_structure(onnx_path: Path, expected_input_dim: int | None) -> dict:
    """G1: the ONNX satisfies the ecosystem's load-time contract.

    HSSM's make_jax_func rejects symbolic dims outright, so a graph with a
    dynamic axis fails at load for every user rather than here.
    """
    import onnx
    import onnxruntime as ort

    try:
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
    except Exception as e:  # noqa: BLE001 - any load failure is a gate failure
        return _result("structure", False, error=f"onnx load/check failed: {e}")

    dims = []
    for graph_input in model.graph.input:
        for dim in graph_input.type.tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                return _result(
                    "structure",
                    False,
                    error=(
                        f"symbolic dim {dim.dim_param!r} in input "
                        f"{graph_input.name!r}; HSSM rejects dynamic axes at load"
                    ),
                )
            dims.append(dim.dim_value)

    try:
        session = ort.InferenceSession(str(onnx_path))
        inputs, outputs = session.get_inputs(), session.get_outputs()
    except Exception as e:  # noqa: BLE001
        return _result("structure", False, input_dims=dims, error=f"ORT: {e}")

    # One tensor in, one out. Everything downstream reads element [0] of each,
    # so an extra tensor would be checked, compared and scored against the
    # wrong one instead of being reported. All 18 published networks are 1/1.
    if len(inputs) != 1 or len(outputs) != 1:
        return _result(
            "structure",
            False,
            error=(
                f"expected exactly 1 input and 1 output, got {len(inputs)} "
                f"and {len(outputs)}; the single-trial contract assumes one of each"
            ),
        )

    input_shape, output_shape = inputs[0].shape, outputs[0].shape
    width = int(input_shape[-1])
    if expected_input_dim is not None and width != expected_input_dim:
        return _result(
            "structure",
            False,
            input_shape=list(input_shape),
            error=(
                f"input width {width} != expected {expected_input_dim} "
                "(len(param_space) + rt + response for a LAN)"
            ),
        )

    # One log-density per trial. A wider output is not the likelihood HSSM
    # calls, and both G2 and G4 would quietly score column 0 alone. A symbolic
    # output width is left alone: there is nothing to compare it against.
    out_width = output_shape[-1] if output_shape else None
    if isinstance(out_width, int) and out_width != 1:
        return _result(
            "structure",
            False,
            input_shape=list(input_shape),
            output_shape=list(output_shape),
            error=f"output width {out_width} != 1 (one log-density per trial)",
        )

    return _result(
        "structure",
        True,
        input_shape=list(input_shape),
        output_shape=list(output_shape),
        input_width=width,
        ops=sorted({node.op_type for node in model.graph.node}),
    )


def gate_parity(
    onnx_path: Path,
    state_file: Path | None,
    network_config_file: Path | None,
    input_width: int,
    n_draws: int = 1000,
    atol: float = DEFAULT_PARITY_ATOL,
) -> dict:
    """G2: the exported graph still computes what the trainer trained.

    Skipped rather than failed when the flax state is absent: torch-trained
    networks and downloaded artifacts legitimately have no *.jax sibling.
    """
    if state_file is None or network_config_file is None:
        return _result(
            "parity", True, skipped=True, reason="no *_train_state.jax + config pair"
        )

    import pickle

    import jax.numpy as jnp
    import onnxruntime as ort
    from lanfactory.trainers import JaxMLPFactory

    try:
        with open(network_config_file, "rb") as f:
            network_config = pickle.load(f)
        # train=False: the eval head, which is what the exporter emits.
        net = JaxMLPFactory(network_config=network_config, train=False)
        forward, _ = net.make_forward_partial(
            input_dim=input_width, state=str(state_file), add_jitted=False
        )
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name

        rng = np.random.default_rng(0)
        draws = rng.standard_normal((n_draws, input_width)).astype(np.float32)
        jax_out = np.asarray(forward(jnp.asarray(draws))).reshape(n_draws, -1)

        # One row per session.run, necessarily: the graph's batch dim is the
        # concrete 1 that G1 just enforced, so ORT rejects a stacked feed
        # outright. jax is shape-polymorphic and does the whole batch at once.
        max_err = 0.0
        for i in range(n_draws):
            row = session.run(None, {input_name: draws[i : i + 1]})[0].reshape(-1)
            if row.shape != jax_out[i].shape:
                return _result(
                    "parity",
                    False,
                    error=f"width mismatch: onnx {row.shape} vs jax {jax_out[i].shape}",
                )
            max_err = max(max_err, float(np.max(np.abs(row - jax_out[i]))))
    except Exception as e:  # noqa: BLE001
        return _result("parity", False, error=str(e))

    return _result(
        "parity", max_err < atol, max_abs_error=max_err, atol=atol, n_draws=n_draws
    )


def gate_hssm_load(
    onnx_path: Path, model_name: str, n_trials: int = 100, seed: int = 0
) -> dict:
    """G3: HSSM accepts the network and produces a finite initial logp.

    This is the integration the whole pipeline exists to serve; everything
    upstream can be correct and still fail here.
    """
    try:
        import hssm
        import ssms

        rng = np.random.default_rng(seed)
        model_config = ssms.config.model_config[model_name]
        lower, upper = (
            np.asarray(b, dtype=float) for b in model_config["param_bounds"]
        )
        theta = lower + (upper - lower) * rng.uniform(size=lower.shape)

        sim = ssms.basic_simulators.simulator.simulator(
            model=model_name, theta=theta, n_samples=n_trials, random_state=seed
        )
        import pandas as pd

        data = pd.DataFrame(
            {
                "rt": np.asarray(sim["rts"]).reshape(-1),
                "response": np.asarray(sim["choices"]).reshape(-1),
            }
        )

        model = hssm.HSSM(
            data=data,
            model=model_name,
            loglik=str(onnx_path),
            loglik_kind="approx_differentiable",
        )
        pymc_model = model.pymc_model
        logp = float(pymc_model.compile_logp()(pymc_model.initial_point()))
    except Exception as e:  # noqa: BLE001
        return _result("hssm_load", False, error=f"{type(e).__name__}: {e}")

    return _result("hssm_load", np.isfinite(logp), initial_logp=logp, n_trials=n_trials)


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance between two discrete distributions on a shared grid."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def gate_density(
    onnx_path: Path,
    model_name: str,
    n_param_draws: int = 5,
    n_sim: int = 20_000,
    n_grid: int = 200,
    shrink: float = 0.1,
    mass_range: tuple[float, float] = DEFAULT_MASS_RANGE,
    hellinger_ratio_max: float = DEFAULT_HELLINGER_RATIO_MAX,
    seed: int = 0,
) -> dict:
    """G4: the implied density integrates to ~1 and tracks simulation.

    Parameters are drawn from the training bounds shrunk by ``shrink`` on each
    side: the network is not expected to be accurate at the very edge of the
    space it was trained on, and testing there produces failures that say
    nothing about ordinary use.

    A network that loads, has the right shape, and returns finite numbers can
    still be untrained noise. This is the gate that notices.
    """
    try:
        import onnxruntime as ort
        import ssms

        rng = np.random.default_rng(seed)
        model_config = ssms.config.model_config[model_name]
        lower, upper = (
            np.asarray(b, dtype=float) for b in model_config["param_bounds"]
        )
        span = upper - lower
        lower_in, upper_in = lower + shrink * span, upper - shrink * span
        declared_choices = list(model_config["choices"])

        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name

        per_draw = []
        for draw in range(n_param_draws):
            theta = lower_in + (upper_in - lower_in) * rng.uniform(size=lower.shape)
            sim = ssms.basic_simulators.simulator.simulator(
                model=model_name, theta=theta, n_samples=n_sim, random_state=seed + draw
            )
            rts = np.asarray(sim["rts"]).reshape(-1)
            choices = np.asarray(sim["choices"]).reshape(-1)

            rt_max = float(np.quantile(rts, 0.999))
            edges = np.linspace(0.0, rt_max, n_grid + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            width = float(edges[1] - edges[0])

            # Compare the JOINT density over (choice, rt), not each choice
            # separately. Under a strong drift one option is chosen a few
            # percent of the time, so its histogram is mostly sampling noise;
            # a per-choice maximum would report that noise as a failure of the
            # network. The joint comparison weights each choice by how often
            # it actually happens, which is also what the likelihood is.
            # Iterate the model's DECLARED choices, not the sampled ones. A
            # choice the simulator happened not to produce still has network
            # mass, and skipping it would hide exactly the failure this gate
            # exists to catch: a network putting weight where nothing happens.
            total_mass = 0.0
            empirical_joint = []
            network_joint = []
            for choice in declared_choices:
                share = float(np.mean(choices == choice))
                counts, _ = np.histogram(rts[choices == choice], bins=edges)
                empirical_density = counts.astype(float)
                if counts.sum() > 0:
                    empirical_density = counts / counts.sum() * share / width

                batch = np.column_stack(
                    [
                        np.tile(theta, (n_grid, 1)),
                        centers,
                        np.full(n_grid, choice),
                    ]
                ).astype(np.float32)
                logp = np.array(
                    [
                        session.run(None, {input_name: batch[i : i + 1]})[0].reshape(
                            -1
                        )[0]
                        for i in range(n_grid)
                    ]
                )
                network_density = np.exp(logp)

                total_mass += float(np.sum(network_density) * width)
                empirical_joint.append(empirical_density)
                network_joint.append(network_density)

            # The sampling floor for THIS theta and grid: an independent
            # simulation of the same distribution, binned identically. Without
            # it the Hellinger number below has no scale.
            sim_b = ssms.basic_simulators.simulator.simulator(
                model=model_name,
                theta=theta,
                n_samples=n_sim,
                random_state=seed + 10_000 + draw,
            )
            rts_b = np.asarray(sim_b["rts"]).reshape(-1)
            choices_b = np.asarray(sim_b["choices"]).reshape(-1)
            floor_joint = []
            for choice in declared_choices:
                share_b = float(np.mean(choices_b == choice))
                counts_b, _ = np.histogram(rts_b[choices_b == choice], bins=edges)
                density_b = counts_b.astype(float)
                if counts_b.sum() > 0:
                    density_b = counts_b / counts_b.sum() * share_b / width
                floor_joint.append(density_b)

            observed = hellinger(
                np.concatenate(empirical_joint), np.concatenate(network_joint)
            )
            floor = hellinger(
                np.concatenate(empirical_joint), np.concatenate(floor_joint)
            )
            per_draw.append(
                {
                    "theta": [float(x) for x in theta],
                    "total_mass": total_mass,
                    "hellinger": observed,
                    "sampling_floor": floor,
                    "ratio": observed / floor if floor > 0 else float("inf"),
                }
            )
    except Exception as e:  # noqa: BLE001
        return _result("density", False, error=f"{type(e).__name__}: {e}")

    worst_mass = max(per_draw, key=lambda d: abs(d["total_mass"] - 1.0))["total_mass"]
    worst_ratio = max(d["ratio"] for d in per_draw)
    # Every draw must be in range, not just the one furthest from 1.0: with an
    # asymmetric mass_range the furthest draw can be the only one inside it.
    passed = (
        all(mass_range[0] <= d["total_mass"] <= mass_range[1] for d in per_draw)
        and worst_ratio <= hellinger_ratio_max
    )
    return _result(
        "density",
        passed,
        worst_total_mass=worst_mass,
        worst_hellinger=max(d["hellinger"] for d in per_draw),
        worst_ratio=worst_ratio,
        mass_range=list(mass_range),
        hellinger_ratio_max=hellinger_ratio_max,
        draws=per_draw,
    )


def find_sibling(folder: Path, suffix: str) -> Path | None:
    """The single file in ``folder`` ending in ``suffix``, or None."""
    matches = sorted(p for p in folder.iterdir() if p.name.endswith(suffix))
    return matches[0] if len(matches) == 1 else None


def validate_network(
    onnx_path: Path,
    model_name: str,
    network_type: str = "lan",
    skip_density: bool = False,
    skip_hssm: bool = False,
    hellinger_ratio_max: float = DEFAULT_HELLINGER_RATIO_MAX,
) -> dict:
    """Run every gate and return the report."""
    onnx_path = Path(onnx_path)
    folder = onnx_path.parent

    # A closed set. Defaulting a typo to 0 extra inputs makes G1 fail with
    # "input width 6 != expected 4", which blames the artifact for a bad flag.
    if network_type not in EXTRA_INPUTS_BY_NETWORK_TYPE:
        raise ValueError(
            f"Unknown network_type {network_type!r}; expected one of "
            f"{sorted(EXTRA_INPUTS_BY_NETWORK_TYPE)}."
        )

    expected_input_dim = None
    try:
        import ssms

        n_params = len(ssms.config.model_config[model_name]["params"])
        expected_input_dim = n_params + EXTRA_INPUTS_BY_NETWORK_TYPE[network_type]
    except Exception as e:  # noqa: BLE001 - an unknown model just weakens G1
        logger.warning(f"Could not resolve the parameter space for {model_name}: {e}")

    gates = [gate_structure(onnx_path, expected_input_dim)]
    input_width = gates[0].get("input_width")

    if input_width is None:
        # Without a usable graph the remaining gates cannot say anything.
        gates += [
            _result(g, False, skipped=True, reason="structure gate failed")
            for g in ("parity", "hssm_load", "density")
        ]
    else:
        gates.append(
            gate_parity(
                onnx_path,
                find_sibling(folder, "_train_state.jax"),
                find_sibling(folder, "_network_config.pickle"),
                input_width,
            )
        )
        gates.append(
            _result("hssm_load", True, skipped=True, reason="--skip-hssm")
            if skip_hssm
            else gate_hssm_load(onnx_path, model_name)
        )
        gates.append(
            _result("density", True, skipped=True, reason="--skip-density")
            if skip_density
            else gate_density(
                onnx_path, model_name, hellinger_ratio_max=hellinger_ratio_max
            )
        )

    return {
        "schema_version": 1,
        "onnx": str(onnx_path),
        "model": model_name,
        "network_type": network_type,
        "passed": all(g["passed"] for g in gates),
        "gates": gates,
    }


@app.command()
def main(
    onnx_path: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
        help=(
            "The .onnx artifact to validate. Its folder must be trusted: the "
            "parity gate unpickles the *_network_config.pickle sibling."
        ),
    ),
    model_name: str = typer.Option(..., help="ssm-simulators model name, e.g. ddm."),
    network_type: str = typer.Option("lan", help="lan | cpn | opn | gonogo."),
    report_path: Path | None = typer.Option(
        None, help="Where to write validation_report.json [default: next to the ONNX]."
    ),
    skip_density: bool = typer.Option(False, "--skip-density"),
    skip_hssm: bool = typer.Option(False, "--skip-hssm"),
    hellinger_ratio_max: float = typer.Option(
        DEFAULT_HELLINGER_RATIO_MAX,
        help="Max Hellinger relative to the measured sampling floor.",
    ),
    log_level: str = typer.Option("WARNING"),
):
    """Validate a trained network; exit non-zero if any gate fails."""
    level = getattr(logging, str(log_level).upper(), None)
    if not isinstance(level, int):
        raise typer.BadParameter(f"Unknown log level {log_level!r}.")
    logging.basicConfig(level=level)

    try:
        report = validate_network(
            onnx_path=onnx_path,
            model_name=model_name,
            network_type=network_type,
            skip_density=skip_density,
            skip_hssm=skip_hssm,
            hellinger_ratio_max=hellinger_ratio_max,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    destination = report_path or onnx_path.parent / "validation_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")

    # One JSON line on stdout, matching gen_sbatch's driver contract. Gates are
    # tri-state, not boolean: a skipped parity gate reports passed=True in the
    # report, and a driver that saw only that would claim it was checked.
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "report": str(destination),
                "gates": {
                    g["gate"]: "skipped"
                    if g.get("skipped")
                    else ("passed" if g["passed"] else "failed")
                    for g in report["gates"]
                },
            }
        )
    )
    if not report["passed"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
