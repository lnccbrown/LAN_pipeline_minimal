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
                    initial logp on simulated data (drawn from the same inset
                    box as G4, redrawn until every rt is positive).
    G4 density    — the network's implied density integrates to ~1 and matches
                    simulation (Hellinger) at several in-bounds parameter
                    draws. This is the only gate that catches a network that
                    loads perfectly and has learned nothing.

G1-G3 are mechanical. G4 is statistical, and it calibrates itself against the
sampling noise of the simulator rather than against a fixed number — see
DEFAULT_HELLINGER_RATIO_MAX.

A LAN gets a fifth, advisory gate after G4:

    G5 mass_survey — the network's total mass (summed over choices, integrated
                    over rt on LANfactory's onset-refined grid) is surveyed at
                    20 000 θ over the FULL training box, and the tails of the
                    |total − 1| distribution are judged. G4 samples five θ from
                    the 10 %-shrunk box, so it cannot see the defective regions
                    a survey of the published ddm LAN found at the box edge
                    (1.9 % of the box beyond 0.05, up to +0.22); it would fail
                    that network with probability 0.06 %. The survey needs
                    ``lanfactory.derive`` and skips itself, with a stated
                    reason, in an environment whose LANfactory lacks it.

Auxiliary networks (cpn / opn / gonogo) are not densities, so G3 and G4 do not
apply to them. They get their own pair after G1 and G2:

    A3 hssm-missing-load — hssm.HSSM accepts the network as
                    ``loglik_missing_data`` next to the base LAN and produces a
                    finite initial logp on simulated data with missing rows,
                    at p_outlier = 0 and at HSSM's default lapse.
    A4 accuracy   — the network's probability matches a Monte-Carlo truth at
                    several in-bounds parameter draws (see aux_truth). This is
                    the gate that notices an untrained or mis-categorised net.

The auxiliary contract every gate here enforces (``--model-name`` is always the
BASE model; the ``_deadline`` variant is derived internally):

    cpn     input [θ in list_params order, extra_fields..., choice], width
            n_params + 1, output log P_s(choice | θ)
    opn     input [θ..., deadline], width n_params + 1,
            output log S_s(deadline | θ) = log P_s(rt > deadline | θ)
    gonogo  input [θ..., deadline], width n_params + 1,
            output log P_s(nogo ∪ omission | θ) — derived only, no HSSM consumer

Outputs are SSM-only marginals with the log-sigmoid baked into the graph, so
every value is ≤ 0; HSSM applies the lapse mixture outside the network.

On the cpn truth: ssms does NOT censor a base model at max_t. A trial that has
not terminated by max_t comes back at rt ≈ max_t + t with its sign-implied
choice, while a derived CPN integrates the first-passage density only up to
max_t (20 s). aux_truth therefore reports both ``truth`` (every trial) and
``truth_rt_lt_max_t`` (trials that finished inside the window); the gate judges
``truth``, and the two only differ for parameter draws where a visible share of
trials outlasts 20 s.

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

# G5 thresholds — PROVISIONAL: freeze after the four modern LANs (ddm_sdv,
# gamma_drift, gamma_drift_angle, angle_extended) are surveyed. Set from the
# one survey that exists, the Hub ddm LAN at 20 000 θ on the onset-refined
# grid: median |total − 1| 0.0037, p99 0.080, min 0.877 / max 1.219, 1.9 % of
# the box beyond 0.05, all of it in two regions at the box edge. That network
# is in production and demonstrably usable, so it must WARN, not FAIL: the
# warn line sits under its p99 and the fail line above it. A network whose
# p99 is past 0.10, or that is off by more than 0.10 on over 2 % of its box,
# is mis-normalised somewhere a user will sample. Exposed as CLI flags so the
# freezing run needs no code change; do not loosen them to pass a candidate.
DEFAULT_MASS_SURVEY_P99_MAX = 0.10
DEFAULT_MASS_SURVEY_FRAC_GT_0_10_MAX = 0.02
MASS_SURVEY_WARN_P99 = 0.05
MASS_SURVEY_WARN_FRAC_GT_0_05 = 0.01
MASS_SURVEY_N_THETA = 20_000
# The survey lives in LANfactory's derive package, which the lock pins. An
# environment synced to an older LANfactory skips the gate with this reason
# rather than failing a network it could not survey.
LANFACTORY_DERIVE_SKIP_REASON = (
    "lanfactory.derive not available; this environment's LANfactory predates it"
)

# G3 redraws its parameter draw when the simulated data contain a response time
# HSSM refuses (rt <= 0, which an untruncated non-decision-time kernel such as
# ddm_normal_st's produces whenever st is large next to t). Budget and seed
# stride are recovery_designs.usable_seed's MAX_TRUTH_REDRAWS and REDRAW_STRIDE,
# duplicated rather than imported because publish imports this module as
# ``validation.validate_network`` while recovery_designs is imported by bare
# name. The tests assert the two stay equal.
HSSM_LOAD_MAX_REDRAWS = 8
HSSM_LOAD_REDRAW_STRIDE = 7_000_003

# Inputs beyond the parameter vector, per network type. The trailing inputs
# come last in the graph, after θ (list_params order) and any extra_fields.
EXTRA_INPUTS_BY_NETWORK_TYPE = {
    "lan": 2,  # rt, response: a density over both
    "cpn": 1,  # choice: log P(choice | θ), one row per choice code
    "opn": 1,  # deadline: log P(rt > deadline | θ), the survival function
    "gonogo": 1,  # deadline: log P(nogo ∪ omission | θ); derived only
}
# What G1 says the width should be made of, per type.
INPUT_LAYOUT_BY_NETWORK_TYPE = {
    "lan": "len(param_space) + rt + response for a LAN",
    "cpn": "len(param_space) + choice for a CPN",
    "opn": "len(param_space) + deadline for an OPN",
    "gonogo": "len(param_space) + deadline for a gonogo network",
}
# The gate names each type reports, in order. The publisher (REQUIRED_GATES)
# and the pinned tests key on a LAN's list; mass_survey was appended to it
# deliberately and is NOT required — a skip there never blocks a publish.
GATES_BY_NETWORK_TYPE = {
    "lan": ("structure", "parity", "hssm_load", "density", "mass_survey"),
    "cpn": ("structure", "parity", "hssm_missing_load", "accuracy"),
    "opn": ("structure", "parity", "hssm_missing_load", "accuracy"),
    "gonogo": ("structure", "parity", "hssm_missing_load", "accuracy"),
}
# What an auxiliary network's OUTPUT is the probability of. This is the one
# vocabulary the whole pipeline shares: LANfactory's derived corpora record it
# as generator_config["source"]["aux_category"], the MLflow training run logs
# it as a param, and the publisher writes it into the provenance and the model
# card. It names the output, not the trailing input: a cpn takes a choice and
# outputs P(choice | θ); an opn takes a deadline and outputs P(rt > deadline).
# Under that contract each type has exactly one possibility, so each defaults
# to it; an explicit value is still checked so a mislabelled net is refused.
AUX_CATEGORIES_BY_NETWORK_TYPE = {
    "cpn": ("choice",),
    "opn": ("omission",),
    "gonogo": ("nogo",),
}
GONOGO_SKIP_REASON = "HSSM has no gonogo consumer"

# HSSM ≥ 0.6.0 feeds ``response`` to a width n_params + 1 CPN; every earlier
# release calls the CPN with the parameters only, so a contract-conformant CPN
# cannot even load there. The cpn A3 gate skips itself below that version and
# activates on its own when the lock moves.
HSSM_CPN_RESPONSE_MIN_VERSION = "0.6.0"
HSSM_CPN_SKIP_REASON = (
    f"HSSM < {HSSM_CPN_RESPONSE_MIN_VERSION} ignores response on missing rows"
)

# ssms marks an omission in ``rts`` only — ``choices`` still carries the
# sign-implied choice — and HSSM's missing_data marker is the same value.
OMISSION_RT = -999.0
# The window a derived auxiliary network integrates over, and the simulator's
# default max_t. See the module docstring for why the two matter together.
AUX_MAX_T = 20.0
# ssms' bounds for the deadline parameter (DEADLINE_PARAM_CONFIG). A deadline
# outside them is outside anything the derived net was built for.
DEADLINE_BOUNDS = (0.001, 10.0)

# A4 thresholds — PROVISIONAL. A derived ddm_sdv CPN measured 0.0012 mean |err|
# against a truth SE of ~0.0016 at n_sim=100_000, so a correct network sits an
# order of magnitude under the mean bound; a constant-0.5 net errs ~0.25 and a
# wrong-category net ~0.5, both far above the max bound. The numbers are to be
# calibrated ONCE on the first ddm_sdv gate run and then frozen, exactly as the
# density gate's DEFAULT_HELLINGER_RATIO_MAX was. Exposed as CLI flags so that
# calibration run needs no code change; do not loosen them to pass a candidate.
DEFAULT_ACCURACY_MEAN_ABS_MAX = 0.01
DEFAULT_ACCURACY_MAX_ABS_MAX = 0.03


def _result(name: str, passed: bool, **details: Any) -> dict:
    return {"gate": name, "passed": bool(passed), **details}


def _draw_theta(
    model_config: dict, rng: np.random.Generator, shrink: float = 0.0
) -> np.ndarray:
    """One uniform parameter draw from the training box shrunk by ``shrink``.

    ``shrink`` is the fraction cut from each side: 0.0 is the full box, 0.1
    leaves the central 80%. Networks are not expected to be accurate at the
    very edge of the space they were trained on.
    """
    lower, upper = (np.asarray(b, dtype=float) for b in model_config["param_bounds"])
    span = upper - lower
    lower_in, upper_in = lower + shrink * span, upper - shrink * span
    return lower_in + (upper_in - lower_in) * rng.uniform(size=lower.shape)


def _param_bounds_dict(model_config: dict) -> dict[str, tuple[float, float]]:
    """The training box keyed by parameter name: ``{name: (lo, hi)}``.

    ssms keeps ``param_bounds`` as two positional lists (``[lows, highs]``);
    HSSM's ``model_config["bounds"]`` and ``lanfactory.derive.survey``'s
    ``param_bounds`` both want it by name. Built from the lists rather than
    read from ssms' ``param_bounds_dict`` so a hand-written config (tests,
    a model absent from ssms) needs only the positional form.
    """
    lows, highs = model_config["param_bounds"]
    return {
        p: (float(lo), float(hi))
        for p, lo, hi in zip(model_config["params"], lows, highs)
    }


def _inside_shrunk_box(theta: np.ndarray, model_config: dict, shrink: float) -> bool:
    """Whether ``theta`` lies in the training box shrunk by ``shrink`` a side."""
    lower, upper = (np.asarray(b, dtype=float) for b in model_config["param_bounds"])
    span = upper - lower
    return bool(
        np.all(theta >= lower + shrink * span)
        and np.all(theta <= upper - shrink * span)
    )


def _draw_theta_edge(
    model_config: dict,
    rng: np.random.Generator,
    shrink: float,
    max_tries: int = 10_000,
) -> np.ndarray:
    """One uniform draw from the full box that lies OUTSIDE the shrunk box.

    Rejection-sampled: a full-box draw is kept only when at least one of its
    coordinates falls in the ``shrink`` margin. For ddm's four parameters
    that is 59 % of the box, and it only grows with the parameter count, so
    the loop ends in a handful of tries; ``max_tries`` is a guard against a
    ``shrink`` so small that the margin has effectively no volume.
    """
    if shrink <= 0.0:
        raise ValueError("an edge stratum needs shrink > 0: nothing lies outside")
    for _ in range(max_tries):
        theta = _draw_theta(model_config, rng)
        if not _inside_shrunk_box(theta, model_config, shrink):
            return theta
    raise RuntimeError(f"no edge draw in {max_tries} tries at shrink={shrink}")


def _lan_total_mass(lan_onnx: Path | None, model_config: dict):
    """A ``theta -> total mass`` function for the base LAN, or None.

    None when there is no LAN path to evaluate or ``lanfactory.derive`` is
    not importable (the environment's LANfactory predates it). The mass is
    ``choice_mass`` on LANfactory's onset-refined grid, built around the
    onset ``theta[:, t_idx]`` when the model has a ``t`` parameter (less
    ``scale * theta[:, param]`` when the config declares an
    ``ndt_edge_shift``, as the survey does) and around 0 otherwise, so a
    failing accuracy draw can be attributed: a cpn/opn derived from a LAN that is
    mis-normalised at that θ inherits the error, and the number says so.
    Advisory only — any failure here is logged and yields None, never a
    failed gate.
    """
    if lan_onnx is None:
        return None
    try:
        from lanfactory.derive import choice_mass, load_onnx_predictor
    except ImportError:
        return None
    try:
        predictor = load_onnx_predictor(lan_onnx)
        params = list(model_config["params"])
        t_idx = params.index("t") if "t" in params else None
        shift = model_config.get("ndt_edge_shift")
        shift_idx = params.index(shift["param"]) if shift is not None else None
        choices = list(model_config["choices"])
    except Exception as e:  # noqa: BLE001 - attribution must not break the gate
        logger.warning(f"LAN total_mass unavailable for the accuracy draws: {e}")
        return None

    def total(theta: np.ndarray) -> float | None:
        row = np.atleast_2d(np.asarray(theta, dtype=float))
        onset = row[:, t_idx] if t_idx is not None else 0.0
        if shift_idx is not None:
            onset = onset - shift["scale"] * row[:, shift_idx]
        try:
            return float(choice_mass(predictor, row, choices, onset).total[0])
        except Exception as e:  # noqa: BLE001
            logger.warning(f"LAN total_mass failed at theta={theta.tolist()}: {e}")
            return None

    return total


def _hssm_model_kwargs(model_name: str, model_config: dict) -> dict:
    """Extra ``hssm.HSSM`` kwargs for a model absent from HSSM's registry.

    list_params must be ssms' positional order — for custom models it defines
    the likelihood input order, which is the order the network was trained on.
    """
    from hssm.modelconfig import list_models

    if model_name in list_models():
        return {}
    return {
        "model_config": {
            "list_params": list(model_config["params"]),
            "bounds": _param_bounds_dict(model_config),
            "backend": "jax",
        }
    }


def _simulate(
    model_name: str,
    theta: np.ndarray,
    n_samples: int,
    random_state: int,
    max_t: float = AUX_MAX_T,
) -> tuple[np.ndarray, np.ndarray]:
    """Flat (rts, choices) from one ssms run of ``model_name`` at ``theta``."""
    import ssms

    sim = ssms.basic_simulators.simulator.simulator(
        model=model_name,
        theta=theta,
        n_samples=n_samples,
        random_state=random_state,
        max_t=max_t,
    )
    return (
        np.asarray(sim["rts"]).reshape(-1),
        np.asarray(sim["choices"]).reshape(-1),
    )


def gate_structure(
    onnx_path: Path,
    expected_input_dim: int | None,
    width_layout: str = INPUT_LAYOUT_BY_NETWORK_TYPE["lan"],
) -> dict:
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
                f"input width {width} != expected {expected_input_dim} ({width_layout})"
            ),
        )

    # One log-density per trial, measured rather than read off the graph.
    # HSSM never inspects the declared output shape — onnx2jax validates
    # graph.input dims and resolves outputs by name — so an exporter that left
    # the output symbolic says nothing about the artifact. Feeding one row and
    # counting what comes back is definitive, and the session already exists.
    try:
        probe = np.zeros([int(d) for d in input_shape], dtype=np.float32)
        n_out = int(np.asarray(session.run(None, {inputs[0].name: probe})[0]).size)
    except Exception as e:  # noqa: BLE001 - a graph that cannot run is a failure
        return _result(
            "structure",
            False,
            input_shape=list(input_shape),
            error=f"inference on a single trial failed: {e}",
        )
    if n_out != 1:
        return _result(
            "structure",
            False,
            input_shape=list(input_shape),
            output_shape=list(output_shape),
            error=(
                f"{n_out} values returned for one trial, expected 1 "
                "(a single log-density)"
            ),
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


def hssm_load_data(
    model_name: str, n_trials: int = 100, seed: int = 0, shrink: float = 0.1
) -> dict:
    """Simulate G3's dataset from the first parameter draw HSSM can accept.

    theta comes from the training bounds shrunk by ``shrink`` on each side, the
    same inset box G4 draws from: the edge of the box is where the network is
    least accurate and, for models whose st may exceed t, where the simulator
    returns response times at or below zero. HSSM rejects such data outright,
    so a draw there fails G3 for a reason that says nothing about the network.

    A draw whose data still contain an rt <= 0 (or a non-finite rt) is
    redrawn, the way recovery_designs.usable_seed redraws a degenerate truth:
    attempt 0 uses ``seed`` unchanged and each later attempt steps it by
    HSSM_LOAD_REDRAW_STRIDE, so the dataset is a pure function of
    (model, seed). Trials are never dropped; every rejected draw is returned
    so the report shows it.

    Returns ``{"data": DataFrame | None, "theta", "seed", "attempts",
    "rejected"}``; ``data`` is None when the budget ran out.
    """
    import pandas as pd
    import ssms

    model_config = ssms.config.model_config[model_name]
    rejected = []
    for attempt in range(HSSM_LOAD_MAX_REDRAWS + 1):
        resolved = seed + attempt * HSSM_LOAD_REDRAW_STRIDE
        theta = _draw_theta(model_config, np.random.default_rng(resolved), shrink)
        rts, choices = _simulate(model_name, theta, n_trials, resolved)
        rts = rts.astype(float)
        n_bad = int(np.sum(~(np.isfinite(rts) & (rts > 0))))
        if n_bad == 0:
            data = pd.DataFrame({"rt": rts, "response": choices})
            return {
                "data": data,
                "theta": [float(x) for x in theta],
                "seed": resolved,
                "attempts": attempt + 1,
                "rejected": rejected,
            }
        rejected.append(
            {
                "seed": resolved,
                "theta": [float(x) for x in theta],
                "n_nonpositive_rt": n_bad,
            }
        )
    return {
        "data": None,
        "theta": None,
        "seed": None,
        "attempts": HSSM_LOAD_MAX_REDRAWS + 1,
        "rejected": rejected,
    }


def gate_hssm_load(
    onnx_path: Path,
    model_name: str,
    n_trials: int = 100,
    seed: int = 0,
    shrink: float = 0.1,
) -> dict:
    """G3: HSSM accepts the network and produces a finite initial logp.

    This is the integration the whole pipeline exists to serve; everything
    upstream can be correct and still fail here. The data come from
    `hssm_load_data`: an inset-box draw whose response times are all positive,
    redrawn when they are not, with every redraw recorded in the result.
    """
    draw = {"attempts": 0, "rejected": []}
    try:
        import hssm
        import ssms

        model_config = ssms.config.model_config[model_name]
        draw = hssm_load_data(model_name, n_trials=n_trials, seed=seed, shrink=shrink)
        if draw["data"] is None:
            # Skipped, not failed: no network was loaded, so there is no
            # verdict on it. Like a skipped parity gate this reports
            # passed=True, and like any skipped gate it still blocks publish,
            # which requires hssm_load to have run.
            return _result(
                "hssm_load",
                True,
                skipped=True,
                n_trials=n_trials,
                attempts=draw["attempts"],
                rejected_draws=draw["rejected"],
                reason=(
                    f"no usable dataset in {draw['attempts']} draws from the "
                    f"{shrink:.0%}-inset box: every draw simulated an rt <= 0 or "
                    "a non-finite rt, which HSSM rejects. This is a fact about "
                    "the box, not the network; the network was never loaded."
                ),
            )
        data = draw["data"]

        model = hssm.HSSM(
            data=data,
            model=model_name,
            loglik=str(onnx_path),
            loglik_kind="approx_differentiable",
            **_hssm_model_kwargs(model_name, model_config),
        )
        pymc_model = model.pymc_model
        logp = float(pymc_model.compile_logp()(pymc_model.initial_point()))
    except Exception as e:  # noqa: BLE001
        # theta/seed are None when the failure came before a usable draw.
        return _result(
            "hssm_load",
            False,
            n_trials=n_trials,
            theta=draw.get("theta"),
            seed=draw.get("seed"),
            attempts=draw["attempts"],
            rejected_draws=draw["rejected"],
            error=f"{type(e).__name__}: {e}",
        )

    return _result(
        "hssm_load",
        np.isfinite(logp),
        initial_logp=logp,
        n_trials=n_trials,
        theta=draw["theta"],
        seed=draw["seed"],
        attempts=draw["attempts"],
        rejected_draws=draw["rejected"],
    )


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
        declared_choices = list(model_config["choices"])

        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name

        per_draw = []
        for draw in range(n_param_draws):
            theta = _draw_theta(model_config, rng, shrink)
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


def gate_mass_survey(
    onnx_path: Path,
    model_config: dict | None,
    *,
    n_theta: int = MASS_SURVEY_N_THETA,
    seed: int = 0,
    p99_max: float = DEFAULT_MASS_SURVEY_P99_MAX,
    frac_gt_0_10_max: float = DEFAULT_MASS_SURVEY_FRAC_GT_0_10_MAX,
) -> dict:
    """G5: the total mass is ~1 over the WHOLE box, not just where G4 looked.

    Runs LANfactory's ``survey`` — ``n_theta`` uniform draws over the full
    training box, total mass on the onset-refined grid (a uniform 1000-point
    grid carries 15-25 % quadrature error at a < 0.5, which would be blamed
    on the network) — and judges the tails of |total − 1|:

        FAIL  total.mean not finite  (a NaN or infinite total anywhere: a NaN
              share under 1 % would otherwise pass both tail checks)
        FAIL  total.p99_abs_dev > p99_max  or  total.frac_gt_0.10 > frac_gt_0_10_max
        WARN  total.p99_abs_dev > MASS_SURVEY_WARN_P99
              or total.frac_gt_0.05 > MASS_SURVEY_WARN_FRAC_GT_0_05
              (passed=True, with a ``warning`` detail)

    A model whose config declares ``ndt_edge_shift`` (ddm_st, full_ddm*)
    starts its response times at ``t - scale * st``, and the survey builds
    its grid and measures the leak from there.

    Skipped, with LANFACTORY_DERIVE_SKIP_REASON, while ``lanfactory.derive``
    is not importable; the survey dict is small and recorded whole, so a
    reader can see which parameter bins carry the deviation.
    """
    try:
        from lanfactory.derive import load_onnx_predictor, survey
    except ImportError:
        return _result(
            "mass_survey", True, skipped=True, reason=LANFACTORY_DERIVE_SKIP_REASON
        )
    if model_config is None:
        return _result(
            "mass_survey",
            False,
            error="the model's parameter space could not be resolved from ssms",
        )

    try:
        params = list(model_config["params"])
        # survey() wants the box by name, not ssms' positional [lows, highs].
        result = survey(
            load_onnx_predictor(onnx_path),
            _param_bounds_dict(model_config),
            params,
            list(model_config["choices"]),
            n_theta=n_theta,
            seed=seed,
            onset_param="t" if "t" in params else None,
            ndt_edge_shift=model_config.get("ndt_edge_shift"),
        )
        total = result["total"]
        mean = float(total["mean"])
        p99 = float(total["p99_abs_dev"])
        frac_gt_0_10 = float(total["frac_gt_0.10"])
        frac_gt_0_05 = float(total["frac_gt_0.05"])
    except Exception as e:  # noqa: BLE001
        return _result("mass_survey", False, error=f"{type(e).__name__}: {e}")

    details: dict[str, Any] = {
        "p99_abs_dev": p99,
        "frac_gt_0.10": frac_gt_0_10,
        "frac_gt_0.05": frac_gt_0_05,
        "p99_max": p99_max,
        "frac_gt_0_10_max": frac_gt_0_10_max,
        "warn_p99": MASS_SURVEY_WARN_P99,
        "warn_frac_gt_0_05": MASS_SURVEY_WARN_FRAC_GT_0_05,
        "n_theta": result.get("n_theta", n_theta),
        "seconds": result.get("seconds"),
        "survey": result,
    }
    failures = []
    if not np.isfinite(mean):
        failures.append(f"total.mean {mean} is not finite")
    if p99 > p99_max:
        failures.append(f"total.p99_abs_dev {p99:.4f} > {p99_max}")
    if frac_gt_0_10 > frac_gt_0_10_max:
        failures.append(f"total.frac_gt_0.10 {frac_gt_0_10:.4f} > {frac_gt_0_10_max}")
    if failures:
        return _result(
            "mass_survey", False, verdict="fail", error="; ".join(failures), **details
        )

    warnings = []
    if p99 > MASS_SURVEY_WARN_P99:
        warnings.append(f"total.p99_abs_dev {p99:.4f} > {MASS_SURVEY_WARN_P99}")
    if frac_gt_0_05 > MASS_SURVEY_WARN_FRAC_GT_0_05:
        warnings.append(
            f"total.frac_gt_0.05 {frac_gt_0_05:.4f} > {MASS_SURVEY_WARN_FRAC_GT_0_05}"
        )
    if warnings:
        return _result(
            "mass_survey", True, verdict="warn", warning="; ".join(warnings), **details
        )
    return _result("mass_survey", True, verdict="pass", **details)


def _hssm_version_supports_cpn_response() -> tuple[bool, str]:
    """Whether the installed HSSM feeds ``response`` to a CPN; and its version.

    Read from package metadata rather than ``hssm.__version__`` so the check
    costs nothing: it decides whether to import the inference stack at all.
    Raises ``importlib.metadata.PackageNotFoundError`` when hssm is absent.

    numpy's version parser is used because numpy is already a module-level
    import here; ``packaging`` is only in the tree transitively, and this
    repo does not rely on transitive dependencies. Pre-releases compare below
    their final version, so ``0.6.0rc1`` does not count as ``0.6.0``.
    """
    from importlib.metadata import version

    from numpy.lib import NumpyVersion

    installed = version("hssm")
    supported = NumpyVersion(installed) >= NumpyVersion(HSSM_CPN_RESPONSE_MIN_VERSION)
    return bool(supported), installed


def gate_hssm_missing_load(
    onnx_path: Path,
    model_name: str,
    network_type: str,
    lan_onnx: Path | None = None,
    n_trials: int = 100,
    seed: int = 0,
) -> dict:
    """A3: HSSM accepts the network as ``loglik_missing_data`` and gets a
    finite initial logp, with and without the default lapse mixture.

    opn: the base model is simulated first so the deadline can be its median
    RT — about half the trials then omit, which exercises both branches of
    HSSM's assembled likelihood. cpn: every 5th trial has its rt blanked to
    the missing marker while its response is kept, which is the shape of a
    choice-only row.

    The base LAN is resolved by name through HSSM (a download for registry
    models) unless ``lan_onnx`` points at one; models outside HSSM's registry
    need it, since there is nothing to download. Either way the pairing is
    LAN + auxiliary net: ``loglik_kind`` is always ``approx_differentiable``,
    because for ddm-family models HSSM would otherwise default to its
    analytical likelihood and the jax assembly users actually hit would never
    be exercised.
    """
    from importlib.metadata import PackageNotFoundError

    try:
        if network_type == "cpn":
            supported, installed = _hssm_version_supports_cpn_response()
            if not supported:
                return _result(
                    "hssm_missing_load",
                    True,
                    skipped=True,
                    reason=HSSM_CPN_SKIP_REASON,
                    hssm_version=installed,
                )

        import hssm
        import pandas as pd
        import ssms

        rng = np.random.default_rng(seed)
        model_config = ssms.config.model_config[model_name]
        theta = _draw_theta(model_config, rng)
        rts, choices = _simulate(model_name, theta, n_trials, seed)

        hssm_kwargs: dict[str, Any] = {
            "missing_data": True,
            "loglik_missing_data": str(onnx_path),
            "loglik_kind": "approx_differentiable",
            **_hssm_model_kwargs(model_name, model_config),
        }
        if lan_onnx is not None:
            hssm_kwargs["loglik"] = str(lan_onnx)
        elif "model_config" in hssm_kwargs:
            return _result(
                "hssm_missing_load",
                False,
                error=(
                    f"{model_name!r} is not in HSSM's registry, so there is no "
                    "base LAN to download; pass --lan-onnx"
                ),
            )

        if network_type == "opn":
            deadline = float(np.median(rts))
            rts, choices = _simulate(
                f"{model_name}_deadline", np.append(theta, deadline), n_trials, seed
            )
            data = pd.DataFrame({"rt": rts, "response": choices, "deadline": deadline})
            hssm_kwargs["deadline"] = True
        elif network_type == "cpn":
            rts = rts.copy()
            rts[::5] = OMISSION_RT
            data = pd.DataFrame({"rt": rts, "response": choices})
        else:
            raise ValueError(f"no HSSM missing-data consumer for {network_type!r}")

        n_missing = int(np.sum(data["rt"] == OMISSION_RT))
        initial_logp = {}
        # 0.0 exercises the network alone; the default lapse is what a user
        # who never touches p_outlier gets, and the mixture must not turn a
        # finite marginal into NaN.
        for p_outlier in (0.0, 0.05):
            model = hssm.HSSM(
                data=data,
                model=model_name,
                p_outlier=p_outlier,
                **hssm_kwargs,
            )
            pymc_model = model.pymc_model
            initial_logp[str(p_outlier)] = float(
                pymc_model.compile_logp()(pymc_model.initial_point())
            )
    except PackageNotFoundError:
        return _result(
            "hssm_missing_load",
            False,
            error="hssm is not installed; install the validate dependency group",
        )
    except Exception as e:  # noqa: BLE001
        return _result("hssm_missing_load", False, error=f"{type(e).__name__}: {e}")

    return _result(
        "hssm_missing_load",
        all(np.isfinite(v) for v in initial_logp.values()),
        initial_logp_by_p_outlier=initial_logp,
        n_trials=n_trials,
        n_missing=n_missing,
        lan=str(lan_onnx) if lan_onnx is not None else f"{model_name} (by name)",
    )


def aux_truth(
    model_name: str,
    network_type: str,
    theta: np.ndarray,
    *,
    choice: float | None = None,
    deadline: float | None = None,
    n_sim: int = 100_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Monte-Carlo truth for one auxiliary-network input row.

    cpn: P(choice | θ) from the base model at max_t = AUX_MAX_T. Reported
    twice — ``truth`` counts every trial, ``truth_rt_lt_max_t`` only those
    that finished inside the window, because ssms does not censor the base
    model at max_t while a derived CPN integrates only up to it (see the
    module docstring). opn: P(rt > deadline | θ) as the omission rate of the
    ``_deadline`` variant, read from ``rts`` — ssms marks an omission there
    only, never in ``choices``.

    ``truth_mc_se`` is the binomial standard error of the estimate, so a
    caller can tell network error from simulation noise.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    random_state = int(rng.integers(2**31 - 1))
    theta = np.asarray(theta, dtype=float)

    if network_type == "cpn":
        if choice is None:
            raise ValueError("a cpn truth needs a choice")
        rts, choices = _simulate(model_name, theta, n_sim, random_state)
        hit = choices == choice
        p = float(np.mean(hit))
        extra = {"truth_rt_lt_max_t": float(np.mean(hit & (rts < AUX_MAX_T)))}
    elif network_type == "opn":
        if deadline is None:
            raise ValueError("an opn truth needs a deadline")
        rts, _ = _simulate(
            f"{model_name}_deadline", np.append(theta, deadline), n_sim, random_state
        )
        p = float(np.mean(rts == OMISSION_RT))
        extra = {}
    else:
        raise ValueError(f"no truth is defined for {network_type!r}")

    return {
        "truth": p,
        "truth_mc_se": float(np.sqrt(p * (1.0 - p) / n_sim)),
        "n_sim": n_sim,
        **extra,
    }


def gate_accuracy(
    onnx_path: Path,
    model_name: str,
    network_type: str,
    n_param_draws: int = 20,
    n_sim: int = 100_000,
    shrink: float = 0.1,
    mean_abs_max: float = DEFAULT_ACCURACY_MEAN_ABS_MAX,
    max_abs_max: float = DEFAULT_ACCURACY_MAX_ABS_MAX,
    seed: int = 0,
    lan_onnx: Path | None = None,
) -> dict:
    """A4: the network's probability tracks simulation across the box.

    One (1, D) row per draw through onnxruntime, compared to aux_truth. cpn
    cycles the choice code per draw over the model's declared choices so a
    net that only learned the majority option is caught. opn takes its
    deadline from the RT quantile at u ~ U(0.1, 0.9) of a base-model
    simulation (clipped to ssms' deadline bounds): a deadline drawn uniformly
    from those bounds makes most DDM draws trivially ≈ 0 and the comparison
    says nothing.

    The draws are stratified over the FULL box, alternating: even draws come
    from the box shrunk by ``shrink`` a side (the ``core`` stratum, as
    before), odd draws from the full box but outside the shrunk box (the
    ``edge`` stratum), so 20 draws are 10 + 10. A cpn's choice code cycles
    on the within-stratum index, so each stratum sees every declared choice
    — cycling on the draw index would alias choice with stratum for every
    two-choice model and never test P(+1 | θ) in the core. A survey of the published
    ddm LAN found its mass off by 10-20 % in two regions at the very edge of
    the box — regions the shrunk box never samples — and an auxiliary net
    derived from that LAN inherits the error exactly there. Each record says
    which ``stratum`` it came from, and, when ``lan_onnx`` is given and
    ``lanfactory.derive`` is importable, the base LAN's ``total_mass`` at
    that θ, so a failing draw can be attributed to the LAN rather than to the
    derived net; otherwise ``total_mass`` is None. The pass rule is unchanged.

    Every output must be finite and ≤ 0 — a raw-logit export fails here
    before any truth is simulated.
    """
    try:
        import onnxruntime as ort
        import ssms

        rng = np.random.default_rng(seed)
        model_config = ssms.config.model_config[model_name]
        declared_choices = list(model_config["choices"])
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        lan_mass = _lan_total_mass(lan_onnx, model_config)

        per_draw = []
        for draw in range(n_param_draws):
            if draw % 2 == 0 or shrink <= 0.0:
                stratum, theta = "core", _draw_theta(model_config, rng, shrink)
            else:
                stratum, theta = "edge", _draw_theta_edge(model_config, rng, shrink)
            record: dict[str, Any] = {
                "theta": [float(x) for x in theta],
                "stratum": stratum,
                "total_mass": None if lan_mass is None else lan_mass(theta),
            }
            if network_type == "cpn":
                within = draw // 2  # the index inside this draw's stratum
                trailing = float(declared_choices[within % len(declared_choices)])
                record["choice"] = trailing
                truth_kwargs = {"choice": trailing}
            else:
                base_rts, _ = _simulate(model_name, theta, 10_000, seed + draw)
                u = rng.uniform(0.1, 0.9)
                trailing = float(np.clip(np.quantile(base_rts, u), *DEADLINE_BOUNDS))
                record["deadline"] = trailing
                truth_kwargs = {"deadline": trailing}

            row = np.append(theta, trailing).astype(np.float32).reshape(1, -1)
            logp = float(session.run(None, {input_name: row})[0].reshape(-1)[0])
            if not (np.isfinite(logp) and logp <= 0.0):
                return _result(
                    "accuracy",
                    False,
                    draw=draw,
                    **record,
                    error=(
                        f"network output {logp} is not a log-probability; every "
                        "output must be finite and <= 0 (a raw-logit export "
                        "fails here)"
                    ),
                )

            truth = aux_truth(
                model_name, network_type, theta, n_sim=n_sim, rng=rng, **truth_kwargs
            )
            record.update(
                network_logp=logp,
                network_value=float(np.exp(logp)),
                truth=truth["truth"],
                truth_mc_se=truth["truth_mc_se"],
                abs_error=abs(float(np.exp(logp)) - truth["truth"]),
            )
            if "truth_rt_lt_max_t" in truth:
                record["truth_rt_lt_max_t"] = truth["truth_rt_lt_max_t"]
            per_draw.append(record)
    except Exception as e:  # noqa: BLE001
        return _result("accuracy", False, error=f"{type(e).__name__}: {e}")

    mean_abs = float(np.mean([d["abs_error"] for d in per_draw]))
    max_abs = float(np.max([d["abs_error"] for d in per_draw]))
    return _result(
        "accuracy",
        mean_abs <= mean_abs_max and max_abs <= max_abs_max,
        mean_abs_error=mean_abs,
        max_abs_error=max_abs,
        mean_abs_max=mean_abs_max,
        max_abs_max=max_abs_max,
        n_param_draws=n_param_draws,
        n_core_draws=sum(d["stratum"] == "core" for d in per_draw),
        n_edge_draws=sum(d["stratum"] == "edge" for d in per_draw),
        shrink=shrink,
        n_sim=n_sim,
        draws=per_draw,
    )


def find_sibling(folder: Path, suffix: str) -> Path | None:
    """The single file in ``folder`` ending in ``suffix``, or None."""
    matches = sorted(p for p in folder.iterdir() if p.name.endswith(suffix))
    return matches[0] if len(matches) == 1 else None


def resolve_aux_category(network_type: str, aux_category: str | None) -> str | None:
    """What the network's output is the probability of, validated.

    The vocabulary is the one shared with LANfactory's derived corpora and the
    publisher's provenance: ``choice`` for a cpn, ``omission`` for an opn,
    ``nogo`` for a gonogo. None for a LAN. Every auxiliary type has exactly
    one possibility and defaults to it; an explicit value must match, so a
    record that says the net is something else is rejected here rather than
    written into the report.
    """
    allowed = AUX_CATEGORIES_BY_NETWORK_TYPE.get(network_type)
    if allowed is None:
        if aux_category is not None:
            raise ValueError(
                f"--aux-category applies to auxiliary networks only, not {network_type!r}."
            )
        return None
    if aux_category is None:
        return allowed[0]
    if aux_category not in allowed:
        raise ValueError(
            f"Unknown --aux-category {aux_category!r} for {network_type!r}; "
            f"expected one of {list(allowed)}."
        )
    return aux_category


def validate_network(
    onnx_path: Path,
    model_name: str,
    network_type: str = "lan",
    skip_density: bool = False,
    skip_hssm: bool = False,
    hellinger_ratio_max: float = DEFAULT_HELLINGER_RATIO_MAX,
    aux_category: str | None = None,
    lan_onnx: Path | None = None,
    skip_accuracy: bool = False,
    accuracy_mean_abs_max: float = DEFAULT_ACCURACY_MEAN_ABS_MAX,
    accuracy_max_abs_max: float = DEFAULT_ACCURACY_MAX_ABS_MAX,
    skip_mass_survey: bool = False,
    mass_survey_p99_max: float = DEFAULT_MASS_SURVEY_P99_MAX,
    mass_survey_frac_gt_0_10_max: float = DEFAULT_MASS_SURVEY_FRAC_GT_0_10_MAX,
) -> dict:
    """Run every gate for the network type and return the report."""
    onnx_path = Path(onnx_path)
    folder = onnx_path.parent

    # A closed set. Defaulting a typo to 0 extra inputs makes G1 fail with
    # "input width 6 != expected 4", which blames the artifact for a bad flag.
    if network_type not in EXTRA_INPUTS_BY_NETWORK_TYPE:
        raise ValueError(
            f"Unknown network_type {network_type!r}; expected one of "
            f"{sorted(EXTRA_INPUTS_BY_NETWORK_TYPE)}."
        )
    # The deadline variant is derived from the base model wherever it is
    # needed; naming it here would double the deadline in every simulation.
    if model_name.endswith("_deadline"):
        raise ValueError(
            f"--model-name must be the base model, not {model_name!r}: the "
            "_deadline variant is derived internally for opn/gonogo."
        )
    aux_category = resolve_aux_category(network_type, aux_category)

    expected_input_dim = None
    model_config = None
    try:
        import ssms

        model_config = ssms.config.model_config[model_name]
        n_params = len(model_config["params"])
        expected_input_dim = n_params + EXTRA_INPUTS_BY_NETWORK_TYPE[network_type]
    except Exception as e:  # noqa: BLE001 - an unknown model just weakens G1
        logger.warning(f"Could not resolve the parameter space for {model_name}: {e}")

    gate_names = GATES_BY_NETWORK_TYPE[network_type]
    gates = [
        gate_structure(
            onnx_path, expected_input_dim, INPUT_LAYOUT_BY_NETWORK_TYPE[network_type]
        )
    ]
    input_width = gates[0].get("input_width")

    if input_width is None:
        # Without a usable graph the remaining gates cannot say anything.
        gates += [
            _result(g, False, skipped=True, reason="structure gate failed")
            for g in gate_names[1:]
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
        if network_type == "lan":
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
            gates.append(
                _result("mass_survey", True, skipped=True, reason="--skip-mass-survey")
                if skip_mass_survey
                else gate_mass_survey(
                    onnx_path,
                    model_config,
                    p99_max=mass_survey_p99_max,
                    frac_gt_0_10_max=mass_survey_frac_gt_0_10_max,
                )
            )
        elif network_type == "gonogo":
            # Nothing in HSSM consumes a gonogo network, and without a consumer
            # there is no truth to hold it to either. Reported, not silent, so
            # the publisher can see exactly what was not checked.
            gates += [
                _result(g, True, skipped=True, reason=GONOGO_SKIP_REASON)
                for g in ("hssm_missing_load", "accuracy")
            ]
        else:
            gates.append(
                _result("hssm_missing_load", True, skipped=True, reason="--skip-hssm")
                if skip_hssm
                else gate_hssm_missing_load(
                    onnx_path, model_name, network_type, lan_onnx=lan_onnx
                )
            )
            gates.append(
                _result("accuracy", True, skipped=True, reason="--skip-accuracy")
                if skip_accuracy
                else gate_accuracy(
                    onnx_path,
                    model_name,
                    network_type,
                    mean_abs_max=accuracy_mean_abs_max,
                    max_abs_max=accuracy_max_abs_max,
                    lan_onnx=lan_onnx,
                )
            )

    return {
        "schema_version": 1,
        "onnx": str(onnx_path),
        "model": model_name,
        "network_type": network_type,
        "aux_category": aux_category,
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
    model_name: str = typer.Option(
        ...,
        help="ssm-simulators BASE model name, e.g. ddm (never a _deadline variant).",
    ),
    network_type: str = typer.Option("lan", help="lan | cpn | opn | gonogo."),
    aux_category: str | None = typer.Option(
        None,
        help=(
            "What an auxiliary network's output is the probability of: choice "
            "for a cpn, omission for an opn, nogo for a gonogo. Each type "
            "defaults to its value; an explicit one must match it."
        ),
    ),
    lan_onnx: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help=(
            "Base LAN for the hssm_missing_load gate [default: HSSM resolves it "
            "by model name]. Required for models outside HSSM's registry."
        ),
    ),
    report_path: Path | None = typer.Option(
        None, help="Where to write validation_report.json [default: next to the ONNX]."
    ),
    skip_density: bool = typer.Option(False, "--skip-density"),
    skip_hssm: bool = typer.Option(False, "--skip-hssm"),
    skip_accuracy: bool = typer.Option(False, "--skip-accuracy"),
    skip_mass_survey: bool = typer.Option(False, "--skip-mass-survey"),
    hellinger_ratio_max: float = typer.Option(
        DEFAULT_HELLINGER_RATIO_MAX,
        help="Max Hellinger relative to the measured sampling floor.",
    ),
    mass_survey_p99_max: float = typer.Option(
        DEFAULT_MASS_SURVEY_P99_MAX,
        "--mass-survey-p99-max",
        help="Max p99 of |total mass - 1| over the survey's full-box draws (provisional).",
    ),
    mass_survey_frac_gt_0_10_max: float = typer.Option(
        DEFAULT_MASS_SURVEY_FRAC_GT_0_10_MAX,
        "--mass-survey-frac-gt-0.10-max",
        help="Max fraction of the box where |total mass - 1| > 0.10 (provisional).",
    ),
    accuracy_mean_abs_max: float = typer.Option(
        DEFAULT_ACCURACY_MEAN_ABS_MAX,
        help="Max mean |network - truth| over the accuracy gate's draws.",
    ),
    accuracy_max_abs_max: float = typer.Option(
        DEFAULT_ACCURACY_MAX_ABS_MAX,
        help="Max single-draw |network - truth| in the accuracy gate.",
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
            aux_category=aux_category,
            lan_onnx=lan_onnx,
            skip_accuracy=skip_accuracy,
            accuracy_mean_abs_max=accuracy_mean_abs_max,
            accuracy_max_abs_max=accuracy_max_abs_max,
            skip_mass_survey=skip_mass_survey,
            mass_survey_p99_max=mass_survey_p99_max,
            mass_survey_frac_gt_0_10_max=mass_survey_frac_gt_0_10_max,
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
