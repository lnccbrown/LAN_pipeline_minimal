"""Synthetic datasets of increasing design complexity, for recovery testing.

Recovering the parameters that generated a dataset is the acceptance test that
matters for a likelihood network: users fit models, they do not evaluate
densities. But a recovery failure has two causes that look identical from the
outside — the network is wrong, or the *model is not identifiable under that
design* — and only one of them is our problem.

The ladder here exists to separate them, by crossing two axes:

    L0  one condition, all five parameters free. The naive test, and the one
        most likely to leave `sv` weakly determined.
    L1  four conditions where the drift rate varies and everything else is
        shared — the LAN paper's remedy for weak identifiability.

    x   500 and 2000 total trials, so "not enough data" and "not enough design"
        are distinguishable rather than confounded.

Truths for the shared parameters are held identical across all four cells at a
given dataset index, so a difference between cells is the design, never the
draw. If L0 fails and L1 recovers, the design was the problem. If both fail,
compare against the analytical arm before blaming the network.

Ground truth is drawn from **HSSM's** declared bounds for the ONNX likelihood,
not from the box the networks were trained on. Those disagree: HSSM caps `sv`
at 1.0 while training went to 2.5. Drawing outside HSSM's box would produce
truths the sampler cannot represent, and the resulting "failure" would say
nothing about the network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# HSSM's bounds for ddm_sdv under `approx_differentiable`, copied from
# hssm/modelconfig/ddm_sdv_config.py. Pinned here rather than imported so the
# module stays importable without the inference stack, matching the way
# validate_network keeps its heavy imports function-local.
ONNX_BOUNDS: dict[str, tuple[float, float]] = {
    "v": (-3.0, 3.0),
    "a": (0.3, 2.5),
    "z": (0.1, 0.9),
    "t": (0.0, 2.0),
    "sv": (0.0, 1.0),
}

PARAM_ORDER = ("v", "a", "z", "t", "sv")

# The same 10% inset gate_density uses. A network is not expected to be
# accurate at the very edge of its training region, and a truth drawn there
# tests the boundary rather than the model.
SHRINK = 0.1


@dataclass(frozen=True)
class Design:
    """One rung of the ladder."""

    name: str
    n_trials: int
    n_conditions: int

    @property
    def varies_by_condition(self) -> tuple[str, ...]:
        """Parameters that take a separate value per condition."""
        return ("v",) if self.n_conditions > 1 else ()

    @property
    def trials_per_condition(self) -> int:
        return self.n_trials // self.n_conditions


# 500 and 2000 total trials, crossed with 1 and 4 conditions. The counts are
# deliberately modest: a real experiment is a few hundred trials per subject,
# and a ladder calibrated at 4000 would prove identifiability nobody can buy.
#
# Crossing the two axes rather than only varying structure is what separates
# "not enough data" from "not enough design". Read the 2x2:
#
#                   1 condition        4 conditions
#     500 total     L0_n500            L1_n500   (125/condition)
#     2000 total    L0_n2000           L1_n2000  (500/condition)
#
# Down a column, only the design changes at fixed data. Along the bottom row
# sits the realistic case: 500 trials per condition, which is what you would
# actually run. L1_n500's 125/condition is deliberately below Ratcliff &
# McKoon's ~200/condition floor, so the ladder should visibly fail there — a
# ladder that never fails is not measuring anything.
DESIGNS: dict[str, Design] = {
    d.name: d
    for d in (
        Design("L0_n500", n_trials=500, n_conditions=1),
        Design("L0_n2000", n_trials=2000, n_conditions=1),
        Design("L1_n500", n_trials=500, n_conditions=4),
        Design("L1_n2000", n_trials=2000, n_conditions=4),
    )
}


def shrunk_bounds(shrink: float = SHRINK) -> dict[str, tuple[float, float]]:
    """HSSM's bounds pulled in by `shrink` on each side."""
    out = {}
    for name, (lo, hi) in ONNX_BOUNDS.items():
        span = hi - lo
        out[name] = (lo + shrink * span, hi - shrink * span)
    return out


def draw_truth(design: Design, seed: int) -> dict[str, float | list[float]]:
    """Ground-truth parameters for one dataset.

    Condition-varying parameters come back as a list with one entry per
    condition; everything else is a scalar shared across the whole dataset.
    """
    # Two independent streams, so a parameter's truth does not depend on how
    # many draws some *other* parameter needed. Sharing one stream would make
    # L1 (four drift values) consume different randomness from L0 (one), and
    # the shared parameters would land on different truths at the same index —
    # so "L1 recovers sv better" could just mean "L1 drew an easier sv". The
    # ladder holds trials constant precisely to avoid that kind of confound,
    # and the truths have to be held constant with them.
    shared_rng = np.random.default_rng(seed)
    condition_rng = np.random.default_rng(seed + 500_000)
    bounds = shrunk_bounds()

    truth: dict[str, float | list[float]] = {}
    for name in PARAM_ORDER:
        lo, hi = bounds[name]
        # Draw from the shared stream for every parameter, in a fixed order,
        # then overwrite the condition-varying ones. That keeps the shared
        # stream's consumption identical across levels.
        shared_value = float(shared_rng.uniform(lo, hi))
        if name in design.varies_by_condition:
            truth[name] = [
                float(x) for x in condition_rng.uniform(lo, hi, design.n_conditions)
            ]
        else:
            truth[name] = shared_value
    return truth


def build_dataset(design: Design, seed: int):
    """Simulate one dataset. Returns (DataFrame, truth).

    The frame carries `rt`, `response`, and — for multi-condition designs — a
    `condition` column. The simulator is handed a full (n_trials, n_params)
    theta with `n_samples=1`, which is the trial-wise path: row order is
    preserved, so the condition labels line up with the rows they generated.
    """
    import pandas as pd
    from ssms.basic_simulators.simulator import simulator

    truth = draw_truth(design, seed)
    condition = np.repeat(
        np.arange(design.n_conditions), design.trials_per_condition
    ).astype(int)
    n_trials = condition.size  # exact, even if n_trials % n_conditions != 0

    columns = []
    for name in PARAM_ORDER:
        value = truth[name]
        if isinstance(value, list):
            columns.append(np.asarray(value, dtype=float)[condition])
        else:
            columns.append(np.full(n_trials, value, dtype=float))
    theta = np.column_stack(columns)

    # `random_state` alone does NOT make ddm_sdv reproducible. Its `sv` is
    # applied through simulator_param_mappings as `norm.rvs(loc=0, scale=sv)`,
    # and scipy's global RandomState is what that draws from — a stream
    # `random_state` never touches. Measured: back-to-back calls with the same
    # random_state differ for ddm_sdv, agree for plain ddm, and agree for
    # ddm_sdv at sv=0. Seeding the global RNG closes the gap.
    #
    # This matters more than reproducibility for its own sake: the analytical
    # and network arms must see byte-identical data, or the comparison that
    # makes recovery interpretable is no longer paired.
    # Upstream: ssm-simulators should thread random_state into the mappings.
    np.random.seed(seed)
    sim = simulator(theta=theta, model="ddm_sdv", n_samples=1, random_state=seed)
    data = pd.DataFrame(
        {
            "rt": np.asarray(sim["rts"]).reshape(-1),
            "response": np.asarray(sim["choices"]).reshape(-1),
        }
    )
    if design.n_conditions > 1:
        # Wrapped in C(...) at formula time; a bare integer column would be
        # read as a linear covariate instead of a factor.
        data["condition"] = condition
    return data, truth


def model_spec(design: Design) -> dict:
    """The `include` / `global_formula` arguments for `hssm.HSSM`.

    Priors are pinned to HSSM's ONNX bounds for *every* arm, including the
    analytical one whose own default bounds are wider (`sv` is unbounded
    above there). Without this the arms would differ in their priors as well
    as their likelihoods, and the comparison that makes recovery interpretable
    would no longer be paired.
    """
    include = []
    for name in PARAM_ORDER:
        lo, hi = ONNX_BOUNDS[name]
        uniform = {"name": "Uniform", "lower": lo, "upper": hi}
        if name in design.varies_by_condition:
            include.append(
                {
                    # 0 + C(...): one coefficient per condition and no
                    # intercept, so each cell's value is read straight off the
                    # posterior instead of being an offset from a reference.
                    "name": name,
                    "formula": f"{name} ~ 0 + C(condition)",
                    "prior": {"C(condition)": uniform},
                    "link": "identity",
                    "bounds": (lo, hi),
                }
            )
        else:
            include.append({"name": name, "prior": uniform, "bounds": (lo, hi)})
    return {"include": include}


def posterior_names(design: Design) -> dict[str, list[str]]:
    """Map each ddm_sdv parameter to the posterior variables holding it.

    A shared parameter is one variable; a condition-varying one is a single
    vector variable whose entries are the per-condition values, which the
    caller indexes.
    """
    names = {}
    for name in PARAM_ORDER:
        if name in design.varies_by_condition:
            names[name] = [f"{name}_C(condition)"]
        else:
            names[name] = [name]
    return names
