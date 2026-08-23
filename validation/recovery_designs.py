"""Parameter recovery as a reusable recipe, and the design ladder it runs on.

This module is not about any one model. It is the machinery behind a recipe we
want to apply to every network we ship, so the recipe comes first and the model
under test is an argument.

The recipe
----------
1. Simulate datasets from known truths, fit them back, and ask two questions
   per parameter — never one:

                       narrow posterior       wide posterior
       covers truth    identifiable           unidentifiable, but HONEST
       misses truth    THE LIKELIHOOD IS WRONG        wrong and vague

   **Coverage tests the likelihood**: a calibrated one puts the truth inside its
   94% interval 94% of the time, however wide that interval is. **Contraction
   (posterior sd / prior sd) tests the design**: it says how much the data
   narrowed the prior. Only the bottom-left cell — confidently wrong — should
   ever block a release. Gating on sharpness would punish a network for being
   honest about what a hard dataset supports.

2. Hold a **reference that contains no network**, so a failure can be
   attributed. In order of preference:

   a. The same model's *analytical* likelihood, fit on byte-identical data with
      identical priors. This is the ceiling; a shortfall it shares is the
      design's identifiability limit, and one only the network shows is the
      network's. Most models in the catalogue have no analytical form, so:
   b. The **ladder** below. A design limit relaxes when you add data or
      structure; a broken likelihood does not. Coverage that is flat across
      every rung is a network signature.

3. Walk a ladder of **increasing design complexity**, not increasing data
   alone. Weak identifiability is usually a property of the design, and the
   standard remedy — several conditions in which **one parameter varies while
   the rest are shared** — is exactly what the ladder encodes.

   *Which* parameter varies is a free choice, and it is the interesting knob,
   not an implementation detail. It changes what the rung can tell you, in two
   ways at once. The varying parameter gets direct experimental leverage: the
   design asks whether it is recoverable when something actually moves it.
   Every *other* parameter is simultaneously pooled across all the conditions,
   so it is constrained by the full dataset instead of one cell — which is how
   a multi-condition design rescues a parameter it never manipulates. So if
   `sv` comes back badly, `L1@v` asks "does pooling fix sv?" and `L1@sv` asks
   "is sv recoverable when we manipulate it?", and those are different
   questions with different answers.

   Drift is only the *default* because it is what experiments usually
   manipulate. Any parameter of the model is a legitimate choice, and running
   several L1 variants against the same L0 is the point of the design, not an
   abuse of it — each variant is its own rung and they are scored separately.

The ladder
----------
Two axes crossed, with total trials held constant down each column so that
"not enough data" and "not enough design" cannot be confused:

                      1 condition        C conditions
        500 total     L0_n500            L1_n500
        2000 total    L0_n2000           L1_n2000

Truths for the shared parameters are identical across all four cells at a given
dataset index, so a difference between cells is the design and never the draw.

Applying it to a new model
--------------------------
    model = load_model("angle")                 # or any HSSM/ssms model name
    data, truth = build_dataset(model, DESIGNS["L1_n500"], seed=10_007)

    theta_varies = load_model("angle", condition_param="theta")   # any parameter

`load_model` reads the parameter list, the bounds and the available likelihood
kinds from HSSM's own model config, so nothing here has to be updated per
model. The one judgement call it cannot read off a config is which parameter
the conditions should vary — hence `condition_param=`, which accepts any
parameter the model has. The default guess is only a convenience.

Ground truth is drawn from the bounds HSSM declares for the likelihood kind
under test. For a network that box is the region it was trained on: outside it
a LAN extrapolates and returns a finite but wrong density, so a "failure" there
would say nothing about the network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# The same 10% inset gate_density uses. A network is not expected to be
# accurate at the very edge of its training region, and a truth drawn there
# tests the boundary rather than the model.
SHRINK = 0.1

# The decision-time budget handed to the simulator. Passed explicitly rather
# than left to ssms' default so the censoring criterion below is ours and
# cannot drift under us. A trial whose decision process never terminated comes
# back at `max_t + t`, so `rt >= SIMULATOR_MAX_T` is exact: no uncensored trial
# can reach the budget without having consumed all of it. (Measured on ddm_sdv
# at a=2.5, t=1.9: RTs saturate at 21.9, and 153 of 4000 sit at or above 20.)
SIMULATOR_MAX_T = 20.0


@dataclass(frozen=True)
class ModelUnderTest:
    """Everything the harness needs to know about one model.

    Built by `load_model` from HSSM's config in normal use; constructed
    directly in tests, which is why it is plain data with no HSSM import
    anywhere in its definition.
    """

    name: str
    # ssms' parameter order. theta columns are positional, so this — not
    # HSSM's `list_params` — is what the simulator must be handed. The two
    # agree today for every model, but only one of them is load-bearing.
    params: tuple[str, ...]
    bounds: dict[str, tuple[float, float]]
    # Which parameter the L1 conditions vary. See `_default_condition_param`.
    condition_param: str
    n_choices: int = 2
    # Likelihood kinds HSSM can build for this model. The harness fits one arm
    # per kind, and `analytical` in here is what makes the paired comparison in
    # step 2a of the recipe available at all.
    likelihood_kinds: tuple[str, ...] = ("approx_differentiable",)
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        missing = set(self.params) - set(self.bounds)
        if missing:
            raise ValueError(f"{self.name}: no bounds for {sorted(missing)}")
        if self.condition_param not in self.params:
            raise ValueError(
                f"{self.name}: condition_param {self.condition_param!r} is not a "
                f"parameter of this model. Have: {list(self.params)}"
            )
        # A non-finite or inverted bound is not a box to draw truths from. Left
        # unchecked, `shrunk_bounds` returns (inf, nan) without complaint and
        # the run dies much later inside numpy, in an error naming neither the
        # model nor the parameter.
        for name in self.params:
            lo, hi = self.bounds[name]
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise ValueError(f"{self.name}/{name}: non-finite bound ({lo}, {hi})")
            if not lo < hi:
                raise ValueError(f"{self.name}/{name}: empty bound ({lo}, {hi})")

    @property
    def has_analytical(self) -> bool:
        """Whether the no-network reference arm of step 2a is available."""
        return "analytical" in self.likelihood_kinds

    def shrunk_bounds(self, shrink: float = SHRINK) -> dict[str, tuple[float, float]]:
        """Bounds pulled in by `shrink` on each side, for drawing truths."""
        out = {}
        for name in self.params:
            lo, hi = self.bounds[name]
            span = hi - lo
            out[name] = (lo + shrink * span, hi - shrink * span)
        return out


def _default_condition_param(params: tuple[str, ...]) -> str:
    """A starting guess at which parameter to vary across conditions.

    A guess, and nothing more. Any parameter of the model is a valid choice and
    each one asks a different question of the design, so this exists only so
    that `load_model("angle")` does something sensible without an argument.
    Drift wins the default because it is what experiments usually manipulate,
    not because the ladder is about drift.
    """
    # ponytail: name-prefix heuristic. Replace with an explicit per-model entry
    # only once a model appears whose drift is not spelled `v...`.
    drifts = [p for p in params if p.startswith("v")]
    return drifts[0] if drifts else params[0]


def load_model(
    name: str,
    *,
    loglik_kind: str = "approx_differentiable",
    condition_param: str | None = None,
) -> ModelUnderTest:
    """Assemble a `ModelUnderTest` from HSSM's and ssms' own configs.

    Bounds come from HSSM, because those are what the sampler will be allowed
    to explore. The parameter *order* comes from ssms, because that is what the
    simulator reads positionally. The two are cross-checked as sets: a model
    where they disagree would otherwise recover garbage for reasons that have
    nothing to do with the likelihood.
    """
    # Imported here, not at module scope: the ladder and its tests must stay
    # usable without the inference stack, which CI does not install.
    from hssm.modelconfig import get_default_model_config
    from ssms.config import model_config as ssms_config

    config = get_default_model_config(name)
    likelihoods = config["likelihoods"]
    if loglik_kind not in likelihoods:
        raise ValueError(
            f"{name} has no {loglik_kind!r} likelihood in HSSM. "
            f"Have: {sorted(likelihoods)}"
        )

    sim_config = ssms_config.get(name)
    if sim_config is None:
        raise ValueError(f"{name} is an HSSM model but ssms cannot simulate it.")
    params = tuple(sim_config["params"])
    if set(params) != set(config["list_params"]):
        raise ValueError(
            f"{name}: HSSM and ssms disagree on the parameter set — "
            f"{sorted(config['list_params'])} vs {sorted(params)}."
        )

    bounds = likelihoods[loglik_kind].get("bounds")
    if not bounds:
        raise ValueError(
            f"{name}/{loglik_kind} declares no bounds, so there is no box to "
            "draw truths from."
        )

    return ModelUnderTest(
        name=name,
        params=params,
        bounds={p: tuple(float(x) for x in bounds[p]) for p in params},
        condition_param=condition_param or _default_condition_param(params),
        n_choices=len(config.get("choices", [-1, 1])),
        likelihood_kinds=tuple(sorted(likelihoods)),
        notes={"loglik_kind": loglik_kind},
    )


@dataclass(frozen=True)
class Design:
    """One rung of the ladder. Deliberately says nothing about the model."""

    name: str
    n_trials: int
    n_conditions: int

    @property
    def trials_per_condition(self) -> int:
        return self.n_trials // self.n_conditions


# The counts are deliberately modest: a real experiment is a few hundred trials
# per subject, and a ladder calibrated at 4000 would prove an identifiability
# nobody can buy. L1_n500's 125/condition is below Ratcliff & McKoon's
# ~200/condition floor on purpose, so the ladder should visibly fail there — a
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


def varies_by_condition(model: ModelUnderTest, design: Design) -> tuple[str, ...]:
    """Parameters that take a separate value per condition."""
    return (model.condition_param,) if design.n_conditions > 1 else ()


def design_id(model: ModelUnderTest, design: Design) -> str:
    """Full identity of one rung: the design AND what varies across it.

    `L1_n500` is not a design — `L1_n500@v` and `L1_n500@sv` are two different
    designs that happen to share a trial count and a condition count. They
    generate different data, answer different questions, and must never be
    pooled. Without this they collide twice over: identical shard filenames, so
    the second run silently overwrites the first, and identical cell keys for
    every shared parameter, so their coverage numbers average together.

    L0 has nothing varying, so its identity is just the design name — otherwise
    passing `--condition-param` would split the L0 cells that the L1 variants
    are all supposed to be compared against.
    """
    if design.n_conditions <= 1:
        return design.name
    return f"{design.name}@{model.condition_param}"


def draw_truth(
    model: ModelUnderTest, design: Design, seed: int
) -> dict[str, float | list[float]]:
    """Ground-truth parameters for one dataset.

    Condition-varying parameters come back as a list with one entry per
    condition; everything else is a scalar shared across the whole dataset.
    """
    # Two independent streams, so a parameter's truth does not depend on how
    # many draws some *other* parameter needed. Sharing one stream would make
    # L1 consume different randomness from L0, and the shared parameters would
    # land on different truths at the same index — so "L1 recovers this better"
    # could just mean "L1 drew an easier one". The ladder holds trials constant
    # precisely to avoid that kind of confound, and the truths have to be held
    # constant with them.
    shared_rng = np.random.default_rng(seed)
    condition_rng = np.random.default_rng(seed + 500_000)
    bounds = model.shrunk_bounds()
    varying = varies_by_condition(model, design)

    truth: dict[str, float | list[float]] = {}
    for name in model.params:
        lo, hi = bounds[name]
        # Draw from the shared stream for every parameter, in a fixed order,
        # then overwrite the condition-varying ones. That keeps the shared
        # stream's consumption identical across levels.
        shared_value = float(shared_rng.uniform(lo, hi))
        if name in varying:
            truth[name] = [
                float(x) for x in condition_rng.uniform(lo, hi, design.n_conditions)
            ]
        else:
            truth[name] = shared_value
    return truth


def build_dataset(model: ModelUnderTest, design: Design, seed: int):
    """Simulate one dataset. Returns (DataFrame, truth).

    The frame carries `rt`, `response`, and — for multi-condition designs — a
    `condition` column. The simulator is handed a full (n_trials, n_params)
    theta with `n_samples=1`, which is the trial-wise path: row order is
    preserved, so the condition labels line up with the rows they generated.
    """
    import pandas as pd
    from ssms.basic_simulators.simulator import simulator

    truth = draw_truth(model, design, seed)
    condition = np.repeat(
        np.arange(design.n_conditions), design.trials_per_condition
    ).astype(int)
    n_trials = condition.size  # exact, even if n_trials % n_conditions != 0

    columns = []
    for name in model.params:
        value = truth[name]
        if isinstance(value, list):
            columns.append(np.asarray(value, dtype=float)[condition])
        else:
            columns.append(np.full(n_trials, value, dtype=float))
    theta = np.column_stack(columns)

    # `random_state` alone does NOT make every ssms model reproducible. Any
    # parameter applied through `simulator_param_mappings` — ddm_sdv's `sv` is
    # `norm.rvs(loc=0, scale=sv)` — draws from scipy's *global* RandomState, a
    # stream `random_state` never touches. Measured: back-to-back calls with the
    # same random_state differ for ddm_sdv, agree for plain ddm, and agree for
    # ddm_sdv at sv=0. Seeding the global RNG closes the gap for every model.
    #
    # This matters more than reproducibility for its own sake: the reference and
    # network arms must see byte-identical data, or the paired comparison that
    # makes recovery interpretable is gone.
    # Upstream: ssm-simulators should thread random_state into the mappings.
    np.random.seed(seed)
    sim = simulator(
        theta=theta,
        model=model.name,
        n_samples=1,
        random_state=seed,
        max_t=SIMULATOR_MAX_T,
    )
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


def model_spec(model: ModelUnderTest, design: Design) -> dict:
    """The `include` argument for `hssm.HSSM`.

    Priors are pinned to the same box for *every* arm, including an analytical
    one whose own default bounds may be wider. Without this the arms would
    differ in their priors as well as their likelihoods, and the comparison
    that makes recovery interpretable would no longer be paired.
    """
    varying = varies_by_condition(model, design)
    include = []
    for name in model.params:
        lo, hi = model.bounds[name]
        uniform = {"name": "Uniform", "lower": lo, "upper": hi}
        if name in varying:
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


def posterior_names(model: ModelUnderTest, design: Design) -> dict[str, list[str]]:
    """Map each parameter to the posterior variables holding it.

    A shared parameter is one variable; a condition-varying one is a single
    vector variable whose entries are the per-condition values, which the
    caller indexes.
    """
    varying = varies_by_condition(model, design)
    return {
        name: [f"{name}_C(condition)" if name in varying else name]
        for name in model.params
    }
