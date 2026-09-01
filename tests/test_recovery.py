"""Tests for the recovery harness.

Fast and offline throughout. The statistical core — does a fit recover the
truth — is not unit-testable in CI; it is exercised by the pilot and by the
harness self-test described in the plan (analytical L0, where recovery is well
established, so a failure indicts the harness rather than the model). What is
tested here is everything that decides *whether the numbers mean anything*: the
design construction, the convergence filter, and the attribution rules that
charge a failure to the network only when a no-network reference did better.

Two models are used throughout rather than one, because the harness is a recipe
and a recipe that has only ever run on `ddm_sdv` is a `ddm_sdv` script. `angle`
adds a parameter with a negative lower bound; `race_no_bias_angle_4` has four
choices, eight parameters and a drift called `v0`.
"""

import dataclasses
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

import aggregate_recovery as agg
import recover_parameters as rp
import recovery_designs as rd

# Built by hand rather than through `load_model`, which needs HSSM — CI installs
# only the default dependency group. The values mirror what the PINNED HSSM
# declares for each model's `approx_differentiable` likelihood, and duplicated
# literals rot: TestFixtureDrift below is what keeps them honest wherever HSSM
# is importable. `sv` is the live example — HSSM lnccbrown/HSSM#1230 widens it
# to (0.0, 2.5) to match the network's training box, and when that lands and
# the pipeline bumps its pin, the drift test is what will say so.
DDM_SDV = rd.ModelUnderTest(
    name="ddm_sdv",
    params=("v", "a", "z", "t", "sv"),
    bounds={
        "v": (-3.0, 3.0),
        "a": (0.3, 2.5),
        "z": (0.1, 0.9),
        "t": (0.0, 2.0),
        "sv": (0.0, 1.0),
    },
    condition_param="v",
    likelihood_kinds=("analytical", "approx_differentiable", "blackbox"),
)

ANGLE = rd.ModelUnderTest(
    name="angle",
    params=("v", "a", "z", "t", "theta"),
    bounds={
        "v": (-3.0, 3.0),
        "a": (0.3, 3.0),
        "z": (0.1, 0.9),
        "t": (0.001, 2.0),
        "theta": (-0.1, 1.3),
    },
    condition_param="v",
    likelihood_kinds=("approx_differentiable",),  # no analytical arm
)

RACE4 = rd.ModelUnderTest(
    name="race_no_bias_angle_4",
    params=("v0", "v1", "v2", "v3", "a", "z", "t", "theta"),
    bounds={
        "v0": (0.0, 2.5),
        "v1": (0.0, 2.5),
        "v2": (0.0, 2.5),
        "v3": (0.0, 2.5),
        "a": (1.0, 3.0),
        "z": (0.0, 0.9),
        "t": (0.0, 2.0),
        "theta": (-0.1, 1.45),
    },
    condition_param="v0",
    n_choices=4,
    likelihood_kinds=("approx_differentiable",),
)

MODELS = [DDM_SDV, ANGLE, RACE4]


# For the aggregator, a shard's model, arm and parameter names are opaque
# labels -- `aggregate_recovery` says so in its own docstring and
# TestNamesAreOpaqueToTheAggregator asserts it directly. So what these vary is
# the structure it actually PARSES: whether a design carries an `@variant`
# suffix, since `_rung_of` splits on it and `design_id != design` only at
# multi-condition rungs. Both aggregator bugs this file has caught lived on
# exactly that axis. Names vary along the way because it is free, and it keeps
# the opacity evident to the next reader.
@dataclasses.dataclass(frozen=True)
class Shape:
    """The identity fields a synthetic shard carries."""

    id: str
    model: str
    design: str
    label: str

    @property
    def rung(self) -> str:
        return self.design.split("@", 1)[0]

    @property
    def has_variant(self) -> bool:
        return self.rung != self.design


SHAPES = [
    Shape("bare-rung", "ddm_sdv", "L0_n500", "v"),
    Shape("variant-rung", "race_no_bias_angle_4", "L1_n250@v0", "v0"),
    Shape("variant-rung-long", "conflict_stimflex", "L1_n1000@tcoh", "vt"),
]

# What `shard()` uses unless a test class opts into the sweep below.
SHAPE = SHAPES[0]

# For tests whose subject is "two distinct identities must not be pooled". Its
# only requirement is being different from whatever shape is active.
OTHER_MODEL = "a_different_model"


def cell_key(arm, *, model=None, design=None, label=None):
    """The report key for one cell, built the way `aggregate_recovery` builds it.

    Defaults to the active shape, so a test that names a cell says *which* cell
    without pinning the identity -- and goes through `_join_key`, so the tests
    cannot drift from the separator the report actually uses.
    """
    return agg._join_key(
        model or SHAPE.model,
        arm,
        design or SHAPE.design,
        label or SHAPE.label,
    )


def arm_key(arm, *, model=None, design=None):
    """The report key for one (model, arm, design) bucket in `attempted`."""
    return agg._join_key(model or SHAPE.model, arm, design or SHAPE.design)


@pytest.fixture(params=SHAPES, ids=lambda s: s.id)
def shape(request, monkeypatch):
    """Run a test once per identity shape, and point `shard()` at each one.

    A class whose subject is the aggregator's arithmetic rather than any
    particular model opts in with a single line --

        pytestmark = pytest.mark.usefixtures("shape")

    -- and needs no other edit: its `shard()` calls pick up that shape's model,
    design and label automatically, so the tests read as they did while running
    across all three. A test that needs the values names `shape` as an argument;
    one that needs a specific design still passes it explicitly.
    """
    monkeypatch.setattr(sys.modules[__name__], "SHAPE", request.param)
    return request.param


class TestModelUnderTest:
    @pytest.mark.parametrize("model", MODELS)
    def test_shrunk_bounds_inset_both_sides_for_every_parameter(self, model):
        shrunk = model.shrunk_bounds(0.1)
        for name in model.params:
            lo, hi = model.bounds[name]
            s_lo, s_hi = shrunk[name]
            assert lo < s_lo < s_hi < hi
            assert (s_hi - s_lo) == pytest.approx(0.8 * (hi - lo))

    def test_a_negative_lower_bound_shrinks_upward_not_toward_zero(self):
        # angle's theta starts at -0.1. Shrinking must move it toward the
        # interior of the box, which is *up*; a sign-blind implementation that
        # multiplies bounds instead of the span gets this backwards.
        lo, hi = ANGLE.shrunk_bounds(0.1)["theta"]
        assert -0.1 < lo < hi < 1.3

    @pytest.mark.parametrize(
        "bad", [(0.0, float("inf")), (float("nan"), 1.0), (1.0, 1.0), (2.0, 1.0)]
    )
    def test_non_finite_or_empty_bounds_are_refused_at_construction(self, bad):
        # Unguarded, shrunk_bounds returns (inf, nan) without complaint, the
        # prior becomes Uniform(0, inf), and the run dies much later inside
        # numpy in an error naming neither the model nor the parameter.
        with pytest.raises(ValueError, match="bound"):
            rd.ModelUnderTest(
                name="broken",
                params=("v",),
                bounds={"v": bad},
                condition_param="v",
            )

    def test_a_likelihood_missing_one_parameters_bounds_names_it(self):
        # The dict comprehension in load_model would otherwise raise a bare
        # KeyError on one parameter, which reads as a harness bug rather than
        # as the model config being incomplete.
        pytest.importorskip("hssm", reason="validate dependency group not installed")
        import hssm.modelconfig as mc

        real = mc.get_default_model_config

        def without_sv(name):
            config = real(name)
            bounds = dict(config["likelihoods"]["approx_differentiable"]["bounds"])
            bounds.pop("sv")
            config["likelihoods"]["approx_differentiable"]["bounds"] = bounds
            return config

        mc.get_default_model_config = without_sv
        try:
            with pytest.raises(ValueError, match=r"no bounds for \['sv'\]"):
                rd.load_model("ddm_sdv")
        finally:
            mc.get_default_model_config = real

    def test_missing_bounds_are_refused_at_construction(self):
        with pytest.raises(ValueError, match="no bounds"):
            rd.ModelUnderTest(
                name="broken",
                params=("v", "a"),
                bounds={"v": (0.0, 1.0)},
                condition_param="v",
            )

    def test_a_condition_param_the_model_does_not_have_is_refused(self):
        # Silently ignoring it would produce an L1 design that is secretly L0.
        with pytest.raises(ValueError, match="not a parameter"):
            rd.ModelUnderTest(
                name="broken",
                params=("v", "a"),
                bounds={"v": (0.0, 1.0), "a": (0.0, 1.0)},
                condition_param="drift",
            )

    def test_has_analytical_reports_whether_the_paired_arm_exists(self):
        assert DDM_SDV.has_analytical
        assert not ANGLE.has_analytical

    def test_condition_param_defaults_to_the_first_drift_like_parameter(self):
        assert rd._default_condition_param(("v", "a", "z", "t")) == "v"
        assert rd._default_condition_param(("v0", "v1", "a", "z", "t")) == "v0"
        # LBA leads with A and b, so the drift is not first in the list.
        assert rd._default_condition_param(("A", "b", "v0", "v1")) == "v0"
        # Nothing drift-like at all: fall back rather than crash.
        assert rd._default_condition_param(("alpha", "beta")) == "alpha"


class TestDesigns:
    def test_every_design_divides_evenly_into_conditions(self):
        # A ragged final condition would give one cell fewer trials and quietly
        # weaken exactly the comparison the ladder exists to make.
        for design in rd.DESIGNS.values():
            assert design.n_trials % design.n_conditions == 0
            assert design.trials_per_condition * design.n_conditions == design.n_trials

    def test_a_design_that_does_not_divide_is_refused_at_construction(self):
        # 250 trials over 4 conditions is the live example: it would silently
        # become 248 and break the "same total down the column" guarantee.
        with pytest.raises(ValueError, match="do not divide"):
            rd.Design("L1_n250_bad", n_trials=250, n_conditions=4)

    def test_the_default_ladder_tops_out_at_1000_trials(self):
        # 1000 trials is already a long session. Anything above it is a
        # deliberate "is this recoverable at all" question, not routine
        # validation, so it is opt-in.
        assert set(rd.DEFAULT_LADDER) == {
            "L0_n250",
            "L0_n500",
            "L0_n1000",
            "L1_n250",
            "L1_n500",
            "L1_n1000",
        }
        assert max(rd.DESIGNS[n].n_trials for n in rd.DEFAULT_LADDER) == 1000
        # Optional rungs stay addressable -- they are just not swept.
        assert rd.DESIGNS["L1_n2000"].optional
        assert "L1_n2000" not in rd.DEFAULT_LADDER

    def test_simulating_does_not_leave_the_global_rng_seeded(self):
        # The scipy-global-RNG workaround has to be scoped to the call. Left in
        # place it makes the CALLER's later draws deterministic, which couples
        # a sweep looping over models -- or a test session sharing a process.
        np.random.seed(999)
        before = np.random.random()
        np.random.seed(999)
        rd.build_dataset(DDM_SDV, rd.DESIGNS["L0_n250"], seed=3)
        assert np.random.random() == before

    def test_the_bottom_rung_trades_conditions_for_trials_per_condition(self):
        # 250 split four ways is 62 per condition, which is useless. Down a
        # column the total is what is held constant, and it is exact.
        for n in (250, 500, 1000):
            assert (
                rd.DESIGNS[f"L0_n{n}"].n_trials == rd.DESIGNS[f"L1_n{n}"].n_trials == n
            )
        assert rd.DESIGNS["L1_n250"].trials_per_condition == 125
        assert rd.DESIGNS["L1_n500"].trials_per_condition == 125
        assert rd.DESIGNS["L1_n1000"].trials_per_condition == 250

    @pytest.mark.parametrize("model", MODELS)
    def test_only_the_condition_parameter_varies_and_only_at_l1(self, model):
        l0 = rd.draw_truth(model, rd.DESIGNS["L0_n500"], seed=3)
        l1 = rd.draw_truth(model, rd.DESIGNS["L1_n500"], seed=3)
        assert isinstance(l0[model.condition_param], float)
        assert isinstance(l1[model.condition_param], list)
        assert len(l1[model.condition_param]) == 4
        for param in model.params:
            if param != model.condition_param:
                assert isinstance(l1[param], float), param

    @pytest.mark.parametrize("model", MODELS)
    def test_truth_lands_inside_the_shrunk_box(self, model):
        # A truth at the edge of the training region tests the boundary rather
        # than the model, and outside it the network extrapolates.
        for seed in range(20):
            truth = rd.draw_truth(model, rd.DESIGNS["L1_n500"], seed)
            for param, value in truth.items():
                lo, hi = model.shrunk_bounds()[param]
                for v in value if isinstance(value, list) else [value]:
                    assert lo <= v <= hi, f"{model.name}/{param}={v}"

    @pytest.mark.parametrize("model", MODELS)
    def test_shared_truths_are_identical_across_ladder_levels(self, model):
        # The ladder holds the trial count fixed so design structure is the only
        # difference between L0 and L1. If the shared parameters also drifted
        # between levels, "L1 recovers this better" could just mean "L1 drew an
        # easier one", and the whole comparison would be worthless.
        for n in (500, 2000):
            l0 = rd.draw_truth(model, rd.DESIGNS[f"L0_n{n}"], seed=11)
            l1 = rd.draw_truth(model, rd.DESIGNS[f"L1_n{n}"], seed=11)
            for param in model.params:
                if param != model.condition_param:
                    assert l0[param] == l1[param], param

    def test_l0_has_no_condition_column_and_l1_does(self):
        l0, _ = rd.build_dataset(DDM_SDV, rd.DESIGNS["L0_n500"], seed=1)
        l1, _ = rd.build_dataset(DDM_SDV, rd.DESIGNS["L1_n500"], seed=1)
        assert "condition" not in l0.columns
        assert sorted(l1["condition"].unique()) == [0, 1, 2, 3]
        assert len(l0) == len(l1) == 500

    def test_conditions_are_balanced(self):
        data, _ = rd.build_dataset(DDM_SDV, rd.DESIGNS["L1_n2000"], seed=2)
        counts = data["condition"].value_counts().to_numpy()
        assert set(counts.tolist()) == {500}

    @pytest.mark.parametrize("model", [DDM_SDV, ANGLE, RACE4])
    def test_same_seed_gives_identical_data_so_arms_are_paired(self, model):
        # ddm_sdv's sv goes through a scipy global-RNG mapping that
        # `random_state` does not reach, so this is a real regression guard and
        # not a tautology.
        design = rd.DESIGNS["L0_n500"]
        first, truth_a = rd.build_dataset(model, design, seed=7)
        second, truth_b = rd.build_dataset(model, design, seed=7)
        assert truth_a == truth_b
        assert np.array_equal(first["rt"].to_numpy(), second["rt"].to_numpy())
        assert np.array_equal(
            first["response"].to_numpy(), second["response"].to_numpy()
        )

    def test_a_multi_choice_model_simulates_more_than_two_responses(self):
        # The harness must not quietly assume a binary choice anywhere.
        data, _ = rd.build_dataset(RACE4, rd.DESIGNS["L0_n2000"], seed=5)
        assert data["response"].nunique() > 2

    @pytest.mark.parametrize("model", MODELS)
    def test_model_spec_pins_priors_to_the_same_box_for_every_parameter(self, model):
        # Both arms must share priors, or the comparison is not paired: an
        # analytical likelihood's own default bounds are often wider.
        spec = rd.model_spec(model, rd.DESIGNS["L1_n500"])
        by_name = {entry["name"]: entry for entry in spec["include"]}
        assert set(by_name) == set(model.params)
        for name, entry in by_name.items():
            lo, hi = model.bounds[name]
            prior = entry["prior"]
            if name == model.condition_param:
                assert entry["formula"] == f"{name} ~ 0 + C(condition)"
                prior = prior["C(condition)"]
            else:
                assert "formula" not in entry
            assert prior == {"name": "Uniform", "lower": lo, "upper": hi}

    @pytest.mark.parametrize("model", MODELS)
    def test_posterior_names_match_the_formula(self, model):
        p = model.condition_param
        assert rd.posterior_names(model, rd.DESIGNS["L0_n500"])[p] == p
        assert rd.posterior_names(model, rd.DESIGNS["L1_n500"])[p] == (
            f"{p}_C(condition)"
        )


class TestLoadModel:
    """`load_model` is the only part that needs the inference stack."""

    @pytest.fixture(autouse=True)
    def _needs_hssm(self):
        pytest.importorskip("hssm", reason="validate dependency group not installed")

    @pytest.mark.parametrize("name", ["ddm_sdv", "angle", "levy"])
    def test_reads_a_usable_model_off_hssms_own_config(self, name):
        model = rd.load_model(name)
        assert model.name == name
        assert set(model.bounds) == set(model.params)
        assert model.condition_param in model.params

    def test_parameter_order_follows_ssms_because_theta_is_positional(self):
        from ssms.config import model_config

        for name in ("ddm_sdv", "levy", "ornstein", "race_no_bias_angle_4"):
            assert rd.load_model(name).params == tuple(model_config[name]["params"])

    def test_a_likelihood_kind_the_model_lacks_is_refused(self):
        with pytest.raises(ValueError, match="no 'analytical' likelihood"):
            rd.load_model("angle", loglik_kind="analytical")

    def test_analytical_availability_is_read_not_assumed(self):
        assert rd.load_model("ddm_sdv").has_analytical
        assert not rd.load_model("angle").has_analytical


# Models ssms simulates but HSSM ships no config for — the state every model
# is in while its network trains, before the config PR that network justifies.
# A spread of shapes, not one instance: 7-10 params, with and without a
# collapsing bound, one whose drift-like parameters are not what varies in an
# experiment. When HSSM learns one of these names, the registry guard in the
# first test flags it for promotion to the hand-built fixtures above.
UNREGISTERED = [
    "gamma_drift",  # 7 params, DMC-style gamma drift bump
    "gamma_drift_angle",  # 8 params, + collapsing bound
    "shrink_spot_simple",  # 8 params, shrinking-spotlight flanker
    "conflict_stimflex",  # 10 params, target/distractor streams
]


class TestUnregisteredModel:
    """The ssms fallback is a property of the registry gap, not of one model."""

    @pytest.fixture(autouse=True)
    def _needs_hssm(self):
        pytest.importorskip("hssm", reason="validate dependency group not installed")

    @pytest.mark.parametrize("name", UNREGISTERED)
    def test_falls_back_to_ssms_when_hssm_does_not_know_the_model(self, name):
        from hssm.modelconfig import list_models
        from ssms.config import model_config

        # The premise, asserted rather than assumed: when an HSSM release
        # learns this name, this line is what says "promote the model to the
        # registered fixtures" instead of the fallback silently never running.
        assert name not in list_models()

        model = rd.load_model(name)
        assert model.params == tuple(model_config[name]["params"])
        assert model.likelihood_kinds == ("approx_differentiable",)
        assert not model.has_analytical
        assert model.notes["hssm_registered"] is False
        assert model.condition_param in model.params

    @pytest.mark.parametrize("name", UNREGISTERED)
    def test_fallback_bounds_are_the_ssms_simulator_box(self, name):
        from ssms.config import model_config

        model = rd.load_model(name)
        lows, highs = model_config[name]["param_bounds"]
        assert model.bounds == {
            p: (float(lo), float(hi)) for p, lo, hi in zip(model.params, lows, highs)
        }

    @pytest.mark.parametrize("name", UNREGISTERED)
    def test_a_dataset_actually_builds_for_the_fallback_model(self, name):
        # Parameter ORDER is the load-bearing field — theta reaches the
        # simulator positionally — and only simulating proves it round-trips.
        model = rd.load_model(name)
        data, truth = rd.build_dataset(model, rd.DESIGNS["L0_n250"], seed=11_003)
        assert len(data) == 250
        assert set(truth) == set(model.params)
        assert data["rt"].gt(0).all()

    @pytest.mark.parametrize("name", UNREGISTERED)
    def test_only_the_network_arm_exists(self, name):
        with pytest.raises(ValueError, match="only the"):
            rd.load_model(name, loglik_kind="analytical")

    def test_a_name_neither_side_knows_is_refused(self):
        with pytest.raises(ValueError, match="Neither HSSM nor ssms"):
            rd.load_model("no_such_model_anywhere")

    def test_condition_param_override_still_works(self):
        # `c` is gamma_drift's conflict knob — the parameter an experiment
        # actually manipulates — and the L1@c arm depends on this override.
        model = rd.load_model("gamma_drift", condition_param="c")
        assert model.condition_param == "c"

    def test_registered_models_do_not_take_the_fallback(self):
        assert "hssm_registered" not in rd.load_model("ddm_sdv").notes


class TestHssmModelConfig:
    """The dict handed to hssm.HSSM for models it cannot look up itself."""

    def test_registered_models_get_none_so_hssms_own_config_wins(self):
        assert rd.hssm_model_config(DDM_SDV) is None

    def test_unregistered_models_get_the_full_identity(self):
        model = dataclasses.replace(
            ANGLE, notes={"hssm_registered": False, "bounds_source": "ssms"}
        )
        config = rd.hssm_model_config(model)
        assert config["list_params"] == list(model.params)
        assert config["backend"] == "jax"
        assert config["bounds"] == {p: model.bounds[p] for p in model.params}


def varying(model, param):
    """The same model with a different parameter varying across conditions."""
    return dataclasses.replace(model, condition_param=param)


class TestAnyParameterCanVary:
    """The ladder is about design structure, not about drift.

    Drift is only the default. Varying a different parameter asks a different
    question — the varying one gets direct experimental leverage, and every
    other one is pooled across all four conditions — so each choice is its own
    design and every one of them has to work.
    """

    @pytest.mark.parametrize("param", ["v", "a", "z", "t", "sv"])
    def test_every_parameter_of_the_model_is_a_legal_choice(self, param):
        model = varying(DDM_SDV, param)
        design = rd.DESIGNS["L1_n2000"]
        truth = rd.draw_truth(model, design, seed=4)
        assert isinstance(truth[param], list) and len(truth[param]) == 4
        for other in model.params:
            if other != param:
                assert isinstance(truth[other], float), other

    @staticmethod
    def _moment_correlation(model, design, param, seed):
        """|r| between a parameter's per-condition truth and an RT moment.

        Which moment moves is the parameter's own business, and assuming it is
        always the mean would be the drift-centric habit this test exists to
        break -- so take whichever of mean and spread tracks it better.
        """
        data, truth = rd.build_dataset(model, design, seed=seed)
        groups = [data.loc[data["condition"] == c, "rt"] for c in range(4)]
        return max(
            abs(np.corrcoef(truth[param], [f(g) for g in groups])[0, 1])
            for f in (np.mean, np.std)
        )

    @pytest.mark.parametrize("param", ["a", "t"])
    def test_a_non_drift_parameter_really_moves_the_data(self, param):
        # The check that matters: it is not enough for the formula to name the
        # parameter, the simulated conditions have to actually differ.
        #
        # Measured over seeds 1-8 at 500 trials per condition, `a` and `t`
        # track the mean on every seed (min r = 0.995 and 0.982), so the bar is
        # the worst seed, not a lucky one.
        model = varying(DDM_SDV, param)
        design = rd.DESIGNS["L1_n2000"]
        moved = [
            self._moment_correlation(model, design, param, seed) for seed in range(1, 9)
        ]
        assert min(moved) > 0.95, f"{param} left the RT distribution unchanged"

    def test_sv_moves_the_data_but_only_on_average(self):
        # sv gets its own bar because the honest measurement is much weaker:
        # over the same eight seeds it ranges 0.42 to 0.995 (mean 0.73).
        # Inter-trial drift variability is a dispersion effect and a faint one
        # at this sample size, which is a large part of why sv is the hard
        # parameter to recover -- so a single-seed bar here would be testing the
        # draw, not the design. It previously "passed" at 0.9 on one seed only
        # because that seed drew a dataset in which one response never occurred;
        # the degeneracy redraw removed it and exposed the real spread.
        model = varying(DDM_SDV, "sv")
        design = rd.DESIGNS["L1_n2000"]
        moved = [
            self._moment_correlation(model, design, "sv", seed) for seed in range(1, 9)
        ]
        assert sum(moved) / len(moved) > 0.5, "sv left the RT distribution unchanged"

    @pytest.mark.parametrize("param", ["a", "theta"])
    def test_the_formula_and_posterior_name_follow_the_choice(self, param):
        model = varying(ANGLE, param)
        design = rd.DESIGNS["L1_n500"]
        by_name = {e["name"]: e for e in rd.model_spec(model, design)["include"]}
        assert by_name[param]["formula"] == f"{param} ~ 0 + C(condition)"
        assert rd.posterior_names(model, design)[param] == f"{param}_C(condition)"
        # And drift is now an ordinary shared parameter.
        if param != "v":
            assert "formula" not in by_name["v"]
            assert rd.posterior_names(model, design)["v"] == "v"

    def test_shared_truths_still_line_up_with_l0_whichever_one_varies(self):
        # The ladder's core guarantee has to hold for every variant, or a
        # variant cannot be compared against L0 at all.
        for param in DDM_SDV.params:
            model = varying(DDM_SDV, param)
            l0 = rd.draw_truth(model, rd.DESIGNS["L0_n2000"], seed=11)
            l1 = rd.draw_truth(model, rd.DESIGNS["L1_n2000"], seed=11)
            for other in model.params:
                if other != param:
                    assert l0[other] == l1[other], f"{param}/{other}"

    def test_two_l1_variants_are_different_designs(self):
        # They share a trial count and a condition count and nothing else: the
        # data differ and the questions differ, so the identities must differ.
        l1 = rd.DESIGNS["L1_n2000"]
        assert rd.design_id(varying(DDM_SDV, "v"), l1) == "L1_n2000@v"
        assert rd.design_id(varying(DDM_SDV, "sv"), l1) == "L1_n2000@sv"
        by_v, _ = rd.build_dataset(varying(DDM_SDV, "v"), l1, seed=4)
        by_sv, _ = rd.build_dataset(varying(DDM_SDV, "sv"), l1, seed=4)
        assert not np.array_equal(by_v["rt"].to_numpy(), by_sv["rt"].to_numpy())

    def test_l0_identity_ignores_the_choice_so_variants_share_a_baseline(self):
        # Nothing varies at L0, so splitting its cells by a flag that had no
        # effect would leave every L1 variant with no baseline to beat.
        l0 = rd.DESIGNS["L0_n500"]
        assert (
            rd.design_id(varying(DDM_SDV, "v"), l0)
            == rd.design_id(varying(DDM_SDV, "sv"), l0)
            == "L0_n500"
        )


class TestFixtureDrift:
    """The hand-built fixtures duplicate HSSM's numbers, so they can rot."""

    def test_fixtures_still_match_what_hssm_declares(self):
        pytest.importorskip("hssm", reason="validate dependency group not installed")
        for fixture in MODELS:
            live = rd.load_model(fixture.name)
            assert live.params == fixture.params, fixture.name
            assert live.n_choices == fixture.n_choices, fixture.name
            assert set(live.likelihood_kinds) == set(fixture.likelihood_kinds), (
                fixture.name
            )
            for param, bound in fixture.bounds.items():
                assert live.bounds[param] == pytest.approx(bound), (
                    f"{fixture.name}/{param}: fixture says {bound}, "
                    f"HSSM says {live.bounds[param]}"
                )


class TestConditionBroadcast:
    """The one line that makes an L1 dataset actually multi-condition."""

    def test_l1_trials_are_simulated_from_their_own_condition_value(self):
        # If the per-condition broadcast broke, every L1 dataset would be
        # simulated from a single drift value while still carrying a condition
        # column -- silently turning the whole ladder into four copies of L0.
        design = rd.DESIGNS["L1_n2000"]
        model = DDM_SDV
        data, truth = rd.build_dataset(model, design, seed=4)
        drifts = truth[model.condition_param]
        assert len(set(drifts)) == 4

        # Choice proportion has to track each condition's own drift. Tested as
        # a correlation rather than strict monotonicity: two conditions can
        # draw near-identical drifts, and then their order is sampling noise.
        shares = [
            (data.loc[data["condition"] == c, "response"] > 0).mean() for c in range(4)
        ]
        assert np.corrcoef(drifts, shares)[0, 1] > 0.95, list(zip(drifts, shares))
        assert max(shares) - min(shares) > 0.2, list(zip(drifts, shares))

    def test_l0_and_l1_differ_in_data_even_at_the_same_seed_and_size(self):
        l0, _ = rd.build_dataset(DDM_SDV, rd.DESIGNS["L0_n2000"], seed=4)
        l1, _ = rd.build_dataset(DDM_SDV, rd.DESIGNS["L1_n2000"], seed=4)
        assert not np.array_equal(l0["rt"].to_numpy(), l1["rt"].to_numpy())


class TestDegenerateRedraw:
    """Drawing a truth that makes one response unreachable is a wasted fit."""

    def test_the_draw_and_the_exclusion_use_the_same_threshold(self):
        # These live in two modules because aggregate_recovery imports
        # recovery_designs and not the other way round. If they drift apart,
        # build_dataset would happily hand back datasets the aggregator then
        # throws away -- the exact waste the redraw exists to stop.
        assert rd.MIN_CHOICE_SHARE == agg.MIN_CHOICE_SHARE

    @pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name)
    def test_the_share_counts_missing_responses_against_the_model(self, model):
        # A response that never appeared contributes 0, not a missing key. The
        # count that matters is model.n_choices, not the number of distinct
        # values observed -- otherwise a 4-choice dataset in which only two
        # alternatives were ever taken would score as perfectly balanced.
        all_one = np.ones(100, dtype=int)
        assert rd.min_choice_share(all_one, model) == 0.0
        every = np.arange(model.n_choices).repeat(25)
        assert rd.min_choice_share(every, model) == pytest.approx(1 / model.n_choices)

    def test_a_clean_draw_is_untouched_by_the_redraw(self):
        # Attempt 0 uses the seed unchanged, so re-running a sweep moves the
        # degenerate datasets and nothing else.
        design = rd.DESIGNS["L0_n500"]
        resolved, attempts = rd.usable_seed(DDM_SDV, 1)
        assert (resolved, attempts) == (1, 1)
        plain, plain_truth = rd._simulate_once(DDM_SDV, design, 1)
        data, truth = rd.build_dataset(DDM_SDV, design, seed=1)
        assert np.array_equal(data["rt"].to_numpy(), plain["rt"].to_numpy())
        assert truth == plain_truth

    def test_a_degenerate_draw_is_redrawn(self, monkeypatch):
        seen = []
        real = rd._simulate

        def fake(model, theta, seed):
            seen.append(seed)
            rt, response = real(model, theta, seed)
            # Force the first two probes to look one-sided.
            if len(seen) <= 2:
                response = np.ones_like(response)
            return rt, response

        monkeypatch.setattr(rd, "_simulate", fake)
        resolved, attempts = rd.usable_seed(DDM_SDV, 3)
        # Distinct seeds in first-seen order: an attempt runs one probe per
        # shape in PROBE_DESIGNS, so a seed appears once per probe it reaches
        # and the count of calls is not the count of attempts.
        tried = list(dict.fromkeys(seen))
        assert tried == [3, 3 + rd.REDRAW_STRIDE, 3 + 2 * rd.REDRAW_STRIDE]
        assert (resolved, attempts) == (3 + 2 * rd.REDRAW_STRIDE, 3)

    def test_exhausting_the_budget_returns_the_last_draw_instead_of_raising(
        self, monkeypatch
    ):
        # An array task that dies takes its shard with it; one that returns a
        # degenerate dataset gets excluded by the aggregator and stays visible
        # in the report as an excluded fit. The second is the useful failure.
        real = rd._simulate

        def always_degenerate(model, theta, seed):
            rt, response = real(model, theta, seed)
            return rt, np.ones_like(response)

        monkeypatch.setattr(rd, "_simulate", always_degenerate)
        resolved, attempts = rd.usable_seed(DDM_SDV, 3)
        assert attempts == rd.MAX_TRUTH_REDRAWS + 1
        assert resolved == 3 + rd.MAX_TRUTH_REDRAWS * rd.REDRAW_STRIDE
        # Still returns a usable dataset rather than blowing up the task.
        data, truth = rd.build_dataset(DDM_SDV, rd.DESIGNS["L0_n500"], seed=3)
        assert set(truth) == set(DDM_SDV.params)


class TestChoiceCountIsNotAssumedBinary:
    """`n_choices` decides whether the degeneracy guard works at all.

    `min_choice_share` reports 0 only when FEWER response categories were
    observed than the model has. Understate the count and a dataset that never
    produced one of its responses scores as perfectly balanced, so both the
    redraw and the aggregator's exclusion wave it through -- silently, and
    for exactly the multi-alternative models where a missing category is most
    likely.
    """

    # ssms models that carry no `choices` key at all, with what `nchoices`
    # says. The old `len(config.get("choices", [-1, 1]))` declared every one of
    # them binary.
    NO_CHOICES_KEY = {
        "lba_angle_3": 3,
        "lca_3": 3,
        "dev_rlwm_lba_pw_v1": 3,
        "dev_rlwm_lba_race_v2": 3,
        "tradeoff_weibull_no_bias": 4,
    }

    def test_every_ssms_config_resolves_to_the_count_it_declares(self):
        # The general form: all 113 of them, not a fixture list. Asserted on
        # `_n_choices` directly rather than through `load_model`, so it needs no
        # inference stack and therefore runs in CI -- which is where a
        # config-shape assertion is worth the most, since CI is where a new ssms
        # release lands first.
        from ssms.config import model_config

        for name, config in model_config.items():
            assert rd._n_choices(config) == config["nchoices"], name
        assert len(model_config) > 100, "the ssms registry got suspiciously small"

    def test_the_models_with_no_choices_key_are_still_read_correctly(self):
        pytest.importorskip("hssm", reason="validate dependency group not installed")
        from ssms.config import model_config

        for name, expected in self.NO_CHOICES_KEY.items():
            # The premise, so this fails loudly if ssms starts shipping the key
            # rather than passing for a reason that no longer exists.
            assert "choices" not in model_config[name], name
            assert rd.load_model(name).n_choices == expected, name

    def test_load_model_threads_the_count_through_for_every_model(self):
        # The same claim one layer up: `_n_choices` being right is no use if
        # `load_model` does not use it. Needs HSSM, because `load_model` asks its
        # registry which branch a model takes.
        pytest.importorskip("hssm", reason="validate dependency group not installed")
        from ssms.config import model_config

        checked, skipped = 0, 0
        for name, config in model_config.items():
            try:
                live = rd.load_model(name)
            except ValueError:
                # Legitimately unavailable: HSSM knows the model but declares no
                # approx_differentiable likelihood. Not a choice-count claim.
                skipped += 1
                continue
            assert live.n_choices == config["nchoices"], (
                f"{name}: harness says {live.n_choices}, ssms says {config['nchoices']}"
            )
            checked += 1
        # Not vacuous: skipping everything would otherwise be a green run.
        assert checked > 100, f"only {checked} models checked, {skipped} skipped"

    def test_a_config_carrying_neither_key_raises_instead_of_guessing(self):
        with pytest.raises(KeyError):
            rd._n_choices({"params": ["v"]})

    def test_a_missing_response_category_is_degenerate_at_three_choices(self):
        # The consequence, stated directly. Two of three categories present:
        # binary says 0.5 and sails through, three-way says 0 and is rejected.
        two_of_three = np.array([0] * 250 + [1] * 250)
        three = dataclasses.replace(ANGLE, n_choices=3)
        assert rd.min_choice_share(two_of_three, ANGLE) == 0.5
        assert rd.min_choice_share(two_of_three, three) == 0.0
        assert rd.min_choice_share(two_of_three, three) < rd.MIN_CHOICE_SHARE


class TestRedrawKeepsTheLadderPaired:
    """The redraw must not decide differently for L0 than for L1.

    Degeneracy is far more common at L0 than at L1 -- varying a parameter
    across conditions makes a wholly one-sided design much harder to draw --
    so a redraw that looked at the requested design would redraw L0 where it
    left L1 alone. The two levels would then hold different shared truths at
    the same dataset index, and "L1 recovers this better" could just mean "L1
    drew an easier one": the confound `draw_truth`'s two independent streams
    exist to prevent, reintroduced one layer down. It would also be
    directional, since the surviving L0 truths are the more balanced ones.
    """

    def test_the_resolution_does_not_take_a_design_at_all(self):
        # The structural guarantee, asserted structurally: if a design ever
        # becomes an argument, every empirical test below is one refactor away
        # from being reassuring for the wrong reason.
        import inspect

        assert list(inspect.signature(rd.usable_seed).parameters) == ["model", "seed"]

    @pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name)
    def test_every_design_resolves_to_the_same_shared_truth(self, model):
        for seed in range(1, 9):
            resolved, _ = rd.usable_seed(model, seed)
            expected = rd.shared_draw(model, resolved)
            for design in rd.DESIGNS.values():
                truth = rd.draw_truth(model, design, resolved)
                shared = [p for p in model.params if not isinstance(truth[p], list)]
                assert shared, f"{design.name} varies every parameter"
                for name in shared:
                    assert truth[name] == expected[name], (design.name, name, seed)

    @pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name)
    def test_shared_truths_agree_across_levels_at_every_seed(self, model):
        l0, l1 = rd.DESIGNS["L0_n2000"], rd.DESIGNS["L1_n2000"]
        for seed in range(1, 9):
            _, t0 = rd.build_dataset(model, l0, seed)
            _, t1 = rd.build_dataset(model, l1, seed)
            shared = [p for p in model.params if not isinstance(t1[p], list)]
            for name in shared:
                assert t0[name] == t1[name], (name, seed)

    def test_the_probe_covers_the_condition_stream_not_only_the_shared_one(self):
        """The second probe shape, and the measurement that forced it.

        `draw_truth` discards the shared draw of the condition parameter and
        replaces it from a second stream (`seed + 500_000`). A probe built only
        from the shared draw never touches that stream, so it is blind to
        exactly the parameter the L1 rungs vary -- and that parameter is the
        drift, which is what makes a dataset one-sided.

        The two-condition rung is where it showed, because two values landing
        on the same side of zero is common while four are protected by their
        own spread. On the sweep's own twenty seeds, before the second probe:
        L1_n250 degenerated 3/20 for gamma_drift, 2/20 for gamma_drift_angle
        and 1/20 for ddm_sdv, with every other rung clean. This test walks the
        whole ladder rather than the widest rung, which is what let that
        through.
        """
        pytest.importorskip("hssm", reason="validate dependency group not installed")
        model = rd.load_model("gamma_drift")
        # The sweep's real seeds: recover_parameters uses 10_000 + index.
        for seed in range(10_000, 10_010):
            for name, design in rd.DESIGNS.items():
                if design.optional:
                    continue
                data, _ = rd.build_dataset(model, design, seed)
                share = rd.min_choice_share(data["response"].to_numpy(), model)
                assert share >= rd.MIN_CHOICE_SHARE, (name, seed, share)

    def test_the_probes_are_constants_and_span_the_vulnerable_shape(self):
        # Structural, so the fix cannot be quietly undone: the probes must not
        # be the requested design (that is what keeps the pairing), and one of
        # them must be multi-condition (that is what closes the blind spot).
        assert all(d.name.startswith("__probe") for d in rd.PROBE_DESIGNS)
        assert any(d.n_conditions > 1 for d in rd.PROBE_DESIGNS)
        assert min(d.n_conditions for d in rd.PROBE_DESIGNS) == 1

    def test_the_levels_disagree_about_degeneracy_so_this_is_not_vacuous(self):
        # The tripwire. If no seed in this range makes L0 degenerate while L1
        # is fine, the pairing tests above could pass for the trivial reason
        # that nothing ever needed a redraw, and this whole class would be
        # measuring nothing. Measured on gamma_drift: L0 degenerates on 11 of
        # 20 draws against L1's 1.
        pytest.importorskip("hssm", reason="validate dependency group not installed")
        model = rd.load_model("gamma_drift")
        disagreeing = []
        for seed in range(1, 21):
            shares = {}
            for key in ("L0_n2000", "L1_n2000"):
                data, _ = rd._simulate_once(model, rd.DESIGNS[key], seed)
                shares[key] = rd.min_choice_share(data["response"].to_numpy(), model)
            usable = {k: v >= rd.MIN_CHOICE_SHARE for k, v in shares.items()}
            if usable["L0_n2000"] != usable["L1_n2000"]:
                disagreeing.append(seed)
        assert disagreeing, "no seed forces unequal redraw outcomes any more"

        # And on exactly those seeds -- where a design-aware redraw would have
        # split the levels apart -- the shared truths still match.
        for seed in disagreeing:
            _, t0 = rd.build_dataset(model, rd.DESIGNS["L0_n2000"], seed)
            _, t1 = rd.build_dataset(model, rd.DESIGNS["L1_n2000"], seed)
            shared = [p for p in model.params if not isinstance(t1[p], list)]
            for name in shared:
                assert t0[name] == t1[name], (name, seed)


class TestArmNamesStayUsable:
    """An arm reaches two places that constrain it, and both fail silently.

    It is joined into the aggregator's cell identity, where the separator would
    break the round trip, and interpolated into the shard filename, where a
    path separator writes the file somewhere `load_shards` does not look: a `/`
    nests it out of the non-recursive glob's reach, and a `..` eats the
    `recovery_` prefix the glob matches on. Either way the sweep finishes having
    quietly lost those fits -- the failure this module exists to stop.
    """

    @pytest.mark.parametrize(
        "arm",
        [
            "net|a",  # the cell-identity separator
            "net/a",  # nests the shard below the glob
            "net\\a",
            "../escape",  # eats the recovery_ prefix
            "sp ace",
            # Not "": an empty --arm is falsy, so it reads as "unset" and falls
            # back to `_default_arm`, whose output is valid by construction.
        ],
    )
    def test_an_unusable_arm_is_refused_at_parse_time(self, tmp_path, arm):
        # At parse time, because aggregation happens after a whole sweep has
        # run -- which is the most expensive moment to learn about it.
        from typer.testing import CliRunner

        result = CliRunner().invoke(
            rp.app,
            [
                "--model",
                "ddm_sdv",
                "--design",
                "L0_n500",
                "--dataset-index",
                "0",
                "--arm",
                arm,
                "--out-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 2, result.output
        assert not list(tmp_path.glob("*.json"))

    @pytest.mark.parametrize(
        "stem",
        ["net_a", "gamma|drift", "a/b", "../../x", "sp ace", "n\u00dcll", "x@y"],
    )
    def test_a_derived_arm_is_always_one_the_check_would_accept(self, stem):
        # The anti-drift assertion: `_default_arm` sanitises to the alphabet
        # the explicit path validates against, so the two cannot disagree about
        # what a usable arm is. If they did, a perfectly ordinary ONNX filename
        # could produce an arm the CLI refuses.
        derived = rp._default_arm("approx_differentiable", Path(f"/n/{stem}.onnx"))
        assert re.fullmatch(f"[{rp.ARM_CHARS}]+", derived), derived

    def test_a_usable_arm_still_gets_through(self, tmp_path, monkeypatch):
        def explode(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(rp, "run_one", explode)
        from typer.testing import CliRunner

        result = CliRunner().invoke(
            rp.app,
            [
                "--model",
                "ddm_sdv",
                "--design",
                "L0_n500",
                "--dataset-index",
                "0",
                "--arm",
                "approx_differentiable@b50k",
                "--out-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 1, result.output  # the fit failed, not the parse
        written = list(tmp_path.glob("recovery_*.json"))
        assert len(written) == 1, "the shard must land where load_shards looks"


class TestSilenceCannotPass:
    """A fit that leaves no shard leaves no failure either.

    `verdict` is a pure function of the shards it is handed, so an arm whose
    jobs all died is not wrong in the report -- it is absent from it, and a
    driver gating on the exit code would ship a network that was never fitted.
    Two guards, at the two ends: the submitter refuses an identity that would
    make every task die, and the aggregator can be told what it should have
    received.
    """

    def test_a_sweep_missing_shards_wholesale_fails(self, tmp_path):
        for i in range(10):
            (tmp_path / f"recovery_m_a_L0_n500_{i:04d}.json").write_text(
                json.dumps(shard("approx_differentiable", index=i))
            )
        from typer.testing import CliRunner

        base = ["--shard-dir", str(tmp_path), "--out", str(tmp_path / "r.json")]
        # Silent today: ten shards, no complaint about the ten that never ran.
        quiet = CliRunner().invoke(agg.app, base)
        assert "were expected" not in quiet.output
        # Told what to expect, the same ten shards are a failure.
        loud = CliRunner().invoke(agg.app, base + ["--expect-fits", "20"])
        assert loud.exit_code == 1
        assert "10 fits left no shard at all" in loud.output

    def test_the_submitter_refuses_an_identity_that_would_kill_every_task(self):
        # The other end: a mistyped rung fails inside all 20 array tasks, and
        # tasks that die before writing leave nothing for the aggregator to
        # notice. Cheaper to refuse the submission.
        import gen_sbatch
        from typer.testing import CliRunner

        for flag, value in (("--design", "L1_n50"), ("--arm", "net|0")):
            argv = [
                "recover",
                "--model",
                "ddm_sdv",
                "--design",
                "L0_n500",
                "--output-path",
                "/tmp/unused",
                "--script-only",
            ]
            if flag == "--design":
                argv[argv.index("L0_n500")] = value
            else:
                argv += [flag, value]
            result = CliRunner().invoke(gen_sbatch.app, argv)
            assert result.exit_code == 2, (flag, result.output)


class TestNamesAreOpaqueToTheAggregator:
    """`aggregate_recovery` claims to work "for any model". This checks it.

    The claim is that a shard's model, arm and parameter names are labels the
    module carries but never reads. That is worth one direct assertion, because
    it is what licenses every other aggregator test to use whatever names are
    convenient -- and it is a stronger statement than any amount of
    parametrising over model names, which only ever samples the space.
    """

    def test_renaming_everything_changes_nothing(self):
        base = [
            shard("approx_differentiable", design="L1_n500@v", index=i)
            for i in range(20)
        ]
        renamed = []
        for original in base:
            copy = dict(original)
            copy["model"] = "conflict_stimflex"
            copy["design"], copy["design_id"] = "L1_n1000", "L1_n1000@tcoh"
            copy["parameters"] = {"vt": next(iter(original["parameters"].values()))}
            renamed.append(copy)
        assert agg.verdict(agg.summarise(base)) == agg.verdict(agg.summarise(renamed))

    def test_what_is_not_opaque_is_the_structure(self):
        # Two things inside those strings ARE parsed, which is why the shared
        # SHAPES fixture spans them rather than spanning model names.
        assert agg._rung_of("L1_n500@v") == "L1_n500"
        assert agg._rung_of("L0_n500") == "L0_n500"


class TestDeadShardsKeepTheirDesignIdentity:
    """A dead fit must land in the same bucket as its healthy siblings.

    Cells are keyed on `design_id`, not `design`, because `L1_n500@v` and
    `L1_n500@c` are different experiments and pooling them would average two
    ladders into one coverage number. The worker's failure path used to record
    only `design`, so a dead shard at any multi-condition rung opened a bucket
    of its own with no cells behind it -- and `verdict` fails an arm that was
    attempted but yielded nothing to judge. Two OOM-killed fits out of twenty
    were enough to fail a sweep whose other eighteen were fine, and only at a
    variant rung: the identical sweep at a bare rung passed, because there
    `design_id == design`.
    """

    ARM = "approx_differentiable"

    @classmethod
    def _dead(cls, model, design, index, *, with_design_id=False, with_model=True):
        """Exactly what the worker's `except` branch writes."""
        record = {
            "schema_version": 2,
            "design": design.split("@", 1)[0],
            "dataset_index": index,
            "likelihood": cls.ARM,
            "arm": cls.ARM,
            "error": "RuntimeError: boom",
        }
        if with_model:
            record["model"] = model
        if with_design_id:
            record["design_id"] = design
        return record

    @classmethod
    def _healthy(cls, model, design, label, n):
        return [
            shard(cls.ARM, model=model, design=design, label=label, index=i)
            for i in range(n)
        ]

    @pytest.mark.parametrize("with_design_id", [True, False])
    def test_a_few_dead_fits_do_not_fail_a_healthy_sweep(self, shape, with_design_id):
        model, design, label = shape.model, shape.design, shape.label
        # Both shard shapes: what the worker writes now, and what the sweeps
        # already on disk contain.
        shards = self._healthy(model, design, label, 18)
        shards += [
            self._dead(model, design, i, with_design_id=with_design_id)
            for i in (18, 19)
        ]
        summary = agg.summarise(shards)
        assert list(summary["attempted"]) == [arm_key(self.ARM)], (
            "the dead shards opened a bucket of their own"
        )
        passed, failures = agg.verdict(summary)
        assert passed, failures

    def test_the_error_count_reaches_the_cell_it_belongs_to(self, shape):
        model, design, label = shape.model, shape.design, shape.label
        # `_errors_for` matches on the design token, so an error filed under
        # the bare rung is invisible to the cell that actually lost the fits.
        shards = self._healthy(model, design, label, 18)
        shards += [self._dead(model, design, i) for i in (18, 19)]
        summary = agg.summarise(shards)
        assert agg._errors_for(summary, model, self.ARM, design) == 2

    def test_a_shard_written_before_the_model_field_existed_still_adopts(self, shape):
        model, design, label = shape.model, shape.design, shape.label
        # `_key` defaults a missing `model`, because shards written before the
        # field existed all came from one network. Adoption has to default it
        # the same way: when only `_key` did, the legacy shard keyed as the
        # default while its adoption entry was filed under None, the lookup
        # missed, and the split bucket came straight back. So the healthy
        # siblings here carry the default name -- that is the only case in
        # which a legacy shard can be attributed at all.
        legacy_name = agg._model_of({})
        shards = self._healthy(legacy_name, design, label, 18)
        shards.append(self._dead(model, design, 18, with_model=False))
        summary = agg.summarise(shards)
        assert list(summary["attempted"]) == [arm_key(self.ARM, model=legacy_name)]
        passed, failures = agg.verdict(summary)
        assert passed, failures

    def test_the_identity_helpers_agree_with_the_cell_key(self):
        # The structural form of the same claim: whatever `_key` uses to
        # identify a shard is what adoption must file it under.
        bare = {"design": "L1_n500", "likelihood": self.ARM}
        model, arm, _ = agg._key(bare)
        assert (model, arm) == (agg._model_of(bare), agg._arm_of(bare))

    def test_a_sweep_where_everything_died_still_fails(self, shape):
        model, design = shape.model, shape.design
        # The guard against over-correcting: with no healthy sibling there is
        # nothing to adopt from, and nothing to judge either.
        shards = [self._dead(model, design, i) for i in range(20)]
        passed, failures = agg.verdict(agg.summarise(shards))
        assert not passed
        assert "none usable" in failures[0]

    def test_two_variants_of_one_rung_are_left_alone(self, shape):
        model, design, label = shape.model, shape.design, shape.label
        # Two variants of one rung in the same sweep: nothing in a bare shard
        # says which of them died, and filing the error against the wrong
        # experiment is worse than leaving it unattributed. A bare rung has no
        # variants, so it is the one shape where this cannot arise.
        if not shape.has_variant:
            pytest.skip("a bare rung has no variants to be ambiguous between")
        rung = shape.rung
        shards = [
            *self._healthy(model, design, label, 18),
            *self._healthy(model, f"{rung}@other", label, 18),
        ]
        shards.append(self._dead(model, rung, 18))
        summary = agg.summarise(shards)
        assert arm_key(self.ARM, design=rung) in summary["attempted"]

    def test_adoption_does_not_cross_arms_or_models(self):
        adopted = agg.adopted_design_ids(
            [
                shard("approx_differentiable", design="L1_n500@v", index=0),
                shard("analytical", design="L1_n500@sv", index=0),
            ]
        )
        assert adopted[("ddm_sdv", "approx_differentiable", "L1_n500")] == "L1_n500@v"
        assert adopted[("ddm_sdv", "analytical", "L1_n500")] == "L1_n500@sv"

    NAMING_CASES = [
        ("L0_n500", None, "L0_n500"),
        ("L1_n500", None, "L1_n500@v"),
        ("L1_n500", "sv", "L1_n500@sv"),
    ]

    @staticmethod
    def _invoke_failing_worker(tmp_path, monkeypatch, design, condition_param):
        """Run the CLI with the fit itself raising. Returns the written shard."""
        from typer.testing import CliRunner

        def explode(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(rp, "run_one", explode)
        argv = [
            "--model",
            "ddm_sdv",
            "--design",
            design,
            "--dataset-index",
            "0",
            "--out-dir",
            str(tmp_path),
        ]
        if condition_param:
            argv += ["--condition-param", condition_param]
        result = CliRunner().invoke(rp.app, argv)
        # 1, not 0: a dead shard is written AND reported as a failure.
        assert result.exit_code == 1, result.output
        written = list(tmp_path.glob("*.json"))
        assert len(written) == 1
        return written[0], json.loads(written[0].read_text())

    @pytest.mark.parametrize(("design", "condition_param", "expected"), NAMING_CASES)
    def test_the_worker_names_the_design_without_an_inference_stack(
        self, tmp_path, monkeypatch, design, condition_param, expected
    ):
        # The root cause, fixed at the source: a shard written by a fit that
        # died still has to say which experiment it belonged to.
        #
        # Stubbed rather than skipped. `load_model` needs HSSM, which CI does
        # not install, but what is under test here is the failure path writing
        # the identity -- not the registry lookup that supplies it. Stubbing
        # keeps this covered in CI; the test below runs the same cases against
        # the real lookup wherever HSSM is available.
        def stub(name, *, condition_param=None, **kwargs):
            return dataclasses.replace(
                DDM_SDV, condition_param=condition_param or DDM_SDV.condition_param
            )

        monkeypatch.setattr(rd, "load_model", stub)
        path, record = self._invoke_failing_worker(
            tmp_path, monkeypatch, design, condition_param
        )
        assert record["error"].startswith("RuntimeError: boom")
        assert record["design_id"] == expected
        # And the filename follows the identity, so two variants of one rung
        # cannot overwrite each other's dead shards.
        assert expected in path.name

    @pytest.mark.parametrize(("design", "condition_param", "expected"), NAMING_CASES)
    def test_the_worker_names_the_design_on_the_failure_path(
        self, tmp_path, monkeypatch, design, condition_param, expected
    ):
        # The same cases against the real registry lookup, so the stub above
        # cannot drift away from what `load_model` actually returns.
        pytest.importorskip("hssm", reason="validate dependency group not installed")
        path, record = self._invoke_failing_worker(
            tmp_path, monkeypatch, design, condition_param
        )
        assert record["error"].startswith("RuntimeError: boom")
        assert record["design_id"] == expected
        assert expected in path.name

    def test_a_failure_to_name_the_design_does_not_replace_the_real_error(
        self, tmp_path, monkeypatch
    ):
        # Whatever went wrong in the fit is the thing worth reporting. Naming
        # the design is best-effort and must never take its place.
        def no_model(*a, **k):
            raise ValueError("cannot load")

        monkeypatch.setattr(rd, "load_model", no_model)
        _, record = self._invoke_failing_worker(tmp_path, monkeypatch, "L1_n500", None)
        assert record["error"].startswith("RuntimeError: boom")
        assert "design_id" not in record


class TestSummarise:
    """_summarise produces every number in the report, so it gets pinned."""

    @pytest.fixture(autouse=True)
    def _needs_arviz(self):
        pytest.importorskip("arviz")

    # A one-parameter stand-in: _summarise reads only the parameter list, the
    # bounds (for the prior sd, 6/sqrt(12) here) and which parameter varies, so
    # the real fixtures would add nothing but noise.
    ONE = rd.ModelUnderTest(
        name="t", params=("v",), bounds={"v": (-3.0, 3.0)}, condition_param="v"
    )

    def summarised(self, draws: dict, design: str, truth: dict):
        """_summarise over a hand-built posterior of (chain, draw[, condition])."""
        xr = pytest.importorskip("xarray")
        posterior = xr.Dataset(
            {
                name: (("chain", "draw") + (("cond",) if a.ndim == 3 else ()), a)
                for name, a in draws.items()
            }
        )
        return rp._summarise(
            {"posterior": posterior}, self.ONE, rd.DESIGNS[design], truth
        )

    def test_a_condition_vector_expands_into_one_record_per_condition(self):
        rng = np.random.default_rng(0)
        drift = rng.normal(loc=[0.5, 1.0, 1.5, 2.0], scale=0.05, size=(2, 500, 4))
        truth = {"v": [0.5, 1.0, 1.5, 2.0]}
        out = self.summarised({"v_C(condition)": drift}, "L1_n500", truth)
        assert sorted(out) == ["v[0]", "v[1]", "v[2]", "v[3]"]
        # Each condition is scored against ITS OWN truth, not the first one.
        for i, true_i in enumerate(truth["v"]):
            assert out[f"v[{i}]"]["truth"] == true_i
            assert out[f"v[{i}]"]["mean"] == pytest.approx(true_i, abs=0.02)
            assert out[f"v[{i}]"]["covered"]

    def test_the_interval_is_a_94_percent_hdi_not_arviz_default_89_eti(self):
        # arviz 1.x defaults to an 89% equal-tailed interval, a different
        # statistic that would silently change every coverage verdict.
        rng = np.random.default_rng(1)
        out = self.summarised(
            {"v": rng.normal(0.0, 1.0, size=(2, 20000))}, "L0_n500", {"v": 0.0}
        )["v"]
        # 94% of a standard normal is +/-1.881; 89% would be +/-1.598.
        assert out["hdi_hi"] - out["hdi_lo"] == pytest.approx(2 * 1.881, abs=0.1)
        assert rp.HDI_PROB == 0.94

    def test_contraction_is_measured_against_the_uniform_prior_sd(self):
        rng = np.random.default_rng(2)
        out = self.summarised(
            {"v": rng.normal(0.0, 0.1, size=(2, 5000))}, "L0_n500", {"v": 0.0}
        )["v"]
        assert out["contraction"] == pytest.approx(0.1 / (6.0 / 12**0.5), rel=0.05)


class TestShardHygiene:
    def test_two_candidate_networks_get_different_arm_labels(self):
        # Same likelihood kind, same model, same design -- but a different
        # network. Without this they write the same filename and the second
        # silently replaces the first, dropping half a sweep.
        from pathlib import Path as _P

        a = rp._default_arm("approx_differentiable", _P("/nets/b50k_cosine.onnx"))
        b = rp._default_arm("approx_differentiable", _P("/nets/b500k_cosine.onnx"))
        assert a != b
        assert a == "approx_differentiable@b50k_cosine"

    def test_arm_labels_survive_a_filename(self):
        from pathlib import Path as _P

        arm = rp._default_arm("approx_differentiable", _P("/n/we ird/na*me.onnx"))
        assert "/" not in arm and "*" not in arm and " " not in arm

    def test_analytical_and_blackbox_no_longer_collapse_together(self):
        assert rp._default_arm("analytical", None) != rp._default_arm("blackbox", None)

    def test_non_finite_numbers_are_nulled_so_the_shard_is_valid_json(self):
        # json.dumps writes bare NaN/Infinity, which RFC 8259 forbids. rhat is
        # NaN for a single-chain fit and z is inf when the posterior sd is 0.
        cleaned = rp._finite(
            {"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": 2.0}}
        )
        text = json.dumps(cleaned)
        assert "NaN" not in text and "Infinity" not in text
        assert cleaned == {"a": None, "b": [1.0, None], "c": {"d": 2.0}}


class TestSanity:
    # Swept across every identity shape --
    # the data checks read numbers off a shard, never its names.
    pytestmark = pytest.mark.usefixtures("shape")

    def _frame(self, responses):
        import pandas as pd

        return pd.DataFrame(
            {"rt": np.full(len(responses), 0.5), "response": np.asarray(responses)}
        )

    def test_a_lopsided_dataset_is_flagged_by_min_choice_share(self):
        # The classic DDM failure mode: nearly all mass on one side, so the
        # parameters only the other side identifies cannot come back and the
        # likelihood is not what is at fault.
        data = self._frame([1] * 99 + [-1])
        out = rp._sanity(data, DDM_SDV)
        assert out["min_choice_share"] == pytest.approx(0.01)
        assert out["choice_shares"] == {
            "-1": pytest.approx(0.01),
            "1": pytest.approx(0.99),
        }

    def test_a_choice_that_never_occurred_scores_zero_not_a_missing_key(self):
        data = self._frame([0] * 50 + [1] * 50)
        out = rp._sanity(data, RACE4)
        assert out["n_choices_observed"] == 2
        assert out["min_choice_share"] == 0.0

    def test_the_censoring_criterion_is_exact_not_a_fudge_factor(self):
        # A censored trial comes back at max_t + t, so it is always at or above
        # the budget; a trial that terminated on its own never reaches it. The
        # old `>= max_rt - 0.1` counted slow-but-finished trials as censored,
        # and 20.0 was the wrong ceiling anyway (measured: RTs reach 21.9).
        import pandas as pd

        data = pd.DataFrame(
            {"rt": [0.5, 19.95, 20.0, 21.9], "response": [1, 1, -1, -1]}
        )
        assert rp._sanity(data, DDM_SDV)["n_rt_at_ceiling"] == 2
        assert rd.SIMULATOR_MAX_T == 20.0


def shard(
    likelihood,
    model=None,
    design=None,
    index=0,
    *,
    arm=None,
    label=None,
    covered=True,
    z=0.5,
    rhat=1.0,
    ess=1000.0,
    divergence_rate=0.0,
    contraction=0.05,
    min_choice_share=0.5,
    truth=1.0,
):
    """One synthetic shard with a single parameter.

    `model`, `design` and `label` default to the active SHAPE rather than to
    fixed names: `aggregate_recovery` never reads them, which
    TestNamesAreOpaqueToTheAggregator asserts directly, so a class that opts
    into the `shape` fixture sweeps all three identities without changing a
    line of its own. Tests whose subject IS the structure inside those strings
    -- a design's `@variant` suffix, a ladder rung's position -- pass their own
    values.
    """

    model = SHAPE.model if model is None else model
    design = SHAPE.design if design is None else design
    label = SHAPE.label if label is None else label
    return {
        "schema_version": 2,
        "model": model,
        "design": design.split("@", 1)[0],
        "design_id": design,
        "likelihood": likelihood,
        "arm": arm or likelihood,
        "dataset_index": index,
        "data": {"min_choice_share": min_choice_share},
        "sampler": {"divergence_rate": divergence_rate, "divergences": 0},
        "parameters": {
            label: {
                "truth": truth,
                "mean": truth + z * 0.1,
                "sd": 0.1,
                "hdi_lo": 0.0,
                "hdi_hi": 2.0,
                "covered": covered,
                "z": z,
                "contraction": contraction,
                "rhat": rhat,
                "ess_bulk": ess,
            }
        },
    }


def errored_shard(likelihood, error="boom", **kw):
    """What the worker writes when a fit dies: no parameters block, an error."""
    dead = shard(likelihood, **kw)
    del dead["parameters"], dead["data"], dead["sampler"]
    return dead | {"error": error}


def arm_shards(likelihood, n=20, *, covered_count=None, **kw):
    """`n` shards for one arm; the first `covered_count` cover the truth."""
    if covered_count is None:
        covered_count = n
    return [
        shard(likelihood, index=i, covered=(i < covered_count), **kw) for i in range(n)
    ]


class TestBand:
    def test_the_floor_is_an_exact_binomial_quantile_not_a_normal_one(self):
        # At n=20, p=0.94 the normal approximation is invalid: n*p*(1-p) = 1.13.
        low, _ = agg._binomial_band(20, n_tests=1)
        # The floor is a realisable count over n, never an arbitrary real.
        assert low * 20 == pytest.approx(round(low * 20))

    def test_more_tests_lower_the_floor_so_the_family_wise_rate_holds(self):
        # The run fails if ANY cell fails. Without the correction a perfectly
        # calibrated network fails the gate more often than not.
        floors = [agg._binomial_band(20, n_tests=m)[0] for m in (1, 5, 26)]
        assert floors == sorted(floors, reverse=True)

        def family_rate(n_tests, n_cells):
            low = agg._binomial_band(20, n_tests=n_tests)[0]
            per = agg._binomial_cdf(round(low * 20) - 1, 20, agg.NOMINAL_COVERAGE)
            return 1 - (1 - per) ** n_cells

        assert family_rate(1, 26) > 0.5  # measured 0.534 -- worse than a coin
        assert family_rate(26, 26) < agg.FAMILY_ALPHA

    def test_a_cell_with_no_fits_is_not_given_a_floor(self):
        assert agg._binomial_band(0) == (0.0, 1.0)


class TestAggregation:
    # Swept across every identity shape --
    # coverage and bias arithmetic is identity-blind.
    pytestmark = pytest.mark.usefixtures("shape")

    def test_two_models_in_one_directory_do_not_share_cells(self):
        shards = [shard("analytical", index=i) for i in range(3)]
        shards += [shard("analytical", model=OTHER_MODEL, index=i) for i in range(3)]
        cells = agg.summarise(shards)["cells"]
        assert set(cells) == {
            cell_key("analytical"),
            cell_key("analytical", model=OTHER_MODEL),
        }

    def test_two_candidate_networks_are_separate_arms(self):
        # Pooling them would average a good network with a bad one into one
        # meaningless coverage number, and the shard files would collide.
        shards = arm_shards("approx_differentiable", arm="approx_differentiable@b50k")
        shards += arm_shards("approx_differentiable", arm="approx_differentiable@b500k")
        cells = agg.summarise(shards)["cells"]
        assert len(cells) == 2
        assert all(cell["n_fits"] == 20 for cell in cells.values())

    def test_a_shard_without_model_or_arm_is_read_as_the_legacy_single_network(self):
        # The one place a concrete model name is the SUBJECT rather than a
        # label: shards predating the field all came from one network, so the
        # default is a compatibility constant. Read from the code, not spelled
        # out here, so the two cannot drift.
        legacy = shard("analytical")
        del legacy["model"], legacy["arm"]
        cells = agg.summarise([legacy])["cells"]
        assert cell_key("analytical", model=agg._model_of({})) in cells

    def test_non_converged_fits_are_excluded_not_failed(self):
        shards = [shard("analytical", index=i, rhat=1.5) for i in range(5)]
        cell = agg.summarise(shards)["cells"][cell_key("analytical")]
        assert cell["n_fits"] == 5
        assert cell["n_converged"] == 0
        assert cell["coverage"] is None

    def test_a_nan_rhat_is_excluded_rather_than_admitted(self):
        # Single-chain fits report rhat NaN, and `NaN <= 1.01` is False -- but
        # only by accident of IEEE semantics, so it is pinned.
        cell = agg.summarise(
            [shard("analytical", index=i, rhat=float("nan")) for i in range(20)]
        )["cells"][cell_key("analytical")]
        assert cell["n_converged"] == 0

    def test_divergent_fits_are_dropped_before_scoring(self):
        summary = agg.summarise([shard("analytical", index=0, divergence_rate=0.5)])
        assert summary["cells"] == {}
        assert summary["excluded_for_divergences"][arm_key("analytical")] == 1

    def test_a_dataset_missing_a_response_category_is_excluded(self):
        # _sanity's docstring promises "the aggregator decides what to do with
        # it". Before this it decided nothing -- the key was never read.
        summary = agg.summarise(
            [shard("analytical", index=i, min_choice_share=0.0) for i in range(20)]
        )
        assert summary["cells"] == {}
        assert summary["excluded_for_degenerate_data"][arm_key("analytical")] == 20

    def test_errored_shards_are_collected_rather_than_crashing(self):
        summary = agg.summarise([errored_shard("analytical", model="angle", index=3)])
        assert summary["errors"][0]["error"] == "boom"
        assert summary["attempted"][arm_key("analytical", model="angle")] == 1

    def test_a_shard_with_no_parameters_block_is_an_error_not_a_crash(self):
        broken = shard("analytical")
        del broken["parameters"]
        summary = agg.summarise([broken])
        assert summary["cells"] == {}
        assert "no parameters" in summary["errors"][0]["error"]

    def test_median_contraction_is_the_median_not_the_upper_middle(self):
        # contractions[n // 2] biases upward on an even count, and this number
        # is compared across runs and against the reference arm.
        shards = [
            shard("analytical", index=i, contraction=c)
            for i, c in enumerate([0.1, 0.2, 0.3, 0.4])
        ]
        cell = agg.summarise(shards)["cells"][cell_key("analytical")]
        assert cell["median_contraction"] == pytest.approx(0.25)

    def test_correlation_is_none_when_truth_does_not_vary(self):
        assert agg._corr([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
        assert agg._corr([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_report_round_trips_through_json(self, tmp_path):
        for i in range(3):
            (tmp_path / f"recovery_x_{i}.json").write_text(
                json.dumps(shard("analytical", index=i))
            )
        loaded = agg.load_shards(tmp_path)
        assert len(loaded) == 3
        assert json.loads(json.dumps(agg.summarise(loaded)))["cells"]


class TestVerdict:
    # Swept across every identity shape --
    # the pass/fail rules are identity-blind.
    pytestmark = pytest.mark.usefixtures("shape")

    def test_a_network_missing_coverage_the_exact_arm_reaches_fails(self):
        shards = arm_shards("analytical")
        shards += arm_shards("approx_differentiable", covered_count=8)
        passed, failures = agg.verdict(agg.summarise(shards))
        assert not passed
        assert "coverage" in failures[0]

    def test_a_shortfall_the_exact_arm_shares_is_not_the_networks_fault(self):
        # Both arms cover 40% of the time: the design cannot identify this
        # parameter, which is a fact about the model, not about the network.
        shards = arm_shards("analytical", covered_count=8)
        shards += arm_shards("approx_differentiable", covered_count=8)
        passed, failures = agg.verdict(agg.summarise(shards))
        assert passed, failures

    def test_a_blackbox_arm_is_a_reference_not_a_defendant(self):
        # blackbox is an exact simulation-based likelihood with no network in
        # it. Judging it as "the network" would blame it for the model. Its
        # own coverage shortfall here must not produce a failure, while the
        # network alongside it is judged normally.
        shards = arm_shards("blackbox", covered_count=8)
        shards += arm_shards("approx_differentiable", covered_count=8)
        passed, failures = agg.verdict(agg.summarise(shards))
        assert passed, failures
        assert not any("blackbox" in f for f in failures)

    def test_a_blackbox_arm_can_anchor_an_attribution(self):
        shards = arm_shards("blackbox")
        shards += arm_shards("approx_differentiable", covered_count=8)
        passed, failures = agg.verdict(agg.summarise(shards))
        assert not passed
        assert "exact likelihood reaches" in failures[0]

    def test_a_wide_but_covering_posterior_passes(self):
        # Honesty is not punished: a posterior wide because the data are
        # uninformative is the model telling the truth, and that is reported.
        shards = arm_shards("analytical", contraction=0.7)
        shards += arm_shards("approx_differentiable", contraction=0.7)
        summary = agg.summarise(shards)
        passed, failures = agg.verdict(summary)
        assert passed, failures
        assert summary["cells"][cell_key("approx_differentiable")][
            "median_contraction"
        ] == pytest.approx(0.7)

    def test_a_likelihood_that_moved_nothing_cannot_pass_on_coverage_alone(self):
        # Truths come from a box inset 10% per side while the prior spans the
        # full box, so the PRIOR's own 94% interval contains every possible
        # truth. A network whose likelihood is constant therefore scores
        # coverage 1.00 -- perfect, and completely uninformative.
        shards = arm_shards("approx_differentiable", contraction=0.99)
        passed, failures = agg.verdict(agg.summarise(shards))
        assert not passed
        assert "moved it almost not at all" in failures[0]

    def test_a_network_much_vaguer_than_the_exact_arm_fails(self):
        shards = arm_shards("analytical", contraction=0.10)
        shards += arm_shards("approx_differentiable", contraction=0.50)
        passed, failures = agg.verdict(agg.summarise(shards))
        assert not passed
        assert "wider than the exact likelihood" in failures[0]

    def test_the_exact_arm_is_judged_against_its_own_floor(self):
        """The floor depends on n, and the two arms need not have the same n.

        Here the reference ran 50 fits and covered 41 (0.82): below its own
        0.86 floor, but above the 0.80 floor of the 10-fit network arm. So the
        reference misses too and the shortfall is the design's. Judging the
        reference against the NETWORK's floor -- which is what the code did --
        reads it as a clean reference and blames the network instead.
        """

        shards = arm_shards("analytical", n=50, covered_count=41)
        shards += arm_shards("approx_differentiable", n=10, covered_count=7)
        summary = agg.summarise(shards)
        ref = summary["cells"][cell_key("analytical")]
        net = summary["cells"][cell_key("approx_differentiable")]
        assert ref["coverage"] < ref["coverage_band"][0]  # misses its own floor
        assert ref["coverage"] > net["coverage_band"][0]  # clears the network's
        assert net["coverage"] < net["coverage_band"][0]  # network misses too
        passed, failures = agg.verdict(summary)
        assert passed, failures

    def test_a_biased_network_fails_even_while_covering(self):
        shards = arm_shards("analytical", z=0.2)
        shards += arm_shards("approx_differentiable", z=5.0)
        passed, failures = agg.verdict(agg.summarise(shards))
        assert not passed
        assert any("bias rate" in f for f in failures)


class TestSilenceIsNotAPass:
    """The gate must distinguish "calibrated" from "nothing ran"."""

    # Swept across every identity shape --
    # so is the guard against silent non-evidence.
    pytestmark = pytest.mark.usefixtures("shape")

    def test_a_sweep_where_every_fit_errored_does_not_pass(self):
        shards = [
            errored_shard(
                "approx_differentiable",
                "RuntimeError: onnx session init failed",
                index=i,
            )
            for i in range(20)
        ]
        passed, failures = agg.verdict(agg.summarise(shards))
        assert not passed
        assert "nothing to judge" in failures[0]

    def test_a_sweep_where_every_fit_diverged_does_not_pass(self):
        shards = arm_shards("approx_differentiable", divergence_rate=0.9)
        passed, failures = agg.verdict(agg.summarise(shards))
        assert not passed

    def test_a_sweep_where_nothing_converged_does_not_pass(self):
        shards = arm_shards("approx_differentiable", rhat=1.5)
        passed, failures = agg.verdict(agg.summarise(shards))
        assert not passed

    def test_the_errored_count_is_for_this_cell_not_every_cell_sharing_an_arm(self):
        # An arm label says nothing about which model or design it ran on, so
        # counting errors by arm alone inflates the one number an operator
        # reads to decide whether a sweep is worth rerunning.
        shards = [errored_shard("approx_differentiable", index=i) for i in range(3)]
        shards += [
            errored_shard("approx_differentiable", model=OTHER_MODEL, index=i)
            for i in range(9)
        ]
        _, failures = agg.verdict(agg.summarise(shards))
        where = arm_key("approx_differentiable").replace(agg.KEY_SEPARATOR, "/")
        assert any(f"{where}: 3 fits" in f for f in failures)
        assert any("(3 errored)" in f for f in failures)
        assert any("(9 errored)" in f for f in failures)
        assert not any("(12 errored)" in f for f in failures)

    def test_an_empty_report_does_not_pass(self):
        passed, failures = agg.verdict(agg.summarise([]))
        assert not passed
        assert "nothing here to pass" in failures[0]

    def test_attrition_below_the_floor_is_inconclusive_not_a_pass(self):
        # Three survivors out of twenty used to clear a band wide enough to
        # admit almost anything, and the stdout line still said n_shards: 20.
        shards = arm_shards("approx_differentiable", n=3)
        summary = agg.summarise(shards)
        assert not summary["cells"][cell_key("approx_differentiable")]["eligible"]
        passed, failures = agg.verdict(summary)
        assert not passed
        assert "inconclusive" in failures[0]

    def test_a_non_converged_exact_arm_does_not_silently_downgrade_the_gate(self):
        # The old code fell through to the ladder here and its message then
        # claimed there was no exact arm -- for a model that has one.
        shards = arm_shards("analytical", n=4)
        shards += arm_shards("approx_differentiable", covered_count=8)
        passed, failures = agg.verdict(agg.summarise(shards))
        assert not passed
        assert any("fix the reference arm first" in f for f in failures)


class TestLadderAttribution:
    """What stands in for an exact likelihood on models that have none."""

    def test_richer_rungs_need_at_least_as_much_data_and_design(self):
        assert set(agg._richer_rungs("L0_n500")) == {
            "L0_n1000",
            "L0_n2000",
            "L1_n500",
            "L1_n1000",
            "L1_n2000",
        }
        # More conditions but fewer trials is not richer -- it is a trade, so
        # L1_n250 does not rescue L0_n500 and L0_n1000 does not rescue L1_n250.
        assert "L1_n250" not in agg._richer_rungs("L0_n500")
        assert set(agg._richer_rungs("L1_n250")) == {
            "L1_n500",
            "L1_n1000",
            "L1_n2000",
        }
        assert agg._richer_rungs("L0_n2000") == ["L1_n2000"]
        assert agg._richer_rungs("L1_n2000") == []

    def test_the_optional_top_rung_is_still_a_richer_rung_if_you_ran_it(self):
        assert "L1_n2000" in agg._richer_rungs("L1_n1000")

    def _arm(self, design, covered_count, n=20, **kw):
        # angle has no analytical arm, so every attribution here goes through
        # the ladder -- which is the point of this class.
        return arm_shards(
            "approx_differentiable",
            n,
            covered_count=covered_count,
            model="angle",
            design=design,
            **kw,
        )

    def test_a_shortfall_a_richer_rung_repairs_is_charged_to_the_design(self):
        # angle has no exact likelihood. L0_n500 misses, adding conditions
        # fixes it: that is what an identifiability limit looks like.
        shards = self._arm("L0_n500", 8) + self._arm("L1_n500", 20)
        passed, failures = agg.verdict(agg.summarise(shards))
        assert passed, failures

    def test_the_ladder_crosses_the_l0_to_l1_boundary_for_the_drift_itself(self):
        # THE case the ladder exists for. The condition parameter is labelled
        # `v` at L0 and `v[0]`..`v[3]` at L1, so an exact-label lookup never
        # finds the rung that rescues it and a correct network gets blamed for
        # the model's identifiability limit.
        shards = self._arm("L0_n2000", 8, label="v")
        for i in range(4):
            shards += self._arm("L1_n2000", 20, label=f"v[{i}]")
        passed, failures = agg.verdict(agg.summarise(shards))
        assert passed, failures

    def test_one_condition_recovering_is_not_enough_to_excuse_the_rest(self):
        shards = self._arm("L0_n2000", 8, label="v")
        shards += self._arm("L1_n2000", 20, label="v[0]")
        for i in (1, 2, 3):
            shards += self._arm("L1_n2000", 8, label=f"v[{i}]")
        passed, failures = agg.verdict(agg.summarise(shards))
        assert not passed

    def test_two_l1_variants_do_not_pool_into_one_cell(self):
        # L1_n500@v and L1_n500@sv share a trial count and a condition count
        # and nothing else. Pooling their shared-parameter cells would average
        # two different experiments into one coverage number.
        shards = self._arm("L1_n500@v", 20, label="z")
        shards += self._arm("L1_n500@sv", 8, label="z")
        cells = agg.summarise(shards)["cells"]
        assert set(cells) == {
            cell_key(
                "approx_differentiable", model="angle", design="L1_n500@v", label="z"
            ),
            cell_key(
                "approx_differentiable", model="angle", design="L1_n500@sv", label="z"
            ),
        }

    def test_one_variant_recovering_is_enough_to_excuse_the_rung_below(self):
        # Across variants the question is whether SOME richer design recovers
        # the parameter, and one that does is an answer. (Within a variant the
        # rule is the opposite -- every condition must clear.) Asserted on the
        # rule itself rather than end-to-end, because the variant that fails
        # here is also a gated cell that fails on its own merits, and a
        # whole-run verdict could not tell the two reasons apart.
        cells = agg.summarise(
            self._arm("L1_n500@sv", 8, label="z")
            + self._arm("L1_n500@v", 20, label="z")
        )["cells"]
        assert agg._rung_recovers(
            cells, "angle", "approx_differentiable", "L1_n500", "z"
        )

    def test_no_variant_recovering_means_the_rung_does_not_rescue(self):
        cells = agg.summarise(
            self._arm("L1_n500@sv", 8, label="z") + self._arm("L1_n500@v", 8, label="z")
        )["cells"]
        assert not agg._rung_recovers(
            cells, "angle", "approx_differentiable", "L1_n500", "z"
        )

    def test_no_variant_recovering_still_charges_the_network(self):
        shards = self._arm("L0_n500", 8, label="z")
        shards += self._arm("L1_n500@sv", 8, label="z")
        shards += self._arm("L1_n500@v", 8, label="z")
        passed, failures = agg.verdict(agg.summarise(shards))
        assert not passed
        assert any("unconfirmed" in f for f in failures)

    def test_the_ladder_crosses_into_a_variant_that_varies_a_shared_parameter(self):
        # L0 leaves sv weakly determined; the rung that rescues it is the one
        # that MANIPULATES sv, not the drift-varying default. An exact-label or
        # exact-design lookup would never find it.
        shards = self._arm("L0_n2000", 8, label="sv")
        for i in range(4):
            shards += self._arm("L1_n2000@sv", 20, label=f"sv[{i}]")
        passed, failures = agg.verdict(agg.summarise(shards))
        assert passed, failures

    def test_a_shortfall_flat_across_the_ladder_is_charged_to_the_network(self):
        shards = self._arm("L0_n500", 8) + self._arm("L1_n2000", 8)
        passed, failures = agg.verdict(agg.summarise(shards))
        assert not passed
        # The weaker evidence has to read as weaker.
        assert any("unconfirmed" in f for f in failures)

    def test_a_richer_rung_must_actually_recover_not_merely_exist(self):
        shards = self._arm("L0_n500", 8) + self._arm("L1_n500", 9)
        passed, _ = agg.verdict(agg.summarise(shards))
        assert not passed

    def test_a_thin_richer_rung_cannot_whitewash_a_shortfall(self):
        # Two surviving fits clear their own band trivially. That must not
        # excuse twenty fits missing at the rung below.
        shards = self._arm("L0_n500", 8) + self._arm("L1_n500", 2, n=2)
        passed, _ = agg.verdict(agg.summarise(shards))
        assert not passed

    def test_bias_without_an_exact_arm_still_has_an_absolute_bar(self):
        shards = [
            shard("approx_differentiable", model="angle", index=i, z=5.0)
            for i in range(20)
        ]
        passed, failures = agg.verdict(agg.summarise(shards))
        assert not passed
        assert any("no usable exact-likelihood reference" in f for f in failures)

    def test_an_unbiased_network_without_an_exact_arm_passes(self):
        shards = [
            shard("approx_differentiable", model="angle", index=i, z=0.2)
            for i in range(20)
        ]
        passed, failures = agg.verdict(agg.summarise(shards))
        assert passed, failures
