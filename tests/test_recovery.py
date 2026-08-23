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
        assert rd.posterior_names(model, rd.DESIGNS["L0_n500"])[p] == [p]
        assert rd.posterior_names(model, rd.DESIGNS["L1_n500"])[p] == [
            f"{p}_C(condition)"
        ]


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

    @pytest.mark.parametrize("param", ["a", "t", "sv"])
    def test_a_non_drift_parameter_really_moves_the_data(self, param):
        # The check that matters: it is not enough for the formula to name the
        # parameter, the simulated conditions have to actually differ.
        #
        # Which RT moment moves is the parameter's own business, and assuming
        # it is always the mean would be the drift-centric habit this test
        # exists to break. Measured at n=2000: `a` and `t` track the mean
        # (r = 0.998, 1.000) while `sv` barely does (0.872) and tracks the
        # spread instead (0.955) -- inter-trial drift variability is a
        # dispersion effect, which is a large part of why sv is the hard one.
        model = varying(DDM_SDV, param)
        design = rd.DESIGNS["L1_n2000"]
        data, truth = rd.build_dataset(model, design, seed=4)
        groups = [data.loc[data["condition"] == c, "rt"] for c in range(4)]
        moved = max(
            abs(np.corrcoef(truth[param], [f(g) for g in groups])[0, 1])
            for f in (np.mean, np.std)
        )
        assert moved > 0.9, f"{param} left the RT distribution unchanged"

    @pytest.mark.parametrize("param", ["a", "theta"])
    def test_the_formula_and_posterior_name_follow_the_choice(self, param):
        model = varying(ANGLE, param)
        design = rd.DESIGNS["L1_n500"]
        by_name = {e["name"]: e for e in rd.model_spec(model, design)["include"]}
        assert by_name[param]["formula"] == f"{param} ~ 0 + C(condition)"
        assert rd.posterior_names(model, design)[param] == [f"{param}_C(condition)"]
        # And drift is now an ordinary shared parameter.
        if param != "v":
            assert "formula" not in by_name["v"]
            assert rd.posterior_names(model, design)["v"] == ["v"]

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


class TestSummarise:
    """_summarise produces every number in the report, so it gets pinned."""

    def _posterior(self, values: dict):
        xr = pytest.importorskip("xarray")
        return xr.Dataset(
            {
                name: (("chain", "draw") + (("cond",) if arr.ndim == 3 else ()), arr)
                for name, arr in values.items()
            }
        )

    def test_a_condition_vector_expands_into_one_record_per_condition(self):
        pytest.importorskip("arviz")
        rng = np.random.default_rng(0)
        drift = rng.normal(loc=[0.5, 1.0, 1.5, 2.0], scale=0.05, size=(2, 500, 4))
        posterior = self._posterior({"v_C(condition)": drift})
        truth = {"v": [0.5, 1.0, 1.5, 2.0]}
        out = rp._summarise(
            {"posterior": posterior},
            rd.ModelUnderTest(
                name="t", params=("v",), bounds={"v": (-3.0, 3.0)}, condition_param="v"
            ),
            rd.DESIGNS["L1_n500"],
            truth,
        )
        assert sorted(out) == ["v[0]", "v[1]", "v[2]", "v[3]"]
        # Each condition is scored against ITS OWN truth, not the first one.
        for i, true_i in enumerate(truth["v"]):
            assert out[f"v[{i}]"]["truth"] == true_i
            assert out[f"v[{i}]"]["mean"] == pytest.approx(true_i, abs=0.02)
            assert out[f"v[{i}]"]["covered"]

    def test_the_interval_is_a_94_percent_hdi_not_arviz_default_89_eti(self):
        # arviz 1.x defaults to an 89% equal-tailed interval, a different
        # statistic that would silently change every coverage verdict.
        pytest.importorskip("arviz")
        rng = np.random.default_rng(1)
        draws = rng.normal(0.0, 1.0, size=(2, 20000))
        out = rp._summarise(
            {"posterior": self._posterior({"v": draws})},
            rd.ModelUnderTest(
                name="t", params=("v",), bounds={"v": (-3.0, 3.0)}, condition_param="v"
            ),
            rd.DESIGNS["L0_n500"],
            {"v": 0.0},
        )["v"]
        # 94% of a standard normal is +/-1.881; 89% would be +/-1.598.
        assert out["hdi_hi"] - out["hdi_lo"] == pytest.approx(2 * 1.881, abs=0.1)
        assert rp.HDI_PROB == 0.94

    def test_contraction_is_measured_against_the_uniform_prior_sd(self):
        pytest.importorskip("arviz")
        rng = np.random.default_rng(2)
        draws = rng.normal(0.0, 0.1, size=(2, 5000))
        out = rp._summarise(
            {"posterior": self._posterior({"v": draws})},
            rd.ModelUnderTest(
                name="t", params=("v",), bounds={"v": (-3.0, 3.0)}, condition_param="v"
            ),
            rd.DESIGNS["L0_n500"],
            {"v": 0.0},
        )["v"]
        prior_sd = 6.0 / (12**0.5)
        assert out["contraction"] == pytest.approx(0.1 / prior_sd, rel=0.05)


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
    model="ddm_sdv",
    design="L0_n500",
    index=0,
    *,
    arm=None,
    label="v",
    covered=True,
    z=0.5,
    rhat=1.0,
    ess=1000.0,
    divergence_rate=0.0,
    contraction=0.05,
    min_choice_share=0.5,
    truth=1.0,
):
    """One synthetic shard with a single parameter."""
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
    def test_two_models_in_one_directory_do_not_share_cells(self):
        shards = [shard("analytical", model="ddm_sdv", index=i) for i in range(3)]
        shards += [shard("analytical", model="angle", index=i) for i in range(3)]
        cells = agg.summarise(shards)["cells"]
        assert set(cells) == {
            "ddm_sdv|analytical|L0_n500|v",
            "angle|analytical|L0_n500|v",
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
        legacy = shard("analytical")
        del legacy["model"], legacy["arm"]
        assert "ddm_sdv|analytical|L0_n500|v" in agg.summarise([legacy])["cells"]

    def test_non_converged_fits_are_excluded_not_failed(self):
        shards = [shard("analytical", index=i, rhat=1.5) for i in range(5)]
        cell = agg.summarise(shards)["cells"]["ddm_sdv|analytical|L0_n500|v"]
        assert cell["n_fits"] == 5
        assert cell["n_converged"] == 0
        assert cell["coverage"] is None

    def test_a_nan_rhat_is_excluded_rather_than_admitted(self):
        # Single-chain fits report rhat NaN, and `NaN <= 1.01` is False -- but
        # only by accident of IEEE semantics, so it is pinned.
        cell = agg.summarise(
            [shard("analytical", index=i, rhat=float("nan")) for i in range(20)]
        )["cells"]["ddm_sdv|analytical|L0_n500|v"]
        assert cell["n_converged"] == 0

    def test_divergent_fits_are_dropped_before_scoring(self):
        summary = agg.summarise([shard("analytical", index=0, divergence_rate=0.5)])
        assert summary["cells"] == {}
        assert summary["excluded_for_divergences"]["ddm_sdv|analytical|L0_n500"] == 1

    def test_a_dataset_missing_a_response_category_is_excluded(self):
        # _sanity's docstring promises "the aggregator decides what to do with
        # it". Before this it decided nothing -- the key was never read.
        summary = agg.summarise(
            [shard("analytical", index=i, min_choice_share=0.0) for i in range(20)]
        )
        assert summary["cells"] == {}
        assert (
            summary["excluded_for_degenerate_data"]["ddm_sdv|analytical|L0_n500"] == 20
        )

    def test_errored_shards_are_collected_rather_than_crashing(self):
        summary = agg.summarise(
            [
                {
                    "model": "angle",
                    "design": "L0_n500",
                    "likelihood": "analytical",
                    "arm": "analytical",
                    "dataset_index": 3,
                    "error": "boom",
                }
            ]
        )
        assert summary["errors"][0]["error"] == "boom"
        assert summary["attempted"]["angle|analytical|L0_n500"] == 1

    def test_a_shard_with_no_parameters_block_is_an_error_not_a_crash(self):
        broken = shard("analytical")
        del broken["parameters"]
        summary = agg.summarise([broken])
        assert summary["cells"] == {}
        assert "no parameters" in summary["errors"][0]["error"]

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
        assert summary["cells"]["ddm_sdv|approx_differentiable|L0_n500|v"][
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

    def test_a_biased_network_fails_even_while_covering(self):
        shards = arm_shards("analytical", z=0.2)
        shards += arm_shards("approx_differentiable", z=5.0)
        passed, failures = agg.verdict(agg.summarise(shards))
        assert not passed
        assert any("bias rate" in f for f in failures)


class TestSilenceIsNotAPass:
    """The gate must distinguish "calibrated" from "nothing ran"."""

    def test_a_sweep_where_every_fit_errored_does_not_pass(self):
        shards = [
            {
                "model": "ddm_sdv",
                "design": "L0_n500",
                "likelihood": "approx_differentiable",
                "arm": "approx_differentiable",
                "dataset_index": i,
                "error": "RuntimeError: onnx session init failed",
            }
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

    def test_an_empty_report_does_not_pass(self):
        passed, failures = agg.verdict(agg.summarise([]))
        assert not passed
        assert "nothing here to pass" in failures[0]

    def test_attrition_below_the_floor_is_inconclusive_not_a_pass(self):
        # Three survivors out of twenty used to clear a band wide enough to
        # admit almost anything, and the stdout line still said n_shards: 20.
        shards = arm_shards("approx_differentiable", n=3)
        summary = agg.summarise(shards)
        assert not summary["cells"]["ddm_sdv|approx_differentiable|L0_n500|v"][
            "eligible"
        ]
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

    def _arm(self, design, covered_count, n=20, label="v", contraction=0.05):
        return [
            shard(
                "approx_differentiable",
                model="angle",
                design=design,
                index=i,
                label=label,
                contraction=contraction,
                covered=(i < covered_count),
            )
            for i in range(n)
        ]

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
            "angle|approx_differentiable|L1_n500@v|z",
            "angle|approx_differentiable|L1_n500@sv|z",
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
