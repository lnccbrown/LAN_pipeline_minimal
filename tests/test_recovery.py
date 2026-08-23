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

import json

import numpy as np
import pytest

import aggregate_recovery as agg
import recover_parameters as rp
import recovery_designs as rd

# Built by hand rather than through `load_model`, which needs HSSM — CI installs
# only the default dependency group. The values mirror what HSSM declares for
# the `approx_differentiable` likelihood of each model.
DDM_SDV = rd.ModelUnderTest(
    name="ddm_sdv",
    params=("v", "a", "z", "t", "sv"),
    bounds={
        "v": (-3.0, 3.0),
        "a": (0.3, 2.5),
        "z": (0.1, 0.9),
        "t": (0.0, 2.0),
        "sv": (0.0, 2.5),
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

    def test_the_rt_ceiling_comes_from_the_model_not_a_constant(self):
        import pandas as pd

        data = pd.DataFrame({"rt": [0.5, 19.95, 20.0], "response": [1, 1, -1]})
        assert rp._sanity(data, DDM_SDV)["n_rt_at_ceiling"] == 2


def shard(
    likelihood,
    model="ddm_sdv",
    design="L0_n500",
    index=0,
    *,
    covered=True,
    z=0.5,
    rhat=1.0,
    ess=1000.0,
    divergence_rate=0.0,
    contraction=0.05,
    truth=1.0,
):
    """One synthetic shard with a single parameter, `v`."""
    return {
        "schema_version": 2,
        "model": model,
        "design": design,
        "likelihood": likelihood,
        "dataset_index": index,
        "sampler": {"divergence_rate": divergence_rate, "divergences": 0},
        "parameters": {
            "v": {
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


class TestAggregation:
    def test_coverage_band_widens_as_datasets_shrink(self):
        low_20, high_20 = agg._binomial_band(20)
        low_200, high_200 = agg._binomial_band(200)
        assert low_20 < low_200 < agg.NOMINAL_COVERAGE < high_200 <= high_20
        # 20 datasets cannot distinguish 0.90 from nominal, and the gate must
        # not pretend otherwise.
        assert low_20 < 0.90

    def test_two_models_in_one_directory_do_not_share_cells(self):
        shards = [shard("analytical", model="ddm_sdv", index=i) for i in range(3)]
        shards += [shard("analytical", model="angle", index=i) for i in range(3)]
        cells = agg.summarise(shards)["cells"]
        assert set(cells) == {
            "ddm_sdv|analytical|L0_n500|v",
            "angle|analytical|L0_n500|v",
        }
        assert all(cell["n_fits"] == 3 for cell in cells.values())

    def test_a_shard_without_a_model_field_is_read_as_ddm_sdv(self):
        legacy = shard("analytical")
        del legacy["model"]
        assert "ddm_sdv|analytical|L0_n500|v" in agg.summarise([legacy])["cells"]

    def test_non_converged_fits_are_excluded_not_failed(self):
        shards = [shard("analytical", index=i, rhat=1.5) for i in range(5)]
        cell = agg.summarise(shards)["cells"]["ddm_sdv|analytical|L0_n500|v"]
        assert cell["n_fits"] == 5
        assert cell["n_converged"] == 0
        assert cell["coverage"] is None

    def test_divergent_fits_are_dropped_before_scoring(self):
        summary = agg.summarise([shard("analytical", index=0, divergence_rate=0.5)])
        assert summary["cells"] == {}
        assert summary["excluded_for_divergences"]["ddm_sdv|analytical|L0_n500"] == 1

    def test_errored_shards_are_collected_rather_than_crashing(self):
        shards = [
            {
                "model": "angle",
                "design": "L0_n500",
                "likelihood": "analytical",
                "dataset_index": 3,
                "error": "boom",
            }
        ]
        summary = agg.summarise(shards)
        assert summary["errors"][0]["error"] == "boom"
        assert summary["attempted"]["angle|analytical|L0_n500"] == 1

    def test_a_network_missing_coverage_the_analytical_arm_reaches_fails(self):
        shards = [shard("analytical", index=i, covered=True) for i in range(20)]
        shards += [
            shard("approx_differentiable", index=i, covered=(i < 8)) for i in range(20)
        ]
        passed, failures = agg.verdict(agg.summarise(shards))
        assert not passed
        assert "coverage" in failures[0]

    def test_a_shortfall_the_analytical_arm_shares_is_not_the_networks_fault(self):
        # Both arms cover 40% of the time: the design cannot identify this
        # parameter, which is a fact about the model, not about the network.
        shards = [shard("analytical", index=i, covered=(i < 8)) for i in range(20)]
        shards += [
            shard("approx_differentiable", index=i, covered=(i < 8)) for i in range(20)
        ]
        passed, failures = agg.verdict(agg.summarise(shards))
        assert passed, failures

    def test_a_wide_but_covering_posterior_passes(self):
        # Honesty is not punished: contraction near 1 means the data barely
        # moved the prior, and that is reported, never gated.
        shards = [shard("analytical", index=i, contraction=0.95) for i in range(20)]
        shards += [
            shard("approx_differentiable", index=i, contraction=0.95) for i in range(20)
        ]
        summary = agg.summarise(shards)
        passed, _ = agg.verdict(summary)
        assert passed
        assert summary["cells"]["ddm_sdv|approx_differentiable|L0_n500|v"][
            "median_contraction"
        ] == pytest.approx(0.95)

    def test_a_biased_network_fails_even_while_covering(self):
        shards = [shard("analytical", index=i, z=0.2) for i in range(20)]
        shards += [shard("approx_differentiable", index=i, z=5.0) for i in range(20)]
        passed, failures = agg.verdict(agg.summarise(shards))
        assert not passed
        assert any("bias rate" in f for f in failures)

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


class TestLadderAttribution:
    """What stands in for the analytical arm on models that have none."""

    def test_richer_rungs_need_at_least_as_much_data_and_design(self):
        assert set(agg._richer_rungs("L0_n500")) == {
            "L0_n2000",
            "L1_n500",
            "L1_n2000",
        }
        # More conditions but fewer trials is not richer — it is a trade.
        assert agg._richer_rungs("L0_n2000") == ["L1_n2000"]
        assert agg._richer_rungs("L1_n2000") == []

    def _arm(self, design, covered_count, n=20):
        return [
            shard(
                "approx_differentiable",
                model="angle",
                design=design,
                index=i,
                covered=(i < covered_count),
            )
            for i in range(n)
        ]

    def test_a_shortfall_a_richer_rung_repairs_is_charged_to_the_design(self):
        # No analytical arm anywhere here: angle has none. L0_n500 misses, but
        # adding conditions fixes it, which is what an identifiability limit
        # looks like.
        shards = self._arm("L0_n500", 8) + self._arm("L1_n500", 20)
        passed, failures = agg.verdict(agg.summarise(shards))
        assert passed, failures

    def test_a_shortfall_flat_across_the_ladder_is_charged_to_the_network(self):
        shards = self._arm("L0_n500", 8) + self._arm("L1_n2000", 8)
        passed, failures = agg.verdict(agg.summarise(shards))
        assert not passed
        # The weaker evidence has to read as weaker.
        assert any("unconfirmed" in f for f in failures)

    def test_bias_without_an_analytical_arm_still_has_an_absolute_bar(self):
        shards = [
            shard("approx_differentiable", model="angle", index=i, z=5.0)
            for i in range(20)
        ]
        passed, failures = agg.verdict(agg.summarise(shards))
        assert not passed
        assert any("no analytical arm" in f for f in failures)

    def test_an_unbiased_network_without_an_analytical_arm_passes(self):
        shards = [
            shard("approx_differentiable", model="angle", index=i, z=0.2)
            for i in range(20)
        ]
        passed, failures = agg.verdict(agg.summarise(shards))
        assert passed, failures
