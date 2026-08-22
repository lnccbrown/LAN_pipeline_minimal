"""Tests for the recovery harness.

Fast and offline throughout. The statistical core — does a fit recover the
truth — is not unit-testable in CI; it is exercised by the pilot and by the
harness self-test described in the plan (analytical L0 at N=4000, where
recovery is well established, so a failure indicts the harness rather than the
model). What is tested here is everything that decides *whether the numbers
mean anything*: the design construction, the convergence filter, and the
attribution rule that charges a failure to the network only when the
analytical arm did better.
"""

import json

import numpy as np
import pytest

import aggregate_recovery as agg
import recovery_designs as rd


class TestDesigns:
    def test_every_design_divides_evenly_into_conditions(self):
        # A ragged final condition would give one cell fewer trials and quietly
        # weaken exactly the comparison the ladder exists to make.
        for design in rd.DESIGNS.values():
            assert design.n_trials % design.n_conditions == 0

    def test_l0_has_no_condition_column_and_l1_does(self):
        l0, _ = rd.build_dataset(rd.DESIGNS["L0_n500"], seed=1)
        l1, _ = rd.build_dataset(rd.DESIGNS["L1_n500"], seed=1)
        assert "condition" not in l0.columns
        assert sorted(l1["condition"].unique()) == [0, 1, 2, 3]
        assert len(l0) == len(l1) == 500

    def test_conditions_are_balanced(self):
        data, _ = rd.build_dataset(rd.DESIGNS["L1_n2000"], seed=2)
        counts = data["condition"].value_counts().to_numpy()
        assert set(counts.tolist()) == {500}

    def test_truth_lands_inside_hssm_bounds_not_the_training_box(self):
        # sv is the one that matters: HSSM caps the ONNX likelihood at 1.0
        # while the networks were trained to 2.5. A truth above 1.0 could not
        # be represented by the sampler at all.
        for name, design in rd.DESIGNS.items():
            for seed in range(20):
                truth = rd.draw_truth(design, seed)
                for param, value in truth.items():
                    lo, hi = rd.shrunk_bounds()[param]
                    values = value if isinstance(value, list) else [value]
                    for v in values:
                        assert lo <= v <= hi, f"{name}/{param}={v} outside [{lo},{hi}]"
            assert truth["sv"] <= 1.0

    def test_drift_varies_by_condition_only_in_l1(self):
        l0 = rd.draw_truth(rd.DESIGNS["L0_n500"], seed=3)
        l1 = rd.draw_truth(rd.DESIGNS["L1_n500"], seed=3)
        assert isinstance(l0["v"], float)
        assert isinstance(l1["v"], list) and len(l1["v"]) == 4
        # Everything else stays shared, which is the point of the design.
        for param in ("a", "z", "t", "sv"):
            assert isinstance(l1[param], float)

    def test_same_seed_gives_identical_data_so_arms_are_paired(self):
        design = rd.DESIGNS["L0_n500"]
        first, truth_a = rd.build_dataset(design, seed=7)
        second, truth_b = rd.build_dataset(design, seed=7)
        assert truth_a == truth_b
        assert np.array_equal(first["rt"].to_numpy(), second["rt"].to_numpy())

    def test_model_spec_pins_priors_to_the_onnx_box_for_every_parameter(self):
        # Both arms must share priors, or the comparison is not paired: the
        # analytical likelihood's own default bounds leave sv unbounded above.
        spec = rd.model_spec(rd.DESIGNS["L1_n500"])
        by_name = {entry["name"]: entry for entry in spec["include"]}
        assert set(by_name) == set(rd.PARAM_ORDER)
        assert by_name["sv"]["prior"] == {"name": "Uniform", "lower": 0.0, "upper": 1.0}
        assert by_name["v"]["formula"] == "v ~ 0 + C(condition)"
        assert "formula" not in by_name["a"]

    def test_posterior_names_match_the_formula(self):
        assert rd.posterior_names(rd.DESIGNS["L0_n500"])["v"] == ["v"]
        assert rd.posterior_names(rd.DESIGNS["L1_n500"])["v"] == ["v_C(condition)"]


def shard(
    likelihood,
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
        "schema_version": 1,
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

    def test_non_converged_fits_are_excluded_not_failed(self):
        shards = [shard("analytical", index=i, rhat=1.5) for i in range(5)]
        summary = agg.summarise(shards)
        cell = summary["cells"]["analytical|L0_n500|v"]
        assert cell["n_fits"] == 5
        assert cell["n_converged"] == 0
        assert cell["coverage"] is None

    def test_divergent_fits_are_dropped_before_scoring(self):
        shards = [shard("analytical", index=0, divergence_rate=0.5)]
        summary = agg.summarise(shards)
        assert summary["cells"] == {}
        assert summary["excluded_for_divergences"]["analytical|L0_n500"] == 1

    def test_errored_shards_are_collected_rather_than_crashing(self):
        shards = [
            {
                "design": "L0_n500",
                "likelihood": "analytical",
                "dataset_index": 3,
                "error": "boom",
            }
        ]
        summary = agg.summarise(shards)
        assert summary["errors"][0]["error"] == "boom"
        assert summary["attempted"]["analytical|L0_n500"] == 1

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
        assert summary["cells"]["approx_differentiable|L0_n500|v"][
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
        summary = agg.summarise(loaded)
        assert json.loads(json.dumps(summary))["cells"]

    def test_shared_truths_are_identical_across_ladder_levels(self):
        # The ladder holds the trial count fixed so that design structure is
        # the only difference between L0 and L1. If the shared parameters also
        # drifted between levels, "L1 recovers sv better" could just mean "L1
        # drew an easier sv", and the whole comparison would be worthless.
        for n in (500, 2000):
            l0 = rd.draw_truth(rd.DESIGNS[f"L0_n{n}"], seed=11)
            l1 = rd.draw_truth(rd.DESIGNS[f"L1_n{n}"], seed=11)
            for param in ("a", "z", "t", "sv"):
                assert l0[param] == l1[param], param
        # Drift is the parameter the design deliberately changes, so it is the
        # one thing allowed to differ.
        assert isinstance(l1["v"], list)
