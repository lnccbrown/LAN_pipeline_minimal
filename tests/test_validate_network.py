"""Tests for the validation gate.

The fast tests build tiny ONNX graphs by hand, so they need neither HSSM nor a
network. The end-to-end test against the real production ddm.onnx is opt-in
(`-m production`): it downloads from HuggingFace and imports the whole
inference stack, which does not belong in the default suite.
"""

import importlib.metadata
import json
import sys
import types

import numpy as np
import onnx
import pandas as pd
import pytest
from onnx import TensorProto, helper

import validate_network as vn
from validate_network import (
    GONOGO_SKIP_REASON,
    gate_accuracy,
    gate_hssm_load,
    gate_parity,
    gate_structure,
    hellinger,
    hssm_load_data,
    validate_network,
)


def make_onnx(path, input_dims, output_dims=(1, 1), real_out_width=None):
    """A minimal MatMul graph with the requested input and output dims.

    dims entries may be ints (concrete) or strings (symbolic), which is the
    distinction G1 exists to enforce. The weight is sized from both, so an
    output width other than 1 produces a *valid* graph that G1 must reject on
    the contract rather than on the checker.

    ``real_out_width`` sizes the weight independently of the declared output
    shape, which is how a graph whose annotation and behaviour disagree gets
    built — the case that decides whether the gate reads metadata or measures.
    """
    in_width = input_dims[-1] if isinstance(input_dims[-1], int) else 6
    out_width = real_out_width or (
        output_dims[-1] if isinstance(output_dims[-1], int) else 1
    )

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(input_dims))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(output_dims))
    weight = helper.make_tensor(
        "w",
        TensorProto.FLOAT,
        [in_width, out_width],
        np.zeros((in_width, out_width), dtype=np.float32).ravel().tolist(),
    )
    node = helper.make_node("MatMul", ["x", "w"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y], initializer=[weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def make_constant_onnx(path, width, value):
    """A bias-only graph: zero weights, so every input maps to ``value``.

    The shape of an auxiliary network that has learned nothing, and the
    cheapest way to hand gate_accuracy a known output.
    """
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, width])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])
    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [width, 1], np.zeros(width, dtype=np.float32).tolist()
    )
    bias = helper.make_tensor("b", TensorProto.FLOAT, [1, 1], [float(value)])
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["h"]),
        helper.make_node("Add", ["h", "b"], ["y"]),
    ]
    graph = helper.make_graph(nodes, "g", [x], [y], initializer=[weight, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def make_two_output_onnx(path, input_dims=(1, 6)):
    """A valid graph that emits two separate 1-wide tensors."""
    width = input_dims[-1]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(input_dims))
    outs = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 1]) for n in ("y", "z")
    ]
    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [width, 1], np.zeros(width, dtype=np.float32).tolist()
    )
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["y"]),
        helper.make_node("Identity", ["y"], ["z"]),
    ]
    graph = helper.make_graph(nodes, "g", [x], outs, initializer=[weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


class TestStructureGate:
    def test_accepts_a_concrete_single_trial_graph(self, tmp_path):
        path = make_onnx(tmp_path / "good.onnx", (1, 6))
        result = gate_structure(path, expected_input_dim=6)
        assert result["passed"], result
        assert result["input_width"] == 6

    def test_rejects_a_symbolic_batch_dim(self, tmp_path):
        # HSSM's make_jax_func raises on symbolic dims, so this would fail at
        # load for every user rather than here.
        path = make_onnx(tmp_path / "dyn.onnx", ("batch", 6))
        result = gate_structure(path, expected_input_dim=6)
        assert not result["passed"]
        assert "symbolic" in result["error"]

    def test_rejects_a_width_that_contradicts_the_parameter_space(self, tmp_path):
        # ddm has 4 params, so a LAN must take 4 + rt + response = 6.
        path = make_onnx(tmp_path / "narrow.onnx", (1, 5))
        result = gate_structure(path, expected_input_dim=6)
        assert not result["passed"]
        assert "expected 6" in result["error"]

    def test_rejects_an_output_wider_than_one_log_density(self, tmp_path):
        # A per-choice output head is a valid ONNX graph and the wrong
        # likelihood; G2 and G4 would score its first column and say nothing.
        path = make_onnx(tmp_path / "wide.onnx", (1, 6), output_dims=(1, 2))
        result = gate_structure(path, expected_input_dim=6)
        assert not result["passed"]
        assert "2 values returned for one trial" in result["error"]

    def test_a_symbolic_output_annotation_is_judged_on_what_it_returns(self, tmp_path):
        # HSSM never reads the declared output shape (onnx2jax validates
        # graph.input dims and resolves outputs by name), so a symbolic
        # annotation over a genuinely 1-wide output is cosmetic. The gate runs
        # the graph instead of trusting either annotation.
        good = make_onnx(tmp_path / "sym_ok.onnx", (1, 6), output_dims=(1, "d"))
        assert gate_structure(good, expected_input_dim=6)["passed"]

        # The case metadata cannot catch: annotated 1 wide, actually 2 wide.
        lying = make_onnx(
            tmp_path / "lying.onnx", (1, 6), output_dims=(1, 1), real_out_width=2
        )
        result = gate_structure(lying, expected_input_dim=6)
        assert not result["passed"]
        assert "2 values returned for one trial" in result["error"]

    def test_rejects_a_graph_with_more_than_one_output(self, tmp_path):
        path = make_two_output_onnx(tmp_path / "two.onnx")
        result = gate_structure(path, expected_input_dim=6)
        assert not result["passed"]
        assert "exactly 1 input and 1 output" in result["error"]

    def test_reports_a_corrupt_file_rather_than_raising(self, tmp_path):
        path = tmp_path / "junk.onnx"
        path.write_bytes(b"not an onnx file")
        result = gate_structure(path, expected_input_dim=6)
        assert not result["passed"]
        assert "error" in result


class TestParityGate:
    def test_skips_without_a_flax_state(self, tmp_path):
        # Torch-trained and downloaded networks have no .jax sibling; that is
        # a skip, not a failure.
        result = gate_parity(tmp_path / "m.onnx", None, None, input_width=6)
        assert result["passed"] and result["skipped"]


class TestHellinger:
    def test_identical_distributions_are_zero(self):
        assert hellinger([1, 2, 3], [1, 2, 3]) == 0.0

    def test_disjoint_distributions_are_one(self):
        assert hellinger([1, 0], [0, 1]) == pytest.approx(1.0)

    def test_is_scale_invariant(self):
        # Inputs are normalized, so unnormalized histograms compare correctly.
        assert hellinger([1, 1], [50, 50]) == pytest.approx(0.0)

    def test_is_symmetric(self):
        a, b = [0.7, 0.2, 0.1], [0.3, 0.4, 0.3]
        assert hellinger(a, b) == pytest.approx(hellinger(b, a))


class TestWiring:
    def test_a_broken_graph_short_circuits_the_expensive_gates(self, tmp_path):
        """G3/G4 load HSSM and run simulations; there is nothing to learn from
        running them once the graph itself is unusable."""
        path = tmp_path / "junk.onnx"
        path.write_bytes(b"nope")
        report = validate_network(path, model_name="ddm", network_type="lan")
        assert not report["passed"]
        gates = {g["gate"]: g for g in report["gates"]}
        assert not gates["structure"]["passed"]
        for later in ("parity", "hssm_load", "density", "mass_survey"):
            assert gates[later].get("skipped"), later
            assert gates[later]["reason"] == "structure gate failed"

    def test_an_unknown_network_type_is_rejected_not_defaulted(self, tmp_path):
        # Defaulting to 0 extra inputs would surface as "input width 6 !=
        # expected 4", blaming the artifact for a mistyped flag.
        path = make_onnx(tmp_path / "good.onnx", (1, 6))
        with pytest.raises(ValueError, match="Unknown network_type"):
            validate_network(path, model_name="ddm", network_type="LAN")

    def test_report_shape_is_stable(self, tmp_path, no_derive):
        path = make_onnx(tmp_path / "good.onnx", (1, 6))
        report = validate_network(
            path, model_name="ddm", skip_hssm=True, skip_density=True
        )
        assert report["schema_version"] == 1
        # mass_survey was APPENDED deliberately: the publisher's REQUIRED_GATES
        # still name only structure/hssm_load/density, so a skip there never
        # blocks a publish, and every earlier name keeps its position.
        assert [g["gate"] for g in report["gates"]] == [
            "structure",
            "parity",
            "hssm_load",
            "density",
            "mass_survey",
        ]
        # The one top-level key the auxiliary gates added; null for a LAN.
        assert report["aux_category"] is None
        # Without lanfactory.derive (forced here, so the shape does not
        # depend on the environment) the survey skips itself with the
        # reason that says what is missing.
        survey = report["gates"][-1]
        assert survey["skipped"] and survey["passed"]
        assert survey["reason"] == (
            "lanfactory.derive not available; this environment's LANfactory predates it"
        )
        assert report["passed"]


AUX_SKIPS = {"skip_hssm": True, "skip_accuracy": True}


class TestAuxiliaryWidths:
    """ddm has 4 params: an auxiliary net takes them plus one trailing input."""

    @pytest.mark.parametrize(
        "network_type, aux_category",
        [("cpn", "choice"), ("opn", None), ("gonogo", None)],
    )
    def test_n_params_plus_one_is_accepted(self, tmp_path, network_type, aux_category):
        path = make_onnx(tmp_path / "aux.onnx", (1, 5))
        report = validate_network(
            path,
            model_name="ddm",
            network_type=network_type,
            aux_category=aux_category,
            **AUX_SKIPS,
        )
        assert report["gates"][0]["passed"], report["gates"][0]

    @pytest.mark.parametrize(
        "network_type, aux_category", [("cpn", "choice"), ("opn", None)]
    )
    def test_the_old_params_only_width_is_rejected(
        self, tmp_path, network_type, aux_category
    ):
        # The pre-contract CPN/OPN took the parameters alone. HSSM's fixture
        # ddm_cpn.onnx is still that shape, and this is what refuses it.
        path = make_onnx(tmp_path / "old.onnx", (1, 4))
        report = validate_network(
            path,
            model_name="ddm",
            network_type=network_type,
            aux_category=aux_category,
            **AUX_SKIPS,
        )
        structure = report["gates"][0]
        assert not structure["passed"]
        assert "expected 5" in structure["error"]

    def test_a_lan_still_takes_rt_and_response(self, tmp_path):
        path = make_onnx(tmp_path / "lan.onnx", (1, 6))
        report = validate_network(
            path, model_name="ddm", skip_hssm=True, skip_density=True
        )
        assert report["gates"][0]["passed"]


class TestAuxiliaryWiring:
    @pytest.mark.parametrize(
        "network_type, aux_category", [("cpn", "choice"), ("opn", None)]
    )
    def test_an_aux_report_lists_exactly_the_aux_gates(
        self, tmp_path, network_type, aux_category
    ):
        path = make_onnx(tmp_path / "aux.onnx", (1, 5))
        report = validate_network(
            path,
            model_name="ddm",
            network_type=network_type,
            aux_category=aux_category,
            **AUX_SKIPS,
        )
        assert [g["gate"] for g in report["gates"]] == [
            "structure",
            "parity",
            "hssm_missing_load",
            "accuracy",
        ]
        assert report["network_type"] == network_type
        assert (
            report["aux_category"] == {"cpn": "choice", "opn": "omission"}[network_type]
        )
        # The skips must actually take effect: a broken skip would leave this
        # suite green while it imported HSSM and simulated 100k trials a draw.
        gates = {g["gate"]: g for g in report["gates"]}
        assert gates["hssm_missing_load"]["skipped"]
        assert gates["hssm_missing_load"]["reason"] == "--skip-hssm"
        assert gates["accuracy"]["skipped"]
        assert gates["accuracy"]["reason"] == "--skip-accuracy"

    @pytest.mark.parametrize(
        "network_type, expected",
        [("cpn", "choice"), ("opn", "omission"), ("gonogo", "nogo")],
    )
    def test_each_type_defaults_to_its_output_category(
        self, tmp_path, network_type, expected
    ):
        # The vocabulary names the OUTPUT probability, shared with LANfactory's
        # corpora and the publisher's provenance; each type has exactly one.
        path = make_onnx(tmp_path / "aux.onnx", (1, 5))
        report = validate_network(
            path, model_name="ddm", network_type=network_type, **AUX_SKIPS
        )
        assert report["aux_category"] == expected

    def test_gonogo_reports_its_two_unrunnable_gates_as_skipped(self, tmp_path):
        path = make_onnx(tmp_path / "gonogo.onnx", (1, 5))
        report = validate_network(path, model_name="ddm", network_type="gonogo")
        gates = {g["gate"]: g for g in report["gates"]}
        for name in ("hssm_missing_load", "accuracy"):
            assert gates[name]["skipped"], name
            assert gates[name]["reason"] == GONOGO_SKIP_REASON
            assert gates[name]["passed"], name
        # A structurally valid gonogo with everything else skipped must not
        # exit non-zero: a skip is a skip, not a veto.
        assert report["passed"]

    def test_a_deadline_model_name_is_rejected(self, tmp_path):
        # The deadline variant is derived where it is needed; naming it would
        # double the deadline in every simulation.
        path = make_onnx(tmp_path / "opn.onnx", (1, 5))
        with pytest.raises(ValueError, match="base model"):
            validate_network(
                path, model_name="ddm_deadline", network_type="opn", **AUX_SKIPS
            )

    @pytest.mark.parametrize(
        "network_type, bad, expected",
        [
            ("cpn", "omission", "choice"),
            ("opn", "deadline", "omission"),
            ("gonogo", "choice", "nogo"),
        ],
    )
    def test_a_category_the_type_cannot_output_is_rejected_naming_the_expected(
        self, tmp_path, network_type, bad, expected
    ):
        # "deadline" is the old input-naming vocabulary; a provenance that
        # still carries it must be refused, not recorded in the report.
        path = make_onnx(tmp_path / "aux.onnx", (1, 5))
        with pytest.raises(ValueError, match=f"Unknown --aux-category.*{expected}"):
            validate_network(
                path,
                model_name="ddm",
                network_type=network_type,
                aux_category=bad,
                **AUX_SKIPS,
            )

    def test_a_category_on_a_lan_is_rejected(self, tmp_path):
        path = make_onnx(tmp_path / "lan.onnx", (1, 6))
        with pytest.raises(ValueError, match="--aux-category"):
            validate_network(
                path,
                model_name="ddm",
                aux_category="choice",
                skip_hssm=True,
                skip_density=True,
            )

    def test_a_broken_aux_graph_short_circuits_the_aux_gates(self, tmp_path):
        path = tmp_path / "junk.onnx"
        path.write_bytes(b"nope")
        report = validate_network(path, model_name="ddm", network_type="opn")
        gates = {g["gate"]: g for g in report["gates"]}
        for later in ("parity", "hssm_missing_load", "accuracy"):
            assert gates[later].get("skipped"), later

    def test_cpn_missing_load_waits_for_an_hssm_that_passes_response(
        self, tmp_path, monkeypatch
    ):
        # Under the locked hssm 0.4.0 HSSM calls a CPN with the parameters
        # only, so a contract-conformant width n_params + 1 CPN cannot load.
        # The gate must say so as a skip. Only the below-0.6.0 direction is
        # patched here; the supported direction drives the real gate against
        # a stubbed hssm in TestHssmMissingLoadGate.
        path = make_onnx(tmp_path / "cpn.onnx", (1, 5))
        monkeypatch.setattr(
            vn, "_hssm_version_supports_cpn_response", lambda: (False, "0.4.0")
        )
        report = validate_network(
            path,
            model_name="ddm",
            network_type="cpn",
            aux_category="choice",
            skip_accuracy=True,
        )
        gate = {g["gate"]: g for g in report["gates"]}["hssm_missing_load"]
        assert gate["skipped"] and gate["passed"]
        assert gate["reason"] == "HSSM < 0.6.0 ignores response on missing rows"
        assert gate["hssm_version"] == "0.4.0"

    @pytest.mark.parametrize(
        "installed, supported",
        [("0.5.9", False), ("0.6.0rc1", False), ("0.6.0", True), ("0.7.1", True)],
    )
    def test_the_version_gate_flips_exactly_at_0_6_0(
        self, monkeypatch, installed, supported
    ):
        # Patched rather than read from the environment: CI installs without
        # the validate group, and a real 0.4.0 could not see an off-by-one at
        # the boundary anyway.
        monkeypatch.setattr(importlib.metadata, "version", lambda name: installed)
        assert vn._hssm_version_supports_cpn_response() == (supported, installed)

    def test_validate_network_hands_the_accuracy_flags_and_type_to_the_gate(
        self, tmp_path, monkeypatch
    ):
        # opn, so a hard-coded "cpn" in the wiring would show up in what the
        # truth stub receives.
        path = make_constant_onnx(tmp_path / "half.onnx", 5, np.log(0.5))
        seen = []

        def truth(model_name, network_type, theta, **kwargs):
            seen.append(
                (network_type, sorted(k for k in kwargs if k not in ("n_sim", "rng")))
            )
            return {"truth": 0.5, "truth_mc_se": 0.0016, "n_sim": 100_000}

        monkeypatch.setattr(vn, "aux_truth", truth)
        monkeypatch.setattr(
            vn,
            "_simulate",
            lambda *a, **k: (np.linspace(0.3, 2.3, 10_000), np.ones(10_000)),
        )
        report = validate_network(
            path,
            model_name="ddm",
            network_type="opn",
            skip_hssm=True,
            accuracy_mean_abs_max=0.011,
            accuracy_max_abs_max=0.033,
        )
        gate = {g["gate"]: g for g in report["gates"]}["accuracy"]
        assert not gate.get("skipped")
        assert gate["passed"]
        assert gate["mean_abs_max"] == 0.011 and gate["max_abs_max"] == 0.033
        assert gate["n_param_draws"] == 20
        assert seen and all(t == ("opn", ["deadline"]) for t in seen)


def fixed_simulate(model_name, theta, n_samples, random_state, max_t=20.0):
    """A stand-in simulator: RTs spread over (0.3, 2.3), choices alternating."""
    return (
        np.linspace(0.3, 2.3, n_samples),
        np.where(np.arange(n_samples) % 2 == 0, 1.0, -1.0),
    )


@pytest.fixture
def fake_hssm(monkeypatch):
    """A stub ``hssm`` in sys.modules that records every HSSM(...) call.

    Drives the real gate_hssm_missing_load without the inference stack: what
    the gate hands HSSM (data layout, kwargs) is the contract under test, and
    the logp it gets back is set on the fake's class.
    """

    class Calls(list):
        cls = None

    calls = Calls()

    class FakePyMC:
        def __init__(self, logp):
            self._logp = logp

        def compile_logp(self):
            return lambda point: self._logp

        def initial_point(self):
            return {}

    class FakeHSSM:
        logp = -12.5

        def __init__(self, data, model, p_outlier, **kwargs):
            calls.append(
                {"data": data, "model": model, "p_outlier": p_outlier, **kwargs}
            )
            self.pymc_model = FakePyMC(FakeHSSM.logp)

    hssm = types.ModuleType("hssm")
    hssm.HSSM = FakeHSSM
    modelconfig = types.ModuleType("hssm.modelconfig")
    modelconfig.list_models = lambda: ["ddm"]
    hssm.modelconfig = modelconfig
    monkeypatch.setitem(sys.modules, "hssm", hssm)
    monkeypatch.setitem(sys.modules, "hssm.modelconfig", modelconfig)
    calls.cls = FakeHSSM
    return calls


class TestHssmMissingLoadGate:
    """gate_hssm_missing_load against a stubbed hssm and simulator."""

    def test_cpn_blanks_every_fifth_rt_and_keeps_its_response(
        self, tmp_path, monkeypatch, fake_hssm
    ):
        monkeypatch.setattr(
            vn, "_hssm_version_supports_cpn_response", lambda: (True, "0.6.0")
        )
        monkeypatch.setattr(vn, "_simulate", fixed_simulate)
        path = make_onnx(tmp_path / "cpn.onnx", (1, 5))
        gate = vn.gate_hssm_missing_load(path, "ddm", "cpn", n_trials=100)
        assert gate["passed"] and not gate.get("skipped")
        assert gate["n_missing"] == 20
        assert gate["initial_logp_by_p_outlier"] == {"0.0": -12.5, "0.05": -12.5}
        assert [c["p_outlier"] for c in fake_hssm] == [0.0, 0.05]
        data = fake_hssm[0]["data"]
        assert (data["rt"].iloc[::5] == vn.OMISSION_RT).all()
        mask = np.ones(100, dtype=bool)
        mask[::5] = False
        assert (data["rt"].to_numpy()[mask] > 0).all()
        assert (
            data["response"].tolist() == fixed_simulate("ddm", None, 100, 0)[1].tolist()
        )
        assert fake_hssm[0]["missing_data"] is True
        assert fake_hssm[0]["loglik_missing_data"] == str(path)
        assert "deadline" not in fake_hssm[0]

    def test_opn_places_the_deadline_at_the_base_median_and_simulates_the_variant(
        self, tmp_path, monkeypatch, fake_hssm
    ):
        seen = []

        def simulate(model_name, theta, n_samples, random_state, max_t=20.0):
            seen.append((model_name, np.asarray(theta).tolist()))
            if model_name.endswith("_deadline"):
                rts = np.where(np.arange(n_samples) % 2 == 0, vn.OMISSION_RT, 0.4)
                return rts, np.ones(n_samples)
            return fixed_simulate(model_name, theta, n_samples, random_state)

        monkeypatch.setattr(vn, "_simulate", simulate)
        path = make_onnx(tmp_path / "opn.onnx", (1, 5))
        gate = vn.gate_hssm_missing_load(path, "ddm", "opn", n_trials=100)
        assert gate["passed"]
        assert [m for m, _ in seen] == ["ddm", "ddm_deadline"]
        median = float(np.median(np.linspace(0.3, 2.3, 100)))
        assert seen[1][1] == pytest.approx(seen[0][1] + [median])
        assert gate["n_missing"] == 50
        assert fake_hssm[0]["deadline"] is True
        assert np.allclose(fake_hssm[0]["data"]["deadline"], median)

    def test_by_name_pairs_the_net_with_the_base_lan_not_the_analytical_likelihood(
        self, tmp_path, monkeypatch, fake_hssm
    ):
        # Without loglik_kind HSSM defaults ddm-family models to the analytical
        # likelihood, and the jax LAN + aux-net assembly users hit would never
        # have been exercised while the report claimed a LAN pairing.
        monkeypatch.setattr(vn, "_simulate", fixed_simulate)
        gate = vn.gate_hssm_missing_load(
            make_onnx(tmp_path / "opn.onnx", (1, 5)), "ddm", "opn", n_trials=10
        )
        assert gate["passed"]
        assert gate["lan"] == "ddm (by name)"
        assert fake_hssm[0]["loglik_kind"] == "approx_differentiable"
        assert "loglik" not in fake_hssm[0]
        assert "model_config" not in fake_hssm[0]

    def test_a_lan_path_is_handed_to_hssm_with_the_custom_model_config(
        self, tmp_path, monkeypatch, fake_hssm
    ):
        monkeypatch.setattr(vn, "_simulate", fixed_simulate)
        sys.modules["hssm.modelconfig"].list_models = list
        lan = make_onnx(tmp_path / "ddm.onnx", (1, 6))
        gate = vn.gate_hssm_missing_load(
            make_onnx(tmp_path / "opn.onnx", (1, 5)), "ddm", "opn", lan_onnx=lan
        )
        assert gate["passed"]
        assert gate["lan"] == str(lan)
        call = fake_hssm[0]
        assert call["loglik"] == str(lan)
        assert call["loglik_kind"] == "approx_differentiable"
        assert call["model_config"]["list_params"] == ["v", "a", "z", "t"]

    def test_a_model_outside_the_registry_needs_a_lan_path(
        self, tmp_path, monkeypatch, fake_hssm
    ):
        monkeypatch.setattr(vn, "_simulate", fixed_simulate)
        sys.modules["hssm.modelconfig"].list_models = list
        gate = vn.gate_hssm_missing_load(
            make_onnx(tmp_path / "opn.onnx", (1, 5)), "ddm", "opn"
        )
        assert not gate["passed"]
        assert "--lan-onnx" in gate["error"]
        assert fake_hssm == []

    def test_a_non_finite_logp_fails(self, tmp_path, monkeypatch, fake_hssm):
        monkeypatch.setattr(vn, "_simulate", fixed_simulate)
        fake_hssm.cls.logp = float("nan")
        gate = vn.gate_hssm_missing_load(
            make_onnx(tmp_path / "opn.onnx", (1, 5)), "ddm", "opn"
        )
        assert not gate["passed"]
        assert "error" not in gate

    def test_a_missing_hssm_is_a_failed_gate_not_a_traceback(
        self, tmp_path, monkeypatch
    ):
        # The cpn version probe runs before HSSM is imported; without the
        # validate group it must still land in the JSON as a failure.
        def absent(name):
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "version", absent)
        gate = vn.gate_hssm_missing_load(
            make_onnx(tmp_path / "cpn.onnx", (1, 5)), "ddm", "cpn"
        )
        assert not gate["passed"]
        assert "validate dependency group" in gate["error"]

    def test_gonogo_has_no_consumer_to_load_into(
        self, tmp_path, monkeypatch, fake_hssm
    ):
        monkeypatch.setattr(vn, "_simulate", fixed_simulate)
        gate = vn.gate_hssm_missing_load(
            make_onnx(tmp_path / "gonogo.onnx", (1, 5)), "ddm", "gonogo"
        )
        assert not gate["passed"]
        assert "gonogo" in gate["error"]
        assert fake_hssm == []


class TestAccuracyGate:
    """gate_accuracy against a bias-only graph and a monkeypatched truth.

    cpn needs no simulation once the truth is patched; the opn tests also
    patch _simulate, since the deadline is placed on a base-model simulation.
    """

    @staticmethod
    def constant_truth(p, **extra):
        def fake_truth(model_name, network_type, theta, **kwargs):
            assert network_type == "cpn"
            assert "choice" in kwargs
            return {"truth": p, "truth_mc_se": 0.0016, "n_sim": 100_000, **extra}

        return fake_truth

    def test_passes_when_the_network_matches_the_truth(self, tmp_path, monkeypatch):
        path = make_constant_onnx(tmp_path / "half.onnx", 5, np.log(0.5))
        monkeypatch.setattr(
            vn, "aux_truth", self.constant_truth(0.5, truth_rt_lt_max_t=0.49)
        )
        result = gate_accuracy(path, "ddm", "cpn", n_param_draws=4)
        assert result["passed"], result
        assert result["max_abs_error"] == pytest.approx(0.0, abs=1e-6)
        assert len(result["draws"]) == 4
        # Every record carries what a reader needs to tell network error from
        # simulation noise, plus the cpn-only windowed proportion.
        for draw in result["draws"]:
            assert draw["truth_mc_se"] == 0.0016
            assert draw["truth"] == 0.5
            assert draw["truth_rt_lt_max_t"] == 0.49
            assert draw["network_logp"] == pytest.approx(np.log(0.5), abs=1e-6)
            assert draw["network_value"] == pytest.approx(0.5, abs=1e-6)
            assert draw["abs_error"] == pytest.approx(0.0, abs=1e-6)
            assert len(draw["theta"]) == 4
        # The choice code cycles over the model's declared choices WITHIN each
        # stratum (draws alternate core/edge, so the pattern is -1,-1,+1,+1):
        # each stratum sees every choice — see TestAccuracyStrata.
        assert [d["choice"] for d in result["draws"]] == [-1.0, -1.0, 1.0, 1.0]
        assert [d["stratum"] for d in result["draws"]] == ["core", "edge"] * 2

    def test_fails_when_the_network_is_far_from_the_truth(self, tmp_path, monkeypatch):
        path = make_constant_onnx(tmp_path / "half.onnx", 5, np.log(0.5))
        monkeypatch.setattr(vn, "aux_truth", self.constant_truth(0.9))
        result = gate_accuracy(path, "ddm", "cpn", n_param_draws=4)
        assert not result["passed"]
        assert result["mean_abs_error"] == pytest.approx(0.4, abs=1e-6)
        assert result["mean_abs_max"] == vn.DEFAULT_ACCURACY_MEAN_ABS_MAX
        assert result["max_abs_max"] == vn.DEFAULT_ACCURACY_MAX_ABS_MAX

    def test_one_bad_draw_fails_on_max_even_when_the_mean_is_fine(
        self, tmp_path, monkeypatch
    ):
        # A constant truth cannot separate the two bounds (mean == max), so
        # the truth varies with the cycled choice: -1 exact, +1 off by 0.04.
        path = make_constant_onnx(tmp_path / "half.onnx", 5, np.log(0.5))

        def truth(model_name, network_type, theta, **kwargs):
            p = 0.5 if kwargs["choice"] == -1.0 else 0.54
            return {"truth": p, "truth_mc_se": 0.0016, "n_sim": 100_000}

        monkeypatch.setattr(vn, "aux_truth", truth)
        result = gate_accuracy(
            path, "ddm", "cpn", n_param_draws=4, mean_abs_max=0.05, max_abs_max=0.03
        )
        assert result["mean_abs_error"] == pytest.approx(0.02, abs=1e-6)
        assert result["max_abs_error"] == pytest.approx(0.04, abs=1e-6)
        assert not result["passed"]

    def test_thresholds_are_parameters_not_constants(self, tmp_path, monkeypatch):
        path = make_constant_onnx(tmp_path / "half.onnx", 5, np.log(0.5))
        monkeypatch.setattr(vn, "aux_truth", self.constant_truth(0.52))
        assert not gate_accuracy(path, "ddm", "cpn", n_param_draws=2)["passed"]
        loose = gate_accuracy(
            path, "ddm", "cpn", n_param_draws=2, mean_abs_max=0.05, max_abs_max=0.05
        )
        assert loose["passed"]

    def test_a_positive_output_is_not_a_log_probability(self, tmp_path, monkeypatch):
        # A raw-logit export (no log-sigmoid in the graph) emits values above
        # zero. That is a contract failure before any truth is worth computing.
        path = make_constant_onnx(tmp_path / "logit.onnx", 5, 0.3)
        called = []
        monkeypatch.setattr(
            vn, "aux_truth", lambda *a, **k: called.append(1) or {"truth": 0.5}
        )
        result = gate_accuracy(path, "ddm", "cpn", n_param_draws=4)
        assert not result["passed"]
        assert "finite and <= 0" in result["error"]
        assert called == []

    def test_a_non_finite_output_fails_the_same_way(self, tmp_path, monkeypatch):
        path = make_constant_onnx(tmp_path / "nan.onnx", 5, float("nan"))
        monkeypatch.setattr(vn, "aux_truth", self.constant_truth(0.5))
        result = gate_accuracy(path, "ddm", "cpn", n_param_draws=1)
        assert not result["passed"]
        assert "finite and <= 0" in result["error"]

    def test_a_certain_prediction_of_exactly_zero_logp_is_accepted(
        self, tmp_path, monkeypatch
    ):
        # A confident float32 log-sigmoid rounds to exactly 0.0 (P = 1); the
        # contract is <= 0, and a strict < 0 would reject such a network.
        path = make_constant_onnx(tmp_path / "one.onnx", 5, 0.0)
        monkeypatch.setattr(vn, "aux_truth", self.constant_truth(1.0))
        result = gate_accuracy(path, "ddm", "cpn", n_param_draws=2)
        assert result["passed"], result
        assert "error" not in result

    def test_opn_deadline_is_clipped_to_ssms_bounds_and_shared_with_the_truth(
        self, tmp_path, monkeypatch
    ):
        path = make_constant_onnx(tmp_path / "half.onnx", 5, np.log(0.5))
        asked = []

        def truth(model_name, network_type, theta, **kwargs):
            asked.append(kwargs["deadline"])
            return {"truth": 0.5, "truth_mc_se": 0.0016, "n_sim": 100_000}

        monkeypatch.setattr(vn, "aux_truth", truth)
        # Every base RT past the deadline ceiling: the quantile must be clipped.
        monkeypatch.setattr(
            vn, "_simulate", lambda *a, **k: (np.full(10_000, 50.0), np.ones(10_000))
        )
        result = gate_accuracy(path, "ddm", "opn", n_param_draws=3)
        ceiling = vn.DEADLINE_BOUNDS[1]
        assert [d["deadline"] for d in result["draws"]] == [ceiling] * 3
        assert asked == [ceiling] * 3
        assert all("choice" not in d for d in result["draws"])

    def test_opn_deadline_sits_inside_the_base_rt_spread(self, tmp_path, monkeypatch):
        path = make_constant_onnx(tmp_path / "half.onnx", 5, np.log(0.5))
        monkeypatch.setattr(
            vn,
            "aux_truth",
            lambda *a, **k: {"truth": 0.5, "truth_mc_se": 0.0016, "n_sim": 100_000},
        )
        monkeypatch.setattr(
            vn,
            "_simulate",
            lambda *a, **k: (np.linspace(0.3, 2.3, 10_000), np.ones(10_000)),
        )
        result = gate_accuracy(path, "ddm", "opn", n_param_draws=5)
        # The quantile at u ~ U(0.1, 0.9) of RTs spread over (0.3, 2.3).
        for d in result["draws"]:
            assert 0.5 <= d["deadline"] <= 2.1


class TestAuxTruth:
    def test_truth_carries_its_monte_carlo_se(self, monkeypatch):
        # A stand-in simulator: 30% of trials pick +1, none run past max_t.
        def fake_simulate(model_name, theta, n_samples, random_state, max_t=20.0):
            choices = np.where(np.arange(n_samples) % 10 < 3, 1, -1)
            return np.full(n_samples, 0.5), choices

        monkeypatch.setattr(vn, "_simulate", fake_simulate)
        truth = vn.aux_truth("ddm", "cpn", np.zeros(4), choice=1, n_sim=1000)
        assert truth["truth"] == pytest.approx(0.3)
        assert truth["truth_mc_se"] == pytest.approx(np.sqrt(0.3 * 0.7 / 1000))
        assert truth["truth_rt_lt_max_t"] == pytest.approx(0.3)

    def test_cpn_truth_separates_trials_that_outlast_max_t(self, monkeypatch):
        # ssms hands an un-terminated trial back at rt ≈ max_t + t with a
        # sign-implied choice; the gate's truth keeps it, the windowed one
        # does not, and both are reported.
        def fake_simulate(model_name, theta, n_samples, random_state, max_t=20.0):
            rts = np.where(np.arange(n_samples) % 2 == 0, 0.5, 20.4)
            return rts, np.ones(n_samples)

        monkeypatch.setattr(vn, "_simulate", fake_simulate)
        truth = vn.aux_truth("ddm", "cpn", np.zeros(4), choice=1, n_sim=100)
        assert truth["truth"] == pytest.approx(1.0)
        assert truth["truth_rt_lt_max_t"] == pytest.approx(0.5)

    def test_opn_truth_is_the_omission_rate_of_the_deadline_variant(self, monkeypatch):
        seen = {}

        def fake_simulate(model_name, theta, n_samples, random_state, max_t=20.0):
            seen["model"], seen["theta"] = model_name, np.asarray(theta)
            rts = np.where(np.arange(n_samples) % 4 == 0, -999.0, 0.7)
            # An omission is marked in rts only; choices stay sign-implied.
            return rts, np.ones(n_samples)

        monkeypatch.setattr(vn, "_simulate", fake_simulate)
        truth = vn.aux_truth("ddm", "opn", np.arange(4.0), deadline=1.5, n_sim=400)
        assert seen["model"] == "ddm_deadline"
        assert seen["theta"].tolist() == [0.0, 1.0, 2.0, 3.0, 1.5]
        assert truth["truth"] == pytest.approx(0.25)
        assert "truth_rt_lt_max_t" not in truth

    def test_gonogo_has_no_truth(self):
        with pytest.raises(ValueError, match="gonogo"):
            vn.aux_truth("ddm", "gonogo", np.zeros(4), deadline=1.0)


def canned_survey(p99=0.02, frac_gt_0_05=0.0, frac_gt_0_10=0.0, seconds=41.5):
    """A survey dict in the shape LANfactory L1's ``survey`` returns."""
    total = {
        "mean": 1.001,
        "p50_abs_dev": 0.0037,
        "p90_abs_dev": 0.015,
        "p99_abs_dev": p99,
        "min": 0.95,
        "max": 1.05,
        "frac_gt_0.02": 0.1,
        "frac_gt_0.05": frac_gt_0_05,
        "frac_gt_0.10": frac_gt_0_10,
    }
    return {
        "n_theta": 20_000,
        "grid": {"kind": "onset", "n_points": 1000},
        "seconds": seconds,
        "total": total,
        "shrunk_box": {**total, "frac_of_theta": 0.41},
        "leak_below_onset": {"mean": 1e-4, "p99": 1e-3, "max": 2e-3},
        "by_param": {
            "a": [
                {"lo": 0.3, "hi": 0.52, "mean_dev": 0.001, "max_abs_dev": 0.02, "n": 2}
            ]
        },
        "worst_cell": {"a": [2.2, 2.5], "v": [-0.5, 0.5], "mean_dev": 0.2},
    }


class FakeDerive:
    """State behind the ``lanfactory.derive`` stub: what it returns, what it saw."""

    def __init__(self):
        self.survey_result = canned_survey()
        self.survey_error = None
        self.total_mass = 0.97
        self.calls = []


@pytest.fixture
def fake_derive(monkeypatch):
    """A stub ``lanfactory.derive`` in sys.modules.

    A stub, not a real LAN, so the mass-survey gate's verdicts and the
    accuracy gate's total_mass attribution are cheap and exact to drive.
    ``survey`` returns whatever ``survey_result`` holds and records its
    arguments; ``choice_mass`` returns ``total_mass`` for every θ.
    """
    state = FakeDerive()

    class Predictor:
        input_width = 6

        def __init__(self, path):
            self.path = str(path)

    class Mass:
        def __init__(self, total):
            self.total = np.asarray([total])

    def load_onnx_predictor(path):
        state.calls.append(("load", str(path)))
        return Predictor(path)

    def survey(predictor, param_bounds, params, choices, **kwargs):
        # L1: param_bounds is {name: (lo, hi)} and must cover every param;
        # ssms' positional [lows, highs] raises "param_bounds lacks bounds".
        assert isinstance(param_bounds, dict), type(param_bounds)
        assert set(param_bounds) == set(params), (param_bounds, params)
        assert all(len(b) == 2 and b[0] <= b[1] for b in param_bounds.values())
        state.calls.append(("survey", predictor, param_bounds, params, choices, kwargs))
        if state.survey_error is not None:
            raise state.survey_error
        return state.survey_result

    def choice_mass(predictor, theta, choices, onset, grid=None):
        # L1: onset is required (0 for a model without one), must be finite
        # and must broadcast to (n_theta,); the grid defaults to OnsetGrid().
        theta = np.asarray(theta)
        assert theta.ndim == 2, theta.shape
        onset = np.broadcast_to(np.asarray(onset, dtype=float), (len(theta),))
        if not np.all(np.isfinite(onset)):
            raise ValueError("onset must be finite")
        state.calls.append(("mass", theta[0].tolist(), choices, onset.tolist()))
        return Mass(state.total_mass)

    derive = types.ModuleType("lanfactory.derive")
    derive.load_onnx_predictor = load_onnx_predictor
    derive.survey = survey
    derive.choice_mass = choice_mass
    lanfactory = types.ModuleType("lanfactory")
    lanfactory.derive = derive
    monkeypatch.setitem(sys.modules, "lanfactory", lanfactory)
    monkeypatch.setitem(sys.modules, "lanfactory.derive", derive)
    return state


@pytest.fixture
def no_derive(monkeypatch):
    """Force ``from lanfactory.derive import ...`` to raise ImportError.

    A None entry in sys.modules makes the import fail deterministically, so
    the skip path is pinned regardless of which LANfactory the lock holds —
    the same way the cpn A3 tests force HSSM's version rather than read it.
    """
    monkeypatch.setitem(sys.modules, "lanfactory.derive", None)


DDM_CONFIG = {
    "params": ["v", "a", "z", "t"],
    "param_bounds": [[-3.0, 0.3, 0.1, 0.0], [3.0, 2.5, 0.9, 2.0]],
    "choices": [-1, 1],
}
DDM_ST_CONFIG = {
    "params": ["v", "a", "z", "t", "st"],
    "param_bounds": [[-3.0, 0.3, 0.1, 0.25, 0.0], [3.0, 2.5, 0.9, 2.25, 0.25]],
    "choices": [-1, 1],
    "ndt_edge_shift": {"param": "st", "scale": 1.0},
}
# What L1's survey() wants: ssms' param_bounds_dict, keyed by name.
DDM_BOUNDS = {"v": (-3.0, 3.0), "a": (0.3, 2.5), "z": (0.1, 0.9), "t": (0.0, 2.0)}


class TestMassSurveyGate:
    """gate_mass_survey against the stubbed lanfactory.derive."""

    def test_skips_with_the_lock_reason_when_derive_is_absent(
        self, tmp_path, no_derive
    ):
        # The import failure is forced, not read from the environment: this
        # pins the skip path after the lock moves too.
        gate = vn.gate_mass_survey(make_onnx(tmp_path / "lan.onnx", (1, 6)), DDM_CONFIG)
        assert gate["skipped"] and gate["passed"]
        assert gate["reason"] == vn.LANFACTORY_DERIVE_SKIP_REASON
        assert gate["reason"] == (
            "lanfactory.derive not available; this environment's LANfactory predates it"
        )
        assert "survey" not in gate

    def test_the_locked_lanfactory_has_what_the_gates_import(self):
        # The ONE environment-dependent test, by design: the lock must pin a
        # LANfactory whose derive package has the names the gates import, or
        # the survey and the accuracy draws' LAN mass skip unnoticed.
        from lanfactory.derive import (  # noqa: F401
            choice_mass,
            load_onnx_predictor,
            survey,
        )

    def test_hands_the_model_space_and_onset_param_to_the_survey(
        self, tmp_path, fake_derive
    ):
        path = make_onnx(tmp_path / "lan.onnx", (1, 6))
        gate = vn.gate_mass_survey(path, DDM_CONFIG, n_theta=500, seed=7)
        assert gate["passed"] and not gate.get("skipped")
        assert fake_derive.calls[0] == ("load", str(path))
        _, predictor, bounds, params, choices, kwargs = fake_derive.calls[1]
        assert predictor.path == str(path)
        # By name, as survey() requires — NOT ssms' positional [lows, highs].
        assert bounds == DDM_BOUNDS
        assert params == ["v", "a", "z", "t"]
        assert choices == [-1, 1]
        assert kwargs == {
            "n_theta": 500,
            "seed": 7,
            "onset_param": "t",
            "ndt_edge_shift": None,
        }

    def test_hands_an_st_models_edge_shift_to_the_survey(self, tmp_path, fake_derive):
        # ddm_st's rts start at t - st; without the shift the survey reports
        # the model's own early mass as leak and misplaces its grid.
        vn.gate_mass_survey(make_onnx(tmp_path / "lan.onnx", (1, 7)), DDM_ST_CONFIG)
        kwargs = fake_derive.calls[1][-1]
        assert kwargs["ndt_edge_shift"] == {"param": "st", "scale": 1.0}

    def test_a_model_without_t_gets_no_onset_param(self, tmp_path, fake_derive):
        config = {
            "params": ["v", "a", "z"],
            "param_bounds": [[-3.0, 0.3, 0.1], [3.0, 2.5, 0.9]],
            "choices": [-1, 1],
        }
        vn.gate_mass_survey(make_onnx(tmp_path / "lan.onnx", (1, 5)), config)
        assert fake_derive.calls[1][-1]["onset_param"] is None

    def test_records_the_whole_survey_and_its_seconds(self, tmp_path, fake_derive):
        gate = vn.gate_mass_survey(make_onnx(tmp_path / "lan.onnx", (1, 6)), DDM_CONFIG)
        assert gate["survey"] == canned_survey()
        assert gate["seconds"] == 41.5
        assert gate["n_theta"] == 20_000
        assert gate["verdict"] == "pass"
        assert gate["p99_abs_dev"] == 0.02
        assert gate["frac_gt_0.10"] == 0.0 and gate["frac_gt_0.05"] == 0.0
        assert gate["p99_max"] == vn.DEFAULT_MASS_SURVEY_P99_MAX == 0.10
        assert gate["frac_gt_0_10_max"] == vn.DEFAULT_MASS_SURVEY_FRAC_GT_0_10_MAX
        assert gate["frac_gt_0_10_max"] == 0.02
        assert "warning" not in gate and "error" not in gate

    @pytest.mark.parametrize(
        "p99, frac_05, expected_in_warning",
        [
            # The Hub ddm LAN itself: p99 0.080, 1.9 % beyond 0.05 — a WARN.
            (0.080, 0.019, "total.p99_abs_dev 0.0800 > 0.05"),
            (0.051, 0.0, "total.p99_abs_dev 0.0510 > 0.05"),
            (0.02, 0.011, "total.frac_gt_0.05 0.0110 > 0.01"),
        ],
    )
    def test_warns_but_passes_past_the_warn_lines(
        self, tmp_path, fake_derive, p99, frac_05, expected_in_warning
    ):
        fake_derive.survey_result = canned_survey(p99=p99, frac_gt_0_05=frac_05)
        gate = vn.gate_mass_survey(make_onnx(tmp_path / "lan.onnx", (1, 6)), DDM_CONFIG)
        assert gate["passed"] and not gate.get("skipped")
        assert gate["verdict"] == "warn"
        assert expected_in_warning in gate["warning"]
        assert "error" not in gate

    @pytest.mark.parametrize(
        "p99, frac_10, expected_in_error",
        [
            (0.101, 0.0, "total.p99_abs_dev 0.1010 > 0.1"),
            (0.02, 0.021, "total.frac_gt_0.10 0.0210 > 0.02"),
            (0.3, 0.1, "total.p99_abs_dev 0.3000 > 0.1; total.frac_gt_0.10 0.1000"),
        ],
    )
    def test_fails_past_the_fail_lines(
        self, tmp_path, fake_derive, p99, frac_10, expected_in_error
    ):
        fake_derive.survey_result = canned_survey(
            p99=p99, frac_gt_0_05=0.05, frac_gt_0_10=frac_10
        )
        gate = vn.gate_mass_survey(make_onnx(tmp_path / "lan.onnx", (1, 6)), DDM_CONFIG)
        assert not gate["passed"] and not gate.get("skipped")
        assert gate["verdict"] == "fail"
        assert expected_in_error in gate["error"]
        # The survey is still recorded on a failure: that is what says where.
        assert gate["survey"]["total"]["p99_abs_dev"] == p99

    def test_exactly_at_a_fail_line_is_not_past_it(self, tmp_path, fake_derive):
        fake_derive.survey_result = canned_survey(
            p99=0.10, frac_gt_0_05=0.01, frac_gt_0_10=0.02
        )
        gate = vn.gate_mass_survey(make_onnx(tmp_path / "lan.onnx", (1, 6)), DDM_CONFIG)
        assert gate["passed"]
        # Past the warn line on p99 (0.05) but not past the fail lines.
        assert gate["verdict"] == "warn"
        assert "error" not in gate

    def test_exactly_at_a_warn_line_is_not_past_it(self, tmp_path, fake_derive):
        # Both warn comparisons are strict: p99 == 0.05 and frac_gt_0.05 ==
        # 0.01 together are a clean pass with no warning at all.
        fake_derive.survey_result = canned_survey(p99=0.05, frac_gt_0_05=0.01)
        gate = vn.gate_mass_survey(make_onnx(tmp_path / "lan.onnx", (1, 6)), DDM_CONFIG)
        assert gate["passed"]
        assert gate["verdict"] == "pass"
        assert "warning" not in gate and "error" not in gate

    def test_thresholds_are_parameters_not_constants(self, tmp_path, fake_derive):
        fake_derive.survey_result = canned_survey(p99=0.12, frac_gt_0_10=0.03)
        path = make_onnx(tmp_path / "lan.onnx", (1, 6))
        assert not vn.gate_mass_survey(path, DDM_CONFIG)["passed"]
        loose = vn.gate_mass_survey(
            path, DDM_CONFIG, p99_max=0.2, frac_gt_0_10_max=0.05
        )
        assert loose["passed"]
        assert loose["p99_max"] == 0.2 and loose["frac_gt_0_10_max"] == 0.05

    def test_a_survey_that_raises_is_a_failed_gate_not_a_traceback(
        self, tmp_path, fake_derive
    ):
        fake_derive.survey_error = ValueError("predictor expects rows of width 6")
        gate = vn.gate_mass_survey(make_onnx(tmp_path / "lan.onnx", (1, 6)), DDM_CONFIG)
        assert not gate["passed"] and not gate.get("skipped")
        assert gate["error"] == "ValueError: predictor expects rows of width 6"

    def test_an_unresolved_parameter_space_fails_rather_than_surveying_nothing(
        self, tmp_path, fake_derive
    ):
        gate = vn.gate_mass_survey(make_onnx(tmp_path / "lan.onnx", (1, 6)), None)
        assert not gate["passed"]
        assert "parameter space" in gate["error"]
        assert fake_derive.calls == []


class TestMassSurveyWiring:
    def test_a_lan_report_runs_the_survey_after_density(self, tmp_path, fake_derive):
        fake_derive.survey_result = canned_survey(p99=0.080, frac_gt_0_05=0.019)
        report = validate_network(
            make_onnx(tmp_path / "lan.onnx", (1, 6)),
            model_name="ddm",
            skip_hssm=True,
            skip_density=True,
        )
        assert [g["gate"] for g in report["gates"]][-2:] == ["density", "mass_survey"]
        gate = report["gates"][-1]
        assert gate["verdict"] == "warn" and gate["passed"]
        assert report["passed"]
        # The real ssms config for ddm went to the survey, t included.
        _, _, bounds, params, choices, kwargs = fake_derive.calls[1]
        assert params == ["v", "a", "z", "t"]
        assert kwargs["onset_param"] == "t"
        assert kwargs["n_theta"] == 20_000
        assert bounds == DDM_BOUNDS and list(choices) == [-1, 1]

    def test_a_failing_survey_fails_the_report(self, tmp_path, fake_derive):
        fake_derive.survey_result = canned_survey(p99=0.2, frac_gt_0_10=0.1)
        report = validate_network(
            make_onnx(tmp_path / "lan.onnx", (1, 6)),
            model_name="ddm",
            skip_hssm=True,
            skip_density=True,
        )
        assert not report["passed"]
        assert not report["gates"][-1]["passed"]

    def test_the_flags_reach_the_gate(self, tmp_path, fake_derive):
        fake_derive.survey_result = canned_survey(p99=0.12, frac_gt_0_10=0.03)
        report = validate_network(
            make_onnx(tmp_path / "lan.onnx", (1, 6)),
            model_name="ddm",
            skip_hssm=True,
            skip_density=True,
            mass_survey_p99_max=0.15,
            mass_survey_frac_gt_0_10_max=0.04,
        )
        gate = report["gates"][-1]
        assert gate["passed"]
        assert gate["p99_max"] == 0.15 and gate["frac_gt_0_10_max"] == 0.04

    def test_skip_mass_survey_is_reported_as_a_skip(self, tmp_path, fake_derive):
        report = validate_network(
            make_onnx(tmp_path / "lan.onnx", (1, 6)),
            model_name="ddm",
            skip_hssm=True,
            skip_density=True,
            skip_mass_survey=True,
        )
        gate = report["gates"][-1]
        assert gate["skipped"] and gate["passed"]
        assert gate["reason"] == "--skip-mass-survey"
        # The skip must actually take effect: nothing was loaded or surveyed.
        assert fake_derive.calls == []

    def test_aux_reports_do_not_carry_the_survey(self, tmp_path, fake_derive):
        report = validate_network(
            make_onnx(tmp_path / "opn.onnx", (1, 5)),
            model_name="ddm",
            network_type="opn",
            **AUX_SKIPS,
        )
        assert "mass_survey" not in {g["gate"] for g in report["gates"]}
        assert fake_derive.calls == []

    def test_cli_flags_are_spelled_as_documented(self, tmp_path, fake_derive):
        from typer.testing import CliRunner

        fake_derive.survey_result = canned_survey(p99=0.12, frac_gt_0_10=0.03)
        path = make_onnx(tmp_path / "lan.onnx", (1, 6))
        result = CliRunner().invoke(
            vn.app,
            [
                "--onnx-path",
                str(path),
                "--model-name",
                "ddm",
                "--skip-hssm",
                "--skip-density",
                "--mass-survey-p99-max",
                "0.15",
                "--mass-survey-frac-gt-0.10-max",
                "0.04",
            ],
        )
        assert result.exit_code == 0, result.output
        line = json.loads(result.output.strip().splitlines()[-1])
        assert line["gates"]["mass_survey"] == "passed"
        report = json.loads((tmp_path / "validation_report.json").read_text())
        assert report["gates"][-1]["p99_max"] == 0.15
        assert report["gates"][-1]["frac_gt_0_10_max"] == 0.04

        result = CliRunner().invoke(
            vn.app,
            ["--onnx-path", str(path), "--model-name", "ddm", "--skip-hssm"]
            + ["--skip-density", "--skip-mass-survey"],
        )
        assert result.exit_code == 0, result.output
        line = json.loads(result.output.strip().splitlines()[-1])
        assert line["gates"]["mass_survey"] == "skipped"


class TestAccuracyStrata:
    """The accuracy draws cover the full box, half of them at its edge."""

    def test_twenty_draws_are_ten_core_and_ten_edge(self, tmp_path, monkeypatch):
        path = make_constant_onnx(tmp_path / "half.onnx", 5, np.log(0.5))
        monkeypatch.setattr(vn, "aux_truth", TestAccuracyGate.constant_truth(0.5))
        result = gate_accuracy(path, "ddm", "cpn")
        assert result["passed"], result
        assert result["n_param_draws"] == 20
        assert result["n_core_draws"] == 10 and result["n_edge_draws"] == 10
        assert result["shrink"] == 0.1
        strata = [d["stratum"] for d in result["draws"]]
        assert strata == ["core", "edge"] * 10
        # Each stratum sees every declared choice. With two choices and two
        # strata, cycling the choice on the draw index would alias the two
        # (-1 only in the core, +1 only at the edge) and a cpn wrong on the
        # +1 output in the core would pass.
        for stratum in ("core", "edge"):
            choices = [d["choice"] for d in result["draws"] if d["stratum"] == stratum]
            assert choices == [-1.0, 1.0] * 5, stratum

    def test_edge_draws_lie_outside_the_shrunk_box_and_core_draws_inside(
        self, tmp_path, monkeypatch
    ):
        import ssms

        path = make_constant_onnx(tmp_path / "half.onnx", 5, np.log(0.5))
        monkeypatch.setattr(vn, "aux_truth", TestAccuracyGate.constant_truth(0.5))
        config = ssms.config.model_config["ddm"]
        lower, upper = (np.asarray(b, dtype=float) for b in config["param_bounds"])
        result = gate_accuracy(path, "ddm", "cpn", n_param_draws=40, seed=3)
        for d in result["draws"]:
            theta = np.asarray(d["theta"])
            assert np.all(theta >= lower) and np.all(theta <= upper)
            inside = vn._inside_shrunk_box(theta, config, 0.1)
            assert inside == (d["stratum"] == "core"), d

    def test_total_mass_is_none_without_lanfactory_derive(
        self, tmp_path, monkeypatch, no_derive
    ):
        path = make_constant_onnx(tmp_path / "half.onnx", 5, np.log(0.5))
        monkeypatch.setattr(vn, "aux_truth", TestAccuracyGate.constant_truth(0.5))
        lan = make_onnx(tmp_path / "ddm.onnx", (1, 6))
        result = gate_accuracy(path, "ddm", "cpn", n_param_draws=4, lan_onnx=lan)
        assert result["passed"]
        assert [d["total_mass"] for d in result["draws"]] == [None] * 4

    def test_total_mass_is_the_lans_mass_at_each_theta_on_the_onset_grid(
        self, tmp_path, monkeypatch, fake_derive
    ):
        path = make_constant_onnx(tmp_path / "half.onnx", 5, np.log(0.5))
        monkeypatch.setattr(vn, "aux_truth", TestAccuracyGate.constant_truth(0.5))
        lan = make_onnx(tmp_path / "ddm.onnx", (1, 6))
        result = gate_accuracy(path, "ddm", "cpn", n_param_draws=4, lan_onnx=lan)
        assert result["passed"]
        assert [d["total_mass"] for d in result["draws"]] == [0.97] * 4
        assert fake_derive.calls[0] == ("load", str(lan))
        masses = [c for c in fake_derive.calls if c[0] == "mass"]
        assert [m[1] for m in masses] == [d["theta"] for d in result["draws"]]
        # ddm's t is its fourth parameter: the onset handed to choice_mass is
        # that draw's t, as onset=theta[:, 3].
        assert [m[3] for m in masses] == [[d["theta"][3]] for d in result["draws"]]
        assert all(list(m[2]) == [-1, 1] for m in masses)

    def test_a_model_without_t_integrates_from_onset_zero(self, tmp_path, fake_derive):
        config = {
            "params": ["v", "a", "z"],
            "param_bounds": [[-3.0, 0.3, 0.1], [3.0, 2.5, 0.9]],
            "choices": [-1, 1],
        }
        total = vn._lan_total_mass(make_onnx(tmp_path / "lan.onnx", (1, 5)), config)
        assert total(np.array([0.5, 1.0, 0.5])) == 0.97
        (_, theta, _, onset) = next(c for c in fake_derive.calls if c[0] == "mass")
        assert onset == [0.0]
        assert theta == [0.5, 1.0, 0.5]

    def test_an_st_model_integrates_from_t_minus_st(self, tmp_path, fake_derive):
        lan = make_onnx(tmp_path / "lan.onnx", (1, 7))
        total = vn._lan_total_mass(lan, DDM_ST_CONFIG)
        assert total(np.array([0.5, 1.0, 0.5, 0.8, 0.2])) == 0.97
        (_, _, _, onset) = next(c for c in fake_derive.calls if c[0] == "mass")
        assert onset == pytest.approx([0.6])

    def test_total_mass_needs_a_lan_to_evaluate(
        self, tmp_path, monkeypatch, fake_derive
    ):
        path = make_constant_onnx(tmp_path / "half.onnx", 5, np.log(0.5))
        monkeypatch.setattr(vn, "aux_truth", TestAccuracyGate.constant_truth(0.5))
        result = gate_accuracy(path, "ddm", "cpn", n_param_draws=2)
        assert [d["total_mass"] for d in result["draws"]] == [None, None]
        assert fake_derive.calls == []

    def test_validate_network_hands_the_lan_path_to_the_accuracy_gate(
        self, tmp_path, monkeypatch, fake_derive
    ):
        path = make_constant_onnx(tmp_path / "half.onnx", 5, np.log(0.5))
        monkeypatch.setattr(
            vn,
            "aux_truth",
            lambda *a, **k: {"truth": 0.5, "truth_mc_se": 0.0016, "n_sim": 100_000},
        )
        lan = make_onnx(tmp_path / "ddm.onnx", (1, 6))
        report = validate_network(
            path, model_name="ddm", network_type="cpn", skip_hssm=True, lan_onnx=lan
        )
        gate = {g["gate"]: g for g in report["gates"]}["accuracy"]
        assert gate["passed"]
        assert all(d["total_mass"] == 0.97 for d in gate["draws"])
        assert {d["stratum"] for d in gate["draws"]} == {"core", "edge"}

    def test_an_edge_draw_needs_a_margin(self):
        with pytest.raises(ValueError, match="shrink > 0"):
            vn._draw_theta_edge(DDM_CONFIG, np.random.default_rng(0), shrink=0.0)


def fake_simulator(rts_for_call):
    """An ssms simulator stand-in. ``rts_for_call(i, theta, n)`` gives the rts
    of the i-th call; choices alternate. Records each call's theta."""
    calls = []

    def simulator(model, theta, n_samples, random_state, **kwargs):
        theta = np.asarray(theta, dtype=float)
        calls.append({"theta": theta, "random_state": random_state})
        rts = np.asarray(rts_for_call(len(calls) - 1, theta, n_samples), dtype=float)
        choices = np.where(np.arange(n_samples) % 2 == 0, 1, -1)
        return {"rts": rts.reshape(-1, 1), "choices": choices.reshape(-1, 1)}

    simulator.calls = calls
    return simulator


@pytest.fixture
def fake_hssm_g3(monkeypatch):
    """A stand-in for the hssm package with HSSM's data check and nothing else.

    hssm.HSSM raises on any rt <= 0 before it touches the likelihood, so that
    check is the whole of what G3's data handling has to satisfy. Records the
    frames it was handed.
    """
    seen = []

    class FakePymcModel:
        def initial_point(self):
            return {}

        def compile_logp(self):
            return lambda point: -123.0

    class HSSM:
        def __init__(self, data, model, loglik, loglik_kind, **kwargs):
            seen.append(data.copy())
            if (data["rt"] <= 0).any():
                raise ValueError("Data contains negative response times.")
            self.pymc_model = FakePymcModel()

    module = types.ModuleType("hssm")
    module.HSSM = HSSM
    modelconfig = types.ModuleType("hssm.modelconfig")
    modelconfig.list_models = lambda: ["ddm"]
    module.modelconfig = modelconfig
    monkeypatch.setitem(sys.modules, "hssm", module)
    monkeypatch.setitem(sys.modules, "hssm.modelconfig", modelconfig)
    return seen


def ddm_box():
    import ssms

    lower, upper = (
        np.asarray(b, dtype=float)
        for b in ssms.config.model_config["ddm"]["param_bounds"]
    )
    return lower, upper


class TestHssmLoadGate:
    """G3 must not fail a working network because its data were unusable.

    hssm.HSSM refuses data with rt <= 0. An untruncated non-decision-time
    kernel (ddm_normal_st) produces such trials whenever st is large next to t,
    and the full ssms box allows exactly that; the gate used to draw there.
    """

    def test_redraw_constants_match_recovery(self):
        import recovery_designs as rd

        assert vn.HSSM_LOAD_MAX_REDRAWS == rd.MAX_TRUTH_REDRAWS
        assert vn.HSSM_LOAD_REDRAW_STRIDE == rd.REDRAW_STRIDE

    def test_theta_is_drawn_from_the_inset_box(self, monkeypatch):
        sim = fake_simulator(lambda i, theta, n: np.full(n, 0.5))
        monkeypatch.setattr("ssms.basic_simulators.simulator.simulator", sim)
        lower, upper = ddm_box()
        span = upper - lower
        for seed in range(25):
            hssm_load_data("ddm", n_trials=10, seed=seed)
        for call in sim.calls:
            assert np.all(call["theta"] >= lower + 0.1 * span - 1e-12)
            assert np.all(call["theta"] <= upper - 0.1 * span + 1e-12)

    def test_a_draw_with_nonpositive_rts_is_redrawn_and_recorded(self, monkeypatch):
        sim = fake_simulator(
            lambda i, theta, n: np.linspace(-0.2, 1.0, n) if i == 0 else np.full(n, 0.5)
        )
        monkeypatch.setattr("ssms.basic_simulators.simulator.simulator", sim)
        draw = hssm_load_data("ddm", n_trials=100, seed=0)
        assert draw["attempts"] == 2
        assert (draw["data"]["rt"] > 0).all()
        assert len(draw["rejected"]) == 1
        assert draw["rejected"][0]["seed"] == 0
        assert draw["rejected"][0]["n_nonpositive_rt"] > 0
        # Attempt 0 keeps the seed unchanged; the redraw steps by the stride.
        assert [c["random_state"] for c in sim.calls] == [
            0,
            vn.HSSM_LOAD_REDRAW_STRIDE,
        ]
        assert draw["seed"] == vn.HSSM_LOAD_REDRAW_STRIDE

    def test_gate_passes_a_network_whose_first_draw_was_unusable(
        self, tmp_path, monkeypatch, fake_hssm_g3
    ):
        # The regression: before the fix the negative-rt frame went straight
        # to hssm.HSSM and the gate failed with "negative response times".
        sim = fake_simulator(
            lambda i, theta, n: np.full(n, -0.05) if i == 0 else np.full(n, 0.5)
        )
        monkeypatch.setattr("ssms.basic_simulators.simulator.simulator", sim)
        result = gate_hssm_load(tmp_path / "m.onnx", "ddm")
        assert result["passed"], result
        assert result["attempts"] == 2
        assert len(result["rejected_draws"]) == 1
        assert len(fake_hssm_g3) == 1 and (fake_hssm_g3[0]["rt"] > 0).all()

    def test_an_exhausted_budget_skips_rather_than_fails(
        self, tmp_path, monkeypatch, fake_hssm_g3
    ):
        # The gate's own message calls this "a fact about the box, not the
        # network", so it must not report a network failure. It skips, the
        # way parity does, and skipped gates report passed=True.
        sim = fake_simulator(lambda i, theta, n: np.full(n, -0.05))
        monkeypatch.setattr("ssms.basic_simulators.simulator.simulator", sim)
        result = gate_hssm_load(tmp_path / "m.onnx", "ddm")
        assert result["passed"], result
        assert result.get("skipped") is True, result
        assert "box" in result["reason"]
        assert "error" not in result
        # The evidence is still in the report.
        assert result["attempts"] == vn.HSSM_LOAD_MAX_REDRAWS + 1
        assert len(result["rejected_draws"]) == vn.HSSM_LOAD_MAX_REDRAWS + 1
        assert fake_hssm_g3 == []

    def test_an_exhausted_budget_still_blocks_publish(
        self, tmp_path, monkeypatch, fake_hssm_g3
    ):
        # Skipping is not passing: G3 is a required gate, so a network HSSM
        # never loaded cannot be published on the strength of a skipped G3.
        from publish.publish_network import gate_verdict

        sim = fake_simulator(lambda i, theta, n: np.full(n, -0.05))
        monkeypatch.setattr("ssms.basic_simulators.simulator.simulator", sim)
        others = [
            {"gate": name, "passed": True}
            for name in ("structure", "parity", "density")
        ]
        report = {"gates": [*others, gate_hssm_load(tmp_path / "m.onnx", "ddm")]}
        assert all(g["passed"] for g in report["gates"])

        ok, reason = gate_verdict(report)
        assert not ok
        assert "hssm_load" in reason and "not actually checked" in reason

    def test_an_hssm_failure_reports_the_draw_it_was_handed(
        self, tmp_path, monkeypatch, fake_hssm_g3
    ):
        sim = fake_simulator(
            lambda i, theta, n: np.full(n, -0.05) if i == 0 else np.full(n, 0.5)
        )
        monkeypatch.setattr("ssms.basic_simulators.simulator.simulator", sim)

        def broken_hssm(**kwargs):
            raise RuntimeError("compile failed")

        monkeypatch.setattr(sys.modules["hssm"], "HSSM", broken_hssm)
        result = gate_hssm_load(tmp_path / "m.onnx", "ddm", seed=3)
        assert not result["passed"]
        assert result["error"] == "RuntimeError: compile failed"
        # The failing integration check names the accepted draw, not just
        # the rejected one before it.
        assert result["attempts"] == 2
        assert result["seed"] == 3 + vn.HSSM_LOAD_REDRAW_STRIDE
        assert result["theta"] == pytest.approx(sim.calls[1]["theta"].tolist())
        assert [d["seed"] for d in result["rejected_draws"]] == [3]

    def test_a_failure_before_any_draw_still_reports_cleanly(
        self, tmp_path, monkeypatch, fake_hssm_g3
    ):
        def broken_simulator(**kwargs):
            raise RuntimeError("simulator unavailable")

        monkeypatch.setattr(
            "ssms.basic_simulators.simulator.simulator", broken_simulator
        )
        result = gate_hssm_load(tmp_path / "m.onnx", "ddm")
        assert not result["passed"]
        assert result["error"] == "RuntimeError: simulator unavailable"
        assert result["theta"] is None and result["seed"] is None
        assert result["attempts"] == 0
        assert result["rejected_draws"] == []
        assert fake_hssm_g3 == []

    def test_ddm_normal_st_seed_zero_is_usable(self, monkeypatch):
        """The draw that failed ddm_normal_st in practice, on the real simulator.

        The networks were trained on an st box widened to 0.35 (ssms ships
        0.25 at the pinned version), so the test installs that box. Its
        full-box seed-0 draw (t = 0.283, st = 0.285) simulates an rt below
        zero through the untruncated Normal kernel; the gate's own draw at the
        same seed must not.
        """
        import copy

        import ssms

        if "ddm_normal_st" not in ssms.config.model_config:
            pytest.skip("ssms has no ddm_normal_st")
        config = copy.deepcopy(ssms.config.model_config["ddm_normal_st"])
        config["param_bounds"] = [
            [-3.0, 0.3, 0.3, 0.25, 0.001],
            [3.0, 2.5, 0.7, 2.25, 0.35],
        ]
        monkeypatch.setitem(ssms.config.model_config, "ddm_normal_st", config)

        lower, upper = (np.asarray(b, dtype=float) for b in config["param_bounds"])
        theta = lower + (upper - lower) * np.random.default_rng(0).uniform(
            size=lower.shape
        )
        old = ssms.basic_simulators.simulator.simulator(
            model="ddm_normal_st", theta=theta, n_samples=100, random_state=0
        )
        assert (np.asarray(old["rts"]) <= 0).any(), "full-box draw no longer fails"

        draw = hssm_load_data("ddm_normal_st", n_trials=100, seed=0)
        assert draw["data"] is not None, draw["rejected"]
        assert isinstance(draw["data"], pd.DataFrame)
        assert (draw["data"]["rt"] > 0).all()


@pytest.mark.production
class TestAgainstProduction:
    """The gate's own acceptance test: it must pass a network that works.

    Opt-in (`-m production`): downloads from HuggingFace and imports HSSM.
    """

    def test_production_ddm_passes_every_gate(self):
        from pathlib import Path

        from huggingface_hub import hf_hub_download

        onnx_path = Path(hf_hub_download("franklab/HSSM", "ddm.onnx"))
        report = validate_network(onnx_path, model_name="ddm", network_type="lan")
        gates = {g["gate"]: g for g in report["gates"]}
        assert gates["structure"]["passed"], gates["structure"]
        assert gates["hssm_load"]["passed"], gates["hssm_load"]
        assert gates["density"]["passed"], gates["density"]
        # Comfortably inside the bound, not scraping it.
        assert gates["density"]["worst_ratio"] < 2.5
        assert report["passed"]
