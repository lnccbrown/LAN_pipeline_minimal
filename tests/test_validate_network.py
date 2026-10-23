"""Tests for the validation gate.

The fast tests build tiny ONNX graphs by hand, so they need neither HSSM nor a
network. The end-to-end test against the real production ddm.onnx is opt-in
(`-m production`): it downloads from HuggingFace and imports the whole
inference stack, which does not belong in the default suite.
"""

import sys
import types

import numpy as np
import onnx
import pandas as pd
import pytest
from onnx import TensorProto, helper

import validate_network as vn
from validate_network import (
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
        for later in ("parity", "hssm_load", "density"):
            assert gates[later].get("skipped"), later

    def test_an_unknown_network_type_is_rejected_not_defaulted(self, tmp_path):
        # Defaulting to 0 extra inputs would surface as "input width 6 !=
        # expected 4", blaming the artifact for a mistyped flag.
        path = make_onnx(tmp_path / "good.onnx", (1, 6))
        with pytest.raises(ValueError, match="Unknown network_type"):
            validate_network(path, model_name="ddm", network_type="LAN")

    def test_report_shape_is_stable(self, tmp_path):
        path = make_onnx(tmp_path / "good.onnx", (1, 6))
        report = validate_network(
            path, model_name="ddm", skip_hssm=True, skip_density=True
        )
        assert report["schema_version"] == 1
        assert [g["gate"] for g in report["gates"]] == [
            "structure",
            "parity",
            "hssm_load",
            "density",
        ]


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
def fake_hssm(monkeypatch):
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
        self, tmp_path, monkeypatch, fake_hssm
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
        assert len(fake_hssm) == 1 and (fake_hssm[0]["rt"] > 0).all()

    def test_an_exhausted_budget_fails_with_a_named_reason(
        self, tmp_path, monkeypatch, fake_hssm
    ):
        sim = fake_simulator(lambda i, theta, n: np.full(n, -0.05))
        monkeypatch.setattr("ssms.basic_simulators.simulator.simulator", sim)
        result = gate_hssm_load(tmp_path / "m.onnx", "ddm")
        assert not result["passed"]
        assert "no usable dataset" in result["error"]
        assert result["attempts"] == vn.HSSM_LOAD_MAX_REDRAWS + 1
        assert len(result["rejected_draws"]) == vn.HSSM_LOAD_MAX_REDRAWS + 1
        # HSSM is never handed data it would reject.
        assert fake_hssm == []

    @pytest.mark.xfail(
        strict=True, reason="PR #64 review: exhausted G3 budget still blocks publish"
    )
    def test_an_exhausted_budget_skips_rather_than_fails(
        self, tmp_path, monkeypatch, fake_hssm
    ):
        # The gate's own message calls this "a fact about the box, not the
        # network", so it must not block publish the way a real HSSM failure
        # does. Skipped gates report passed=True, as parity's does.
        sim = fake_simulator(lambda i, theta, n: np.full(n, -0.05))
        monkeypatch.setattr("ssms.basic_simulators.simulator.simulator", sim)
        result = gate_hssm_load(tmp_path / "m.onnx", "ddm")
        assert result["passed"], result
        assert result.get("skipped") is True, result
        assert "box" in result["reason"]
        # The evidence is still in the report.
        assert len(result["rejected_draws"]) == vn.HSSM_LOAD_MAX_REDRAWS + 1
        assert fake_hssm == []

    def test_an_hssm_failure_reports_the_draw_it_was_handed(
        self, tmp_path, monkeypatch, fake_hssm
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
        self, tmp_path, monkeypatch, fake_hssm
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
        assert fake_hssm == []

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
