"""Tests for the validation gate.

The fast tests build tiny ONNX graphs by hand, so they need neither HSSM nor a
network. The end-to-end test against the real production ddm.onnx is opt-in
(`-m production`): it downloads from HuggingFace and imports the whole
inference stack, which does not belong in the default suite.
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from validate_network import (
    gate_parity,
    gate_structure,
    hellinger,
    validate_network,
)


def make_onnx(path, input_dims, output_dims=(1, 1)):
    """A minimal Identity-shaped graph with the requested input dims.

    dims entries may be ints (concrete) or strings (symbolic), which is the
    distinction G1 exists to enforce.
    """

    def dim_list(dims):
        return [d if isinstance(d, int) else d for d in dims]

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, dim_list(input_dims))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, dim_list(output_dims))
    weight = helper.make_tensor(
        "w",
        TensorProto.FLOAT,
        [input_dims[-1] if isinstance(input_dims[-1], int) else 6, 1],
        np.zeros(
            (input_dims[-1] if isinstance(input_dims[-1], int) else 6, 1),
            dtype=np.float32,
        )
        .ravel()
        .tolist(),
    )
    node = helper.make_node("MatMul", ["x", "w"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y], initializer=[weight])
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
