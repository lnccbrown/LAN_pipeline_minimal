"""Tests for the local runner: stage order, the flags each stage receives,
state persistence, and failure handling. Subprocesses are replaced by a fake
runner that records argv, so nothing here needs the upstream packages, a GPU,
or a tracking store beyond an empty sqlite file.
"""

import json
import shutil
from pathlib import Path

import experiment as xp
import pytest
import run_local as rl

mlflow = pytest.importorskip("mlflow")


@pytest.fixture
def exp(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ARTIFACT_LOCATION", raising=False)
    e = xp.scaffold_experiment("quick", tmp_path, model="ddm", quick=True)
    # The runner looks console scripts up on PATH; make them exist.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("generate", "jaxtrain", "torchtrain", "lan-validate", "lan-recover"):
        script = bindir / name
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{Path(shutil.which('sh')).parent}")
    return e


class FakeRunner:
    """Records argv; fabricates the side effects each real tool would have."""

    def __init__(self, exp, fail_stage=None, gate_passed=True):
        self.exp = exp
        self.calls = []
        self.fail_stage = fail_stage
        self.gate_passed = gate_passed

    def __call__(self, argv, env, capture_output, text):
        name = Path(argv[0]).name
        if argv[1:] == ["--help"]:  # preflight: advertise the lineage flags

            class Help:
                returncode = 0
                stdout = "--lineage-id --track"
                stderr = ""

            return Help()
        self.calls.append(argv)
        rc, out = 0, ""
        if name == "generate":
            if self.fail_stage == "generate":
                rc = 1
            else:
                folder = (
                    self.exp.stage_output("generate") / "data" / "training_data" / "ddm"
                )
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "training_data_x.pickle").write_bytes(b"x")
                self._mlflow_experiment("ddm-data-generation")
        elif name in ("jaxtrain", "torchtrain"):
            folder = self.exp.stage_output("train") / "lan" / "ddm"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "abc123_lan_ddm__model.onnx").write_bytes(b"onnx")
            exp_id = self._mlflow_experiment("ddm-training")
            with mlflow.start_run(experiment_id=exp_id) as run:
                mlflow.set_tags({"run_uuid": "abc123", "phase": "train"})
                self.train_run_id = run.info.run_id
        elif name == "lan-validate":
            out = json.dumps({"passed": self.gate_passed, "report": "r.json"})
            rc = 0 if self.gate_passed else 1
        elif name == "lan-recover":
            out_dir = Path(argv[argv.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            shard = out_dir / f"shard_{len(self.calls)}.json"
            shard.write_text("{}")
            out = json.dumps({"shard": str(shard), "error": None})

        class Result:
            returncode = rc
            stdout = out
            stderr = "boom" if rc else ""

        return Result()

    def _mlflow_experiment(self, name):
        mlflow.set_tracking_uri(xp.mlflow_tracking_uri(self.exp))
        e = mlflow.get_experiment_by_name(name)
        return e.experiment_id if e else mlflow.create_experiment(name)

    def argv_for(self, tool):
        return [c for c in self.calls if Path(c[0]).name == tool]


def _flag(argv, name):
    return argv[argv.index(name) + 1]


class TestStageWiring:
    def test_generate_then_train_share_lineage_and_state(self, exp, monkeypatch):
        monkeypatch.setattr(rl, "_has_analytical", lambda model: False)
        runner = FakeRunner(exp)
        state = rl.run_local(exp, ["generate", "train"], runner=runner)

        gen = runner.argv_for("generate")[0]
        train = runner.argv_for("jaxtrain")[0]
        assert (
            _flag(gen, "--lineage-id") == exp.lineage_id == _flag(train, "--lineage-id")
        )
        assert _flag(gen, "--mlflow-experiment-name") == "ddm-data-generation"
        assert _flag(train, "--mlflow-experiment-name") == "ddm-training"
        assert _flag(train, "--training-data-folder") == state["training_data_folder"]
        assert (
            _flag(train, "--data-generation-experiment-id")
            == (state["data_generation_experiment_id"])
        )
        assert state["run_uuid"] == "abc123"
        assert state["mlflow_run_id_train"] == runner.train_run_id
        assert Path(state["onnx_path"]).name == "abc123_lan_ddm__model.onnx"

    def test_recover_uses_state_and_reference_arm(self, exp, monkeypatch):
        monkeypatch.setattr(rl, "_has_analytical", lambda model: True)
        monkeypatch.setattr(
            rl, "_aggregate", lambda out_dir: {"recovery_verdict": True}
        )
        exp.stages.recover.n_datasets = 2
        exp.save()
        runner = FakeRunner(exp)
        rl.run_local(exp, ["generate", "train", "recover"], runner=runner)
        recs = runner.argv_for("lan-recover")
        assert len(recs) == 4  # 1 design x (network + analytical) x 2 datasets
        arms = {_flag(r, "--likelihood") for r in recs}
        assert arms == {"approx_differentiable", "analytical"}
        for r in recs:
            assert "--track" in r
            assert _flag(r, "--mlflow-experiment-name") == "ddm-inference"
            assert _flag(r, "--lineage-id") == exp.lineage_id
            assert _flag(r, "--mlflow-run-id-train") == runner.train_run_id
            if _flag(r, "--likelihood") == "analytical":
                assert "--onnx-path" not in r
            else:
                assert _flag(r, "--onnx-path").endswith(".onnx")
        assert {_flag(r, "--dataset-index") for r in recs} == {"0", "1"}

    def test_experiment_tags_reach_every_stage(self, exp, monkeypatch):
        monkeypatch.setattr(rl, "_has_analytical", lambda model: False)
        exp.mlflow.tags = {"project": "pilot"}
        exp.save()
        runner = FakeRunner(exp)
        rl.run_local(exp, ["generate", "train"], runner=runner)
        for argv in runner.calls:
            assert _flag(argv, "--mlflow-tag") == "project=pilot"

    def test_env_points_tools_at_the_experiment_store(self, exp):
        runner = FakeRunner(exp)
        captured = {}

        def spy(argv, env, **kw):
            if argv[1:] != ["--help"]:  # skip the preflight probes
                captured.update(env)
            return runner(argv, env, **kw)

        rl.run_local(exp, ["generate"], runner=spy)
        assert captured["MLFLOW_TRACKING_URI"] == f"sqlite:///{exp.dir / 'mlflow.db'}"
        assert "MLFLOW_ARTIFACT_LOCATION" not in captured
        assert "VIRTUAL_ENV" not in captured


class TestControlFlow:
    def test_dry_run_executes_nothing_and_writes_no_state(self, exp):
        calls = []
        rl.run_local(
            exp,
            ["generate", "train"],
            dry_run=True,
            runner=lambda *a, **k: calls.append(a),
        )
        assert calls == []
        assert not (exp.dir / "state.json").exists()

    def test_stage_failure_reports_the_tools_json_and_keeps_state(self, exp):
        runner = FakeRunner(exp, gate_passed=False)
        with pytest.raises(rl.StageFailed) as info:
            rl.run_local(exp, ["generate", "train", "validate"], runner=runner)
        assert info.value.stage == "validate"
        assert '"passed": false' in str(info.value)
        state = xp.read_state(exp)
        assert "onnx_path" in state  # earlier stages persisted

    def test_continue_on_gate_failure(self, exp, monkeypatch):
        monkeypatch.setattr(rl, "_has_analytical", lambda model: False)
        monkeypatch.setattr(rl, "_aggregate", lambda out_dir: {})
        runner = FakeRunner(exp, gate_passed=False)
        state = rl.run_local(
            exp,
            ["generate", "train", "validate", "recover"],
            continue_on_gate_failure=True,
            runner=runner,
        )
        assert state["validation_passed"] is False
        assert runner.argv_for("lan-recover")

    def test_unknown_stage_rejected(self, exp):
        with pytest.raises(ValueError, match="unknown stages"):
            rl.run_local(exp, ["generate", "deploy"], runner=FakeRunner(exp))

    def test_train_without_data_is_a_clear_error(self, exp):
        with pytest.raises(rl.StageFailed, match="generate stage first"):
            rl.run_local(exp, ["train"], runner=FakeRunner(exp))

    def test_preflight_rejects_tools_without_lineage_flags(self, exp):
        def old_tool(argv, env, capture_output, text):
            class R:
                returncode = 0
                stdout = "Usage: generate [OPTIONS]\n--config-path"
                stderr = ""

            return R()

        with pytest.raises(rl.StageFailed, match="does not accept --lineage-id"):
            rl.run_local(exp, ["generate"], runner=old_tool)

    def test_missing_console_script_names_the_group(self, exp, monkeypatch):
        monkeypatch.setenv("PATH", "/nonexistent")
        with pytest.raises(rl.StageFailed, match="uv sync --group validate"):
            rl._cli("lan-recover")
