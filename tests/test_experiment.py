"""Tests for the experiment directory: scaffold, load/validate, lineage, state.

No MLflow, no SLURM, no upstream CLIs: everything here is file I/O, which is
what lets `--script-only` rendering and these tests share one code path.
"""

import json

import experiment as xp
import pytest
import yaml


@pytest.fixture
def scaffolded(tmp_path):
    return xp.scaffold_experiment("pilot", tmp_path, model="ddm", quick=True)


class TestScaffold:
    def test_writes_the_three_files(self, scaffolded):
        names = sorted(p.name for p in scaffolded.dir.iterdir())
        assert names == [
            "data_generation.yaml",
            "experiment.yaml",
            "network_training.yaml",
        ]

    def test_rewrites_model_and_network_type_keeping_comments(self, tmp_path):
        exp = xp.scaffold_experiment(
            "cpn-run", tmp_path, model="angle", quick=True, network_type="cpn"
        )
        gen = yaml.safe_load((exp.dir / "data_generation.yaml").read_text())
        train_text = (exp.dir / "network_training.yaml").read_text()
        train = yaml.safe_load(train_text)
        assert gen["MODEL"] == "angle"
        assert train["MODEL"] == "angle"
        assert train["NETWORK_TYPE"] == "cpn"
        # the template's comments survive the rewrite
        assert "# Quick Test Network Training Configuration" in train_text

    def test_quick_uses_quick_test_templates(self, scaffolded):
        train = yaml.safe_load((scaffolded.dir / "network_training.yaml").read_text())
        assert train["N_EPOCHS"] == 2  # quick_test value, examples uses 20
        # a minutes-long network cannot pass the density gate
        assert scaffolded.stages.validate.skip_density is True
        assert (
            xp.scaffold_experiment(
                "big", scaffolded.dir.parent
            ).stages.validate.skip_density
            is False
        )

    def test_examples_are_the_default_templates(self, tmp_path):
        exp = xp.scaffold_experiment("big", tmp_path, model="ddm")
        train = yaml.safe_load((exp.dir / "network_training.yaml").read_text())
        assert train["N_EPOCHS"] == 20

    def test_lineage_minted_once_and_persisted(self, scaffolded):
        assert len(scaffolded.lineage_id) == 32
        int(scaffolded.lineage_id, 16)
        again = xp.load_experiment(scaffolded.dir)
        assert again.lineage_id == scaffolded.lineage_id

    def test_refuses_non_empty_dir_without_force(self, tmp_path, scaffolded):
        with pytest.raises(xp.ExperimentError, match="not empty"):
            xp.scaffold_experiment("pilot", tmp_path, quick=True)
        forced = xp.scaffold_experiment("pilot", tmp_path, quick=True, force=True)
        assert forced.lineage_id != scaffolded.lineage_id

    @pytest.mark.parametrize("bad", ["../x", "a b", "", "/abs"])
    def test_rejects_unsafe_names(self, tmp_path, bad):
        with pytest.raises(xp.ExperimentError):
            xp.scaffold_experiment(bad, tmp_path, quick=True)

    def test_rejects_unknown_network_type_and_trainer(self, tmp_path):
        with pytest.raises(xp.ExperimentError, match="network_type"):
            xp.scaffold_experiment("a", tmp_path, quick=True, network_type="mlp")
        with pytest.raises(xp.ExperimentError, match="trainer"):
            xp.scaffold_experiment("b", tmp_path, quick=True, trainer="keras")


class TestLoad:
    def test_round_trips_and_interpolates_names(self, scaffolded):
        exp = xp.load_experiment(scaffolded.dir / "experiment.yaml")
        assert exp.model == "ddm"
        assert exp.experiment_names() == {
            "data_generation": "ddm-data-generation",
            "training": "ddm-training",
            "inference": "ddm-inference",
        }
        assert (
            exp.stage_config("generate") == (exp.dir / "data_generation.yaml").resolve()
        )
        assert exp.stage_output("train") == (exp.dir / "networks").resolve()

    def test_custom_experiment_name_override(self, scaffolded):
        data = yaml.safe_load(scaffolded.path.read_text())
        data["mlflow"]["experiments"]["training"] = "lab/{model}/train"
        scaffolded.path.write_text(yaml.safe_dump(data))
        assert xp.load_experiment(scaffolded.dir).experiment_names()["training"] == (
            "lab/ddm/train"
        )

    def _corrupt(self, exp, mutate):
        data = yaml.safe_load(exp.path.read_text())
        mutate(data)
        exp.path.write_text(yaml.safe_dump(data))

    def test_model_mismatch_with_stage_config_is_an_error(self, scaffolded):
        self._corrupt(scaffolded, lambda d: d.__setitem__("model", "angle"))
        with pytest.raises(xp.ExperimentError, match="MODEL 'ddm' disagrees"):
            xp.load_experiment(scaffolded.dir)

    def test_network_type_mismatch_is_an_error(self, scaffolded):
        self._corrupt(scaffolded, lambda d: d["network"].__setitem__("type", "cpn"))
        with pytest.raises(xp.ExperimentError, match="NETWORK_TYPE"):
            xp.load_experiment(scaffolded.dir)

    def test_reserved_tag_rejected(self, scaffolded):
        self._corrupt(
            scaffolded, lambda d: d["mlflow"].__setitem__("tags", {"phase": "x"})
        )
        with pytest.raises(xp.ExperimentError, match="reserved"):
            xp.load_experiment(scaffolded.dir)

    def test_unknown_design_rejected(self, scaffolded):
        self._corrupt(
            scaffolded, lambda d: d["stages"]["recover"].__setitem__("designs", ["L9"])
        )
        with pytest.raises(xp.ExperimentError, match="unknown recovery designs"):
            xp.load_experiment(scaffolded.dir)

    def test_unknown_key_rejected(self, scaffolded):
        self._corrupt(scaffolded, lambda d: d["network"].__setitem__("gpus", 2))
        with pytest.raises(xp.ExperimentError, match="unknown keys"):
            xp.load_experiment(scaffolded.dir)

    def test_missing_directory_has_a_helpful_message(self, tmp_path):
        with pytest.raises(xp.ExperimentError, match="lan-sbatch init"):
            xp.load_experiment(tmp_path / "nope")

    def test_wrong_schema_version(self, scaffolded):
        self._corrupt(scaffolded, lambda d: d.__setitem__("schema_version", 99))
        with pytest.raises(xp.ExperimentError, match="schema_version"):
            xp.load_experiment(scaffolded.dir)


class TestLineage:
    def test_ensure_writes_back_only_when_absent(self, scaffolded):
        before = scaffolded.path.read_text()
        assert xp.ensure_lineage_id(scaffolded) == scaffolded.lineage_id
        assert scaffolded.path.read_text() == before

        scaffolded.lineage_id = None
        minted = xp.ensure_lineage_id(scaffolded)
        assert len(minted) == 32
        assert yaml.safe_load(scaffolded.path.read_text())["lineage_id"] == minted


class TestState:
    def test_round_trip_and_none_dropping(self, scaffolded):
        assert xp.read_state(scaffolded) == {}
        xp.write_state(scaffolded, data_generation_experiment_id="7", onnx_path=None)
        xp.write_state(scaffolded, training_data_folder="/d")
        assert xp.read_state(scaffolded) == {
            "data_generation_experiment_id": "7",
            "training_data_folder": "/d",
        }

    def test_corrupt_state_is_reported(self, scaffolded):
        (scaffolded.dir / "state.json").write_text("{not json")
        with pytest.raises(xp.ExperimentError, match="not valid JSON"):
            xp.read_state(scaffolded)


class TestArtifacts:
    def test_run_uuid_from_both_trainer_conventions(self):
        assert xp.run_uuid_from_filename("abc_lan_ddm__model.onnx") == "abc"
        assert xp.run_uuid_from_filename("ddm_lan_abc_model.onnx") == "abc"
        assert xp.run_uuid_from_filename("model.onnx") is None

    def test_find_onnx_newest_wins(self, scaffolded):
        folder = scaffolded.stage_output("train") / "lan" / "ddm"
        folder.mkdir(parents=True)
        assert xp.find_onnx(scaffolded) is None
        old = folder / "old_lan_ddm__model.onnx"
        new = folder / "new_lan_ddm__model.onnx"
        old.write_bytes(b"a")
        new.write_bytes(b"b")
        import os

        os.utime(old, (1, 1))
        assert xp.find_onnx(scaffolded) == new


class TestTrackingUri:
    def test_precedence(self, scaffolded):
        assert xp.mlflow_tracking_uri(scaffolded, env={}) == (
            f"sqlite:///{scaffolded.dir / 'mlflow.db'}"
        )
        assert (
            xp.mlflow_tracking_uri(
                scaffolded, env={"MLFLOW_TRACKING_URI": "http://h:5"}
            )
            == "http://h:5"
        )
        scaffolded.mlflow.tracking_uri = "sqlite:////x.db"
        assert (
            xp.mlflow_tracking_uri(
                scaffolded, env={"MLFLOW_TRACKING_URI": "http://h:5"}
            )
            == "sqlite:////x.db"
        )


def test_state_and_outputs_are_gitignored():
    gitignore = (xp.REPO_ROOT / ".gitignore").read_text()
    for pattern in (
        "experiments/*/state.json",
        "experiments/*/data/",
        "experiments/*/networks/",
    ):
        assert pattern in gitignore
    assert json.loads('{"ok": true}')  # keep json import honest for state tests
