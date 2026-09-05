"""Tests for the publish orchestrator.

All fast and offline: the pieces that talk to HuggingFace and MLflow are
exercised by hand against the staging repo, but the decisions that stop a bad
publish are pure functions and belong here.
"""

import pytest

from publish.publish_network import (
    PRODUCTION_REPOS,
    PublishError,
    gate_verdict,
    stage_artifacts,
    upload_include_patterns,
)


def report(**gates):
    """A validation report with the given gates; value is (passed, skipped)."""
    return {
        "gates": [
            {"gate": name, "passed": passed, **({"skipped": True} if skipped else {})}
            for name, (passed, skipped) in gates.items()
        ]
    }


ALL_RAN = dict(
    structure=(True, False),
    parity=(True, False),
    hssm_load=(True, False),
    density=(True, False),
)


class TestGateVerdict:
    def test_accepts_a_report_where_every_gate_ran(self):
        ok, reason = gate_verdict(report(**ALL_RAN))
        assert ok, reason

    def test_a_skipped_required_gate_is_not_a_pass(self):
        # The trap this function exists for: skipped gates report passed=True,
        # so report["passed"] is True for a network nothing checked.
        skipped_density = {**ALL_RAN, "density": (True, True)}
        assert all(g["passed"] for g in report(**skipped_density)["gates"])

        ok, reason = gate_verdict(report(**skipped_density))
        assert not ok
        assert "density" in reason and "not actually checked" in reason

    def test_parity_may_skip_because_torch_runs_have_no_jax_state(self):
        ok, reason = gate_verdict(report(**{**ALL_RAN, "parity": (True, True)}))
        assert ok, reason

    def test_a_gate_missing_from_the_report_is_not_a_pass(self):
        # A truncated report or a schema change leaves no trace in the gate
        # list, so absent has to count the same as skipped.
        without_density = {k: v for k, v in ALL_RAN.items() if k != "density"}
        ok, reason = gate_verdict(report(**without_density))
        assert not ok
        assert "density" in reason

    def test_a_failed_gate_is_reported_with_its_error(self):
        failing = {**ALL_RAN, "density": (False, False)}
        r = report(**failing)
        r["gates"][-1]["error"] = "worst_ratio 13.4 > 3.0"
        ok, reason = gate_verdict(r)
        assert not ok
        assert "worst_ratio 13.4" in reason


class TestStaging:
    def make_run(self, folder, uuid, kinds=("__model.onnx", "__train_state.jax")):
        folder.mkdir(parents=True, exist_ok=True)
        for kind in kinds:
            (folder / f"{uuid}_lan_ddm{kind}").write_bytes(b"x")

    def test_copies_only_the_requested_run(self, tmp_path):
        # LANfactory writes every run of a model into one flat folder, so the
        # source almost always holds other runs' files too.
        source = tmp_path / "shared"
        self.make_run(source, "a" * 32)
        self.make_run(source, "b" * 32)

        onnx = stage_artifacts(source, "a" * 32, tmp_path / "staged")

        staged = sorted(p.name for p in (tmp_path / "staged").iterdir())
        assert len(staged) == 2
        assert all("a" * 32 in name for name in staged)
        assert onnx.name == f"{'a' * 32}_lan_ddm__model.onnx"

    def test_matches_the_uuid_anywhere_in_the_name(self, tmp_path):
        # jax writes {uuid}_{nt}_{model}__kind, torch writes
        # {model}_{nt}_{uuid}_kind — neither a prefix nor a suffix glob works.
        source = tmp_path / "torch"
        source.mkdir()
        (source / f"ddm_lan_{'c' * 32}_model.onnx").write_bytes(b"x")

        onnx = stage_artifacts(source, "c" * 32, tmp_path / "staged")
        assert onnx.name == f"ddm_lan_{'c' * 32}_model.onnx"

    def test_copies_rather_than_links(self, tmp_path):
        # lanfactory resolves the canonical ONNX's parent and compares it to
        # the folder; resolve() follows links, so a link fails that check.
        source = tmp_path / "src"
        self.make_run(source, "d" * 32)
        onnx = stage_artifacts(source, "d" * 32, tmp_path / "staged")
        assert not onnx.is_symlink()
        assert onnx.resolve().parent == (tmp_path / "staged").resolve()

    def test_refuses_when_nothing_matches(self, tmp_path):
        source = tmp_path / "src"
        self.make_run(source, "a" * 32)
        with pytest.raises(PublishError, match="No artifacts matching"):
            stage_artifacts(source, "e" * 32, tmp_path / "staged")

    def test_refuses_a_run_with_two_onnx_files(self, tmp_path):
        source = tmp_path / "src"
        self.make_run(source, "f" * 32, kinds=("__model.onnx", "__other.onnx"))
        with pytest.raises(PublishError, match="exactly one .onnx"):
            stage_artifacts(source, "f" * 32, tmp_path / "staged")

    def test_reports_a_missing_source_directory(self, tmp_path):
        with pytest.raises(PublishError, match="does not exist"):
            stage_artifacts(tmp_path / "nope", "a" * 32, tmp_path / "staged")

    def test_refuses_a_staging_directory_holding_another_run(self, tmp_path):
        # The whole staging dir is uploaded, so a leftover from a previous run
        # would be published as part of this network and land in the manifest.
        source = tmp_path / "src"
        self.make_run(source, "a" * 32)
        staged = tmp_path / "staged"
        staged.mkdir()
        (staged / "leftover_from_another_run.pickle").write_bytes(b"x")

        with pytest.raises(PublishError, match="did not produce"):
            stage_artifacts(source, "a" * 32, staged)
        assert (staged / "leftover_from_another_run.pickle").exists()

    def test_a_dry_run_does_not_lock_the_staging_directory(self, tmp_path):
        # A dry run stages exactly the files the real publish would, then
        # returns early leaving them there. Refusing on the second call would
        # make "--dry-run, look, publish" impossible against one --staging-dir,
        # and would also throw away the report the operator just read.
        source = tmp_path / "src"
        self.make_run(source, "a" * 32)
        staged = tmp_path / "staged"

        stage_artifacts(source, "a" * 32, staged)
        (staged / "validation_report.json").write_text("{}")

        assert stage_artifacts(source, "a" * 32, staged).exists()

    def test_stages_an_operator_authored_model_card(self, tmp_path):
        # The card has no run_uuid in its name, so the uuid glob cannot find
        # it. Without it lanfactory renders a default whose usage example
        # assumes the model is already a built-in HSSM name — which does not
        # run for a model published before its HSSM config ships.
        source = tmp_path / "src"
        self.make_run(source, "a" * 32)
        (source / "model_card.yaml").write_text("title: gamma_drift (LAN)\n")
        staged = tmp_path / "staged"

        stage_artifacts(source, "a" * 32, staged)

        assert (staged / "model_card.yaml").read_text() == "title: gamma_drift (LAN)\n"

    def test_stages_the_recovery_report(self, tmp_path):
        # Same problem as the card: no run_uuid in the name, so the glob cannot
        # find it. It matters because validation_report.json cannot see a
        # recovery failure -- the gate has no recovery check in it -- so a
        # network whose recovery verdict is false would otherwise ship with
        # only the evidence that could not have caught the problem.
        source = tmp_path / "src"
        self.make_run(source, "a" * 32)
        (source / "recovery_report.json").write_text('{"passed": false}')
        staged = tmp_path / "staged"

        stage_artifacts(source, "a" * 32, staged)

        assert (staged / "recovery_report.json").read_text() == '{"passed": false}'

    def test_a_staged_recovery_report_is_not_a_foreign_leftover(self, tmp_path):
        # It is copied from the source folder, so re-running the identical
        # command finds it already there. Refusing on it would make the command
        # unrepeatable, the same way the card once did.
        source = tmp_path / "src"
        self.make_run(source, "a" * 32)
        (source / "recovery_report.json").write_text('{"passed": false}')
        staged = tmp_path / "staged"
        stage_artifacts(source, "a" * 32, staged)

        assert stage_artifacts(source, "a" * 32, staged).exists()

    def test_absent_model_card_is_not_an_error(self, tmp_path):
        source = tmp_path / "src"
        self.make_run(source, "a" * 32)

        stage_artifacts(source, "a" * 32, tmp_path / "staged")

        assert not (tmp_path / "staged" / "model_card.yaml").exists()

    def test_a_sidecar_removed_from_source_is_removed_from_staging(self, tmp_path):
        # The staging directory survives between runs. A recovery report staged
        # last time and since deleted from the source would otherwise linger
        # and upload as if it described this network -- shipping another run's
        # evidence, which is the exact failure this publish step exists to
        # prevent.
        source = tmp_path / "src"
        self.make_run(source, "a" * 32)
        (source / "recovery_report.json").write_text('{"passed": true}')
        staged = tmp_path / "staged"

        stage_artifacts(source, "a" * 32, staged)
        assert (staged / "recovery_report.json").exists()

        (source / "recovery_report.json").unlink()
        stage_artifacts(source, "a" * 32, staged)

        assert not (staged / "recovery_report.json").exists()

    def test_a_successful_publish_does_not_poison_its_staging_directory(self, tmp_path):
        # lanfactory renders the card and README into the staging dir during
        # upload. Treating those as foreign leftovers made the identical
        # command unrepeatable: the second run refused on files the first run
        # had itself written.
        source = tmp_path / "src"
        self.make_run(source, "a" * 32)
        staged = tmp_path / "staged"
        stage_artifacts(source, "a" * 32, staged)
        for produced in ("validation_report.json", "model_card.yaml", "README.md"):
            (staged / produced).write_text("written by the upload")

        assert stage_artifacts(source, "a" * 32, staged).exists()

    def test_absent_recovery_report_is_not_an_error(self, tmp_path):
        source = tmp_path / "src"
        self.make_run(source, "a" * 32)

        stage_artifacts(source, "a" * 32, tmp_path / "staged")

        assert not (tmp_path / "staged" / "recovery_report.json").exists()

    def test_refuses_a_staging_path_that_is_not_a_directory(self, tmp_path):
        source = tmp_path / "src"
        self.make_run(source, "a" * 32)
        staged = tmp_path / "staged"
        staged.write_text("not a directory")

        with pytest.raises(PublishError, match="not a directory"):
            stage_artifacts(source, "a" * 32, staged)


class TestPublishRecord:
    """Against a throwaway sqlite store — no network, no HuggingFace."""

    @pytest.fixture
    def training_run_id(self, tmp_path, monkeypatch):
        """A finished training run in a store this test owns outright.

        MLflow keeps the tracking URI and the active experiment in module
        globals, so a store set up here would leak into every test that runs
        after it — and, in the other direction, an experiment id another test
        left active does not exist in this fresh database. Hence: restore the
        URI afterwards, and never rely on the ambient experiment.
        """
        import mlflow

        # chdir too: MLflow resolves an experiment's artifact root relative to
        # the working directory, and without this the suite writes ./mlruns
        # into the checkout.
        monkeypatch.chdir(tmp_path)
        previous = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
        experiment = mlflow.create_experiment(
            "ddm-training", artifact_location=str(tmp_path / "artifacts")
        )
        # Closed before yielding: publish_network starts its own run, and
        # MLflow refuses to start one while another is active.
        with mlflow.start_run(experiment_id=experiment) as training:
            run_id = training.info.run_id
        yield run_id
        mlflow.set_tracking_uri(previous)

    def record(self, tmp_path, training_run_id, verified):
        import mlflow

        from publish.publish_network import publish_network

        publish_run_id = publish_network(
            onnx_path=tmp_path / "ddm_lan_model.onnx",
            model="ddm",
            network_type="lan",
            repo_id="franklab/HSSM_staging",
            training_run_id=training_run_id,
            hf_commit="0" * 40,
            hf_commit_verified=verified,
            artifact_location=str(tmp_path / "artifacts"),
        )
        client = mlflow.MlflowClient()
        return client.get_run(publish_run_id), client.get_run(training_run_id)

    def test_a_confirmed_sha_is_recorded_as_hf_commit(self, tmp_path, training_run_id):
        publish, training = self.record(tmp_path, training_run_id, verified=True)
        assert publish.data.params["hf_commit"] == "0" * 40
        assert training.data.tags["hf_commit"] == "0" * 40
        assert training.data.tags["published"] == "true"

    def test_an_unconfirmed_sha_never_lands_under_hf_commit(
        self, tmp_path, training_run_id
    ):
        # The sha is read back from the repo head, so an unverified one may be
        # somebody else's push. Everything downstream queries hf_commit, which
        # is exactly why a lead must not be filed there — and the lead is still
        # not thrown away.
        publish, training = self.record(tmp_path, training_run_id, verified=False)
        assert "hf_commit" not in publish.data.params
        assert "hf_commit" not in training.data.tags
        assert publish.data.params["hf_commit_candidate"] == "0" * 40
        assert training.data.tags["hf_commit_candidate"] == "0" * 40
        assert publish.data.tags["hf_commit_verified"] == "false"

    def test_a_store_that_refuses_tags_does_not_sink_a_live_publish(
        self, tmp_path, training_run_id, monkeypatch
    ):
        # By this point the upload is on HuggingFace. The tags are
        # back-references; losing them must not cost the caller the run id of
        # a publish that succeeded.
        import mlflow

        def refuse(*_args, **_kwargs):
            raise RuntimeError("read-only tracking store")

        monkeypatch.setattr(mlflow.MlflowClient, "set_tag", refuse)
        publish, _ = self.record(tmp_path, training_run_id, verified=True)
        assert publish.info.run_id


class TestProductionGuard:
    def test_the_production_repo_is_named(self):
        # Every released HSSM downloads from this repo's root at main with no
        # revision pin, so a bad file there is live for everyone immediately.
        assert "franklab/HSSM" in PRODUCTION_REPOS

    def test_publishing_to_production_is_refused(self):
        from publish.publish_network import run_publish

        with pytest.raises(PublishError, match="production repo"):
            run_publish(hf_repo="franklab/HSSM", model="ddm", network_type="lan")

    @pytest.mark.parametrize(
        "spelling",
        ["Franklab/HSSM", "FRANKLAB/HSSM", "franklab/HSSM/", " franklab/HSSM "],
    )
    def test_a_differently_spelled_production_repo_is_still_production(self, spelling):
        # HuggingFace namespaces are case-insensitively unique, so none of
        # these is some other repo — each is production, spelled the way a
        # copy-paste or a shift key leaves it. An exact string match is the
        # difference between the guard holding and one capital letter
        # overwriting the file every released HSSM downloads.
        from publish.publish_network import run_publish

        with pytest.raises(PublishError, match="production repo"):
            run_publish(hf_repo=spelling, model="ddm", network_type="lan")

    def test_the_guard_runs_before_any_heavy_import(self, monkeypatch):
        # A safety check an ImportError can preempt is not a safety check.
        import builtins

        from publish.publish_network import run_publish

        real_import = builtins.__import__

        # Every module run_publish reaches, not just the ones imported in its
        # own body: mlflow and huggingface_hub come in one frame deeper, via
        # resolve_training_run and resolve_hf_commit. Without them the guard
        # could be moved below an `import mlflow` and this test would not know.
        heavy = {"lanfactory", "validation", "tempfile", "mlflow", "huggingface_hub"}

        def explode(name, *args, **kwargs):
            if name.split(".")[0] in heavy:
                raise ImportError(f"pretend {name} is not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", explode)
        with pytest.raises(PublishError, match="production repo"):
            run_publish(hf_repo="franklab/HSSM", model="ddm", network_type="lan")


class TestUploadIncludePatterns:
    """What the pipeline asks lanfactory to upload.

    Staging the recovery report is only half the job: lanfactory filters the
    staging directory by pattern, so a file staged but not matched is silently
    dropped.
    """

    def test_the_recovery_report_is_uploaded(self):
        assert "recovery_report.json" in upload_include_patterns(["*.onnx"])

    def test_lanfactory_defaults_are_preserved(self):
        # Extended, never replaced: the defaults name the trainer's own
        # artifacts, and re-listing them here would be a second copy to drift.
        defaults = ["*.onnx", "*.pt", "model_card.yaml"]

        assert upload_include_patterns(defaults)[: len(defaults)] == defaults

    def test_the_pipeline_names_its_own_report(self):
        # The ownership decision, asserted rather than left to a comment:
        # parameter recovery is this repo's concept, so the filename lives at
        # this call site and not in lanfactory's defaults.
        from lanfactory.hf.upload import DEFAULT_INCLUDE_PATTERNS

        assert "recovery_report.json" not in DEFAULT_INCLUDE_PATTERNS
        assert "recovery_report.json" in upload_include_patterns(
            DEFAULT_INCLUDE_PATTERNS
        )
