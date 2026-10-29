"""Tests for the publish orchestrator.

All fast and offline: the pieces that talk to HuggingFace and MLflow are
exercised by hand against the staging repo, but the decisions that stop a bad
publish are pure functions and belong here.
"""

import json

import click
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


ALL_RAN = {
    "structure": (True, False),
    "parity": (True, False),
    "hssm_load": (True, False),
    "density": (True, False),
    "mass_survey": (True, False),
}


class TestGateVerdict:
    def test_accepts_a_report_where_every_gate_ran(self):
        ok, reason = gate_verdict(report(**ALL_RAN))
        assert ok, reason

    def test_the_mass_survey_is_advisory_so_its_skip_does_not_block(self):
        # Under the locked LANfactory the survey always skips itself; it is
        # deliberately absent from REQUIRED_GATES until the lock moves.
        skipped_survey = {**ALL_RAN, "mass_survey": (True, True)}
        ok, reason = gate_verdict(report(**skipped_survey))
        assert ok, reason
        # A report from before the gate existed carries no such entry at all.
        without_survey = {k: v for k, v in ALL_RAN.items() if k != "mass_survey"}
        ok, reason = gate_verdict(report(**without_survey))
        assert ok, reason

    def test_a_failed_mass_survey_still_refuses_like_any_failed_gate(self):
        failing = {**ALL_RAN, "mass_survey": (False, False)}
        r = report(**failing)
        next(g for g in r["gates"] if g["gate"] == "mass_survey")["error"] = (
            "total.p99_abs_dev 0.2000 > 0.1"
        )
        ok, reason = gate_verdict(r)
        assert not ok
        assert "mass_survey" in reason and "0.2000" in reason

    def test_a_skipped_required_gate_is_not_a_pass(self):
        # The trap this function exists for: skipped gates report passed=True,
        # so report["passed"] is True for a network nothing checked.
        skipped_density = {**ALL_RAN, "density": (True, True)}
        assert all(g["passed"] for g in report(**skipped_density)["gates"])

        ok, reason = gate_verdict(report(**skipped_density))
        assert not ok
        assert "density" in reason and "not actually checked" in reason

    def test_a_skipped_hssm_load_is_not_a_pass(self):
        # G3 skips when no inset-box draw gives data HSSM accepts. The network
        # was never loaded, so it must not publish.
        ok, reason = gate_verdict(report(**{**ALL_RAN, "hssm_load": (True, True)}))
        assert not ok
        assert "hssm_load" in reason and "not actually checked" in reason

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
        next(g for g in r["gates"] if g["gate"] == "density")["error"] = (
            "worst_ratio 13.4 > 3.0"
        )
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
        (source / "recovery_report.json").write_text(
            self.report_for(next(source.glob("*.onnx")).name)
        )
        staged = tmp_path / "staged"

        stage_artifacts(source, "a" * 32, staged)

        assert (staged / "recovery_report.json").read_text() == (
            source / "recovery_report.json"
        ).read_text()

    def test_a_staged_recovery_report_is_not_a_foreign_leftover(self, tmp_path):
        # It is copied from the source folder, so re-running the identical
        # command finds it already there. Refusing on it would make the command
        # unrepeatable, the same way the card once did.
        source = tmp_path / "src"
        self.make_run(source, "a" * 32)
        (source / "recovery_report.json").write_text(
            self.report_for(next(source.glob("*.onnx")).name)
        )
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
        (source / "recovery_report.json").write_text(
            self.report_for(next(source.glob("*.onnx")).name)
        )
        staged = tmp_path / "staged"

        stage_artifacts(source, "a" * 32, staged)
        assert (staged / "recovery_report.json").exists()

        (source / "recovery_report.json").unlink()
        stage_artifacts(source, "a" * 32, staged)

        assert not (staged / "recovery_report.json").exists()

    def test_a_directory_squatting_on_a_sidecar_name_is_refused(self, tmp_path):
        # unlink() raises on a directory no matter what missing_ok says, and
        # copy2() onto one silently copies INTO it -- either branch, the
        # publish must stop with a clear refusal instead.
        source = tmp_path / "src"
        self.make_run(source, "a" * 32)
        staged = tmp_path / "staged"
        (staged / "model_card.yaml").mkdir(parents=True)

        with pytest.raises(PublishError, match="is a directory"):
            stage_artifacts(source, "a" * 32, staged)

    def test_a_symlink_squatting_on_a_sidecar_name_is_refused(self, tmp_path):
        # copy2() follows a symlink destination and overwrites whatever it
        # points at -- for a reused --staging-dir that can be a file outside
        # the staging area entirely. Refused, and the target stays untouched.
        source = tmp_path / "src"
        self.make_run(source, "a" * 32)
        (source / "model_card.yaml").write_text("title: new (LAN)\n")
        outside = tmp_path / "precious.yaml"
        outside.write_text("do not touch")
        staged = tmp_path / "staged"
        staged.mkdir()
        (staged / "model_card.yaml").symlink_to(outside)

        with pytest.raises(PublishError, match="is a symlink"):
            stage_artifacts(source, "a" * 32, staged)

        assert outside.read_text() == "do not touch"

    def report_for(self, *onnx_names):
        import json

        return json.dumps({"passed": True, "onnx_files": list(onnx_names)})

    def test_a_report_naming_the_staged_onnx_is_accepted(self, tmp_path):
        source = tmp_path / "src"
        self.make_run(source, "a" * 32)
        onnx_name = next(source.glob("*.onnx")).name
        (source / "recovery_report.json").write_text(self.report_for(onnx_name))

        assert stage_artifacts(source, "a" * 32, tmp_path / "staged").exists()

    def test_a_report_for_another_run_is_refused(self, tmp_path):
        # The exact hazard: two runs' artifacts share the source directory,
        # the ONNX is selected by run_uuid, the report by fixed name. Without
        # the binding, --run-id ships one run's network with the other run's
        # verdict.
        source = tmp_path / "src"
        self.make_run(source, "a" * 32)
        (source / "recovery_report.json").write_text(
            self.report_for("model_lan_bbbb_model.onnx")
        )

        with pytest.raises(PublishError, match="belongs to another run"):
            stage_artifacts(source, "a" * 32, tmp_path / "staged")

    def test_an_unbound_report_is_refused(self, tmp_path):
        # A report from before the binding existed cannot prove anything
        # about the network beside it; re-aggregating is cheap.
        source = tmp_path / "src"
        self.make_run(source, "a" * 32)
        (source / "recovery_report.json").write_text('{"passed": true}')

        with pytest.raises(PublishError, match="does not say which network"):
            stage_artifacts(source, "a" * 32, tmp_path / "staged")

    def test_a_string_onnx_files_cannot_substring_match(self, tmp_path):
        # `in` on a string is substring search: a report whose onnx_files is
        # one long string containing the staged name would false-ACCEPT. Only
        # a list makes membership mean membership.
        import json

        source = tmp_path / "src"
        self.make_run(source, "a" * 32)
        onnx_name = next(source.glob("*.onnx")).name
        (source / "recovery_report.json").write_text(
            json.dumps({"passed": True, "onnx_files": f"prefix_{onnx_name}"})
        )

        with pytest.raises(PublishError, match="not a list"):
            stage_artifacts(source, "a" * 32, tmp_path / "staged")

    def test_a_report_that_is_not_an_object_is_refused(self, tmp_path):
        # Valid JSON, wrong shape: .get on a list is an AttributeError
        # traceback, not a refusal, without the isinstance guard.
        source = tmp_path / "src"
        self.make_run(source, "a" * 32)
        (source / "recovery_report.json").write_text('["not", "an", "object"]')

        with pytest.raises(PublishError, match="does not say which network"):
            stage_artifacts(source, "a" * 32, tmp_path / "staged")

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

    def test_allow_production_opens_the_guard(self, monkeypatch):
        # The refusal branch is covered four ways above; the escape hatch had
        # no coverage at all, so nothing proved it actually opens -- a typo in
        # the parameter name would have looked exactly like a working guard.
        # Block the first import *after* the guard: reaching it is the proof.
        import builtins

        from publish.publish_network import run_publish

        real_import = builtins.__import__

        def explode(name, *args, **kwargs):
            if name.split(".")[0] == "lanfactory":
                raise ImportError("got past the guard")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", explode)
        with pytest.raises(ImportError, match="got past the guard"):
            run_publish(
                hf_repo="franklab/HSSM",
                model="ddm",
                network_type="lan",
                allow_production=True,
            )


class TestRunPublishRefusals:
    def test_a_type_the_validator_has_no_gates_for_is_refused_before_staging(
        self, tmp_path, monkeypatch
    ):
        # lanfactory.hf still names gonogo; the validator dropped it. Without
        # the check run_publish staged the run and died in a ValueError.
        from types import SimpleNamespace

        from publish import publish_network

        run = SimpleNamespace(
            info=SimpleNamespace(run_id="r"),
            data=SimpleNamespace(
                params={"model": "ddm", "network_type": "gonogo"},
                tags={"run_uuid": "a" * 32},
            ),
        )
        monkeypatch.setattr(publish_network, "resolve_training_run", lambda **kw: run)
        with pytest.raises(PublishError, match="'gonogo' is not one of"):
            publish_network.run_publish(
                hf_repo="example/HSSM_staging",
                run_id="r",
                artifact_dir=tmp_path,
                staging_dir=tmp_path / "staged",
            )
        assert not (tmp_path / "staged").exists()


class TestProductionConfirmation:
    """--allow-production is necessary but not sufficient.

    A flag survives shell history and a copied runbook line, which is the
    "ordinary CLI invocation" the refusal exists to prevent. Retyping the repo
    id does not survive either.
    """

    def invoke(self, monkeypatch, argv, stdin=None, tty=True):
        """Run the CLI with run_publish stubbed; returns (result, calls).

        `tty` fakes an interactive terminal, which the confirmation now
        requires. CliRunner's stdin never is one, so without this every
        confirmation path would be untestable.
        """
        from typer.testing import CliRunner

        from publish import publish_network

        monkeypatch.setattr(publish_network, "_stdin_is_a_terminal", lambda: tty)
        calls = []
        monkeypatch.setattr(
            publish_network,
            "run_publish",
            lambda **kwargs: calls.append(kwargs) or {"published": True},
        )
        result = CliRunner().invoke(publish_network.app, argv, input=stdin)
        return result, calls

    def test_piped_input_does_not_promote(self, monkeypatch):
        # `echo franklab/HSSM | lan-publish ... --allow-production` answers the
        # prompt perfectly and is exactly the scripted invocation the whole
        # check exists to stop. A pipe is not a person.
        result, calls = self.invoke(
            monkeypatch,
            ["--hf-repo", "franklab/HSSM", "--model", "ddm", "--allow-production"],
            stdin="franklab/HSSM\n",
            tty=False,
        )
        assert result.exit_code == 1
        assert "interactive terminal" in result.stdout
        assert calls == []

    def test_an_aborted_prompt_still_emits_the_json_line(self, monkeypatch):
        # Ctrl-C at the prompt. Every caller parses the JSON line on stdout, so
        # dying on a traceback would break the contract, not just the publish.
        from publish import publish_network

        def abort(*args, **kwargs):
            raise click.Abort()

        monkeypatch.setattr(publish_network.typer, "prompt", abort)
        result, calls = self.invoke(
            monkeypatch,
            ["--hf-repo", "franklab/HSSM", "--model", "ddm", "--allow-production"],
        )
        assert result.exit_code == 1
        assert json.loads(result.stdout.strip().splitlines()[-1])["published"] is False
        assert calls == []

    def test_a_non_interactive_invocation_does_not_promote(self, monkeypatch):
        # The flag on its own, with no one at the keyboard -- a cron job, a CI
        # step, a runbook line pasted into a script.
        result, calls = self.invoke(
            monkeypatch,
            ["--hf-repo", "franklab/HSSM", "--model", "ddm", "--allow-production"],
            tty=False,
        )
        assert result.exit_code != 0
        assert calls == []

    def test_a_near_miss_does_not_promote(self, monkeypatch):
        # The staging repo is one token away on the keyboard and in muscle
        # memory, and it is the repo the operator was just working in.
        result, calls = self.invoke(
            monkeypatch,
            ["--hf-repo", "franklab/HSSM", "--model", "ddm", "--allow-production"],
            stdin="franklab/HSSM_staging\n",
        )
        assert result.exit_code == 1
        assert "confirmation did not match" in result.stdout
        assert calls == []

    def test_retyping_the_repo_id_proceeds(self, monkeypatch):
        from publish import publish_network

        monkeypatch.setattr(publish_network, "_stdin_is_a_terminal", lambda: True)
        seen = {}

        def record(**kwargs):
            seen.update(kwargs)
            return {"published": True}

        monkeypatch.setattr(publish_network, "run_publish", record)
        from typer.testing import CliRunner

        result = CliRunner().invoke(
            publish_network.app,
            ["--hf-repo", "franklab/HSSM", "--model", "ddm", "--allow-production"],
            input="franklab/HSSM\n",
        )
        assert result.exit_code == 0
        assert seen["allow_production"] is True

    def test_a_staging_repo_is_never_prompted(self, monkeypatch):
        from publish import publish_network

        # No tty fake: a staging publish must not consult stdin at all.
        called = {}

        def record(**kwargs):
            called.update(kwargs)
            return {"published": True}

        monkeypatch.setattr(publish_network, "run_publish", record)
        from typer.testing import CliRunner

        # No input supplied: a prompt here would hang or abort, so passing
        # proves the confirmation stays out of the ordinary staging path.
        result = CliRunner().invoke(
            publish_network.app,
            ["--hf-repo", "franklab/HSSM_staging", "--model", "ddm"],
        )
        assert result.exit_code == 0
        assert called["hf_repo"] == "franklab/HSSM_staging"

    def test_a_dry_run_is_not_prompted(self, monkeypatch):
        # --dry-run touches neither HF nor MLflow (it does stage files and
        # rewrite the gate report locally), so gating it behind a prompt is
        # friction that buys nothing -- and it is the rehearsal that catches
        # the things --dry-run *can* see before a real promote.
        from publish import publish_network

        monkeypatch.setattr(
            publish_network, "run_publish", lambda **kw: {"published": False}
        )
        from typer.testing import CliRunner

        result = CliRunner().invoke(
            publish_network.app,
            [
                "--hf-repo",
                "franklab/HSSM",
                "--model",
                "ddm",
                "--allow-production",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0


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
