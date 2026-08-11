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


class TestProductionGuard:
    def test_the_production_repo_is_named(self):
        # Every released HSSM downloads from this repo's root at main with no
        # revision pin, so a bad file there is live for everyone immediately.
        assert "franklab/HSSM" in PRODUCTION_REPOS

    def test_publishing_to_production_is_refused(self):
        from publish.publish_network import run_publish

        with pytest.raises(PublishError, match="production repo"):
            run_publish(hf_repo="franklab/HSSM", model="ddm", network_type="lan")
