"""Tests for the publish orchestrator.

All fast and offline: the pieces that talk to HuggingFace and MLflow are
exercised by hand against the staging repo, but the decisions that stop a bad
publish are pure functions and belong here.
"""

import json

import click
import pytest

from publish.publish_network import (
    AUX_PROVENANCE_KEYS,
    PRODUCTION_REPOS,
    REQUIRED_GATES,
    REQUIRED_GATES_BY_NETWORK_TYPE,
    PublishError,
    aux_provenance,
    forwarded_tags,
    gate_verdict,
    stage_artifacts,
    upload_include_patterns,
    write_aux_model_card,
)


def report(network_type="lan", **gates):
    """A validation report with the given gates; value is (passed, skipped)."""
    return {
        "network_type": network_type,
        "gates": [
            {"gate": name, "passed": passed, **({"skipped": True} if skipped else {})}
            for name, (passed, skipped) in gates.items()
        ],
    }


ALL_RAN = {
    "structure": (True, False),
    "parity": (True, False),
    "hssm_load": (True, False),
    "density": (True, False),
    "mass_survey": (True, False),
}

AUX_ALL_RAN = dict(
    structure=(True, False),
    parity=(True, True),
    hssm_missing_load=(True, False),
    accuracy=(True, False),
)

# What LANfactory logs on a training run started from a derive-aux corpus.
PROVENANCE = {
    "derivation_method": "derived-from-lan",
    "aux_category": "choice",
    "source_lan_run_uuid": "f" * 32,
    "source_lan_sha256": "a1b2" * 16,
    "source_lan_hf_commit": "c" * 40,
    "integration_grid": "1000",
    "integration_max_t": "20.0",
}


def aux_report(network_type="cpn", model="ddm_sdv"):
    """A passing auxiliary report with the numbers the card and metrics read."""
    return {
        "schema_version": 1,
        "onnx": f"/staged/{model}_{network_type}.onnx",
        "model": model,
        "network_type": network_type,
        "aux_category": "choice" if network_type == "cpn" else "omission",
        "passed": True,
        "gates": [
            {"gate": "structure", "passed": True, "input_width": 6},
            {"gate": "parity", "passed": True, "skipped": True, "reason": "torch"},
            {
                "gate": "hssm_missing_load",
                "passed": True,
                "initial_logp_by_p_outlier": {"0.0": -123.4, "0.05": -120.1},
                "n_trials": 100,
                "n_missing": 20,
                "lan": f"{model} (by name)",
            },
            {
                "gate": "accuracy",
                "passed": True,
                "mean_abs_error": 0.0012,
                "max_abs_error": 0.0031,
                "mean_abs_max": 0.01,
                "max_abs_max": 0.03,
                "n_param_draws": 20,
                "n_sim": 100_000,
                "draws": [
                    {"theta": [1.0], "choice": 1.0, "truth_mc_se": 0.0016},
                    {"theta": [1.0], "choice": -1.0, "truth_mc_se": 0.0009},
                ],
            },
        ],
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

    def test_the_lan_gate_set_is_unchanged(self):
        # The alias other modules and the pinned tests import.
        assert REQUIRED_GATES == ("structure", "hssm_load", "density")
        assert REQUIRED_GATES_BY_NETWORK_TYPE["lan"] is REQUIRED_GATES

    def test_a_report_without_network_type_is_refused(self):
        # A pre-P1 report cannot say which gate set judged it; assuming "lan"
        # would let an auxiliary report from an old validator walk through.
        r = report(**ALL_RAN)
        del r["network_type"]
        ok, reason = gate_verdict(r)
        assert not ok
        assert "no network_type" in reason and "re-run the validator" in reason

    @pytest.mark.parametrize("network_type", ["cpn", "opn"])
    def test_an_auxiliary_report_needs_the_auxiliary_gates(self, network_type):
        ok, reason = gate_verdict(report(network_type, **AUX_ALL_RAN))
        assert ok, reason

        # The LAN gates prove nothing about a params-only graph.
        ok, reason = gate_verdict(report(network_type, **ALL_RAN))
        assert not ok
        assert "hssm_missing_load" in reason and "accuracy" in reason

    @pytest.mark.parametrize("network_type", ["cpn", "opn"])
    @pytest.mark.parametrize("gate", ["structure", "hssm_missing_load", "accuracy"])
    def test_a_skipped_required_gate_refuses_an_auxiliary_network(
        self, network_type, gate
    ):
        # Every required gate, not just the one with a built-in skip: the cpn
        # missing-load gate skips itself under HSSM < 0.6.0, and --skip-accuracy
        # skips accuracy on purpose. Either is a network nothing has checked.
        skipped = {**AUX_ALL_RAN, gate: (True, True)}
        assert all(g["passed"] for g in report(network_type, **skipped)["gates"])

        ok, reason = gate_verdict(report(network_type, **skipped))
        assert not ok
        assert gate in reason and "not actually checked" in reason

    def test_a_gonogo_report_is_refused_for_lack_of_a_consumer(self):
        # Even one where every gate ran: nothing in HSSM loads a gonogo, and
        # a root filename on the Hub cannot be taken back.
        ok, reason = gate_verdict(report("gonogo", **AUX_ALL_RAN))
        assert not ok
        assert "no HSSM consumer" in reason and "gonogo" in reason

    def test_an_unknown_network_type_is_refused(self):
        ok, reason = gate_verdict(report("mlp", **AUX_ALL_RAN))
        assert not ok
        assert "mlp" in reason


class TestAuxProvenance:
    def test_a_lan_has_no_provenance(self):
        assert aux_provenance(PROVENANCE, "lan") == {}

    def test_exactly_the_required_keys_are_read(self):
        params = {**PROVENANCE, "model": "ddm_sdv", "n_epochs": "10"}
        assert aux_provenance(params, "cpn") == PROVENANCE
        assert set(PROVENANCE) == set(AUX_PROVENANCE_KEYS)

    def test_the_optional_run_id_is_included_when_present(self):
        params = {**PROVENANCE, "source_lan_run_id": "0123456789abcdef"}
        assert aux_provenance(params, "cpn")["source_lan_run_id"] == "0123456789abcdef"

    @pytest.mark.parametrize("missing", AUX_PROVENANCE_KEYS)
    def test_each_missing_required_key_is_named(self, missing):
        params = {k: v for k, v in PROVENANCE.items() if k != missing}
        with pytest.raises(PublishError, match=missing) as excinfo:
            aux_provenance(params, "cpn")
        # A run started from a simulated corpus is the common way to get
        # here, and the message must say what to do about it.
        assert "simulated corpus" in str(excinfo.value)

    @pytest.mark.parametrize("absent", ["", "None", "null", " "])
    def test_an_empty_or_none_value_counts_as_missing(self, absent):
        # MLflow stores a param logged as None as the string "None": a corpus
        # derived from a local ONNX with no Hub revision would otherwise pass
        # and publish with source_lan_hf_commit="None" on the run and card.
        with pytest.raises(PublishError, match="source_lan_hf_commit"):
            aux_provenance({**PROVENANCE, "source_lan_hf_commit": absent}, "cpn")

    @pytest.mark.parametrize("absent", ["", "None", "null"])
    def test_a_legacy_lan_may_have_no_run_uuid(self, absent):
        # The 2023 Hub artifacts predate run uuids; LANfactory logs the key
        # as "" for a corpus derived from one. The key must exist, its value
        # may be empty, and it comes back normalised to "".
        provenance = aux_provenance(
            {**PROVENANCE, "source_lan_run_uuid": absent}, "cpn"
        )
        assert provenance["source_lan_run_uuid"] == ""
        with pytest.raises(PublishError, match="source_lan_run_uuid"):
            aux_provenance(
                {k: v for k, v in PROVENANCE.items() if k != "source_lan_run_uuid"},
                "cpn",
            )

    def test_a_none_valued_optional_key_is_left_out(self):
        params = {**PROVENANCE, "source_lan_run_id": "None"}
        assert "source_lan_run_id" not in aux_provenance(params, "cpn")

    def test_a_gonogo_has_no_provenance_to_read(self):
        # run_publish refuses a gonogo before it gets here; the function must
        # not be the thing that refuses it, or the message would be "relabel
        # your provenance" for a network that can never ship.
        assert aux_provenance({}, "gonogo") == {}

    def test_trained_from_simulation_is_recognised_but_not_publishable(self):
        # The contract allows the value, but no writer emits it and the card
        # would assert a LAN lineage the network does not have.
        with pytest.raises(PublishError, match="trained-from-simulation") as excinfo:
            aux_provenance(
                {**PROVENANCE, "derivation_method": "trained-from-simulation"},
                "cpn",
            )
        assert "no publish path" in str(excinfo.value)
        assert "derived-from-lan" in str(excinfo.value)

    def test_a_mis_categorised_network_is_refused(self):
        # A cpn is published as ddm_sdv_cpn.onnx and HSSM feeds it a choice;
        # a run that says its output is an omission probability is a
        # different network wearing that filename.
        with pytest.raises(PublishError, match="omission"):
            aux_provenance({**PROVENANCE, "aux_category": "omission"}, "cpn")
        assert aux_provenance({**PROVENANCE, "aux_category": "omission"}, "opn")

    def test_an_unknown_derivation_method_is_refused(self):
        with pytest.raises(PublishError, match="derivation_method"):
            aux_provenance({**PROVENANCE, "derivation_method": "magic"}, "cpn")

    def test_only_the_known_training_tags_are_forwarded(self):
        tags = {
            "derive_total_mass_mean": "0.998",
            "derive_total_mass_max": "1.0",
            "data_origin": "derived",
            "run_uuid": "f" * 32,
            "mlflow.user": "someone",
        }
        assert forwarded_tags(tags) == {
            "derive_total_mass_mean": "0.998",
            "derive_total_mass_max": "1.0",
            "data_origin": "derived",
        }

    def test_the_tail_policy_record_travels_when_present_and_only_then(self):
        # LANfactory L2/L3 log the corpus manifest's fallback fraction, the
        # simulated mass past max_t and the onset leak beside the mass stats;
        # an older training run has none of them and none is invented.
        record = {
            "derive_fallback_frac": "0.012",
            "derive_sim_past_max_t_max": "0.004",
            "derive_leak_below_onset_p99": "0.0007",
        }
        forwarded = forwarded_tags({**record, "run_uuid": "f" * 32})
        assert forwarded == record

        older = forwarded_tags(
            {"derive_total_mass_mean": "0.998", "run_uuid": "f" * 32}
        )
        assert older == {"derive_total_mass_mean": "0.998"}
        assert not set(record) & set(older)

    def test_a_tag_stored_as_none_is_absent_not_forwarded(self):
        # LANfactory stringifies every derive_stats entry, and the survey
        # reports leak_below_onset as None for a model with no onset param,
        # so the tag arrives as the string "None". That is an absence, not a
        # value to put on the publish run.
        tags = {
            "derive_total_mass_mean": "0.998",
            "derive_fallback_frac": "",
            "derive_leak_below_onset_p99": "None",
            "data_origin": "null",
        }
        assert forwarded_tags(tags) == {"derive_total_mass_mean": "0.998"}


class TestAuxModelCard:
    """The card generated when the operator staged none."""

    def load(self, tmp_path, network_type, training_tags=None):
        import yaml

        provenance = {
            **PROVENANCE,
            "aux_category": "choice" if network_type == "cpn" else "omission",
        }
        path = write_aux_model_card(
            tmp_path,
            "ddm_sdv",
            network_type,
            provenance,
            aux_report(network_type),
            training_tags,
        )
        assert path == tmp_path / "model_card.yaml"
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_the_cpn_card_names_its_source_and_contract(self, tmp_path):
        card = self.load(tmp_path, "cpn")
        # Multi-line strings as `|` blocks, pinned on the raw text: safe_load
        # erases the scalar style, and a description folded into a quoted
        # scalar with doubled apostrophes is not something a reviewer reads.
        raw = (tmp_path / "model_card.yaml").read_text(encoding="utf-8")
        assert "description: |" in raw and "usage_example: |" in raw
        assert "''" not in raw
        assert card["title"] == "ddm_sdv (CPN)"
        assert card["license"] == "bsd-2-clause"
        assert card["library_name"] == "onnx"
        assert "cpn" in card["tags"] and "derived-from-lan" in card["tags"]
        description = card["description"]
        assert "a1b2" * 16 in description
        assert "f" * 32 in description and "c" * 40 in description
        assert "[θ in list_params order, choice]" in description
        assert "1000-point grid" in description and "max_t = 20.0" in description
        assert "log P(choice | θ)" in description
        # The gate numbers and the noise they are judged against.
        assert "0.0012" in description and "0.0031" in description
        assert "0.0016" in description
        assert "renormalised by the source LAN's own total" in description
        assert 'model="ddm_sdv"' in card["usage_example"]
        assert "missing_data=True" in card["usage_example"]
        assert "-999.0" in card["usage_example"]
        assert "deadline=True" not in card["usage_example"]
        # Left to lanfactory, which fills them from the pickled configs.
        assert "architecture" not in card and "training" not in card

    def test_the_opn_card_names_the_deadline_contract(self, tmp_path):
        card = self.load(tmp_path, "opn")
        assert card["title"] == "ddm_sdv (OPN)"
        assert "[θ in list_params order, deadline]" in card["description"]
        assert "log P(rt > deadline | θ)" in card["description"]
        assert "deadline=True" in card["usage_example"]
        assert "missing_data=True" in card["usage_example"]

    @pytest.mark.parametrize("network_type", ["cpn", "opn"])
    def test_the_card_renders_through_lanfactory(self, tmp_path, network_type):
        # The loader and renderer that run at upload time, on the file they
        # will read: a card that crashes there crashes halfway into a commit.
        from lanfactory.hf.model_card import generate_readme, load_model_card_yaml

        self.load(tmp_path, network_type)
        readme = generate_readme(load_model_card_yaml(tmp_path), "ddm_sdv")
        assert f"# ddm_sdv ({network_type.upper()})" in readme
        assert "missing_data=True" in readme

    @pytest.mark.parametrize(
        "error",
        [
            ValueError("'architecture' must be a mapping"),
            KeyError("license"),
            TypeError("bad tag"),
        ],
        ids=lambda e: type(e).__name__,
    )
    def test_a_card_lanfactory_would_reject_is_refused_and_removed(
        self, tmp_path, monkeypatch, error
    ):
        # The guard catches whatever the loader raises -- the breadth is the
        # point, since any of these would surface halfway into a Hub commit.
        import lanfactory.hf.model_card as model_card

        def reject(folder):
            raise error

        monkeypatch.setattr(model_card, "load_model_card_yaml", reject)
        with pytest.raises(PublishError, match="would fail upload"):
            write_aux_model_card(tmp_path, "ddm_sdv", "cpn", PROVENANCE, aux_report())
        assert not (tmp_path / "model_card.yaml").exists()

    def test_the_tail_policy_quotes_the_masses_only_when_recorded(self, tmp_path):
        # forwarded_tags only forwards derive_total_mass_* when present, so a
        # card claiming they are "recorded on the training and publish runs"
        # would be false for a run without them.
        without = self.load(tmp_path, "cpn")["description"]
        assert "renormalised by the source LAN's own total" in without
        assert "carries no derive_total_mass" in without

        tags = {
            "derive_total_mass_mean": "0.998",
            "derive_total_mass_min": "0.990",
            "derive_total_mass_max": "1.0",
        }
        with_tags = self.load(tmp_path, "cpn", tags)["description"]
        assert "renormalised by the source LAN's own total" in with_tags
        assert "0.998 / 0.99 / 1" in with_tags
        assert "carries no derive_total_mass" not in with_tags

        # One tag alone is not a record; the statement stays honest.
        partial = self.load(tmp_path, "cpn", {"derive_total_mass_mean": "0.998"})
        assert "carries no derive_total_mass" in partial["description"]

    def test_the_tail_policy_reports_the_fallback_and_the_leak_when_recorded(
        self, tmp_path
    ):
        # The renormalisation is only trusted where the LAN's total sat in
        # (0.98, 1.03); the card says how much of the box fell back to
        # simulation and how much mass leaked below the onset, or that the
        # training run never recorded either.
        without = self.load(tmp_path, "cpn")["description"]
        assert "carries no derive_fallback_frac" in without
        assert "carries no derive_leak_below_onset_p99" in without
        assert "labelled by simulation because" not in without
        assert "derive_sim_past_max_t_max" not in without

        tags = {
            "derive_fallback_frac": "0.012",
            "derive_sim_past_max_t_max": "0.004",
            "derive_leak_below_onset_p99": "0.0007",
        }
        with_tags = self.load(tmp_path, "cpn", tags)["description"]
        assert (
            "1.2 % of the parameter box was labelled by simulation because the "
            "LAN's total was outside (0.98, 1.03)" in with_tags
        )
        assert "leaking below the onset is 0.0007" in with_tags
        assert "past max_t was 0.004 (derive_sim_past_max_t_max)" in with_tags
        assert "carries no derive_fallback_frac" not in with_tags
        assert "carries no derive_leak_below_onset_p99" not in with_tags

        # Each figure stands on its own: a run that recorded only the fallback
        # states the leak's absence, not a made-up value.
        only_fallback = self.load(tmp_path, "opn", {"derive_fallback_frac": "0"})
        description = only_fallback["description"]
        assert "0 % of the parameter box was labelled by simulation" in description
        assert "carries no derive_leak_below_onset_p99" in description

    def test_a_tag_stored_as_none_is_stated_absent_not_crashed_on(self, tmp_path):
        # forwarded_tags drops these before the card sees them, but a caller
        # handing the card raw run tags must get the absence sentence, not a
        # ValueError from float("None") halfway through staging.
        stringified = {
            "derive_total_mass_mean": "None",
            "derive_total_mass_min": "None",
            "derive_total_mass_max": "None",
            "derive_fallback_frac": "",
            "derive_sim_past_max_t_max": "None",
            "derive_leak_below_onset_p99": "None",
        }
        description = self.load(tmp_path, "cpn", stringified)["description"]
        assert "carries no derive_total_mass" in description
        assert "carries no derive_fallback_frac" in description
        assert "carries no derive_leak_below_onset_p99" in description
        assert "derive_sim_past_max_t_max" not in description
        assert "None" not in description

    def test_a_card_without_gate_numbers_still_renders(self, tmp_path):
        # Defensive only: the card is written after the verdict, so a report
        # with no accuracy gate never reaches it. It must not crash if one did.
        thin = {"network_type": "cpn", "gates": []}
        write_aux_model_card(tmp_path, "ddm_sdv", "cpn", PROVENANCE, thin)
        assert "n/a" in (tmp_path / "model_card.yaml").read_text()


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


class TestAuxiliaryPublish:
    """run_publish end to end for a cpn/opn: a throwaway sqlite store, a fake
    artifact folder, the validator and the uploader stubbed.

    What is real: run resolution, the provenance refusals, staging, the
    verdict, the generated card, the publish record and its back-references.
    """

    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        """Returns ``training_run(params, tags) -> run_id`` in an owned store."""
        import mlflow

        monkeypatch.chdir(tmp_path)
        previous = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
        monkeypatch.setenv("MLFLOW_ARTIFACT_LOCATION", str(tmp_path / "artifacts"))
        experiment = mlflow.create_experiment(
            "ddm_sdv-training", artifact_location=str(tmp_path / "artifacts")
        )

        def training_run(params, tags):
            with mlflow.start_run(experiment_id=experiment) as run:
                mlflow.log_params(params)
                mlflow.set_tags(tags)
            return run.info.run_id

        yield training_run
        mlflow.set_tracking_uri(previous)

    @pytest.fixture
    def stubs(self, monkeypatch):
        """Stub the validator and the uploader; returns their recorded calls."""
        import lanfactory.hf.upload as upload
        import validation.validate_network as validator

        from publish import publish_network

        calls = {"validate": [], "upload": []}

        def fake_validate(**kwargs):
            calls["validate"].append(kwargs)
            r = aux_report(kwargs["network_type"], kwargs["model_name"])
            # Honour the flag the way the real validator does: a skipped
            # accuracy gate reports passed=True with skipped=True and none of
            # the numbers. Without this the stub would let --skip-accuracy
            # complete a live publish, which is the opposite of the contract.
            if kwargs.get("skip_accuracy"):
                r["gates"][-1] = {
                    "gate": "accuracy",
                    "passed": True,
                    "skipped": True,
                    "reason": "--skip-accuracy",
                }
            return r

        def fake_upload(**kwargs):
            calls["upload"].append(kwargs)
            return "https://huggingface.co/example/HSSM_staging/commit/abc"

        monkeypatch.setattr(validator, "validate_network", fake_validate)
        monkeypatch.setattr(upload, "upload_model", fake_upload)
        monkeypatch.setattr(
            publish_network, "resolve_hf_commit", lambda *a: ("0" * 40, True)
        )
        return calls

    UUID = "e" * 32
    TAGS = {
        "run_uuid": UUID,
        "derive_total_mass_mean": "0.998",
        "derive_total_mass_min": "0.990",
        "derive_total_mass_max": "1.000",
        "derive_fallback_frac": "0.012",
        "derive_sim_past_max_t_max": "0.004",
        "derive_leak_below_onset_p99": "0.0007",
        "data_origin": "derived",
    }

    def params(self, network_type="cpn", model="ddm_sdv", **provenance):
        """A training run's params: the derive-aux keys for an auxiliary
        type, bare identity for a LAN."""
        if network_type == "lan":
            return {"model": model, "network_type": "lan", **provenance}
        category = {"cpn": "choice", "opn": "omission", "gonogo": "nogo"}[network_type]
        return {
            "model": model,
            "network_type": network_type,
            **PROVENANCE,
            "aux_category": category,
            **provenance,
        }

    def artifacts(self, tmp_path, network_type="cpn", model="ddm_sdv"):
        source = tmp_path / "trained"
        source.mkdir()
        (source / f"{model}_{network_type}_{self.UUID}_model.onnx").write_bytes(b"x")
        return source

    def publish(self, run_id, source, tmp_path, **kwargs):
        from publish.publish_network import run_publish

        return run_publish(
            hf_repo="example/HSSM_staging",
            run_id=run_id,
            artifact_dir=source,
            staging_dir=tmp_path / "staged",
            **kwargs,
        )

    @pytest.mark.parametrize("network_type", ["opn", "cpn", "lan"])
    def test_a_deadline_model_is_refused_before_anything_runs(
        self, tmp_path, store, stubs, network_type
    ):
        # HSSM asks for {base_model}{suffix}.onnx; a file named after the
        # deadline variant is unreachable, and its root filename is permanent.
        # Type-independent: the check must not narrow itself to the opn, the
        # one type a deadline is naturally associated with.
        run_id = store(self.params(network_type, model="ddm_sdv_deadline"), self.TAGS)
        source = self.artifacts(tmp_path, network_type, "ddm_sdv_deadline")

        with pytest.raises(PublishError, match="_deadline") as excinfo:
            self.publish(run_id, source, tmp_path)

        assert "'ddm_sdv'" in str(excinfo.value)
        assert stubs["validate"] == [] and stubs["upload"] == []
        assert not (tmp_path / "staged").exists()

    def test_a_run_without_provenance_is_refused_by_name(self, tmp_path, store, stubs):
        # A training run started from a simulated corpus: LANfactory logged
        # none of the derive-aux keys. Refused before staging or validation.
        params = {k: v for k, v in self.params().items() if k != "source_lan_sha256"}
        run_id = store(params, self.TAGS)
        source = self.artifacts(tmp_path)

        with pytest.raises(PublishError, match="source_lan_sha256"):
            self.publish(run_id, source, tmp_path)

        assert stubs["validate"] == [] and stubs["upload"] == []
        assert not (tmp_path / "staged").exists()

    def test_skip_accuracy_can_rehearse_but_never_publish(self, tmp_path, store, stubs):
        # The flag's contract: it shortens a dry run and can never produce an
        # upload, because accuracy is a required gate and a skipped required
        # gate is a refusal. Without --dry-run, to prove the refusal is what
        # stands in front of the uploader.
        run_id = store(self.params("cpn"), self.TAGS)
        source = self.artifacts(tmp_path, "cpn")

        result = self.publish(run_id, source, tmp_path, skip_accuracy=True)

        assert result["published"] is False
        assert (
            "accuracy" in result["error"] and "not actually checked" in result["error"]
        )
        assert stubs["upload"] == []
        # No card either: there are no gate numbers to put on one.
        assert not (tmp_path / "staged" / "model_card.yaml").exists()

    def test_a_cpn_with_provenance_publishes_under_its_root_filename(
        self, tmp_path, store, stubs
    ):
        import mlflow

        run_id = store(self.params("cpn"), self.TAGS)
        source = self.artifacts(tmp_path, "cpn")

        result = self.publish(run_id, source, tmp_path)

        assert result["published"] is True
        assert result["root_filename"] == "ddm_sdv_cpn.onnx"
        assert result["provenance"] == {**PROVENANCE, "aux_category": "choice"}
        (upload,) = stubs["upload"]
        assert (upload["network_type"], upload["model_name"]) == ("cpn", "ddm_sdv")
        assert upload["model_folder"] == tmp_path / "staged"
        # The validator was told what the cpn predicts; the verdict needed it.
        (validate,) = stubs["validate"]
        assert validate["aux_category"] == "choice"
        assert validate["network_type"] == "cpn"

        client = mlflow.MlflowClient()
        publish = client.get_run(result["publish_run_id"])
        for key, value in PROVENANCE.items():
            if key != "aux_category":
                assert publish.data.params[key] == value
        assert publish.data.params["aux_category"] == "choice"
        assert publish.data.params["source_training_run_id"] == run_id
        assert publish.data.params["source_run_uuid"] == self.UUID
        assert publish.data.metrics["gate_accuracy_mean_abs_error"] == 0.0012
        assert publish.data.metrics["gate_accuracy_max_abs_error"] == 0.0031
        assert publish.data.metrics["gate_hssm_missing_initial_logp_p0"] == -123.4
        assert publish.data.metrics["gate_hssm_missing_initial_logp_p05"] == -120.1
        assert publish.data.tags["derive_total_mass_mean"] == "0.998"
        assert publish.data.tags["derive_fallback_frac"] == "0.012"
        assert publish.data.tags["derive_sim_past_max_t_max"] == "0.004"
        assert publish.data.tags["derive_leak_below_onset_p99"] == "0.0007"
        assert publish.data.tags["data_origin"] == "derived"
        assert "run_uuid" not in publish.data.tags
        assert client.get_run(run_id).data.tags["published"] == "true"

    def test_an_opn_is_validated_with_its_provenance_category_and_its_lan(
        self, tmp_path, store, stubs
    ):
        # Provenance and validator share the output vocabulary, so the
        # provenance's "omission" is handed straight through for an opn too:
        # the validator then refuses a mislabelled run on its own.
        run_id = store(self.params("opn"), self.TAGS)
        source = self.artifacts(tmp_path, "opn")
        lan = tmp_path / "ddm_sdv.onnx"
        lan.write_bytes(b"lan")

        # dry_run: with skip_accuracy honoured by the stub this cannot publish,
        # and the passthrough is what is under test here.
        result = self.publish(
            run_id, source, tmp_path, lan_onnx=lan, skip_accuracy=True, dry_run=True
        )

        assert result["root_filename"] == "ddm_sdv_opn.onnx"
        assert result["published"] is False
        (validate,) = stubs["validate"]
        assert validate["aux_category"] == "omission"
        assert validate["lan_onnx"] == lan
        assert validate["skip_accuracy"] is True

    @pytest.mark.parametrize("with_provenance", [False, True], ids=["bare", "labelled"])
    def test_a_gonogo_run_is_refused_for_lack_of_a_consumer(
        self, tmp_path, store, stubs, with_provenance
    ):
        # Before provenance, staging or validation, and the same refusal
        # whether or not the run carries derive-aux keys: a gonogo built from
        # simulation (the common case) must not be told to relabel a network
        # that can never ship.
        params = (
            self.params("gonogo")
            if with_provenance
            else {"model": "ddm_sdv", "network_type": "gonogo"}
        )
        run_id = store(params, self.TAGS)
        source = self.artifacts(tmp_path, "gonogo")

        with pytest.raises(PublishError, match="no HSSM consumer") as excinfo:
            self.publish(run_id, source, tmp_path)

        assert "gonogo" in str(excinfo.value)
        assert "relabel" not in str(excinfo.value)
        assert stubs["validate"] == [] and stubs["upload"] == []
        assert not (tmp_path / "staged").exists()

    def test_the_cli_forwards_lan_onnx_and_skip_accuracy(self, tmp_path, store, stubs):
        # The main -> run_publish boundary is the one the operator drives;
        # the run_publish -> validate_network passthrough is pinned above.
        from typer.testing import CliRunner

        from publish import publish_network

        run_id = store(self.params("opn"), self.TAGS)
        source = self.artifacts(tmp_path, "opn")
        lan = tmp_path / "ddm_sdv.onnx"
        lan.write_bytes(b"lan")

        result = CliRunner().invoke(
            publish_network.app,
            [
                "--hf-repo",
                "example/HSSM_staging",
                "--run-id",
                run_id,
                "--artifact-dir",
                str(source),
                "--staging-dir",
                str(tmp_path / "staged"),
                "--lan-onnx",
                str(lan),
                "--skip-accuracy",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0, result.output
        (validate,) = stubs["validate"]
        assert validate["lan_onnx"] == lan
        assert validate["skip_accuracy"] is True
        plan = json.loads(result.stdout.strip().splitlines()[-1])
        assert plan["published"] is False and "accuracy" in plan["gate"]
        assert "model_card.yaml" not in plan["staged"]
        assert stubs["upload"] == []

    def test_a_dry_run_shows_the_provenance_and_the_generated_card(
        self, tmp_path, store, stubs
    ):
        # Through the CLI: the one JSON line on stdout is the contract a
        # driver reads, and the card is what a reviewer opens before the
        # real publish.
        import yaml
        from typer.testing import CliRunner

        from publish import publish_network

        run_id = store(self.params("cpn"), self.TAGS)
        source = self.artifacts(tmp_path, "cpn")
        staged = tmp_path / "staged"

        result = CliRunner().invoke(
            publish_network.app,
            [
                "--hf-repo",
                "example/HSSM_staging",
                "--run-id",
                run_id,
                "--artifact-dir",
                str(source),
                "--staging-dir",
                str(staged),
                "--dry-run",
            ],
        )

        assert result.exit_code == 0, result.output
        plan = json.loads(result.stdout.strip().splitlines()[-1])
        assert plan["dry_run"] is True and plan["published"] is False
        assert plan["provenance"] == {**PROVENANCE, "aux_category": "choice"}
        assert plan["root_filename"] == "ddm_sdv_cpn.onnx"
        assert "model_card.yaml" in plan["staged"]
        assert "validation_report.json" in plan["staged"]
        # The tags the publish run would carry, so a training run missing
        # its tail-policy record is visible before anything is uploaded.
        assert plan["forwarded_tags"] == {
            key: value for key, value in self.TAGS.items() if key != "run_uuid"
        }
        card = yaml.safe_load((staged / "model_card.yaml").read_text(encoding="utf-8"))
        assert card["title"] == "ddm_sdv (CPN)"
        # The training run's tail-policy tags reached the card, not just the
        # record.
        assert "0.998 / 0.99 / 1" in card["description"]
        assert (
            "1.2 % of the parameter box was labelled by simulation"
            in (card["description"])
        )
        assert "leaking below the onset is 0.0007" in card["description"]
        assert stubs["upload"] == []

    def test_a_dry_run_plan_omits_tags_the_training_run_lacks(
        self, tmp_path, store, stubs
    ):
        # An older run logged only the mass stats: the plan shows exactly
        # those, and the card says the rest is unrecorded rather than
        # inventing it.
        import yaml

        older = {
            key: value
            for key, value in self.TAGS.items()
            if not key.startswith(("derive_fallback", "derive_sim", "derive_leak"))
        }
        run_id = store(self.params("cpn"), older)
        source = self.artifacts(tmp_path, "cpn")

        plan = self.publish(run_id, source, tmp_path, dry_run=True)

        assert plan["forwarded_tags"] == {
            key: value for key, value in older.items() if key != "run_uuid"
        }
        assert "derive_fallback_frac" not in plan["forwarded_tags"]
        card = yaml.safe_load(
            (tmp_path / "staged" / "model_card.yaml").read_text(encoding="utf-8")
        )
        assert "carries no derive_fallback_frac" in card["description"]
        assert "carries no derive_leak_below_onset_p99" in card["description"]

    def test_an_operator_card_is_never_overwritten(self, tmp_path, store, stubs):
        run_id = store(self.params("cpn"), self.TAGS)
        source = self.artifacts(tmp_path, "cpn")
        (source / "model_card.yaml").write_text("title: ddm_sdv CPN, reviewed\n")

        self.publish(run_id, source, tmp_path, dry_run=True)

        staged_card = (tmp_path / "staged" / "model_card.yaml").read_text()
        assert staged_card == "title: ddm_sdv CPN, reviewed\n"

    def test_a_lan_plan_carries_an_empty_provenance_block(
        self, tmp_path, store, stubs, monkeypatch
    ):
        # Same key in every plan, so a driver never has to test for it.
        import validation.validate_network as validator

        monkeypatch.setattr(
            validator, "validate_network", lambda **kw: report(**ALL_RAN)
        )
        run_id = store(
            {"model": "ddm_sdv", "network_type": "lan"}, {"run_uuid": self.UUID}
        )
        source = self.artifacts(tmp_path, "lan")

        result = self.publish(run_id, source, tmp_path, dry_run=True)

        assert result["provenance"] == {}
        assert result["forwarded_tags"] == {}
        assert result["root_filename"] == "ddm_sdv.onnx"
        assert not (tmp_path / "staged" / "model_card.yaml").exists()

    def test_a_validator_refusal_is_a_publish_error(
        self, tmp_path, store, stubs, monkeypatch
    ):
        # validate_network raises ValueError on a bad flag combination; main
        # catches PublishError alone, and must still print its JSON line.
        import validation.validate_network as validator

        def refuse(**kwargs):
            raise ValueError("--aux-category is required for a cpn")

        monkeypatch.setattr(validator, "validate_network", refuse)
        run_id = store(self.params("cpn"), self.TAGS)
        source = self.artifacts(tmp_path, "cpn")

        with pytest.raises(PublishError, match="Validation refused"):
            self.publish(run_id, source, tmp_path)
