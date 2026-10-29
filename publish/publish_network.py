#!/usr/bin/env python3
"""Publish a trained network to HuggingFace, and record that it happened.

This is the step that turns a training run into a file every released HSSM
will download. It runs from the laptop, never the cluster: the HuggingFace
token is laptop-only, and the validation gate needs the laptop's HSSM
environment anyway.

The flow:

    resolve   — find the training run in MLflow by id, or by model +
                network type. The discriminator for "training actually
                finished" is the presence of the ``run_uuid`` tag, not the
                run's status: the submitting process ends its own run right
                after sbatch returns, so a job that has not started yet
                already reads FINISHED.
    stage     — copy every artifact whose name carries that run_uuid into a
                private directory. LANfactory writes all runs of a model into
                one flat folder, so publishing straight from it would upload
                other runs' files, silently disable the parity gate, and
                write a model card into the training output.
    validate  — run the gate against the staged ONNX. A gate that skipped is
                not a gate that passed; see ``gate_verdict``.
    upload    — lanfactory's dual-layout upload: the full artifact set under
                ``{network_type}/{model}/`` plus the canonical ONNX at the
                repo root under the name HSSM looks for.
    record    — a publish run in MLflow holding the source run, the resulting
                commit and the gate scores, and tags on the training run
                saying where it went.

An auxiliary network (cpn / opn) takes the same path with three differences:
the required gates are the auxiliary set (``REQUIRED_GATES_BY_NETWORK_TYPE``),
the training run must carry the provenance of the LAN it was derived from
(``aux_provenance``), and a model card is generated from that provenance and
the gate report when the operator staged none (``write_aux_model_card``).
A network type without a gate set (anything but lan / cpn / opn) and a
``_deadline`` model name are refused, the latter since HSSM builds the root
filename from the base model. Both refusals fire before anything is staged or
validated.

One caveat worth knowing: the record is written wherever MLFLOW_TRACKING_URI
points. If that is a *mirror* pulled down from the cluster, these writes live
only in the local copy and the next pull discards them. Point it at the
authoritative store when the record needs to survive.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import click
import typer

logger = logging.getLogger("publish_network")

app = typer.Typer(add_completion=False)

PUBLISH_EXPERIMENT = "publishing"

# Writing these is not reversible in any way that matters: every released HSSM
# downloads from the repo root at `main` with no revision pin, so a bad file is
# live for every user the moment it lands. Reaching it takes --allow-production
# *and* retyping the repo id at the prompt: the point is that a promotion
# cannot be collapsed into an ordinary CLI invocation, not that it is
# impossible.
PRODUCTION_REPOS = frozenset({"franklab/HSSM"})

# Parity is allowed to skip — it needs a *_train_state.jax sibling, which
# torch-trained networks legitimately do not have. The other three have no
# excuse: if one of them did not run, the network is unproven.
#
# Keyed on network type because an auxiliary network is not a density: a
# cpn/opn graph takes [θ..., choice] or [θ..., deadline] and no rt, so it can
# never clear hssm_load or density, and holding it to them made every
# auxiliary report a refusal. A type with no entry here
# has nothing to be cleared by, so it is refused.
REQUIRED_GATES_BY_NETWORK_TYPE = {
    "lan": ("structure", "hssm_load", "density"),
    "cpn": ("structure", "hssm_missing_load", "accuracy"),
    "opn": ("structure", "hssm_missing_load", "accuracy"),
}
# The LAN alias, kept for existing imports.
REQUIRED_GATES = REQUIRED_GATES_BY_NETWORK_TYPE["lan"]

# Provenance an auxiliary network carries from the LAN it was derived from.
# The names are a contract shared with LANfactory (which logs them as params
# on the training run) and the derived corpora's generator_config["source"];
# this module reads them and never invents them, so a training run that lacks
# them is refused rather than published with an unknown origin.
AUX_PROVENANCE_KEYS = (
    "derivation_method",
    "aux_category",
    "source_lan_run_uuid",
    "source_lan_sha256",
    "source_lan_hf_commit",
    "integration_grid",
    "integration_max_t",
)
# Keys that must exist on the run but may be empty: the legacy LANs on the
# Hub (the 2023 artifacts, bare ``{model}.onnx``) predate training-run uuids,
# so a network derived from one has a sha256 and a Hub commit but no
# run_uuid. LANfactory logs the key as None in that case, which MLflow stores
# as "None" (see _ABSENT_PARAM_VALUES); the card says so.
AUX_PROVENANCE_MAY_BE_EMPTY = ("source_lan_run_uuid",)
# The contract's vocabulary, and the subset this module can publish. Only a
# derived-from-lan run has the source_lan_* / integration_* keys the card is
# written from; no writer emits "trained-from-simulation" yet, and publishing
# one through this path would put a LAN lineage on a network that has none.
DERIVATION_METHODS = ("derived-from-lan", "trained-from-simulation")
PUBLISHABLE_DERIVATION_METHODS = ("derived-from-lan",)
# What MLflow stores for a param that was logged as None (log_params
# stringifies it), plus the spellings a hand-relabelled run might use.
_ABSENT_PARAM_VALUES = frozenset({"", "None", "null"})
# Tags on the training run that travel to the publish run when present: the
# tail-policy record of the derived corpus, corpus-wide over its θ — the
# total mass the source LAN put on the integration grid before its labels
# were renormalised by that total (mean / min / max), the share of θ that fell
# back to simulation because the total was outside (0.98, 1.03), the largest
# share of simulated trials past max_t among those fallback θ, the p99 of the
# mass leaking below the onset — and where the corpus came from. LANfactory logs these on the training run from the
# corpus manifest; they are forwarded when present and never invented.
AUX_FORWARDED_TAGS = (
    "derive_total_mass_mean",
    "derive_total_mass_min",
    "derive_total_mass_max",
    "derive_fallback_frac",
    "derive_sim_past_max_t_max",
    "derive_leak_below_onset_p99",
    "data_origin",
)
# What each auxiliary type's output is the probability of, in the provenance
# vocabulary. A cpn labelled "omission" was derived for something other than
# what its root filename promises, so the label is checked, not just copied.
AUX_CATEGORY_BY_NETWORK_TYPE = {"cpn": "choice", "opn": "omission"}


def _normalize_repo(hf_repo: str) -> str:
    """Strip what a copy-paste adds without naming a different repo."""
    return hf_repo.strip().strip("/")


def _stdin_is_a_terminal() -> bool:
    """Whether a person is actually at a keyboard.

    A function rather than an inline `sys.stdin.isatty()` so tests can fake it:
    click's CliRunner replaces `sys.stdin` during a run, so patching the real
    one has no effect and the check would go untested -- which for a check
    standing in front of a production write is the same as not having it.
    """
    return sys.stdin.isatty()


def _is_production(hf_repo: str) -> bool:
    """Whether this names the repo every released HSSM downloads from.

    Compared case-insensitively: HuggingFace namespaces are case-insensitively
    unique, so "Franklab/HSSM" is not some other repo -- it is this one with a
    capital letter.
    """
    return _normalize_repo(hf_repo).casefold() in {
        repo.casefold() for repo in PRODUCTION_REPOS
    }


class PublishError(RuntimeError):
    """Anything that should stop a publish before it touches HuggingFace."""


def resolve_training_run(
    run_id: str | None = None,
    model: str | None = None,
    network_type: str | None = None,
):
    """Find the training run to publish.

    By id, or by (model, network_type) taking the most recent. Searches every
    experiment: local test runs land in experiment 0 while cluster runs land
    in ``{model}-training``.
    """
    import mlflow

    if run_id:
        run = mlflow.get_run(run_id)
    else:
        if not (model and network_type):
            raise PublishError("Pass --run-id, or both --model and --network-type.")
        # MLflow's filter grammar has no working escape for the quote that
        # delimits a value, so a quote in either name produces a parser error —
        # an MlflowException, which is not a PublishError and so escapes main's
        # handler along with its print-JSON-on-stdout contract.
        if "'" in model or "'" in network_type:
            raise PublishError(
                f"Quotes are not allowed in --model or --network-type "
                f"(model={model!r}, network_type={network_type!r})."
            )
        filter_string = (
            f"params.model = '{model}' and params.network_type = '{network_type}'"
        )
        found = mlflow.search_runs(
            search_all_experiments=True,
            filter_string=filter_string,
            order_by=["attributes.start_time DESC"],
            output_format="list",
        )
        # A run without run_uuid never got as far as writing artifacts.
        found = [r for r in found if r.data.tags.get("run_uuid")]
        if not found:
            raise PublishError(
                f"No completed training run for model={model!r} "
                f"network_type={network_type!r}. Is MLFLOW_TRACKING_URI right?"
            )
        run = found[0]
        if len(found) > 1:
            logger.warning(
                f"{len(found)} matching runs; taking the most recent "
                f"({run.info.run_id}, started {run.info.start_time})."
            )

    run_uuid = run.data.tags.get("run_uuid")
    if not run_uuid:
        raise PublishError(
            f"Run {run.info.run_id} has no run_uuid tag, so its artifacts cannot "
            "be identified. Training probably never finished."
        )
    return run


MODEL_CARD = "model_card.yaml"
RECOVERY_REPORT = "recovery_report.json"

# Files the artifact folder may carry whose names do not embed the run_uuid,
# so the glob in stage_artifacts cannot find them. Copied through explicitly.
SIDECAR_FILES = (MODEL_CARD, RECOVERY_REPORT)

# Names the publish itself writes into the staging directory: the gate report,
# and the card and README lanfactory renders during upload. They are not
# leftovers from someone else's run, so finding them is not a reason to refuse
# — without this a *successful* publish poisons its own staging directory and
# the identical command can never be run again.
PUBLISH_WRITES = frozenset({"validation_report.json", *SIDECAR_FILES, "README.md"})


def upload_include_patterns(defaults: Sequence[str]) -> list[str]:
    """lanfactory's upload patterns, plus the evidence this repo produces.

    Named here rather than added to lanfactory's defaults: parameter recovery
    is this pipeline's concept, and lanfactory has no notion of it — it neither
    produces nor reads the report. Baking the filename into the uploader would
    make an upstream package carry a downstream one's vocabulary.

    The gate report is the counter-example, already in lanfactory's defaults
    for the same non-reason. Left alone here rather than moved, to keep this
    change from altering what other callers upload.
    """
    return [*defaults, RECOVERY_REPORT]


def stage_artifacts(source: Path, run_uuid: str, destination: Path) -> Path:
    """Copy one run's artifacts out of the shared training folder.

    Returns the staged ONNX. Copies rather than links: lanfactory resolves the
    canonical ONNX's parent and compares it to the folder, and ``resolve()``
    follows symlinks, so a linked file would look like it came from elsewhere.
    """
    source = Path(source)
    if not source.is_dir():
        raise PublishError(f"Artifact directory does not exist: {source}")

    if destination.exists() and not destination.is_dir():
        raise PublishError(f"Staging path {destination} is not a directory.")

    # Both trainers embed the uuid but in opposite positions —
    # jax: {uuid}_{nt}_{model}__{kind}, torch: {model}_{nt}_{uuid}_{kind} —
    # so it has to be matched anywhere in the name.
    matches = sorted(p for p in source.glob(f"*{run_uuid}*") if p.is_file())
    if not matches:
        raise PublishError(
            f"No artifacts matching run_uuid {run_uuid} in {source}. "
            "If training ran on the cluster, fetch them first and pass "
            "--artifact-dir."
        )

    # Everything in here is uploaded, so a file from a *different* run would be
    # published as part of this one and recorded in the manifest as this
    # network's artifact set. What this call is about to write is not that: a
    # dry run or a failed gate leaves exactly these names behind, and making
    # the operator clear them before the real publish buys no safety and costs
    # them the report they were about to read. Refusing rather than clearing
    # anything else — the path is user-supplied, and deleting its contents is
    # not ours to decide.
    ours = {p.name for p in matches} | PUBLISH_WRITES
    leftovers = (
        sorted(p.name for p in destination.iterdir() if p.name not in ours)
        if destination.exists()
        else []
    )
    if leftovers:
        raise PublishError(
            f"Staging directory {destination} holds files this publish did not "
            f"produce ({', '.join(leftovers[:5])}). Its whole contents get "
            "uploaded, so they would be published as part of this network. "
            "Remove them or pass a different --staging-dir."
        )

    destination.mkdir(parents=True, exist_ok=True)
    for path in matches:
        shutil.copy2(path, destination / path.name)

    # Sidecars the artifact folder may carry. Neither name embeds the run_uuid,
    # so the glob above cannot find them.
    #
    # The card: without it lanfactory generates a default whose usage example
    # assumes the model is already a built-in HSSM name, which does not run for
    # a model published before its HSSM config ships.
    #
    # The recovery report: validation_report.json cannot see a recovery
    # failure — the gate has no recovery check in it — so without this the only
    # evidence travelling with the network is the one that could not have
    # caught the problem.
    # Copy-or-delete, not copy-if-present: the staging directory survives
    # between runs (PUBLISH_WRITES exists so a re-run does not refuse its own
    # leftovers), so a sidecar staged last time and since removed from the
    # source would otherwise stay behind and upload as if it described this
    # network. For the recovery report that is the exact failure this publish
    # step exists to prevent -- shipping evidence that belongs to another run.
    for name in SIDECAR_FILES:
        staged = destination / name
        if staged.is_symlink() or staged.is_dir():
            # Guarding both branches at once: unlink() raises on a directory
            # regardless of missing_ok, and copy2() onto one silently copies
            # INTO it -- producing model_card.yaml/model_card.yaml that the
            # upload would then miss. A symlink is worse: copy2() follows it
            # and overwrites whatever it points at, which for a reused
            # --staging-dir can be a file outside the staging area entirely.
            # Either way the operator put it there, so the operator removes
            # it. Symlink first: a link to a directory passes is_dir() too,
            # and the message should name what is actually on disk.
            kind = "symlink" if staged.is_symlink() else "directory"
            raise PublishError(
                f"{staged} is a {kind}, not a staged sidecar. "
                "Remove it or pass a different --staging-dir."
            )
        sidecar = source / name
        if sidecar.is_file():
            shutil.copy2(sidecar, staged)
        else:
            staged.unlink(missing_ok=True)

    onnx = [p for p in destination.iterdir() if p.suffix == ".onnx"]
    if len(onnx) != 1:
        raise PublishError(
            f"Expected exactly one .onnx among the {len(matches)} staged "
            f"artifacts, found {len(onnx)}: {[p.name for p in onnx]}"
        )

    # The report is selected by fixed filename from a directory that can hold
    # several runs' artifacts, while the ONNX is selected by run_uuid -- so
    # without this check an explicit --run-id publishes one run's network
    # with another run's recovery verdict. The aggregator records which ONNX
    # files its shards fit; the one travelling beside the report must be
    # among them.
    report_path = destination / RECOVERY_REPORT
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError as e:
            raise PublishError(f"{report_path} is not valid JSON: {e}") from e
        judged = report.get("onnx_files") if isinstance(report, dict) else None
        # The shape checks are load-bearing, not defensive fluff: a string
        # here would turn the membership test below into substring matching,
        # which can false-ACCEPT -- "..._model.onnx" is a substring of a
        # longer artifact name. Only a list makes `in` mean what it says.
        if not isinstance(judged, list) or not judged:
            raise PublishError(
                f"{report_path} does not say which network it judged "
                "(onnx_files missing, empty, or not a list). Re-run "
                "validation/aggregate_recovery.py over this run's shards so "
                "the report is bound to its network."
            )
        if onnx[0].name not in judged:
            raise PublishError(
                f"{report_path} judges {judged}, not the staged "
                f"{onnx[0].name}. It belongs to another run -- aggregate "
                "this run's shards instead."
            )

    logger.info(f"Staged {len(matches)} artifacts to {destination}")
    return onnx[0]


def gate_verdict(report: dict) -> tuple[bool, str]:
    """Decide whether a validation report clears the network for publishing.

    Not the same as ``report["passed"]``: a skipped gate reports passed=True,
    so a report where everything skipped is "passed" and proves nothing.

    Which gates are required depends on ``report["network_type"]``. A report
    without one predates the auxiliary gate set and cannot say which set it
    was judged by, so it is refused rather than assumed to be a LAN's.
    """
    network_type = report.get("network_type")
    if network_type is None:
        return False, "report has no network_type; re-run the validator (P1 or later)"
    required = REQUIRED_GATES_BY_NETWORK_TYPE.get(network_type)
    if required is None:
        return False, f"no publishable gate set for network_type {network_type!r}"

    # .get throughout: the reports this function is defending against are the
    # malformed ones, so a missing "gates" key or a gate with no "passed" has
    # to come out as a refusal, not a KeyError that main does not catch.
    gates = {g["gate"]: g for g in report.get("gates", [])}
    failed = [name for name, g in gates.items() if not g.get("passed")]
    if failed:
        details = "; ".join(
            f"{n}: {gates[n].get('error', 'did not pass')}" for n in failed
        )
        return False, f"gate failed — {details}"

    # Absent counts the same as skipped. A report missing a gate entirely — a
    # schema change, a truncated file — is exactly the "looks passed, proves
    # nothing" case this function exists to catch.
    absent = {"skipped": True}
    unchecked = [n for n in required if gates.get(n, absent).get("skipped")]
    if unchecked:
        return False, (
            f"not actually checked: {', '.join(unchecked)}. "
            "A skipped or missing gate is not a passed gate."
        )
    return True, "all required gates ran and passed"


def _present(value: object) -> bool:
    """Whether a param value carries information.

    MLflow stores a param logged as ``None`` as the string ``"None"``, so a
    corpus derived from a local ONNX with no Hub revision would otherwise pass
    the required-key check and publish with ``source_lan_hf_commit="None"`` on
    the run and on the card.
    """
    return value is not None and str(value).strip() not in _ABSENT_PARAM_VALUES


def aux_provenance(params: Mapping[str, str], network_type: str) -> dict[str, str]:
    """The derivation provenance an auxiliary network must publish with.

    Read from the training run's params, where LANfactory logs them for a run
    started from a derived corpus. A LAN (or anything else that is not a
    cpn/opn) has none and gets ``{}``. A cpn/opn run without them is refused
    by the first missing key: a run trained from a simulated corpus carries
    none of these, and until it is relabelled its network has no source LAN
    to be traced to. An empty, ``None`` or ``null`` value counts as missing,
    except for ``source_lan_run_uuid``, which a legacy LAN legitimately lacks
    (``AUX_PROVENANCE_MAY_BE_EMPTY``): the key must still exist on the run, and
    an absent value is normalised to ``""``.
    """
    if network_type not in AUX_CATEGORY_BY_NETWORK_TYPE:
        return {}
    missing = [
        key
        for key in AUX_PROVENANCE_KEYS
        if (key not in params)
        or (key not in AUX_PROVENANCE_MAY_BE_EMPTY and not _present(params[key]))
    ]
    if missing:
        raise PublishError(
            f"Training run has no {missing[0]!r} param, so the {network_type}'s "
            "provenance is unknown. A run started from a simulated corpus "
            "carries none of the derive-aux keys "
            f"({', '.join(AUX_PROVENANCE_KEYS)}); relabel it with the source "
            "LAN's identity before publishing."
        )
    provenance = {
        key: str(params[key]) if _present(params[key]) else ""
        for key in AUX_PROVENANCE_KEYS
    }
    method = provenance["derivation_method"]
    if method not in DERIVATION_METHODS:
        raise PublishError(
            f"derivation_method {method!r} is not one of {list(DERIVATION_METHODS)}."
        )
    if method not in PUBLISHABLE_DERIVATION_METHODS:
        raise PublishError(
            f"derivation_method {method!r} has no publish path yet: the "
            "generated card names the source LAN's sha256, run_uuid, Hub "
            "commit and integration grid, which a network trained from "
            "simulation does not have. Only "
            f"{', '.join(PUBLISHABLE_DERIVATION_METHODS)} publishes."
        )
    expected = AUX_CATEGORY_BY_NETWORK_TYPE[network_type]
    if provenance["aux_category"] != expected:
        raise PublishError(
            f"Training run says aux_category {provenance['aux_category']!r}, but "
            f"a {network_type} is published as the probability of {expected!r}. "
            "The network was derived for something other than its root "
            "filename would promise."
        )
    return provenance


def forwarded_tags(tags: Mapping[str, str]) -> dict[str, str]:
    """The training-run tags that travel to the publish run, when present.

    A tag MLflow stored as ``"None"`` or ``""`` is absent, not a value:
    LANfactory stringifies every derive_stats entry, and
    ``derive_sim_past_max_t_max`` is ``None`` when no θ in the corpus fell
    back to simulation, so that tag arrives as the string ``"None"``. Forwarding it verbatim would put
    a ``"None"`` tag on the publish run and crash the card at ``float()``.
    """
    return {
        key: str(tags[key])
        for key in AUX_FORWARDED_TAGS
        if key in tags and _present(tags[key])
    }


def _card_number(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}g}" if value is not None else "n/a"


def _dump_card(card: Mapping[str, object]) -> str:
    """The card as YAML, multi-line strings as ``|`` blocks.

    The card is reviewed by eye before the real publish, and a description
    folded into a quoted scalar with doubled apostrophes is not something a
    reviewer can read. yaml is imported here, not at module level, so the
    publisher's import-time surface stays what it was before cards.
    """
    import yaml

    class BlockScalarDumper(yaml.SafeDumper):
        pass

    def represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
        style = "|" if "\n" in value else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)

    BlockScalarDumper.add_representer(str, represent_str)
    return yaml.dump(
        dict(card), Dumper=BlockScalarDumper, sort_keys=False, allow_unicode=True
    )


def write_aux_model_card(
    staging_dir: Path,
    model: str,
    network_type: str,
    provenance: Mapping[str, str],
    report: Mapping,
    training_tags: Mapping[str, str] | None = None,
) -> Path:
    """Write ``model_card.yaml`` for an auxiliary network into the staging dir.

    Only for runs that staged no operator card. Carries exactly the keys
    LANfactory's card renderer reads (title, tags, library_name, license,
    description, usage_example); architecture and training are left out so
    the renderer fills them from the pickled configs, which are the
    authoritative record. ``training_tags`` are the tags forwarded from the
    training run (``forwarded_tags``): the ``derive_total_mass_*`` values, the
    ``derive_fallback_frac`` and the ``derive_leak_below_onset_p99`` are
    quoted in the tail-policy paragraph when present, and their absence is
    stated rather than papered over. The result is round-tripped through
    LANfactory's loader before it is accepted, so a card that would crash the
    upload is caught here, on the laptop, not halfway through a commit to the
    Hub.
    """
    from lanfactory.hf import DEFAULT_LICENSE

    gates = {g["gate"]: g for g in report.get("gates", [])}
    accuracy = gates.get("accuracy", {})
    draws = accuracy.get("draws") or []
    truth_se = max((d.get("truth_mc_se", 0.0) for d in draws), default=None)
    kind = network_type.upper()
    grid, max_t = provenance["integration_grid"], provenance["integration_max_t"]

    if network_type == "cpn":
        what = (
            f"Choice-probability network (CPN) for the ssm-simulators model "
            f"`{model}`: log P(choice | θ), the probability that the process "
            f"terminates at `choice` within {max_t} s. HSSM uses it as "
            "`loglik_missing_data` on rows whose RT is missing but whose "
            "response is known."
        )
        contract = (
            "Input: one row of width n_params + 1, `[θ in list_params order, "
            "choice]` — the trailing column is the choice code."
        )
        usage = (
            "import hssm\n"
            "\n"
            "# Rows with a known response but no RT carry rt == -999.0.\n"
            'data.loc[data["rt"].isna(), "rt"] = -999.0\n'
            "model = hssm.HSSM(\n"
            "    data,\n"
            f'    model="{model}",\n'
            '    loglik_kind="approx_differentiable",\n'
            f"    missing_data=True,  # downloads {model}_cpn.onnx\n"
            "    p_outlier=0.05,\n"
            ")\n"
        )
    else:
        what = (
            f"Omission-probability network (OPN) for the ssm-simulators model "
            f"`{model}`: log P(rt > deadline | θ), the survival function of the "
            "base model at the deadline. HSSM uses it as `loglik_missing_data` "
            "on trials that outlasted their deadline."
        )
        contract = (
            "Input: one row of width n_params + 1, `[θ in list_params order, "
            "deadline]` — the trailing column is the deadline in seconds."
        )
        usage = (
            "import hssm\n"
            "\n"
            "# One `deadline` column per trial; a trial that outlasted it\n"
            "# carries rt == -999.0.\n"
            "model = hssm.HSSM(\n"
            "    data,\n"
            f'    model="{model}",\n'
            '    loglik_kind="approx_differentiable",\n'
            "    missing_data=True,\n"
            f"    deadline=True,  # downloads {model}_opn.onnx\n"
            ")\n"
        )

    # The tail-policy record, only claimed when the training run carries it:
    # forwarded_tags forwards these tags when present, so a card saying they
    # are "recorded on the training and publish runs" would be false for a
    # run that has none. The total-mass stats, the fallback fraction and the
    # onset leak are each stated or their absence is: the mass stats say how
    # far the source LAN was from normalised before its labels were
    # renormalised, the fallback fraction says how much of the box that
    # renormalisation was not trusted for, and the leak says how much mass sat
    # below the onset. The mass past max_t is quoted only when present. A tag
    # stored as "None" (see forwarded_tags) counts as absent here too, so a
    # caller handing the card raw run tags gets an absence sentence, not a
    # ValueError.
    tags = {
        key: value for key, value in (training_tags or {}).items() if _present(value)
    }
    masses = {
        stat: tags.get(f"derive_total_mass_{stat}") for stat in ("mean", "min", "max")
    }
    if all(masses.values()):
        tail_record = [
            (
                "the derived corpus's total mass before renormalisation, over "
                "its θ, is recorded on the training and publish runs as "
                "derive_total_mass_{mean,min,max} = "
                f"{_card_number(masses['mean'])} / {_card_number(masses['min'])} / "
                f"{_card_number(masses['max'])}."
            )
        ]
    else:
        tail_record = [
            (
                "the training run carries no derive_total_mass_{mean,min,max} "
                "tags, so the derived corpus's total mass is not recorded "
                "here."
            )
        ]
    fallback = tags.get("derive_fallback_frac")
    if fallback is not None:
        tail_record.append(
            f"{_card_number(float(fallback) * 100, 3)} % of the parameter box was "
            "labelled by simulation because the LAN's total was outside "
            "(0.98, 1.03) (derive_fallback_frac)."
        )
    else:
        tail_record.append(
            "The training run carries no derive_fallback_frac tag, so the share "
            "of the parameter box labelled by simulation is not recorded here."
        )
    sim_past = tags.get("derive_sim_past_max_t_max")
    if sim_past is not None:
        tail_record.append(
            "Of the θ labelled by simulation, the largest share of trials "
            f"past max_t was {_card_number(sim_past)} "
            "(derive_sim_past_max_t_max)."
        )
    leak = tags.get("derive_leak_below_onset_p99")
    if leak is not None:
        tail_record.append(
            "The p99, over θ, of the mass leaking below the onset is "
            f"{_card_number(leak)} (derive_leak_below_onset_p99)."
        )
    else:
        tail_record.append(
            "The training run carries no derive_leak_below_onset_p99 tag, so the "
            "mass leaking below the onset is not recorded here."
        )

    run_uuid = (
        provenance["source_lan_run_uuid"]
        or "unknown (a legacy artifact without a training run)"
    )
    description = "\n\n".join(
        [
            what,
            contract
            + (
                " Output: one log-probability (every value ≤ 0; the log-sigmoid "
                "is baked into the graph). HSSM applies the lapse mixture "
                "outside the network."
            ),
            (
                f"Derivation: {provenance['derivation_method']} — from the "
                f"`{model}` LAN with sha256 {provenance['source_lan_sha256']}, "
                f"training run_uuid {run_uuid}, Hub commit "
                f"{provenance['source_lan_hf_commit']}; integrated on a "
                f"{grid}-point grid up to max_t = {max_t} s."
            ),
            (
                f"Validation (accuracy gate): mean |network − truth| "
                f"{_card_number(accuracy.get('mean_abs_error'))}, max "
                f"{_card_number(accuracy.get('max_abs_error'))} over "
                f"{accuracy.get('n_param_draws', 'n/a')} parameter draws, against a "
                f"Monte-Carlo truth with standard error ≤ {_card_number(truth_se)} "
                f"(n_sim = {accuracy.get('n_sim', 'n/a')} per draw)."
            ),
            "Tail policy: labels were renormalised by the source LAN's own total "
            "on the integration grid, so the LAN's mass past max_t is folded "
            "back in by that total rather than dropped; " + " ".join(tail_record),
        ]
    )
    card = {
        "title": f"{model} ({kind})",
        "tags": [
            network_type,
            "ssm",
            "hssm",
            "missing-data",
            provenance["derivation_method"],
        ],
        "library_name": "onnx",
        "license": DEFAULT_LICENSE,
        "description": description,
        "usage_example": usage,
    }
    path = Path(staging_dir) / MODEL_CARD
    # utf-8 explicitly: the description carries θ, ≤ and a real minus sign,
    # and a UnicodeEncodeError under a non-UTF-8 locale is not a PublishError,
    # so it would escape main's handler without the JSON line.
    path.write_text(_dump_card(card), encoding="utf-8")

    # The loader that will run at upload time, on the file it will read.
    try:
        from lanfactory.hf.model_card import generate_readme, load_model_card_yaml
    except ImportError:  # pragma: no cover - older lanfactory without the module
        return path
    try:
        generate_readme(load_model_card_yaml(Path(staging_dir)), model)
    except Exception as e:  # whatever it is, upload would hit it
        path.unlink(missing_ok=True)
        raise PublishError(f"The generated model card would fail upload: {e}") from e
    return path


def resolve_hf_commit(repo_id: str, commit_message: str) -> tuple[str | None, bool]:
    """Recover the sha of the commit just made, and say whether it is certain.

    lanfactory's upload returns a browser URL and discards the CommitInfo, so
    the sha has to be read back. Matching on the commit message turns the
    read-back race into something detectable rather than a quietly wrong sha.
    """
    from huggingface_hub import HfApi

    try:
        head = HfApi().list_repo_commits(repo_id)[0]
    except Exception as e:  # noqa: BLE001 - a missing sha must not fail a publish
        logger.warning(f"Could not read back the commit sha: {e}")
        return None, False
    return head.commit_id, head.title == commit_message


def _publish_experiment_id(artifact_location: str | None) -> str:
    """The publishing experiment, with an absolute artifact location.

    Created implicitly, MLflow would bake in a path relative to whatever
    directory this happened to run from, and reports published from elsewhere
    would land somewhere else.
    """
    import mlflow

    experiment = mlflow.get_experiment_by_name(PUBLISH_EXPERIMENT)
    if experiment is not None:
        return experiment.experiment_id
    location = str(Path(artifact_location).absolute()) if artifact_location else None
    return mlflow.create_experiment(PUBLISH_EXPERIMENT, artifact_location=location)


def publish_network(
    onnx_path: Path,
    model: str,
    network_type: str,
    repo_id: str,
    training_run_id: str | None = None,
    run_uuid: str | None = None,
    report: dict | None = None,
    hf_url: str | None = None,
    hf_commit: str | None = None,
    hf_commit_verified: bool = False,
    artifact_location: str | None = None,
    provenance: Mapping[str, str] | None = None,
    training_tags: Mapping[str, str] | None = None,
) -> str:
    """Record the publish in MLflow and stamp the training run. Returns run id.

    ``provenance`` (an auxiliary network's derive-aux keys) is logged as params
    beside the source run identity; ``training_tags`` (the derive_* tail-policy
    record and data_origin tags forwarded from the training run,
    ``AUX_FORWARDED_TAGS``) as tags.
    """
    import mlflow

    published_at = datetime.now(UTC).isoformat()
    # An unconfirmed sha is a lead, not a fact. Recording it under hf_commit
    # would put a possibly-wrong value in the field everything else trusts, so
    # it goes somewhere that reads as uncertain.
    commit_key = "hf_commit" if hf_commit_verified else "hf_commit_candidate"

    with mlflow.start_run(
        experiment_id=_publish_experiment_id(artifact_location),
        run_name=f"publish-{model}-{network_type}",
    ) as publish_run:
        mlflow.log_params(
            {
                "model": model,
                "network_type": network_type,
                "hf_repo": repo_id,
                commit_key: hf_commit or "unknown",
                "source_training_run_id": training_run_id or "unknown",
                "source_run_uuid": run_uuid or "unknown",
                "onnx_filename": Path(onnx_path).name,
                **dict(provenance or {}),
            }
        )
        mlflow.set_tags(
            {
                "schema_version": "1",
                "phase": "publish",
                "hf_commit_verified": str(hf_commit_verified).lower(),
                "published_at": published_at,
                **dict(training_tags or {}),
            }
        )
        if hf_url:
            mlflow.set_tag("hf_url", hf_url)

        if report:
            mlflow.log_dict(report, "validation_report.json")
            gates = {g["gate"]: g for g in report["gates"]}
            # .get throughout: a skipped or errored gate carries no scores.
            missing_logp = gates.get("hssm_missing_load", {}).get(
                "initial_logp_by_p_outlier", {}
            )
            scores = {
                "gate_parity_max_abs_error": gates.get("parity", {}).get(
                    "max_abs_error"
                ),
                "gate_hssm_initial_logp": gates.get("hssm_load", {}).get(
                    "initial_logp"
                ),
                "gate_density_worst_ratio": gates.get("density", {}).get("worst_ratio"),
                "gate_density_worst_mass": gates.get("density", {}).get(
                    "worst_total_mass"
                ),
                "gate_accuracy_mean_abs_error": gates.get("accuracy", {}).get(
                    "mean_abs_error"
                ),
                "gate_accuracy_max_abs_error": gates.get("accuracy", {}).get(
                    "max_abs_error"
                ),
                "gate_hssm_missing_initial_logp_p0": missing_logp.get("0.0"),
                "gate_hssm_missing_initial_logp_p05": missing_logp.get("0.05"),
            }
            for key, value in scores.items():
                if value is not None:
                    mlflow.log_metric(key, float(value))
            mlflow.set_tag(
                "gates_run",
                ",".join(g["gate"] for g in report["gates"] if not g.get("skipped")),
            )

        publish_run_id = publish_run.info.run_id

    if training_run_id:
        # Back-references only — the publish run above already records which
        # training run this came from, and the upload is already live. A store
        # that refuses these writes (a read-only mirror, a deleted run) must
        # not cost the caller the hf_url of a publish that did succeed: main
        # catches PublishError alone, so anything else here would exit with a
        # traceback and no JSON at all.
        try:
            client = mlflow.MlflowClient()
            client.set_tag(training_run_id, "published", "true")
            client.set_tag(training_run_id, "published_at", published_at)
            client.set_tag(training_run_id, "publish_run_id", publish_run_id)
            if hf_commit:
                client.set_tag(training_run_id, commit_key, hf_commit)
            client.set_tag(training_run_id, "hf_repo", repo_id)
        except Exception as e:  # noqa: BLE001 - the upload already happened
            logger.warning(
                f"Published, but could not stamp training run {training_run_id}: {e}"
            )

    return publish_run_id


@app.command()
def main(
    hf_repo: str = typer.Option(
        ...,
        help="Target HuggingFace repo, e.g. franklab/HSSM_staging. No default: "
        "publishing to the wrong repo is not undoable.",
    ),
    run_id: str = typer.Option(None, help="MLflow training run id to publish."),
    model: str = typer.Option(None, help="Model name, if not using --run-id."),
    network_type: str = typer.Option(
        None,
        help="lan | cpn | opn. Any other type is refused before anything is staged.",
    ),
    artifact_dir: Path = typer.Option(
        None,
        help="Folder holding the trained artifacts. Defaults to the training "
        "run's output_path, which only works if that path exists locally.",
    ),
    staging_dir: Path = typer.Option(
        None, help="Where to assemble this run's files [default: a temp dir]."
    ),
    lan_onnx: Path = typer.Option(
        None,
        help="cpn/opn only: the base LAN the hssm_missing_load gate pairs the "
        "network with [default: HSSM resolves it by model name]. Required for "
        "models outside HSSM's registry.",
    ),
    skip_density: bool = typer.Option(
        False,
        # Density is a required gate, so skipping it makes gate_verdict refuse.
        # The flag cannot produce a publish at all — it only saves the cost of
        # the slowest gate while dry-running the resolve/stage/plan path.
        help="Skip G4. No publish is possible with this set: it shortens a "
        "--dry-run, which then exits 1.",
    ),
    skip_accuracy: bool = typer.Option(
        False,
        help="Skip A4, the auxiliary analogue of --skip-density. No publish is "
        "possible with this set: it shortens a --dry-run, which then exits 1.",
    ),
    dry_run: bool = typer.Option(
        False, help="Validate and show the plan; touch neither HF nor MLflow."
    ),
    overwrite_root: bool = typer.Option(
        False,
        help="Replace an existing root {model}.onnx. Needed to republish a "
        "model that is already in the target repo.",
    ),
    allow_production: bool = typer.Option(
        False,
        help="Permit writing to the production repo. Requires retyping the "
        "repo id at an interactive prompt -- promoting is a review decision, "
        "not a flag.",
    ),
    log_level: str = typer.Option("INFO"),
):
    """Validate a trained network and publish it to HuggingFace."""
    level = getattr(logging, str(log_level).upper(), None)
    if not isinstance(level, int):
        raise typer.BadParameter(f"Unknown log level {log_level!r}.")
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")

    # A flag on its own is exactly the "ordinary CLI invocation" the refusal
    # exists to prevent: it survives shell history, a copied runbook line, and
    # a re-run of the wrong command. Retyping the repo id does not. Skipped
    # under --dry-run, which touches neither HF nor MLflow -- it still writes
    # local staging files and the gate report, but nothing irreversible.
    if allow_production and not dry_run and _is_production(hf_repo):
        # The prompt is only a check if a person answers it. Piping the answer
        # in -- `echo franklab/HSSM | lan-publish ... --allow-production` -- is
        # a script, which is precisely what this is guarding against, so a
        # non-terminal stdin is refused before anything is asked.
        error = None
        if not _stdin_is_a_terminal():
            error = (
                "--allow-production needs an interactive terminal: the "
                "confirmation is what makes a promotion deliberate, and piped "
                "input is not a person. Run it from a terminal."
            )
        else:
            try:
                typed = typer.prompt(f"Retype {_normalize_repo(hf_repo)} to confirm")
            except (click.Abort, EOFError):
                # Ctrl-C or a closed stdin. Without this the CLI dies on a
                # traceback and skips the JSON line every caller parses.
                typed = ""
            if _normalize_repo(typed).casefold() != _normalize_repo(hf_repo).casefold():
                error = "production confirmation did not match; nothing was written"
        if error is not None:
            logger.error(error)
            print(json.dumps({"published": False, "error": error}))
            raise typer.Exit(code=1)

    try:
        result = run_publish(
            hf_repo=hf_repo,
            run_id=run_id,
            model=model,
            network_type=network_type,
            artifact_dir=artifact_dir,
            staging_dir=staging_dir,
            lan_onnx=lan_onnx,
            skip_density=skip_density,
            skip_accuracy=skip_accuracy,
            dry_run=dry_run,
            overwrite_root=overwrite_root,
            allow_production=allow_production,
        )
    except PublishError as e:
        logger.error(str(e))
        print(json.dumps({"published": False, "error": str(e)}))
        raise typer.Exit(code=1) from e

    print(json.dumps(result))
    # A refusal fails a dry run too: the dry run is the rehearsal, and one that
    # exits 0 on a refused gate reads as a green light to whoever scripts it.
    if "error" in result or not (result["published"] or dry_run):
        raise typer.Exit(code=1)


def run_publish(
    hf_repo: str,
    run_id: str | None = None,
    model: str | None = None,
    network_type: str | None = None,
    artifact_dir: Path | None = None,
    staging_dir: Path | None = None,
    lan_onnx: Path | None = None,
    skip_density: bool = False,
    skip_accuracy: bool = False,
    dry_run: bool = False,
    overwrite_root: bool = False,
    allow_production: bool = False,
) -> dict:
    """The whole flow, importable so it can be tested without a CLI."""
    # Before any import: a safety check that an ImportError can preempt is not
    # a safety check, and this way the refusal is testable without the whole
    # inference stack installed.
    # Normalized once and used from here on, not compared raw: HuggingFace
    # namespaces are case-insensitively unique, so "Franklab/HSSM" is not some
    # other repo — it is this one with a capital letter, and a raw string match
    # lets it walk past the only check standing in front of an irreversible
    # write. Surrounding whitespace and a trailing slash survive a copy-paste
    # and name the same repo too.
    hf_repo = _normalize_repo(hf_repo)
    if _is_production(hf_repo) and not allow_production:
        raise PublishError(
            f"{hf_repo} is the production repo every released HSSM downloads "
            "from. Publish to a staging repo and promote deliberately."
        )

    import tempfile

    from lanfactory.hf.upload import (
        DEFAULT_INCLUDE_PATTERNS,
        RootArtifactExistsError,
        canonical_root_filename,
        upload_model,
    )

    from validation.validate_network import validate_network

    run = resolve_training_run(run_id=run_id, model=model, network_type=network_type)
    run_uuid = run.data.tags["run_uuid"]
    # From params, not the filename: the trainers derive network_type from the
    # output layer and can write 'unknown' into the name.
    model = run.data.params.get("model", model)
    network_type = run.data.params.get("network_type", network_type)
    # network_type has a closed set to check against; model has none, but it
    # has to at least exist — it becomes a path segment and the root filename
    # HSSM downloads by, so an unset one gets as far as a TypeError building
    # the artifact path, or a network published as "None.onnx".
    if not model:
        raise PublishError(
            f"Run {run.info.run_id} records no model param and none was given. "
            "Pass --model."
        )
    # Before provenance, staging or validation: a type with no gate set (a
    # gonogo, say, which lanfactory.hf still names) can never ship whatever its
    # run carries, so it must not be told to relabel its provenance.
    # gate_verdict refuses it again for a hand-fed report.
    if network_type not in REQUIRED_GATES_BY_NETWORK_TYPE:
        raise PublishError(
            f"network_type {network_type!r} is not one of "
            f"{list(REQUIRED_GATES_BY_NETWORK_TYPE)}."
        )
    # The root filename is always {base_model}{suffix}.onnx: HSSM builds it
    # from the model name it was given plus the type's suffix, and nothing
    # ever asks for ddm_deadline_opn.onnx. A network published under that
    # name is unreachable — and its root filename is permanent.
    if model.endswith("_deadline"):
        raise PublishError(
            f"model {model!r} is a _deadline variant; publish under the base "
            f"model {model[: -len('_deadline')]!r}. The root filename is "
            "always {base_model}_{network_type}.onnx, which is what HSSM "
            "downloads."
        )
    # Before staging: a run with no provenance is refused whatever its ONNX
    # says, so there is no point copying and validating it first.
    provenance = aux_provenance(run.data.params, network_type)
    training_tags = forwarded_tags(run.data.tags)
    logger.info(
        f"Publishing {model}/{network_type} from run {run.info.run_id} "
        f"(run_uuid {run_uuid})"
    )

    source = Path(artifact_dir) if artifact_dir else None
    if source is None:
        output_path = run.data.params.get("output_path")
        if not output_path:
            raise PublishError(
                "The training run records no output_path; pass --artifact-dir."
            )
        source = Path(output_path) / network_type / model
        if not source.is_dir():
            raise PublishError(
                f"The run's artifacts are at {source}, which does not exist on "
                "this machine — they are probably still on the cluster. Fetch "
                "them and pass --artifact-dir."
            )

    with tempfile.TemporaryDirectory(prefix="lan-publish-") as tmp:
        staging = Path(staging_dir) if staging_dir else Path(tmp) / "staged"
        onnx_path = stage_artifacts(source, run_uuid, staging)

        logger.info(f"Validating {onnx_path.name}")
        try:
            report = validate_network(
                onnx_path=onnx_path,
                model_name=model,
                network_type=network_type,
                skip_density=skip_density,
                # Provenance and validator share one vocabulary (what the
                # output is the probability of), so the validator's own check
                # agrees with aux_provenance's by construction.
                aux_category=provenance.get("aux_category"),
                lan_onnx=lan_onnx,
                skip_accuracy=skip_accuracy,
            )
        except ValueError as e:
            raise PublishError(f"Validation refused the candidate: {e}") from e
        # Written into the staging dir so it is uploaded alongside the network.
        (staging / "validation_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )

        ok, reason = gate_verdict(report)
        # Only once the gate numbers exist to put on it, and only when the
        # operator did not stage a card of their own (stage_artifacts has
        # already copied or removed that one).
        if ok and provenance and not (staging / MODEL_CARD).exists():
            write_aux_model_card(
                staging, model, network_type, provenance, report, training_tags
            )
        root_name = canonical_root_filename(network_type, model)
        plan = {
            "model": model,
            "network_type": network_type,
            "hf_repo": hf_repo,
            "root_filename": root_name,
            "training_run_id": run.info.run_id,
            "run_uuid": run_uuid,
            "staged": sorted(p.name for p in staging.iterdir()),
            "gate": reason,
            "provenance": provenance,
            # What the publish run will carry as tags, shown on the plan so a
            # dry run reveals a training run that lacks the tail-policy
            # record before anything is uploaded.
            "forwarded_tags": training_tags,
        }
        if not ok:
            logger.error(f"Not publishing: {reason}")
            return {"published": False, "error": reason, **plan}

        if dry_run:
            logger.info(f"Dry run: would publish {root_name} to {hf_repo}")
            return {"published": False, "dry_run": True, **plan}

        # Unique enough to identify on read-back, since the sha is not returned.
        commit_message = f"Publish {model} ({network_type}) from run {run.info.run_id}"
        try:
            hf_url = upload_model(
                model_folder=staging,
                network_type=network_type,
                model_name=model,
                repo_id=hf_repo,
                commit_message=commit_message,
                overwrite_root=overwrite_root,
                include_patterns=upload_include_patterns(DEFAULT_INCLUDE_PATTERNS),
            )
        except RootArtifactExistsError as e:
            # Not re-raising lanfactory's text: it is written for lanfactory's
            # own CLI and suggests flags this one does not have, including
            # "publish to a staging repo" when we may already be doing that.
            raise PublishError(
                f"{hf_repo} already has {root_name} at its root. Re-run with "
                "--overwrite-root to replace it."
            ) from e

        hf_commit, verified = resolve_hf_commit(hf_repo, commit_message)
        if hf_commit and not verified:
            logger.warning(
                "The repo head does not match the commit just made — someone "
                "else pushed in between. Recorded sha may not be this upload."
            )

        publish_run_id = publish_network(
            onnx_path=onnx_path,
            model=model,
            network_type=network_type,
            repo_id=hf_repo,
            training_run_id=run.info.run_id,
            run_uuid=run_uuid,
            report=report,
            hf_url=hf_url,
            hf_commit=hf_commit,
            hf_commit_verified=verified,
            artifact_location=os.environ.get("MLFLOW_ARTIFACT_LOCATION"),
            provenance=provenance,
            training_tags=training_tags,
        )

    return {
        "published": True,
        "hf_url": hf_url,
        # Same rule as the MLflow record: a driver reading hf_commit gets a sha
        # that was confirmed, or no hf_commit at all.
        ("hf_commit" if verified else "hf_commit_candidate"): hf_commit,
        "hf_commit_verified": verified,
        "publish_run_id": publish_run_id,
        **plan,
    }


if __name__ == "__main__":
    app()
