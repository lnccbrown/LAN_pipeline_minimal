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
from datetime import datetime, timezone
from pathlib import Path

import typer

logger = logging.getLogger("publish_network")

app = typer.Typer(add_completion=False)

PUBLISH_EXPERIMENT = "publishing"

# Writing these is not reversible in any way that matters: every released HSSM
# downloads from the repo root at `main` with no revision pin, so a bad file is
# live for every user the moment it lands. Publishing here is deliberately not
# reachable from the CLI in this version.
PRODUCTION_REPOS = frozenset({"franklab/HSSM"})

# Parity is allowed to skip — it needs a *_train_state.jax sibling, which
# torch-trained networks legitimately do not have. The other three have no
# excuse: if one of them did not run, the network is unproven.
REQUIRED_GATES = ("structure", "hssm_load", "density")


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
    ours = {p.name for p in matches} | {"validation_report.json"}
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

    onnx = [p for p in destination.iterdir() if p.suffix == ".onnx"]
    if len(onnx) != 1:
        raise PublishError(
            f"Expected exactly one .onnx among the {len(matches)} staged "
            f"artifacts, found {len(onnx)}: {[p.name for p in onnx]}"
        )
    logger.info(f"Staged {len(matches)} artifacts to {destination}")
    return onnx[0]


def gate_verdict(report: dict) -> tuple[bool, str]:
    """Decide whether a validation report clears the network for publishing.

    Not the same as ``report["passed"]``: a skipped gate reports passed=True,
    so a report where everything skipped is "passed" and proves nothing.
    """
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
    unchecked = [n for n in REQUIRED_GATES if gates.get(n, absent).get("skipped")]
    if unchecked:
        return False, (
            f"not actually checked: {', '.join(unchecked)}. "
            "A skipped or missing gate is not a passed gate."
        )
    return True, "all required gates ran and passed"


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
) -> str:
    """Record the publish in MLflow and stamp the training run. Returns run id."""
    import mlflow

    published_at = datetime.now(timezone.utc).isoformat()
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
            }
        )
        mlflow.set_tags(
            {
                "schema_version": "1",
                "phase": "publish",
                "hf_commit_verified": str(hf_commit_verified).lower(),
                "published_at": published_at,
            }
        )
        if hf_url:
            mlflow.set_tag("hf_url", hf_url)

        if report:
            mlflow.log_dict(report, "validation_report.json")
            gates = {g["gate"]: g for g in report["gates"]}
            # .get throughout: a skipped or errored gate carries no scores.
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
    network_type: str = typer.Option(None, help="lan | cpn | opn | gonogo."),
    artifact_dir: Path = typer.Option(
        None,
        help="Folder holding the trained artifacts. Defaults to the training "
        "run's output_path, which only works if that path exists locally.",
    ),
    staging_dir: Path = typer.Option(
        None, help="Where to assemble this run's files [default: a temp dir]."
    ),
    skip_density: bool = typer.Option(
        False,
        # Density is a required gate, so skipping it makes gate_verdict refuse.
        # The flag cannot produce a publish at all — it only saves the cost of
        # the slowest gate while dry-running the resolve/stage/plan path.
        help="Skip G4. No publish is possible with this set; it only "
        "shortens --dry-run.",
    ),
    dry_run: bool = typer.Option(
        False, help="Validate and show the plan; touch neither HF nor MLflow."
    ),
    overwrite_root: bool = typer.Option(
        False,
        help="Replace an existing root {model}.onnx. Needed to republish a "
        "model that is already in the target repo.",
    ),
    log_level: str = typer.Option("INFO"),
):
    """Validate a trained network and publish it to HuggingFace."""
    level = getattr(logging, str(log_level).upper(), None)
    if not isinstance(level, int):
        raise typer.BadParameter(f"Unknown log level {log_level!r}.")
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")

    try:
        result = run_publish(
            hf_repo=hf_repo,
            run_id=run_id,
            model=model,
            network_type=network_type,
            artifact_dir=artifact_dir,
            staging_dir=staging_dir,
            skip_density=skip_density,
            dry_run=dry_run,
            overwrite_root=overwrite_root,
        )
    except PublishError as e:
        logger.error(str(e))
        print(json.dumps({"published": False, "error": str(e)}))
        raise typer.Exit(code=1) from e

    print(json.dumps(result))
    if not result["published"] and not dry_run:
        raise typer.Exit(code=1)


def run_publish(
    hf_repo: str,
    run_id: str | None = None,
    model: str | None = None,
    network_type: str | None = None,
    artifact_dir: Path | None = None,
    staging_dir: Path | None = None,
    skip_density: bool = False,
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
    hf_repo = hf_repo.strip().strip("/")
    production = {repo.casefold() for repo in PRODUCTION_REPOS}
    if hf_repo.casefold() in production and not allow_production:
        raise PublishError(
            f"{hf_repo} is the production repo every released HSSM downloads "
            "from. Publish to a staging repo and promote deliberately."
        )

    import tempfile

    from lanfactory.hf import VALID_NETWORK_TYPES
    from lanfactory.hf.upload import (
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
    if network_type not in VALID_NETWORK_TYPES:
        raise PublishError(
            f"network_type {network_type!r} is not one of {list(VALID_NETWORK_TYPES)}."
        )
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
        report = validate_network(
            onnx_path=onnx_path,
            model_name=model,
            network_type=network_type,
            skip_density=skip_density,
        )
        # Written into the staging dir so it is uploaded alongside the network.
        (staging / "validation_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )

        ok, reason = gate_verdict(report)
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
