"""Run an experiment's stages on this machine, no Slurm.

`lan-sbatch run --experiment DIR --local` drives the same console scripts the
Slurm jobs call — `generate`, `jaxtrain`/`torchtrain`, `lan-validate`,
`lan-recover` — as subprocesses, in order, passing the experiment's lineage id
and MLflow experiment names to each, and records what each stage produced in
the experiment's `state.json`. The final stage queries MLflow for every run
carrying the lineage id and checks the cross-stage links.

Subprocesses rather than imports: the upstream packages resolve their own
configuration and MLflow state per process, exactly as they do inside a job.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from sbatch_scripts import experiment as xp
except ImportError:  # pragma: no cover - path import in tests
    import experiment as xp

logger = logging.getLogger("run_local")

STAGES = ("generate", "train", "validate", "recover", "summary")

# Console scripts each stage needs and the uv group that provides them.
_NEEDS_VALIDATE_GROUP = {"lan-validate", "lan-recover"}


class StageFailed(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(f"{stage}: {message}")
        self.stage = stage


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _cli(name: str) -> str:
    """Path of a console script in this environment, or a clear error."""
    path = shutil.which(name)
    if path:
        return path
    hint = (
        " Install it with `uv sync --group validate`."
        if name in _NEEDS_VALIDATE_GROUP
        else " Is the environment synced (`uv sync`)?"
    )
    raise StageFailed("setup", f"console script {name!r} not found on PATH.{hint}")


# Flags each tool must accept for lineage tracking; the released packages in
# the lockfile predate them, so check before running anything.
_REQUIRED_FLAGS = {
    "generate": "--lineage-id",
    "jaxtrain": "--lineage-id",
    "torchtrain": "--lineage-id",
    "lan-recover": "--track",
}


def _preflight(tools: list[str], runner) -> None:
    """Fail early, in plain words, if an installed tool cannot track lineage."""
    for tool in tools:
        flag = _REQUIRED_FLAGS.get(tool)
        if flag is None:
            continue
        path = _cli(tool)
        result = runner(
            [path, "--help"], env=dict(os.environ), capture_output=True, text=True
        )
        if flag not in (result.stdout or "") + (result.stderr or ""):
            raise StageFailed(
                "setup",
                f"the installed {tool!r} does not accept {flag}, so it cannot record "
                "lineage. The pipeline's lockfile still pins releases that predate "
                "MLflow schema v2; install the tracking-capable checkouts into this "
                "environment (see the user guide, 'Setup') and run with "
                "`uv run --no-sync` so uv does not replace them.",
            )


def _env(exp: xp.Experiment) -> dict:
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)  # uv must resolve each tool's own environment
    env["MLFLOW_TRACKING_URI"] = xp.mlflow_tracking_uri(exp)
    env["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    # Local runs never own an artifact root; the tracking store's default (or
    # the server's --serve-artifacts) does.
    env.pop("MLFLOW_ARTIFACT_LOCATION", None)
    return env


def _run(stage: str, argv: list[str], env: dict, dry_run: bool, runner) -> str:
    logger.info("%s: %s", stage, " ".join(argv))
    if dry_run:
        return ""
    result = runner(argv, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        # The tools print one JSON line on stdout saying what failed; prefer
        # that over a stderr tail full of library warnings.
        record = _last_json(result.stdout or "")
        if record:
            raise StageFailed(stage, f"exit {result.returncode}: {json.dumps(record)}")
        tail = "\n".join((result.stderr or result.stdout or "").splitlines()[-25:])
        raise StageFailed(stage, f"exit {result.returncode}\n{tail}")
    return result.stdout or ""


def _tag_args(exp: xp.Experiment) -> list[str]:
    out: list[str] = []
    for key, value in (exp.mlflow.tags or {}).items():
        out += ["--mlflow-tag", f"{key}={value}"]
    return out


def _mlflow(exp: xp.Experiment):
    import mlflow

    mlflow.set_tracking_uri(xp.mlflow_tracking_uri(exp))
    return mlflow


def _find_training_data_folder(exp: xp.Experiment) -> Path | None:
    root = exp.stage_output("generate")
    hits = [
        Path(p)
        for p in glob.glob(str(root / "**" / exp.model), recursive=True)
        if Path(p).is_dir() and any(Path(p).glob("*.pickle"))
    ]
    return sorted(hits, key=lambda p: p.stat().st_mtime)[-1] if hits else None


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def stage_generate(exp, env, state, dry_run, runner) -> dict:
    lineage = exp.lineage_id
    names = exp.experiment_names()
    n_files = exp.stages.generate.n_files or 2
    argv = [
        _cli("generate"),
        "--config-path",
        str(exp.stage_config("generate")),
        "--output",
        str(exp.stage_output("generate")),
        "--n-files",
        str(n_files),
        "--n-cpus",
        str(max(1, min(4, os.cpu_count() or 1))),
        "--mlflow-experiment-name",
        names["data_generation"],
        "--mlflow-run-name",
        f"{exp.model}-local-worker-1",
        "--lineage-id",
        lineage,
        "--log-level",
        "WARNING",
        *_tag_args(exp),
    ]
    _run("generate", argv, env, dry_run, runner)
    if dry_run:
        return {}
    mlflow = _mlflow(exp)
    experiment = mlflow.get_experiment_by_name(names["data_generation"])
    folder = _find_training_data_folder(exp)
    if folder is None:
        raise StageFailed(
            "generate",
            f"no pickles for {exp.model!r} under {exp.stage_output('generate')}",
        )
    return {
        "data_generation_experiment_id": experiment.experiment_id
        if experiment
        else None,
        "training_data_folder": str(folder),
    }


def stage_train(exp, env, state, dry_run, runner) -> dict:
    names = exp.experiment_names()
    trainer = exp.network.trainer
    argv = [
        _cli(trainer),
        "--config-path",
        str(exp.stage_config("train")),
        "--networks-path-base",
        str(exp.stage_output("train")),
        "--network-id",
        str(exp.network.network_id),
        "--dl-workers",
        "0",
        "--mlflow-experiment-name",
        names["training"],
        "--mlflow-run-name",
        f"{trainer}_network_{exp.network.network_id}",
        "--lineage-id",
        exp.lineage_id,
        "--log-level",
        "WARNING",
        *_tag_args(exp),
    ]
    folder = state.get("training_data_folder") or (
        str(_find_training_data_folder(exp) or "")
    )
    if folder:
        argv += ["--training-data-folder", folder]
    if state.get("data_generation_experiment_id"):
        argv += [
            "--data-generation-experiment-id",
            state["data_generation_experiment_id"],
        ]
    if not folder and not state.get("data_generation_experiment_id") and not dry_run:
        raise StageFailed("train", "no training data: run the generate stage first")
    _run("train", argv, env, dry_run, runner)
    if dry_run:
        return {}
    onnx = xp.find_onnx(exp)
    if onnx is None:
        raise StageFailed("train", f"no ONNX written under {exp.stage_output('train')}")
    run_uuid = xp.run_uuid_from_filename(onnx.name)
    mlflow = _mlflow(exp)
    train_run_id = None
    experiment = mlflow.get_experiment_by_name(names["training"])
    if run_uuid and experiment:
        found = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.run_uuid = '{run_uuid}'",
            max_results=1,
        )
        if len(found):
            train_run_id = found.iloc[0]["run_id"]
    return {
        "onnx_path": str(onnx),
        "run_uuid": run_uuid,
        "mlflow_run_id_train": train_run_id,
        "training_experiment_id": experiment.experiment_id if experiment else None,
    }


def stage_validate(exp, env, state, dry_run, runner, continue_on_gate_failure) -> dict:
    onnx = state.get("onnx_path") or (str(xp.find_onnx(exp) or ""))
    if not onnx:
        raise StageFailed("validate", "no ONNX to validate: run the train stage first")
    argv = [
        _cli("lan-validate"),
        "--onnx-path",
        onnx,
        "--model-name",
        exp.model,
        "--network-type",
        exp.network.type,
        "--log-level",
        "WARNING",
    ]
    if exp.stages.validate.skip_density:
        argv.append("--skip-density")
    if exp.stages.validate.skip_hssm:
        argv.append("--skip-hssm")
    try:
        out = _run("validate", argv, env, dry_run, runner)
    except StageFailed as e:
        # lan-validate exits non-zero when a gate fails; its JSON line still
        # says which. Surface that and honour --continue-on-gate-failure.
        if not continue_on_gate_failure:
            raise
        logger.warning("validation failed; continuing as requested: %s", e)
        return {"validation_passed": False}
    if dry_run:
        return {}
    record = _last_json(out)
    return {
        "validation_passed": bool(record.get("passed")),
        "validation_report": record.get("report"),
    }


def stage_recover(exp, env, state, dry_run, runner) -> dict:
    rec = exp.stages.recover
    onnx = state.get("onnx_path") or (str(xp.find_onnx(exp) or ""))
    likelihoods = list(rec.likelihoods)
    if (
        rec.add_reference_arm
        and "analytical" not in likelihoods
        and _has_analytical(exp.model)
    ):
        likelihoods.append("analytical")
    if any(lk != "analytical" for lk in likelihoods) and not onnx:
        raise StageFailed(
            "recover", "no ONNX for the network arm: run the train stage first"
        )
    names = exp.experiment_names()
    out_dir = exp.recovery_dir()
    shards = []
    for design in rec.designs:
        for lk in likelihoods:
            for i in range(rec.n_datasets):
                argv = [
                    _cli("lan-recover"),
                    "--model",
                    exp.model,
                    "--design",
                    design,
                    "--dataset-index",
                    str(i),
                    "--likelihood",
                    lk,
                    "--out-dir",
                    str(out_dir),
                    "--draws",
                    str(rec.draws),
                    "--tune",
                    str(rec.tune),
                    "--chains",
                    str(rec.chains),
                    "--target-accept",
                    str(rec.target_accept),
                    "--track",
                    "--mlflow-experiment-name",
                    names["inference"],
                    "--mlflow-run-name",
                    f"{exp.model}-recovery-{design}-{lk}-{i}",
                    "--lineage-id",
                    exp.lineage_id,
                    "--log-level",
                    "WARNING",
                    *_tag_args(exp),
                ]
                if lk != "analytical":
                    argv += ["--onnx-path", onnx]
                if rec.p_outlier is not None:
                    argv += ["--p-outlier", str(rec.p_outlier)]
                if rec.condition_param:
                    argv += ["--condition-param", rec.condition_param]
                if state.get("mlflow_run_id_train"):
                    argv += ["--mlflow-run-id-train", state["mlflow_run_id_train"]]
                out = _run("recover", argv, env, dry_run, runner)
                if not dry_run:
                    shards.append(_last_json(out).get("shard"))
    if dry_run:
        return {}
    summary = _aggregate(out_dir)
    return {
        "recovery_shards": shards,
        "recovery_summary": str(out_dir / "summary.json"),
        **summary,
    }


def stage_summary(exp, env, state, dry_run, runner, ran: set[str]) -> dict:
    if dry_run:
        return {}
    mlflow = _mlflow(exp)
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.lineage_id = '{exp.lineage_id}'",
        order_by=["start_time ASC"],
    )
    exps = {e.experiment_id: e.name for e in mlflow.search_experiments()}
    print(
        f"\n  {'experiment':<26}{'phase':<9}{'user':<12}{'status':<10}run",
        file=sys.stderr,
    )
    for _, r in runs.iterrows():
        print(
            f"  {exps.get(r['experiment_id'], '?'):<26}{r.get('tags.phase', '?'):<9}"
            f"{str(r.get('tags.user', '?')):<12}{r['status']:<10}{r['run_id']}",
            file=sys.stderr,
        )
    phases = set(runs["tags.phase"]) if len(runs) else set()
    expected = {"generate": "datagen", "train": "train", "recover": "infer"}
    missing = [expected[s] for s in ran if s in expected and expected[s] not in phases]
    checks = {}
    client = mlflow.MlflowClient()
    if "train" in phases and "datagen" in phases:
        tr = client.get_run(runs[runs["tags.phase"] == "train"]["run_id"].iloc[-1])
        dg_ids = set(runs[runs["tags.phase"] == "datagen"]["run_id"])
        linked = set(tr.data.tags.get("data_generation_run_ids", "").split(","))
        checks["train links the datagen runs"] = bool(linked & dg_ids)
    if "infer" in phases and "train" in phases:
        inf = client.get_run(runs[runs["tags.phase"] == "infer"]["run_id"].iloc[-1])
        tr_id = runs[runs["tags.phase"] == "train"]["run_id"].iloc[-1]
        checks["fits link the training run"] = (
            inf.data.tags.get("mlflow_run_id_train") == tr_id
        )
    for name, ok in checks.items():
        print(f"  {'✓' if ok else '✗'} {name}", file=sys.stderr)
    if missing:
        raise StageFailed(
            "summary",
            f"no MLflow runs with phase {missing} for lineage {exp.lineage_id}",
        )
    if not all(checks.values()):
        raise StageFailed("summary", "cross-stage links are missing (see above)")
    return {"phases": sorted(phases), "n_runs": int(len(runs)), "checks": checks}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _has_analytical(model: str) -> bool:
    try:
        sys.path.insert(0, str(xp.REPO_ROOT / "validation"))
        import recovery_designs as rd  # type: ignore

        return bool(rd.load_model(model).has_analytical)
    except Exception:  # noqa: BLE001 - hssm missing or model unknown to it
        return False


def _aggregate(out_dir: Path) -> dict:
    try:
        sys.path.insert(0, str(xp.REPO_ROOT / "validation"))
        import aggregate_recovery as agg  # type: ignore

        shards = agg.load_shards(out_dir)
        summary = agg.summarise(shards)
        passed, reasons = agg.verdict(summary)
        (out_dir / "summary.json").write_text(
            json.dumps(
                {"passed": passed, "reasons": reasons, "summary": summary},
                indent=2,
                default=str,
            )
        )
        return {"recovery_verdict": passed, "recovery_reasons": reasons}
    except Exception as e:  # noqa: BLE001
        logger.warning("could not aggregate recovery shards: %s", e)
        return {}


def _last_json(text: str) -> dict:
    for line in reversed(text.strip().splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


def run_local(
    exp: xp.Experiment,
    stages: list[str],
    *,
    dry_run: bool = False,
    continue_on_gate_failure: bool = False,
    runner=subprocess.run,
) -> dict:
    """Run the requested stages in order; return the final state."""
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown stages {unknown}; choose from {list(STAGES)}")
    ordered = [s for s in STAGES if s in stages]
    if not dry_run:
        xp.ensure_lineage_id(exp)
        tools = []
        if "generate" in ordered:
            tools.append("generate")
        if "train" in ordered:
            tools.append(exp.network.trainer)
        if "recover" in ordered:
            tools.append("lan-recover")
        _preflight(tools, runner)
    env = _env(exp)
    state = xp.read_state(exp)
    ran: set[str] = set()
    for stage in ordered:
        logger.info("==> %s", stage)
        if stage == "generate":
            update = stage_generate(exp, env, state, dry_run, runner)
        elif stage == "train":
            update = stage_train(exp, env, state, dry_run, runner)
        elif stage == "validate":
            update = stage_validate(
                exp, env, state, dry_run, runner, continue_on_gate_failure
            )
        elif stage == "recover":
            update = stage_recover(exp, env, state, dry_run, runner)
        else:
            update = stage_summary(exp, env, state, dry_run, runner, ran)
        ran.add(stage)
        if not dry_run and update:
            state = xp.write_state(exp, **update)
    return state
