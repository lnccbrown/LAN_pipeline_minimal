#!/usr/bin/env python3
"""
Generates and submits individual sbatch job for generating simulated data for a given model,
and/or for training neural network on simulated data.

MLflow Integration:
- For data generation: Each SBATCH array task creates its own MLflow run using --mlflow-run-name
- For training: The orchestrator creates a parent run and passes --mlflow-run-id to continue logging
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import logging
import re
import shlex
import subprocess
import typer
import yaml
import os

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

app = typer.Typer(add_completion=False)

# SBATCH template
SBATCH_TEMPLATE = """#!/bin/bash

#SBATCH --account={account}
#SBATCH -p {partition} --gres=gpu:{num_gpus}
#SBATCH -c {cores}
#SBATCH --mem={mem}
#SBATCH -J {job_name}
#SBATCH --time={time}
#SBATCH --output={output}
#SBATCH --error={error}
#SBATCH --array=1-{n_jobs_in_array}

{module_loads}

# The uv project is resolved from the working directory, and SLURM starts the
# job in whatever directory sbatch was invoked from — $HOME for a driver that
# submits over ssh. Pin it to the checkout that generated this script.
cd {project_root} || exit 1

# MLflow environment variables
{env_vars}

if ! python -m uv --version >/dev/null 2>&1; then
  if [ -n "${{VIRTUAL_ENV:-}}" ]; then
    python -m pip install uv
  else
    python -m pip install --user uv
  fi
fi
python -m uv run {command}
"""
# Notes on the install block above:
# - `python -m pip`, not bare `pip`: after `module load python` the two can
#   resolve to different interpreters, and uv must land in the one that runs
#   the job on the next line.
# - `--user` only outside a virtualenv. pip *hard-errors* on `--user` inside
#   one ("User site-packages are not visible in this virtualenv"), and sbatch
#   propagates the submitting environment by default — so an unconditional
#   `--user` fails for anyone submitting from their project venv.
# - Skipped entirely when uv is already importable, which is the common case
#   on a node that has it.
# Braces are doubled because this string is consumed by str.format().


# Values that must stay shell-expandable inside the job, because SLURM
# substitutes per-task. Only the variables in ALLOWED_SHELL_VARS expand; every
# other character is quoted literally (see quote_param_value).
SHELL_EXPANDED_PARAMS = frozenset({"mlflow-run-name"})

# The only shell expansions allowed inside SHELL_EXPANDED_PARAMS values.
ALLOWED_SHELL_VARS = ("SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_JOB_ID", "SLURM_JOB_ID")
_ALLOWED_VAR_RE = re.compile(r"\$(?:" + "|".join(ALLOWED_SHELL_VARS) + r")\b")

# Historical CLI defaults per job kind, used when no --cluster-config is given.
# A cluster config's job_defaults section overrides these; explicit CLI flags
# override both.
JOB_KIND_FALLBACKS = {
    "generate": {
        "account": "default",
        "partition": "batch",
        "num_gpus": 0,
        "cores": 1,
        "mem": "16G",
        "time": "00:30:00",
    },
    "jaxtrain": {
        "account": "default",
        "partition": "batch",
        "num_gpus": 0,
        "cores": 1,
        "mem": "16G",
        "time": "00:30:00",
    },
    "torchtrain": {
        "account": "default",
        "partition": "batch",
        "num_gpus": 0,
        "cores": 1,
        "mem": "16G",
        "time": "00:30:00",
    },
}

DEFAULT_MODULES = ["python", "gcc"]

# The checkout this script lives in — baked into generated scripts so the job
# runs against the right uv project regardless of the submission directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# SLURM wall-time syntax accepted for the `time` resource.
_SLURM_TIME_RE = re.compile(r"^(\d+|\d+:\d{2}|\d+:\d{2}:\d{2}|\d+-\d+(:\d{2}){0,2})$")


def absolutize_tracking_uri(uri: str) -> str:
    """Make a relative sqlite MLflow URI absolute.

    ``sqlite:///mlflow.db`` resolves against the *current* working directory,
    which differs between the submitting process and each compute node's job
    (SLURM starts jobs in the submission directory). Left relative, workers
    silently create their own database and the experiment id reported in the
    JSON contract points at a file nothing ever wrote to.
    """
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return uri  # server URIs and already-absolute sqlite:////... are fine
    path_part = uri[len(prefix) :]
    if path_part.startswith("/"):
        return uri
    return f"{prefix}{Path(path_part).resolve()}"


def quote_param_value(key: str, value) -> str:
    """Quote a CLI parameter value for embedding in the sbatch script.

    Default: shlex.quote, so the value is one inert literal argument.

    For SHELL_EXPANDED_PARAMS the value must keep expanding SLURM's per-task
    variables, so it is split into segments: occurrences of ALLOWED_SHELL_VARS
    are emitted bare (inside double quotes) and *everything else* is
    shlex.quoted. Adjacent quoted segments concatenate into one shell word, so
    e.g. `ddm-worker-$SLURM_ARRAY_TASK_ID` becomes

        'ddm-worker-'"$SLURM_ARRAY_TASK_ID"

    Blanket double-quoting would be wrong here: the model name comes from a
    user-supplied pipeline YAML, and inside double quotes bash still executes
    backticks and `$(...)` and expands every other `$VAR` — so a model named
    ``ddm`hostname` `` would run a command on the compute node.
    """
    text = str(value)
    if key not in SHELL_EXPANDED_PARAMS:
        return shlex.quote(text)

    segments = []
    position = 0
    for match in _ALLOWED_VAR_RE.finditer(text):
        if match.start() > position:
            segments.append(shlex.quote(text[position : match.start()]))
        segments.append(f'"{match.group(0)}"')
        position = match.end()
    if position < len(text):
        segments.append(shlex.quote(text[position:]))
    return "".join(segments) if segments else "''"


def create_command(command_name: str, **params) -> str:
    """Create CLI command string from parameters, shell-quoted."""
    parts = [command_name]
    parts += [
        f"--{key} {quote_param_value(key, value)}" for key, value in params.items()
    ]
    return " ".join(parts)


def resolve_resources(
    command_name: str,
    cluster_config_path: Path | None,
    cli_values: dict,
    logger=None,
) -> dict:
    """Resolve job resources: builtin fallbacks < cluster config < CLI flags.

    Within the cluster config, a per-job-kind ``modules`` list beats the
    top-level one (specific beats generic), and an explicitly empty list means
    "load no modules" — distinct from the key being absent, which means "use
    the defaults".

    cli_values holds only flags the user actually passed (None = not passed).
    """
    resolved = dict(JOB_KIND_FALLBACKS[command_name])
    modules = list(DEFAULT_MODULES)

    if cluster_config_path is not None:
        with open(cluster_config_path, "rb") as f:
            cluster_config = yaml.safe_load(f) or {}
        job_defaults = (cluster_config.get("job_defaults") or {}).get(command_name, {})

        known = set(resolved) | {"modules"}
        unknown = set(job_defaults) - known
        if unknown and logger is not None:
            logger.warning(
                f"Ignoring unknown job_defaults keys for {command_name!r} in "
                f"{cluster_config_path}: {sorted(unknown)}"
            )
        resolved.update({k: v for k, v in job_defaults.items() if k in resolved})

        # Presence, not truthiness: `modules: []` disables module loading.
        if "modules" in cluster_config:
            modules = list(cluster_config["modules"] or [])
        if "modules" in job_defaults:
            modules = list(job_defaults["modules"] or [])

    resolved.update({k: v for k, v in cli_values.items() if v is not None})
    resolved["modules"] = modules
    validate_resources(resolved, cluster_config_path)
    return resolved


def validate_resources(resolved: dict, source) -> None:
    """Reject resource values SLURM would silently misread.

    Chiefly YAML 1.1 sexagesimal: an unquoted ``time: 12:00:00`` loads as the
    integer 43200, which SLURM reads as 43200 *minutes* — a 60x wall-time
    request. Fail loudly at generation time instead.
    """
    time_value = resolved.get("time")
    if isinstance(time_value, int) or not _SLURM_TIME_RE.match(str(time_value)):
        hint = ""
        if isinstance(time_value, int):
            hours, remainder = divmod(time_value, 3600)
            minutes, seconds = divmod(remainder, 60)
            hint = (
                f" YAML parsed it as the integer {time_value}; quote it "
                f'("{hours:02d}:{minutes:02d}:{seconds:02d}") in {source}.'
            )
        raise typer.BadParameter(
            f"Invalid SLURM wall time {time_value!r}. Expected forms: "
            f"MM, HH:MM, HH:MM:SS, D-HH, D-HH:MM, D-HH:MM:SS.{hint}"
        )


def create_sbatch_script(
    account="default",
    partition="batch",
    num_gpus=0,
    cores=1,
    mem="4G",
    job_name="job",
    output="output.txt",
    error="error.txt",
    time="01:00:00",
    command="",
    n_jobs_in_array=1,
    env_vars="",
    modules=None,
    project_root=None,
):
    # `is None` (absent), not falsy: an empty list means "load no modules".
    module_list = DEFAULT_MODULES if modules is None else modules
    module_loads = "\n".join(f"module load {module}" for module in module_list)
    sbatch_script = SBATCH_TEMPLATE.format(
        account=account,
        partition=partition,
        num_gpus=num_gpus,
        cores=cores,
        mem=mem,
        job_name=job_name,
        time=time,
        # SLURM splits #SBATCH directive lines on whitespace; quoting keeps a
        # path with spaces in one piece (a no-op for ordinary paths).
        output=shlex.quote(str(output)),
        error=shlex.quote(str(error)),
        command=command,
        n_jobs_in_array=n_jobs_in_array,
        env_vars=env_vars,
        module_loads=module_loads,
        project_root=shlex.quote(str(project_root or PROJECT_ROOT)),
    )
    return sbatch_script


def write_sbatch(script, sbatch_script):
    with open(script, "w") as f:
        f.write(sbatch_script)


def submit_sbatch(script, logger) -> int | None:
    """Submit the script via sbatch; return the SLURM job id, or None on failure.

    The job id is parsed from sbatch's "Submitted batch job <id>" stdout line.
    Failures (nonzero exit, missing sbatch binary, unparseable output) are
    logged and yield None — the caller decides the process exit code.
    """
    try:
        result = subprocess.run(
            ["sbatch", str(script)], capture_output=True, text=True, check=False
        )
    except OSError as e:  # sbatch binary missing / not executable
        logger.error(f"Failed to submit job: {e}")
        return None
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.stderr:
        logger.error(result.stderr.strip())
    if result.returncode != 0:
        logger.error(f"sbatch exited with code {result.returncode}")
        return None
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match is None:
        logger.error(
            "sbatch succeeded but its output did not contain "
            f"'Submitted batch job <id>': {result.stdout!r}"
        )
        return None
    return int(match.group(1))


def get_basic_config_from_yaml(yaml_config_path: str | Path):
    with open(yaml_config_path, "rb") as f:
        basic_config = yaml.safe_load(f)
    return basic_config


def get_parameters_setup(
    command: str,
    config_path: Path,
    output_path: Path,
    log_level: str,
    training_data_folder: Path = None,
    network_id: int = 0,
    dl_workers: int = 1,
    # MLflow parameters - different for generate vs training
    mlflow_run_name: str = None,  # For data generation (ssm-simulators)
    mlflow_experiment_name: str = None,  # For data generation (ssm-simulators)
    mlflow_run_id: str = None,  # For training (LANfactory) - to resume a run
    data_generation_experiment_id: str = None,  # For training lineage
):
    """
    Prepare CLI arguments for the command based on the command type.

    For data generation (ssm-simulators):
        - Uses --mlflow-run-name and --mlflow-experiment-name
        - Each worker creates its own run

    For training (LANfactory):
        - Uses --mlflow-run-id to continue logging to parent run
        - Uses --data-generation-experiment-id for lineage tracking
    """
    params = {"config-path": config_path.resolve(), "log-level": log_level}

    if command == "generate":
        params["output"] = output_path.resolve()
        # ssm-simulators uses --mlflow-run-name and --mlflow-experiment-name
        if mlflow_run_name:
            params["mlflow-run-name"] = mlflow_run_name
        if mlflow_experiment_name:
            params["mlflow-experiment-name"] = mlflow_experiment_name

    elif command in ["jaxtrain", "torchtrain"]:
        params.update(
            {
                "networks-path-base": output_path.resolve(),
                "training-data-folder": training_data_folder.resolve(),
                "network-id": network_id,
                "dl-workers": dl_workers,
            }
        )
        # LANfactory uses --mlflow-run-id to resume the parent run
        if mlflow_run_id:
            params["mlflow-run-id"] = mlflow_run_id
        # Add data generation experiment ID for training lineage
        if data_generation_experiment_id:
            params["data-generation-experiment-id"] = data_generation_experiment_id

    return params


def handle_job(
    command_name: str,
    config_path: Path,
    output_path: Path,
    log_level: str,
    script_only: bool,
    account: str | None = None,
    partition: str | None = None,
    num_gpus: int | None = None,
    cores: int | None = None,
    mem: str | None = None,
    time: str | None = None,
    cluster_config: Path = None,
    n_jobs_in_array: int = 1,
    n_files: int = None,
    training_data_folder: Path = None,
    network_id: int = 0,
    dl_workers: int = 1,
    data_generation_experiment_id: str = None,
):
    # Case-insensitive, and validated: getattr(logging, "info") returns the
    # *function*, which basicConfig rejects with a TypeError traceback — and
    # the JSON contract line would never be printed.
    level = getattr(logging, str(log_level).upper(), None)
    if not isinstance(level, int):
        raise typer.BadParameter(
            f"Unknown log level {log_level!r}. Expected one of: "
            "DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )
    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    logger = logging.getLogger("gen_sbatch")
    target = output_path.resolve()

    resources = resolve_resources(
        command_name,
        cluster_config,
        {
            "account": account,
            "partition": partition,
            "num_gpus": num_gpus,
            "cores": cores,
            "mem": mem,
            "time": time,
        },
        logger=logger,
    )

    # Initialize MLflow parameters
    mlflow_run_id = None  # For training: parent run ID
    mlflow_run_name = None  # For data generation: run name template
    mlflow_experiment_name = None  # For data generation: experiment name
    mlflow_experiment_id = None  # For data generation: to report to user
    env_vars = ""

    # --script-only is side-effect-free: no experiment is created, no run is
    # started, and no MLflow wiring is embedded in the generated script. A
    # previous version created an empty orphan run per --script-only call.
    if MLFLOW_AVAILABLE and not script_only:
        try:
            # Set tracking URI from environment or use default (SQLite)
            # Absolute, so the submitting process and every compute-node
            # worker agree on which database the experiment lives in.
            tracking_uri = absolutize_tracking_uri(
                os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
            )
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI: {tracking_uri}")

            # Get artifact location from environment (optional)
            artifact_location = os.getenv("MLFLOW_ARTIFACT_LOCATION", None)

            # Get model name for experiment naming
            basic_config = get_basic_config_from_yaml(config_path)
            model_name = basic_config.get("MODEL", "unknown")

            # Create/get experiment based on command type
            if command_name == "generate":
                # Data generation experiment
                experiment_name = f"{model_name}-data-generation"

                # Check if experiment exists, create with artifact location if needed
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    if artifact_location:
                        artifact_location_abs = str(Path(artifact_location).absolute())
                        mlflow.create_experiment(
                            experiment_name, artifact_location=artifact_location_abs
                        )
                        logger.info(
                            f"Created experiment: {experiment_name} "
                            f"(artifacts: {artifact_location_abs})"
                        )
                    else:
                        mlflow.create_experiment(experiment_name)
                        logger.info(
                            f"Created experiment: {experiment_name} "
                            "(default artifact location)"
                        )
                    experiment = mlflow.get_experiment_by_name(experiment_name)

                mlflow.set_experiment(experiment_name)
                mlflow_experiment_id = experiment.experiment_id
                logger.info(
                    f"Data generation experiment: {experiment_name} "
                    f"(ID: {mlflow_experiment_id})"
                )

                # For data generation, we DON'T create a parent run here.
                # Each SBATCH array worker will create its own run.
                # We pass --mlflow-run-name with $SLURM_ARRAY_TASK_ID for unique names.
                mlflow_run_name = f"{model_name}-worker-$SLURM_ARRAY_TASK_ID"
                mlflow_experiment_name = experiment_name

                # Environment variables for the SBATCH script
                env_vars = (
                    f"export MLFLOW_EXPERIMENT_NAME={shlex.quote(experiment_name)}\n"
                    f"export MLFLOW_TRACKING_URI={shlex.quote(tracking_uri)}"
                )
                if artifact_location:
                    env_vars += (
                        "\nexport MLFLOW_ARTIFACT_LOCATION="
                        f"{shlex.quote(artifact_location)}"
                    )

            elif command_name in ["jaxtrain", "torchtrain"]:
                # Training experiment
                experiment_name = f"{model_name}-training"

                # Check if experiment exists, create with artifact location if needed
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    if artifact_location:
                        artifact_location_abs = str(Path(artifact_location).absolute())
                        mlflow.create_experiment(
                            experiment_name, artifact_location=artifact_location_abs
                        )
                        logger.info(
                            f"Created experiment: {experiment_name} "
                            f"(artifacts: {artifact_location_abs})"
                        )
                    else:
                        mlflow.create_experiment(experiment_name)
                        logger.info(
                            f"Created experiment: {experiment_name} "
                            "(default artifact location)"
                        )
                    experiment = mlflow.get_experiment_by_name(experiment_name)

                mlflow.set_experiment(experiment_name)
                mlflow_experiment_id = experiment.experiment_id
                logger.info(
                    f"Training experiment: {experiment_name} "
                    f"(ID: {mlflow_experiment_id})"
                )

                # Create a parent run for the training job
                # The SBATCH worker will continue logging to this run via --mlflow-run-id
                run = mlflow.start_run(run_name=f"{command_name}_network_{network_id}")
                mlflow_run_id = run.info.run_id

                # Log configuration
                mlflow.log_param("command", command_name)
                mlflow.log_param("config_path", str(config_path))
                mlflow.log_param("output_path", str(output_path))
                mlflow.log_param("network_id", network_id)

                # If data_generation_experiment_id is provided, log it
                if data_generation_experiment_id:
                    mlflow.set_tag(
                        "data_generation_experiment_id", data_generation_experiment_id
                    )
                    logger.info(
                        f"Linked to data generation experiment: "
                        f"{data_generation_experiment_id}"
                    )

                logger.info(f"MLflow training run initialized: {mlflow_run_id}")

                # Prepare env vars for sbatch script
                env_vars = (
                    f"export MLFLOW_EXPERIMENT_NAME={shlex.quote(experiment_name)}\n"
                    f"export MLFLOW_TRACKING_URI={shlex.quote(tracking_uri)}"
                )
                if artifact_location:
                    env_vars += (
                        "\nexport MLFLOW_ARTIFACT_LOCATION="
                        f"{shlex.quote(artifact_location)}"
                    )

        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")

    params = get_parameters_setup(
        command=command_name,
        config_path=config_path,
        output_path=output_path,
        log_level=log_level,
        training_data_folder=training_data_folder,
        network_id=network_id,
        dl_workers=dl_workers,
        mlflow_run_name=mlflow_run_name,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_run_id=mlflow_run_id,
        data_generation_experiment_id=data_generation_experiment_id,
    )
    if command_name == "generate" and n_files is not None:
        params["n-files"] = n_files
    command = create_command(command_name, **params)
    logger.info(f"Generated command: {command}")
    basic_config = get_basic_config_from_yaml(params["config-path"])
    job_name = f"{basic_config['MODEL']}_{command_name}_sbatch"

    # Scripts and SLURM logs live under <output_path>/runs/, timestamped —
    # repeated invocations never overwrite each other (previously the script
    # landed in CWD under a fixed name).
    runs_dir = target / "runs"
    runs_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    script = runs_dir / f"{timestamp}_{job_name}.sh"

    sbatch_kwargs = dict(
        account=resources["account"],
        partition=resources["partition"],
        num_gpus=resources["num_gpus"],
        cores=resources["cores"],
        mem=resources["mem"],
        job_name=job_name,
        # %A_%a = array job id + task index. The template always emits
        # --array, so every task would otherwise append to one shared pair of
        # files and interleave its output with the others'.
        output=str(runs_dir / f"{job_name}_%A_%a.out"),
        error=str(runs_dir / f"{job_name}_%A_%a.err"),
        time=resources["time"],
        command=command,
        n_jobs_in_array=n_jobs_in_array,
        env_vars=env_vars,
        modules=resources["modules"],
    )
    sbatch_script = create_sbatch_script(**sbatch_kwargs)
    write_sbatch(script, sbatch_script)

    # The one machine-readable line on stdout, regardless of log level: the
    # laptop driver's API. Logging goes to stderr, so stdout carries only this.
    result_record = {
        "command": command,
        "job_id": None,
        "mlflow_experiment_id": mlflow_experiment_id,
        "mlflow_run_id": mlflow_run_id,
        "sbatch_script": str(script),
        "output_path": str(target),
        "account": resources["account"],
        "partition": resources["partition"],
    }

    if script_only:
        logger.info(f"Generated sbatch script: {script}")
        print(json.dumps(result_record))
        return

    if command_name == "generate":
        logger.info(f"Simulated data output folder: {target}")
    else:
        logger.info(f"Trained networks output folder: {target}")
    job_id = submit_sbatch(script, logger)
    result_record["job_id"] = job_id

    if MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.end_run()

    print(json.dumps(result_record))
    if job_id is None:
        logger.error("Job submission failed")
        raise typer.Exit(code=1)
    logger.info("Job submitted successfully")


@app.command()
def generate(
    config_path: Path = typer.Option(
        ..., help="Path to configuration .yaml file for running commands"
    ),
    output_path: Path = typer.Option(
        ..., help="Path to output folder for simulated data"
    ),
    n_jobs_in_array: int = typer.Option(1, help="Size of the job array"),
    n_files: int = typer.Option(
        None, help="Number of files each worker generates (passed to `generate`)"
    ),
    cluster_config: Path = typer.Option(
        None,
        help="Cluster inventory YAML (e.g. configs/cluster/oscar.yaml); its "
        "job_defaults section supplies account/partition/resources for this "
        "job kind. Explicit flags below override it.",
    ),
    account: str = typer.Option(
        None, help="Condo to run the SBATCH job on [default: from cluster config]"
    ),
    partition: str = typer.Option(
        None,
        help="Partition to run the SBATCH script on [default: from cluster config]",
    ),
    num_gpus: int = typer.Option(
        None, help="Number of GPUs requested [default: from cluster config]"
    ),
    mem: str = typer.Option(
        None, help="Memory limit for each job [default: from cluster config]"
    ),
    time: str = typer.Option(
        None, help="Wall time limit for each job [default: from cluster config]"
    ),
    cores: int = typer.Option(
        None, help="Number of cores per job [default: from cluster config]"
    ),
    script_only: bool = typer.Option(
        False,
        help="Generate the sbatch script without submitting the job. "
        "Side-effect-free: no MLflow experiment/run is created and no MLflow "
        "wiring is embedded in the script.",
    ),
    log_level: str = typer.Option(
        "WARNING", help="Set the log level", show_default=True
    ),
):
    """Generate SBATCH script for data generation using ssm-simulators."""
    handle_job(
        command_name="generate",
        config_path=config_path,
        output_path=output_path,
        log_level=log_level,
        time=time,
        script_only=script_only,
        cluster_config=cluster_config,
        account=account,
        partition=partition,
        num_gpus=num_gpus,
        cores=cores,
        mem=mem,
        n_jobs_in_array=n_jobs_in_array,
        n_files=n_files,
    )


def train_command(command_name: str):
    def train(
        config_path: Path = typer.Option(
            ..., help="Path to configuration .yaml file for running commands"
        ),
        output_path: Path = typer.Option(
            ..., help="Path to output folder for trained neural network"
        ),
        training_data_folder: Path = typer.Option(
            ..., help="Path to folder with data to train the neural network on"
        ),
        network_id: int = typer.Option(0, help="Id for the neural network to train"),
        data_generation_experiment_id: str = typer.Option(
            None,
            "--data-generation-experiment-id",
            help="MLflow Experiment ID of the data generation experiment. "
            "If provided, training data lineage will be logged.",
        ),
        cluster_config: Path = typer.Option(
            None,
            help="Cluster inventory YAML (e.g. configs/cluster/oscar.yaml); its "
            "job_defaults section supplies account/partition/resources for "
            "this job kind. Explicit flags below override it.",
        ),
        account: str = typer.Option(
            None, help="Condo to run the SBATCH job on [default: from cluster config]"
        ),
        partition: str = typer.Option(
            None,
            help="Partition to run the SBATCH script on [default: from cluster config]",
        ),
        num_gpus: int = typer.Option(
            None, help="Number of GPUs requested [default: from cluster config]"
        ),
        cores: int = typer.Option(
            None, help="Number of cores per job [default: from cluster config]"
        ),
        dl_workers: int = typer.Option(
            1, help="Number of cores to use with the dataloader class"
        ),
        time: str = typer.Option(
            None, help="Wall time limit for each job [default: from cluster config]"
        ),
        mem: str = typer.Option(
            None, help="Memory limit for each job [default: from cluster config]"
        ),
        script_only: bool = typer.Option(
            False,
            help="Generate the sbatch script without submitting the job. "
            "Side-effect-free: no MLflow experiment/run is created and no "
            "MLflow wiring is embedded in the script.",
        ),
        log_level: str = typer.Option(
            "WARNING", help="Set the log level", show_default=True
        ),
    ):
        """Generate SBATCH script for neural network training using LANfactory."""
        handle_job(
            command_name=command_name,
            config_path=config_path,
            output_path=output_path,
            log_level=log_level,
            time=time,
            script_only=script_only,
            cluster_config=cluster_config,
            account=account,
            partition=partition,
            num_gpus=num_gpus,
            cores=cores,
            mem=mem,
            training_data_folder=training_data_folder,
            network_id=network_id,
            dl_workers=dl_workers,
            data_generation_experiment_id=data_generation_experiment_id,
        )

    return train


app.command("jaxtrain")(train_command("jaxtrain"))
app.command("torchtrain")(train_command("torchtrain"))

if __name__ == "__main__":
    app()
