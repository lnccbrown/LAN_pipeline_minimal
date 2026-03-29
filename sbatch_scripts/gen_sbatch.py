#!/usr/bin/env python3
"""
Generates and submits individual sbatch job for generating simulated data for a given model,
and/or for training neural network on simulated data.

MLflow Integration:
- For data generation: Each SBATCH array task creates its own MLflow run using --mlflow-run-name
- For training: The orchestrator creates a parent run and passes --mlflow-run-id to continue logging
"""

from pathlib import Path
import logging
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

module load python
module load gcc

# MLflow environment variables
{env_vars}

pip install uv
python -m uv run {command}
"""


def create_command(command_name: str, **params: dict):
    """Create CLI command string from parameters."""
    command = f"{command_name} "
    command += " ".join([f"--{key} {value}" for key, value in params.items()])
    return command


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
):
    sbatch_script = SBATCH_TEMPLATE.format(
        account=account,
        partition=partition,
        num_gpus=num_gpus,
        cores=cores,
        mem=mem,
        job_name=job_name,
        time=time,
        output=output,
        error=error,
        command=command,
        n_jobs_in_array=n_jobs_in_array,
        env_vars=env_vars,
    )
    return sbatch_script


def write_sbatch(script, sbatch_script):
    with open(script, "w") as f:
        f.write(sbatch_script)


def submit_sbatch(script, logger):
    try:
        result = subprocess.run(
            ["sbatch", script], capture_output=True, text=True, check=False
        )
        logger.info(result.stdout)
        if result.stderr:
            logger.error(result.stderr)
    except Exception as e:
        logger.error(f"Failed to submit job: {e}")


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
    time: str,
    script_only: bool,
    account: str = "default",
    partition: str = "batch",
    num_gpus: int = 0,
    cores: int = 1,
    mem: str = "16G",
    n_jobs_in_array: int = 1,
    training_data_folder: Path = None,
    network_id: int = 0,
    dl_workers: int = 1,
    data_generation_experiment_id: str = None,
):
    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, log_level),
    )
    logger = logging.getLogger("gen_sbatch")
    target = output_path.resolve()

    # Initialize MLflow parameters
    mlflow_run_id = None  # For training: parent run ID
    mlflow_run_name = None  # For data generation: run name template
    mlflow_experiment_name = None  # For data generation: experiment name
    mlflow_experiment_id = None  # For data generation: to report to user
    env_vars = ""

    if MLFLOW_AVAILABLE:
        try:
            # Set tracking URI from environment or use default (SQLite)
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
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
                env_vars = f"export MLFLOW_EXPERIMENT_NAME={experiment_name}\n"
                env_vars += f"export MLFLOW_TRACKING_URI={tracking_uri}"
                if artifact_location:
                    env_vars += f"\nexport MLFLOW_ARTIFACT_LOCATION={artifact_location}"

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
                env_vars = f"export MLFLOW_EXPERIMENT_NAME={experiment_name}\n"
                env_vars += f"export MLFLOW_TRACKING_URI={tracking_uri}"
                if artifact_location:
                    env_vars += f"\nexport MLFLOW_ARTIFACT_LOCATION={artifact_location}"

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
    command = create_command(command_name, **params)
    logger.info(f"Generated command: {command}")
    basic_config = get_basic_config_from_yaml(params["config-path"])
    job_name = f"{basic_config['MODEL']}_{command_name}_sbatch"
    script = f"{basic_config['MODEL']}_{command_name}_sbatch.sh"
    sbatch_kwargs = dict(
        account=account,
        partition=partition,
        num_gpus=num_gpus,
        cores=cores,
        mem=mem,
        job_name=job_name,
        output=f"{job_name}.out",
        error=f"{job_name}.err",
        time=time,
        command=command,
        n_jobs_in_array=n_jobs_in_array,
        env_vars=env_vars,
    )
    sbatch_script = create_sbatch_script(**sbatch_kwargs)
    write_sbatch(script, sbatch_script)
    if script_only:
        logger.info(f"Generated sbatch script: {script}")
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.end_run()
        return
    target.mkdir(exist_ok=True, parents=True)
    if command_name == "generate":
        logger.info(f"Simulated data output folder: {target}")
    else:
        logger.info(f"Trained networks output folder: {target}")
    submit_sbatch(script, logger)

    if MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.end_run()

    # Return experiment ID for data generation (useful for chaining to training)
    if command_name == "generate" and mlflow_experiment_id:
        logger.info("=" * 60)
        logger.info(f"DATA GENERATION EXPERIMENT ID: {mlflow_experiment_id}")
        logger.info(
            "Use this ID with training commands via --data-generation-experiment-id"
        )
        logger.info("=" * 60)

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
    account: str = typer.Option("default", help="Condo to run the SBATCH job on"),
    partition: str = typer.Option(
        "batch", help="Partition to run the SBATCH script on"
    ),
    num_gpus: int = typer.Option(
        0, help="Number of GPUs requested (for use on gpu partition)"
    ),
    mem: str = typer.Option("16G", help="Memory limit for each job"),
    time: str = typer.Option("00:30:00", help="Wall time limit for each job"),
    cores: int = typer.Option(1, help="Number of tasks (cores) to run in parallel"),
    script_only: bool = typer.Option(
        False, help="Generate the sbatch script without submitting the job"
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
        account=account,
        partition=partition,
        num_gpus=num_gpus,
        cores=cores,
        mem=mem,
        n_jobs_in_array=n_jobs_in_array,
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
        account: str = typer.Option("default", help="Condo to run the SBATCH job on"),
        partition: str = typer.Option(
            "batch", help="Partition to run the SBATCH script on"
        ),
        num_gpus: int = typer.Option(
            0, help="Number of GPUs requested (for use on gpu partition)"
        ),
        cores: int = typer.Option(1, help="Number of tasks (cores) to run in parallel"),
        dl_workers: int = typer.Option(
            1, help="Number of cores to use with the dataloader class"
        ),
        time: str = typer.Option("00:30:00", help="Wall time limit for each job"),
        mem: str = typer.Option("16G", help="Memory limit for each job"),
        script_only: bool = typer.Option(
            False, help="Generate the sbatch script without submitting the job"
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
