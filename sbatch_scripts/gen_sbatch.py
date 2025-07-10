#!/usr/bin/env python3
"""
Generates and submits individual sbatch job for generating simulated data for a given model, and/or for training neural network on simulated data.
"""

from pathlib import Path
import logging
import subprocess
import typer
import yaml

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

pip install uv
python -m uv run {command}
"""

def create_command(command_name: str, **params: dict):
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
    )
    return sbatch_script

def write_sbatch(script, sbatch_script):
    with open(script, "w") as f:
        f.write(sbatch_script)

def submit_sbatch(script, logger):
    try:
        result = subprocess.run(["sbatch", script], capture_output=True, text=True)
        logger.info(result.stdout)
        if result.stderr:
            logger.error(result.stderr)
    except Exception as e:
        logger.error(f"Failed to submit job: {e}")

def get_basic_config_from_yaml(yaml_config_path: str | Path):
    basic_config = yaml.safe_load(open(yaml_config_path, "rb"))
    return basic_config

def get_parameters_setup(
    command: str,
    config_path: Path,
    output_path: Path,
    log_level: str,
    training_data_folder: Path = None,
    network_id: int = 0,
    dl_workers: int = 1,
):
    """
    Prepare cli arguments for the command based on the command type.
    """
    params = {"config-path": config_path.resolve(), "log-level": log_level}
    if command == "generate":
        params["output"] = output_path.resolve()
    elif command in ["jaxtrain", "torchtrain"]:
        params.update(
            {
                "networks-path-base": output_path.resolve(),
                "training-data-folder": training_data_folder.resolve(),
                "network-id": network_id,
                "dl-workers": dl_workers,
            }
        )
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
):
    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, log_level),
    )
    logger = logging.getLogger("gen_sbatch")
    target = output_path.resolve()
    params = get_parameters_setup(
        command=command_name,
        config_path=config_path,
        output_path=output_path,
        log_level=log_level,
        training_data_folder=training_data_folder,
        network_id=network_id,
        dl_workers=dl_workers,
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
        n_jobs_in_array=n_jobs_in_array
    )
    sbatch_script = create_sbatch_script(**sbatch_kwargs)
    write_sbatch(script, sbatch_script)
    if script_only:
        logger.info(f"Generated sbatch script: {script}")
        return
    target.mkdir(exist_ok=True, parents=True)
    if command_name == "generate":
        logger.info(f"Simulated data output folder: {target}")
    else:
        logger.info(f"Trained networks output folder: {target}")
    submit_sbatch(script, logger)
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
    partition: str = typer.Option("batch", help="Partition to run the SBATCH script on"),
    num_gpus: int = typer.Option(0, help="Number of GPUs requested (for use on gpu partition)"),
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
        account: str = typer.Option("default", help="Condo to run the SBATCH job on"),
        partition: str = typer.Option("batch", help="Partition to run the SBATCH script on"),
        num_gpus: int = typer.Option(0, help="Number of GPUs requested (for use on gpu partition)"),
        cores: int = typer.Option(1, help="Number of tasks (cores) to run in parallel"),
        dl_workers: int = typer.Option(1, help="Number of cores to use with the dataloader class"),
        time: str = typer.Option("00:30:00", help="Wall time limit for each job"),
        mem: str = typer.Option("16G", help="Memory limit for each job"),
        ntasks: int = typer.Option(1, help="Number of tasks (cores) to run in parallel"),
        script_only: bool = typer.Option(False, help="Generate the sbatch script without submitting the job"),
        log_level: str = typer.Option("WARNING", help="Set the log level", show_default=True),
    ):
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
        )

    return train


app.command("jaxtrain")(train_command("jaxtrain"))
app.command("torchtrain")(train_command("torchtrain"))

if __name__ == "__main__":
    app()
