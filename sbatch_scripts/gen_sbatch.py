from pathlib import Path
import logging
import subprocess


import typer
import yaml

app = typer.Typer(add_completion=False)

# SBATCH template
SBATCH_TEMPLATE = """#!/bin/bash

#SBATCH --mem={mem}
#SBATCH -J {job_name}
#SBATCH --time={time}
#SBATCH --output={output}
#SBATCH --error={error}
#SBATCH --array=1-{array_size}

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
    job_name="job",
    output="output.txt",
    error="error.txt",
    time="01:00:00",
    mem="4G",
    command="",
    array_size=1,
):
    sbatch_script = SBATCH_TEMPLATE.format(
        mem=mem,
        job_name=job_name,
        time=time,
        output=output,
        error=error,
        command=command,
        array_size=array_size,
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
    params = {"config-path": config_path.resolve(), "log-level": log_level}
    if command == "generate":
        params["output"] = output_path.resolve()
    elif command in ["jaxtrain", "torchtrain"]:
        params.update(
            {
                "networks-path-base": output_path.resolve(),
                "training-data-folder": training_data_folder.resolve()
                if training_data_folder
                else None,
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
    sh_only: bool,
    array_size: int = 1,
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
        command_name,
        config_path,
        output_path,
        log_level,
        training_data_folder,
        network_id,
        dl_workers,
    )
    command = create_command(command_name, **params)
    logger.info(f"Generated command: {command}")
    bc = get_basic_config_from_yaml(params["config-path"])
    job_name = f"{bc['MODEL']}_{command_name}_sbatch"
    script = f"{bc['MODEL']}_{command_name}_sbatch.sh"
    sbatch_kwargs = dict(
        job_name=job_name,
        output=f"{job_name}.out",
        error=f"{job_name}.err",
        time=time,
        command=command,
        mem="16G",
    )
    if command_name == "generate":
        sbatch_kwargs["array_size"] = array_size
    sbatch_script = create_sbatch_script(**sbatch_kwargs)
    write_sbatch(script, sbatch_script)
    if sh_only:
        logger.info(f"Generated sbatch script: {script}")
        return
    target.mkdir(exist_ok=True, parents=True)
    if command_name == "generate":
        logger.info(f"Simulated data output folder: {target}")
        submit_sbatch(script, logger)
        logger.info("Job submitted successfully")
    else:
        logger.info(f"Trained networks output folder: {target}")
        submit_sbatch(script, logger)


@app.command()
def generate(
    config_path: Path = typer.Option(
        ..., help="Path to configuration .yaml file for running commands"
    ),
    output_path: Path = typer.Option(
        ..., help="Path to output folder for simulated data"
    ),
    array_size: int = typer.Option(1, help="Size of the job array"),
    time: str = typer.Option("00:30:00", help="Wall time limit for each job"),
    sh_only: bool = typer.Option(
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
        sh_only=sh_only,
        array_size=array_size,
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
        dl_workers: int = typer.Option(
            1, help="Number of cores to use with the dataloader class"
        ),
        time: str = typer.Option("00:30:00", help="Wall time limit for each job"),
        sh_only: bool = typer.Option(
            False, help="Generate the sbatch script without submitting the job"
        ),
        log_level: str = typer.Option(
            "WARNING", help="Set the log level", show_default=True
        ),
    ):
        handle_job(
            command_name=command_name,
            config_path=config_path,
            output_path=output_path,
            log_level=log_level,
            time=time,
            sh_only=sh_only,
            training_data_folder=training_data_folder,
            network_id=network_id,
            dl_workers=dl_workers,
        )


if __name__ == "__main__":
    app()
