#!/usr/bin/env python3
import typer
from pathlib import Path
import subprocess
from typing import Optional
from rich.console import Console
from rich import print as rprint
from enum import Enum

app = typer.Typer(help="LAN Pipeline CLI tools")
console = Console()


class Backend(str, Enum):
    jax = "jax"
    torch = "torch"


def generate_sbatch_script(
    job_name: str,
    output_dir: Path,
    time: str,
    memory: str,
    cores: int,
    nodes: int,
    config_path: Path,
    data_gen_base_path: Path,
    account: Optional[str] = None,
) -> str:
    """Generate SBATCH script with the given parameters."""

    script_content = f"""#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J {job_name}

# output file
#SBATCH --output {output_dir}/slurm_{job_name}_%A_%a.out

# Request runtime, memory, cores
#SBATCH --time={time}
#SBATCH --mem={memory}
#SBATCH -c {cores}
#SBATCH -N {nodes}
{f"#SBATCH --account={account}" if account else ""}

# BASIC SETUP

set -x  # Enable command echo
set -e  # Exit on error

echo "Starting job at: $(date)"
echo "Running on node: $(hostname)"
echo "Current working directory: $(pwd)"

# Load Python module
module load python

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

config_path="{config_path}"
data_gen_base_path="{data_gen_base_path}"

echo "The config file supplied is: $config_path"

python -m uv run scripts/data_generation_script.py --config-path $config_path \\
                                            --data-gen-base-path $data_gen_base_path

echo "Job completed at: $(date)"
"""
    return script_content


def generate_training_script(
    job_name: str,
    output_dir: Path,
    time: str,
    memory: str,
    cores: int,
    nodes: int,
    config_path: Path,
    networks_path_base: Path,
    n_networks: int,
    backend: Backend,
    dl_workers: int,
    use_gpu: bool = False,
    array_range: Optional[str] = None,
    account: Optional[str] = None,
) -> str:
    """Generate SBATCH script for network training with the given parameters."""

    script_content = f"""#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J {job_name}

# output file
#SBATCH --output {output_dir}/slurm_{job_name}_%A_%a.out

# Request runtime, memory, cores
#SBATCH --time={time}
#SBATCH --mem={memory}
#SBATCH -c {cores}
#SBATCH -N {nodes}
{f"#SBATCH --account={account}" if account else ""}

{f"#SBATCH -p gpu --gres=gpu:1" if use_gpu else "##SBATCH -p gpu --gres=gpu:1"}
{f"#SBATCH --array={array_range}" if array_range else "##SBATCH --array=0-8"}

# --------------------------------------------------------------------------------------

set -x  # Enable command echo
set -e  # Exit on error

echo "Starting job at: $(date)"
echo "Running on node: $(hostname)"
echo "Current working directory: $(pwd)"

# Load Python module
module load python

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

config_path="{config_path}"
networks_path_base="{networks_path_base}"
dl_workers={dl_workers}
n_networks={n_networks}
backend="{backend}"

echo "Using config path: $config_path"
echo "Using networks path: $networks_path_base"
echo "Number of networks: $n_networks"
echo "Backend: $backend"

# Use uv to run Python scripts
if [ -z "$SLURM_ARRAY_TASK_ID" ];
then
    for ((i = 1; i <= $n_networks; i++))
        do
            echo "NOW TRAINING NETWORK: $i of $n_networks"
            echo "No array ID"
            
            if [ "$backend" == "jax" ]; then
                echo "Running JAX training script..."
                python -m uv run scripts/jax_training_script.py --config-path $config_path \\
                                                         --network-id 0 \\
                                                         --networks-path-base $networks_path_base \\
                                                         --dl-workers $dl_workers
            elif [ "$backend" == "torch" ]; then
                echo "Running PyTorch training script..."
                python -m uv run scripts/torch_training_script.py --config-path $config_path \\
                                                           --network-id 0 \\
                                                           --networks-path-base $networks_path_base \\
                                                           --dl-workers $dl_workers
            fi
        done
else
    for ((i = 1; i <= $n_networks; i++))
        do
            echo "NOW TRAINING NETWORK: $i of $n_networks"
            echo "Array ID is $SLURM_ARRAY_TASK_ID"
            
            if [ "$backend" == "jax" ]; then
                echo "Running JAX training script with array ID..."
                python -m uv run scripts/jax_training_script.py --config-path $config_path \\
                                                         --network-id $SLURM_ARRAY_TASK_ID \\
                                                         --networks-path-base $networks_path_base \\
                                                         --dl-workers $dl_workers
            elif [ "$backend" == "torch" ]; then
                echo "Running PyTorch training script with array ID..."
                python -m uv run scripts/torch_training_script.py --config-path $config_path \\
                                                           --network-id $SLURM_ARRAY_TASK_ID \\
                                                           --networks-path-base $networks_path_base \\
                                                           --dl-workers $dl_workers
            fi
        done
fi

echo "Job completed at: $(date)"
"""
    return script_content


@app.command()
def generate(
    config_path: Path = typer.Argument(..., help="Path to configuration file", exists=True),
    job_name: str = typer.Option("data_generator", "--job-name", "-j", help="Name of the SLURM job"),
    output_dir: Path = typer.Option(Path("logs"), "--output-dir", "-o", help="Directory for SLURM output files"),
    time: str = typer.Option("48:00:00", "--time", "-t", help="Requested runtime in format HH:MM:SS"),
    memory: str = typer.Option("16G", "--memory", "-m", help="Requested memory (e.g., 16G)"),
    cores: int = typer.Option(12, "--cores", "-c", help="Number of CPU cores"),
    nodes: int = typer.Option(1, "--nodes", "-n", help="Number of nodes"),
    data_gen_base_path: Path = typer.Option(
        Path("data"), 
        "--data-base",
        "-d",
        help="Base path for data generation"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only generate script without running"),
    account: Optional[str] = typer.Option(None, "--account", "-a", help="SLURM account to use"),
):
    """Generate and optionally submit a SLURM job for data generation."""

    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate script content
        script_content = generate_sbatch_script(
            job_name=job_name,
            output_dir=output_dir,
            time=time,
            memory=memory,
            cores=cores,
            nodes=nodes,
            config_path=config_path,
            data_gen_base_path=data_gen_base_path,
            account=account,
        )

        # Write script to temporary file
        script_path = output_dir / f"temp_{job_name}.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        rprint(f"[green]Generated SBATCH script at:[/green] {script_path}")

        if not dry_run:
            with console.status("Submitting job to SLURM..."):
                subprocess.run(["sbatch", str(script_path)], check=True)
            rprint("[green]Job submitted successfully! :rocket:[/green]")
        else:
            rprint("[yellow]Dry run - script generated but not submitted[/yellow]")

    except Exception as e:
        rprint(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def train(
    config_path: Path = typer.Argument(..., help="Path to configuration file", exists=True),
    job_name: str = typer.Option("model_trainer", "--job-name", "-j", help="Name of the SLURM job"),
    output_dir: Path = typer.Option(Path("logs"), "--output-dir", "-o", help="Directory for SLURM output files"),
    time: str = typer.Option("32:00:00", "--time", "-t", help="Requested runtime in format HH:MM:SS"),
    memory: str = typer.Option("32G", "--memory", "-m", help="Requested memory (e.g., 32G)"),
    cores: int = typer.Option(12, "--cores", "-c", help="Number of CPU cores"),
    nodes: int = typer.Option(1, "--nodes", "-n", help="Number of nodes"),
    networks_path_base: Path = typer.Option(
        Path("networks"),
        "--networks-base",
        help="Base path for networks"
    ),
    n_networks: int = typer.Option(
        2, "--n-networks", "-N", help="Number of networks to train"
    ),
    backend: Backend = typer.Option(
        Backend.jax, "--backend", "-b", help="Deep learning backend to use"
    ),
    dl_workers: int = typer.Option(
        4, "--dl-workers", "-w", help="Number of dataloader workers"
    ),
    use_gpu: bool = typer.Option(False, "--gpu", help="Whether to use GPU resources"),
    array_range: Optional[str] = typer.Option(
        None, "--array", help="SLURM array range (e.g., '0-8')"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Only generate script without running"
    ),
    account: Optional[str] = typer.Option(None, "--account", "-a", help="SLURM account to use"),
):
    """Generate and optionally submit a SLURM job for network training."""

    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate script content
        script_content = generate_training_script(
            job_name=job_name,
            output_dir=output_dir,
            time=time,
            memory=memory,
            cores=cores,
            nodes=nodes,
            config_path=config_path,
            networks_path_base=networks_path_base,
            n_networks=n_networks,
            backend=backend,
            dl_workers=dl_workers,
            use_gpu=use_gpu,
            array_range=array_range,
            account=account,
        )

        # Write script to temporary file
        script_path = output_dir / f"temp_{job_name}.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        rprint(f"[green]Generated SBATCH script at:[/green] {script_path}")

        if not dry_run:
            with console.status("Submitting job to SLURM..."):
                subprocess.run(["sbatch", str(script_path)], check=True)
            rprint("[green]Job submitted successfully! :rocket:[/green]")
        else:
            rprint("[yellow]Dry run - script generated but not submitted[/yellow]")

    except Exception as e:
        rprint(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
