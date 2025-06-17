#!/usr/bin/env python3
"""
Generates and submits individual sbatch job for generating simulations of a model, and/or for training neural network on those simulations
"""
# Imports
import argparse
from argparse import RawDescriptionHelpFormatter
from pathlib import Path
import logging
import subprocess
import yaml

# SBATCH template
SBATCH_TEMPLATE = """#!/bin/bash

# specifies sbatch resources needed for the job

#SBATCH --mem={mem}
#SBATCH -J {job_name}
#SBATCH --time={time}
#SBATCH --output={output}
#SBATCH --error={error}

# Your commands here
{command}
"""

def create_command(command_name, **params):
    """
    Creates a full CLI command to be run using the SBATCH script.

    Parameters:
        command_name (string): name of the Python file you are going to run (generate.py or jax_train.py)
        params (dictionary): dictionary of parameters to be used with the Python file

    Returns:
        command (string): full CLI command, with the script to be run, and all parameters

    """
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
):
    """
    Creates a SBATCH script using the SBATCH template.

    Parameters:
        job_name (string): Name for SBATCH job
        outputs (string): Name for output file after running SBATCH job
        error (string): Name for the error file after running SBATCH job
        time (string): Time limit for job
        mem (string): Amount of memory in GB to be used with SBATCH job
        command (string): Python command(s) to run after setting up all SBATCH metadata

    Returns:
        sbatch_script (sbatch_template): SBATCH script with metadata used for running SBATCH jobs

    """

    sbatch_script = SBATCH_TEMPLATE.format(
        mem=mem,
        job_name=job_name,
        time=time,
        output=output,
        error=error,
        command=command,
    )
    return sbatch_script

def write_sbatch(script, sbatch_script):
    """
    Adds SBATCH metadata to .sh script

    Parameters:
        script (string): .sh file that will be turned into an SBATCH script
        sbatch_script (SBATCH file): output from the create_sbatch_script function with SBATCH metadata

    Returns:
        None.
    """
    with open(script, "w") as f:
        f.write(sbatch_script)

def submit_sbatch(script, logger):
    """
    Submits SBATCH .sh file to OSCAR.

    Parameters:
        script (SBATCH .sh): script with SBATCH metadata added after running write_sbatch
        logger (Logger): logger from file
    """
    try:
        result = subprocess.run(["sbatch", script], capture_output=True, text=True)
        logger.info(result.stdout)
        if result.stderr:
            logger.error(result.stderr)
    except Exception as e:
        logger.error(f"Failed to submit job: {e}")

def get_basic_config_from_yaml(
    yaml_config_path: str | Path
):
    """
    Load the basic configuration from a YAML file. Modified from the generate.py file
    
    Paramters:
        yaml_config_path (string or Path): path to the .yaml config file for use with generate.py

    Returns:
        basic_config (dictionary?): basic configuration dictionary, taken from the config.yaml file
    
    """
    basic_config = yaml.safe_load(open(yaml_config_path, "rb"))
    return basic_config

    

def main():

    # params = {"config-path": "/path/to/config.yaml", "output-path" : "/path/to/output_folder"}

    # command = create_command("generate.py", **params)

    # print(command)

    # sbatch_script = create_sbatch_script
    # print(sbatch_script)

    # Setting up argument parsing 

    description = __doc__ # docstring for gen_sbatch
    prog = "gen_sbatch" # program name

    epilog = (
        f"Example:\n    {prog} generate.py --config-path path/to/config.yaml --output-path path/to/output/folder --log-level INFO\n"
        f"    {prog} jax_train.py --config-path path/to/config.yaml --training-data-folder path/to/training_data --network-id 0 --dl-workers 4 --networks-path-base path/to/trained_network_output --log-level INFO\n" # examples for generate and jaxtrain scripts
    )
    parser = argparse.ArgumentParser(
        description = description,
        prog = prog,
        epilog = epilog,
        formatter_class=RawDescriptionHelpFormatter 
    )

    # Common arguments for both generate and jaxtrain
    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument(
        "--config-path",
        default="dump",
        help="path to config file for running commands (default: 'dump')",
        type=str,
        required=True
    )
    parent_parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="set the log level",
    )
    parent_parser.add_argument(
        "--sh-only",
        action="store_true",
        help="generate sbatch script only, do not submit the job",
    )
    parent_parser.add_argument(
        "--time", help="time limit for each job (default: 24:00:00)", default="24:00:00"
    )

    # Subparsers for arguments specific to generate and jaxtrain
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
    )
    # Generate args
    generate_parser = subparsers.add_parser(
        "generate", help = "Generates simulated data from model parameters", parents=[parent_parser]
    )
    generate_parser.add_argument(
        "--output-path",
        help="Path to output folder for simulated data",
        type=str,
        required=True
    )

    # Jaxtrain args
    jax_train_parser = subparsers.add_parser(
        "jax_train", help = "Trains a neural network using simulated data", parents=[parent_parser]
    )
    jax_train_parser.add_argument(
        "--training-data-folder",
        help="Path to folder with data to train the neural network on",
        type=str,
        required=True
    )
    jax_train_parser.add_argument(
        "--network-id",
        type=int,
        help="Id for the neural network to train (default=0)",
        default=0,
    )
    jax_train_parser.add_argument(
        "--dl-workers",
        type=int,
        help="Number of cores to use with the dataloader class (default=1)",
        default=1,
        required=True
    )
    jax_train_parser.add_argument(
        "--networks-path-base",
        type=str,
        help="output folder for the trained neural net",
        required=True
    )

    args = parser.parse_args()

    # Setting up logging
    logger = logging.getLogger(__file__)
    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, args.log_level)
    )

    if args.command == "generate":
        
        # target folder for generate
        target = Path(args.output_path)
        target.mkdir(exist_ok=True, parents=True)
        print("target:", target)
        #print("Vars(args)", vars(args))

        # get parameters from command from the arguments parser
        params = {
            "config-path": args.config_path,
            "output-path": args.output_path,
            "log-level": args.log_level,
            "sh-only": args.sh_only, 
            "time": args.time
        }
        #print("Parameter dictionary:", params)

        command = create_command("generate.py", **params)
        print("Command:", command)

        # Something feels very circular about my logic here. I am running sbatch.py, and then the command "generate". Then I am specifying the arguments needed to run the generate.py script AND the arguments for the sbatch.py script. Those arguments are used to create a command that's going to be added to my Sbatch template, but why can't I just take the string of what I wrote after gen_sbatch.py generate <all arguments?> 

        bc = get_basic_config_from_yaml(params["config-path"])
        print(bc)
        
        job_name = f"{bc["MODEL"]}_generate_sbatch" #TODO: need to figure out how to get better job names and script names later
        script = f"{bc["MODEL"]}_generate_sbatch.sh"

        sbatch_script = create_sbatch_script(
                job_name=job_name,
                output=f"{target / job_name}.out",
                error=f"{target / job_name}.err",
                time=args.time,
                command=command,
                mem="16G"
            )

        # Run sbatch
        write_sbatch(script, sbatch_script)

        # # Generate sbatch only, do not submit job
        # if args.sh_only:
        #     logger.info(f"Generated sbatch script: {script}")
        #     return

        # # Submits job
        # submit_sbatch(script, logger)

    elif args.command == "jax_train":
        print("Vars(args):", vars(args))
        
        # target folder for jaxtrain
        target = Path(args.networks_path_base)
        target.mkdir(exist_ok=True, parents=True)

        params = {
            "config-path": args.config_path,
            "training-data-folder": args.training_data_folder,
            "network-id": args.network_id,
            "dl-workers": args.dl_workers,
            "log-level": args.log_level,
            "sh-only": args.sh_only, 
            "time": args.time
        }
        print("Parameter dictionary:", params)

        command = create_command("jax_train.py", **params)
        print("Command:", command)

        bc = get_basic_config_from_yaml(params["config-path"])
        #print(bc)

        job_name = f"{bc["MODEL"]}_jaxtrain_sbatch" #TODO: need to figure out how to get better job names and script names later
        script = f"{bc["MODEL"]}_jaxtrain_sbatch.sh"

        sbatch_script = create_sbatch_script(
                job_name=job_name,
                output=f"{target / job_name}.out",
                error=f"{target / job_name}.err",
                time=args.time,
                command=command,
                mem="16G"
            )

        # Run sbatch
        write_sbatch(script, sbatch_script)

        # # Generate sbatch only, do not submit job
        # if args.sh_only:
        #     logger.info(f"Generated sbatch script: {script}")
        #     return

        # # Submits job
        # submit_sbatch(script, logger)

# Main
if __name__ == "__main__":
    main()

