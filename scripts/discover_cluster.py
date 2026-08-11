#!/usr/bin/env python3
"""Ask the cluster what *this* user may actually schedule, and write it down.

The committed `configs/cluster/oscar.yaml` records facts that are the same for
everyone in the lab. What differs per person is which associations they hold and
therefore how much they can run at once — and nobody can answer that from
memory, because the numbers that matter live in the QOS, not the partition
(`scontrol show partition` reports MaxTime=UNLIMITED for everything).

This writes `configs/cluster/oscar.local.yaml`, which is gitignored: it is
*your* allocation, not the lab's. `gen_sbatch` merges it over the committed
file automatically.

    # from a login node
    uv run python scripts/discover_cluster.py

    # or from a laptop, over the ssh config entry
    uv run python scripts/discover_cluster.py --ssh-host oscar

A "lane" is one (account, partition, QOS) triple you are allowed to submit to.
Its cap is `MaxTRESPU` on the QOS, which SLURM applies *per QOS*, so two lanes
are two independent budgets — running in both really does give you their sum.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "configs/cluster/oscar.local.yaml"
)


def run(command: str, ssh_host: str | None) -> str:
    """Run a read-only cluster query, locally or over ssh."""
    argv = (
        ["ssh", "-o", "BatchMode=yes", ssh_host, command]
        if ssh_host
        else ["bash", "-lc", command]
    )
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"`{command}` failed ({result.returncode}): {result.stderr.strip()[:200]}\n"
            + (
                "Run this on a login node, or pass --ssh-host."
                if ssh_host is None
                else "Check that your ssh session is authenticated (one Duo prompt)."
            )
        )
    return result.stdout


# QOS that exist on every account but are not for pipeline work: debug queues
# are time-boxed, vnc is for interactive desktops.
NON_COMPUTE_QOS = {"debug", "gpu-debug", "vnc"}


def parse_cap(tres: str) -> dict:
    """`cpu=208,mem=2500G,gres/gpu=75` -> {'cores': 208, 'mem': '2500G', 'gpus': 75}."""
    cap: dict = {}
    for item in filter(None, tres.split(",")):
        key, _, value = item.partition("=")
        if key == "cpu":
            cap["cores"] = int(value)
        elif key == "mem":
            cap["mem"] = value
        elif key.startswith("gres/gpu"):
            cap["gpus"] = int(value)
    return cap


def discover(ssh_host: str | None) -> dict:
    associations = run(
        "sacctmgr -nP show associations user=$USER format=Account,Partition,QOS",
        ssh_host,
    )
    lanes: list[dict] = []
    qos_names = set()
    for line in associations.strip().splitlines():
        fields = line.split("|")
        if len(fields) < 3 or not fields[1]:
            continue  # rows without a partition are the account-level parent
        account, partition, qos = fields[0], fields[1], fields[2]
        lanes.append({"account": account, "partition": partition, "qos": qos})
        qos_names.add(qos)

    if not lanes:
        raise RuntimeError("No associations found — is this the right cluster account?")

    qos_rows = run(
        "sacctmgr -nP show qos where name="
        + ",".join(sorted(qos_names))
        + " format=Name,Priority,MaxWall,MaxTRESPU,MaxSubmitJobsPU",
        ssh_host,
    )
    qos_info = {}
    for line in qos_rows.strip().splitlines():
        f = (line.split("|") + [""] * 5)[:5]
        qos_info[f[0]] = {
            "priority": int(f[1]) if f[1].isdigit() else 0,
            "max_walltime": f[2] or None,
            "cap": parse_cap(f[3]),
            "max_submit_jobs": int(f[4]) if f[4].isdigit() else None,
        }

    for lane in lanes:
        info = qos_info.get(lane["qos"], {})
        lane["priority"] = info.get("priority", 0)
        lane["max_walltime"] = info.get("max_walltime")
        lane.update(info.get("cap", {}))
        if info.get("max_submit_jobs"):
            lane["max_submit_jobs"] = info["max_submit_jobs"]

    # Best lane first: highest priority, then most cores.
    lanes.sort(key=lambda x: (-x["priority"], -x.get("cores", 0)))
    return {"lanes": lanes}


def suggest_job_defaults(lanes: list[dict]) -> dict:
    """Map the discovered lanes onto the pipeline's job kinds.

    Datagen is embarrassingly parallel and CPU-bound, so it can use every CPU
    lane at once. Training wants one GPU on the highest-priority GPU lane;
    spilling a single training job across lanes would not help it.
    """
    # A usable compute lane needs a known core cap and must not be a debug or
    # vnc queue: those exist for everyone and would inflate the totals.
    usable = [x for x in lanes if x["qos"] not in NON_COMPUTE_QOS and x.get("cores")]
    cpu = [x for x in usable if not x.get("gpus")]
    gpu = [x for x in usable if x.get("gpus")]

    defaults: dict = {}
    if cpu:
        best = cpu[0]
        defaults["generate"] = {
            "account": best["account"],
            "partition": best["partition"],
            "lanes": [
                {
                    "account": x["account"],
                    "partition": x["partition"],
                    "max_cores": x["cores"],
                    "priority": x["priority"],
                }
                for x in cpu
            ],
        }
    if gpu:
        best = gpu[0]
        for kind in ("jaxtrain", "torchtrain"):
            defaults[kind] = {
                "account": best["account"],
                "partition": best["partition"],
                "num_gpus": 1,
            }
    return defaults


def to_yaml(data: dict, ssh_host: str | None) -> str:
    import yaml

    header = f"""# Per-user cluster lanes — GENERATED, gitignored, do not hand-edit.
#
# Written by `scripts/discover_cluster.py{" --ssh-host " + ssh_host if ssh_host else ""}`.
# These are YOUR associations and caps; the lab-wide facts live in the
# committed oscar.yaml, which this file is merged over.
#
# `cores` on each lane is MaxTRESPU from its QOS. SLURM applies that cap per
# QOS, so separate lanes are separate budgets and their capacity adds up.
# Re-run discovery whenever your allocations change.
"""
    return header + yaml.safe_dump(data, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ssh-host", help="Run the queries over ssh (e.g. oscar).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--print", action="store_true", help="Print instead of writing."
    )
    args = parser.parse_args()

    try:
        discovered = discover(args.ssh_host)
    except RuntimeError as e:
        print(f"discovery failed: {e}", file=sys.stderr)
        return 1

    lanes = discovered["lanes"]
    discovered["job_defaults"] = suggest_job_defaults(lanes)

    print(f"Found {len(lanes)} lanes:", file=sys.stderr)
    for lane in lanes:
        gpus = f", {lane['gpus']} gpus" if lane.get("gpus") else ""
        print(
            f"  {lane['account']:26s} {lane['partition']:10s} "
            f"qos={lane['qos']:16s} priority={lane['priority']:<6d} "
            f"{lane.get('cores', '?')} cores{gpus}",
            file=sys.stderr,
        )
    cpu_lanes = [
        x
        for x in lanes
        if x["qos"] not in NON_COMPUTE_QOS and x.get("cores") and not x.get("gpus")
    ]
    total = sum(x["cores"] for x in cpu_lanes)
    print(
        f"  -> {total} CPU cores usable for datagen across {len(cpu_lanes)} lane(s)\n",
        file=sys.stderr,
    )

    text = to_yaml(discovered, args.ssh_host)
    if args.print:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"wrote {args.output}", file=sys.stderr)
        print(
            "gen_sbatch picks it up automatically; check with:\n"
            "  uv run python sbatch_scripts/gen_sbatch.py generate --help",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
