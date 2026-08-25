# Configure your cluster resources

The committed `configs/cluster/oscar.yaml` describes lab-wide facts. Your
accounts, QOS limits, and usable account/partition combinations are personal,
so discovery writes them to the gitignored
`configs/cluster/oscar.local.yaml`.

!!! info "Execution status"

    Repository CI tests local configuration merging and renders Slurm scripts
    with `--script-only`. It cannot query your allocation or exercise an SSH
    host. Run the discovery commands below manually from an authenticated
    operator environment whenever allocations or QOS limits change.

## Discover your lanes

On an Oscar login node, run:

```bash
uv run python scripts/discover_cluster.py
```

From a laptop with an authenticated SSH config entry:

```bash
uv run python scripts/discover_cluster.py --ssh-host oscar
```

The command makes read-only Slurm queries, reports the lanes it found, and
writes the local overlay. A **lane** is one account/partition/QOS combination.
QOS limits apply independently, so separate CPU lanes provide additive array
capacity even when one is lower priority.

!!! note "Re-run discovery when allocations change"

    The limits that matter are `MaxTRESPU` values on the QOS. They cannot be
    recovered reliably from the partition listing or from another operator's
    config. Do not hand-edit the generated local file; discovery overwrites it.

## Review the merge

Passing the committed path is enough:

```bash
uv run python sbatch_scripts/gen_sbatch.py generate \
  --config-path configs/quick_test/data_generation.yaml \
  --output-path /tmp/lan-pipeline-plan \
  --cluster-config configs/cluster/oscar.yaml \
  --script-only
```

`gen_sbatch` automatically merges the adjacent local overlay. The
[configuration reference](../reference/configuration.md#personal-overlay-and-precedence)
owns the exact merge and precedence contract; explicit CLI resource flags are
the final overrides.

Review the JSON object's `account` and `partition`, then read the generated
script to confirm cores, memory, GPU count, wall time, and loaded modules.

## Map work onto lanes

Discovery chooses defaults according to the workload:

- `generate` uses the highest-priority CPU lane and records every usable CPU
  lane for optional array fan-out;
- `jaxtrain` and `torchtrain` use one GPU on the highest-priority GPU lane;
- debug and VNC QOS entries are excluded from pipeline capacity.

Data generation is embarrassingly parallel, so `--use-all-lanes` can split its
array proportionally to lane core caps. A single training job cannot span lanes;
the flag is therefore available only for generation.

## Keep personal state out of the repository

Do not put these values in the committed cluster config:

- usernames, personal account memberships, or per-user caps;
- tokens or SSH details;
- home or shared-storage paths;
- local MLflow database paths.

Pass paths at invocation time and set tracking locations through environment
variables. The local overlay itself is gitignored; verify that with `git status`
before committing unrelated changes.
