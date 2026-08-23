# Pipeline configuration files

The [configuration reference](https://lnccbrown.github.io/LAN_pipeline_minimal/reference/configuration/)
is the canonical description of generation, training, cluster, and personal
overlay schemas. For task-oriented instructions, use the guides to
[configure cluster resources](https://lnccbrown.github.io/LAN_pipeline_minimal/how-to/configure-cluster/)
and [generate Slurm jobs](https://lnccbrown.github.io/LAN_pipeline_minimal/how-to/submit-slurm-jobs/).

## Directory map

```text
configs/
├── examples/       # larger generation and LAN/CPN training templates
├── quick_test/     # small configs exercised by local_test_run.sh and CI
└── cluster/
    ├── oscar.yaml        # committed lab/cluster inventory and defaults
    └── oscar.local.yaml  # generated personal overlay; gitignored
```

Start with `quick_test/` when checking a checkout. Copy and review an example
before adapting it to a scientific run; the example scale is not a universal
production recommendation.

Generate your personal cluster overlay rather than editing it:

```bash
uv run python scripts/discover_cluster.py --ssh-host oscar
```

Passing `--cluster-config configs/cluster/oscar.yaml` automatically merges the
adjacent local overlay. The configuration reference owns the complete resource
precedence and merge contract.

Quote Slurm wall times in YAML, for example `time: "12:00:00"`. Do not commit
personal accounts, paths, quotas, or credentials.
