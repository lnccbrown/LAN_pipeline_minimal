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
├── production_ddm_sdv/
├── production_gamma_drift/
├── production_gamma_drift_angle/
└── cluster/
    ├── oscar.yaml        # committed lab/cluster inventory and defaults
    └── oscar.local.yaml  # generated personal overlay; gitignored
```

Production-scale configurations are committed as `production_<model>/` pairs
so a completed run or predeclared candidate is reproducible from the repo
alone. A directory preserves the intended configuration; its presence does not
by itself prove that the run completed or that an artifact shipped.
`tests/test_production_configs.py`
checks every pair for the mistakes that are only wrong in context — a model
name the simulator does not know, an architecture without matching
activations, a GPU batch that leaves a remainder on a training file — because
none of those fail loudly, and all of them fail late. The 2023-era legacy networks on
franklab/HSSM predate this pipeline and have no recorded configs; their
provenance is a registry concern, not something to reconstruct here.

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
