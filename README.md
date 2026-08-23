# LAN_pipeline_minimal

Operational orchestration for generating sequential-sampling-model training
data, training likelihood approximation networks, validating ONNX candidates,
and publishing approved artifacts to a staging repository.

> [!WARNING]
> This project is active infrastructure. Its lockfile currently pins unreleased
> `main` revisions of ssm-simulators and LANfactory, and the committed Oscar
> defaults describe one lab allocation. Rehearse locally and inspect generated
> scripts before scheduling production-scale work.

## Documentation

The [LAN pipeline documentation](https://lnccbrown.github.io/LAN_pipeline_minimal/)
is the canonical source for:

- the [CI-backed local tutorial](https://lnccbrown.github.io/LAN_pipeline_minimal/learn/local-workflow/);
- [Slurm and cluster operation](https://lnccbrown.github.io/LAN_pipeline_minimal/how-to/);
- [validation and safe staging](https://lnccbrown.github.io/LAN_pipeline_minimal/how-to/validate-network/);
- [architecture and lineage](https://lnccbrown.github.io/LAN_pipeline_minimal/explanations/);
- [CLI, configuration, environment, and JSON contracts](https://lnccbrown.github.io/LAN_pipeline_minimal/reference/).

README files in this repository are concise entry points. Keep durable operator
and interface guidance on the documentation site.

## Quick local check

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. ssm-simulators
also needs GSL and a working compiler.

```bash
git clone https://github.com/lnccbrown/LAN_pipeline_minimal.git
cd LAN_pipeline_minimal
uv sync --locked
bash local_test_run.sh
```

The script uses the tracked quick-test configs to generate a small DDM dataset,
train a small JAX LAN, and record both stages in local MLflow. It removes and
recreates `local_test_data/`, so do not store personal files there.

## Main entry points

```bash
uv run lan-sbatch generate --help
uv run lan-sbatch jaxtrain --help
uv run lan-sbatch torchtrain --help
uv run lan-publish --help
```

To build or preview the documentation without installing the CUDA/git-sourced
project dependency stack:

```bash
./scripts/docs.sh build
./scripts/docs.sh serve
```

## Development checks

```bash
uv sync --locked --group dev
uv run pytest tests/ -q
uv run ruff check .
```

LAN_pipeline_minimal is distributed under the terms in [LICENSE](LICENSE).
