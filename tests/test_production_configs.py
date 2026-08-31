"""Every committed `production_<model>/` pair has to be internally coherent.

Recording production runs as committed configs is only worth the convention if
the recording is checked. It is not hypothetical: backfilling
`production_ddm_sdv/` from the cluster caught two values that had been copied
from `configs/examples/` and never corrected -- `SHUFFLE: True` (wrong for a
loader that advances files on batch index) and `CPU_BATCH_SIZE: 1000` (a
fiftieth of the GPU batch, so a CPU fallback would not have been the same
problem). Neither would have failed loudly; both would have quietly made the
next run un-reproducible from the repo.

These are cheap structural checks, not a schema. The pipeline's own loaders own
the schema; this owns the things that are only wrong in context.
"""

from pathlib import Path

import pytest
import yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PRODUCTION = sorted(p for p in CONFIGS.glob("production_*") if p.is_dir())


def _load(directory: Path, name: str) -> dict:
    return yaml.safe_load((directory / name).read_text())


@pytest.fixture(params=PRODUCTION, ids=lambda p: p.name)
def run_config(request):
    directory = request.param
    return (
        directory,
        _load(directory, "data_generation.yaml"),
        _load(directory, "network_training.yaml"),
    )


def test_at_least_one_production_run_is_recorded():
    # Guards the glob itself: a rename that emptied it would turn every test
    # below into a silent no-op.
    assert PRODUCTION, f"no production_* config directories under {CONFIGS}"


class TestIdentity:
    def test_the_directory_name_is_the_model_name(self, run_config):
        directory, generation, training = run_config
        model = directory.name[len("production_") :]
        assert generation["MODEL"] == model
        assert training["MODEL"] == model

    def test_the_model_is_one_the_simulator_knows(self, run_config):
        # A typo here survives review and dies three hours into an array job.
        from ssms.config import model_config

        _, generation, _ = run_config
        assert generation["MODEL"] in model_config


class TestArchitectureGrid:
    """`--network-id` indexes both lists, so they have to stay parallel."""

    def test_every_architecture_has_matching_activations(self, run_config):
        _, _, training = run_config
        sizes, activations = training["LAYER_SIZES"], training["ACTIVATIONS"]
        assert len(sizes) == len(activations)
        for layers, acts in zip(sizes, activations):
            # One activation per hidden layer: the final size-1 layer is the
            # log-density output and carries none.
            assert len(acts) == len(layers) - 1, (layers, acts)

    def test_every_architecture_ends_in_a_scalar_output(self, run_config):
        _, _, training = run_config
        for layers in training["LAYER_SIZES"]:
            assert layers[-1] == 1, layers


class TestBatchSize:
    def test_the_gpu_batch_divides_a_training_file_exactly(self, run_config):
        # A remainder batch is a short batch, and the loader has no path for
        # one: rows per file is N_PARAMETER_SETS x N_SAMPLES_PER_PARAM.
        _, generation, training = run_config
        rows = (
            generation["PIPELINE"]["N_PARAMETER_SETS"]
            * (generation["TRAINING"]["N_SAMPLES_PER_PARAM"])
        )
        assert rows % training["GPU_BATCH_SIZE"] == 0, (
            rows,
            training["GPU_BATCH_SIZE"],
        )

    def test_the_cpu_batch_matches_the_gpu_batch(self, run_config):
        # So a CPU fallback runs the same problem rather than a different one.
        _, _, training = run_config
        assert training["CPU_BATCH_SIZE"] == training["GPU_BATCH_SIZE"]


class TestLoaderContract:
    def test_shuffle_is_off(self, run_config):
        # DatasetTorch advances files on batch index, so a shuffled index
        # stream reads rows from whichever file happens to be loaded. Rows are
        # already bootstrap-resampled on every file load.
        _, _, training = run_config
        assert training["SHUFFLE"] is False

    def test_the_label_floor_stays_a_quoted_expression(self, run_config):
        # It is eval'd with numpy in scope; a bare numeric literal raises
        # TypeError at load, deep inside a submitted job.
        _, _, training = run_config
        assert isinstance(training["LABELS_LOWER_BOUND"], str)
