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

It also owns `derived_<model>/network_training_{cpn,opn}.yaml`, which have no
generation pair and are checked against derive-aux's fixed rows-per-file
instead.
"""

import re
from pathlib import Path

import pytest
import yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CONFIGURATION_REFERENCE = (
    Path(__file__).resolve().parents[1] / "docs/reference/configuration.md"
)
PRODUCTION = sorted(p for p in CONFIGS.glob("production_*") if p.is_dir())
PRODUCTION_REFERENCE_ROW = re.compile(
    r"^\| `configs/(?P<directory>production_[^`/]+)/` \| `(?P<model>[^`]+)` \|$",
    re.MULTILINE,
)


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


def test_every_production_pair_is_named_in_the_configuration_reference():
    """Keep the rendered operational reference exact, current, and unambiguous."""
    reference = CONFIGURATION_REFERENCE.read_text()
    expected = {
        (directory.name, _load(directory, "data_generation.yaml")["MODEL"])
        for directory in PRODUCTION
    }
    documented = {
        (match["directory"], match["model"])
        for match in PRODUCTION_REFERENCE_ROW.finditer(reference)
    }
    assert documented == expected, (
        f"documented production configs differ: {documented ^ expected}"
    )


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


# ---------------------------------------------------------------------------
# derived_<model>/ — training configs for auxiliary networks whose corpus is
# integrated from a LAN by LANfactory's derive-aux, not simulated. No
# data_generation.yaml exists to pair them with, which is why the prefix is
# not `production_`: the pairing fixture above would fail to find one.
# ---------------------------------------------------------------------------

DERIVED = sorted(CONFIGS.glob("derived_*/network_training_*.yaml"))
# derive-aux's default θ per file. A cpn file holds one row per (θ, choice);
# an opn file one row per θ.
DERIVED_THETA_PER_FILE = 4096


def test_derived_configs_are_not_paired_as_production():
    assert not any(p.name.startswith("derived_") for p in PRODUCTION)
    assert DERIVED, f"no derived_*/network_training_*.yaml under {CONFIGS}"


@pytest.fixture(params=DERIVED, ids=lambda p: f"{p.parent.name}/{p.name}")
def derived_config(request):
    path = request.param
    return path, yaml.safe_load(path.read_text())


class TestDerivedConfigs:
    def test_the_directory_and_file_names_say_what_the_config_trains(
        self, derived_config
    ):
        path, training = derived_config
        assert training["MODEL"] == path.parent.name[len("derived_") :]
        network_type = path.stem[len("network_training_") :]
        assert training["NETWORK_TYPE"] == network_type
        assert network_type in ("cpn", "opn")

    def test_the_batch_divides_a_derived_file_exactly(self, derived_config):
        # DatasetTorch raises at load on any remainder, and a derived corpus
        # has a fixed row count per file: 4096 θ, times the number of choices
        # for a cpn (one row per choice code).
        from ssms.config import model_config

        _, training = derived_config
        n_choices = len(model_config[training["MODEL"]]["choices"])
        rows = DERIVED_THETA_PER_FILE * (
            n_choices if training["NETWORK_TYPE"] == "cpn" else 1
        )
        assert rows % training["GPU_BATCH_SIZE"] == 0, (
            rows,
            training["GPU_BATCH_SIZE"],
        )
        assert rows % training["CPU_BATCH_SIZE"] == 0, (
            rows,
            training["CPU_BATCH_SIZE"],
        )
        assert training["CPU_BATCH_SIZE"] == training["GPU_BATCH_SIZE"]

    def test_the_loader_contract_holds(self, derived_config):
        _, training = derived_config
        assert training["SHUFFLE"] is False
        assert isinstance(training["LABELS_LOWER_BOUND"], str)
        for layers, acts in zip(training["LAYER_SIZES"], training["ACTIVATIONS"]):
            assert len(acts) == len(layers) - 1 and layers[-1] == 1

    def test_the_training_folder_is_a_placeholder(self, derived_config):
        # A derive-aux output is a local path; committing one would pin a
        # laptop. --training-data-folder supplies it at submission.
        _, training = derived_config
        assert training["TRAINING_DATA_FOLDER"] == ""
