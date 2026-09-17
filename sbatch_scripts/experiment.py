"""An *experiment* is a directory: one `experiment.yaml` plus the per-stage
configs it points at, and a `state.json` the stages fill in as they run.

`lan-sbatch init` scaffolds one; `lan-sbatch generate/jaxtrain/torchtrain/recover
--experiment DIR` read it so the model, lineage id, experiment names and config
paths are typed once. The lineage id is the MLflow schema-v2 join key
(HSSMSpine `_docs/mlflow-schema.md`): minted here, passed to every stage, and
carried by ssm-simulators into each training pickle, by LANfactory into the
network's config pickles, and by `hssm.track` onto every fit.

Nothing in this module talks to MLflow or SLURM. It is plain data + file I/O so
`--script-only` rendering stays side-effect-free and unit tests need no store.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SCHEMA_VERSION = 1
EXPERIMENT_FILE = "experiment.yaml"
STATE_FILE = "state.json"

# Mirror of the schema's reserved tags: users may add MLflow tags to an
# experiment, but not the ones the emitters own.
RESERVED_MLFLOW_TAGS = frozenset({"schema_version", "phase", "lineage_id"})

NETWORK_TYPES = ("lan", "cpn", "opn", "gonogo")
TRAINERS = ("jaxtrain", "torchtrain")

# `{model}` is interpolated. These are the pipeline's historical names; the
# ecosystem schema doc follows them.
DEFAULT_EXPERIMENT_NAMES = {
    "data_generation": "{model}-data-generation",
    "training": "{model}-training",
    "inference": "{model}-inference",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIRS = {
    "examples": REPO_ROOT / "configs" / "examples",
    "quick_test": REPO_ROOT / "configs" / "quick_test",
}

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ExperimentError(ValueError):
    """A malformed or inconsistent experiment directory."""


@dataclass
class NetworkSpec:
    type: str = "lan"
    trainer: str = "jaxtrain"
    network_id: int = 0


@dataclass
class MlflowSpec:
    experiments: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_EXPERIMENT_NAMES)
    )
    tracking_uri: str | None = None
    artifact_location: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class GenerateStage:
    config: str = "data_generation.yaml"
    output: str = "data"
    n_jobs_in_array: int = 1
    n_files: int | None = None


@dataclass
class TrainStage:
    config: str = "network_training.yaml"
    output: str = "networks"
    dl_workers: int = 1


@dataclass
class ValidateStage:
    skip_density: bool = False
    skip_hssm: bool = False


@dataclass
class RecoverStage:
    designs: list[str] = field(default_factory=lambda: ["L0_n250"])
    likelihoods: list[str] = field(default_factory=lambda: ["approx_differentiable"])
    add_reference_arm: bool = True
    n_datasets: int = 1
    draws: int = 200
    tune: int = 200
    chains: int = 2
    target_accept: float = 0.9
    p_outlier: float | None = None
    condition_param: str | None = None
    out_dir: str = "recovery"


@dataclass
class Stages:
    generate: GenerateStage = field(default_factory=GenerateStage)
    train: TrainStage = field(default_factory=TrainStage)
    validate: ValidateStage = field(default_factory=ValidateStage)
    recover: RecoverStage = field(default_factory=RecoverStage)


@dataclass
class Experiment:
    dir: Path
    name: str
    model: str
    lineage_id: str | None = None
    network: NetworkSpec = field(default_factory=NetworkSpec)
    mlflow: MlflowSpec = field(default_factory=MlflowSpec)
    stages: Stages = field(default_factory=Stages)

    # -- derived -----------------------------------------------------------

    @property
    def path(self) -> Path:
        return self.dir / EXPERIMENT_FILE

    def experiment_names(self) -> dict[str, str]:
        """MLflow experiment name per phase, `{model}` interpolated."""
        names = dict(DEFAULT_EXPERIMENT_NAMES)
        names.update(self.mlflow.experiments or {})
        return {k: v.format(model=self.model) for k, v in names.items()}

    def stage_config(self, stage: str) -> Path:
        rel = getattr(self.stages, stage).config
        return (self.dir / rel).resolve()

    def stage_output(self, stage: str) -> Path:
        rel = getattr(self.stages, stage).output
        return (self.dir / rel).resolve()

    def recovery_dir(self) -> Path:
        return (self.dir / self.stages.recover.out_dir).resolve()

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "model": self.model,
            "lineage_id": self.lineage_id,
            "network": _asdict(self.network),
            "mlflow": _asdict(self.mlflow),
            "stages": {
                "generate": _asdict(self.stages.generate),
                "train": _asdict(self.stages.train),
                "validate": _asdict(self.stages.validate),
                "recover": _asdict(self.stages.recover),
            },
        }

    def save(self) -> None:
        self.path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)
        )


def _asdict(obj) -> dict:
    return {k: getattr(obj, k) for k in obj.__dataclass_fields__}


def _build(cls, data: dict | None, where: str):
    """Construct a dataclass from a mapping, rejecting unknown keys."""
    data = data or {}
    if not isinstance(data, dict):
        raise ExperimentError(f"{where} must be a mapping, got {type(data).__name__}")
    unknown = set(data) - set(cls.__dataclass_fields__)
    if unknown:
        raise ExperimentError(f"{where} has unknown keys: {sorted(unknown)}")
    return cls(**data)


# --------------------------------------------------------------------------- #
# Load / validate
# --------------------------------------------------------------------------- #


def experiment_dir(path: Path) -> Path:
    """Accept either the directory or its experiment.yaml."""
    path = Path(path)
    return path.parent if path.name == EXPERIMENT_FILE else path


def load_experiment(path: Path) -> Experiment:
    """Read and validate an experiment directory.

    Raises ExperimentError with a message fit for the CLI boundary.
    """
    directory = experiment_dir(path).resolve()
    file = directory / EXPERIMENT_FILE
    if not file.is_file():
        raise ExperimentError(
            f"{file} not found. Create an experiment with `lan-sbatch init <name>`."
        )
    with open(file, "rb") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ExperimentError(f"{file} must contain a mapping at the top level")

    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ExperimentError(
            f"{file}: schema_version {version!r} is not supported "
            f"(this pipeline reads {SCHEMA_VERSION})"
        )

    model = raw.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ExperimentError(f"{file}: 'model' must be a non-empty string")
    name = raw.get("name") or directory.name
    if not _NAME_RE.match(str(name)):
        raise ExperimentError(
            f"{file}: 'name' {name!r} may only contain letters, digits, '.', '_', '-'"
        )

    lineage_id = raw.get("lineage_id")
    if lineage_id is not None and (
        not isinstance(lineage_id, str) or not lineage_id.strip()
    ):
        raise ExperimentError(f"{file}: 'lineage_id' must be a non-empty string")

    stages_raw = raw.get("stages") or {}
    if not isinstance(stages_raw, dict):
        raise ExperimentError(f"{file}: 'stages' must be a mapping")
    unknown_stages = set(stages_raw) - {"generate", "train", "validate", "recover"}
    if unknown_stages:
        raise ExperimentError(f"{file}: unknown stages {sorted(unknown_stages)}")

    exp = Experiment(
        dir=directory,
        name=str(name),
        model=model.strip(),
        lineage_id=lineage_id.strip() if lineage_id else None,
        network=_build(NetworkSpec, raw.get("network"), "network"),
        mlflow=_build(MlflowSpec, raw.get("mlflow"), "mlflow"),
        stages=Stages(
            generate=_build(
                GenerateStage, stages_raw.get("generate"), "stages.generate"
            ),
            train=_build(TrainStage, stages_raw.get("train"), "stages.train"),
            validate=_build(
                ValidateStage, stages_raw.get("validate"), "stages.validate"
            ),
            recover=_build(RecoverStage, stages_raw.get("recover"), "stages.recover"),
        ),
    )
    _validate(exp)
    return exp


def _validate(exp: Experiment) -> None:
    where = exp.path
    if exp.network.type not in NETWORK_TYPES:
        raise ExperimentError(
            f"{where}: network.type {exp.network.type!r} not in {list(NETWORK_TYPES)}"
        )
    if exp.network.trainer not in TRAINERS:
        raise ExperimentError(
            f"{where}: network.trainer {exp.network.trainer!r} not in {list(TRAINERS)}"
        )

    reserved = RESERVED_MLFLOW_TAGS & set(exp.mlflow.tags or {})
    if reserved:
        raise ExperimentError(
            f"{where}: mlflow.tags may not set reserved schema tags {sorted(reserved)}"
        )
    for key, value in (exp.mlflow.tags or {}).items():
        if not isinstance(key, str) or not key or not isinstance(value, str):
            raise ExperimentError(
                f"{where}: mlflow.tags must map non-empty string keys to strings"
            )
    for phase, template in (exp.mlflow.experiments or {}).items():
        if phase not in DEFAULT_EXPERIMENT_NAMES:
            raise ExperimentError(
                f"{where}: mlflow.experiments has unknown phase {phase!r}; "
                f"expected {sorted(DEFAULT_EXPERIMENT_NAMES)}"
            )
        if not isinstance(template, str) or not template.strip():
            raise ExperimentError(
                f"{where}: mlflow.experiments.{phase} must be a string"
            )

    for stage in ("generate", "train"):
        config = exp.stage_config(stage)
        if not config.is_file():
            raise ExperimentError(
                f"{where}: stages.{stage}.config {config} does not exist"
            )
        cfg_model = _read_yaml(config).get("MODEL")
        if cfg_model is not None and str(cfg_model) != exp.model:
            raise ExperimentError(
                f"{config}: MODEL {cfg_model!r} disagrees with experiment model "
                f"{exp.model!r}"
            )
    train_cfg = _read_yaml(exp.stage_config("train"))
    cfg_type = train_cfg.get("NETWORK_TYPE")
    if cfg_type is not None and str(cfg_type) != exp.network.type:
        raise ExperimentError(
            f"{exp.stage_config('train')}: NETWORK_TYPE {cfg_type!r} disagrees with "
            f"network.type {exp.network.type!r}"
        )

    rec = exp.stages.recover
    if rec.n_datasets < 1:
        raise ExperimentError(f"{where}: stages.recover.n_datasets must be >= 1")
    if not rec.designs:
        raise ExperimentError(f"{where}: stages.recover.designs must not be empty")
    known = _known_designs()
    if known is not None:
        bad = [d for d in rec.designs if d not in known]
        if bad:
            raise ExperimentError(
                f"{where}: unknown recovery designs {bad}; have {sorted(known)}"
            )


def _known_designs() -> set[str] | None:
    """Design names from the recovery harness; None if it cannot be imported.

    Lazy so loading an experiment never needs the validate dependency group.
    """
    try:
        import sys

        sys.path.insert(0, str(REPO_ROOT / "validation"))
        import recovery_designs as rd  # type: ignore

        return set(rd.DESIGNS)
    except Exception:  # noqa: BLE001 - validation is optional at load time
        return None


def _read_yaml(path: Path) -> dict:
    with open(path, "rb") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


# --------------------------------------------------------------------------- #
# Lineage
# --------------------------------------------------------------------------- #


def mint_lineage_id() -> str:
    return uuid.uuid4().hex


def ensure_lineage_id(exp: Experiment) -> str:
    """Return the experiment's lineage id, minting and persisting one if absent.

    Every stage of one experiment must share the id, so it is written back to
    `experiment.yaml` the first time it is needed. Callers that must stay
    side-effect-free (`--script-only`) should read `exp.lineage_id` instead and
    error when it is None.
    """
    if exp.lineage_id:
        return exp.lineage_id
    exp.lineage_id = mint_lineage_id()
    exp.save()
    return exp.lineage_id


# --------------------------------------------------------------------------- #
# Scaffold
# --------------------------------------------------------------------------- #


def scaffold_experiment(
    name: str,
    root: Path,
    *,
    model: str = "ddm",
    quick: bool = False,
    network_type: str = "lan",
    trainer: str = "jaxtrain",
    force: bool = False,
) -> Experiment:
    """Create `<root>/<name>/` with experiment.yaml and both stage configs.

    Templates come from configs/quick_test (``quick``) or configs/examples;
    their MODEL (and the training config's NETWORK_TYPE) are rewritten to match
    the experiment so the two never disagree.
    """
    if not _NAME_RE.match(name):
        raise ExperimentError(
            f"experiment name {name!r} may only contain letters, digits, '.', '_', '-'"
        )
    if network_type not in NETWORK_TYPES:
        raise ExperimentError(
            f"network_type {network_type!r} not in {list(NETWORK_TYPES)}"
        )
    if trainer not in TRAINERS:
        raise ExperimentError(f"trainer {trainer!r} not in {list(TRAINERS)}")

    directory = (Path(root) / name).resolve()
    if directory.exists() and any(directory.iterdir()) and not force:
        raise ExperimentError(
            f"{directory} already exists and is not empty (use --force)"
        )
    directory.mkdir(parents=True, exist_ok=True)

    templates = TEMPLATE_DIRS["quick_test" if quick else "examples"]
    gen_src = templates / "data_generation.yaml"
    if quick:
        train_src = templates / "network_training.yaml"
    else:
        train_src = templates / f"network_training_{network_type}.yaml"
        if not train_src.is_file():
            # Only lan/cpn ship as examples; fall back to lan and rewrite the type.
            train_src = templates / "network_training_lan.yaml"

    gen_dst = directory / "data_generation.yaml"
    train_dst = directory / "network_training.yaml"
    shutil.copyfile(gen_src, gen_dst)
    shutil.copyfile(train_src, train_dst)
    _rewrite_top_level_keys(gen_dst, {"MODEL": model})
    _rewrite_top_level_keys(train_dst, {"MODEL": model, "NETWORK_TYPE": network_type})

    exp = Experiment(
        dir=directory,
        name=name,
        model=model,
        lineage_id=mint_lineage_id(),
        network=NetworkSpec(type=network_type, trainer=trainer),
    )
    if quick:
        # A minutes-long network cannot pass the density gate; a quick
        # experiment exists to prove the chain, not the network.
        exp.stages.validate.skip_density = True
    exp.save()
    return exp


def _rewrite_top_level_keys(path: Path, updates: dict[str, str]) -> None:
    """Replace `KEY: value` lines in place, keeping the template's comments.

    Round-tripping through yaml.safe_dump would drop every comment in the
    template, and those comments are the documentation a new user reads.
    """
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    seen: set[str] = set()
    for i, line in enumerate(lines):
        for key, value in updates.items():
            if re.match(rf"^{re.escape(key)}\s*:", line):
                comment = ""
                if "#" in line.split(":", 1)[1]:
                    comment = "  #" + line.split("#", 1)[1].rstrip("\n")
                lines[i] = f'{key}: "{value}"{comment}\n'
                seen.add(key)
    for key, value in updates.items():
        if key not in seen:
            lines.append(f'{key}: "{value}"\n')
    path.write_text("".join(lines))


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


def read_state(exp: Experiment) -> dict:
    path = exp.dir / STATE_FILE
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text()) or {}
    except json.JSONDecodeError as e:
        raise ExperimentError(f"{path} is not valid JSON: {e}") from e


def write_state(exp: Experiment, **updates) -> dict:
    """Merge `updates` into state.json (None values are dropped) and return it."""
    state = read_state(exp)
    state.update({k: v for k, v in updates.items() if v is not None})
    (exp.dir / STATE_FILE).write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n"
    )
    return state


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #

# Both trainer filename conventions embed the run_uuid next to the network
# type; see LANfactory hf/upload.py `_ARTIFACT_NAME_PATTERNS`.
_RUN_UUID_PATTERNS = (
    re.compile(r"^(?P<run_uuid>[0-9a-f]+)_(?:lan|cpn|opn|gonogo)_.+?__"),  # jax
    re.compile(r"^.+?_(?:lan|cpn|opn|gonogo)_(?P<run_uuid>[0-9a-f]+)_"),  # torch
)


def run_uuid_from_filename(filename: str) -> str | None:
    for pattern in _RUN_UUID_PATTERNS:
        if match := pattern.match(filename):
            return match.group("run_uuid")
    return None


def find_onnx(exp: Experiment) -> Path | None:
    """Newest ONNX the training stage wrote for this experiment, or None."""
    folder = exp.stage_output("train") / exp.network.type / exp.model
    candidates = sorted(folder.glob("*.onnx"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def mlflow_tracking_uri(exp: Experiment, env: dict | None = None) -> str:
    """experiment.yaml > $MLFLOW_TRACKING_URI > a sqlite file inside the experiment."""
    import os

    env = os.environ if env is None else env
    if exp.mlflow.tracking_uri:
        return exp.mlflow.tracking_uri
    if env.get("MLFLOW_TRACKING_URI"):
        return env["MLFLOW_TRACKING_URI"]
    return f"sqlite:///{exp.dir / 'mlflow.db'}"
