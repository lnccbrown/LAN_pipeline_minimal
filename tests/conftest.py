"""Shared fixtures for the gen_sbatch test suite.

sbatch_scripts/ is not a package (the repo becomes an installable tool in a
later PR), so the module is imported by path.
"""

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "sbatch_scripts"))
sys.path.insert(0, str(REPO_ROOT / "validation"))


@pytest.fixture
def model_config(tmp_path):
    """Minimal pipeline YAML config (the MODEL key is what gen_sbatch reads)."""
    path = tmp_path / "config.yaml"
    path.write_text("MODEL: ddm\n")
    return path


@pytest.fixture
def cluster_config(tmp_path):
    """A cluster inventory file exercising job_defaults + modules."""
    path = tmp_path / "oscar.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "condos": {
                    "test-condo": {
                        "partitions": ["batch", "gpu"],
                        "suited_for": ["generate"],
                        "verified": False,
                    }
                },
                "job_defaults": {
                    "generate": {
                        "account": "test-condo",
                        "partition": "batch",
                        "cores": 4,
                        "mem": "8G",
                        "num_gpus": 0,
                        "time": "02:00:00",
                    },
                    "jaxtrain": {
                        "account": "test-condo",
                        "partition": "gpu",
                        "cores": 2,
                        "mem": "32G",
                        "num_gpus": 1,
                        "time": "08:00:00",
                    },
                },
                "modules": ["python", "gcc", "cuda"],
            }
        )
    )
    return path


@pytest.fixture
def isolated_mlflow(tmp_path, monkeypatch):
    """Point MLflow at a throwaway sqlite DB; return its path."""
    db = tmp_path / "mlflow_test.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db}")
    monkeypatch.delenv("MLFLOW_ARTIFACT_LOCATION", raising=False)
    return db


@pytest.fixture
def fake_sbatch_ok(monkeypatch):
    """subprocess.run stub: sbatch succeeds with job id 12345; records calls."""
    import gen_sbatch

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)

        class Result:
            returncode = 0
            stdout = "Submitted batch job 12345\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(gen_sbatch.subprocess, "run", fake_run)
    return calls


@pytest.fixture
def fake_sbatch_fail(monkeypatch):
    """subprocess.run stub: sbatch fails (bad account)."""
    import gen_sbatch

    def fake_run(cmd, **kwargs):
        class Result:
            returncode = 1
            stdout = ""
            stderr = "sbatch: error: Invalid account or account/partition combination specified\n"

        return Result()

    monkeypatch.setattr(gen_sbatch.subprocess, "run", fake_run)
