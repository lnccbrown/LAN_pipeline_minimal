"""Tests for gen_sbatch: quoting, job-id capture, the JSON contract,
--script-only purity, and cluster-config resource resolution.

The JSON line on stdout is the laptop driver's API — its shape is the
load-bearing contract here.
"""

import json
import logging
import shlex
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

import gen_sbatch
from gen_sbatch import (
    absolutize_tracking_uri,
    app,
    load_cluster_config,
    split_across_lanes,
    create_command,
    quote_param_value,
    resolve_resources,
    submit_sbatch,
)

runner = CliRunner()

JSON_KEYS = {
    "command",
    "job_id",
    "mlflow_experiment_id",
    "mlflow_run_id",
    "sbatch_script",
    "output_path",
    "account",
    "partition",
    # Added with lane fan-out. Additive: a single-lane run still emits exactly
    # one line, with lane=0 and n_lanes=1, so existing consumers are unaffected.
    "array_size",
    "lane",
    "n_lanes",
}


def last_json_line(output: str) -> dict:
    lines = [ln for ln in output.strip().splitlines() if ln.startswith("{")]
    assert lines, f"no JSON line on stdout: {output!r}"
    return json.loads(lines[-1])


class TestQuoting:
    def test_paths_with_spaces_are_quoted(self):
        cmd = create_command(
            "generate", **{"config-path": "/tmp/my configs/config.yaml"}
        )
        assert "--config-path '/tmp/my configs/config.yaml'" in cmd

    def test_shell_metacharacters_are_neutralized(self):
        cmd = create_command("generate", **{"output": "/tmp/$(rm -rf x); echo"})
        # single-quoted → the shell sees one literal argument, nothing executes
        assert "'/tmp/$(rm -rf x); echo'" in cmd
        assert shlex.split(cmd) == [
            "generate",
            "--output",
            "/tmp/$(rm -rf x); echo",
        ]

    def test_mlflow_run_name_keeps_slurm_var_expandable(self):
        # SLURM substitutes $SLURM_ARRAY_TASK_ID inside the job; single-quoting
        # would freeze the literal string and every worker would share a name.
        cmd = create_command(
            "generate", **{"mlflow-run-name": "ddm-worker-$SLURM_ARRAY_TASK_ID"}
        )
        assert '--mlflow-run-name ddm-worker-"$SLURM_ARRAY_TASK_ID"' in cmd

    def test_expandable_value_quotes_literal_part(self):
        # The SLURM var is isolated in its own double-quoted segment; the
        # literal part goes through shlex.quote, which adds nothing when the
        # text is already shell-safe (as an ordinary model name is).
        assert (
            quote_param_value("mlflow-run-name", "ddm-worker-$SLURM_ARRAY_TASK_ID")
            == 'ddm-worker-"$SLURM_ARRAY_TASK_ID"'
        )
        # ...and quotes it when it is not
        assert (
            quote_param_value("mlflow-run-name", "ddm x-worker-$SLURM_ARRAY_TASK_ID")
            == "'ddm x-worker-'\"$SLURM_ARRAY_TASK_ID\""
        )

    @pytest.mark.parametrize(
        "hostile",
        [
            "ddm`id`",
            "ddm$(id)",
            "ddm; id",
            'ddm"; id; "',
            "ddm$HOME",
            "ddm\\`id\\`",
        ],
    )
    def test_command_substitution_in_expandable_value_is_inert(self, hostile):
        """The model name comes from a user-supplied YAML and reaches the run
        name; nothing in it may execute on the compute node."""
        import subprocess

        rendered = quote_param_value(
            "mlflow-run-name", f"{hostile}-worker-$SLURM_ARRAY_TASK_ID"
        )
        # Ask a real bash what argument the job would actually receive.
        out = subprocess.run(
            ["bash", "-c", f"printf '%s' {rendered}"],
            capture_output=True,
            text=True,
            env={"SLURM_ARRAY_TASK_ID": "7", "PATH": "/usr/bin:/bin"},
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout == f"{hostile}-worker-7", out.stdout

    def test_dollar_in_non_expandable_param_stays_literal(self):
        import subprocess

        rendered = quote_param_value("output", "/data/$USER/out")
        out = subprocess.run(
            ["bash", "-c", f"printf '%s' {rendered}"],
            capture_output=True,
            text=True,
            env={"USER": "someone", "PATH": "/usr/bin:/bin"},
        )
        assert out.stdout == "/data/$USER/out"


class TestSubmitSbatch:
    def test_parses_job_id(self, fake_sbatch_ok, tmp_path):
        job_id = submit_sbatch(tmp_path / "job.sh", logging.getLogger("t"))
        assert job_id == 12345

    def test_failure_returns_none(self, fake_sbatch_fail, tmp_path):
        assert submit_sbatch(tmp_path / "job.sh", logging.getLogger("t")) is None

    def test_missing_binary_returns_none(self, monkeypatch, tmp_path):
        def raise_oserror(cmd, **kwargs):
            raise FileNotFoundError("sbatch")

        monkeypatch.setattr(gen_sbatch.subprocess, "run", raise_oserror)
        assert submit_sbatch(tmp_path / "job.sh", logging.getLogger("t")) is None

    def test_unparseable_stdout_returns_none(self, monkeypatch, tmp_path):
        def fake_run(cmd, **kwargs):
            class Result:
                returncode = 0
                stdout = "something unexpected\n"
                stderr = ""

            return Result()

        monkeypatch.setattr(gen_sbatch.subprocess, "run", fake_run)
        assert submit_sbatch(tmp_path / "job.sh", logging.getLogger("t")) is None


class TestJsonContract:
    def test_submit_emits_json_with_job_id(
        self, model_config, tmp_path, fake_sbatch_ok, isolated_mlflow
    ):
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            ["generate", "--config-path", str(model_config), "--output-path", str(out)],
        )
        assert result.exit_code == 0, result.output
        record = last_json_line(result.output)
        assert set(record) == JSON_KEYS
        assert record["job_id"] == 12345
        assert record["output_path"] == str(out.resolve())
        assert record["sbatch_script"].endswith(".sh")
        # generate creates a per-model experiment; its id is the chaining key
        assert record["mlflow_experiment_id"] is not None

    def test_json_emitted_even_at_default_log_level(
        self, model_config, tmp_path, fake_sbatch_ok, isolated_mlflow
    ):
        # No --log-level: WARNING suppresses all info logging, the JSON line
        # must still be there (this was the suppressed-experiment-id bug).
        result = runner.invoke(
            app,
            [
                "generate",
                "--config-path",
                str(model_config),
                "--output-path",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 0
        assert last_json_line(result.output)["mlflow_experiment_id"] is not None

    def test_submit_failure_exits_nonzero_with_json(
        self, model_config, tmp_path, fake_sbatch_fail, isolated_mlflow
    ):
        result = runner.invoke(
            app,
            [
                "generate",
                "--config-path",
                str(model_config),
                "--output-path",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 1
        assert last_json_line(result.output)["job_id"] is None


class TestScriptOnly:
    def invoke_script_only(self, model_config, out, extra=()):
        return runner.invoke(
            app,
            [
                "generate",
                "--config-path",
                str(model_config),
                "--output-path",
                str(out),
                "--script-only",
                *extra,
            ],
        )

    def test_writes_script_under_output_runs(self, model_config, tmp_path):
        out = tmp_path / "out"
        result = self.invoke_script_only(model_config, out)
        assert result.exit_code == 0, result.output
        record = last_json_line(result.output)
        scripts = list((out / "runs").glob("*.sh"))
        assert len(scripts) == 1
        assert record["sbatch_script"] == str(scripts[0])
        assert record["job_id"] is None

    def test_repeated_invocations_do_not_overwrite(self, model_config, tmp_path):
        out = tmp_path / "out"
        assert self.invoke_script_only(model_config, out).exit_code == 0
        assert self.invoke_script_only(model_config, out).exit_code == 0
        assert len(list((out / "runs").glob("*.sh"))) == 2

    def test_no_mlflow_side_effects(self, model_config, tmp_path, isolated_mlflow):
        # Previously each --script-only train call left an empty orphan run.
        result = self.invoke_script_only(model_config, tmp_path / "out")
        assert result.exit_code == 0
        assert not isolated_mlflow.exists(), "script-only must not touch MLflow"

    def test_train_script_only_no_orphan_run(
        self, model_config, tmp_path, isolated_mlflow
    ):
        result = runner.invoke(
            app,
            [
                "jaxtrain",
                "--config-path",
                str(model_config),
                "--output-path",
                str(tmp_path / "out"),
                "--training-data-folder",
                str(tmp_path),
                "--script-only",
            ],
        )
        assert result.exit_code == 0, result.output
        assert not isolated_mlflow.exists()
        assert last_json_line(result.output)["mlflow_run_id"] is None


class TestResourceResolution:
    def test_builtin_fallbacks_without_config(self):
        resources = resolve_resources(
            "generate",
            None,
            dict.fromkeys(("account", "partition", "num_gpus", "cores", "mem", "time")),
        )
        assert resources["account"] == "default"
        assert resources["partition"] == "batch"
        assert resources["modules"] == ["python", "gcc"]

    def test_cluster_config_supplies_job_kind_defaults(
        self, model_config, cluster_config, tmp_path
    ):
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "generate",
                "--config-path",
                str(model_config),
                "--output-path",
                str(out),
                "--cluster-config",
                str(cluster_config),
                "--script-only",
            ],
        )
        assert result.exit_code == 0, result.output
        record = last_json_line(result.output)
        assert record["account"] == "test-condo"
        assert record["partition"] == "batch"
        script = (out / "runs").glob("*.sh").__next__().read_text()
        assert "#SBATCH --account=test-condo" in script
        assert "#SBATCH -c 4" in script
        assert "#SBATCH --mem=8G" in script
        assert "#SBATCH --time=02:00:00" in script
        assert "module load cuda" in script

    def test_job_kind_selects_its_own_defaults(
        self, model_config, cluster_config, tmp_path
    ):
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "jaxtrain",
                "--config-path",
                str(model_config),
                "--output-path",
                str(out),
                "--training-data-folder",
                str(tmp_path),
                "--cluster-config",
                str(cluster_config),
                "--script-only",
            ],
        )
        assert result.exit_code == 0, result.output
        script = (out / "runs").glob("*.sh").__next__().read_text()
        assert "#SBATCH -p gpu --gres=gpu:1" in script
        assert record_partition(result) == "gpu"

    def test_explicit_flag_overrides_cluster_config(
        self, model_config, cluster_config, tmp_path
    ):
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "generate",
                "--config-path",
                str(model_config),
                "--output-path",
                str(out),
                "--cluster-config",
                str(cluster_config),
                "--account",
                "my-other-condo",
                "--cores",
                "16",
                "--script-only",
            ],
        )
        assert result.exit_code == 0, result.output
        assert last_json_line(result.output)["account"] == "my-other-condo"
        script = (out / "runs").glob("*.sh").__next__().read_text()
        assert "#SBATCH --account=my-other-condo" in script
        assert "#SBATCH -c 16" in script
        # non-overridden keys still come from the config
        assert "#SBATCH --mem=8G" in script

    def test_repo_oscar_yaml_parses_and_resolves(self):
        from pathlib import Path

        repo_yaml = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "cluster"
            / "oscar.yaml"
        )
        for kind in ("generate", "jaxtrain", "torchtrain"):
            resources = resolve_resources(kind, repo_yaml, {})
            # Real account, seeded from sacctmgr (not the placeholder guess).
            assert resources["account"] == "carney-mjfrank-condo2"
            assert set(resources) >= {
                "account",
                "partition",
                "num_gpus",
                "cores",
                "mem",
                "time",
                "modules",
            }


class TestNFilesPassthrough:
    def test_n_files_reaches_the_generate_command(self, model_config, tmp_path):
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "generate",
                "--config-path",
                str(model_config),
                "--output-path",
                str(out),
                "--n-files",
                "7",
                "--script-only",
            ],
        )
        assert result.exit_code == 0, result.output
        script = (out / "runs").glob("*.sh").__next__().read_text()
        assert "--n-files 7" in script

    def test_omitted_n_files_is_absent(self, model_config, tmp_path):
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "generate",
                "--config-path",
                str(model_config),
                "--output-path",
                str(out),
                "--script-only",
            ],
        )
        assert result.exit_code == 0
        script = (out / "runs").glob("*.sh").__next__().read_text()
        assert "--n-files" not in script


def record_partition(result) -> str:
    return last_json_line(result.output)["partition"]


class TestGeneratedScriptRuntime:
    """Properties the script must have when SLURM runs it on a compute node."""

    def script_for(self, model_config, tmp_path, extra=()):
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "generate",
                "--config-path",
                str(model_config),
                "--output-path",
                str(out),
                "--script-only",
                *extra,
            ],
        )
        assert result.exit_code == 0, result.output
        return next((out / "runs").glob("*.sh")).read_text()

    def test_pins_the_uv_project_directory(self, model_config, tmp_path):
        # SLURM starts the job in the sbatch submission directory ($HOME for a
        # driver submitting over ssh); `uv run <cmd>` there fails with
        # "Failed to spawn" or silently resolves a foreign pyproject.toml.
        script = self.script_for(model_config, tmp_path)
        repo_root = Path(gen_sbatch.__file__).resolve().parents[1]
        assert f"cd {repo_root} || exit 1" in script

    def test_output_path_with_spaces_is_rejected_upfront(self, model_config, tmp_path):
        # SLURM reads #SBATCH --output= literally to end of line and does not
        # dequote, so whitespace cannot be expressed at all. Fail at
        # generation time with the reason, not at submission with a parse error.
        out = tmp_path / "out dir with spaces"
        result = runner.invoke(
            app,
            [
                "generate",
                "--config-path",
                str(model_config),
                "--output-path",
                str(out),
                "--script-only",
            ],
        )
        assert result.exit_code != 0
        assert "whitespace" in result.output

    def test_slurm_log_directives_are_unquoted(self, model_config, tmp_path):
        script = self.script_for(model_config, tmp_path)
        directives = [
            line
            for line in script.splitlines()
            if line.startswith(("#SBATCH --output=", "#SBATCH --error="))
        ]
        assert len(directives) == 2
        for line in directives:
            value = line.split("=", 1)[1]
            assert not value.startswith(("'", '"')), line

    def test_tracking_uri_embedded_absolute(self, model_config, tmp_path, monkeypatch):
        # A relative sqlite URI resolves against each worker's own CWD, so
        # workers would write to a different database than the one whose
        # experiment id the JSON contract reports.
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        monkeypatch.chdir(tmp_path)
        assert absolutize_tracking_uri("sqlite:///mlflow.db") == (
            f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}"
        )

    def test_absolutize_leaves_server_and_absolute_uris_alone(self):
        assert (
            absolutize_tracking_uri("http://localhost:5000") == "http://localhost:5000"
        )
        assert (
            absolutize_tracking_uri("sqlite:////abs/path/mlflow.db")
            == "sqlite:////abs/path/mlflow.db"
        )


class TestModulesResolution:
    def write_config(self, tmp_path, payload):
        import yaml as _yaml

        path = tmp_path / "cluster.yaml"
        path.write_text(_yaml.safe_dump(payload))
        return path

    def test_per_job_kind_modules_beat_top_level(self, tmp_path):
        config = self.write_config(
            tmp_path,
            {
                "job_defaults": {"jaxtrain": {"modules": ["python", "gcc", "cuda"]}},
                "modules": ["python", "gcc"],
            },
        )
        resources = resolve_resources("jaxtrain", config, {})
        assert resources["modules"] == ["python", "gcc", "cuda"]

    def test_empty_list_disables_module_loading(self, tmp_path):
        config = self.write_config(tmp_path, {"modules": []})
        assert resolve_resources("generate", config, {})["modules"] == []

    def test_empty_modules_emits_no_module_load_lines(self, model_config, tmp_path):
        config = self.write_config(tmp_path, {"modules": []})
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "generate",
                "--config-path",
                str(model_config),
                "--output-path",
                str(out),
                "--cluster-config",
                str(config),
                "--script-only",
            ],
        )
        assert result.exit_code == 0, result.output
        script = next((out / "runs").glob("*.sh")).read_text()
        assert "module load" not in script

    def test_absent_modules_key_uses_defaults(self, tmp_path):
        config = self.write_config(tmp_path, {"job_defaults": {}})
        assert resolve_resources("generate", config, {})["modules"] == ["python", "gcc"]


class TestResourceValidation:
    def test_sexagesimal_time_is_rejected_with_a_hint(self, tmp_path):
        # PyYAML (YAML 1.1) loads unquoted 12:00:00 as the integer 43200,
        # which SLURM would read as 43200 MINUTES.
        config = tmp_path / "cluster.yaml"
        config.write_text("job_defaults:\n  generate:\n    time: 12:00:00\n")
        with pytest.raises(typer.BadParameter) as excinfo:
            resolve_resources("generate", config, {})
        assert "43200" in str(excinfo.value)
        assert "12:00:00" in str(excinfo.value)  # the quoted form to use

    def test_bare_integer_time_is_rejected_with_both_readings(self, tmp_path):
        # `time: 30` is ambiguous after parsing: the user may have written 30
        # (minutes) or 0:30, which YAML also loads as 30. The message must
        # offer both quoted forms rather than guess.
        config = tmp_path / "cluster.yaml"
        config.write_text("job_defaults:\n  generate:\n    time: 30\n")
        with pytest.raises(typer.BadParameter) as excinfo:
            resolve_resources("generate", config, {})
        message = str(excinfo.value)
        assert 'time: "30"' in message
        assert 'time: "00:00:30"' in message

    @pytest.mark.parametrize(
        "value", ["30", "4:00", "12:00:00", "1-00", "2-06:30", "1-00:00:00"]
    )
    def test_valid_slurm_time_forms_accepted(self, value, tmp_path):
        # Quoted strings, including a bare minute count, pass through.
        config = tmp_path / "cluster.yaml"
        config.write_text(f'job_defaults:\n  generate:\n    time: "{value}"\n')
        assert resolve_resources("generate", config, {})["time"] == value

    def test_missing_cluster_config_is_a_clean_cli_error(self, model_config, tmp_path):
        # A mistyped path must not traceback: the driver would get no JSON.
        result = runner.invoke(
            app,
            [
                "generate",
                "--config-path",
                str(model_config),
                "--output-path",
                str(tmp_path / "out"),
                "--cluster-config",
                str(tmp_path / "nope.yaml"),
                "--script-only",
            ],
        )
        assert result.exit_code != 0
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "does not exist" in result.output


class TestModelNameSanitization:
    @pytest.mark.parametrize(
        ("model", "expected_fragment"),
        [
            ("../../etc/ddm", "_.._etc_ddm"),  # leading dots also stripped
            ("ddm/../../x", "ddm_.._.._x"),
            ("my ddm", "my_ddm"),
            ("ddm;rm -rf /", "ddm_rm_-rf__"),
        ],
    )
    def test_hostile_model_names_stay_inside_runs_dir(
        self, tmp_path, model, expected_fragment
    ):
        # MODEL comes from a user-supplied YAML and lands in the script
        # filename, the SLURM job name and the log filenames.
        config = tmp_path / "config.yaml"
        config.write_text(f'MODEL: "{model}"\n')
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "generate",
                "--config-path",
                str(config),
                "--output-path",
                str(out),
                "--script-only",
            ],
        )
        assert result.exit_code == 0, result.output
        script_path = Path(last_json_line(result.output)["sbatch_script"])
        assert script_path.parent == (out / "runs").resolve(), script_path
        assert expected_fragment in script_path.name
        assert script_path.exists()

    def test_job_name_directive_is_a_single_token(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text('MODEL: "my ddm"\n')
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "generate",
                "--config-path",
                str(config),
                "--output-path",
                str(out),
                "--script-only",
            ],
        )
        assert result.exit_code == 0, result.output
        script = next((out / "runs").glob("*.sh")).read_text()
        job_line = next(
            line for line in script.splitlines() if line.startswith("#SBATCH -J ")
        )
        assert len(job_line.split()) == 3, job_line


class TestLogLevel:
    def test_lowercase_log_level_works(self, model_config, tmp_path):
        # The driver passes the same log-level string to both this tool and
        # the worker CLI; a TypeError traceback here would leave the driver
        # with no JSON line to parse.
        result = runner.invoke(
            app,
            [
                "generate",
                "--config-path",
                str(model_config),
                "--output-path",
                str(tmp_path / "out"),
                "--script-only",
                "--log-level",
                "info",
            ],
        )
        assert result.exit_code == 0, result.output
        assert last_json_line(result.output)["job_id"] is None

    def test_invalid_log_level_is_a_clean_error(self, model_config, tmp_path):
        result = runner.invoke(
            app,
            [
                "generate",
                "--config-path",
                str(model_config),
                "--output-path",
                str(tmp_path / "out"),
                "--script-only",
                "--log-level",
                "VERBOSE",
            ],
        )
        assert result.exit_code != 0
        assert "Unknown log level" in result.output


@pytest.fixture(autouse=True)
def _no_cwd_litter(tmp_path, monkeypatch):
    """Every test runs from a scratch CWD; assert nothing is written there."""
    monkeypatch.chdir(tmp_path)
    yield
    stray = [p for p in tmp_path.iterdir() if p.suffix in (".sh", ".out", ".err")]
    assert not stray, f"artifacts leaked into CWD: {stray}"


CONDO = {
    "account": "my-condo",
    "partition": "batch",
    "max_cores": 208,
    "priority": 10000,
}
SPILL = {"account": "default", "partition": "batch", "max_cores": 64, "priority": 0}


class TestLaneSplit:
    def test_splits_in_proportion_to_core_caps(self):
        plan = split_across_lanes([CONDO, SPILL], 100)
        assert [(lane["account"], size) for lane, size in plan] == [
            ("my-condo", 77),
            ("default", 23),
        ]

    @pytest.mark.parametrize("n_jobs", [2, 3, 5, 7, 10, 33, 100, 501])
    def test_every_task_is_allocated_exactly_once(self, n_jobs):
        plan = split_across_lanes([CONDO, SPILL], n_jobs)
        assert sum(size for _, size in plan) == n_jobs
        assert all(size > 0 for _, size in plan)

    def test_highest_priority_lane_leads(self):
        # Passed worst-first; the split must still favour the condo.
        plan = split_across_lanes([SPILL, CONDO], 100)
        assert plan[0][0]["account"] == "my-condo"
        assert plan[0][1] > plan[1][1]

    def test_a_single_task_does_not_fan_out(self):
        # Splitting one task across two lanes would just add queue latency.
        assert len(split_across_lanes([CONDO, SPILL], 1)) == 1

    def test_lanes_without_a_cap_are_ignored(self):
        plan = split_across_lanes([CONDO, {"account": "vnc", "partition": "vnc"}], 10)
        assert len(plan) == 1


class TestLocalConfigLayering:
    def write(self, path, payload):
        import yaml as _yaml

        path.write_text(_yaml.safe_dump(payload))
        return path

    def test_local_file_merges_over_the_committed_one(self, tmp_path):
        shared = self.write(
            tmp_path / "oscar.yaml",
            {
                "job_defaults": {"generate": {"account": "lab-condo", "cores": 1}},
                "modules": ["python", "gcc"],
            },
        )
        self.write(
            tmp_path / "oscar.local.yaml",
            {"job_defaults": {"generate": {"account": "my-condo", "lanes": [CONDO]}}},
        )
        merged = load_cluster_config(shared)
        generate = merged["job_defaults"]["generate"]
        assert generate["account"] == "my-condo"  # personal wins
        assert generate["cores"] == 1  # shared survives
        assert generate["lanes"] == [CONDO]
        assert merged["modules"] == ["python", "gcc"]

    def test_absent_local_file_is_not_an_error(self, tmp_path):
        shared = self.write(tmp_path / "oscar.yaml", {"modules": ["python"]})
        assert load_cluster_config(shared)["modules"] == ["python"]

    def test_local_lanes_reach_the_resolver(self, tmp_path):
        shared = self.write(
            tmp_path / "oscar.yaml", {"job_defaults": {"generate": {"cores": 2}}}
        )
        self.write(
            tmp_path / "oscar.local.yaml",
            {"job_defaults": {"generate": {"lanes": [CONDO, SPILL]}}},
        )
        resources = resolve_resources("generate", shared, {})
        assert len(resources["lanes"]) == 2
        assert resources["cores"] == 2


class TestFanOut:
    def cluster_files(self, tmp_path, lanes):
        import yaml as _yaml

        shared = tmp_path / "oscar.yaml"
        shared.write_text(
            _yaml.safe_dump(
                {
                    "job_defaults": {
                        "generate": {
                            "account": "lab-condo",
                            "partition": "batch",
                            "cores": 1,
                            "mem": "16G",
                            "num_gpus": 0,
                            "time": "04:00:00",
                        }
                    }
                }
            )
        )
        (tmp_path / "oscar.local.yaml").write_text(
            _yaml.safe_dump({"job_defaults": {"generate": {"lanes": lanes}}})
        )
        return shared

    def invoke(self, model_config, tmp_path, extra=()):
        cluster = self.cluster_files(tmp_path, [CONDO, SPILL])
        out = tmp_path / "out"
        return runner.invoke(
            app,
            [
                "generate",
                "--config-path",
                str(model_config),
                "--output-path",
                str(out),
                "--cluster-config",
                str(cluster),
                "--n-jobs-in-array",
                "100",
                "--script-only",
                *extra,
            ],
        ), out

    def test_one_json_line_per_lane(self, model_config, tmp_path):
        result, out = self.invoke(model_config, tmp_path, ["--use-all-lanes"])
        assert result.exit_code == 0, result.output
        records = [
            json.loads(ln) for ln in result.output.splitlines() if ln.startswith("{")
        ]
        assert len(records) == 2
        assert [r["account"] for r in records] == ["my-condo", "default"]
        assert sum(r["array_size"] for r in records) == 100
        assert {r["n_lanes"] for r in records} == {2}
        # distinct scripts, distinct array sizes in the emitted sbatch headers
        scripts = sorted((out / "runs").glob("*.sh"))
        assert len(scripts) == 2
        arrays = sorted(
            line
            for script in scripts
            for line in script.read_text().splitlines()
            if line.startswith("#SBATCH --array")
        )
        assert arrays == ["#SBATCH --array=1-23", "#SBATCH --array=1-77"]

    def test_without_the_flag_it_stays_single_lane(self, model_config, tmp_path):
        """Fan-out must be opt-in: it changes how many jobs land on the cluster."""
        result, out = self.invoke(model_config, tmp_path)
        assert result.exit_code == 0, result.output
        records = [
            json.loads(ln) for ln in result.output.splitlines() if ln.startswith("{")
        ]
        assert len(records) == 1
        assert records[0]["account"] == "lab-condo"
        assert records[0]["array_size"] == 100
        assert records[0]["n_lanes"] == 1
        assert len(list((out / "runs").glob("*.sh"))) == 1

    def test_worker_run_names_do_not_collide_across_lanes(
        self, model_config, tmp_path, fake_sbatch_ok, isolated_mlflow
    ):
        """Both arrays number tasks from 1, so the lane must enter the name."""
        cluster = self.cluster_files(tmp_path, [CONDO, SPILL])
        result = runner.invoke(
            app,
            [
                "generate",
                "--config-path",
                str(model_config),
                "--output-path",
                str(tmp_path / "out"),
                "--cluster-config",
                str(cluster),
                "--n-jobs-in-array",
                "10",
                "--use-all-lanes",
            ],
        )
        assert result.exit_code == 0, result.output
        names = [
            ln.split("--mlflow-run-name ")[1].split(" --")[0]
            for ln in result.output.splitlines()
            if "--mlflow-run-name" in ln
        ]
        assert len(names) == 2 and names[0] != names[1], names

    def test_a_failed_lane_makes_the_whole_submission_fail(
        self, model_config, tmp_path, fake_sbatch_fail, isolated_mlflow
    ):
        """Partial success is failure: some lanes run, some do not, and only
        the JSON lines say which."""
        cluster = self.cluster_files(tmp_path, [CONDO, SPILL])
        result = runner.invoke(
            app,
            [
                "generate",
                "--config-path",
                str(model_config),
                "--output-path",
                str(tmp_path / "out"),
                "--cluster-config",
                str(cluster),
                "--n-jobs-in-array",
                "10",
                "--use-all-lanes",
            ],
        )
        assert result.exit_code == 1
        records = [
            json.loads(ln) for ln in result.output.splitlines() if ln.startswith("{")
        ]
        assert len(records) == 2
        assert all(r["job_id"] is None for r in records)


class TestUvResolution:
    """The generated script has to find uv on a real cluster node.

    Oscar's `module load python` gives a spack Python 3.13 with neither pip
    nor uv, so the original `python -m uv` + pip-bootstrap died on "No module
    named pip" while a working uv sat unused in ~/.local/bin.
    """

    def test_prefers_the_uv_binary_over_the_python_module(self):
        script = gen_sbatch.SBATCH_TEMPLATE
        binary_check = script.index("command -v uv")
        module_check = script.index("python -m uv --version")
        assert binary_check < module_check

    def test_puts_the_user_install_dir_on_path(self):
        # A non-interactive batch shell does not read the profile that adds it.
        assert 'export PATH="$HOME/.local/bin:$PATH"' in gen_sbatch.SBATCH_TEMPLATE

    def test_fails_with_the_install_command_when_uv_cannot_be_found(self):
        script = gen_sbatch.SBATCH_TEMPLATE
        assert "astral.sh/uv/install.sh" in script
        assert "exit 1" in script.split("could not be installed")[1]

    def test_the_pip_bootstrap_is_verified_before_being_trusted(self):
        """pip can fail — compute nodes often have no outbound network.

        The first version set UV="python -m uv" straight after the install and
        fell through to `$UV run`, landing back at the ModuleNotFoundError this
        block exists to prevent. Resolution must happen *after* the install.
        """
        script = gen_sbatch.SBATCH_TEMPLATE
        install = script.index("pip install")
        # The last resolve_uv call, and the failure branch, both come after it.
        assert script.rindex("if ! resolve_uv; then") > install
        assert script.index("could not be installed") > install
        # And nothing assigns UV unconditionally on the way out of the install.
        assert 'UV="python -m uv"\nfi' not in script

    def test_resolution_is_defined_once(self):
        # One helper, one failure message — the earlier shape duplicated both.
        assert gen_sbatch.SBATCH_TEMPLATE.count("resolve_uv() {{") == 1
        assert gen_sbatch.SBATCH_TEMPLATE.count("could not be installed") == 1

    def test_runs_through_the_resolved_uv(self):
        # Not a hardcoded `python -m uv run`, which is what broke.
        assert "$UV run {command}" in gen_sbatch.SBATCH_TEMPLATE


class TestNCpusPassthrough:
    """The generate command states its worker count instead of inferring it.

    ssm-simulators defaults n_cpus to "all", which reads sched_getaffinity —
    correct under SLURM, but it grabs the whole machine anywhere else, and
    either way the number never reaches the submission record.
    """

    def test_n_cpus_matches_the_requested_cores(self, model_config, tmp_path):
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "generate",
                "--config-path",
                str(model_config),
                "--output-path",
                str(out),
                "--cores",
                "12",
                "--script-only",
            ],
        )
        assert result.exit_code == 0, result.output
        script = next((out / "runs").glob("*.sh")).read_text()
        # The CLI value and the SBATCH directive must not drift apart.
        assert "--n-cpus 12" in script
        assert "#SBATCH -c 12" in script

    def test_training_commands_do_not_get_n_cpus(self, model_config, tmp_path):
        # jaxtrain/torchtrain have no such option; passing it would be an error.
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "jaxtrain",
                "--config-path",
                str(model_config),
                "--output-path",
                str(out),
                "--training-data-folder",
                str(tmp_path / "data"),
                "--script-only",
            ],
        )
        assert result.exit_code == 0, result.output
        script = next((out / "runs").glob("*.sh")).read_text()
        assert "--n-cpus" not in script
