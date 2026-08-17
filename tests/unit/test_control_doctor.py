from __future__ import annotations

import sys
from pathlib import Path

from shardgrid.common.config import ClusterConfig
from shardgrid.common.enums import Health
from shardgrid.control import doctor as doctor_module
from shardgrid.control.doctor import run_control_doctor


class _Result:
    def __init__(self, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.ok = exit_code == 0


def _cluster_config(jobs_root: Path) -> ClusterConfig:
    payload = {
        "control": {"machine_id": "machine-a", "hostname": "control-a.local"},
        "jobs_root": str(jobs_root),
        "ssh": {},
        "runtime": {},
        "network": {},
        "backend_preference": {},
        "manual_override": {},
        "workers": [
            {
                "id": "gpu4060",
                "machine_id": "machine-c",
                "physical_os": "windows",
                "runtime_os": "wsl2_linux",
                "runtime": "wsl2",
                "host": "machine-c.local",
                "ssh_user": "shardgrid",
            }
        ],
    }
    return ClusterConfig.from_dict(payload)


def _conda_run(stdout: str, stderr: str = "", exit_code: int = 0) -> _Result:
    return _Result(stdout=stdout, stderr=stderr, exit_code=exit_code)


def _all_deps_present(python_executable: str | None) -> dict[str, bool]:
    return {dep: True for dep in ("shardgrid", "yaml", "pytest", "ruff", "mypy")}


def _all_deps_missing(python_executable: str | None) -> dict[str, bool]:
    return {dep: False for dep in ("shardgrid", "yaml", "pytest", "ruff", "mypy")}


def _patch_basics(monkeypatch) -> None:
    monkeypatch.setattr(doctor_module, "detect_python", lambda: (sys.executable, "3.13.5"))
    monkeypatch.setattr(doctor_module, "detect_tool_versions", lambda: {
        "ssh": "OpenSSH_9.6p1",
        "git": "git version 2.43.0",
        "iperf3": "not_installed",
    })
    monkeypatch.setattr(doctor_module, "_local_address", lambda: "10.0.0.5")


def test_control_doctor_healthy(tmp_path, monkeypatch) -> None:
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    config = _cluster_config(jobs_root)
    _patch_basics(monkeypatch)
    monkeypatch.setattr(
        doctor_module, "detect_conda", lambda: ("/opt/conda/bin/conda", "base", sys.prefix)
    )
    monkeypatch.setattr(
        doctor_module,
        "_run",
        lambda cmd: _conda_run(
            "conda 25.5.1"
            if cmd[-1] == "--version"
            else f"base  {sys.prefix}\nother   /tmp/other-env\n"
        ),
    )
    monkeypatch.setattr(doctor_module, "_check_dependencies", _all_deps_present)

    report = run_control_doctor(config)

    assert report.health == Health.HEALTHY
    assert report.exit_code == 0
    assert report.environment["selected_environment"] == "base"
    jobs_check = next(check for check in report.checks if check.name == "jobs_root")
    assert jobs_check.status == "ok"


def test_control_doctor_conda_missing_blocks(monkeypatch) -> None:
    _patch_basics(monkeypatch)
    monkeypatch.setattr(doctor_module, "detect_conda", lambda: (None, None, None))
    monkeypatch.setattr(doctor_module, "_check_dependencies", _all_deps_present)

    report = run_control_doctor(None)

    assert report.health == Health.BLOCKED_MANUAL_ACTION
    assert report.exit_code == 2
    conda_check = next(check for check in report.checks if check.name == "conda")
    assert conda_check.manual_action_required is True


def test_control_doctor_missing_deps_is_degraded(monkeypatch) -> None:
    _patch_basics(monkeypatch)
    monkeypatch.setattr(
        doctor_module, "detect_conda", lambda: ("/opt/conda/bin/conda", "base", sys.prefix)
    )
    monkeypatch.setattr(
        doctor_module,
        "_run",
        lambda cmd: _conda_run(
            "conda 25.5.1"
            if cmd[-1] == "--version"
            else f"base  {sys.prefix}\nother   /tmp/other-env\n"
        ),
    )
    monkeypatch.setattr(doctor_module, "_check_dependencies", _all_deps_missing)

    report = run_control_doctor(None)

    assert report.health == Health.DEGRADED
    assert report.exit_code == 1
    deps_check = next(
        check for check in report.checks if check.name == "project_dependencies"
    )
    assert deps_check.status == "degraded"


def test_control_doctor_jobs_root_missing_is_degraded(tmp_path, monkeypatch) -> None:
    jobs_root = tmp_path / "jobs-not-created"
    config = _cluster_config(jobs_root)
    _patch_basics(monkeypatch)
    monkeypatch.setattr(
        doctor_module, "detect_conda", lambda: ("/opt/conda/bin/conda", "base", sys.prefix)
    )
    monkeypatch.setattr(
        doctor_module,
        "_run",
        lambda cmd: _conda_run(
            "conda 25.5.1"
            if cmd[-1] == "--version"
            else f"base  {sys.prefix}\nother   /tmp/other-env\n"
        ),
    )
    monkeypatch.setattr(doctor_module, "_check_dependencies", _all_deps_present)

    report = run_control_doctor(config)

    assert report.health == Health.DEGRADED
    assert report.exit_code == 1
    jobs_check = next(check for check in report.checks if check.name == "jobs_root")
    assert jobs_check.status == "degraded"
    assert any("mkdir" in action for action in report.manual_actions)


def test_control_doctor_without_config_marks_jobs_root_not_checked(monkeypatch) -> None:
    _patch_basics(monkeypatch)
    monkeypatch.setattr(
        doctor_module, "detect_conda", lambda: ("/opt/conda/bin/conda", "base", sys.prefix)
    )
    monkeypatch.setattr(
        doctor_module,
        "_run",
        lambda cmd: _conda_run(
            "conda 25.5.1"
            if cmd[-1] == "--version"
            else f"base  {sys.prefix}\nother   /tmp/other-env\n"
        ),
    )
    monkeypatch.setattr(doctor_module, "_check_dependencies", _all_deps_present)

    report = run_control_doctor(None)

    jobs_check = next(check for check in report.checks if check.name == "jobs_root")
    assert jobs_check.status == "not_checked"


def test_control_doctor_report_is_json_serializable(monkeypatch) -> None:
    import json

    _patch_basics(monkeypatch)
    monkeypatch.setattr(
        doctor_module, "detect_conda", lambda: ("/opt/conda/bin/conda", "base", sys.prefix)
    )
    monkeypatch.setattr(
        doctor_module,
        "_run",
        lambda cmd: _conda_run(
            "conda 25.5.1"
            if cmd[-1] == "--version"
            else f"base  {sys.prefix}\nother   /tmp/other-env\n"
        ),
    )
    monkeypatch.setattr(doctor_module, "_check_dependencies", _all_deps_present)

    report = run_control_doctor(None)
    payload = report.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["target"] == "control"
    assert payload["health"] == "healthy"