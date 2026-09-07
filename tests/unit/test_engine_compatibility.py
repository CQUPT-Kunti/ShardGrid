from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from shardgrid.common.process import ProcessResult
from shardgrid.engines.compatibility import (
    GALVATRON_PYPI_INSTALL,
    CompatibilityStatus,
    GalvatronEvidence,
    _parse_install_preflight,
    _parse_version,
    build_worker_version_script,
    collect_galvatron_declared_requirements,
    collect_galvatron_evidence,
    collect_worker_version_evidence,
    evaluate_galvatron,
    load_galvatron_evidence,
    parse_worker_version_evidence,
    run_galvatron_check,
    save_galvatron_evidence,
)

TORCH_OK = (
    "torch_version 2.7.1+cu118\n"
    "torch_cuda_version 11.8\n"
    "torch_cuda_available True"
)
TORCH_OK_2 = (
    "torch_version 2.7.1+cu118\n"
    "torch_cuda_version 11.8\n"
    "torch_cuda_available False"
)
PIP_SHOW_OFFICIAL = (
    "Name: galvatron\n"
    "Version: 0.9.1\n"
    "Summary: Galvatron training framework\n"
    "Home-page: https://github.com/PKU-DAIR/Hetu-Galvatron\n"
    "Location: /opt/conda/envs/shardgrid/lib/python3.12/site-packages\n"
)
IMPORT_OK = (
    "file: /opt/conda/envs/shardgrid/lib/python3.12/site-packages/"
    "galvatron/__init__.py\nversion: unknown"
)
IMPORT_NOT_FOUND = (
    "Traceback (most recent call last):\n"
    "ModuleNotFoundError: No module named 'galvatron'"
)
IMPORT_BROKEN = (
    "Traceback (most recent call last):\n"
    "RuntimeError: Galvatron requires torch>=2.0 but a broken build was detected"
)


class _FakeResult(ProcessResult):
    def __init__(self, *, exit_code: int = 0, stdout: str = "", stderr: str = "") -> None:
        super().__init__(
            args=(),
            recorded_command="",
            shell=False,
            cwd=None,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=False,
            runtime_environment={},
        )


class FakeRunner:
    """Scripted runner: command -> ProcessResult, or raises if configured."""

    def __init__(
        self,
        table: dict[str, ProcessResult],
        *,
        raise_on: list[str] | None = None,
    ) -> None:
        self.table = table
        self.raise_on = raise_on or []
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], *, timeout: float) -> ProcessResult:
        self.calls.append(command)
        rendered = " ".join(command)
        for pattern in self.raise_on:
            if pattern in rendered:
                raise RuntimeError(f"simulated failure for {rendered}")
        if rendered in self.table:
            return self.table[rendered]
        if any(pattern in rendered for pattern in self.table):
            for pattern, result in self.table.items():
                if pattern in rendered:
                    return result
        return _FakeResult(exit_code=1, stderr=f"unscripted command: {rendered}")


def _python_probe() -> str:
    return "3.12.13"


def _torch_cmd(python: str) -> list[str]:
    return [python, "-c", TORCH_OK]


def _pip_show_cmd(python: str) -> list[str]:
    return [python, "-m", "pip", "show", "galvatron"]


def _import_cmd(python: str) -> list[str]:
    return [python, "-c", "import galvatron; print('file:', galvatron.__file__)"]


def _run(
    monkeypatch: pytest.MonkeyPatch,
    table: dict[str, ProcessResult],
    *,
    python: str = "/opt/conda/envs/shardgrid/bin/python",
    conda: tuple[str | None, str | None, str | None] = (
        "/opt/conda/bin/conda",
        "shardgrid",
        "/opt/conda/envs/shardgrid",
    ),
    runner: FakeRunner | None = None,
    **kwargs: Any,
) -> tuple[GalvatronEvidence, Any]:
    monkeypatch.setattr(
        "shardgrid.engines.compatibility.detect_conda", lambda: conda
    )
    monkeypatch.setattr(
        "shardgrid.engines.compatibility.detect_python", lambda: (python, "3.12.13")
    )
    fake = runner or FakeRunner(table)
    evidence = collect_galvatron_evidence(python_executable=python, runner=fake)
    result = evaluate_galvatron(evidence, runner=fake, **kwargs)
    return evidence, result


def test_galvatron_present_official_pypi_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _, result = _run(
        monkeypatch,
        {
            "torch_version": _FakeResult(stdout=TORCH_OK),
            "pip show galvatron": _FakeResult(stdout=PIP_SHOW_OFFICIAL),
            "import galvatron": _FakeResult(stdout=IMPORT_OK),
        },
    )
    assert result.status == CompatibilityStatus.AVAILABLE
    assert result.galvatron_installed is True
    assert result.galvatron_version == "0.9.1"
    assert result.galvatron_source == "github:PKU-DAIR/Hetu-Galvatron"
    assert result.torch_version == "2.7.1+cu118"
    assert result.torch_cuda_version == "11.8"
    assert result.torch_cuda_available is True
    assert not result.blockers
    assert "T056-T060" in " ".join(result.limitations)


def test_galvatron_present_git_origin_official(monkeypatch: pytest.MonkeyPatch) -> None:
    _, result = _run(
        monkeypatch,
        {
            "torch_version": _FakeResult(stdout=TORCH_OK),
            "pip show galvatron": _FakeResult(
                stdout="Name: galvatron\nVersion: 0.9.1\nLocation: /opt/checkout\n"
            ),
            "import galvatron": _FakeResult(
                stdout="file: /opt/checkout/Hetu-Galvatron/galvatron/__init__.py\nversion: 0.9.1"
            ),
            "remote get-url origin": _FakeResult(
                stdout="https://github.com/PKU-DAIR/Hetu-Galvatron.git"
            ),
        },
    )
    assert result.status == CompatibilityStatus.AVAILABLE
    assert result.galvatron_version == "0.9.1"
    assert result.galvatron_source == "github:PKU-DAIR/Hetu-Galvatron"
    assert "remote get-url origin" in " ".join(result.commands)


def test_galvatron_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    _, result = _run(
        monkeypatch,
        {
            "torch_version": _FakeResult(stdout=TORCH_OK),
            "pip show galvatron": _FakeResult(exit_code=1, stderr="WARNING: Package(s) not found"),
            "import galvatron": _FakeResult(exit_code=1, stderr=IMPORT_NOT_FOUND),
        },
    )
    assert result.status == CompatibilityStatus.NOT_INSTALLED
    assert result.galvatron_installed is False
    assert result.galvatron_version is None
    assert result.proposed_install_command == GALVATRON_PYPI_INSTALL
    assert result.manual_actions
    assert "official-source" in result.manual_actions[0]


def test_version_parsing_from_attribute_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _, by_attr = _run(
        monkeypatch,
        {
            "torch_version": _FakeResult(stdout=TORCH_OK),
            "pip show galvatron": _FakeResult(stdout="Name: galvatron\nLocation: /opt/x\n"),
            "import galvatron": _FakeResult(
                stdout="file: /opt/x/galvatron/__init__.py\nversion: 9.9.9"
            ),
        },
    )
    assert by_attr.galvatron_version == "9.9.9"

    _, both_missing = _run(
        monkeypatch,
        {
            "torch_version": _FakeResult(stdout=TORCH_OK),
            "pip show galvatron": _FakeResult(stdout="Name: galvatron\nLocation: /opt/x\n"),
            "import galvatron": _FakeResult(
                stdout="file: /opt/x/galvatron/__init__.py\nversion: unknown"
            ),
        },
    )
    assert both_missing.galvatron_version is None
    assert both_missing.status == CompatibilityStatus.AVAILABLE

    assert _parse_version({"version": "0.9.1"}, "") == "0.9.1"
    assert _parse_version({"version": "unknown"}, "version: 1.2.3") == "1.2.3"


def test_command_failure_produces_check_failed_with_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = FakeRunner({}, raise_on=["torch"])
    result = run_galvatron_check(
        python_executable="/opt/conda/envs/shardgrid/bin/python", runner=runner
    )
    assert result.status == CompatibilityStatus.CHECK_FAILED
    assert any("evidence incomplete" in blocker for blocker in result.blockers)
    assert "simulated failure" in " ".join(result.blockers)
    assert any("harness runner raised" in entry.stderr_tail for entry in result.diagnostics)


def test_command_failure_nonzero_exit_preserves_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, result = _run(
        monkeypatch,
        {
            "torch_version": _FakeResult(
                exit_code=1, stderr="ModuleNotFoundError: No module named 'torch'"
            ),
            "pip show galvatron": _FakeResult(exit_code=1),
            "import galvatron": _FakeResult(exit_code=1, stderr=IMPORT_NOT_FOUND),
        },
    )
    assert result.status == CompatibilityStatus.NOT_INSTALLED
    torch_diag = next(entry for entry in result.diagnostics if entry.name == "detect.torch")
    assert torch_diag.exit_code == 1
    assert "No module named 'torch'" in torch_diag.stderr_tail


def test_conda_and_runtime_evidence_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    _, result = _run(
        monkeypatch,
        {
            "torch_version": _FakeResult(stdout=TORCH_OK),
            "pip show galvatron": _FakeResult(exit_code=1),
            "import galvatron": _FakeResult(exit_code=1, stderr=IMPORT_NOT_FOUND),
        },
        conda=("C:\\conda\\conda.exe", "shardgrid", "C:\\envs\\shardgrid"),
        python="C:\\envs\\shardgrid\\python.exe",
    )
    assert result.conda_environment == "shardgrid"
    assert result.conda_prefix == "C:\\envs\\shardgrid"
    assert result.python_executable == "C:\\envs\\shardgrid\\python.exe"
    assert result.python_version == "3.12.13"
    assert result.torch_version == "2.7.1+cu118"
    assert result.torch_cuda_version == "11.8"
    assert result.torch_cuda_available is True


def test_incompatible_when_import_broken(monkeypatch: pytest.MonkeyPatch) -> None:
    _, result = _run(
        monkeypatch,
        {
            "torch_version": _FakeResult(stdout=TORCH_OK),
            "pip show galvatron": _FakeResult(stdout=PIP_SHOW_OFFICIAL),
            "import galvatron": _FakeResult(exit_code=1, stderr=IMPORT_BROKEN),
        },
    )
    assert result.status == CompatibilityStatus.INCOMPATIBLE
    assert result.galvatron_installed is True
    assert any("import failed" in blocker for blocker in result.blockers)
    import_diag = next(
        entry for entry in result.diagnostics if entry.name == "detect.import_galvatron"
    )
    assert "Galvatron requires torch>=2.0" in import_diag.stderr_tail


def test_incompatible_when_torch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _, result = _run(
        monkeypatch,
        {
            "torch_version": _FakeResult(exit_code=1, stderr="ModuleNotFoundError: torch"),
            "pip show galvatron": _FakeResult(stdout=PIP_SHOW_OFFICIAL),
            "import galvatron": _FakeResult(stdout=IMPORT_OK),
        },
    )
    assert result.status == CompatibilityStatus.INCOMPATIBLE
    assert any("torch" in blocker.lower() for blocker in result.blockers)
    assert result.torch_version is None


def test_blocked_when_source_unofficial(monkeypatch: pytest.MonkeyPatch) -> None:
    _, result = _run(
        monkeypatch,
        {
            "torch_version": _FakeResult(stdout=TORCH_OK),
            "pip show galvatron": _FakeResult(
                stdout="Name: galvatron\nVersion: 0.9.1\nHome-page: https://github.com/evil/galvatron\n"
            ),
            "import galvatron": _FakeResult(stdout=IMPORT_OK),
        },
    )
    assert result.status == CompatibilityStatus.BLOCKED
    assert result.galvatron_source == "unofficial/unknown"
    assert any("official" in action for action in result.manual_actions)


def test_blocked_when_install_would_change_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    _, result = _run(
        monkeypatch,
        {
            "torch_version": _FakeResult(stdout=TORCH_OK),
            "pip show galvatron": _FakeResult(exit_code=1),
            "import galvatron": _FakeResult(exit_code=1, stderr=IMPORT_NOT_FOUND),
            "pip install --dry-run": _FakeResult(
                stdout="Would install galvatron-0.9.1 torch-3.0.0+cu130 transformers-4.45.1"
            ),
        },
        allow_install=True,
    )
    assert result.status == CompatibilityStatus.BLOCKED
    assert result.galvatron_installed is False
    assert any("torch-3.0.0" in blocker for blocker in result.blockers)


def test_blocked_when_preflight_dry_run_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _, result = _run(
        monkeypatch,
        {
            "torch_version": _FakeResult(stdout=TORCH_OK),
            "pip show galvatron": _FakeResult(exit_code=1),
            "import galvatron": _FakeResult(exit_code=1, stderr=IMPORT_NOT_FOUND),
            "pip install --dry-run": _FakeResult(exit_code=1, stderr="no such index"),
        },
        allow_install=True,
    )
    assert result.status == CompatibilityStatus.BLOCKED
    assert any("dry-run" in blocker for blocker in result.blockers)


def test_blocked_when_github_install_without_conda(monkeypatch: pytest.MonkeyPatch) -> None:
    _, result = _run(
        monkeypatch,
        {
            "torch_version": _FakeResult(stdout=TORCH_OK),
            "pip show galvatron": _FakeResult(exit_code=1),
            "import galvatron": _FakeResult(exit_code=1, stderr=IMPORT_NOT_FOUND),
        },
        conda=(None, None, None),
        allow_install=True,
        install_source="github",
    )
    assert result.status == CompatibilityStatus.BLOCKED
    assert any("Conda" in blocker for blocker in result.blockers)
    assert "https://github.com/PKU-DAIR/Hetu-Galvatron" in (
        result.proposed_install_command or ""
    )
    assert any("Conda" in action for action in result.manual_actions)


class _InstallStatefulRunner:
    """Simulates a successful official PyPI install: absent -> present."""

    def __init__(self) -> None:
        self.installed = False
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], *, timeout: float) -> ProcessResult:
        self.calls.append(command)
        rendered = " ".join(command)
        if "pip install --dry-run" in rendered:
            return _FakeResult(stdout="Would install galvatron-0.9.1 transformers-4.45.1")
        if "pip install galvatron" in rendered:
            self.installed = True
            return _FakeResult(stdout="Successfully installed galvatron-0.9.1")
        if "pip show galvatron" in rendered:
            if self.installed:
                return _FakeResult(
                    stdout="Name: galvatron\nVersion: 0.9.1\nLocation: /opt/x\n"
                )
            return _FakeResult(exit_code=1, stderr="WARNING: Package(s) not found")
        if "import galvatron" in rendered:
            if self.installed:
                return _FakeResult(
                    stdout="file: /opt/x/galvatron/__init__.py\nversion: 0.9.1"
                )
            return _FakeResult(exit_code=1, stderr=IMPORT_NOT_FOUND)
        if "torch_version" in rendered or "torch" in rendered:
            return _FakeResult(stdout=TORCH_OK)
        return _FakeResult(exit_code=1, stderr=f"unscripted: {rendered}")


def test_install_success_from_official_pypi(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _InstallStatefulRunner()
    monkeypatch.setattr(
        "shardgrid.engines.compatibility.detect_conda",
        lambda: ("/opt/conda/bin/conda", "shardgrid", "/opt/conda/envs/shardgrid"),
    )
    monkeypatch.setattr(
        "shardgrid.engines.compatibility.detect_python",
        lambda: ("/opt/conda/envs/shardgrid/bin/python", "3.12.13"),
    )
    evidence = collect_galvatron_evidence(
        python_executable="/opt/conda/envs/shardgrid/bin/python", runner=runner
    )
    result = evaluate_galvatron(evidence, allow_install=True, runner=runner)
    assert result.status == CompatibilityStatus.AVAILABLE
    assert result.galvatron_installed is True
    assert result.galvatron_version == "0.9.1"
    assert "pip install galvatron" in (result.install_command_used or "")
    assert "install.pypi" in [entry.name for entry in result.diagnostics]


def test_install_failure_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    _, result = _run(
        monkeypatch,
        {
            "torch_version": _FakeResult(stdout=TORCH_OK),
            "pip show galvatron": _FakeResult(exit_code=1),
            "import galvatron": _FakeResult(exit_code=1, stderr=IMPORT_NOT_FOUND),
            "pip install --dry-run": _FakeResult(
                stdout="Would install galvatron-0.9.1 transformers-4.45.1"
            ),
            "pip install galvatron": _FakeResult(
                exit_code=1, stderr="ERROR: No matching distribution found"
            ),
        },
        allow_install=True,
    )
    assert result.status == CompatibilityStatus.BLOCKED
    assert any("install failed" in blocker for blocker in result.blockers)
    install_diag = next(entry for entry in result.diagnostics if entry.name == "install.pypi")
    assert "No matching distribution" in install_diag.stderr_tail


def test_check_failed_without_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "shardgrid.engines.compatibility.detect_conda", lambda: (None, None, None)
    )
    monkeypatch.setattr(
        "shardgrid.engines.compatibility.detect_python", lambda: (None, None)
    )
    evidence = collect_galvatron_evidence(runner=FakeRunner({}))
    result = evaluate_galvatron(evidence)
    assert result.status == CompatibilityStatus.CHECK_FAILED
    assert any("evidence incomplete" in blocker for blocker in result.blockers)


def test_diagnostics_are_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    evidence, result = _run(
        monkeypatch,
        {
            "torch_version": _FakeResult(stdout=TORCH_OK),
            "pip show galvatron": _FakeResult(exit_code=1, stderr="WARNING: not found"),
            "import galvatron": _FakeResult(exit_code=1, stderr=IMPORT_NOT_FOUND),
        },
    )
    names = [entry.name for entry in result.diagnostics]
    assert "detect.torch" in names
    assert "detect.pip_show_galvatron" in names
    assert "detect.import_galvatron" in names
    assert result.commands
    assert any("WARNING: not found" in entry.stderr_tail for entry in result.diagnostics)
    assert any("ModuleNotFoundError" in entry.stderr_tail for entry in result.diagnostics)
    assert len(result.diagnostics) == len(evidence.diagnostics)


def test_save_and_load_evidence_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, result = _run(
        monkeypatch,
        {
            "torch_version": _FakeResult(stdout=TORCH_OK),
            "pip show galvatron": _FakeResult(stdout=PIP_SHOW_OFFICIAL),
            "import galvatron": _FakeResult(stdout=IMPORT_OK),
        },
    )
    saved = save_galvatron_evidence(result, tmp_path)
    assert saved.name.startswith("galvatron-")
    assert (tmp_path / "galvatron-latest.json").exists()
    loaded = load_galvatron_evidence(saved)
    assert loaded.run_id == result.run_id
    assert loaded.status == result.status
    assert loaded.galvatron_version == result.galvatron_version
    assert loaded.conda_environment == result.conda_environment
    assert loaded.diagnostics == result.diagnostics
    payload = json.loads(saved.read_text())
    assert payload["status"] == "AVAILABLE"


def test_spike_report_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    _, result = _run(
        monkeypatch,
        {
            "torch_version": _FakeResult(stdout=TORCH_OK),
            "pip show galvatron": _FakeResult(exit_code=1),
            "import galvatron": _FakeResult(exit_code=1, stderr=IMPORT_NOT_FOUND),
        },
    )
    report = result.to_spike_report()
    assert report.component == "galvatron"
    assert report.results == ["NOT INSTALLED"]
    assert report.versions["galvatron"] == "not_detected"
    assert report.versions["torch"] == "2.7.1+cu118"
    assert report.decision is None
    assert report.recommended_next_action is not None


def test_spike_report_blocked_requires_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    _, result = _run(
        monkeypatch,
        {
            "torch_version": _FakeResult(stdout=TORCH_OK),
            "pip show galvatron": _FakeResult(exit_code=1),
            "import galvatron": _FakeResult(exit_code=1, stderr=IMPORT_NOT_FOUND),
            "pip install --dry-run": _FakeResult(exit_code=1, stderr="no such index"),
        },
        allow_install=True,
    )
    report = result.to_spike_report()
    assert report.status.value == "blocked"
    assert report.decision is not None
    assert report.blockers


def test_preflight_parser_extracts_packages() -> None:
    packages = _parse_install_preflight(
        "Successfully installed (dry-run)\nWould install galvatron-0.9.1 "
        "transformers-4.45.1 torch-2.7.1+cu118"
    )
    assert "galvatron-0.9.1" in packages
    assert "transformers-4.45.1" in packages
    assert "torch-2.7.1+cu118" in packages
    assert _parse_install_preflight("nothing to say") == []


SETUP_PY_SAMPLE = '''
from setuptools import setup, find_packages, Extension

_deps = [
    "torch>=2.0.1",
    "torchvision>=0.15.2",
    "numpy<2.0.0",
    "transformers==4.49.0",
]

if FLASH_ATTN_INSTALL:
    _deps.append("flash-attn>=2.0.8")

setup(
    name="hetu-galvatron",
    version="2.4.1",
    python_requires=">=3.8",
    install_requires=_deps,
)
'''

PYPI_OFFICIAL_JSON = json.dumps(
    {
        "info": {
            "name": "galvatron",
            "version": "0.0.3",
            "home_page": "https://github.com/kyegomez/Galvatron",
            "project_urls": {"Homepage": "https://github.com/kyegomez/Galvatron"},
            "requires_python": "",
            "requires_dist": ["transformers", "torch"],
        }
    }
)


class _Fetch:
    def __init__(self, mapping: dict[str, str | None]) -> None:
        self.mapping = mapping

    def __call__(self, url: str, timeout: float) -> str | None:
        for pattern, text in self.mapping.items():
            if pattern in url:
                return text
        return None


def test_declared_requirements_from_official_github_setup_py() -> None:
    fetch = _Fetch(
        {
            "raw.githubusercontent.com/PKU-DAIR/Hetu-Galvatron": SETUP_PY_SAMPLE,
            "pypi.org": PYPI_OFFICIAL_JSON,
        }
    )
    requirements = collect_galvatron_declared_requirements(fetcher=fetch, timeout=5)
    assert requirements.obtained is True
    assert requirements.source is not None and "PKU-DAIR/Hetu-Galvatron" in requirements.source
    assert requirements.version == "2.4.1"
    assert requirements.python_requires == ">=3.8"
    assert requirements.torch_requirement == "torch>=2.0.1"
    assert requirements.cuda_requirement is None
    assert "torchvision>=0.15.2" in requirements.requires_dist
    assert any("GALVATRON_FLASH_ATTN_INSTALL" in item for item in requirements.diagnostics)


def test_declared_requirements_reject_unofficial_pypi() -> None:
    fetch = _Fetch(
        {
            "raw.githubusercontent.com": None,
            "pypi.org": PYPI_OFFICIAL_JSON,
        }
    )
    requirements = collect_galvatron_declared_requirements(fetcher=fetch, timeout=5)
    assert requirements.obtained is False
    assert any("NOT the official PKU-DAIR" in item for item in requirements.pypi_findings)
    assert requirements.torch_requirement is None


def test_declared_requirements_unobtainable_no_guessing() -> None:
    requirements = collect_galvatron_declared_requirements(
        fetcher=_Fetch({}), timeout=5
    )
    assert requirements.obtained is False
    assert requirements.source is None
    assert requirements.version is None
    assert requirements.python_requires is None
    assert requirements.torch_requirement is None
    assert requirements.diagnostics


def test_split_requirement_parsing() -> None:
    from shardgrid.engines.compatibility import _split_requirement

    assert _split_requirement("torch>=2.0.1") == ("torch", ">=2.0.1")
    assert _split_requirement("numpy<2.0.0") == ("numpy", "<2.0.0")
    assert _split_requirement("six>=1.15.0") == ("six", ">=1.15.0")


def test_parse_worker_version_evidence_marker() -> None:
    payload = parse_worker_version_evidence(
        'banner\nVERSION_EVIDENCE {"worker_id": "gpu4060", "python_version": "3.12.13"}\n'
    )
    assert payload == {"worker_id": "gpu4060", "python_version": "3.12.13"}
    assert parse_worker_version_evidence("nothing here") is None
    assert parse_worker_version_evidence("VERSION_EVIDENCE not-json") is None


def test_worker_version_script_injects_worker_id() -> None:
    script = build_worker_version_script(worker_id="gpu4060")
    assert 'worker_id = "gpu4060"' in script
    assert 'VERSION_EVIDENCE ' in script
    assert "galvatron_installed" in script


class _FakeWrapper:
    def __init__(self, result: ProcessResult) -> None:
        self.result = result

    def run_script(self, script: str, *, timeout: float) -> ProcessResult:
        return self.result


def test_collect_worker_version_evidence_live_payload() -> None:
    stdout = (
        "VERSION_EVIDENCE "
        + json.dumps(
            {
                "worker_id": "gpu1060",
                "conda_environment": "shardgrid",
                "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
                "python_version": "3.12.13",
                "torch_version": "2.7.1+cu118",
                "torch_cuda_version": "11.8",
                "torch_cuda_available": True,
                "gpu_name": "NVIDIA GeForce GTX 1650",
                "compute_capability": "7.5",
                "driver_version": "527.41",
                "galvatron_installed": False,
                "galvatron_source": None,
            }
        )
    )
    evidence = collect_worker_version_evidence(
        _FakeWrapper(_FakeResult(stdout=stdout)), worker_id="gpu1060"
    )
    assert evidence.evidence_status == "live"
    assert evidence.torch_version == "2.7.1+cu118"
    assert evidence.compute_capability == "7.5"
    assert evidence.galvatron_installed is False


def test_collect_worker_version_evidence_missing_marker_is_pending() -> None:
    evidence = collect_worker_version_evidence(
        _FakeWrapper(_FakeResult(stdout="nothing", stderr="boom")), worker_id="gpu1060"
    )
    assert evidence.evidence_status == "pending"
    assert any("no VERSION_EVIDENCE marker" in item for item in evidence.diagnostics)


def test_collect_worker_version_evidence_wrapper_failure_is_pending() -> None:
    class _BrokenWrapper:
        def run_script(self, script: str, *, timeout: float) -> ProcessResult:
            raise RuntimeError("ssh failed")

    evidence = collect_worker_version_evidence(_BrokenWrapper(), worker_id="gpu4060")
    assert evidence.evidence_status == "pending"
    assert any("runtime wrapper failed" in item for item in evidence.diagnostics)
