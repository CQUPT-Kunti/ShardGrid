"""Galvatron compatibility-spike harness (T054).

The harness is detect-first and reuse-first.  It checks the *selected* Conda
runtime for an already-installed official Galvatron and records complete
compatibility evidence (commands, versions, Conda/Python/PyTorch/CUDA facts,
timing, and diagnostics) without claiming full capability compatibility from a
successful import alone.

Only official sources are ever considered:

- PyPI ``galvatron`` when its recorded metadata matches the official project
- the official GitHub repository ``PKU-DAIR/Hetu-Galvatron``

The default mode is *check-only* and never changes the active backend or
environment.  Installation is opt-in (``allow_install=True``), runs only the
official command, pre-flights against destructive dependency changes, and stops
with a recorded manual action / blocker whenever the environment would be
damaged.  Full capability validation (RTX 4060 / GTX 1650 / multi-host /
profiler / search / pipeline / checkpoint) belongs to T056-T060, not here.
"""

from __future__ import annotations

import json
import re
import socket
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from packaging.specifiers import SpecifierSet

from shardgrid.common.enums import BackendStatus, FailureStage, SerializableStrEnum
from shardgrid.common.process import ProcessResult, run_process
from shardgrid.engines.models import CompatibilitySpikeReport
from shardgrid.workers.environment_report import detect_conda, detect_python

GALVATRON_OFFICIAL_GITHUB = "https://github.com/PKU-DAIR/Hetu-Galvatron"
GALVATRON_OFFICIAL_GITHUB_ORIGIN = "PKU-DAIR/Hetu-Galvatron"
GALVATRON_PYPI_PACKAGE = "galvatron"

GALVATRON_PYPI_INSTALL = "python -m pip install galvatron"
GALVATRON_GITHUB_INSTALL = (
    f"git clone {GALVATRON_OFFICIAL_GITHUB} <dir> && python -m pip install -e <dir>"
)

GALVATRON_PYPI_JSON_URL = "https://pypi.org/pypi/galvatron/json"
GALVATRON_GITHUB_SETUP_URLS = (
    "https://raw.githubusercontent.com/PKU-DAIR/Hetu-Galvatron/main/setup.py",
    "https://raw.githubusercontent.com/PKU-DAIR/Hetu-Galvatron/master/setup.py",
)

# Declared-dependency names that map to the CUDA comparison dimension.
_CUDA_REQUIREMENT_NAMES = ("cudatoolkit", "cuda-python", "nvidia-cuda")

VERSION_EVIDENCE_MARKER = "VERSION_EVIDENCE "

# Packages whose upgrade/change inside the selected environment is treated as
# destructive and therefore requires a manual action instead of automation.
_RISKY_PACKAGE_PREFIXES = ("torch", "nvidia-", "triton", "cuda-", "tensorrt")

PROBE_TIMEOUT = 30.0
INSTALL_TIMEOUT = 300.0


class CompatibilityStatus(SerializableStrEnum):
    NOT_CHECKED = "NOT CHECKED"
    AVAILABLE = "AVAILABLE"
    NOT_INSTALLED = "NOT INSTALLED"
    INCOMPATIBLE = "INCOMPATIBLE"
    BLOCKED = "BLOCKED"
    CHECK_FAILED = "CHECK FAILED"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:]


def _new_run_id() -> str:
    return f"galvatron-{uuid.uuid4().hex[:12]}"


def socket_name() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown-host"


@dataclass(frozen=True)
class CommandDiagnostic:
    """Preserved per-command evidence, including failure output tails."""

    name: str
    command: str
    exit_code: int | None
    timed_out: bool
    stdout_tail: str
    stderr_tail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


@dataclass(frozen=True)
class GalvatronEvidence:
    """Raw detection evidence collected without modifying the environment."""

    conda_executable: str | None
    conda_environment: str | None
    conda_prefix: str | None
    python_executable: str | None
    python_version: str | None
    python_probe: ProcessResult | None
    pip_show: ProcessResult | None
    import_probe: ProcessResult | None
    torch_probe: ProcessResult | None
    git_origin: ProcessResult | None
    diagnostics: tuple[CommandDiagnostic, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def pip_show_ok(self) -> bool:
        return self.pip_show is not None and self.pip_show.ok

    @property
    def import_probe_ok(self) -> bool:
        return self.import_probe is not None and self.import_probe.ok

    @property
    def import_not_found(self) -> bool:
        return self.import_probe is not None and "ModuleNotFoundError" in (
            self.import_probe.stdout + self.import_probe.stderr
        )

    @property
    def torch_ok(self) -> bool:
        return self.torch_probe is not None and self.torch_probe.ok


def _parse_pip_show(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip().lower()] = value.strip()
    return metadata


def _parse_version(pip_metadata: dict[str, str], import_stdout: str) -> str | None:
    version = pip_metadata.get("version")
    if version and version.lower() not in {"unknown", "none"}:
        return version
    for line in import_stdout.splitlines():
        if line.startswith("version:"):
            value = line.split(":", 1)[1].strip()
            if value and value.lower() != "unknown":
                return value
    return None


def _parse_import_file(import_stdout: str) -> str | None:
    for line in import_stdout.splitlines():
        if line.startswith("file:"):
            return line.split(":", 1)[1].strip()
    return None


def _classify_source(
    pip_metadata: dict[str, str],
    import_file: str | None,
    git_origin: str | None,
) -> str:
    """Return the recorded source label, or ``unofficial/unknown``."""
    if git_origin and GALVATRON_OFFICIAL_GITHUB_ORIGIN in git_origin:
        return "github:PKU-DAIR/Hetu-Galvatron"
    home_page = pip_metadata.get("home-page", "")
    if home_page:
        if GALVATRON_OFFICIAL_GITHUB_ORIGIN in home_page:
            return "github:PKU-DAIR/Hetu-Galvatron"
        return "unofficial/unknown"
    if import_file and "Hetu-Galvatron" in import_file:
        return "github:PKU-DAIR/Hetu-Galvatron (local clone, origin unverified)"
    location = pip_metadata.get("location", "")
    if location and "Hetu-Galvatron" in location:
        return "github:PKU-DAIR/Hetu-Galvatron (local clone, origin unverified)"
    if pip_metadata.get("name") == GALVATRON_PYPI_PACKAGE:
        return "pypi:galvatron"
    return "unofficial/unknown"


def _parse_install_preflight(text: str) -> list[str]:
    """Extract the package names pip would install from a dry-run report."""
    packages: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("would install"):
            rest = (
                stripped.split(":", 1)[1] if ":" in stripped else stripped[13:].strip()
            )
            for item in rest.replace(",", "").split():
                name = item.split("==")[0].split(">")[0].split("<")[0].split("[")[0]
                if name:
                    packages.append(name)
    return packages


def _risky_packages(packages: Sequence[str]) -> list[str]:
    risky: list[str] = []
    for package in packages:
        lowered = package.lower()
        if any(lowered.startswith(prefix) for prefix in _RISKY_PACKAGE_PREFIXES):
            risky.append(package)
    return risky


@dataclass(frozen=True)
class GalvatronCompatibilityResult:
    """Complete compatibility evidence for one Galvatron check run."""

    run_id: str
    status: CompatibilityStatus
    started_at: str
    elapsed_s: float
    galvatron_installed: bool
    galvatron_version: str | None
    galvatron_source: str | None
    conda_environment: str | None
    conda_prefix: str | None
    python_executable: str | None
    python_version: str | None
    torch_version: str | None
    torch_cuda_version: str | None
    torch_cuda_available: bool | None
    commands: tuple[str, ...] = ()
    diagnostics: tuple[CommandDiagnostic, ...] = ()
    blockers: tuple[str, ...] = ()
    manual_actions: tuple[str, ...] = ()
    proposed_install_command: str | None = None
    install_command_used: str | None = None
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "elapsed_s": self.elapsed_s,
            "galvatron_installed": self.galvatron_installed,
            "galvatron_version": self.galvatron_version,
            "galvatron_source": self.galvatron_source,
            "conda_environment": self.conda_environment,
            "conda_prefix": self.conda_prefix,
            "python_executable": self.python_executable,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "torch_cuda_available": self.torch_cuda_available,
            "commands": list(self.commands),
            "diagnostics": [entry.to_dict() for entry in self.diagnostics],
            "blockers": list(self.blockers),
            "manual_actions": list(self.manual_actions),
            "proposed_install_command": self.proposed_install_command,
            "install_command_used": self.install_command_used,
            "limitations": list(self.limitations),
        }

    def to_spike_report(self) -> CompatibilitySpikeReport:
        """Map this check onto the stable spike-report contract (T061)."""
        return CompatibilitySpikeReport(
            report_id=self.run_id,
            component="galvatron",
            stage=FailureStage.PROBE,
            machines_tested=[socket_name()],
            versions={
                "galvatron": self.galvatron_version or "not_detected",
                "galvatron_source": self.galvatron_source or "unknown",
                "conda_environment": self.conda_environment or "unknown",
                "conda_prefix": self.conda_prefix or "unknown",
                "python_executable": self.python_executable or "unknown",
                "python_version": self.python_version or "unknown",
                "torch": self.torch_version or "not_detected",
                "torch_cuda": self.torch_cuda_version or "not_detected",
            },
            commands=list(self.commands),
            results=[self.status.value],
            status=_spike_status(self.status),
            blockers=list(self.blockers),
            decision=(
                None
                if self.status
                in {CompatibilityStatus.AVAILABLE, CompatibilityStatus.NOT_INSTALLED}
                else f"galvatron check: {self.status.value}"
            ),
            recommended_next_action=_recommended_action(self),
            created_at=self.started_at,
        )


def _spike_status(status: CompatibilityStatus) -> BackendStatus:
    if status == CompatibilityStatus.AVAILABLE:
        return BackendStatus.AVAILABLE
    if status == CompatibilityStatus.NOT_INSTALLED:
        return BackendStatus.NOT_CHECKED
    if status == CompatibilityStatus.BLOCKED:
        return BackendStatus.BLOCKED
    return BackendStatus.FAILED


def _recommended_action(result: GalvatronCompatibilityResult) -> str | None:
    if result.status == CompatibilityStatus.NOT_INSTALLED:
        return "confirm official-source install from the recorded command"
    if result.blockers:
        return result.blockers[0]
    return None


def _run_checked(
    runner: Callable[..., ProcessResult],
    name: str,
    command: Sequence[str] | str,
    *,
    timeout: float,
    diagnostics: list[CommandDiagnostic],
    commands: list[str],
) -> ProcessResult:
    recorded = " ".join(command) if isinstance(command, list) else str(command)
    try:
        result = runner(command, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - converted into a harness failure
        diagnostics.append(
            CommandDiagnostic(
                name=name,
                command=recorded,
                exit_code=None,
                timed_out=False,
                stdout_tail="",
                stderr_tail=f"harness runner raised: {exc}",
            )
        )
        commands.append(recorded)
        raise
    commands.append(recorded)
    diagnostics.append(
        CommandDiagnostic(
            name=name,
            command=recorded,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            stdout_tail=_tail(result.stdout),
            stderr_tail=_tail(result.stderr),
        )
    )
    return result


def _safe_probe(
    runner: Callable[..., ProcessResult],
    name: str,
    command: Sequence[str] | str,
    *,
    timeout: float,
    diagnostics: list[CommandDiagnostic],
    commands: list[str],
    errors: list[str],
) -> ProcessResult | None:
    """Run one probe, recording a harness error instead of losing evidence."""
    try:
        return _run_checked(
            runner,
            name,
            command,
            timeout=timeout,
            diagnostics=diagnostics,
            commands=commands,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as CHECK FAILED evidence
        errors.append(f"{name} failed to execute: {exc}")
        return None


def collect_galvatron_evidence(
    *,
    python_executable: str | None = None,
    runner: Callable[..., ProcessResult] = run_process,
    probe_timeout: float = PROBE_TIMEOUT,
) -> GalvatronEvidence:
    """Detect Galvatron and runtime facts in the selected environment.

    Detect-first: the already-selected Conda Python is reused; nothing is
    installed, upgraded, or replaced by this step.
    """
    diagnostics: list[CommandDiagnostic] = []
    commands: list[str] = []
    errors: list[str] = []

    conda_executable, conda_environment, conda_prefix = detect_conda()
    detected_python, detected_version = detect_python()
    if python_executable is None:
        python_executable = detected_python
    python_version = detected_version if python_executable == detected_python else None

    pip_show: ProcessResult | None = None
    import_probe: ProcessResult | None = None
    torch_probe: ProcessResult | None = None
    git_origin: ProcessResult | None = None
    python_probe: ProcessResult | None = None

    if not python_executable:
        errors.append("no python executable resolvable in the selected environment")
    else:
        if python_version is None:
            python_probe = _safe_probe(
                runner,
                "detect.python_version",
                [
                    python_executable,
                    "-c",
                    "import platform; print(platform.python_version())",
                ],
                timeout=probe_timeout,
                diagnostics=diagnostics,
                commands=commands,
                errors=errors,
            )
            if python_probe is not None and python_probe.ok:
                python_version = python_probe.stdout.strip() or None

        torch_probe = _safe_probe(
            runner,
            "detect.torch",
            [
                python_executable,
                "-c",
                (
                    "import torch; "
                    "print('torch_version', torch.__version__); "
                    "print('torch_cuda_version', torch.version.cuda); "
                    "print('torch_cuda_available', torch.cuda.is_available())"
                ),
            ],
            timeout=probe_timeout,
            diagnostics=diagnostics,
            commands=commands,
            errors=errors,
        )

        pip_show = _safe_probe(
            runner,
            "detect.pip_show_galvatron",
            [python_executable, "-m", "pip", "show", "galvatron"],
            timeout=probe_timeout,
            diagnostics=diagnostics,
            commands=commands,
            errors=errors,
        )

        import_probe = _safe_probe(
            runner,
            "detect.import_galvatron",
            [
                python_executable,
                "-c",
                (
                    "import galvatron; "
                    "print('file:', galvatron.__file__); "
                    "print('version:', getattr(galvatron, '__version__', 'unknown'))"
                ),
            ],
            timeout=probe_timeout,
            diagnostics=diagnostics,
            commands=commands,
            errors=errors,
        )

        if import_probe is not None and import_probe.ok:
            import_file = _parse_import_file(import_probe.stdout)
            if import_file:
                check = _safe_probe(
                    runner,
                    "detect.git_origin",
                    ["git", "-C", import_file, "remote", "get-url", "origin"],
                    timeout=probe_timeout,
                    diagnostics=diagnostics,
                    commands=commands,
                    errors=errors,
                )
                if check is not None and check.ok:
                    git_origin = check

    return GalvatronEvidence(
        conda_executable=conda_executable,
        conda_environment=conda_environment,
        conda_prefix=conda_prefix,
        python_executable=python_executable,
        python_version=python_version,
        python_probe=python_probe,
        pip_show=pip_show,
        import_probe=import_probe,
        torch_probe=torch_probe,
        git_origin=git_origin,
        diagnostics=tuple(diagnostics),
        errors=tuple(errors),
    )


def _parse_torch_facts(stdout: str) -> tuple[str | None, str | None, bool | None]:
    torch_version: str | None = None
    cuda_version: str | None = None
    available: bool | None = None
    for line in stdout.splitlines():
        if line.startswith("torch_version "):
            torch_version = line.split(" ", 1)[1].strip() or None
        elif line.startswith("torch_cuda_version "):
            value = line.split(" ", 1)[1].strip()
            cuda_version = value or None
        elif line.startswith("torch_cuda_available "):
            value = line.split(" ", 1)[1].strip()
            available = value.lower() == "true" if value else None
    return torch_version, cuda_version, available


def _preflight_install(
    runner: Callable[..., ProcessResult],
    python_executable: str,
    *,
    source: str,
    timeout: float,
    diagnostics: list[CommandDiagnostic],
    commands: list[str],
) -> tuple[list[str], list[str]]:
    """Return (blockers, packages_would_install) for the official install."""
    blockers: list[str] = []
    if source == "pypi":
        dry_run = _run_checked(
            runner,
            "install.preflight_pypi",
            [python_executable, "-m", "pip", "install", "--dry-run", "galvatron"],
            timeout=timeout,
            diagnostics=diagnostics,
            commands=commands,
        )
        if not dry_run.ok:
            return [
                f"pip dry-run for galvatron failed: {_tail(dry_run.stderr or dry_run.stdout, 500)}"
            ], []
        packages = _parse_install_preflight(dry_run.stdout + dry_run.stderr)
        risky = _risky_packages(packages)
        if risky:
            blockers.append(
                f"install would change risky packages in the selected environment: {sorted(risky)}"
            )
        return blockers, packages

    clone_dir = tempfile.mkdtemp(prefix="galvatron-spike-")
    clone_command = ["git", "clone", "--depth", "1", GALVATRON_OFFICIAL_GITHUB, clone_dir]
    cloned = _run_checked(
        runner,
        "install.preflight_clone",
        clone_command,
        timeout=timeout,
        diagnostics=diagnostics,
        commands=commands,
    )
    if not cloned.ok:
        return [f"git clone of official Galvatron failed: {_tail(cloned.stderr, 500)}"], []
    dry_run = _run_checked(
        runner,
        "install.preflight_github",
        [python_executable, "-m", "pip", "install", "--dry-run", "-e", clone_dir],
        timeout=timeout,
        diagnostics=diagnostics,
        commands=commands,
    )
    if not dry_run.ok:
        detail = _tail(dry_run.stderr or dry_run.stdout, 500)
        return [f"pip dry-run for editable galvatron failed: {detail}"], []
    packages = _parse_install_preflight(dry_run.stdout + dry_run.stderr)
    risky = _risky_packages(packages)
    if risky:
        blockers.append(
            f"install would change risky packages in the selected environment: {sorted(risky)}"
        )
    return blockers, packages


def _install_galvatron(
    runner: Callable[..., ProcessResult],
    python_executable: str,
    *,
    source: str,
    timeout: float,
    diagnostics: list[CommandDiagnostic],
    commands: list[str],
) -> tuple[ProcessResult, str]:
    if source == "pypi":
        command = [python_executable, "-m", "pip", "install", "galvatron"]
        return (
            _run_checked(
                runner,
                "install.pypi",
                command,
                timeout=timeout,
                diagnostics=diagnostics,
                commands=commands,
            ),
            " ".join(command),
        )
    clone_dir = tempfile.mkdtemp(prefix="galvatron-spike-")
    clone_command = ["git", "clone", "--depth", "1", GALVATRON_OFFICIAL_GITHUB, clone_dir]
    cloned = _run_checked(
        runner,
        "install.clone",
        clone_command,
        timeout=timeout,
        diagnostics=diagnostics,
        commands=commands,
    )
    if not cloned.ok:
        return cloned, " ".join(clone_command)
    editable = [python_executable, "-m", "pip", "install", "-e", clone_dir]
    return (
        _run_checked(
            runner,
            "install.editable",
            editable,
            timeout=timeout,
            diagnostics=diagnostics,
            commands=commands,
        ),
        " ".join(editable),
    )


def _finish_result(
    *,
    result_id: str,
    started: str,
    status: CompatibilityStatus,
    installed: bool,
    version: str | None,
    source: str | None,
    evidence: GalvatronEvidence,
    diagnostics: list[CommandDiagnostic],
    commands: list[str],
    blockers: list[str],
    manual_actions: list[str],
    limitations: list[str],
    torch_version: str | None,
    torch_cuda_version: str | None,
    torch_cuda_available: bool | None,
    elapsed_s: float,
    proposed: str | None = None,
    install_used: str | None = None,
) -> GalvatronCompatibilityResult:
    return GalvatronCompatibilityResult(
        run_id=result_id,
        status=status,
        started_at=started,
        elapsed_s=elapsed_s,
        galvatron_installed=installed,
        galvatron_version=version,
        galvatron_source=source,
        conda_environment=evidence.conda_environment,
        conda_prefix=evidence.conda_prefix,
        python_executable=evidence.python_executable,
        python_version=evidence.python_version,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        torch_cuda_available=torch_cuda_available,
        commands=tuple(commands),
        diagnostics=tuple(diagnostics),
        blockers=tuple(blockers),
        manual_actions=tuple(manual_actions),
        proposed_install_command=proposed,
        install_command_used=install_used,
        limitations=tuple(limitations),
    )


def evaluate_galvatron(
    evidence: GalvatronEvidence,
    *,
    allow_install: bool = False,
    install_source: str = "pypi",
    runner: Callable[..., ProcessResult] = run_process,
    install_timeout: float = INSTALL_TIMEOUT,
    probe_timeout: float = PROBE_TIMEOUT,
    started_at: str | None = None,
) -> GalvatronCompatibilityResult:
    """Derive a status from collected evidence; optionally install officially."""
    started = started_at or _now()
    result_id = _new_run_id()
    diagnostics = list(evidence.diagnostics)
    commands = [entry.command for entry in diagnostics]
    blockers: list[str] = []
    manual_actions: list[str] = []
    limitations: list[str] = []

    python_executable = evidence.python_executable

    if evidence.errors or python_executable is None:
        missing = "; ".join(evidence.errors or ["no python executable"])
        blockers.append(f"evidence incomplete: {missing}")
        return _finish_result(
            result_id=result_id,
            started=started,
            status=CompatibilityStatus.CHECK_FAILED,
            installed=False,
            version=None,
            source=None,
            evidence=evidence,
            diagnostics=diagnostics,
            commands=commands,
            blockers=blockers,
            manual_actions=manual_actions,
            limitations=limitations,
            torch_version=None,
            torch_cuda_version=None,
            torch_cuda_available=None,
            elapsed_s=0.0,
            proposed=GALVATRON_PYPI_INSTALL,
        )

    torch_version: str | None = None
    torch_cuda_version: str | None = None
    torch_cuda_available: bool | None = None
    if evidence.torch_probe is not None:
        if not evidence.torch_probe.timed_out and not evidence.torch_probe.ok:
            blockers.append(
                "torch probe failed in the selected environment: "
                + _tail(evidence.torch_probe.stderr or evidence.torch_probe.stdout, 500)
            )
        torch_version, torch_cuda_version, torch_cuda_available = _parse_torch_facts(
            evidence.torch_probe.stdout
        )

    galvatron_installed = evidence.pip_show_ok or evidence.import_probe_ok
    pip_metadata = (
        _parse_pip_show(evidence.pip_show.stdout) if evidence.pip_show is not None else {}
    )
    import_file = (
        _parse_import_file(evidence.import_probe.stdout)
        if evidence.import_probe is not None
        else None
    )
    git_origin = (
        (evidence.git_origin.stdout or "").strip() if evidence.git_origin is not None else None
    )
    galvatron_version = (
        _parse_version(pip_metadata, evidence.import_probe.stdout)
        if evidence.import_probe is not None
        else None
    )
    galvatron_source = _classify_source(pip_metadata, import_file, git_origin)

    if galvatron_installed and not evidence.import_probe_ok:
        if evidence.import_not_found:
            galvatron_installed = False
        elif evidence.import_probe is not None:
            blockers.append(
                "galvatron import failed: "
                + _tail(evidence.import_probe.stderr or evidence.import_probe.stdout, 500)
            )

    if not galvatron_installed:
        if not allow_install:
            manual_actions.append(
                f"confirm official-source Galvatron install: {GALVATRON_PYPI_INSTALL} "
                f"(GitHub: {GALVATRON_GITHUB_INSTALL})"
            )
            limitations.append(
                "detection-level check only; capability validation is T056-T060"
            )
            return _finish_result(
                result_id=result_id,
                started=started,
                status=CompatibilityStatus.NOT_INSTALLED,
                installed=False,
                version=None,
                source=None,
                evidence=evidence,
                diagnostics=diagnostics,
                commands=commands,
                blockers=blockers,
                manual_actions=manual_actions,
                limitations=limitations,
                torch_version=torch_version,
                torch_cuda_version=torch_cuda_version,
                torch_cuda_available=torch_cuda_available,
                elapsed_s=0.0,
                proposed=GALVATRON_PYPI_INSTALL,
            )

        if install_source not in {"pypi", "github"}:
            blockers.append(f"unsupported install source {install_source!r}")
            return _finish_result(
                result_id=result_id,
                started=started,
                status=CompatibilityStatus.BLOCKED,
                installed=False,
                version=None,
                source=None,
                evidence=evidence,
                diagnostics=diagnostics,
                commands=commands,
                blockers=blockers,
                manual_actions=manual_actions,
                limitations=limitations,
                torch_version=torch_version,
                torch_cuda_version=torch_cuda_version,
                torch_cuda_available=torch_cuda_available,
                elapsed_s=0.0,
                proposed=GALVATRON_PYPI_INSTALL,
            )
        if evidence.conda_executable is None and install_source == "github":
            blockers.append("editable install requires Conda environment identity")
            manual_actions.append(
                "Conda executable not detected; editable Galvatron install into the "
                "selected environment requires manual confirmation"
            )
            return _finish_result(
                result_id=result_id,
                started=started,
                status=CompatibilityStatus.BLOCKED,
                installed=False,
                version=None,
                source=None,
                evidence=evidence,
                diagnostics=diagnostics,
                commands=commands,
                blockers=blockers,
                manual_actions=manual_actions,
                limitations=limitations,
                torch_version=torch_version,
                torch_cuda_version=torch_cuda_version,
                torch_cuda_available=torch_cuda_available,
                elapsed_s=0.0,
                proposed=GALVATRON_GITHUB_INSTALL,
            )

        preflight_blockers, _ = _preflight_install(
            runner,
            python_executable,
            source=install_source,
            timeout=install_timeout,
            diagnostics=diagnostics,
            commands=commands,
        )
        if preflight_blockers:
            blockers.extend(preflight_blockers)
            manual_actions.append(
                "resolve the install blocker manually or choose a compatible environment"
            )
            return _finish_result(
                result_id=result_id,
                started=started,
                status=CompatibilityStatus.BLOCKED,
                installed=False,
                version=None,
                source=None,
                evidence=evidence,
                diagnostics=diagnostics,
                commands=commands,
                blockers=blockers,
                manual_actions=manual_actions,
                limitations=limitations,
                torch_version=torch_version,
                torch_cuda_version=torch_cuda_version,
                torch_cuda_available=torch_cuda_available,
                elapsed_s=0.0,
                proposed=(
                    GALVATRON_PYPI_INSTALL
                    if install_source == "pypi"
                    else GALVATRON_GITHUB_INSTALL
                ),
            )

        install_result, install_command = _install_galvatron(
            runner,
            python_executable,
            source=install_source,
            timeout=install_timeout,
            diagnostics=diagnostics,
            commands=commands,
        )
        if not install_result.ok:
            blockers.append(
                "official install failed: "
                + _tail(install_result.stderr or install_result.stdout, 500)
            )
            manual_actions.append(
                "fix the install failure manually; never patch Galvatron source"
            )
            return _finish_result(
                result_id=result_id,
                started=started,
                status=CompatibilityStatus.BLOCKED,
                installed=False,
                version=None,
                source=None,
                evidence=evidence,
                diagnostics=diagnostics,
                commands=commands,
                blockers=blockers,
                manual_actions=manual_actions,
                limitations=limitations,
                torch_version=torch_version,
                torch_cuda_version=torch_cuda_version,
                torch_cuda_available=torch_cuda_available,
                elapsed_s=0.0,
                proposed=GALVATRON_PYPI_INSTALL,
                install_used=install_command,
            )

        redetect = collect_galvatron_evidence(
            python_executable=python_executable,
            runner=runner,
            probe_timeout=probe_timeout,
        )
        diagnostics.extend(redetect.diagnostics)
        commands.extend(entry.command for entry in redetect.diagnostics)
        if not (redetect.pip_show_ok or redetect.import_probe_ok):
            blockers.append("install completed but post-install detection failed")
            manual_actions.append("inspect the selected environment manually")
            return _finish_result(
                result_id=result_id,
                started=started,
                status=CompatibilityStatus.BLOCKED,
                installed=False,
                version=None,
                source=None,
                evidence=evidence,
                diagnostics=diagnostics,
                commands=commands,
                blockers=blockers,
                manual_actions=manual_actions,
                limitations=limitations,
                torch_version=torch_version,
                torch_cuda_version=torch_cuda_version,
                torch_cuda_available=torch_cuda_available,
                elapsed_s=0.0,
                proposed=GALVATRON_PYPI_INSTALL,
                install_used=install_command,
            )
        new_metadata = (
            _parse_pip_show(redetect.pip_show.stdout)
            if redetect.pip_show is not None
            else {}
        )
        new_import_file = (
            _parse_import_file(redetect.import_probe.stdout)
            if redetect.import_probe is not None
            else None
        )
        new_git_origin = (
            (redetect.git_origin.stdout or "").strip()
            if redetect.git_origin is not None
            else None
        )
        new_version = (
            _parse_version(new_metadata, redetect.import_probe.stdout)
            if redetect.import_probe is not None
            else None
        )
        new_source = _classify_source(new_metadata, new_import_file, new_git_origin)
        if new_source == "unofficial/unknown":
            blockers.append("installed Galvatron source could not be verified as official")
            return _finish_result(
                result_id=result_id,
                started=started,
                status=CompatibilityStatus.BLOCKED,
                installed=True,
                version=new_version,
                source=new_source,
                evidence=evidence,
                diagnostics=diagnostics,
                commands=commands,
                blockers=blockers,
                manual_actions=manual_actions,
                limitations=limitations,
                torch_version=torch_version,
                torch_cuda_version=torch_cuda_version,
                torch_cuda_available=torch_cuda_available,
                elapsed_s=0.0,
                install_used=install_command,
            )
        if not redetect.torch_ok:
            blockers.append("post-install torch probe failed in the selected environment")
        if blockers:
            return _finish_result(
                result_id=result_id,
                started=started,
                status=CompatibilityStatus.INCOMPATIBLE,
                installed=True,
                version=new_version,
                source=new_source,
                evidence=evidence,
                diagnostics=diagnostics,
                commands=commands,
                blockers=blockers,
                manual_actions=manual_actions,
                limitations=limitations,
                torch_version=torch_version,
                torch_cuda_version=torch_cuda_version,
                torch_cuda_available=torch_cuda_available,
                elapsed_s=0.0,
                install_used=install_command,
            )
        limitations.append(
            "AVAILABLE is detection-level (import + version + runtime evidence); "
            "full capability validation is T056-T060"
        )
        return _finish_result(
            result_id=result_id,
            started=started,
            status=CompatibilityStatus.AVAILABLE,
            installed=True,
            version=new_version,
            source=new_source,
            evidence=evidence,
            diagnostics=diagnostics,
            commands=commands,
            blockers=blockers,
            manual_actions=manual_actions,
            limitations=limitations,
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
            torch_cuda_available=torch_cuda_available,
            elapsed_s=0.0,
            install_used=install_command,
        )

    # Installed: classify present-but-broken vs usable.
    if galvatron_source == "unofficial/unknown":
        blockers.append("Galvatron source could not be verified as official")
        manual_actions.append("replace with the official PyPI/GitHub source")
        return _finish_result(
            result_id=result_id,
            started=started,
            status=CompatibilityStatus.BLOCKED,
            installed=True,
            version=galvatron_version,
            source=galvatron_source,
            evidence=evidence,
            diagnostics=diagnostics,
            commands=commands,
            blockers=blockers,
            manual_actions=manual_actions,
            limitations=limitations,
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
            torch_cuda_available=torch_cuda_available,
            elapsed_s=0.0,
        )

    if blockers:
        return _finish_result(
            result_id=result_id,
            started=started,
            status=CompatibilityStatus.INCOMPATIBLE,
            installed=True,
            version=galvatron_version,
            source=galvatron_source,
            evidence=evidence,
            diagnostics=diagnostics,
            commands=commands,
            blockers=blockers,
            manual_actions=manual_actions,
            limitations=limitations,
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
            torch_cuda_available=torch_cuda_available,
            elapsed_s=0.0,
        )

    if not evidence.torch_ok or torch_version is None:
        blockers.append("selected environment has no importable PyTorch")
        return _finish_result(
            result_id=result_id,
            started=started,
            status=CompatibilityStatus.INCOMPATIBLE,
            installed=True,
            version=galvatron_version,
            source=galvatron_source,
            evidence=evidence,
            diagnostics=diagnostics,
            commands=commands,
            blockers=blockers,
            manual_actions=manual_actions,
            limitations=limitations,
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
            torch_cuda_available=torch_cuda_available,
            elapsed_s=0.0,
        )

    limitations.append(
        "AVAILABLE is detection-level (import + version + runtime evidence); "
        "full capability validation is T056-T060"
    )
    return _finish_result(
        result_id=result_id,
        started=started,
        status=CompatibilityStatus.AVAILABLE,
        installed=True,
        version=galvatron_version,
        source=galvatron_source,
        evidence=evidence,
        diagnostics=diagnostics,
        commands=commands,
        blockers=blockers,
        manual_actions=manual_actions,
        limitations=limitations,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        torch_cuda_available=torch_cuda_available,
        elapsed_s=0.0,
    )


def run_galvatron_check(
    *,
    python_executable: str | None = None,
    allow_install: bool = False,
    install_source: str = "pypi",
    runner: Callable[..., ProcessResult] = run_process,
    probe_timeout: float = PROBE_TIMEOUT,
    install_timeout: float = INSTALL_TIMEOUT,
) -> GalvatronCompatibilityResult:
    """Run one complete Galvatron compatibility check (check-only by default)."""
    started = _now()
    start = time.monotonic()
    result_id = _new_run_id()
    try:
        evidence = collect_galvatron_evidence(
            python_executable=python_executable,
            runner=runner,
            probe_timeout=probe_timeout,
        )
        evaluated = evaluate_galvatron(
            evidence,
            allow_install=allow_install,
            install_source=install_source,
            runner=runner,
            install_timeout=install_timeout,
            probe_timeout=probe_timeout,
            started_at=started,
        )
    except Exception as exc:  # noqa: BLE001 - harness-level failure
        elapsed = time.monotonic() - start
        return GalvatronCompatibilityResult(
            run_id=result_id,
            status=CompatibilityStatus.CHECK_FAILED,
            started_at=started,
            elapsed_s=round(elapsed, 3),
            galvatron_installed=False,
            galvatron_version=None,
            galvatron_source=None,
            conda_environment=None,
            conda_prefix=None,
            python_executable=python_executable,
            python_version=None,
            torch_version=None,
            torch_cuda_version=None,
            torch_cuda_available=None,
            blockers=(f"harness failed: {exc}",),
            manual_actions=("inspect harness diagnostics before retrying",),
        )
    elapsed = time.monotonic() - start
    return GalvatronCompatibilityResult(
        run_id=evaluated.run_id,
        status=evaluated.status,
        started_at=evaluated.started_at,
        elapsed_s=round(elapsed, 3),
        galvatron_installed=evaluated.galvatron_installed,
        galvatron_version=evaluated.galvatron_version,
        galvatron_source=evaluated.galvatron_source,
        conda_environment=evaluated.conda_environment,
        conda_prefix=evaluated.conda_prefix,
        python_executable=evaluated.python_executable,
        python_version=evaluated.python_version,
        torch_version=evaluated.torch_version,
        torch_cuda_version=evaluated.torch_cuda_version,
        torch_cuda_available=evaluated.torch_cuda_available,
        commands=evaluated.commands,
        diagnostics=evaluated.diagnostics,
        blockers=evaluated.blockers,
        manual_actions=evaluated.manual_actions,
        proposed_install_command=evaluated.proposed_install_command,
        install_command_used=evaluated.install_command_used,
        limitations=evaluated.limitations,
    )


def save_galvatron_evidence(
    result: GalvatronCompatibilityResult,
    output_dir: str | Path,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["timestamp"] = _now()
    path = directory / f"galvatron-{result.run_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    (directory / "galvatron-latest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    return path


def load_galvatron_evidence(path: str | Path) -> GalvatronCompatibilityResult:
    data = json.loads(Path(path).read_text())
    return GalvatronCompatibilityResult(
        run_id=str(data["run_id"]),
        status=CompatibilityStatus(str(data["status"])),
        started_at=str(data["started_at"]),
        elapsed_s=float(data.get("elapsed_s", 0.0)),
        galvatron_installed=bool(data.get("galvatron_installed", False)),
        galvatron_version=data.get("galvatron_version"),
        galvatron_source=data.get("galvatron_source"),
        conda_environment=data.get("conda_environment"),
        conda_prefix=data.get("conda_prefix"),
        python_executable=data.get("python_executable"),
        python_version=data.get("python_version"),
        torch_version=data.get("torch_version"),
        torch_cuda_version=data.get("torch_cuda_version"),
        torch_cuda_available=data.get("torch_cuda_available"),
        commands=tuple(str(item) for item in data.get("commands", [])),
        diagnostics=tuple(
            CommandDiagnostic(
                name=str(entry["name"]),
                command=str(entry["command"]),
                exit_code=entry.get("exit_code"),
                timed_out=bool(entry.get("timed_out", False)),
                stdout_tail=str(entry.get("stdout_tail", "")),
                stderr_tail=str(entry.get("stderr_tail", "")),
            )
            for entry in data.get("diagnostics", [])
        ),
        blockers=tuple(str(item) for item in data.get("blockers", [])),
        manual_actions=tuple(str(item) for item in data.get("manual_actions", [])),
        proposed_install_command=data.get("proposed_install_command"),
        install_command_used=data.get("install_command_used"),
        limitations=tuple(str(item) for item in data.get("limitations", [])),
    )


class ComparisonStatus(SerializableStrEnum):
    MATCH = "MATCH"
    VERSION_MISMATCH = "VERSION MISMATCH"
    REQUIREMENT_UNKNOWN = "REQUIREMENT UNKNOWN"
    NOT_INSTALLED = "NOT INSTALLED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class GalvatronDeclaredRequirements:
    """Declared Galvatron requirements collected from official sources only."""

    source: str | None
    version: str | None
    python_requires: str | None
    requires_dist: tuple[str, ...]
    torch_requirement: str | None
    cuda_requirement: str | None
    obtained: bool
    pypi_findings: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "version": self.version,
            "python_requires": self.python_requires,
            "requires_dist": list(self.requires_dist),
            "torch_requirement": self.torch_requirement,
            "cuda_requirement": self.cuda_requirement,
            "obtained": self.obtained,
            "pypi_findings": list(self.pypi_findings),
            "diagnostics": list(self.diagnostics),
        }


def _fetch_text(url: str, timeout: float) -> str | None:
    request = urllib.request.Request(url, headers={"User-Agent": "shardgrid-t055/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _split_requirement(requirement: str) -> tuple[str, str | None]:
    match = re.match(r"^([A-Za-z0-9._-]+)\s*(.*)$", requirement.strip())
    if not match:
        return requirement.strip(), None
    name, spec = match.group(1), match.group(2).strip()
    return name, spec or None


def _extract_setup_py(text: str) -> tuple[str | None, str | None, str | None, list[str]]:
    name_match = re.search(r'name\s*=\s*"([^"]+)"', text)
    version_match = re.search(r'version\s*=\s*"([^"]+)"', text)
    python_match = re.search(r'python_requires\s*=\s*"([^"]+)"', text)
    deps: list[str] = []
    deps_match = re.search(r"_deps\s*=\s*\[(.*?)\]\n", text, re.DOTALL)
    if deps_match:
        deps = re.findall(r'"([^"]+)"', deps_match.group(1))
    return (
        name_match.group(1) if name_match else None,
        version_match.group(1) if version_match else None,
        python_match.group(1) if python_match else None,
        deps,
    )


def _extract_declared_dimensions(
    requires_dist: Sequence[str],
) -> tuple[str | None, str | None, tuple[str, ...]]:
    torch_requirement: str | None = None
    cuda_requirement: str | None = None
    other: list[str] = []
    for requirement in requires_dist:
        name, _ = _split_requirement(requirement)
        if name == "torch" and torch_requirement is None:
            torch_requirement = requirement
        elif name in _CUDA_REQUIREMENT_NAMES or name.startswith("nvidia-cuda"):
            cuda_requirement = requirement
        else:
            other.append(requirement)
    return torch_requirement, cuda_requirement, tuple(other)


def collect_galvatron_declared_requirements(
    *,
    fetcher: Callable[[str, float], str | None] | None = None,
    timeout: float = 20.0,
) -> GalvatronDeclaredRequirements:
    """Collect Galvatron declared requirements from official sources.

    Official sources only: the official GitHub ``PKU-DAIR/Hetu-Galvatron``
    setup metadata first, then the PyPI JSON API only when its recorded
    metadata matches the official project.  When no official information can be
    obtained, ``obtained=False`` is reported and nothing is guessed.
    """
    fetch = fetcher or _fetch_text
    diagnostics: list[str] = []
    pypi_findings: list[str] = []
    python_requires: str | None = None
    requires_dist: list[str] = []
    source: str | None = None
    version: str | None = None

    setup_text: str | None = None
    for url in GALVATRON_GITHUB_SETUP_URLS:
        setup_text = fetch(url, timeout)
        if setup_text is not None:
            break
    if setup_text is not None:
        name, setup_version, setup_python, deps = _extract_setup_py(setup_text)
        if name and "galvatron" in name.lower() and deps:
            source = f"github:PKU-DAIR/Hetu-Galvatron (setup.py, {setup_version or 'unknown'})"
            version = setup_version
            python_requires = setup_python
            requires_dist = deps
            if "FLASH_ATTN_INSTALL" in setup_text:
                diagnostics.append(
                    "setup.py declares conditional deps via GALVATRON_FLASH_ATTN_INSTALL"
                    " (flash-attn>=2.0.8, packaging); recorded as declared conditional"
                )
        else:
            diagnostics.append(
                "official GitHub setup.py fetched but no parseable Galvatron metadata"
            )
    else:
        diagnostics.append("official GitHub setup.py could not be fetched")

    pypi_text = fetch(GALVATRON_PYPI_JSON_URL, timeout)
    pypi: dict[str, Any] | None = None
    if pypi_text is not None:
        try:
            payload = json.loads(pypi_text)
        except ValueError:
            payload = None
        pypi = payload if isinstance(payload, dict) else None
    if pypi is not None:
        info = pypi.get("info") or {}
        pypi_name = str(info.get("name") or "")
        pypi_home = str(info.get("home_page") or "")
        project_urls = {
            str(key).lower(): str(value)
            for key, value in (info.get("project_urls") or {}).items()
        }
        is_official_pypi = GALVATRON_OFFICIAL_GITHUB_ORIGIN in pypi_home or any(
            GALVATRON_OFFICIAL_GITHUB_ORIGIN in value for value in project_urls.values()
        )
        if not is_official_pypi:
            pypi_findings.append(
                f"PyPI package {pypi_name!r} (version {info.get('version')}) is NOT the "
                f"official PKU-DAIR project (home_page={pypi_home or 'none'}); "
                "PyPI install is rejected as unofficial"
            )
        elif python_requires is None:
            source = "pypi:galvatron"
            version = str(info.get("version") or None)
            python_requires = str(info.get("requires_python") or None) or None
            requires_dist = [
                str(item) for item in (info.get("requires_dist") or []) if item
            ]
    else:
        pypi_findings.append("PyPI JSON API could not be fetched")

    torch_requirement, cuda_requirement, other = _extract_declared_dimensions(
        requires_dist
    )
    return GalvatronDeclaredRequirements(
        source=source,
        version=version,
        python_requires=python_requires,
        requires_dist=tuple(requires_dist),
        torch_requirement=torch_requirement,
        cuda_requirement=cuda_requirement,
        obtained=source is not None and python_requires is not None,
        pypi_findings=tuple(pypi_findings),
        diagnostics=tuple(diagnostics),
    )


@dataclass(frozen=True)
class WorkerVersionEvidence:
    """Real runtime evidence for one physical GPU Worker (WSL selected Conda)."""

    worker_id: str
    physical_os: str | None = None
    runtime_os: str | None = None
    conda_environment: str | None = None
    conda_prefix: str | None = None
    python_executable: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    torch_cuda_available: bool | None = None
    driver_version: str | None = None
    gpu_name: str | None = None
    compute_capability: str | None = None
    galvatron_installed: bool = False
    galvatron_version: str | None = None
    galvatron_source: str | None = None
    diagnostics: tuple[str, ...] = ()
    evidence_status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "physical_os": self.physical_os,
            "runtime_os": self.runtime_os,
            "conda_environment": self.conda_environment,
            "conda_prefix": self.conda_prefix,
            "python_executable": self.python_executable,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "torch_cuda_available": self.torch_cuda_available,
            "driver_version": self.driver_version,
            "gpu_name": self.gpu_name,
            "compute_capability": self.compute_capability,
            "galvatron_installed": self.galvatron_installed,
            "galvatron_version": self.galvatron_version,
            "galvatron_source": self.galvatron_source,
            "diagnostics": list(self.diagnostics),
            "evidence_status": self.evidence_status,
        }


@dataclass(frozen=True)
class VersionComparisonItem:
    component: str
    status: ComparisonStatus
    requirement: str | None = None
    actual: str | None = None
    detail: str | None = None
    impact: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "status": self.status.value,
            "requirement": self.requirement,
            "actual": self.actual,
            "detail": self.detail,
            "impact": self.impact,
        }


@dataclass(frozen=True)
class WorkerVersionComparison:
    worker_id: str
    status: ComparisonStatus
    items: tuple[VersionComparisonItem, ...]
    mismatches: tuple[VersionComparisonItem, ...]
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "status": self.status.value,
            "items": [item.to_dict() for item in self.items],
            "mismatches": [item.to_dict() for item in self.mismatches],
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class GalvatronVersionComparison:
    run_id: str
    created_at: str
    requirements: GalvatronDeclaredRequirements
    workers: tuple[WorkerVersionComparison, ...]
    overall_status: ComparisonStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "requirements": self.requirements.to_dict(),
            "workers": [worker.to_dict() for worker in self.workers],
            "overall_status": self.overall_status.value,
        }

    def to_spike_report(self) -> CompatibilitySpikeReport:
        return CompatibilitySpikeReport(
            report_id=self.run_id,
            component="galvatron-versions",
            stage=FailureStage.PROBE,
            machines_tested=[worker.worker_id for worker in self.workers],
            versions={
                "galvatron_requirement_source": self.requirements.source or "unknown",
                "galvatron_requirement_version": self.requirements.version or "unknown",
                "galvatron_python_requires": self.requirements.python_requires or "unknown",
                "galvatron_torch_requirement": self.requirements.torch_requirement
                or "unknown",
                "galvatron_cuda_requirement": self.requirements.cuda_requirement
                or "unknown",
            },
            commands=[],
            results=[self.overall_status.value],
            status=_spike_status_for_comparison(self.overall_status),
            blockers=self._blockers(),
            decision=(
                None
                if self.overall_status
                in {ComparisonStatus.MATCH, ComparisonStatus.NOT_INSTALLED}
                else f"galvatron version comparison: {self.overall_status.value}"
            ),
            recommended_next_action=self._recommended_action(),
            created_at=self.created_at,
        )

    def _blockers(self) -> list[str]:
        blockers: list[str] = []
        for worker in self.workers:
            for item in worker.mismatches:
                blockers.append(
                    f"{worker.worker_id}/{item.component}: {item.status.value} "
                    f"(requirement={item.requirement!r}, actual={item.actual!r})"
                )
        return blockers

    def _recommended_action(self) -> str | None:
        if self.overall_status == ComparisonStatus.VERSION_MISMATCH:
            return "record mismatch details; do not auto-upgrade the environment"
        if self.overall_status == ComparisonStatus.REQUIREMENT_UNKNOWN:
            return "obtain official Galvatron declared requirements and re-check"
        if self.overall_status == ComparisonStatus.NOT_INSTALLED:
            return "no official Galvatron installed on the Workers; T056 install spike decides"
        if self.overall_status == ComparisonStatus.BLOCKED:
            return "fix the blocked Worker evidence and re-check"
        return None


def _spike_status_for_comparison(status: ComparisonStatus) -> BackendStatus:
    if status == ComparisonStatus.MATCH:
        return BackendStatus.AVAILABLE
    if status == ComparisonStatus.NOT_INSTALLED:
        return BackendStatus.NOT_CHECKED
    if status == ComparisonStatus.BLOCKED:
        return BackendStatus.BLOCKED
    return BackendStatus.FAILED


def _requirement_satisfied(
    requirement: str | None, actual: str | None
) -> tuple[bool, str | None]:
    if not requirement or not actual:
        return False, "requirement or actual value missing"
    _, specifier_text = _split_requirement(requirement)
    specifier_text = specifier_text or requirement
    try:
        specifier = SpecifierSet(specifier_text, prereleases=True)
    except ValueError as exc:
        return False, f"declared requirement {requirement!r} is not parseable: {exc}"
    try:
        return specifier.contains(actual, prereleases=True), None
    except ValueError as exc:
        return False, f"actual version {actual!r} is not parseable: {exc}"


def compare_galvatron_versions(
    requirements: GalvatronDeclaredRequirements,
    workers: Sequence[WorkerVersionEvidence | Mapping[str, Any]],
) -> GalvatronVersionComparison:
    """Compare declared requirements against real per-Worker runtime evidence.

    Pure decision logic: no network, no environment changes, no guessing.
    """
    now = _now()
    comparisons: list[WorkerVersionComparison] = []

    for worker in workers:
        evidence = (
            worker
            if isinstance(worker, WorkerVersionEvidence)
            else WorkerVersionEvidence(
                worker_id=str(worker.get("worker_id") or "unknown"),
                physical_os=_opt_str(worker.get("physical_os")),
                runtime_os=_opt_str(worker.get("runtime_os")),
                conda_environment=_opt_str(worker.get("conda_environment")),
                conda_prefix=_opt_str(worker.get("conda_prefix")),
                python_executable=_opt_str(worker.get("python_executable")),
                python_version=_opt_str(worker.get("python_version")),
                torch_version=_opt_str(worker.get("torch_version")),
                torch_cuda_version=_opt_str(worker.get("torch_cuda_version")),
                torch_cuda_available=worker.get("torch_cuda_available"),
                driver_version=_opt_str(worker.get("driver_version")),
                gpu_name=_opt_str(worker.get("gpu_name")),
                compute_capability=_opt_str(worker.get("compute_capability")),
                galvatron_installed=bool(worker.get("galvatron_installed", False)),
                galvatron_version=_opt_str(worker.get("galvatron_version")),
                galvatron_source=_opt_str(worker.get("galvatron_source")),
                diagnostics=tuple(
                    str(item) for item in worker.get("diagnostics", [])
                ),
                evidence_status=str(worker.get("evidence_status") or "pending"),
            )
        )
        comparisons.append(_compare_worker(evidence, requirements))

    if not comparisons:
        overall = ComparisonStatus.BLOCKED
    elif any(c.status == ComparisonStatus.BLOCKED for c in comparisons):
        overall = ComparisonStatus.BLOCKED
    elif any(c.status == ComparisonStatus.VERSION_MISMATCH for c in comparisons):
        overall = ComparisonStatus.VERSION_MISMATCH
    elif any(c.status == ComparisonStatus.REQUIREMENT_UNKNOWN for c in comparisons):
        overall = ComparisonStatus.REQUIREMENT_UNKNOWN
    elif all(c.status == ComparisonStatus.NOT_INSTALLED for c in comparisons):
        overall = ComparisonStatus.NOT_INSTALLED
    else:
        overall = ComparisonStatus.MATCH

    return GalvatronVersionComparison(
        run_id=f"galvatron-versions-{uuid.uuid4().hex[:12]}",
        created_at=now,
        requirements=requirements,
        workers=tuple(comparisons),
        overall_status=overall,
    )


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _compare_worker(
    evidence: WorkerVersionEvidence, requirements: GalvatronDeclaredRequirements
) -> WorkerVersionComparison:
    if evidence.evidence_status != "live" or evidence.worker_id == "unknown":
        return WorkerVersionComparison(
            worker_id=evidence.worker_id,
            status=ComparisonStatus.BLOCKED,
            items=(
                VersionComparisonItem(
                    component="evidence",
                    status=ComparisonStatus.BLOCKED,
                    detail="worker runtime evidence missing or not live",
                    impact="cannot compare versions without real Worker evidence",
                ),
            ),
            mismatches=(
                VersionComparisonItem(
                    component="evidence",
                    status=ComparisonStatus.BLOCKED,
                    detail="worker runtime evidence missing or not live",
                    impact="cannot compare versions without real Worker evidence",
                ),
            ),
            diagnostics=evidence.diagnostics,
        )

    items: list[VersionComparisonItem] = []
    mismatches: list[VersionComparisonItem] = []

    installed_item = VersionComparisonItem(
        component="galvatron",
        status=(
            ComparisonStatus.MATCH
            if evidence.galvatron_installed
            else ComparisonStatus.NOT_INSTALLED
        ),
        requirement="official source only",
        actual=(
            f"installed {evidence.galvatron_version or 'unknown'} "
            f"({evidence.galvatron_source or 'unknown'})"
            if evidence.galvatron_installed
            else "not installed"
        ),
        impact=(
            None
            if evidence.galvatron_installed
            else "Galvatron is not installed in this WSL Conda training runtime"
        ),
    )
    items.append(installed_item)

    for component, requirement, actual in (
        ("python", requirements.python_requires, evidence.python_version),
        ("pytorch", requirements.torch_requirement, evidence.torch_version),
        ("cuda", requirements.cuda_requirement, evidence.torch_cuda_version),
    ):
        if requirement is None:
            items.append(
                VersionComparisonItem(
                    component=component,
                    status=ComparisonStatus.REQUIREMENT_UNKNOWN,
                    actual=actual,
                    detail="no explicit declared requirement; actual value recorded as fact",
                    impact="cannot verify this dimension without a declared requirement",
                )
            )
            continue
        if actual is None:
            item = VersionComparisonItem(
                component=component,
                status=ComparisonStatus.BLOCKED,
                requirement=requirement,
                actual=actual,
                detail="Worker evidence for this component is missing",
                impact="version comparison impossible without the actual value",
            )
            items.append(item)
            mismatches.append(item)
            continue
        satisfied, error = _requirement_satisfied(requirement, actual)
        if error is not None:
            item = VersionComparisonItem(
                component=component,
                status=ComparisonStatus.REQUIREMENT_UNKNOWN,
                requirement=requirement,
                actual=actual,
                detail=f"comparison could not be evaluated: {error}",
                impact="neither match nor mismatch can be claimed",
            )
            items.append(item)
            continue
        if satisfied:
            items.append(
                VersionComparisonItem(
                    component=component,
                    status=ComparisonStatus.MATCH,
                    requirement=requirement,
                    actual=actual,
                )
            )
        else:
            item = VersionComparisonItem(
                component=component,
                status=ComparisonStatus.VERSION_MISMATCH,
                requirement=requirement,
                actual=actual,
                detail=(
                    f"declared {requirement!r} is not satisfied by actual "
                    f"{actual!r}"
                ),
                impact="declared and actual versions differ; no environment change is made",
            )
            items.append(item)
            mismatches.append(item)

    items.append(
        VersionComparisonItem(
            component="dependencies",
            status=ComparisonStatus.REQUIREMENT_UNKNOWN,
            requirement=(
                "; ".join(requirements.requires_dist) if requirements.requires_dist else None
            ),
            actual="worker dependency versions not collected in T055",
            detail="declared deps recorded verbatim; per-package Worker evidence is T056+",
            impact="not verified and not guessed",
        )
    )

    if mismatches:
        worker_status = ComparisonStatus.VERSION_MISMATCH
    elif any(
        item.status == ComparisonStatus.BLOCKED for item in items
    ):
        worker_status = ComparisonStatus.BLOCKED
    elif not requirements.obtained or any(
        item.component in {"python", "pytorch"}
        and item.status == ComparisonStatus.REQUIREMENT_UNKNOWN
        for item in items
    ):
        worker_status = ComparisonStatus.REQUIREMENT_UNKNOWN
    elif not evidence.galvatron_installed:
        worker_status = ComparisonStatus.NOT_INSTALLED
    else:
        worker_status = ComparisonStatus.MATCH

    return WorkerVersionComparison(
        worker_id=evidence.worker_id,
        status=worker_status,
        items=tuple(items),
        mismatches=tuple(mismatches),
        diagnostics=evidence.diagnostics,
    )


_WORKER_VERSION_SCRIPT = """
import json
import os
import platform
import subprocess
import sys

worker_id = "__WORKER_ID__"
out = {
    "worker_id": worker_id,
    "physical_os": "windows",
    "runtime_os": "wsl2_linux",
    "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
    "conda_prefix": os.environ.get("CONDA_PREFIX"),
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "torch_version": None,
    "torch_cuda_version": None,
    "torch_cuda_available": None,
    "gpu_name": None,
    "compute_capability": None,
    "driver_version": None,
    "galvatron_installed": False,
    "galvatron_version": None,
    "galvatron_source": None,
    "diagnostics": [],
    "error": None,
}
try:
    import torch

    out["torch_version"] = torch.__version__
    out["torch_cuda_version"] = str(torch.version.cuda)
    out["torch_cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.cuda.current_device()
        out["gpu_name"] = torch.cuda.get_device_name(device)
        out["compute_capability"] = ".".join(
            str(part) for part in torch.cuda.get_device_capability(device)
        )
except Exception as exc:
    out["error"] = str(exc)
    out["diagnostics"].append("torch probe failed: %s" % exc)
try:
    import galvatron

    out["galvatron_installed"] = True
    out["galvatron_version"] = getattr(galvatron, "__version__", None)
    out["galvatron_source"] = galvatron.__file__
except ModuleNotFoundError:
    pass
except Exception as exc:
    out["diagnostics"].append("galvatron import failed: %s" % exc)
try:
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
        timeout=30,
    )
    lines = [line.strip() for line in smi.splitlines() if line.strip()]
    out["driver_version"] = lines[0] if lines else None
except Exception as exc:
    out["diagnostics"].append("nvidia-smi failed: %s" % exc)
print("VERSION_EVIDENCE " + json.dumps(out, sort_keys=True))
"""


def build_worker_version_script(*, worker_id: str) -> str:
    return _WORKER_VERSION_SCRIPT.replace("__WORKER_ID__", worker_id)


def parse_worker_version_evidence(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(VERSION_EVIDENCE_MARKER):
            try:
                payload = json.loads(stripped.split(" ", 1)[1])
            except ValueError:
                return None
            if isinstance(payload, dict):
                return payload
    return None


def _classify_worker_galvatron_source(file_path: str | None) -> str | None:
    if not file_path:
        return None
    if GALVATRON_OFFICIAL_GITHUB_ORIGIN in file_path or "Hetu-Galvatron" in file_path:
        return "github:PKU-DAIR/Hetu-Galvatron (local clone)"
    return "pypi:galvatron (unverified)"


def collect_worker_version_evidence(
    wrapper: Any,
    *,
    worker_id: str,
    timeout: float = 120.0,
) -> WorkerVersionEvidence:
    """Collect real runtime evidence from a Worker's WSL selected Conda runtime.

    Reuses the existing SSH + WSL runtime wrapper (T040) and probe conventions
    (T041); no transport, WSL, Conda, or GPU probing is reimplemented here.
    """
    script = build_worker_version_script(worker_id=worker_id)
    try:
        result = wrapper.run_script(script, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - surfaced as BLOCKED evidence
        return WorkerVersionEvidence(
            worker_id=worker_id,
            diagnostics=(f"runtime wrapper failed: {exc}",),
            evidence_status="pending",
        )
    payload = parse_worker_version_evidence(result.stdout)
    if payload is None:
        return WorkerVersionEvidence(
            worker_id=worker_id,
            diagnostics=(
                "no VERSION_EVIDENCE marker in runtime output",
                f"stdout tail: {result.stdout[-500:]}",
                f"stderr tail: {result.stderr[-500:]}",
            ),
            evidence_status="pending",
        )
    payload.setdefault("physical_os", "windows")
    payload.setdefault("runtime_os", "wsl2_linux")
    galvatron_source = _classify_worker_galvatron_source(payload.get("galvatron_source"))
    return WorkerVersionEvidence(
        worker_id=str(payload.get("worker_id") or worker_id),
        physical_os=_opt_str(payload.get("physical_os")),
        runtime_os=_opt_str(payload.get("runtime_os")),
        conda_environment=_opt_str(payload.get("conda_environment")),
        conda_prefix=_opt_str(payload.get("conda_prefix")),
        python_executable=_opt_str(payload.get("python_executable")),
        python_version=_opt_str(payload.get("python_version")),
        torch_version=_opt_str(payload.get("torch_version")),
        torch_cuda_version=_opt_str(payload.get("torch_cuda_version")),
        torch_cuda_available=payload.get("torch_cuda_available"),
        driver_version=_opt_str(payload.get("driver_version")),
        gpu_name=_opt_str(payload.get("gpu_name")),
        compute_capability=_opt_str(payload.get("compute_capability")),
        galvatron_installed=bool(payload.get("galvatron_installed", False)),
        galvatron_version=_opt_str(payload.get("galvatron_version")),
        galvatron_source=galvatron_source,
        diagnostics=tuple(str(item) for item in payload.get("diagnostics", [])),
        evidence_status="live",
    )


def save_galvatron_version_comparison(
    comparison: GalvatronVersionComparison,
    output_dir: str | Path,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = comparison.to_dict()
    payload["timestamp"] = _now()
    path = directory / f"{comparison.run_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    (directory / "galvatron-versions-latest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    return path
