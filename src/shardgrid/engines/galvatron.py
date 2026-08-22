"""Galvatron single-GPU compatibility workload (T056/T057).

The same minimal workload runs independently on each physical GPU Worker
(RTX 4060 -> T056, GTX 1650 -> T057) through the real chain
Windows -> WSL2 Ubuntu -> selected Conda environment -> Python/PyTorch/CUDA
-> Galvatron, reusing ``SSHTransport`` / ``WSLRuntimeWrapper``.

Workload steps (real, minimal, no fake substitutes):

- environment detection (Conda, Python, PyTorch, CUDA, driver, GPU)
- Galvatron detect-first: reuse an already installed official Galvatron;
  otherwise install from the official source only
  (``https://github.com/PKU-DAIR/Hetu-Galvatron``), pre-flighting against
  destructive dependency changes and verifying the installed torch version
  is unchanged
- official import chain: ``import galvatron`` plus
  ``galvatron.core.profiler`` (HardwareProfiler / RuntimeProfiler)
- CUDA visibility and real GPU identity on the Worker
- minimal profiler/runtime path supported by the current version: the
  official ``profile_hardware.py`` entry with ``num_gpus_per_node=1``, which
  generates the profiling scripts and then really runs
  ``profile_overlap.py`` through ``torchrun`` on the Worker's GPU

flash-attn is a conditional dependency (``GALVATRON_FLASH_ATTN_INSTALL``);
it is only handled when the workload explicitly requires it, so the minimal
workload records it as SKIPPED instead of forcing an install.

Each Worker is judged independently: PASS only from real execution on that
Worker, otherwise FAIL or BLOCKED with preserved evidence.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from shardgrid.common.enums import SerializableStrEnum
from shardgrid.common.process import ProcessResult

GALVATRON_OFFICIAL_REPO = "https://github.com/PKU-DAIR/Hetu-Galvatron"
GALVATRON_OFFICIAL_REF = "v2.4.0"
SPIKE_EVIDENCE_MARKER = "GALVATRON_SPIKE_EVIDENCE "

SPIKE_STATUS_PASS = "PASS"
SPIKE_STATUS_FAIL = "FAIL"
SPIKE_STATUS_BLOCKED = "BLOCKED"

# Would-change prefixes that must never be silently upgraded in the selected
# Conda environment.  Exact names (torch, torchvision, torchaudio) are matched
# separately because they pin the installed torch build.
_RISKY_PREFIXES = ("nvidia-", "triton", "cuda-", "tensorrt")
_RISKY_EXACT = ("torch", "torchvision", "torchaudio")

# Packages the minimal compatibility workload really needs beyond torch, all
# compatible with the existing torch build (used only when the default pip
# resolution would change the installed torch/CUDA stack).
MINIMAL_WORKLOAD_DEPS = (
    "numpy<2.0.0",
    "scipy>=1.10.1",
    "hydra-core",
    "pydantic",
    "omegaconf",
    "packaging",
)
DEFAULT_GIT_PROBE_TIMEOUT = 25.0
DEFAULT_CLONE_ATTEMPTS = 2

PROFILE_HARDWARE_ENTRY = (
    "profile_hardware.py",
    "scripts/profile_hardware.yaml",
    "num_gpus_per_node=1",
    "max_tp_size=1",
    "max_pp_deg=1",
)


class SpikeStepStatus(SerializableStrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail(text: str, limit: int = 6000) -> str:
    return text[-limit:]


def _new_run_id() -> str:
    return f"galvatron-spike-{uuid.uuid4().hex[:10]}"


def _galvatron_checkout_dir() -> str:
    return f"$HOME/galvatron-spike-{GALVATRON_OFFICIAL_REF}"


def _evidence_slug(worker_id: str, expected_gpu: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", expected_gpu.lower())
    if normalized:
        return normalized
    return re.sub(r"[^a-z0-9]+", "", worker_id.lower()) or worker_id.lower()


def _network_env(proxy_url: str | None) -> dict[str, str] | None:
    if not proxy_url:
        return None
    return {
        "http_proxy": proxy_url,
        "https_proxy": proxy_url,
        "HTTP_PROXY": proxy_url,
        "HTTPS_PROXY": proxy_url,
        "all_proxy": proxy_url,
        "ALL_PROXY": proxy_url,
    }


@dataclass(frozen=True)
class GalvatronSpikeStep:
    name: str
    status: SpikeStepStatus
    detail: str | None = None
    output_tail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
            "output_tail": self.output_tail,
        }


@dataclass(frozen=True)
class GalvatronSpikeResult:
    run_id: str
    worker_id: str
    expected_gpu: str
    status: str
    started_at: str
    elapsed_s: float
    install_mode: str = "none"
    official_source: str | None = None
    requested_ref: str | None = None
    resolved_commit: str | None = None
    galvatron_version: str | None = None
    galvatron_source: str | None = None
    repo_dir: str | None = None
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
    gpu_matched: bool | None = None
    steps: tuple[GalvatronSpikeStep, ...] = ()
    diagnostics: tuple[str, ...] = ()
    manual_actions: tuple[str, ...] = ()
    stdout_tail: str | None = None
    stderr_tail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "worker_id": self.worker_id,
            "expected_gpu": self.expected_gpu,
            "status": self.status,
            "started_at": self.started_at,
            "elapsed_s": self.elapsed_s,
            "install_mode": self.install_mode,
            "official_source": self.official_source,
            "requested_ref": self.requested_ref,
            "resolved_commit": self.resolved_commit,
            "galvatron_version": self.galvatron_version,
            "galvatron_source": self.galvatron_source,
            "repo_dir": self.repo_dir,
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
            "gpu_matched": self.gpu_matched,
            "steps": [step.to_dict() for step in self.steps],
            "diagnostics": list(self.diagnostics),
            "manual_actions": list(self.manual_actions),
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


_SPIKE_SCRIPT = """
import json
import re
import os
import platform
import subprocess
import sys

worker_id = "__WORKER_ID__"
expected_gpu = "__EXPECTED_GPU__"
out = {
    "worker_id": worker_id,
    "expected_gpu": expected_gpu,
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
    "galvatron_import_error": None,
    "galvatron_submodules_ok": False,
    "diagnostics": [],
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
    out["diagnostics"].append("torch probe failed: %s" % exc)
try:
    import galvatron

    out["galvatron_installed"] = True
    out["galvatron_source"] = galvatron.__file__
except ModuleNotFoundError:
    pass
except Exception as exc:
    out["galvatron_import_error"] = str(exc)
    out["diagnostics"].append("galvatron import failed: %s" % exc)
if out["galvatron_installed"]:
    try:
        from importlib.metadata import PackageNotFoundError, version as _pkg_version

        try:
            out["galvatron_version"] = _pkg_version("hetu-galvatron")
        except PackageNotFoundError:
            try:
                out["galvatron_version"] = _pkg_version("galvatron")
            except PackageNotFoundError:
                pass
    except Exception as exc:
        out["diagnostics"].append("galvatron version lookup failed: %s" % exc)
    try:
        from galvatron.core.profiler import (
            HardwareProfiler,
            RuntimeProfiler,
        )

        out["galvatron_submodules_ok"] = True
    except Exception as exc:
        out["diagnostics"].append(
            "galvatron.core.profiler import failed: %s" % exc
        )
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
print("GALVATRON_SPIKE_EVIDENCE " + json.dumps(out, sort_keys=True))
"""


def build_spike_evidence_script(*, worker_id: str, expected_gpu: str) -> str:
    script = _SPIKE_SCRIPT.replace("__WORKER_ID__", worker_id)
    return script.replace("__EXPECTED_GPU__", expected_gpu)


def parse_spike_evidence(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(SPIKE_EVIDENCE_MARKER):
            try:
                payload = json.loads(stripped.split(" ", 1)[1])
            except ValueError:
                return None
            if isinstance(payload, dict):
                return payload
    return None


_PIP_NAME_VERSION = re.compile(
    r"^(?P<name>[A-Za-z0-9._-]+?)-(?P<version>[0-9][A-Za-z0-9._+-]*)$"
)


def _parse_would_install(text: str) -> list[tuple[str, str | None]]:
    entries: list[tuple[str, str | None]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("would install"):
            rest = (
                stripped.split(":", 1)[1] if ":" in stripped else stripped[13:].strip()
            )
            for item in rest.replace(",", "").split():
                match = _PIP_NAME_VERSION.match(item.strip())
                if match:
                    entries.append((match.group("name"), match.group("version")))
                else:
                    entries.append((item.strip(), None))
    return entries


def preflight_install_blockers(
    dry_run_output: str,
    *,
    current_torch_version: str | None,
) -> list[str]:
    """Return blockers when the official install would change torch-family
    or CUDA-stack packages in the selected environment."""
    blockers: list[str] = []
    would_install = _parse_would_install(dry_run_output)
    for name, version in would_install:
        lowered = name.lower()
        if lowered in _RISKY_EXACT or lowered.startswith(_RISKY_PREFIXES):
            blockers.append(
                f"install would change {name} ({version or 'version unknown'}) "
                "in the selected environment"
            )
    if blockers and current_torch_version:
        for name, version in would_install:
            if name.lower() == "torch":
                blockers.append(
                    f"torch would change from {current_torch_version} to {version}"
                )
    return blockers


def derive_spike_status(steps: Sequence[GalvatronSpikeStep]) -> str:
    statuses = [step.status for step in steps]
    if SpikeStepStatus.BLOCKED in statuses:
        return SPIKE_STATUS_BLOCKED
    if SpikeStepStatus.FAIL in statuses:
        return SPIKE_STATUS_FAIL
    return SPIKE_STATUS_PASS


def _extract_missing_module(error_text: str | None) -> str | None:
    """Return the top-level module name from a ModuleNotFoundError text."""
    if not error_text:
        return None
    match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_text)
    if not match:
        return None
    return match.group(1).split(".")[0]


def _is_forbidden_dependency(module: str) -> bool:
    lowered = module.lower()
    return lowered in _RISKY_EXACT or lowered.startswith(_RISKY_PREFIXES)


def _backfill_missing_python_deps(
    *,
    run_script: Any,
    run_command: Any,
    install_python: str,
    payload: dict[str, Any],
    evidence_timeout: float,
    install_timeout: float,
    diagnostics: list[str],
    max_attempts: int = 5,
) -> tuple[dict[str, Any], bool, list[str]]:
    """Install missing plain-Python modules required by the Galvatron import
    chain and recheck after each install.

    Missing Python dependencies are installed and the test continues.  A
    missing torch/CUDA-stack module (or any non-Python, compile/system-level
    failure) is never auto-installed; the caller stops at that layer.
    """
    current = payload
    installed_modules: list[str] = []
    worker_id = str(payload.get("worker_id") or "worker")
    expected_gpu = str(payload.get("expected_gpu") or "")
    for _ in range(max_attempts):
        if current.get("galvatron_installed") is True:
            return current, True, installed_modules
        error = current.get("galvatron_import_error")
        module = _extract_missing_module(str(error) if error else None)
        if not module:
            return current, False, installed_modules
        if _is_forbidden_dependency(module):
            diagnostics.append(
                f"missing module {module!r} is part of the torch/CUDA stack; "
                "not auto-installing (compile/system layer)"
            )
            return current, False, installed_modules
        diagnostics.append(
            f"auto-installing missing Python dependency {module!r} "
            f"(reason: {str(error)[:120]})"
        )
        result = run_command(
            ["sh", "-c", f"{install_python} -m pip install {module}"],
            install_timeout,
            networked=False,
        )
        if result is None or not result.ok:
            diagnostics.append(
                f"auto-install of missing Python dependency {module!r} failed: "
                + _tail((result.stderr if result is not None else "") or "", 500)
            )
            return current, False, installed_modules
        installed_modules.append(module)
        recheck = run_script(
            build_spike_evidence_script(
                worker_id=worker_id, expected_gpu=expected_gpu
            ),
            evidence_timeout,
        )
        if recheck is None:
            diagnostics.append("recheck after dependency install could not run")
            return current, False, installed_modules
        current = parse_spike_evidence(recheck.stdout) or {}
    return current, current.get("galvatron_installed") is True, installed_modules


def _resolve_checkout_commit(
    run_command: Any,
    *,
    repo_dir: str,
    install_timeout: float,
) -> str | None:
    commit = run_command(
        ["sh", "-c", f"cd {repo_dir} && git rev-parse HEAD"],
        install_timeout,
    )
    if commit is None or not commit.ok:
        return None
    value = commit.stdout.strip()
    return value or None


def run_galvatron_spike(
    wrapper: Any,
    *,
    worker_id: str,
    expected_gpu: str,
    install: bool = True,
    evidence_timeout: float = 120.0,
    install_timeout: float = 900.0,
    profile_timeout: float = 900.0,
    git_probe_timeout: float = DEFAULT_GIT_PROBE_TIMEOUT,
    clone_attempts: int = DEFAULT_CLONE_ATTEMPTS,
    proxy_url: str | None = None,
) -> GalvatronSpikeResult:
    """Run the complete single-GPU Galvatron compatibility workload.

    The wrapper must already route Windows -> WSL2 -> selected Conda
    (``WSLRuntimeWrapper``).  Every step result is recorded; the final status
    is derived only from the real per-Worker evidence.
    """
    started = _now()
    start = time.monotonic()
    run_id = _new_run_id()
    steps: list[GalvatronSpikeStep] = []
    diagnostics: list[str] = []
    manual_actions: list[str] = []

    def add_step(
        name: str,
        status: SpikeStepStatus,
        detail: str | None = None,
        output_tail: str | None = None,
    ) -> None:
        steps.append(
            GalvatronSpikeStep(
                name=name, status=status, detail=detail, output_tail=output_tail
            )
        )

    def run_script(script: str, timeout: float) -> ProcessResult | None:
        try:
            return wrapper.run_script(script, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - surfaced as BLOCKED evidence
            diagnostics.append(f"runtime wrapper failed for {worker_id}: {exc}")
            return None

    def run_command(
        command: Sequence[str] | str,
        timeout: float,
        *,
        networked: bool = False,
    ) -> ProcessResult | None:
        try:
            return wrapper.run(
                command,
                timeout=timeout,
                env=_network_env(proxy_url) if networked else None,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as BLOCKED evidence
            diagnostics.append(f"runtime wrapper failed for {worker_id}: {exc}")
            return None

    payload: dict[str, Any] = {}
    official_source = GALVATRON_OFFICIAL_REPO
    requested_ref = GALVATRON_OFFICIAL_REF
    resolved_commit: str | None = None
    checkout_dir = _galvatron_checkout_dir()
    evidence_result = run_script(
        build_spike_evidence_script(worker_id=worker_id, expected_gpu=expected_gpu),
        evidence_timeout,
    )
    if evidence_result is None:
        add_step(
            "env_detect",
            SpikeStepStatus.BLOCKED,
            "evidence script could not be executed",
        )
        return GalvatronSpikeResult(
            run_id=run_id,
            worker_id=worker_id,
            expected_gpu=expected_gpu,
            status=SPIKE_STATUS_BLOCKED,
            started_at=started,
            elapsed_s=round(time.monotonic() - start, 3),
            steps=tuple(steps),
            diagnostics=tuple(diagnostics),
            manual_actions=tuple(manual_actions),
            stdout_tail=None,
            stderr_tail=None,
        )
    payload = parse_spike_evidence(evidence_result.stdout) or {}
    if not payload:
        add_step(
            "env_detect",
            SpikeStepStatus.BLOCKED,
            "no GALVATRON_SPIKE_EVIDENCE marker in runtime output",
            _tail(evidence_result.stdout) or _tail(evidence_result.stderr),
        )
        return GalvatronSpikeResult(
            run_id=run_id,
            worker_id=worker_id,
            expected_gpu=expected_gpu,
            status=SPIKE_STATUS_BLOCKED,
            started_at=started,
            elapsed_s=round(time.monotonic() - start, 3),
            steps=tuple(steps),
            diagnostics=tuple(diagnostics),
            manual_actions=tuple(manual_actions),
            stdout_tail=_tail(evidence_result.stdout),
            stderr_tail=_tail(evidence_result.stderr),
        )

    add_step(
        "env_detect",
        SpikeStepStatus.PASS,
        (
            f"conda={payload.get('conda_environment') or 'unknown'} "
            f"python={payload.get('python_version') or 'unknown'} "
            f"torch={payload.get('torch_version') or 'unknown'} "
            f"cuda={payload.get('torch_cuda_version') or 'unknown'}"
        ),
    )

    cuda_ok = payload.get("torch_cuda_available") is True
    add_step(
        "cuda_visibility",
        SpikeStepStatus.PASS if cuda_ok else SpikeStepStatus.FAIL,
        (
            f"torch.cuda.is_available()={payload.get('torch_cuda_available')} "
            f"torch_cuda={payload.get('torch_cuda_version')}"
        ),
    )

    gpu_name = payload.get("gpu_name")
    gpu_matched = bool(gpu_name and expected_gpu.lower() in str(gpu_name).lower())
    add_step(
        "gpu_identity",
        SpikeStepStatus.PASS if gpu_matched else SpikeStepStatus.FAIL,
        f"expected {expected_gpu!r}, detected {gpu_name!r} "
        f"cap={payload.get('compute_capability')}",
    )

    galvatron_installed = payload.get("galvatron_installed") is True
    galvatron_version: str | None = payload.get("galvatron_version")
    galvatron_source: str | None = payload.get("galvatron_source")
    install_mode = "reused"
    repo_dir: str | None = None

    if galvatron_installed:
        add_step(
            "galvatron_detect",
            SpikeStepStatus.PASS,
            f"already installed version={galvatron_version or 'unknown'} "
            f"source={galvatron_source or 'unknown'}",
        )
    else:
        add_step(
            "galvatron_detect",
            SpikeStepStatus.PASS,
            "not installed; official-source install required",
        )
        if not install:
            add_step(
                "galvatron_install",
                SpikeStepStatus.BLOCKED,
                "Galvatron absent and install disabled",
            )
            return GalvatronSpikeResult(
                run_id=run_id,
                worker_id=worker_id,
                expected_gpu=expected_gpu,
                status=SPIKE_STATUS_BLOCKED,
                started_at=started,
                elapsed_s=round(time.monotonic() - start, 3),
                install_mode="none",
                official_source=official_source,
                requested_ref=requested_ref,
                galvatron_version=galvatron_version,
                galvatron_source=galvatron_source,
                conda_environment=payload.get("conda_environment"),
                conda_prefix=payload.get("conda_prefix"),
                python_executable=payload.get("python_executable"),
                python_version=payload.get("python_version"),
                torch_version=payload.get("torch_version"),
                torch_cuda_version=payload.get("torch_cuda_version"),
                torch_cuda_available=payload.get("torch_cuda_available"),
                driver_version=payload.get("driver_version"),
                gpu_name=gpu_name,
                compute_capability=payload.get("compute_capability"),
                gpu_matched=gpu_matched,
                steps=tuple(steps),
                diagnostics=tuple(diagnostics),
            manual_actions=tuple(manual_actions),
                stdout_tail=_tail(evidence_result.stdout),
                stderr_tail=_tail(evidence_result.stderr),
            )

        install_mode = "installed"
        repo_dir = checkout_dir
        exists_check = run_command(
            [
                "sh",
                "-c",
                f"test -f {checkout_dir}/galvatron/site_package/megatron/"
                "core/datasets/helpers.cpp",
            ],
            install_timeout,
        )
        if exists_check is not None and exists_check.ok:
            diagnostics.append(
                "reusing existing official Galvatron checkout "
                f"({checkout_dir}, ref {requested_ref}) with site_package"
            )
        else:
            probe = run_command(
                [
                    "git",
                    "ls-remote",
                    "--tags",
                    GALVATRON_OFFICIAL_REPO,
                    requested_ref,
                ],
                git_probe_timeout,
                networked=True,
            )
            if probe is None or not probe.ok:
                add_step(
                    "galvatron_install",
                    SpikeStepStatus.BLOCKED,
                    "official Galvatron repo not reachable from this Worker",
                    _tail(
                        (probe.stderr if probe is not None else "")
                        or (probe.stdout if probe is not None else ""),
                    ),
                )
                diagnostics.append(
                    "git ls-remote preflight failed before clone; "
                    "skipping long clone timeout path"
                )
                return GalvatronSpikeResult(
                    run_id=run_id,
                    worker_id=worker_id,
                    expected_gpu=expected_gpu,
                    status=SPIKE_STATUS_BLOCKED,
                    started_at=started,
                    elapsed_s=round(time.monotonic() - start, 3),
                    install_mode="failed",
                    official_source=official_source,
                    requested_ref=requested_ref,
                    repo_dir=repo_dir,
                    conda_environment=payload.get("conda_environment"),
                    conda_prefix=payload.get("conda_prefix"),
                    python_executable=payload.get("python_executable"),
                    python_version=payload.get("python_version"),
                    torch_version=payload.get("torch_version"),
                    torch_cuda_version=payload.get("torch_cuda_version"),
                    torch_cuda_available=payload.get("torch_cuda_available"),
                    driver_version=payload.get("driver_version"),
                    gpu_name=gpu_name,
                    compute_capability=payload.get("compute_capability"),
                    gpu_matched=gpu_matched,
                    steps=tuple(steps),
                    diagnostics=tuple(diagnostics),
                    manual_actions=tuple(manual_actions),
                    stdout_tail=_tail(probe.stdout if probe is not None else ""),
                    stderr_tail=_tail(probe.stderr if probe is not None else ""),
                )
            clone: ProcessResult | None = None
            for attempt in range(1, clone_attempts + 1):
                clone = run_command(
                    [
                        "sh",
                        "-c",
                        "git clone --depth 1 "
                        f"--branch {requested_ref} "
                        f"{GALVATRON_OFFICIAL_REPO} {checkout_dir}",
                    ],
                    install_timeout,
                    networked=True,
                )
                if clone is not None and clone.ok:
                    break
                diagnostics.append(
                    f"git clone attempt {attempt}/{clone_attempts} failed: "
                    + _tail((clone.stderr if clone is not None else "") or "", 300)
                )
            if clone is None or not clone.ok:
                add_step(
                    "galvatron_install",
                    SpikeStepStatus.BLOCKED,
                    "git clone of the official Galvatron repo failed after "
                    f"{clone_attempts} attempts",
                    _tail(clone.stderr if clone is not None else ""),
                )
                return GalvatronSpikeResult(
                    run_id=run_id,
                    worker_id=worker_id,
                    expected_gpu=expected_gpu,
                    status=SPIKE_STATUS_BLOCKED,
                    started_at=started,
                    elapsed_s=round(time.monotonic() - start, 3),
                    install_mode="failed",
                    official_source=official_source,
                    requested_ref=requested_ref,
                    repo_dir=repo_dir,
                    conda_environment=payload.get("conda_environment"),
                    conda_prefix=payload.get("conda_prefix"),
                    python_executable=payload.get("python_executable"),
                    python_version=payload.get("python_version"),
                    torch_version=payload.get("torch_version"),
                    torch_cuda_version=payload.get("torch_cuda_version"),
                    torch_cuda_available=payload.get("torch_cuda_available"),
                    driver_version=payload.get("driver_version"),
                    gpu_name=gpu_name,
                    compute_capability=payload.get("compute_capability"),
                    gpu_matched=gpu_matched,
                    steps=tuple(steps),
                    diagnostics=tuple(diagnostics),
            manual_actions=tuple(manual_actions),
                    stdout_tail=_tail(clone.stdout if clone is not None else ""),
                    stderr_tail=_tail(clone.stderr if clone is not None else ""),
                )

        resolved_commit = _resolve_checkout_commit(
            run_command,
            repo_dir=repo_dir,
            install_timeout=install_timeout,
        )
        install_python = str(payload.get("python_executable") or "python")
        dry_run = run_command(
            [
                "sh",
                "-c",
                f"{install_python} -m pip install --dry-run -e "
                f"{repo_dir}",
            ],
            install_timeout,
            networked=True,
        )
        blockers: list[str] = []
        if dry_run is None or not dry_run.ok:
            blockers.append(
                "pip dry-run for official Galvatron failed: "
                + _tail((dry_run.stderr if dry_run is not None else "") or "", 500)
            )
        else:
            blockers.extend(
                preflight_install_blockers(
                    dry_run.stdout + dry_run.stderr,
                    current_torch_version=payload.get("torch_version"),
                )
            )
        if blockers:
            diagnostics.append(
                "default resolution would change installed packages; "
                "falling back to constrained minimal install: "
                + "; ".join(blockers)
            )
            # Reuse-first: never replace the installed PyTorch/CUDA stack.
            # Install Galvatron without deps and add only the packages the
            # minimal workload actually needs, all compatible with the
            # existing torch build.
            no_deps = run_command(
                [
                    "sh",
                    "-c",
                    f"{install_python} -m pip install --no-deps "
                    f"{repo_dir}",
                ],
                install_timeout,
                networked=True,
            )
            if no_deps is None or not no_deps.ok:
                combined = (no_deps.stdout if no_deps is not None else "") + (
                    no_deps.stderr if no_deps is not None else ""
                )
                if any(
                    marker in combined
                    for marker in ("Building wheel", "g++", "gcc", "error: command")
                ):
                    diagnostics.append(
                        "install failed while building the official "
                        "galvatron_dp_core extension; a C++ compiler is required"
                    )
                    manual_actions.append(
                        "install build-essential (g++/gcc) in the WSL2 runtime "
                        "on this Worker (requires admin; manual action), "
                        "then re-run the spike"
                    )
                add_step(
                    "galvatron_install",
                    SpikeStepStatus.BLOCKED,
                    "official Galvatron install failed (--no-deps): "
                    + _tail(
                        (no_deps.stderr if no_deps is not None else "") or "", 500
                    ),
                )
                return GalvatronSpikeResult(
                    run_id=run_id,
                    worker_id=worker_id,
                    expected_gpu=expected_gpu,
                    status=SPIKE_STATUS_BLOCKED,
                    started_at=started,
                    elapsed_s=round(time.monotonic() - start, 3),
                    install_mode="failed",
                    official_source=official_source,
                    requested_ref=requested_ref,
                    resolved_commit=resolved_commit,
                    repo_dir=repo_dir,
                    conda_environment=payload.get("conda_environment"),
                    conda_prefix=payload.get("conda_prefix"),
                    python_executable=payload.get("python_executable"),
                    python_version=payload.get("python_version"),
                    torch_version=payload.get("torch_version"),
                    torch_cuda_version=payload.get("torch_cuda_version"),
                    torch_cuda_available=payload.get("torch_cuda_available"),
                    driver_version=payload.get("driver_version"),
                    gpu_name=gpu_name,
                    compute_capability=payload.get("compute_capability"),
                    gpu_matched=gpu_matched,
                    steps=tuple(steps),
                    diagnostics=tuple(diagnostics),
            manual_actions=tuple(manual_actions),
                    stdout_tail=_tail(no_deps.stdout if no_deps is not None else ""),
                    stderr_tail=_tail(no_deps.stderr if no_deps is not None else ""),
                )
            deps_result = run_command(
                [install_python, "-m", "pip", "install", *MINIMAL_WORKLOAD_DEPS],
                install_timeout,
                networked=True,
            )
            if deps_result is None or not deps_result.ok:
                add_step(
                    "galvatron_install",
                    SpikeStepStatus.BLOCKED,
                    "minimal workload dependency install failed: "
                    + _tail(
                        (deps_result.stderr if deps_result is not None else "")
                        or "",
                        500,
                    ),
                )
                return GalvatronSpikeResult(
                    run_id=run_id,
                    worker_id=worker_id,
                    expected_gpu=expected_gpu,
                    status=SPIKE_STATUS_BLOCKED,
                    started_at=started,
                    elapsed_s=round(time.monotonic() - start, 3),
                    install_mode="failed",
                    official_source=official_source,
                    requested_ref=requested_ref,
                    resolved_commit=resolved_commit,
                    repo_dir=repo_dir,
                    conda_environment=payload.get("conda_environment"),
                    conda_prefix=payload.get("conda_prefix"),
                    python_executable=payload.get("python_executable"),
                    python_version=payload.get("python_version"),
                    torch_version=payload.get("torch_version"),
                    torch_cuda_version=payload.get("torch_cuda_version"),
                    torch_cuda_available=payload.get("torch_cuda_available"),
                    driver_version=payload.get("driver_version"),
                    gpu_name=gpu_name,
                    compute_capability=payload.get("compute_capability"),
                    gpu_matched=gpu_matched,
                    steps=tuple(steps),
                    diagnostics=tuple(diagnostics),
            manual_actions=tuple(manual_actions),
                    stdout_tail=(
                        _tail(deps_result.stdout if deps_result is not None else "")
                    ),
                    stderr_tail=(
                        _tail(deps_result.stderr if deps_result is not None else "")
                    ),
                )
            install_mode = "installed-minimal"
            add_step(
                "galvatron_install",
                SpikeStepStatus.PASS,
                "installed from official source with constrained minimal deps "
                "(default resolution would have changed the torch stack)",
                _tail(no_deps.stdout),
            )
            recheck = run_script(
                build_spike_evidence_script(
                    worker_id=worker_id, expected_gpu=expected_gpu
                ),
                evidence_timeout,
            )
            if recheck is None:
                add_step(
                    "galvatron_install",
                    SpikeStepStatus.BLOCKED,
                    "install completed but post-install detection could not run",
                )
                return GalvatronSpikeResult(
                    run_id=run_id,
                    worker_id=worker_id,
                    expected_gpu=expected_gpu,
                    status=SPIKE_STATUS_BLOCKED,
                    started_at=started,
                    elapsed_s=round(time.monotonic() - start, 3),
                    install_mode=install_mode,
                    official_source=official_source,
                    requested_ref=requested_ref,
                    resolved_commit=resolved_commit,
                    repo_dir=repo_dir,
                    conda_environment=payload.get("conda_environment"),
                    conda_prefix=payload.get("conda_prefix"),
                    python_executable=payload.get("python_executable"),
                    python_version=payload.get("python_version"),
                    torch_version=payload.get("torch_version"),
                    torch_cuda_version=payload.get("torch_cuda_version"),
                    torch_cuda_available=payload.get("torch_cuda_available"),
                    driver_version=payload.get("driver_version"),
                    gpu_name=gpu_name,
                    compute_capability=payload.get("compute_capability"),
                    gpu_matched=gpu_matched,
                    steps=tuple(steps),
                    diagnostics=tuple(diagnostics),
            manual_actions=tuple(manual_actions),
                )
            recheck_payload = parse_spike_evidence(recheck.stdout) or {}
            new_torch = recheck_payload.get("torch_version")
            if new_torch != payload.get("torch_version"):
                add_step(
                    "galvatron_install",
                    SpikeStepStatus.BLOCKED,
                    f"torch changed after install: "
                    f"{payload.get('torch_version')} -> {new_torch}",
                )
                return GalvatronSpikeResult(
                    run_id=run_id,
                    worker_id=worker_id,
                    expected_gpu=expected_gpu,
                    status=SPIKE_STATUS_BLOCKED,
                    started_at=started,
                    elapsed_s=round(time.monotonic() - start, 3),
                    install_mode="blocked",
                    official_source=official_source,
                    requested_ref=requested_ref,
                    resolved_commit=resolved_commit,
                    repo_dir=repo_dir,
                    conda_environment=payload.get("conda_environment"),
                    conda_prefix=payload.get("conda_prefix"),
                    python_executable=payload.get("python_executable"),
                    python_version=payload.get("python_version"),
                    torch_version=payload.get("torch_version"),
                    torch_cuda_version=payload.get("torch_cuda_version"),
                    torch_cuda_available=payload.get("torch_cuda_available"),
                    driver_version=payload.get("driver_version"),
                    gpu_name=gpu_name,
                    compute_capability=payload.get("compute_capability"),
                    gpu_matched=gpu_matched,
                    steps=tuple(steps),
                    diagnostics=tuple(diagnostics),
            manual_actions=tuple(manual_actions),
                )
            payload = recheck_payload
            galvatron_installed = payload.get("galvatron_installed") is True
            galvatron_version = payload.get("galvatron_version")
            galvatron_source = payload.get("galvatron_source")
            payload, galvatron_installed, backfilled = _backfill_missing_python_deps(
                run_script=run_script,
                run_command=run_command,
                install_python=install_python,
                payload=payload,
                evidence_timeout=evidence_timeout,
                install_timeout=install_timeout,
                diagnostics=diagnostics,
            )
            galvatron_version = payload.get("galvatron_version")
            galvatron_source = payload.get("galvatron_source")
            if backfilled:
                add_step(
                    "galvatron_install",
                    SpikeStepStatus.PASS,
                    "auto-installed missing Python deps and verified import: "
                    + ", ".join(backfilled),
                )
            if not galvatron_installed:
                add_step(
                    "galvatron_install",
                    SpikeStepStatus.BLOCKED,
                    "install completed but Galvatron still not importable: "
                    + str(payload.get("galvatron_import_error") or "unknown error"),
                )
                return GalvatronSpikeResult(
                    run_id=run_id,
                    worker_id=worker_id,
                    expected_gpu=expected_gpu,
                    status=SPIKE_STATUS_BLOCKED,
                    started_at=started,
                    elapsed_s=round(time.monotonic() - start, 3),
                    install_mode=install_mode,
                    official_source=official_source,
                    requested_ref=requested_ref,
                    resolved_commit=resolved_commit,
                    repo_dir=repo_dir,
                    conda_environment=payload.get("conda_environment"),
                    conda_prefix=payload.get("conda_prefix"),
                    python_executable=payload.get("python_executable"),
                    python_version=payload.get("python_version"),
                    torch_version=payload.get("torch_version"),
                    torch_cuda_version=payload.get("torch_cuda_version"),
                    torch_cuda_available=payload.get("torch_cuda_available"),
                    driver_version=payload.get("driver_version"),
                    gpu_name=gpu_name,
                    compute_capability=payload.get("compute_capability"),
                    gpu_matched=gpu_matched,
                    steps=tuple(steps),
                    diagnostics=tuple(diagnostics),
            manual_actions=tuple(manual_actions),
                    stdout_tail=_tail(recheck.stdout),
                    stderr_tail=_tail(recheck.stderr),
                )
        else:
            installed = run_command(
                [
                    "sh",
                    "-c",
                    f"{install_python} -m pip install {repo_dir}",
                ],
                install_timeout,
                networked=True,
            )
            if installed is None or not installed.ok:
                add_step(
                    "galvatron_install",
                    SpikeStepStatus.BLOCKED,
                    "official Galvatron install failed",
                    _tail(installed.stderr if installed is not None else ""),
                )
                return GalvatronSpikeResult(
                    run_id=run_id,
                    worker_id=worker_id,
                    expected_gpu=expected_gpu,
                    status=SPIKE_STATUS_BLOCKED,
                    started_at=started,
                    elapsed_s=round(time.monotonic() - start, 3),
                    install_mode="failed",
                    official_source=official_source,
                    requested_ref=requested_ref,
                    resolved_commit=resolved_commit,
                    repo_dir=repo_dir,
                    conda_environment=payload.get("conda_environment"),
                    conda_prefix=payload.get("conda_prefix"),
                    python_executable=payload.get("python_executable"),
                    python_version=payload.get("python_version"),
                    torch_version=payload.get("torch_version"),
                    torch_cuda_version=payload.get("torch_cuda_version"),
                    torch_cuda_available=payload.get("torch_cuda_available"),
                    driver_version=payload.get("driver_version"),
                    gpu_name=gpu_name,
                    compute_capability=payload.get("compute_capability"),
                    gpu_matched=gpu_matched,
                    steps=tuple(steps),
                    diagnostics=tuple(diagnostics),
            manual_actions=tuple(manual_actions),
                    stdout_tail=_tail(installed.stdout if installed is not None else ""),
                    stderr_tail=_tail(installed.stderr if installed is not None else ""),
                )

            recheck = run_script(
                build_spike_evidence_script(
                    worker_id=worker_id, expected_gpu=expected_gpu
                ),
                evidence_timeout,
            )
            if recheck is None:
                add_step(
                    "galvatron_install",
                    SpikeStepStatus.BLOCKED,
                    "install completed but post-install detection could not run",
                )
                return GalvatronSpikeResult(
                    run_id=run_id,
                    worker_id=worker_id,
                    expected_gpu=expected_gpu,
                    status=SPIKE_STATUS_BLOCKED,
                    started_at=started,
                    elapsed_s=round(time.monotonic() - start, 3),
                    install_mode="installed",
                    official_source=official_source,
                    requested_ref=requested_ref,
                    resolved_commit=resolved_commit,
                    repo_dir=repo_dir,
                    conda_environment=payload.get("conda_environment"),
                    conda_prefix=payload.get("conda_prefix"),
                    python_executable=payload.get("python_executable"),
                    python_version=payload.get("python_version"),
                    torch_version=payload.get("torch_version"),
                    torch_cuda_version=payload.get("torch_cuda_version"),
                    torch_cuda_available=payload.get("torch_cuda_available"),
                    driver_version=payload.get("driver_version"),
                    gpu_name=gpu_name,
                    compute_capability=payload.get("compute_capability"),
                    gpu_matched=gpu_matched,
                    steps=tuple(steps),
                    diagnostics=tuple(diagnostics),
            manual_actions=tuple(manual_actions),
                )
            recheck_payload = parse_spike_evidence(recheck.stdout) or {}
            new_torch = recheck_payload.get("torch_version")
            if new_torch != payload.get("torch_version"):
                add_step(
                    "galvatron_install",
                    SpikeStepStatus.BLOCKED,
                    f"torch changed after install: "
                    f"{payload.get('torch_version')} -> {new_torch}",
                )
                return GalvatronSpikeResult(
                    run_id=run_id,
                    worker_id=worker_id,
                    expected_gpu=expected_gpu,
                    status=SPIKE_STATUS_BLOCKED,
                    started_at=started,
                    elapsed_s=round(time.monotonic() - start, 3),
                    install_mode="blocked",
                    official_source=official_source,
                    requested_ref=requested_ref,
                    resolved_commit=resolved_commit,
                    repo_dir=repo_dir,
                    conda_environment=payload.get("conda_environment"),
                    conda_prefix=payload.get("conda_prefix"),
                    python_executable=payload.get("python_executable"),
                    python_version=payload.get("python_version"),
                    torch_version=payload.get("torch_version"),
                    torch_cuda_version=payload.get("torch_cuda_version"),
                    torch_cuda_available=payload.get("torch_cuda_available"),
                    driver_version=payload.get("driver_version"),
                    gpu_name=gpu_name,
                    compute_capability=payload.get("compute_capability"),
                    gpu_matched=gpu_matched,
                    steps=tuple(steps),
                    diagnostics=tuple(diagnostics),
            manual_actions=tuple(manual_actions),
                )
            payload = recheck_payload
            galvatron_installed = payload.get("galvatron_installed") is True
            galvatron_version = payload.get("galvatron_version")
            galvatron_source = payload.get("galvatron_source")
            payload, galvatron_installed, backfilled = _backfill_missing_python_deps(
                run_script=run_script,
                run_command=run_command,
                install_python=install_python,
                payload=payload,
                evidence_timeout=evidence_timeout,
                install_timeout=install_timeout,
                diagnostics=diagnostics,
            )
            galvatron_version = payload.get("galvatron_version")
            galvatron_source = payload.get("galvatron_source")
            if backfilled:
                add_step(
                    "galvatron_install",
                    SpikeStepStatus.PASS,
                    "auto-installed missing Python deps and verified import: "
                    + ", ".join(backfilled),
                )
            if not galvatron_installed:
                add_step(
                    "galvatron_install",
                    SpikeStepStatus.BLOCKED,
                    "install completed but Galvatron still not importable: "
                    + str(payload.get("galvatron_import_error") or "unknown error"),
                )
                return GalvatronSpikeResult(
                    run_id=run_id,
                    worker_id=worker_id,
                    expected_gpu=expected_gpu,
                    status=SPIKE_STATUS_BLOCKED,
                    started_at=started,
                    elapsed_s=round(time.monotonic() - start, 3),
                    install_mode="installed",
                    official_source=official_source,
                    requested_ref=requested_ref,
                    resolved_commit=resolved_commit,
                    repo_dir=repo_dir,
                    conda_environment=payload.get("conda_environment"),
                    conda_prefix=payload.get("conda_prefix"),
                    python_executable=payload.get("python_executable"),
                    python_version=payload.get("python_version"),
                    torch_version=payload.get("torch_version"),
                    torch_cuda_version=payload.get("torch_cuda_version"),
                    torch_cuda_available=payload.get("torch_cuda_available"),
                    driver_version=payload.get("driver_version"),
                    gpu_name=gpu_name,
                    compute_capability=payload.get("compute_capability"),
                    gpu_matched=gpu_matched,
                    steps=tuple(steps),
                    diagnostics=tuple(diagnostics),
            manual_actions=tuple(manual_actions),
                    stdout_tail=_tail(recheck.stdout),
                    stderr_tail=_tail(recheck.stderr),
                )
            add_step(
                "galvatron_install",
                SpikeStepStatus.PASS,
                f"installed from official source version={galvatron_version or 'unknown'}",
                _tail(installed.stdout),
            )

    add_step(
        "galvatron_import",
        (
            SpikeStepStatus.PASS
            if galvatron_installed and payload.get("galvatron_submodules_ok") is True
            else SpikeStepStatus.FAIL
        ),
        (
            f"version={galvatron_version or 'unknown'} "
            f"core.profiler importable={payload.get('galvatron_submodules_ok')}"
        )
        + (
            f" import_error={payload.get('galvatron_import_error')}"
            if payload.get("galvatron_import_error")
            else ""
        ),
    )
    for diagnostic in payload.get("diagnostics") or []:
        diagnostics.append(str(diagnostic))

    add_step(
        "flash_attn",
        SpikeStepStatus.SKIPPED,
        "conditional dependency (GALVATRON_FLASH_ATTN_INSTALL); not required by "
        "the minimal compatibility workload",
    )

    profiler_result: ProcessResult | None = None
    if repo_dir is None:
        repo_dir = checkout_dir
        exists_check = run_command(
            ["sh", "-c", f"test -f {repo_dir}/setup.py"],
            install_timeout,
        )
        if exists_check is not None and exists_check.ok:
            checkout: ProcessResult | None = None
        else:
            probe = run_command(
                [
                    "git",
                    "ls-remote",
                    "--tags",
                    GALVATRON_OFFICIAL_REPO,
                    requested_ref,
                ],
                git_probe_timeout,
                networked=True,
            )
            if probe is None or not probe.ok:
                add_step(
                    "profiler_runtime",
                    SpikeStepStatus.BLOCKED,
                    "official repo not reachable for profiler checkout",
                    _tail(
                        (probe.stderr if probe is not None else "")
                        or (probe.stdout if probe is not None else ""),
                    ),
                )
                probe = None
                checkout = None
            else:
                checkout = run_command(
                    [
                        "sh",
                        "-c",
                        "git clone --depth 1 "
                        f"--branch {requested_ref} "
                        f"{GALVATRON_OFFICIAL_REPO} {repo_dir}",
                    ],
                    install_timeout,
                    networked=True,
                )
        if checkout is not None and not checkout.ok:
            add_step(
                "profiler_runtime",
                SpikeStepStatus.BLOCKED,
                "official repo checkout failed (needed to run the official "
                "profile_hardware entry)",
                _tail(checkout.stderr if checkout is not None else ""),
            )
    if repo_dir is not None and (
        profiler_result is None
        and all(step.name != "profiler_runtime" for step in steps)
    ):
        profiler_result = run_command(
            [
                "sh",
                "-c",
                f"cd {repo_dir}/galvatron/profile_hardware && "
                "python profile_hardware.py scripts/profile_hardware.yaml "
                "num_gpus_per_node=1 max_tp_size=1 max_pp_deg=1",
            ],
            profile_timeout,
        )
    if profiler_result is None:
        if all(step.name != "profiler_runtime" for step in steps):
            add_step(
                "profiler_runtime",
                SpikeStepStatus.BLOCKED,
                "official profiler entry could not be executed",
            )
    elif not profiler_result.ok:
        add_step(
            "profiler_runtime",
            SpikeStepStatus.FAIL,
            f"official profile_hardware entry failed (exit={profiler_result.exit_code})",
            _tail(profiler_result.stdout) or _tail(profiler_result.stderr),
        )
    else:
        add_step(
            "profiler_runtime",
            SpikeStepStatus.PASS,
            "official profile_hardware entry completed with num_gpus_per_node=1",
            _tail(profiler_result.stdout),
        )

    status = derive_spike_status(steps)
    return GalvatronSpikeResult(
        run_id=run_id,
        worker_id=worker_id,
        expected_gpu=expected_gpu,
        status=status,
        started_at=started,
        elapsed_s=round(time.monotonic() - start, 3),
        install_mode=install_mode,
        official_source=official_source,
        requested_ref=requested_ref,
        resolved_commit=resolved_commit,
        galvatron_version=galvatron_version,
        galvatron_source=galvatron_source,
        repo_dir=repo_dir,
        conda_environment=payload.get("conda_environment"),
        conda_prefix=payload.get("conda_prefix"),
        python_executable=payload.get("python_executable"),
        python_version=payload.get("python_version"),
        torch_version=payload.get("torch_version"),
        torch_cuda_version=payload.get("torch_cuda_version"),
        torch_cuda_available=payload.get("torch_cuda_available"),
        driver_version=payload.get("driver_version"),
        gpu_name=gpu_name,
        compute_capability=payload.get("compute_capability"),
        gpu_matched=gpu_matched,
        steps=tuple(steps),
        diagnostics=tuple(diagnostics),
        manual_actions=tuple(manual_actions),
        stdout_tail=_tail(evidence_result.stdout),
        stderr_tail=_tail(evidence_result.stderr),
    )


def save_galvatron_spike_evidence(
    result: GalvatronSpikeResult,
    output_dir: str | Path,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["timestamp"] = _now()
    slug = _evidence_slug(result.worker_id, result.expected_gpu)
    path = directory / f"galvatron-spike-{slug}-{result.run_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    (directory / f"galvatron-spike-{slug}-latest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    return path


def load_galvatron_spike_evidence(path: str | Path) -> GalvatronSpikeResult:
    data = json.loads(Path(path).read_text())
    return GalvatronSpikeResult(
        run_id=str(data["run_id"]),
        worker_id=str(data["worker_id"]),
        expected_gpu=str(data["expected_gpu"]),
        status=str(data["status"]),
        started_at=str(data["started_at"]),
        elapsed_s=float(data.get("elapsed_s", 0.0)),
        install_mode=str(data.get("install_mode", "none")),
        official_source=data.get("official_source"),
        requested_ref=data.get("requested_ref"),
        resolved_commit=data.get("resolved_commit"),
        galvatron_version=data.get("galvatron_version"),
        galvatron_source=data.get("galvatron_source"),
        repo_dir=data.get("repo_dir"),
        conda_environment=data.get("conda_environment"),
        conda_prefix=data.get("conda_prefix"),
        python_executable=data.get("python_executable"),
        python_version=data.get("python_version"),
        torch_version=data.get("torch_version"),
        torch_cuda_version=data.get("torch_cuda_version"),
        torch_cuda_available=data.get("torch_cuda_available"),
        driver_version=data.get("driver_version"),
        gpu_name=data.get("gpu_name"),
        compute_capability=data.get("compute_capability"),
        gpu_matched=data.get("gpu_matched"),
        steps=tuple(
            GalvatronSpikeStep(
                name=str(step["name"]),
                status=SpikeStepStatus(str(step["status"])),
                detail=step.get("detail"),
                output_tail=step.get("output_tail"),
            )
            for step in data.get("steps", [])
        ),
        diagnostics=tuple(str(item) for item in data.get("diagnostics", [])),
        manual_actions=tuple(str(item) for item in data.get("manual_actions", [])),
        stdout_tail=data.get("stdout_tail"),
        stderr_tail=data.get("stderr_tail"),
    )
