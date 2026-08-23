"""nnScaler compatibility spike harness for GPU Workers (T064).

T064 evaluates nnScaler only if required by the decision process and records
environment compatibility.  The official nnScaler (microsoft/nnscaler, no
PyPI release) installs through its GitHub ``setup.py``; its dependency
resolution on this environment replaces the fixed torch/CUDA stack
(``torch 2.7.1+cu118`` -> ``torch 2.6.0`` plus the full ``nvidia-cu12`` set),
which is forbidden by ShardGrid's environment-consistency rule.

The harness therefore records a reproducible blocker with the real install
evidence instead of degrading the Worker environment.  Only real execution
can mark a capability PASS; here the recorded outcome is BLOCKED.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from shardgrid.transport.runtime import WSLRuntimeWrapper

SPIKE_STATUS_PASS = "PASS"
SPIKE_STATUS_FAIL = "FAIL"
SPIKE_STATUS_BLOCKED = "BLOCKED"

FORBIDDEN_PACKAGE_MARKERS = (
    "torch-",
    "torchvision-",
    "torchaudio-",
    "nvidia-",
    "triton-",
    "tensorrt",
)


@dataclass
class NnscalerSpikeStep:
    name: str
    status: str
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass
class NnscalerSpikeResult:
    run_id: str
    status: str
    nnscaler_installed: bool | None
    torch_version: str | None
    steps: list[NnscalerSpikeStep] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    started_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "nnscaler_installed": self.nnscaler_installed,
            "torch_version": self.torch_version,
            "steps": [step.to_dict() for step in self.steps],
            "blockers": list(self.blockers),
            "diagnostics": list(self.diagnostics),
            "started_at": self.started_at,
        }


def find_forbidden_package_changes(would_install_text: str) -> list[str]:
    """List forbidden stack changes (torch/nvidia/triton) in a pip plan."""
    changes: list[str] = []
    for token in (would_install_text or "").replace(",", " ").split():
        for marker in FORBIDDEN_PACKAGE_MARKERS:
            if token.startswith(marker):
                changes.append(token)
                break
    return sorted(set(changes))


def preflight_install_blockers(
    would_install_text: str,
    *,
    current_torch_version: str | None,
) -> list[str]:
    blockers: list[str] = []
    changes = find_forbidden_package_changes(would_install_text)
    for change in changes:
        if change.startswith(("torch-", "torchvision-", "torchaudio-")):
            blockers.append(
                f"nnScaler install would replace the PyTorch stack: {change} "
                f"(current torch {current_torch_version or 'unknown'})"
            )
        else:
            blockers.append(
                f"nnScaler install would introduce a CUDA stack change: {change}"
            )
    return blockers


def derive_spike_status(
    *,
    nnscaler_installed: bool,
    blockers: Sequence[str],
) -> str:
    if blockers:
        return SPIKE_STATUS_BLOCKED
    if nnscaler_installed:
        return SPIKE_STATUS_PASS
    return SPIKE_STATUS_BLOCKED


def save_spike_evidence(result: NnscalerSpikeResult, output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "nnscaler-latest.json"
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return path


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_nnscaler_spike(
    wrapper: WSLRuntimeWrapper,
    *,
    would_install_text: str,
    nnscaler_installed: bool = False,
    current_torch_version: str | None = None,
) -> NnscalerSpikeResult:
    """Record the nnScaler environment-compatibility spike outcome.

    ``would_install_text`` is the real pip resolution plan (captured from an
    official-source install attempt); the harness turns it into a blocker when
    it would replace the fixed torch/CUDA stack.
    """
    run_id = _now_utc().replace(":", "-").replace("+00:00", "Z")
    steps: list[NnscalerSpikeStep] = []

    if current_torch_version is None:
        result = wrapper.run(
            'python -c "import torch; print(torch.__version__)"', timeout=60
        )
        current_torch_version = (result.stdout or "").strip() or None
    steps.append(
        NnscalerSpikeStep(
            name="torch_version",
            status="PASS",
            detail=f"torch {current_torch_version}",
        )
    )
    steps.append(
        NnscalerSpikeStep(
            name="nnscaler_installed",
            status="PASS" if nnscaler_installed else "INFO",
            detail=f"nnscaler installed: {nnscaler_installed}",
        )
    )

    blockers = preflight_install_blockers(
        would_install_text, current_torch_version=current_torch_version
    )
    if blockers:
        steps.append(
            NnscalerSpikeStep(
                name="install_preflight",
                status="BLOCKED",
                detail="; ".join(blockers),
            )
        )
    else:
        steps.append(
            NnscalerSpikeStep(
                name="install_preflight",
                status="PASS",
                detail="official install would not replace the torch/CUDA stack",
            )
        )

    status = derive_spike_status(
        nnscaler_installed=nnscaler_installed, blockers=blockers
    )
    if status == SPIKE_STATUS_BLOCKED and not blockers:
        blockers.append(
            "nnScaler is not installed; official-source install requires "
            "dependency resolution that must not replace torch 2.7.1+cu118"
        )

    return NnscalerSpikeResult(
        run_id=run_id,
        status=status,
        nnscaler_installed=nnscaler_installed,
        torch_version=current_torch_version,
        steps=steps,
        blockers=list(blockers),
        diagnostics=[f"would_install_len={len(would_install_text or '')}"],
        started_at=_now_utc(),
    )