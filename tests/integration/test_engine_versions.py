"""Galvatron declared-requirements vs Worker runtime versions (T055).

The comparison logic is exercised here with mocked evidence: one test per
required outcome (compatible, Python mismatch, PyTorch mismatch, CUDA
mismatch, missing requirement, not installed, Worker evidence missing,
heterogeneous Workers, diagnostics preservation).  The live test reads the
real RTX 4060 and GTX 1650 WSL2 selected Conda runtimes from Machine A through
the existing SSH + WSL runtime wrapper and saves the comparison evidence.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any

from shardgrid.engines.compatibility import (
    ComparisonStatus,
    GalvatronDeclaredRequirements,
    WorkerVersionEvidence,
    collect_galvatron_declared_requirements,
    collect_worker_version_evidence,
    compare_galvatron_versions,
    save_galvatron_version_comparison,
)


def _requirements(
    *,
    python: str | None = ">=3.8",
    torch: str | None = "torch>=2.0.1",
    cuda: str | None = None,
    obtained: bool = True,
) -> GalvatronDeclaredRequirements:
    deps: list[str] = []
    if torch:
        deps.append(torch)
    if cuda:
        deps.append(cuda)
    return GalvatronDeclaredRequirements(
        source="github:PKU-DAIR/Hetu-Galvatron (setup.py, 2.4.1)",
        version="2.4.1",
        python_requires=python,
        requires_dist=tuple(deps),
        torch_requirement=torch,
        cuda_requirement=cuda,
        obtained=obtained,
    )


def _evidence(
    worker_id: str,
    *,
    python: str = "3.12.13",
    torch: str = "2.7.1+cu118",
    torch_cuda: str = "11.8",
    galvatron_installed: bool = True,
    status: str = "live",
    **overrides: Any,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "worker_id": worker_id,
        "physical_os": "windows",
        "runtime_os": "wsl2_linux",
        "conda_environment": "shardgrid",
        "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
        "python_executable": "/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
        "python_version": python,
        "torch_version": torch,
        "torch_cuda_version": torch_cuda,
        "torch_cuda_available": True,
        "driver_version": "566.07",
        "gpu_name": "NVIDIA GeForce RTX 4060",
        "compute_capability": "8.9",
        "galvatron_installed": galvatron_installed,
        "galvatron_version": "2.4.1" if galvatron_installed else None,
        "galvatron_source": "github:PKU-DAIR/Hetu-Galvatron" if galvatron_installed else None,
        "evidence_status": status,
    }
    evidence.update(overrides)
    return evidence


def _item(comparison: Any, worker_id: str, component: str) -> Any:
    worker = next(w for w in comparison.workers if w.worker_id == worker_id)
    return next(item for item in worker.items if item.component == component)


def test_compatible_versions_are_match() -> None:
    comparison = compare_galvatron_versions(
        _requirements(),
        [_evidence("gpu4060"), _evidence("gpu1060", torch_cuda="11.8")],
    )
    assert comparison.overall_status == ComparisonStatus.MATCH
    for worker in comparison.workers:
        assert worker.status == ComparisonStatus.MATCH
        python_item = _item(comparison, worker.worker_id, "python")
        torch_item = _item(comparison, worker.worker_id, "pytorch")
        assert python_item.status == ComparisonStatus.MATCH
        assert torch_item.status == ComparisonStatus.MATCH
        assert python_item.requirement == ">=3.8"
        assert python_item.actual == "3.12.13"


def test_python_mismatch_is_reported() -> None:
    comparison = compare_galvatron_versions(
        _requirements(python=">=3.9,<3.12"),
        [_evidence("gpu4060", python="3.10.12"), _evidence("gpu1060", python="3.12.13")],
    )
    assert comparison.overall_status == ComparisonStatus.VERSION_MISMATCH
    gpu1060 = next(w for w in comparison.workers if w.worker_id == "gpu1060")
    assert gpu1060.status == ComparisonStatus.VERSION_MISMATCH
    python_item = _item(comparison, "gpu1060", "python")
    assert python_item.status == ComparisonStatus.VERSION_MISMATCH
    assert python_item.requirement == ">=3.9,<3.12"
    assert python_item.actual == "3.12.13"
    assert python_item.impact
    gpu4060 = next(w for w in comparison.workers if w.worker_id == "gpu4060")
    assert gpu4060.status == ComparisonStatus.MATCH


def test_pytorch_mismatch_is_reported() -> None:
    comparison = compare_galvatron_versions(
        _requirements(torch="torch==2.0.1"),
        [_evidence("gpu4060", torch="2.7.1+cu118")],
    )
    assert comparison.overall_status == ComparisonStatus.VERSION_MISMATCH
    torch_item = _item(comparison, "gpu4060", "pytorch")
    assert torch_item.status == ComparisonStatus.VERSION_MISMATCH
    assert torch_item.requirement == "torch==2.0.1"
    assert torch_item.actual == "2.7.1+cu118"
    assert "declared" in (torch_item.detail or "")
    assert torch_item.impact


def test_cuda_mismatch_is_reported() -> None:
    comparison = compare_galvatron_versions(
        _requirements(cuda="cudatoolkit==11.8"),
        [_evidence("gpu4060", torch_cuda="12.1")],
    )
    assert comparison.overall_status == ComparisonStatus.VERSION_MISMATCH
    cuda_item = _item(comparison, "gpu4060", "cuda")
    assert cuda_item.status == ComparisonStatus.VERSION_MISMATCH
    assert cuda_item.requirement == "cudatoolkit==11.8"
    assert cuda_item.actual == "12.1"
    assert "affected" not in (cuda_item.impact or "")


def test_missing_requirement_is_requirement_unknown() -> None:
    comparison = compare_galvatron_versions(
        _requirements(torch=None),
        [_evidence("gpu4060")],
    )
    assert comparison.overall_status == ComparisonStatus.REQUIREMENT_UNKNOWN
    torch_item = _item(comparison, "gpu4060", "pytorch")
    assert torch_item.status == ComparisonStatus.REQUIREMENT_UNKNOWN
    assert torch_item.actual == "2.7.1+cu118"
    assert "no explicit declared requirement" in (torch_item.detail or "")


def test_requirements_unobtainable_is_requirement_unknown() -> None:
    comparison = compare_galvatron_versions(
        _requirements(obtained=False),
        [_evidence("gpu4060")],
    )
    assert comparison.overall_status == ComparisonStatus.REQUIREMENT_UNKNOWN
    for worker in comparison.workers:
        assert worker.status == ComparisonStatus.REQUIREMENT_UNKNOWN


def test_galvatron_not_installed_is_not_installed() -> None:
    comparison = compare_galvatron_versions(
        _requirements(),
        [
            _evidence("gpu4060", galvatron_installed=False),
            _evidence("gpu1060", galvatron_installed=False),
        ],
    )
    assert comparison.overall_status == ComparisonStatus.NOT_INSTALLED
    for worker in comparison.workers:
        assert worker.status == ComparisonStatus.NOT_INSTALLED
        assert _item(comparison, worker.worker_id, "galvatron").status == (
            ComparisonStatus.NOT_INSTALLED
        )
    python_item = _item(comparison, "gpu4060", "python")
    assert python_item.status == ComparisonStatus.MATCH


def test_worker_evidence_missing_is_blocked() -> None:
    comparison = compare_galvatron_versions(
        _requirements(),
        [
            _evidence("gpu4060"),
            {"worker_id": "gpu1060", "evidence_status": "pending"},
        ],
    )
    assert comparison.overall_status == ComparisonStatus.BLOCKED
    gpu1060 = next(w for w in comparison.workers if w.worker_id == "gpu1060")
    assert gpu1060.status == ComparisonStatus.BLOCKED
    assert any(
        item.component == "evidence" for item in gpu1060.items
    )
    gpu4060 = next(w for w in comparison.workers if w.worker_id == "gpu4060")
    assert gpu4060.status != ComparisonStatus.BLOCKED


def test_two_workers_with_different_environments() -> None:
    comparison = compare_galvatron_versions(
        _requirements(python=">=3.9,<3.12"),
        [
            _evidence("gpu4060", python="3.10.12"),
            _evidence("gpu1060", python="3.12.13"),
        ],
    )
    assert comparison.overall_status == ComparisonStatus.VERSION_MISMATCH
    statuses = {worker.worker_id: worker.status for worker in comparison.workers}
    assert statuses == {"gpu4060": ComparisonStatus.MATCH,
                        "gpu1060": ComparisonStatus.VERSION_MISMATCH}
    gpu1060 = next(w for w in comparison.workers if w.worker_id == "gpu1060")
    assert len(gpu1060.mismatches) == 1
    mismatch = gpu1060.mismatches[0]
    assert mismatch.component == "python"
    assert mismatch.requirement == ">=3.9,<3.12"
    assert mismatch.actual == "3.12.13"
    assert mismatch.impact


def test_mismatch_details_and_diagnostics_preserved() -> None:
    comparison = compare_galvatron_versions(
        _requirements(torch="torch==2.0.1"),
        [
            _evidence(
                "gpu4060",
                torch="2.7.1+cu118",
                diagnostics=("nvidia-smi failed: timeout",),
            ),
        ],
    )
    assert comparison.overall_status == ComparisonStatus.VERSION_MISMATCH
    payload = comparison.to_dict()
    assert payload["overall_status"] == "VERSION MISMATCH"
    worker_payload = payload["workers"][0]
    assert worker_payload["worker_id"] == "gpu4060"
    mismatch_payloads = worker_payload["mismatches"]
    assert any(
        item["component"] == "pytorch"
        and item["requirement"] == "torch==2.0.1"
        and item["actual"] == "2.7.1+cu118"
        for item in mismatch_payloads
    )
    assert worker_payload["mismatches"][0]["impact"]
    assert worker_payload["mismatches"][0]["detail"]
    assert "nvidia-smi failed" in json.dumps(payload)


def test_comparison_accepts_evidence_objects_and_maps() -> None:
    requirements = _requirements()
    evidence_obj = WorkerVersionEvidence(
        worker_id="gpu4060",
        python_version="3.12.13",
        torch_version="2.7.1+cu118",
        torch_cuda_version="11.8",
        galvatron_installed=True,
        galvatron_version="2.4.1",
        galvatron_source="github:PKU-DAIR/Hetu-Galvatron",
        evidence_status="live",
    )
    comparison = compare_galvatron_versions(requirements, [evidence_obj])
    assert comparison.overall_status == ComparisonStatus.MATCH
    assert comparison.workers[0].status == ComparisonStatus.MATCH


def test_spike_report_mapping_for_comparison() -> None:
    comparison = compare_galvatron_versions(
        _requirements(torch="torch==2.0.1"),
        [_evidence("gpu4060", torch="2.7.1+cu118")],
    )
    report = comparison.to_spike_report()
    assert report.component == "galvatron-versions"
    assert report.results == ["VERSION MISMATCH"]
    assert report.decision is not None
    assert report.blockers
    assert report.versions["galvatron_torch_requirement"] == "torch==2.0.1"


def test_save_comparison_evidence(tmp_path: Any) -> None:
    comparison = compare_galvatron_versions(
        _requirements(),
        [_evidence("gpu4060")],
    )
    path = save_galvatron_version_comparison(comparison, tmp_path)
    assert path.name == f"{comparison.run_id}.json"
    payload = json.loads(path.read_text())
    assert payload["overall_status"] == "MATCH"
    assert (tmp_path / "galvatron-versions-latest.json").exists()


def test_live_worker_version_comparison() -> None:
    """Real T055 comparison from Machine A against both GPU Workers."""
    from shardgrid.common.config import load_cluster_config
    from shardgrid.common.models import as_hostname
    from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
    from shardgrid.transport.ssh import SSHOptions, SSHTransport

    config = load_cluster_config("examples/workers.yaml")
    address_book = json.load(open("tests/address.json"))

    def build_wrapper(worker_id: str, gpu_marker: str) -> WSLRuntimeWrapper:
        worker = next(w for w in config.workers if str(w.worker_id) == worker_id)
        entry = next(
            e
            for e in address_book
            if gpu_marker.replace(" ", "") in str(e.get("gpu_model") or "").replace(" ", "")
        )
        worker = replace(
            worker,
            host=as_hostname(str(entry["ip"])),
            ssh_user=str(entry["username"]),
        )
        transport = SSHTransport(
            SSHOptions.from_ssh_config(
                config.ssh,
                host=str(entry["ip"]),
                user=worker.ssh_user,
                port=worker.ssh_port,
            )
        )
        return WSLRuntimeWrapper(
            WSLRuntimeConfig.from_worker_and_runtime(worker, config.runtime), transport
        )

    targets = [("gpu4060", "RTX 4060"), ("gpu1060", "GTX 1650")]
    evidence_list = [
        collect_worker_version_evidence(
            build_wrapper(worker_id, gpu_marker),
            worker_id=worker_id,
            timeout=120.0,
        )
        for worker_id, gpu_marker in targets
    ]

    requirements = collect_galvatron_declared_requirements()
    comparison = compare_galvatron_versions(requirements, evidence_list)

    output_dir = os.environ.get("SHARDGRID_ENGINE_EVIDENCE_DIR") or (
        "/var/tmp/shardgrid/engines"
    )
    path = save_galvatron_version_comparison(comparison, output_dir)

    detail = "\n".join(
        f"{evidence.worker_id}: evidence_status={evidence.evidence_status} "
        f"python={evidence.python_version} torch={evidence.torch_version} "
        f"cuda={evidence.torch_cuda_version} gpu={evidence.gpu_name} "
        f"galvatron={evidence.galvatron_installed} diagnostics={evidence.diagnostics}"
        for evidence in evidence_list
    )
    worker_summary = "\n".join(
        f"{worker.worker_id}: {worker.status.value} "
        f"mismatches={[i.component for i in worker.mismatches]}"
        for worker in comparison.workers
    )
    assert all(
        evidence.evidence_status == "live" for evidence in evidence_list
    ), f"live evidence missing\n{detail}"
    assert comparison.requirements.obtained, (
        "declared requirements could not be obtained: "
        f"{comparison.requirements.diagnostics}\n{detail}"
    )
    assert comparison.overall_status in {
        ComparisonStatus.MATCH,
        ComparisonStatus.NOT_INSTALLED,
        ComparisonStatus.VERSION_MISMATCH,
    }, (
        f"overall={comparison.overall_status.value}; evidence={path}\n"
        f"{detail}\n{worker_summary}"
    )
    assert not any(
        worker.status == ComparisonStatus.BLOCKED for worker in comparison.workers
    ), f"worker blocked\n{detail}\n{worker_summary}"
