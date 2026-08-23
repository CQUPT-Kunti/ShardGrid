"""ParallelEngine adapter contract (T066).

Defines the minimal adapter boundary between ShardGrid orchestration and
parallel-engine frameworks (Galvatron, PyTorch pipelining, DeepSpeed
Pipeline, nnScaler, ...).  Upper layers depend on this interface instead of
hard-coding any framework.  Framework differences stay inside each adapter.

Contract methods (per ``contracts/adapter-contracts.md``):

- ``compatibility_spike`` -> CompatibilitySpikeReport
- ``profile`` -> ProfileResult
- ``plan`` -> ParallelPlan (must preserve the original external plan)
- ``prepare`` -> EnginePreparation
- ``launch_metadata`` -> dict

Rules:

- Every adapter carries its candidate record (name/status/capabilities/
  limitations) so registry consumers can express SUPPORTED / BLOCKED /
  NOT_SELECTED without probing internals.
- Methods a concrete adapter does not support must raise
  :class:`UnsupportedEngineMethodError` - never silently fall back.
- Static-validation plans (explicit parallel configs without profiler-driven
  search) must be labeled ``limited support`` in the plan limitations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from shardgrid.common.enums import BackendStatus
from shardgrid.engines.models import (
    CompatibilitySpikeReport,
    EnginePreparation,
    ParallelEngineCandidate,
    ParallelPlan,
    ProfileResult,
)


class UnsupportedEngineMethodError(NotImplementedError):
    """Raised by an engine adapter for a contract method it does not support.

    Adapters must raise this explicitly instead of silently falling back, so
    unsupported capabilities are never mistaken for support.
    """


@runtime_checkable
class ParallelEngine(Protocol):
    """Adapter contract for a parallel-engine framework (T066).

    Implementations must expose ``engine_id`` and ``candidate`` (the
    registered engine record) plus the five contract methods.  Any method
    that is not implemented for the concrete engine raises
    :class:`UnsupportedEngineMethodError`.
    """

    engine_id: str
    candidate: ParallelEngineCandidate

    def compatibility_spike(self, context: Any) -> CompatibilitySpikeReport: ...

    def profile(self, job: Any, workers: Any) -> ProfileResult: ...

    def plan(self, job: Any, resources: Any, network: Any) -> ParallelPlan: ...

    def prepare(
        self, job_snapshot: Any, execution_plan: Any
    ) -> EnginePreparation: ...

    def launch_metadata(self, parallel_plan: ParallelPlan) -> dict[str, Any]: ...


@dataclass(frozen=True)
class EngineRegistry:
    """Static engine registry reflecting the T065 decision (no adapters yet).

    The registry records each candidate's real status; adapters are added in
    T067.  Selection stays adapter-driven: upper layers read this registry and
    choose by candidate status, never by hard-coded framework names in
    business logic.
    """

    candidates: tuple[ParallelEngineCandidate, ...] = field(default_factory=tuple)

    def by_name(self, engine_id: str) -> ParallelEngineCandidate | None:
        for candidate in self.candidates:
            if candidate.engine_id == engine_id:
                return candidate
        return None

    def supported(self) -> list[ParallelEngineCandidate]:
        return [
            candidate
            for candidate in self.candidates
            if candidate.status
            in (BackendStatus.AVAILABLE, BackendStatus.EXPERIMENTAL)
        ]

    def to_dict(self) -> list[dict[str, Any]]:
        return [candidate.to_dict() for candidate in self.candidates]


def registered_engine_registry() -> EngineRegistry:
    """Registry of engine candidates with the T065 evidence-derived status.

    Statuses come from the published compatibility decision
    (``examples/compatibility/parallel-engine-decision.json``):
    Galvatron SELECTED, PyTorch pipelining SUPPORTED, DeepSpeed Pipeline
    BLOCKED, nnScaler BLOCKED / NOT_SELECTED.
    """
    return EngineRegistry(
        candidates=(
            ParallelEngineCandidate(
                engine_id="galvatron",
                name="galvatron",
                version="v2.4.0",
                source="github:PKU-DAIR/Hetu-Galvatron",
                status=BackendStatus.EXPERIMENTAL,
                capabilities=[
                    "one_gpu_per_host_placement",
                    "pipeline_construction",
                    "runtime_launch",
                    "checkpoint",
                    "model_profiler",
                ],
                limitations=[
                    "hardware profiler BLOCKED_BY_WSL2_CUPTI",
                    "search engine blocked (depends on hardware profiler)",
                    "static validation plans labeled limited support",
                    "GTX 1650 stage budgets <= ~1 GiB",
                ],
                compatibility_report_path=(
                    "examples/compatibility/galvatron-report.json"
                ),
            ),
            ParallelEngineCandidate(
                engine_id="pytorch_pipeline",
                name="pytorch_pipeline",
                version="torch 2.7.1",
                source="torch.distributed.pipelining",
                status=BackendStatus.AVAILABLE,
                capabilities=["two_host_pipeline_placement", "runtime_launch"],
                limitations=["fallback candidate; not the MVP default"],
                compatibility_report_path="docs/compatibility/pytorch-pipeline.md",
            ),
            ParallelEngineCandidate(
                engine_id="deepspeed_pipeline",
                name="deepspeed_pipeline",
                version="0.19.5",
                source="github:microsoft/DeepSpeed",
                status=BackendStatus.BLOCKED,
                capabilities=[],
                limitations=[
                    "two-host WSL2 train_batch deadlock (T062)",
                    "not usable on current environment",
                ],
                compatibility_report_path="docs/compatibility/deepspeed-pipeline.md",
            ),
            ParallelEngineCandidate(
                engine_id="nnscaler",
                name="nnscaler",
                version="0.8",
                source="github:microsoft/nnscaler",
                status=BackendStatus.BLOCKED,
                capabilities=[],
                limitations=[
                    "official install replaces torch 2.7.1+cu118 / nvidia-cu12 (T064)",
                    "not selected on current environment",
                ],
                compatibility_report_path="docs/compatibility/nnscaler.md",
            ),
        )
    )