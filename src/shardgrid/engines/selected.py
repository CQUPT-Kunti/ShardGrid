"""Selected engine adapter and fallback dispatch (T067).

Upper layers obtain the active engine through :func:`select_engine` /
:func:`select_with_fallback`, never by hard-coding a framework.  Rules:

- unknown engine ids fail loudly (``EngineSelectionError``)
- BLOCKED engines fail loudly and are never silently replaced
- fallback dispatch only reacts to a *plan* rejection (accepted -> rejected)
  of a supported engine and tries the next supported candidate in registry
  order
- exactly one selected engine is active per job; the original external plan
  reference is preserved in the returned :class:`ParallelPlan`
"""

from __future__ import annotations

from dataclasses import dataclass

from shardgrid.common.enums import BackendStatus
from shardgrid.engines.base import (
    ParallelEngine,
    UnsupportedEngineMethodError,
    registered_engine_registry,
)
from shardgrid.engines.models import (
    EnginePreparation,
    ParallelEngineCandidate,
    ParallelPlan,
    ProfileResult,
)


class EngineSelectionError(RuntimeError):
    """Raised when an engine cannot be selected (unknown, blocked, or none)."""


@dataclass(frozen=True)
class SelectedEngine:
    job_id: object
    engine: ParallelEngine
    candidate: ParallelEngineCandidate
    parallel_plan: ParallelPlan
    original_plan_path: str | None = None
    rejected_engine_ids: tuple[str, ...] = ()

    @property
    def engine_id(self) -> str:
        return self.candidate.engine_id


class PytorchPipelineEngine:
    """Lightweight ParallelEngine adapter for torch.distributed.pipelining.

    Registered fallback candidate (T063 SUPPORTED).  Only plan/prepare/
    launch_metadata are implemented on the MVP adapter; the spike harness in
    ``src/shardgrid/engines/pytorch_pipeline.py`` covers real execution.
    """

    engine_id = "pytorch_pipeline"

    def __init__(self, candidate: ParallelEngineCandidate | None = None) -> None:
        from shardgrid.engines.base import registered_engine_registry

        self.candidate = candidate or registered_engine_registry().by_name(
            self.engine_id
        )
        if self.candidate is None:
            raise ValueError("pytorch_pipeline engine candidate is not registered")

    def compatibility_spike(self, context: object) -> object:
        from shardgrid.common.enums import FailureStage
        from shardgrid.engines.models import CompatibilitySpikeReport

        del context
        return CompatibilitySpikeReport(
            report_id="pytorch-pipeline-mvp",
            component="pytorch_pipeline",
            stage=FailureStage.PROBE,
            status=self.candidate.status,
            results=["T063 two-host GPipe spike PASS (0.9s)"],
            decision=(
                "torch.distributed.pipelining SUPPORTED on the two-host "
                "environment; fallback candidate"
            ),
        )

    def profile(self, job: object, workers: object) -> ProfileResult:
        raise UnsupportedEngineMethodError(
            "pytorch_pipeline profile integration is not implemented on the "
            "MVP adapter"
        )

    def plan(self, job: object, resources: object, network: object) -> ParallelPlan:
        from shardgrid.common.models import as_engine_name

        del resources, network
        world_size = getattr(job, "requested_world_size", 2)
        external_plan_path = getattr(job, "execution_plan_path", None)
        return ParallelPlan(
            parallel_plan_id="pytorch-pipeline-static-plan",
            engine=as_engine_name(self.engine_id),
            engine_plan_path=external_plan_path,
            model_name=getattr(job, "model", "") or "",
            world_size=int(world_size),
            stages=[f"stage{index}" for index in range(int(world_size))],
            requirements={
                "plan_mode": "static",
                "one_gpu_per_host": "1",
            },
            limitations=[
                "static validation plan labeled limited support "
                "(torch.distributed.pipelining, GPipe schedule)"
            ],
        )

    def prepare(self, job_snapshot: object, execution_plan: object) -> EnginePreparation:
        refs: list[str] = []
        if getattr(job_snapshot, "plan_path", None):
            refs.append(str(job_snapshot.plan_path))
        return EnginePreparation(
            engine_id=self.engine_id,
            status=BackendStatus.AVAILABLE,
            snapshot_artifact_refs=refs,
            diagnostics=["prepare is metadata-only on the MVP adapter"],
        )

    def launch_metadata(self, parallel_plan: ParallelPlan) -> dict[str, object]:
        return {
            "engine": self.engine_id,
            "plan_mode": "static",
            "original_plan_path": parallel_plan.engine_plan_path,
            "world_size": parallel_plan.world_size,
        }


def build_engine_adapter(
    engine_id: str,
    registry: object | None = None,
) -> ParallelEngine:
    """Instantiate the adapter for ``engine_id`` from the registry.

    Unknown ids and BLOCKED engines raise :class:`EngineSelectionError`;
    nothing silently falls back here.
    """
    if registry is None:
        registry = registered_engine_registry()
    candidate = registry.by_name(engine_id)
    if candidate is None:
        raise EngineSelectionError(f"unknown engine: {engine_id!r}")
    if candidate.status in (BackendStatus.BLOCKED, BackendStatus.NOT_CHECKED):
        raise EngineSelectionError(
            f"engine {engine_id!r} is {candidate.status.value} and cannot be "
            "selected (no silent fallback)"
        )
    if engine_id == "galvatron":
        from shardgrid.engines.galvatron import GalvatronEngine

        return GalvatronEngine(candidate=candidate)
    if engine_id == "pytorch_pipeline":
        return PytorchPipelineEngine(candidate=candidate)
    raise EngineSelectionError(f"no adapter implementation for engine {engine_id!r}")


def select_engine(
    engine_id: str,
    job: object,
    resources: object = None,
    network: object = None,
    *,
    registry: object | None = None,
) -> SelectedEngine:
    """Select exactly one engine for a job without fallback."""
    engine = build_engine_adapter(engine_id, registry=registry)
    plan = engine.plan(job, resources, network)
    return SelectedEngine(
        job_id=getattr(job, "job_id", None),
        engine=engine,
        candidate=engine.candidate,
        parallel_plan=plan,
        original_plan_path=plan.engine_plan_path,
    )


def select_with_fallback(
    engine_id: str,
    job: object,
    resources: object = None,
    network: object = None,
    *,
    registry: object | None = None,
) -> SelectedEngine:
    """Select an engine, falling back only on plan rejection.

    A BLOCKED/unknown requested engine is a hard error (no fallback).  If the
    requested supported engine's ``plan`` raises, the next supported
    candidate in registry order is tried; if none succeeds,
    :class:`EngineSelectionError` is raised with the rejected engine ids.
    """
    if registry is None:
        registry = registered_engine_registry()
    first = build_engine_adapter(engine_id, registry=registry)
    candidate_engines: list[tuple[str, ParallelEngine]] = [
        (engine_id, first)
    ]
    for candidate in registry.supported():
        if candidate.engine_id == engine_id:
            continue
        adapter = build_engine_adapter(candidate.engine_id, registry=registry)
        candidate_engines.append((candidate.engine_id, adapter))

    rejected: list[str] = []
    for candidate_id, engine in candidate_engines:
        try:
            plan = engine.plan(job, resources, network)
        except EngineSelectionError as error:
            rejected.append(f"{candidate_id} ({error})")
            continue
        except Exception as error:  # noqa: BLE001 - surfaced as rejection
            rejected.append(f"{candidate_id} ({error.__class__.__name__}: {error})")
            continue
        return SelectedEngine(
            job_id=getattr(job, "job_id", None),
            engine=engine,
            candidate=engine.candidate,
            parallel_plan=plan,
            original_plan_path=plan.engine_plan_path,
            rejected_engine_ids=tuple(rejected),
        )
    raise EngineSelectionError(
        "no supported engine produced a plan; rejected: "
        + "; ".join(rejected)
    )