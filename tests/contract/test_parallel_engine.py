"""ParallelEngine adapter contract tests (T066).

A fake engine implements the full contract; another engine omits one method
and must fail loudly via UnsupportedEngineMethodError instead of silently
falling back.  The registry reflects the T065 decision statuses.
"""

from __future__ import annotations

from dataclasses import dataclass

from shardgrid.common.enums import BackendStatus, FailureStage
from shardgrid.engines.base import (
    ParallelEngine,
    UnsupportedEngineMethodError,
    registered_engine_registry,
)
from shardgrid.engines.models import (
    CompatibilitySpikeReport,
    EnginePreparation,
    ParallelEngineCandidate,
    ParallelPlan,
    ProfileResult,
)


@dataclass
class FakeEngine:
    engine_id: str = "fake"
    candidate: ParallelEngineCandidate = ParallelEngineCandidate(
        engine_id="fake",
        name="fake",
        status=BackendStatus.AVAILABLE,
        capabilities=["planning"],
    )

    def compatibility_spike(self, context: object) -> CompatibilitySpikeReport:
        del context
        return CompatibilitySpikeReport(
            report_id="fake-spike",
            component="fake",
            stage=FailureStage.PROBE,
            status=BackendStatus.AVAILABLE,
        )

    def profile(self, job: object, workers: object) -> ProfileResult:
        del job, workers
        return ProfileResult(engine_id=self.engine_id, status=BackendStatus.AVAILABLE)

    def plan(self, job: object, resources: object, network: object) -> ParallelPlan:
        del job, resources, network
        return ParallelPlan(
            parallel_plan_id="fake-plan",
            engine=self.candidate.name,
            engine_plan_path="/var/tmp/original-plan.json",
            world_size=2,
            stages=["stage0", "stage1"],
        )

    def prepare(self, job_snapshot: object, execution_plan: object) -> EnginePreparation:
        del job_snapshot, execution_plan
        return EnginePreparation(
            engine_id=self.engine_id, status=BackendStatus.AVAILABLE
        )

    def launch_metadata(self, parallel_plan: ParallelPlan) -> dict[str, object]:
        return {"engine": self.engine_id, "plan": parallel_plan.engine_plan_path}


class PartialEngine(FakeEngine):
    """An engine that omits ``profile`` - must raise loudly."""

    def profile(self, job: object, workers: object) -> ProfileResult:
        raise UnsupportedEngineMethodError(
            f"engine {self.engine_id} does not support profiling"
        )


def test_fake_engine_satisfies_protocol() -> None:
    assert isinstance(FakeEngine(), ParallelEngine)


def test_contract_methods_return_contract_types() -> None:
    engine = FakeEngine()
    spike = engine.compatibility_spike(None)
    assert isinstance(spike, CompatibilitySpikeReport)
    profile = engine.profile(None, None)
    assert isinstance(profile, ProfileResult)
    plan = engine.plan(None, None, None)
    assert isinstance(plan, ParallelPlan)
    preparation = engine.prepare(None, None)
    assert isinstance(preparation, EnginePreparation)
    metadata = engine.launch_metadata(plan)
    assert isinstance(metadata, dict)


def test_original_plan_is_preserved() -> None:
    engine = FakeEngine()
    plan = engine.plan(None, None, None)
    assert plan.engine_plan_path == "/var/tmp/original-plan.json"
    metadata = engine.launch_metadata(plan)
    assert metadata["plan"] == "/var/tmp/original-plan.json"


def test_unsupported_method_raises_explicitly() -> None:
    engine = PartialEngine()
    try:
        engine.profile(None, None)
    except UnsupportedEngineMethodError:
        pass
    else:
        raise AssertionError("unsupported method must raise UnsupportedEngineMethodError")


def test_registry_reflects_t065_statuses() -> None:
    registry = registered_engine_registry()
    galvatron = registry.by_name("galvatron")
    assert galvatron is not None
    assert galvatron.status in (
        BackendStatus.AVAILABLE,
        BackendStatus.EXPERIMENTAL,
    )
    assert registry.by_name("deepspeed_pipeline").status == BackendStatus.BLOCKED
    assert registry.by_name("nnscaler").status == BackendStatus.BLOCKED
    assert registry.by_name("pytorch_pipeline").status == BackendStatus.AVAILABLE


def test_registry_supported_candidates() -> None:
    registry = registered_engine_registry()
    supported = {candidate.engine_id for candidate in registry.supported()}
    assert "galvatron" in supported
    assert "pytorch_pipeline" in supported
    assert "deepspeed_pipeline" not in supported
    assert "nnscaler" not in supported


def test_static_validation_labeled_limited_support() -> None:
    registry = registered_engine_registry()
    galvatron = registry.by_name("galvatron")
    assert any(
        "limited support" in limitation for limitation in galvatron.limitations
    )
    plan = FakeEngine().plan(None, None, None)
    assert plan.engine_plan_path  # original external plan retained


def test_registry_round_trip() -> None:
    registry = registered_engine_registry()
    data = registry.to_dict()
    assert len(data) == 4
    names = {entry["engine_id"] for entry in data}
    assert names == {"galvatron", "pytorch_pipeline", "deepspeed_pipeline", "nnscaler"}
    loaded = ParallelEngineCandidate.from_dict(
        next(entry for entry in data if entry["engine_id"] == "galvatron")
    )
    assert loaded.status == BackendStatus.EXPERIMENTAL
    assert loaded.capabilities