"""Selected engine adapter and fallback dispatch contract tests (T067).

Covers: unknown engine, BLOCKED engine hard error (no silent fallback),
Galvatron accepted (single selected engine, original plan preserved),
Galvatron rejected -> fallback to the next supported engine, all-fail
dispatch error, and capability/status consistency with the T065 decision.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from shardgrid.common.enums import BackendStatus, JobState
from shardgrid.common.models import as_backend_name, as_job_id
from shardgrid.engines.base import registered_engine_registry
from shardgrid.engines.galvatron import GalvatronEngine
from shardgrid.engines.selected import (
    EngineSelectionError,
    SelectedEngine,
    build_engine_adapter,
    select_engine,
    select_with_fallback,
)
from shardgrid.jobs.models import TrainingJob


def _job(
    *,
    requested_world_size: int = 2,
    execution_plan_path: str | None = "/var/tmp/original-external-plan.json",
) -> TrainingJob:
    return TrainingJob(
        job_id=as_job_id("job-1"),
        config_path="/var/tmp/job.yaml",
        model="minimal",
        requested_world_size=requested_world_size,
        backend_preference=as_backend_name("nccl"),
        state=JobState.CREATED,
        execution_plan_path=execution_plan_path,
    )


def test_unknown_engine_fails_loudly() -> None:
    with pytest.raises(EngineSelectionError, match="unknown engine"):
        build_engine_adapter("not_an_engine")


def test_blocked_engine_fails_loudly_no_fallback() -> None:
    for engine_id in ("deepspeed_pipeline", "nnscaler"):
        with pytest.raises(EngineSelectionError, match="cannot be selected"):
            build_engine_adapter(engine_id)
        with pytest.raises(EngineSelectionError, match="cannot be selected"):
            select_with_fallback(engine_id, _job())


def test_galvatron_accepted_single_engine_original_plan_preserved() -> None:
    job = _job(execution_plan_path="/var/tmp/external-plan-v1.json")
    selected = select_engine("galvatron", job)
    assert isinstance(selected, SelectedEngine)
    assert selected.engine_id == "galvatron"
    assert isinstance(selected.engine, GalvatronEngine)
    assert selected.original_plan_path == "/var/tmp/external-plan-v1.json"
    assert selected.parallel_plan.engine_plan_path == "/var/tmp/external-plan-v1.json"
    assert selected.parallel_plan.world_size == 2
    assert selected.parallel_plan.stages == ["stage0", "stage1"]
    # exactly one engine is active per job
    assert selected.engine is not None
    assert selected.rejected_engine_ids == ()


def test_galvatron_plan_labeled_limited_support() -> None:
    selected = select_engine("galvatron", _job())
    assert any(
        "limited support" in limitation
        for limitation in selected.parallel_plan.limitations
    )


def test_galvatron_profile_unsupported_raises() -> None:
    engine = build_engine_adapter("galvatron")
    with pytest.raises(NotImplementedError):
        engine.profile(None, None)


def test_galvatron_adapter_contract_types() -> None:
    job = _job()
    engine = build_engine_adapter("galvatron")
    spike = engine.compatibility_spike(None)
    assert spike.status in (
        BackendStatus.AVAILABLE,
        BackendStatus.EXPERIMENTAL,
    )
    plan = engine.plan(job, None, None)
    metadata = engine.launch_metadata(plan)
    assert metadata["engine"] == "galvatron"
    assert metadata["original_plan_path"] == job.execution_plan_path
    preparation = engine.prepare(None, None)
    assert preparation.engine_id == "galvatron"


class _RejectingGalvatron(GalvatronEngine):
    """Galvatron whose plan is rejected (accepted -> rejected state)."""

    def plan(self, job: object, resources: object, network: object) -> object:
        raise EngineSelectionError("galvatron plan rejected for this job")


class _FailingAllEngines(_RejectingGalvatron):
    pass


def test_galvatron_rejected_falls_back_to_next_supported() -> None:
    registry = registered_engine_registry()
    job = _job(execution_plan_path="/var/tmp/external-plan-v2.json")

    class _FallbackRegistry:
        def by_name(self, engine_id: str) -> object:
            candidate = registry.by_name(engine_id)
            if engine_id == "galvatron":
                return replace(
                    candidate, status=BackendStatus.EXPERIMENTAL
                )
            return candidate

        def supported(self) -> list[object]:
            return [candidate for candidate in registry.supported()]


    original_build = build_engine_adapter

    def patched_build(engine_id: str, registry: object | None = None) -> object:
        candidate = _FallbackRegistry().by_name(engine_id)
        if engine_id == "galvatron":
            return _RejectingGalvatron(candidate=candidate)
        return original_build(engine_id, registry=registry)

    # The fallback path must land on pytorch_pipeline (next supported).
    import shardgrid.engines.selected as selected_module

    selected_module.build_engine_adapter = patched_build
    try:
        selected = select_with_fallback(
            "galvatron", job, registry=_FallbackRegistry()
        )
    finally:
        selected_module.build_engine_adapter = original_build

    assert selected.engine_id == "pytorch_pipeline"
    assert selected.parallel_plan.engine_plan_path == "/var/tmp/external-plan-v2.json"
    assert any("galvatron" in rejected for rejected in selected.rejected_engine_ids)


def test_all_engines_rejected_raises() -> None:
    registry = registered_engine_registry()
    job = _job()

    class _AllRejectRegistry:
        def by_name(self, engine_id: str) -> object:
            candidate = registry.by_name(engine_id)
            if engine_id in ("galvatron", "pytorch_pipeline"):
                return replace(candidate, status=BackendStatus.EXPERIMENTAL)
            return candidate

        def supported(self) -> list[object]:
            return [candidate for candidate in registry.supported()]

    original_build = build_engine_adapter
    import shardgrid.engines.selected as selected_module

    def all_reject(engine_id: str, registry: object | None = None) -> object:
        candidate = _AllRejectRegistry().by_name(engine_id)
        if engine_id == "galvatron":
            return _RejectingGalvatron(candidate=candidate)
        if engine_id == "pytorch_pipeline":
            from shardgrid.engines.selected import PytorchPipelineEngine

            class _RejectingPytorch(PytorchPipelineEngine):
                def plan(self, job: object, resources: object, network: object) -> object:
                    raise EngineSelectionError("pytorch plan rejected")

            return _RejectingPytorch(candidate=candidate)
        return original_build(engine_id, registry=registry)

    selected_module.build_engine_adapter = all_reject
    try:
        with pytest.raises(EngineSelectionError, match="no supported engine"):
            select_with_fallback("galvatron", job, registry=_AllRejectRegistry())
    finally:
        selected_module.build_engine_adapter = original_build


def test_registry_capability_status_matches_t065() -> None:
    registry = registered_engine_registry()
    assert registry.by_name("galvatron").status == BackendStatus.EXPERIMENTAL
    assert registry.by_name("pytorch_pipeline").status == BackendStatus.AVAILABLE
    assert registry.by_name("deepspeed_pipeline").status == BackendStatus.BLOCKED
    assert registry.by_name("nnscaler").status == BackendStatus.BLOCKED
    assert "BLOCKED_BY_WSL2_CUPTI" in " ".join(
        registry.by_name("galvatron").limitations
    )
    assert registry.by_name("nnscaler").capabilities == []