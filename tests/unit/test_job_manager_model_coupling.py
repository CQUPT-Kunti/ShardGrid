from __future__ import annotations

import inspect

from shardgrid.common.models import as_engine_name
from shardgrid.control.job_manager import JobManager, PlannerWorkload
from shardgrid.engines.models import ParallelPlan


def test_planner_workload_currently_branches_on_model_type_and_zoo_builders() -> None:
    source = inspect.getsource(JobManager._planner_workload)

    assert "training_config.model.type" in source
    assert '"minimal_sequential"' in source
    assert '"hf_style"' in source
    assert '"large_residual_transformer"' in source
    assert '"generic_dag"' in source
    assert "build_zoo_model" in source
    assert "make_zoo_sample" in source
    assert "zoo_model" in source


def test_generic_captured_planner_workload_bypasses_legacy_model_type_dispatch() -> None:
    model = object()
    workload = PlannerWorkload(
        model=model,
        sample_args=("batch",),
        sample_kwargs={"mask": object()},
        model_name="ordinary_user_model",
    )

    class ManagerWithoutLegacyWorkload:
        def _planner_workload(self, _training_config: object) -> object:
            raise AssertionError("generic captured workload used legacy model-type dispatch")

    result = JobManager._automatic_planner_workload(
        ManagerWithoutLegacyWorkload(),
        object(),
        captured_workload=workload,
    )

    assert result is workload
    assert result.model is model
    assert result.model_name == "ordinary_user_model"
    assert result.source == "captured_context"


def test_automatic_plan_builder_has_generic_captured_workload_entrypoint() -> None:
    source = inspect.getsource(JobManager._build_automatic_parallel_plan)

    assert "captured_workload: PlannerWorkload | None = None" in source
    assert "_automatic_planner_workload" in source
    assert "_build_captured_parallel_plan" in source
    assert "planner_workload_source" in source
    assert "model_name=workload.model_name" in source


def test_captured_parallel_plan_builder_has_no_zoo_workload_dependency() -> None:
    source = inspect.getsource(JobManager._build_captured_parallel_plan)

    assert "training_config.model.type" not in source
    assert "build_zoo_model" not in source
    assert "make_zoo_sample" not in source
    assert "zoo_model" not in source


def _automatic_plan(requirements: dict[str, str]) -> ParallelPlan:
    return ParallelPlan(
        parallel_plan_id="plan-test",
        engine=as_engine_name("pytorch_pipeline"),
        model_name="ordinary",
        world_size=1,
        stages=["stage0"],
        partition_source="automatic",
        requirements=requirements,
    )


def test_launch_command_preserves_legacy_example_model_runtimes() -> None:
    source = inspect.getsource(JobManager._launch_command_for_assignment)

    assert "parallel_plan.partition_source" in source
    assert "generic_dag_runtime" in source
    assert "memory_probe" in source
    assert "examples/models/train_generic_dag.py" in source
    assert "examples/models/train_automatic_plan.py" in source
    assert "examples/models/train_pipeline.py" in source

    manager = object.__new__(JobManager)
    legacy_command = manager._launch_command_for_assignment(
        _automatic_plan({"generic_dag_runtime": "true"}),
        0,
    )

    assert legacy_command == "python examples/models/train_generic_dag.py --rank 0"


def test_launch_command_does_not_use_generic_dag_example_for_production_captured_path() -> None:
    manager = object.__new__(JobManager)
    command = manager._launch_command_for_assignment(
        _automatic_plan({"workload_source": "captured_context"}),
        0,
    )

    assert "examples/models/train_generic_dag.py" not in command
    assert "examples/models/train_automatic_plan.py" not in command
    assert "python -m shardgrid.runtime.generic_bootstrap" in command


def test_consolidated_checkpoint_currently_uses_model_specific_reconstruction() -> None:
    consolidated_source = inspect.getsource(JobManager._write_consolidated_model)
    generic_dag_source = inspect.getsource(JobManager._write_generic_dag_model_state)

    assert "training_config.model.type" in consolidated_source
    assert '"generic_dag"' in consolidated_source
    assert '"minimal_sequential"' in consolidated_source
    assert "MinimalTransformer" in consolidated_source
    assert "MinimalTransformerConfig" in consolidated_source
    assert "_write_generic_dag_model_state" in consolidated_source

    assert "build_zoo_model" in generic_dag_source
    assert "make_zoo_sample" in generic_dag_source
    assert "zoo_model" in generic_dag_source
    assert "load_state_dict" in generic_dag_source
