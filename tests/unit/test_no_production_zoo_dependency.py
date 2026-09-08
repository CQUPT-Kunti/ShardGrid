"""T089 — ordinary production path must not depend on the model Zoo.

The ordinary ``shardgrid run train.py`` captured-entrypoint path must remain
generic: no ``build_zoo_model`` / ``make_zoo_sample`` / ``generic_partition_zoo``
/ model-name dispatch in the production capture/admission/artifact/checkpoint
chain.  Zoo/examples stay as legacy tests and stress references only.
"""

from __future__ import annotations

import inspect

from shardgrid.common.models import as_engine_name
from shardgrid.control.job_manager import JobManager
from shardgrid.engines.models import ParallelPlan


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


def test_run_entrypoint_source_has_no_zoo_dependency() -> None:
    source = inspect.getsource(JobManager.run_entrypoint)

    assert "build_zoo_model" not in source
    assert "make_zoo_sample" not in source
    assert "generic_partition_zoo" not in source
    assert "zoo_model" not in source
    assert "training_config.model.type" not in source


def test_captured_entrypoint_planner_workload_has_no_zoo_dependency() -> None:
    source = inspect.getsource(JobManager._captured_entrypoint_planner_workload)

    assert "build_zoo_model" not in source
    assert "make_zoo_sample" not in source
    assert "generic_partition_zoo" not in source
    assert "zoo_model" not in source
    assert "training_config.model.type" not in source


def test_build_automatic_parallel_plan_keeps_zoo_behind_legacy_model_type() -> None:
    planner_workload = inspect.getsource(JobManager._planner_workload)
    source = inspect.getsource(JobManager._build_automatic_parallel_plan)

    assert '"generic_dag"' in planner_workload
    assert "build_zoo_model" in planner_workload
    # the generic captured path must not route through the legacy dispatch
    assert "_build_captured_parallel_plan" in source
    assert "_automatic_planner_workload" in source
    assert "captured_workload" in source


def test_persist_captured_runtime_artifacts_has_no_model_name_reconstruction() -> None:
    source = inspect.getsource(JobManager._persist_captured_runtime_artifacts)

    assert "build_zoo_model" not in source
    assert "model.type" not in source
    assert "zoo_model" not in source


def test_generic_model_state_finalization_has_no_model_name_reconstruction() -> None:
    source = inspect.getsource(JobManager._write_generic_model_state)

    assert "build_zoo_model" not in source
    assert "make_zoo_sample" not in source
    assert "zoo_model" not in source
    assert "training_config.model.type" not in source


def test_generic_bootstrap_launch_command_is_module_not_example_script() -> None:
    from shardgrid.launchers.ssh import _GENERIC_RUNTIME_BOOTSTRAP_MODULE

    assert _GENERIC_RUNTIME_BOOTSTRAP_MODULE == "shardgrid.runtime.generic_bootstrap"

    manager = object.__new__(JobManager)
    command = manager._launch_command_for_assignment(
        _automatic_plan({"workload_source": "captured_context"}),
        0,
    )
    assert _GENERIC_RUNTIME_BOOTSTRAP_MODULE in command
    assert "train_generic_dag.py" not in command
    assert "train_automatic_plan.py" not in command
    assert "train_pipeline.py" not in command


def test_no_shardgrid_specific_user_model_api_required_in_entrypoint() -> None:
    """The ordinary entrypoint must not require ModelProvider/build_model/etc."""
    source = inspect.getsource(JobManager.run_entrypoint) + inspect.getsource(
        JobManager._captured_entrypoint_planner_workload
    )

    for forbidden_api in (
        "ModelProvider",
        "build_model()",
        "sample_inputs()",
        "ShardGridModel",
        "ShardGridStage",
    ):
        assert forbidden_api not in source, f"production path requires {forbidden_api}"