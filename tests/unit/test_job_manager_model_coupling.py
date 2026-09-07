from __future__ import annotations

import inspect

from shardgrid.control.job_manager import JobManager


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


def test_launch_command_currently_dispatches_to_example_model_runtimes() -> None:
    source = inspect.getsource(JobManager._launch_command_for_assignment)

    assert "parallel_plan.partition_source" in source
    assert "generic_dag_runtime" in source
    assert "memory_probe" in source
    assert "examples/models/train_generic_dag.py" in source
    assert "examples/models/train_automatic_plan.py" in source
    assert "examples/models/train_pipeline.py" in source


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
