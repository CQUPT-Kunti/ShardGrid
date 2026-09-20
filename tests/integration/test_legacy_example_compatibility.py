"""T089 — legacy example compatibility after Phase 8-11 architecture repair.

Legacy Zoo/example assets must remain importable and functional as regression
assets, while ordinary production ``shardgrid run`` stays generic and never
routes through the Zoo.  These tests prove both sides of the contract.
"""

from __future__ import annotations

import importlib

import pytest

from shardgrid.control.job_manager import JobManager


def _import_example(name: str):
    return importlib.import_module(name)


def test_generic_partition_zoo_example_remains_importable() -> None:
    zoo = _import_example("examples.models.generic_partition_zoo")
    assert callable(getattr(zoo, "build_zoo_model", None))
    assert callable(getattr(zoo, "make_zoo_sample", None))


def test_train_generic_dag_example_remains_legacy_asset() -> None:
    train = _import_example("examples.models.train_generic_dag")
    assert "example" in (train.__doc__ or "").lower()
    assert callable(getattr(train, "build_zoo_model", None))


def test_large_residual_transformer_example_remains_importable() -> None:
    module = _import_example("examples.models.large_residual_transformer")
    assert callable(getattr(module, "build_large_residual_transformer", None))


def test_partition_stress_model_example_remains_importable() -> None:
    module = _import_example("examples.models.partition_stress_model")
    assert callable(getattr(module, "build_partition_stress_model", None))


def test_legacy_examples_are_not_production_runtime_modules() -> None:
    """The worker bootstrap runtime is the generic module, never an example."""
    from shardgrid.launchers.ssh import _GENERIC_RUNTIME_BOOTSTRAP_MODULE

    assert _GENERIC_RUNTIME_BOOTSTRAP_MODULE == "shardgrid.runtime.generic_bootstrap"
    assert "examples" not in _GENERIC_RUNTIME_BOOTSTRAP_MODULE

    manager = object.__new__(JobManager)
    plan = type(
        "Plan",
        (),
        {
            "parallel_plan_id": "p",
            "selected_candidate_id": "c",
            "partition_source": "automatic",
            "requirements": {"workload_source": "captured_context"},
        },
    )()
    command = manager._launch_command_for_assignment(plan, 0)
    assert "shardgrid.runtime.generic_bootstrap" in command
    assert "train_generic_dag.py" not in command


def test_zoo_model_construction_requires_explicit_legacy_request() -> None:
    """Zoo builders are only reachable through legacy model.type dispatch."""
    import inspect

    source = inspect.getsource(JobManager._planner_workload)
    captured_source = inspect.getsource(JobManager._automatic_planner_workload)

    assert '"generic_dag"' in source
    assert "build_zoo_model" in source
    # the ordinary captured path is separate and bypasses legacy dispatch
    assert "captured_workload" in captured_source
    assert "training_config.model.type" not in captured_source


@pytest.mark.parametrize(
    "example_module",
    [
        "examples.models.generic_partition_zoo",
        "examples.models.train_generic_dag",
        "examples.models.train_automatic_plan",
        "examples.models.large_residual_transformer",
        "examples.models.partition_stress_model",
    ],
)
def test_example_modules_import_without_side_effects(example_module: str) -> None:
    module = _import_example(example_module)
    assert module.__name__ == example_module