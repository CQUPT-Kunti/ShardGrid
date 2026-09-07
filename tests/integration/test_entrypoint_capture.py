from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

CAPTURE_NOT_IMPLEMENTED = pytest.mark.xfail(
    reason=(
        "T031 contract baseline: entrypoint capture runner is specified "
        "but not implemented until T032"
    ),
    strict=True,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "ordinary_training_scripts"


def _capture_module() -> Any:
    return importlib.import_module("shardgrid.capture.runner")


def _capture(script_name: str, *argv: str) -> Any:
    module = _capture_module()
    return module.capture_entrypoint(
        FIXTURE_ROOT / script_name,
        argv=tuple(argv),
        cwd=FIXTURE_ROOT,
        environment={"SHARDGRID_TEST_CAPTURE": "1"},
    )


@CAPTURE_NOT_IMPLEMENTED
def test_capture_contract_preserves_entrypoint_and_structured_positional_batch() -> None:
    context = _capture(
        "positional_tuple_train.py",
        "--epochs",
        "1",
        "--checkpoint",
        "out/tuple.pt",
    )

    assert context.entrypoint_path.endswith("positional_tuple_train.py")
    assert context.argv == ("--epochs", "1", "--checkpoint", "out/tuple.pt")
    assert context.cwd == str(FIXTURE_ROOT)
    assert context.environment_summary["SHARDGRID_TEST_CAPTURE"] == "present"
    assert context.capture_backend
    assert context.capture_backend_version
    assert context.model_identity["class_name"] == "TupleBatchModel"
    assert context.model_identity["module_name"] == "__main__"
    assert context.parameter_count > 0
    assert context.buffer_count == 0
    assert context.first_batch_structure["kind"] == "tuple"
    assert context.model_call["args"]["kind"] == "tuple"
    assert context.model_call["kwargs"]["kind"] == "dict"
    assert context.tensor_metadata["arg0"]["shape"] == [3, 4]
    assert context.tensor_metadata["arg0"]["dtype"] == "torch.float32"
    assert context.tensor_metadata["arg0"]["device"] == "cpu"
    assert context.tensor_metadata["arg0"]["requires_grad"] is False


@CAPTURE_NOT_IMPLEMENTED
def test_capture_contract_preserves_kwargs_nested_mapping_mask_and_labels() -> None:
    context = _capture(
        "kwargs_hf_mapping_train.py",
        "--config",
        "user-model.yaml",
        "--checkpoint",
        "out/mapping.pt",
    )

    assert context.model_identity["class_name"] == "KeywordClassifier"
    assert context.first_batch_structure["kind"] == "dict"
    assert context.first_batch_structure["fields"]["metadata"]["kind"] == "dict"
    assert context.model_call["args"]["items"] == []
    assert set(context.model_call["kwargs"]["fields"]) >= {
        "input_ids",
        "attention_mask",
        "labels",
    }
    assert context.tensor_metadata["kwarg.input_ids"]["dtype"] == "torch.int64"
    assert context.tensor_metadata["kwarg.attention_mask"]["dtype"] == "torch.bool"
    assert context.tensor_metadata["kwarg.labels"]["shape"] == [2]


@CAPTURE_NOT_IMPLEMENTED
def test_capture_contract_records_optimizer_scheduler_and_state_key_mapping() -> None:
    context = _capture(
        "multi_output_lifecycle_train.py",
        "synthetic",
        "--checkpoint",
        "out/lifecycle.pt",
    )

    assert context.model_identity["class_name"] == "MultiOutputModel"
    assert context.model_output_structure["kind"] == "tuple"
    assert context.optimizer["class_name"] == "Adam"
    assert context.optimizer["param_groups"]
    assert context.optimizer["param_groups"][0]["hyperparameters"]["lr"] == 0.02
    assert context.scheduler["class_name"] == "StepLR"
    assert context.lifecycle["loss_computed_before_backward"] is True
    assert context.lifecycle["backward_called_before_optimizer_step"] is True
    assert context.lifecycle["scheduler_step_observed"] is True
    assert context.distributed_mutation_observed_before_capture is False
    assert context.state_dict_key_to_canonical_state_id
    assert set(context.state_dict_key_to_canonical_state_id) >= {
        "left.weight",
        "right.weight",
        "head.weight",
    }


@CAPTURE_NOT_IMPLEMENTED
def test_capture_contract_returns_structured_unsupported_diagnostics() -> None:
    module = _capture_module()

    result = module.capture_entrypoint(
        FIXTURE_ROOT / "unsupported_dynamic_control_flow_train.py",
        argv=(),
        cwd=FIXTURE_ROOT,
        environment={},
    )

    assert result.ok is False
    assert result.failure.stage == "capture"
    assert result.failure.code
    assert result.failure.message
    assert result.failure.artifact_log_ref
    assert "generic exception" not in result.failure.message.lower()


def test_ordinary_training_script_fixtures_do_not_import_shardgrid_user_api() -> None:
    forbidden = ("shardgrid", "ModelProvider", "sample_inputs", "compute_loss", "build_model")
    offenders: list[str] = []
    for path in sorted(FIXTURE_ROOT.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if any(token in source for token in forbidden):
            offenders.append(path.name)

    assert offenders == []
