from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from shardgrid.artifacts.snapshot import write_capture_context
from shardgrid.jobs.models import JobSnapshot

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "ordinary_training_scripts"


def _capture_module() -> Any:
    return importlib.import_module("shardgrid.bootstrap.runner")


def _capture(script_name: str, *argv: str) -> Any:
    module = _capture_module()
    return module.capture_entrypoint(
        FIXTURE_ROOT / script_name,
        argv=tuple(argv),
        cwd=FIXTURE_ROOT,
        environment={"SHARDGRID_TEST_CAPTURE": "1"},
    )


def _job_snapshot(tmp_path: Path) -> JobSnapshot:
    root = tmp_path / "jobs" / "job-capture"
    return JobSnapshot(
        job_id="job-capture",
        root_path=str(root),
        code_path=str(root / "code"),
        config_path=str(root / "config"),
        plan_path=str(root / "plan"),
        logs_path=str(root / "logs"),
        environment_path=str(root / "environment"),
        checkpoint_path=str(root / "checkpoint"),
        diagnostics_path=str(root / "diagnostics"),
    )


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
    assert context.first_batch_structure["kind"] in {"tuple", "list"}
    assert context.model_call["args"]["kind"] == "tuple"
    assert context.model_call["kwargs"]["kind"] == "dict"
    assert context.tensor_metadata["arg0"]["shape"] == [3, 4]
    assert context.tensor_metadata["arg0"]["dtype"] == "torch.float32"
    assert context.tensor_metadata["arg0"]["device"] == "cpu"
    assert context.tensor_metadata["arg0"]["requires_grad"] is False


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
    assert set(context.optimizer["param_groups"][0]["parameter_keys"]) >= {
        "left.weight",
        "right.weight",
        "head.weight",
    }
    assert "state:0000" in context.optimizer["param_groups"][0]["canonical_state_ids"]
    assert context.scheduler["class_name"] == "StepLR"
    assert context.lifecycle["loss_computed_before_backward"] is True
    assert context.lifecycle["backward_called_before_optimizer_step"] is True
    assert context.lifecycle["scheduler_step_observed"] is True
    assert context.lifecycle["backward_call_count"] == 1
    assert context.distributed_mutation_observed_before_capture is False
    assert context.state_dict_key_to_canonical_state_id
    assert set(context.state_dict_key_to_canonical_state_id) >= {
        "left.weight",
        "right.weight",
        "head.weight",
    }


@pytest.mark.xfail(
    strict=True,
    reason=(
        "T068 expected-red: current dry-run capture enters real nn.Module.__call__ "
        "and runs autograd backward on CPU before planning; T070 must replace this "
        "with bounded metadata capture."
    ),
)
def test_dry_run_capture_does_not_enter_real_cpu_forward_or_backward(
    tmp_path: Path,
) -> None:
    context = _capture_module().capture_entrypoint(
        FIXTURE_ROOT / "capture_execution_sentinel_train.py",
        cwd=FIXTURE_ROOT,
        environment={"SHARDGRID_CAPTURE_SENTINEL_DIR": str(tmp_path)},
        dry_run=True,
    )

    assert not getattr(context, "ok", True) is False
    assert not (tmp_path / "forward_entered").exists()
    assert not (tmp_path / "backward_grad_computed").exists()


def test_dry_run_capture_suppresses_real_cpu_optimizer_step_mutation(
    tmp_path: Path,
) -> None:
    context = _capture_module().capture_entrypoint(
        FIXTURE_ROOT / "capture_execution_sentinel_train.py",
        cwd=FIXTURE_ROOT,
        environment={"SHARDGRID_CAPTURE_SENTINEL_DIR": str(tmp_path)},
        dry_run=True,
    )

    assert not getattr(context, "ok", True) is False
    assert (tmp_path / "optimizer_changed").read_text(encoding="utf-8") == "False"


def test_capture_contract_returns_structured_unsupported_diagnostics(tmp_path: Path) -> None:
    script = tmp_path / "unsupported_dynamic_control_flow_train.py"
    script.write_text("raise RuntimeError('dynamic control flow unsupported')\n", encoding="utf-8")
    module = _capture_module()

    result = module.capture_entrypoint(
        script,
        argv=(),
        cwd=tmp_path,
        environment={},
    )

    assert result.ok is False
    assert result.failure.stage == "capture"
    assert result.failure.code
    assert result.failure.message
    assert result.failure.artifact_log_ref
    assert "generic exception" not in result.failure.message.lower()


def test_capture_context_is_persisted_to_snapshot_plan_artifact(tmp_path: Path) -> None:
    context = _capture(
        "kwargs_hf_mapping_train.py",
        "--config",
        "user-model.yaml",
        "--checkpoint",
        "out/mapping.pt",
    )
    snapshot = _job_snapshot(tmp_path)

    output_path = write_capture_context(snapshot, context)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == Path(snapshot.plan_path) / "capture-context.json"
    assert payload["entrypoint_path"].endswith("kwargs_hf_mapping_train.py")
    assert payload["argv"] == ["--config", "user-model.yaml", "--checkpoint", "out/mapping.pt"]
    assert payload["cwd"] == str(FIXTURE_ROOT)
    assert payload["environment_summary"]["SHARDGRID_TEST_CAPTURE"] == "present"
    assert payload["capture_backend"] == "torch-monkeypatch"
    assert payload["capture_backend_version"] == "v1"
    assert payload["model_identity"]["class_name"] == "KeywordClassifier"
    assert payload["first_batch_structure"]["kind"] == "dict"
    assert payload["model_call"]["kwargs"]["fields"]["input_ids"]["kind"] == "tensor"
    assert payload["tensor_metadata"]["kwarg.input_ids"]["shape"] == [2, 5]
    assert payload["optimizer"]["class_name"] == "AdamW"
    assert payload["lifecycle"]["backward_called_before_optimizer_step"] is True
    assert payload["state_dict_key_to_canonical_state_id"]["embedding.weight"] == "state:0000"


def test_unsupported_capture_diagnostics_are_persisted_without_live_objects(
    tmp_path: Path,
) -> None:
    script = tmp_path / "unsupported_dynamic_control_flow_train.py"
    script.write_text("raise RuntimeError('dynamic control flow unsupported')\n", encoding="utf-8")
    result = _capture_module().capture_entrypoint(script, cwd=tmp_path, environment={})

    output_path = write_capture_context(_job_snapshot(tmp_path), result)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["ok"] is False
    assert payload["failure"]["stage"] == "capture"
    assert payload["failure"]["code"] == "RuntimeError"
    assert payload["failure"]["artifact_log_ref"] == "diagnostics/capture.json"
    assert payload["context"] is None


def test_graph_capture_unsupported_fails_before_distributed_mutation(
    tmp_path: Path,
) -> None:
    script = tmp_path / "dynamic_then_distributed.py"
    script.write_text(
        """
import torch
import torch.distributed as dist
from torch import nn


class DynamicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.positive = nn.Linear(2, 2)
        self.negative = nn.Linear(2, 2)

    def forward(self, x):
        if x.sum() > 0:
            return self.positive(x)
        return self.negative(x)


model = DynamicModel()
model(torch.ones(1, 2))
dist.init_process_group("gloo")
""".lstrip(),
        encoding="utf-8",
    )

    result = _capture_module().capture_entrypoint(script, cwd=tmp_path, environment={})

    assert result.ok is False
    assert result.failure.stage == "graph_capture"
    assert result.failure.code == "DYNAMIC_CONTROL_FLOW_UNSUPPORTED"
    assert result.failure.artifact_log_ref == "diagnostics/capture.json"


def test_gradient_accumulation_boundary_is_captured(tmp_path: Path) -> None:
    script = tmp_path / "gradient_accumulation_train.py"
    script.write_text(
        """
import torch
from torch import nn


class AccumulationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 1)

    def forward(self, x):
        return self.proj(x)


model = AccumulationModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
for value in (1.0, 2.0):
    loss = model(torch.full((2, 2), value)).sum() / 2
    loss.backward()
optimizer.step()
""".lstrip(),
        encoding="utf-8",
    )

    context = _capture_module().capture_entrypoint(script, cwd=tmp_path, environment={})

    assert context.optimizer["class_name"] == "AdamW"
    assert context.optimizer["param_groups"][0]["hyperparameters"]["lr"] == 0.005
    assert context.optimizer["param_groups"][0]["parameter_keys"] == [
        "proj.weight",
        "proj.bias",
    ]
    assert context.lifecycle["backward_call_count"] == 2
    assert context.lifecycle["gradient_accumulation_observed"] is True
    assert context.lifecycle["backward_called_before_optimizer_step"] is True


def test_amp_autocast_and_scaler_intent_are_captured(tmp_path: Path) -> None:
    script = tmp_path / "amp_train.py"
    script.write_text(
        """
import torch
from torch import nn


class AmpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 1)

    def forward(self, x):
        return self.proj(x)


model = AmpModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scaler = torch.amp.GradScaler("cpu", enabled=False)
with torch.amp.autocast("cpu"):
    loss = model(torch.ones(2, 2)).sum()
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
""".lstrip(),
        encoding="utf-8",
    )

    context = _capture_module().capture_entrypoint(script, cwd=tmp_path, environment={})

    assert context.optimizer["class_name"] == "Adam"
    assert context.lifecycle["amp_autocast_observed"] is True
    assert context.lifecycle["amp_scaler_observed"] is True
    assert context.lifecycle["backward_called_before_optimizer_step"] is True


def test_hidden_custom_optimizer_mutation_is_rejected_before_training(
    tmp_path: Path,
) -> None:
    script = tmp_path / "custom_optimizer_train.py"
    script.write_text(
        """
import torch
from torch import nn


class HiddenMutationOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {"lr": 0.1})

    def step(self, closure=None):
        for group in self.param_groups:
            for param in group["params"]:
                param.data.add_(1.0)


model = nn.Linear(2, 1)
optimizer = HiddenMutationOptimizer(model.parameters())
model(torch.ones(1, 2)).sum().backward()
optimizer.step()
""".lstrip(),
        encoding="utf-8",
    )

    result = _capture_module().capture_entrypoint(script, cwd=tmp_path, environment={})

    assert result.ok is False
    assert result.failure.stage == "lifecycle_capture"
    assert result.failure.code == "HIDDEN_OPTIMIZER_MUTATION_UNSUPPORTED"


def test_ordinary_training_script_fixtures_do_not_import_shardgrid_user_api() -> None:
    forbidden = ("shardgrid", "ModelProvider", "sample_inputs", "compute_loss", "build_model")
    offenders: list[str] = []
    for path in sorted(FIXTURE_ROOT.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if any(token in source for token in forbidden):
            offenders.append(path.name)

    assert offenders == []
