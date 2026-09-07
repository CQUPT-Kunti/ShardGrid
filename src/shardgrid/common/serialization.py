"""Stable JSON/YAML serialization and schema validation helpers."""

from __future__ import annotations

import json
import math
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, TypeVar, cast

import yaml
from jsonschema import Draft202012Validator

from shardgrid.common.enums import JobState
from shardgrid.jobs.models import JobStatus
from shardgrid.planner.models import ExecutionPlan
from shardgrid.resources.models import NetworkState, WorkerResource

SchemaName = str
ModelType = TypeVar(
    "ModelType", WorkerResource, NetworkState, ExecutionPlan, JobStatus
)


class SupportsSerde(Protocol):
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any: ...

    def to_dict(self) -> dict[str, Any]: ...


_SCHEMA_FILES: dict[SchemaName, Path] = {
    "execution_plan": Path(
        "specs/001-multi-host-training-mvp/contracts/execution-plan.schema.yaml"
    ),
    "job_status": Path(
        "specs/001-multi-host-training-mvp/contracts/job-status.schema.yaml"
    ),
}


class SchemaValidationError(ValueError):
    """Raised when serialized model data does not satisfy a schema."""



def serialize_json(model: SupportsSerde) -> str:
    return json.dumps(model.to_dict(), indent=2, sort_keys=True)


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(
        to_json_compatible(payload),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )


def to_json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not valid stable JSON metadata")
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_json_compatible(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): to_json_compatible(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_json_compatible(item) for item in value]
    raise TypeError(f"cannot serialize live Python object as stable JSON: {type(value).__name__}")



def serialize_yaml(model: SupportsSerde) -> str:
    return yaml.safe_dump(model.to_dict(), sort_keys=True)



def deserialize_json(text: str, model_type: type[ModelType]) -> ModelType:
    return cast(ModelType, model_type.from_dict(json.loads(text)))



def deserialize_yaml(text: str, model_type: type[ModelType]) -> ModelType:
    return cast(ModelType, model_type.from_dict(yaml.safe_load(text)))



def dump_json(model: SupportsSerde, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.write_text(serialize_json(model))
    return output_path



def dump_yaml(model: SupportsSerde, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.write_text(serialize_yaml(model))
    return output_path



def load_json(path: str | Path, model_type: type[ModelType]) -> ModelType:
    return deserialize_json(Path(path).read_text(), model_type)



def load_yaml(path: str | Path, model_type: type[ModelType]) -> ModelType:
    return deserialize_yaml(Path(path).read_text(), model_type)



def load_schema(name: SchemaName) -> dict[str, Any]:
    schema_path = _SCHEMA_FILES[name]
    return cast(dict[str, Any], yaml.safe_load(schema_path.read_text()))



def validate_schema_data(name: SchemaName, payload: dict[str, Any]) -> None:
    schema = load_schema(name)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        message = "; ".join(error.message for error in errors)
        raise SchemaValidationError(message)

    if name == "execution_plan":
        _validate_execution_plan_payload(payload)
    elif name == "job_status":
        _validate_job_status_payload(payload)


def _validate_execution_plan_payload(payload: dict[str, Any]) -> None:
    workers = payload.get("workers", [])
    world_size = payload.get("world_size")
    ranks = [worker["rank"] for worker in workers]

    if len(ranks) != len(set(ranks)):
        raise SchemaValidationError("worker ranks must be unique")
    if any(worker.get("local_rank") != 0 for worker in workers):
        raise SchemaValidationError("worker local_rank must be 0 in Stage A-C")
    if world_size != len(workers):
        raise SchemaValidationError("world_size must equal number of workers")


def _validate_job_status_payload(payload: dict[str, Any]) -> None:
    state = payload.get("state")
    failure = payload.get("failure")
    checkpoint_ref = payload.get("checkpoint_ref")
    final_metrics = payload.get("final_metrics")

    if state == JobState.FAILED.value and not isinstance(failure, dict):
        raise SchemaValidationError("failed job status must include failure record")
    if state == JobState.COMPLETED.value and not checkpoint_ref:
        raise SchemaValidationError("completed job status must include checkpoint_ref")
    if state == JobState.COMPLETED.value and (
        not isinstance(final_metrics, dict) or "final_loss" not in final_metrics
    ):
        raise SchemaValidationError(
            "completed job status must include final_metrics.final_loss"
        )



def validate_execution_plan(plan: ExecutionPlan) -> None:
    validate_schema_data("execution_plan", plan.to_dict())



def validate_job_status(status: JobStatus) -> None:
    validate_schema_data("job_status", status.to_dict())
