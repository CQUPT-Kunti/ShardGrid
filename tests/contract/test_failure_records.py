from __future__ import annotations

import pytest

from shardgrid.common.enums import FailureStage
from shardgrid.common.models import as_worker_id
from shardgrid.jobs.models import FailureRecord


def test_failure_record_contract_round_trip() -> None:
    record = FailureRecord(
        stage=FailureStage.TRAIN,
        host="machine-d.local",
        worker_id=as_worker_id("gpu1060"),
        command="python3 train.py",
        exit_code=2,
        stdout_path="/tmp/train.stdout",
        stderr_path="/tmp/train.stderr",
        message="training step failed",
        recommended_action="inspect worker logs and retry",
        retryable=False,
        manual_action_required=False,
    )

    restored = FailureRecord.from_dict(record.to_dict())

    assert restored == record
    assert restored.to_dict()["stage"] == "TRAIN"


def test_failure_record_contract_rejects_incomplete_payloads() -> None:
    with pytest.raises(ValueError, match="failure message"):
        FailureRecord(
            stage=FailureStage.BOOTSTRAP,
            host="machine-a.local",
            recommended_action="rerun bootstrap",
        )

    with pytest.raises(ValueError, match="recommended_action"):
        FailureRecord(
            stage=FailureStage.PROBE,
            host="machine-c.local",
            message="probe failed",
        )
