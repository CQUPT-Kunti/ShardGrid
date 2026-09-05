from __future__ import annotations

from pathlib import Path

from tests.multi_host import large_model_gate as gate


def test_wait_for_job_training_steps_returns_when_all_ranks_reach_min_steps(
    monkeypatch,
    tmp_path: Path,
) -> None:
    payloads = [
        {"rank": 0, "train": {"rank": 0, "steps": 53}, "process_state": "alive"},
        {"rank": 1, "train": {"rank": 1, "steps": 55}, "process_state": "alive"},
    ]
    monkeypatch.setattr(gate, "job_status_payload", lambda root, job_id: {"state": "training", "world_size": 2})
    monkeypatch.setattr(gate, "monitor_payloads", lambda root, job_id: payloads)
    sleeps: list[int] = []
    monkeypatch.setattr(gate.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = gate.wait_for_job_training_steps(tmp_path, "job-1", min_steps=3, timeout=5)

    assert result == payloads
    assert sleeps == []


def test_wait_for_job_training_steps_does_not_require_equal_rank_steps(
    monkeypatch,
    tmp_path: Path,
) -> None:
    payloads = [
        {"rank": 0, "train": {"rank": 0, "steps": 3}, "process_state": "alive"},
        {"rank": 1, "train": {"rank": 1, "steps": 5}, "process_state": "alive"},
    ]
    monkeypatch.setattr(
        gate,
        "job_status_payload",
        lambda root, job_id: {"state": "training", "assignments": [{}, {}]},
    )
    monkeypatch.setattr(gate, "monitor_payloads", lambda root, job_id: payloads)
    sleeps: list[int] = []
    monkeypatch.setattr(gate.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = gate.wait_for_job_training_steps(tmp_path, "job-2", min_steps=3, timeout=5)

    assert result == payloads
    assert sleeps == []


def test_wait_for_job_training_steps_returns_empty_on_terminal_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    payloads = [
        {"rank": 0, "process_state": "exited"},
        {"rank": 1, "process_state": "alive"},
    ]
    monkeypatch.setattr(
        gate,
        "job_status_payload",
        lambda root, job_id: {"state": "failed", "assignments": [{}, {}]},
    )
    monkeypatch.setattr(gate, "monitor_payloads", lambda root, job_id: payloads)
    sleeps: list[int] = []
    monkeypatch.setattr(gate.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = gate.wait_for_job_training_steps(tmp_path, "job-failed", min_steps=3, timeout=5)

    assert result == []
    assert sleeps == []


def test_start_train_does_not_capture_pipes(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class DummyProc:
        pass

    def fake_popen(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return DummyProc()

    monkeypatch.setattr(gate.subprocess, "Popen", fake_popen)

    proc = gate.start_train(tmp_path / "workers.yaml", tmp_path / "train.yaml")

    assert isinstance(proc, DummyProc)
    assert captured["kwargs"]["stdout"] is gate.subprocess.DEVNULL
    assert captured["kwargs"]["stderr"] is gate.subprocess.DEVNULL


def test_three_worker_packing_candidates_use_lightweight_long_training_defaults() -> None:
    memory = {
        "gpu4060": 6 * gate.BYTES_PER_MB * 1024,
        "gpu1060": 3 * gate.BYTES_PER_MB * 1024,
        "gpu4060-cqupt": 6 * gate.BYTES_PER_MB * 1024,
    }

    medium = gate.three_worker_medium_packing_candidates(memory)
    small = gate.three_worker_small_packing_candidates(memory)

    assert medium
    assert small
    assert all(candidate["batch_size"] == 1 for candidate in medium + small)
    assert all(candidate["microbatch_count"] == 1 for candidate in medium + small)
    assert all(candidate["training_steps"] == 1000 for candidate in medium + small)
