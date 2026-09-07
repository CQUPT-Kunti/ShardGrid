from __future__ import annotations

import json

from shardgrid.cli.app import main
from shardgrid.cli.commands import dist_test as dist_test_command
from shardgrid.distributed.collectives import RankCollectiveResult
from shardgrid.distributed.fallback import FallbackEligibility

_CONFIG = "--config"
_WORKERS_YAML = "examples/workers.yaml"


def _rank_result(
    rank: int,
    worker_id: str,
    *,
    init_ok: bool,
    backend: str,
    gpu: str,
    interface: str,
) -> RankCollectiveResult:
    result = {
        "init_ok": init_ok,
        "broadcast_ok": init_ok,
        "send_recv_ok": init_ok,
        "all_reduce_ok": init_ok,
        "broadcast_tensor": [11.0, 22.0, 33.0, 44.0],
        "send_recv_tensor": [5.0, 6.0, 7.0, 8.0],
        "all_reduce_tensor": [3.0, 3.0, 3.0, 3.0],
        "error": None if init_ok else f"{backend} rendezvous failed",
        "conda_environment": "shardgrid",
        "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
        "python_executable": "/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
        "python_version": "3.10.12",
        "torch_version": "2.7.1+cu118",
        "torch_cuda_version": "11.8",
        "cuda_available": True,
        "gpu_name": gpu,
        "network_interface": interface,
        "master_addr": "10.87.5.155",
        "master_port": 29500,
        "route_output": f"peer dev {interface} src 10.87.5.155",
        "port_range": "32768 60999",
        "stages": ["BEFORE_INIT", "AFTER_INIT"],
        "elapsed_s": 10.0,
        "backend": backend,
    }
    return RankCollectiveResult(
        rank=rank,
        worker_id=worker_id,
        exit_code=0 if init_ok else 1,
        timed_out=False,
        result=result,
        stdout="",
        stderr="",
    )


def _fake_infos() -> list[dict]:
    return [
        {
            "wrapper": object(),
            "worker_id": "gpu4060",
            "rank": 0,
            "ip": "10.87.5.155",
            "hostname": "LDJ",
            "gpu_model": "NVIDIA GeForce RTX 4060",
        },
        {
            "wrapper": object(),
            "worker_id": "gpu1060",
            "rank": 1,
            "ip": "10.87.5.15",
            "hostname": "LAPTOP-5G3QUOGM",
            "gpu_model": "NVIDIA GeForce GTX 1650",
        },
    ]


def _ok_baseline() -> tuple[FallbackEligibility, dict]:
    eligibility = FallbackEligibility(
        network_ok=True, rendezvous_ok=True, runtime_ok=True
    )
    baseline = {
        "interfaces": {"0": "eth3", "1": "eth0"},
        "workers": {
            "0": {"interface": "eth3", "ssh_reachable": True, "runtime_configured": True},
            "1": {"interface": "eth0", "ssh_reachable": True, "runtime_configured": True},
        },
    }
    return eligibility, baseline


def _blocked_baseline() -> tuple[FallbackEligibility, dict]:
    eligibility = FallbackEligibility(
        network_ok=False, rendezvous_ok=True, runtime_ok=True
    )
    baseline = {
        "interfaces": {"0": "", "1": "eth0"},
        "workers": {
            "0": {"interface": None, "ssh_reachable": False, "runtime_configured": True},
            "1": {"interface": "eth0", "ssh_reachable": True, "runtime_configured": True},
        },
    }
    return eligibility, baseline


def _install_fakes(
    monkeypatch,
    *,
    nccl_ok: bool = True,
    gloo_ok: bool = True,
    baseline: tuple[FallbackEligibility, dict] | None = None,
    called: list[str] | None = None,
) -> None:
    monkeypatch.setattr(
        dist_test_command, "_build_wrappers", lambda config, workers: _fake_infos()
    )
    monkeypatch.setattr(
        dist_test_command,
        "_check_baseline",
        lambda infos: baseline if baseline is not None else _ok_baseline(),
    )

    def fake_run_collectives(
        infos, *, backend, master_addr, master_port, interfaces, timeout=180.0
    ):
        if called is not None:
            called.append(backend)
        ok = {"nccl": nccl_ok, "gloo": gloo_ok}[backend]
        return (
            _rank_result(
                0, "gpu4060", init_ok=ok, backend=backend, gpu="RTX 4060", interface="eth3"
            ),
            _rank_result(
                1, "gpu1060", init_ok=ok, backend=backend, gpu="GTX 1650", interface="eth0"
            ),
        )

    monkeypatch.setattr(dist_test_command, "_run_collectives", fake_run_collectives)


def _dist_test_args(
    *,
    backend: str,
    workers: str,
    json_output: bool = False,
    save_report: str | None = None,
) -> list[str]:
    args = [_CONFIG, _WORKERS_YAML]
    if json_output:
        args.append("--json")
    args.extend(["dist-test", "--backend", backend, "--workers", workers])
    if save_report is not None:
        args.extend(["--save-report", save_report])
    return args


def test_dist_test_is_registered_as_real_command(capsys) -> None:
    try:
        main(["--help"])
    except SystemExit as error:
        assert error.code == 0

    captured = capsys.readouterr()
    assert "dist-test" in captured.out
    assert "placeholder command" not in captured.out


def test_dist_test_requires_config(capsys) -> None:
    exit_code = main(
        ["dist-test", "--backend", "nccl", "--workers", "gpu4060,gpu1060"]
    )

    captured = capsys.readouterr()
    assert exit_code == 3
    assert "requires a cluster config" in captured.out


def test_dist_test_requires_workers(capsys) -> None:
    exit_code = main(
        [_CONFIG, _WORKERS_YAML, "dist-test", "--backend", "nccl"]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--workers is required" in captured.out


def test_dist_test_invalid_backend(capsys) -> None:
    exit_code = main(
        [_CONFIG, _WORKERS_YAML, "dist-test", "--backend", "bogus",
         "--workers", "gpu4060,gpu1060"]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "invalid choice" in captured.err


def test_dist_test_invalid_worker(capsys) -> None:
    exit_code = main(
        [_CONFIG, _WORKERS_YAML, "dist-test", "--backend", "nccl",
         "--workers", "nope,gpu1060"]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "unknown worker id" in captured.out


def test_dist_test_nccl_success(monkeypatch, capsys) -> None:
    _install_fakes(monkeypatch, nccl_ok=True)
    exit_code = main(_dist_test_args(backend="nccl", workers="gpu4060,gpu1060"))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "NCCL SUCCESS" in captured.out


def test_dist_test_nccl_failure_no_fallback(monkeypatch, capsys) -> None:
    called: list[str] = []
    _install_fakes(monkeypatch, nccl_ok=False, called=called)
    exit_code = main(_dist_test_args(backend="nccl", workers="gpu4060,gpu1060"))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert called == ["nccl"]
    assert "NCCL FAILED" in captured.out


def test_dist_test_gloo_direct(monkeypatch, capsys) -> None:
    _install_fakes(monkeypatch, gloo_ok=True)
    exit_code = main(_dist_test_args(backend="gloo", workers="gpu4060,gpu1060"))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "GLOO PASS" in captured.out
    assert "gloo" in captured.out


def test_dist_test_auto_falls_back_to_gloo(monkeypatch, capsys) -> None:
    called: list[str] = []
    _install_fakes(monkeypatch, nccl_ok=False, gloo_ok=True, called=called)
    exit_code = main(
        _dist_test_args(backend="auto", workers="gpu4060,gpu1060", json_output=True)
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert called == ["nccl", "gloo"]
    payload = json.loads(captured.out)
    assert payload["backend_state"] == "GLOO FALLBACK"
    assert payload["backend_actual"] == "gloo"
    assert payload["backend_requested"] == "auto"
    assert payload["status"] == "PASS"
    assert payload["nccl_failure_evidence"] is not None
    assert payload["gloo_fallback"] is not None
    assert payload["nccl_failure_evidence"]["outcome"] == "FAILED"
    assert payload["nccl_failure_evidence"]["backend"] == "nccl"
    assert payload["gloo_fallback"]["outcome"] == "PASS"
    assert payload["gloo_fallback"]["backend"] == "gloo"


def test_dist_test_auto_fallback_failure(monkeypatch, capsys) -> None:
    called: list[str] = []
    _install_fakes(monkeypatch, nccl_ok=False, gloo_ok=False, called=called)
    exit_code = main(
        _dist_test_args(backend="auto", workers="gpu4060,gpu1060", json_output=True)
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert called == ["nccl", "gloo"]
    payload = json.loads(captured.out)
    assert payload["backend_state"] == "FALLBACK FAILED"
    assert payload["status"] == "FAIL"


def test_dist_test_baseline_blocked_no_fallback(monkeypatch, capsys) -> None:
    called: list[str] = []
    _install_fakes(
        monkeypatch, nccl_ok=False, baseline=_blocked_baseline(), called=called
    )
    exit_code = main(
        _dist_test_args(backend="auto", workers="gpu4060,gpu1060", json_output=True)
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert called == []
    payload = json.loads(captured.out)
    assert payload["backend_state"] == "FALLBACK NOT ALLOWED"
    assert payload["status"] == "BLOCKED"


def test_dist_test_save_report(monkeypatch, tmp_path, capsys) -> None:
    _install_fakes(monkeypatch, nccl_ok=True)
    report_path = tmp_path / "reports" / "dist-test.json"
    exit_code = main(
        _dist_test_args(
            backend="nccl",
            workers="gpu4060,gpu1060",
            save_report=str(report_path),
        )
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert report_path.exists()
    payload = json.loads(report_path.read_text())
    assert payload["backend_state"] == "NCCL SUCCESS"
    assert payload["status"] == "PASS"
    assert payload["collectives"]["nccl"]["process_group"]["ok"] is True
    assert payload["collectives"]["nccl"]["broadcast"]["ok"]["rank0"] is True
    assert payload["collectives"]["nccl"]["send_recv"]["tensor"]["rank1"] == [
        5.0, 6.0, 7.0, 8.0
    ]
    assert payload["collectives"]["nccl"]["all_reduce"]["tensor"]["rank0"] == [
        3.0, 3.0, 3.0, 3.0
    ]
    assert payload["runtime_evidence"]["torch_version"] == "2.7.1+cu118"
    assert payload["runtime_evidence"]["conda_environment"] == "shardgrid"
    assert len(payload["workers"]) == 2
    assert "report saved" in captured.out


def test_dist_test_report_has_no_secrets(monkeypatch, tmp_path, capsys) -> None:
    _install_fakes(monkeypatch, nccl_ok=False, gloo_ok=True)
    report_path = tmp_path / "reports" / "dist-test.json"
    main(
        _dist_test_args(
            backend="auto",
            workers="gpu4060,gpu1060",
            save_report=str(report_path),
        )
    )
    capsys.readouterr()

    text = report_path.read_text().lower()
    for secret in (
        "password",
        "private_key",
        "credentials",
        "090157",
        "a2031291160",
        "known_hosts",
    ):
        assert secret not in text, f"report must not contain {secret!r}"


def test_dist_test_json_output_backend_labels(monkeypatch, capsys) -> None:
    _install_fakes(monkeypatch, nccl_ok=False, gloo_ok=True)
    exit_code = main(
        _dist_test_args(backend="auto", workers="gpu4060,gpu1060", json_output=True)
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["backend_state"] != "NCCL SUCCESS"
    assert payload["backend_state"] != "NCCL PASS"
    assert payload["backend_state"] == "GLOO FALLBACK"
    assert payload["elapsed_s"] is not None
    assert payload["network"]["master_addr"] == "10.87.5.155"
    assert payload["network"]["interfaces"] == {"0": "eth3", "1": "eth0"}
