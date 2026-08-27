from __future__ import annotations

from shardgrid.bootstrap.runner import BootstrapExecution
from shardgrid.common.config import ClusterConfig
from shardgrid.common.enums import Health
from shardgrid.control import doctor as doctor_module
from shardgrid.transport.remote_access import RemoteAccessResult, RemoteRuntimeIdentity
from shardgrid.transport.ssh import SSHOptions, SSHTransport


def _config() -> ClusterConfig:
    return ClusterConfig.from_dict(
        {
            "control": {"machine_id": "machine-a", "hostname": "control-a.local"},
            "jobs_root": "/tmp/shardgrid-jobs",
            "ssh": {},
            "runtime": {},
            "network": {"nccl_mtu": 1500},
            "backend_preference": {},
            "manual_override": {},
            "workers": [
                {
                    "id": "gpu4060",
                    "machine_id": "machine-c",
                    "physical_os": "windows",
                    "runtime_os": "wsl2_linux",
                    "runtime": "wsl2",
                    "host": "10.87.5.155",
                    "ssh_user": "shardgrid",
                    "runtime_distro": "Ubuntu-22.04",
                    "conda_environment": "shardgrid",
                    "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
                },
                {
                    "id": "gpu1060",
                    "machine_id": "machine-d",
                    "physical_os": "windows",
                    "runtime_os": "wsl2_linux",
                    "runtime": "wsl2",
                    "host": "10.87.5.15",
                    "ssh_user": "shardgrid",
                    "runtime_distro": "Ubuntu-22.04",
                    "conda_environment": "shardgrid",
                    "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
                },
            ],
        }
    )


def _transport() -> SSHTransport:
    return SSHTransport(SSHOptions(host="10.87.5.155", user="shardgrid"))


def _access(worker_id: str, host: str) -> RemoteAccessResult:
    return RemoteAccessResult(
        status="PASS",
        worker_id=worker_id,
        host=host,
        ssh_user="shardgrid",
        transport=_transport(),
        commands=("ssh host hostname", "ssh host wsl.exe -l -v"),
        windows_identity="LDJ" if worker_id == "gpu4060" else "LAPTOP-5G3QUOGM",
        wsl_distro="Ubuntu-22.04",
        runtime_identity=RemoteRuntimeIdentity(
            windows_identity="LDJ" if worker_id == "gpu4060" else "LAPTOP-5G3QUOGM",
            wsl_distro="Ubuntu-22.04",
            conda_executable="/home/shardgrid/miniconda3/bin/conda",
            conda_environment="shardgrid",
            conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
            python_version="Python 3.12.13",
        ),
    )


def _bootstrap(*, interface: str, mtu: int) -> dict:
    return {
        "conda": {
            "executable": "/home/shardgrid/miniconda3/bin/conda",
            "environments": ["base", "shardgrid"],
            "active_environment": "shardgrid",
            "selected_environment": "shardgrid",
            "selected_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
        },
        "python": {
            "executable": "/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
            "version": "Python 3.12.13",
        },
        "runtime_tools": {
            "git": "git version 2.43.0",
            "iperf3": "iperf 3.12",
        },
        "project_dependencies": {"status": "present"},
        "nccl_path_mtu": {
            "peer_ip": "10.87.5.15",
            "expected_mtu": "1500",
            "route_output": f"10.87.5.15 dev {interface} src 10.87.5.155",
            "interface": interface,
            "interface_mtu_before": str(mtu),
            "interface_mtu_after": str(mtu),
            "status": "PASS" if mtu == 1500 else "NCCL_PATH_MTU_UNSAFE",
            "df_1472": "PASS",
            "df_1473": "EXPECTED_BLOCK",
        },
        "commands_run": ["ip route get 10.87.5.15", f"ip link show dev {interface}"],
        "manual_actions": [],
    }


def _bootstrap_execution(payload: dict, *, execution: str = "skipped", verified: bool = True):
    return BootstrapExecution(
        target="10.87.5.15",
        action="bootstrap-wsl --check",
        before_state=payload,
        execution=execution,
        after_verification=payload,
        verified=verified,
        commands_run=tuple(str(item) for item in payload.get("commands_run", [])),
        manual_action=next(iter(payload.get("manual_actions", [])), None),
    )


def _runtime_probe() -> dict:
    return {
        "python_executable": "/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
        "python_version": "3.12.13",
        "torch_version": "2.7.1+cu118",
        "torch_cuda_version": "11.8",
        "cuda_available": True,
        "nccl_available": True,
        "nccl_version": "2.21.5",
        "nccl_lib_path": "/home/shardgrid/miniconda3/envs/shardgrid/lib/libnccl.so.2",
        "gloo_available": True,
        "gpu_name": "NVIDIA GeForce RTX 4060",
        "gpu_total_memory_mb": 8188,
        "compute_capability": "8.9",
        "driver_version": "555.85",
        "disk_free_bytes": 50 * 1024**3,
    }


def test_worker_doctor_reports_dynamic_interface_and_unsafe_mtu(monkeypatch) -> None:
    config = _config()
    worker = config.workers[0]

    monkeypatch.setattr(
        doctor_module,
        "run_remote_access_check",
        lambda transport, worker, worker_label, preferred_environment=None: _access(
            str(worker.worker_id), str(worker.host)
        ),
    )
    monkeypatch.setattr(
        doctor_module,
        "_bootstrap_worker_runtime",
        lambda wrapper, peer_ip, expected_mtu, fix: _bootstrap_execution(
            _bootstrap(interface="eth3", mtu=2800)
        ),
    )
    monkeypatch.setattr(doctor_module, "_probe_runtime_details", lambda wrapper: _runtime_probe())

    report = doctor_module._run_worker_doctor(config, worker, fix=False)

    assert report.health == Health.FAILED
    assert report.environment["windows_identity"] == "LDJ"
    assert report.environment["wsl_distro"] == "Ubuntu-22.04"
    route_check = next(check for check in report.checks if check.name == "peer_route")
    mtu_check = next(check for check in report.checks if check.name == "nccl_path_mtu")
    assert route_check.status == "PASS"
    assert route_check.detected_value["interface"] == "eth3"
    assert mtu_check.status == "FAIL"
    assert "NCCL_PATH_MTU_UNSAFE" in (mtu_check.failure_reason or "")


def test_worker_doctor_fix_reuses_existing_mtu_helper(monkeypatch) -> None:
    config = _config()
    worker = config.workers[0]
    calls: list[tuple[bool, int]] = []

    monkeypatch.setattr(
        doctor_module,
        "run_remote_access_check",
        lambda transport, worker, worker_label, preferred_environment=None: _access(
            str(worker.worker_id), str(worker.host)
        ),
    )

    def fake_bootstrap(wrapper, peer_ip, expected_mtu, fix):
        calls.append((fix, len(calls)))
        if fix:
            return _bootstrap_execution(
                _bootstrap(interface="eth1", mtu=1500),
                execution="executed",
            )
        return _bootstrap_execution(_bootstrap(interface="eth1", mtu=2800))

    monkeypatch.setattr(doctor_module, "_bootstrap_worker_runtime", fake_bootstrap)
    monkeypatch.setattr(doctor_module, "_probe_runtime_details", lambda wrapper: _runtime_probe())

    report = doctor_module._run_worker_doctor(config, worker, fix=True)

    mtu_check = next(check for check in report.checks if check.name == "nccl_path_mtu")
    assert report.health == Health.HEALTHY
    assert report.exit_code == 0
    assert mtu_check.status == "PASS"
    assert calls[0][0] is False
    assert any(fix for fix, _ in calls)
    assert not report.manual_actions


def test_run_doctor_all_aggregates_control_and_workers(monkeypatch) -> None:
    config = _config()
    control = doctor_module.DoctorSubjectReport(
        subject="control",
        subject_type="control",
        host="control-a.local",
        runtime="control",
        physical_os="linux",
        runtime_os="linux",
        timestamp="2026-08-27T00:00:00+00:00",
        checks=(),
        environment={},
        health=Health.HEALTHY,
        manual_actions=(),
        commands_run=(),
        exit_code=0,
    )
    worker = doctor_module.DoctorSubjectReport(
        subject="gpu4060",
        subject_type="worker",
        host="10.87.5.155",
        runtime="wsl2",
        physical_os="windows",
        runtime_os="wsl2_linux",
        timestamp="2026-08-27T00:00:00+00:00",
        checks=(),
        environment={},
        health=Health.HEALTHY,
        manual_actions=(),
        commands_run=(),
        exit_code=0,
    )
    monkeypatch.setattr(
        doctor_module,
        "run_control_doctor",
        lambda cfg=None, *, fix=False: control,
    )
    monkeypatch.setattr(
        doctor_module,
        "_run_worker_doctor",
        lambda cfg, candidate, fix=False: worker,
    )

    report = doctor_module.run_doctor("all", config=config, fix=False)

    assert report.target == "all"
    assert len(report.subjects) == 3
    assert report.exit_code == 0
