from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXECUTABLE = sys.executable
RUNTIME_DISTRO = "Ubuntu-22.04"
RUNTIME_USER = "shardgrid"
RUNTIME_PREFIX = "/home/shardgrid/miniconda3/envs/shardgrid"
RUNTIME_PYTHON = f"{RUNTIME_PREFIX}/bin/python"


def _require_password() -> str:
    password = os.environ.get("SHARDGRID_TEST_SSH_PASSWORD")
    if not password:
        pytest.skip("set SHARDGRID_TEST_SSH_PASSWORD to run live doctor hardware tests")
    return password


def _run(command: list[str], *, env: dict[str, str] | None = None, timeout: int = 180) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        pytest.fail(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )
    return completed.stdout.strip()


def _ssh_base(password: str, host: str) -> list[str]:
    return [
        "sshpass",
        "-p",
        password,
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        f"{RUNTIME_USER}@{host}",
    ]


def _remote_windows(password: str, host: str, command: str) -> str:
    return _run(_ssh_base(password, host) + [command], timeout=90)


def _remote_wsl(password: str, host: str, command: str) -> str:
    return _run(
        _ssh_base(password, host)
        + [f"wsl.exe -d {RUNTIME_DISTRO} -u {RUNTIME_USER} -- {command}"],
        timeout=120,
    )


def _remote_torch_probe(password: str, host: str) -> dict[str, object]:
    script = (
        "import json,platform,torch;"
        "print(json.dumps({"
        "'python': platform.python_version(),"
        "'torch': torch.__version__,"
        "'cuda': torch.version.cuda,"
        "'cuda_available': torch.cuda.is_available(),"
        "'device_count': torch.cuda.device_count(),"
        "'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
        "'nccl_version': '.'.join(str(x) for x in torch.cuda.nccl.version()) "
        "if torch.cuda.is_available() and torch.cuda.nccl.version() else None"
        "}))"
    )
    output = _remote_wsl(password, host, f'{RUNTIME_PYTHON} -c "{script}"')
    return json.loads(output)


def _worker_doctor_subject(
    tmp_path: Path,
    *,
    password: str,
    worker_id: str,
) -> dict[str, object]:
    config_path = tmp_path / "workers-live.yaml"
    config_path.write_text(
        "\n".join(
            [
                "control:",
                "  machine_id: machine-a",
                "  hostname: control-a.local",
                "jobs_root: /tmp/shardgrid-jobs",
                "ssh:",
                "  strict_host_key_checking: false",
                "  known_hosts_file: /dev/null",
                "  connect_timeout_seconds: 20",
                "runtime:",
                "  conda_environment: shardgrid",
                "network:",
                "  nccl_mtu: 1500",
                "workers:",
                "  - id: gpu4060",
                "    machine_id: machine-c",
                "    physical_os: windows",
                "    runtime_os: wsl2_linux",
                "    runtime: wsl2",
                "    host: 10.87.5.155",
                "    ssh_user: shardgrid",
                "    runtime_distro: Ubuntu-22.04",
                "    conda_environment: shardgrid",
                "    conda_prefix: /home/shardgrid/miniconda3/envs/shardgrid",
                "    labels:",
                "      gpu: NVIDIA GeForce RTX 4060 Laptop GPU",
                "  - id: gpu1060",
                "    machine_id: machine-d",
                "    physical_os: windows",
                "    runtime_os: wsl2_linux",
                "    runtime: wsl2",
                "    host: 10.87.5.15",
                "    ssh_user: shardgrid",
                "    runtime_distro: Ubuntu-22.04",
                "    conda_environment: shardgrid",
                "    conda_prefix: /home/shardgrid/miniconda3/envs/shardgrid",
                "    labels:",
                "      gpu: NVIDIA GeForce GTX 1650",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    wrapper = wrapper_dir / "ssh"
    wrapper.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "args=()",
                "while [[ $# -gt 0 ]]; do",
                "  case \"$1\" in",
                "    -o)",
                "      if [[ $# -ge 2 && \"$2\" == \"BatchMode=yes\" ]]; then",
                "        shift 2",
                "        continue",
                "      fi",
                "      if [[ $# -ge 2 ]]; then",
                "        args+=(\"$1\" \"$2\")",
                "        shift 2",
                "        continue",
                "      fi",
                "      ;;",
                "    -oBatchMode=yes)",
                "      shift",
                "      continue",
                "      ;;",
                "  esac",
                "  args+=(\"$1\")",
                "  shift",
                "done",
                "exec /usr/bin/sshpass -p \"$SHARDGRID_TEST_SSH_PASSWORD\" "
                "/usr/bin/ssh \"${args[@]}\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PATH"] = f"{wrapper_dir}:{env['PATH']}"
    env["SHARDGRID_TEST_SSH_PASSWORD"] = password
    output = _run(
        [
            PYTHON_EXECUTABLE,
            "-c",
            (
                "import json; "
                "from shardgrid.common.config import load_cluster_config; "
                "from shardgrid.control.doctor import _run_worker_doctor; "
                f"config = load_cluster_config({str(config_path)!r}); "
                "worker = next("
                "item for item in config.workers "
                f"if str(item.worker_id) == {worker_id!r}"
                "); "
                "print(json.dumps(_run_worker_doctor(config, worker, fix=False).to_dict()))"
            ),
        ],
        env=env,
        timeout=180,
    )
    return json.loads(output)


def verify_live_worker_doctor(
    tmp_path: Path,
    *,
    worker_id: str,
    host: str,
    peer_host: str,
    expected_gpu: str,
    expected_hostname: str,
    expected_interface: str,
) -> None:
    password = _require_password()
    if not shutil.which("sshpass"):
        pytest.skip("sshpass is required for live doctor hardware tests")
    subject = _worker_doctor_subject(tmp_path, password=password, worker_id=worker_id)
    assert subject["health"] == "healthy"
    checks = {check["name"]: check for check in subject["checks"]}

    hostname = _remote_windows(password, host, "hostname").strip()
    assert hostname.lower() == expected_hostname.lower()
    assert checks["windows_identity"]["status"] == "PASS"
    assert str(checks["windows_identity"]["detected_value"]).lower() == hostname.lower()

    nvidia_smi = _remote_wsl(
        password,
        host,
        "/usr/lib/wsl/lib/nvidia-smi "
        "--query-gpu=name,driver_version,memory.total --format=csv,noheader",
    ).strip()
    gpu_name, driver_version, memory_total = [item.strip() for item in nvidia_smi.split(",")]
    assert expected_gpu in gpu_name
    gpu_check = checks["gpu"]
    assert gpu_check["status"] == "PASS"
    assert gpu_check["detected_value"]["name"] == gpu_name
    assert str(gpu_check["detected_value"]["driver_version"]) == driver_version
    assert memory_total.endswith("MiB")

    torch_probe = _remote_torch_probe(password, host)
    assert checks["pytorch"]["status"] == "PASS"
    assert checks["pytorch"]["detected_value"] == torch_probe["torch"]
    assert checks["cuda_availability"]["status"] == "PASS"
    assert checks["cuda_availability"]["detected_value"] is True
    assert torch_probe["cuda_available"] is True
    assert int(torch_probe["device_count"]) == 1
    assert torch_probe["device_name"] == gpu_name
    assert checks["nccl_availability"]["status"] == "PASS"
    assert checks["nccl_availability"]["detected_value"]["version"] == torch_probe["nccl_version"]

    assert checks["peer_route"]["status"] == "PASS"
    assert checks["peer_route"]["detected_value"]["peer"] == peer_host
    assert checks["peer_route"]["detected_value"]["interface"] == expected_interface

    assert checks["nccl_path_mtu"]["status"] == "PASS"
    assert checks["nccl_path_mtu"]["detected_value"]["peer"] == peer_host
    assert checks["nccl_path_mtu"]["detected_value"]["interface"] == expected_interface
    assert str(checks["nccl_path_mtu"]["detected_value"]["interface_mtu"]) == "1500"
    assert checks["network_readiness"]["status"] == "PASS"

    assert checks["conda_executable"]["status"] == "PASS"
    assert subject["environment"]["conda_executable"] == "/home/shardgrid/miniconda3/bin/conda"
    assert subject["environment"]["selected_environment"] == "shardgrid"
    assert subject["environment"]["python_executable"] == RUNTIME_PYTHON
    assert subject["runtime_os"] == "wsl2_linux"
    assert subject["physical_os"] == "windows"
