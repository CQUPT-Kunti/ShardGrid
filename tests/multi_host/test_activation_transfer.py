"""Two-host activation transfer + forward/loss verification (T072)."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import PurePosixPath
from typing import Any

from examples.models.train_pipeline import _load_t072_runtime_plan
from tests.multi_host.test_stage_placement import _build_wrapper, parse_placement_evidence

from shardgrid.engines.static_validation import load_static_parallel_plan

FORWARD_MARKER = "T072_FORWARD_EVIDENCE "
_INTERFACE_RE = re.compile(r"\bdev\s+(\S+)")


def parse_forward_evidence(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(FORWARD_MARKER):
            try:
                payload = json.loads(stripped[len(FORWARD_MARKER):])
            except ValueError:
                return None
            if isinstance(payload, dict):
                return payload
    return None


def _cleanup_t072_processes(wrapper: Any) -> None:
    remote_script = PurePosixPath("/tmp/t072/examples/models/train_pipeline.py")
    wrapper.run(f"pkill -9 -f '{remote_script}' || true", timeout=15.0)


def _discover_interface(wrapper: Any, peer_ip: str) -> str:
    route = wrapper.run(f"ip route get {peer_ip}", timeout=10)
    text = (route.stdout or "").strip()
    match = _INTERFACE_RE.search(text)
    if match:
        return match.group(1)
    route_json = wrapper.run(f"ip -json route get {peer_ip}", timeout=10)
    payload = json.loads((route_json.stdout or "[]").strip() or "[]")
    if isinstance(payload, list) and payload:
        dev = payload[0].get("dev")
        if isinstance(dev, str) and dev:
            return dev
    raise AssertionError(
        "interface resolution failed: "
        f"route_stdout={route.stdout!r} route_stderr={route.stderr!r} "
        f"route_json_stdout={route_json.stdout!r} route_json_stderr={route_json.stderr!r}"
    )


def _assert_marker(stdout: str, marker: str) -> None:
    assert marker in stdout.splitlines(), f"missing marker {marker!r}"


def _last_marker(stdout: str) -> str | None:
    markers = [
        "STAGE0_FORWARD_BEGIN",
        "STAGE0_FORWARD_END",
        "ACTIVATION_SEND_BEGIN",
        "ACTIVATION_SEND_END",
        "FORWARD_COMPLETION_WAIT_BEGIN",
        "FORWARD_COMPLETION_WAIT_END",
        "ACTIVATION_RECV_BEGIN",
        "ACTIVATION_RECV_END",
        "STAGE1_FORWARD_BEGIN",
        "STAGE1_FORWARD_END",
        "LOSS_READY",
        "FORWARD_COMPLETE_SIGNAL",
        "SHUTDOWN_BEGIN",
        "SHUTDOWN_END",
    ]
    lines = set(stdout.splitlines())
    found = [marker for marker in markers if marker in lines]
    return found[-1] if found else None


def _write_t072_diagnostic_artifacts(
    *,
    output_dir: str,
    results: dict[int, Any],
    preflight: dict[int, str],
    remote_state: dict[int, dict[str, Any]],
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    for rank in (0, 1):
        result = results.get(rank)
        if result is None:
            continue
        with open(os.path.join(output_dir, f"rank{rank}.stdout.txt"), "w", encoding="utf-8") as handle:
            handle.write(result.stdout or "")
        with open(os.path.join(output_dir, f"rank{rank}.stderr.txt"), "w", encoding="utf-8") as handle:
            handle.write(result.stderr or "")
    path = os.path.join(output_dir, "activation-transfer-latest.json")
    payload = {
        "task": "T072",
        "remote_preflight": {
            "rank0": preflight.get(0),
            "rank1": preflight.get(1),
        },
        "results": {},
        "remote_state": {
            str(rank): state for rank, state in remote_state.items()
        },
    }
    for rank in (0, 1):
        result = results.get(rank)
        if result is None:
            payload["results"][str(rank)] = None
            continue
        payload["results"][str(rank)] = {
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "last_marker": _last_marker(result.stdout or ""),
            "placement": parse_placement_evidence(result.stdout or ""),
            "forward": parse_forward_evidence(result.stdout or ""),
        }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def _read_remote_file(wrapper: Any, path: str) -> str:
    result = wrapper.run(f"test -f {path} && cat {path} || true", timeout=30)
    return result.stdout or ""


def _collect_remote_state(wrapper: Any, rank: int) -> dict[str, Any]:
    pid_path = f"/tmp/t072/rank{rank}.pid"
    stdout_path = f"/tmp/t072/rank{rank}.stdout"
    stderr_path = f"/tmp/t072/rank{rank}.stderr"
    pid_text = _read_remote_file(wrapper, pid_path).strip()
    pid = int(pid_text) if pid_text.isdigit() else None
    state: dict[str, Any] = {
        "pid": pid,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "stdout": _read_remote_file(wrapper, stdout_path),
        "stderr": _read_remote_file(wrapper, stderr_path),
        "alive": False,
        "python_stack": "",
        "wait_state": "",
        "ps": "",
    }
    if pid is None:
        return state
    alive = wrapper.run(f"kill -0 {pid}", timeout=15)
    state["alive"] = alive.ok
    if not alive.ok:
        return state
    wrapper.run(f"kill -USR1 {pid}", timeout=15)
    state["stderr"] = _read_remote_file(wrapper, stderr_path)
    state["python_stack"] = state["stderr"]
    ps = wrapper.run(
        f"ps -p {pid} -o pid=,stat=,wchan=,cmd=",
        timeout=15,
    )
    state["ps"] = ps.stdout or ""
    state["wait_state"] = (ps.stdout or "").strip()
    return state


def _t072_remote_files() -> dict[str, str]:
    return {
        "minimal_transformer.py": open(
            "examples/models/minimal_transformer.py", encoding="utf-8"
        ).read(),
        "static_parallel_plan.yaml": open(
            "examples/models/static_parallel_plan.yaml", encoding="utf-8"
        ).read(),
        "stage0.py": open("examples/models/stage0.py", encoding="utf-8").read(),
        "stage1.py": open("examples/models/stage1.py", encoding="utf-8").read(),
        "train_pipeline.py": open(
            "examples/models/train_pipeline.py", encoding="utf-8"
        ).read(),
    }


def _preflight_t072_remote_tree(wrapper: Any) -> str:
    command = (
        "set -e; "
        "ls -la /tmp/t072/examples/models/; "
        "test -f /tmp/t072/examples/models/train_pipeline.py; "
        "test -f /tmp/t072/examples/models/static_parallel_plan.yaml; "
        "test -f /tmp/t072/examples/models/stage0.py; "
        "test -f /tmp/t072/examples/models/stage1.py; "
        "test -f /tmp/t072/examples/models/minimal_transformer.py"
    )
    result = wrapper.run(command, timeout=30)
    assert result.ok, (
        "remote T072 preflight failed: "
        f"stdout={(result.stdout or '')[-1200:]}; "
        f"stderr={(result.stderr or '')[-1200:]}"
    )
    return result.stdout or ""


def _stage_t072_remote_tree(wrapper: Any, files: dict[str, str]) -> str:
    from shardgrid.transport.runtime import wrap_wsl_direct_command

    def install_file(path: str, content: str) -> None:
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        script = (
            "import base64\n"
            "from pathlib import Path\n\n"
            f"Path({path!r}).write_bytes(base64.b64decode({encoded!r}))\n"
        )
        result = wrapper.run_script(script, timeout=60)
        assert result.ok, (
            f"failed to install {path}: "
            f"stdout={(result.stdout or '')[-400:]}; "
            f"stderr={(result.stderr or '')[-400:]}"
        )
        check = wrapper.run(f"test -f {path}", timeout=15)
        assert check.ok, f"remote file missing after install: {path}"

    init = wrap_wsl_direct_command(
        wrapper.config.distro,
        wrapper.config.user or "shardgrid",
        "mkdir -p /tmp/t072/examples/models && "
        "touch /tmp/t072/__init__.py /tmp/t072/examples/__init__.py "
        "/tmp/t072/examples/models/__init__.py",
    )
    init_result = wrapper.executor.run(init, timeout=60)
    assert init_result.ok, (
        f"failed to initialize /tmp/t072 tree: "
        f"stdout={(init_result.stdout or '')[-400:]}; "
        f"stderr={(init_result.stderr or '')[-400:]}"
    )
    for name, content in files.items():
        install_file(f"/tmp/t072/examples/models/{name}", content)
    listing = _preflight_t072_remote_tree(wrapper)
    print(listing, flush=True)
    _cleanup_t072_processes(wrapper)
    return listing


def test_parse_forward_evidence() -> None:
    payload = {
        "rank": 1,
        "stage_id": "stage1",
        "output_shape": [2, 8, 1024],
        "loss": 7.0,
        "loss_isfinite": True,
    }
    parsed = parse_forward_evidence(
        "noise\n" + FORWARD_MARKER + json.dumps(payload) + "\n"
    )
    assert parsed is not None
    assert parsed["stage_id"] == "stage1"
    assert parsed["loss_isfinite"] is True
    assert parse_forward_evidence("nothing") is None


def test_static_plan_activation_boundary_for_t072() -> None:
    plan = load_static_parallel_plan("examples/models/static_parallel_plan.yaml")
    stage0 = next(stage for stage in plan.stages if stage.id == "stage0")
    stage1 = next(stage for stage in plan.stages if stage.id == "stage1")
    assert stage0.activation_shape == ("batch", "seq", 128)
    assert stage1.activation_shape == ("batch", "seq", 128)
    assert stage0.activation_dtype == "float32"
    assert stage1.activation_dtype == "float32"


def test_runtime_plan_drives_t072_route_and_tensor_boundary() -> None:
    runtime_plan = _load_t072_runtime_plan()
    stage_by_rank = runtime_plan["stage_by_rank"]
    assert runtime_plan["producer_id"] == "stage0"
    assert runtime_plan["consumer_id"] == "stage1"
    assert runtime_plan["producer_rank"] == 0
    assert runtime_plan["consumer_rank"] == 1
    assert stage_by_rank[0]["id"] == "stage0"
    assert stage_by_rank[1]["id"] == "stage1"
    assert runtime_plan["activation_shape"] == (2, 8, 128)
    assert runtime_plan["activation_dtype_name"] == "float32"


def test_parse_forward_evidence_contains_lifecycle_fields() -> None:
    payload = {
        "rank": 0,
        "stage_id": "stage0",
        "lifecycle": {
            "stage0_forward_begin": 1.0,
            "stage0_forward_end": 2.0,
            "activation_send_begin": 3.0,
            "activation_send_end": 4.0,
        },
    }
    parsed = parse_forward_evidence(FORWARD_MARKER + json.dumps(payload))
    assert parsed is not None
    assert parsed["lifecycle"]["activation_send_end"] == 4.0


def test_t072_remote_file_manifest() -> None:
    files = _t072_remote_files()
    assert set(files) == {
        "minimal_transformer.py",
        "static_parallel_plan.yaml",
        "stage0.py",
        "stage1.py",
        "train_pipeline.py",
    }
    assert "PLAN_PATH" in files["train_pipeline.py"]


def test_live_t072_remote_staging_preflight() -> None:
    files = _t072_remote_files()
    for worker_id in ("gpu4060", "gpu1060"):
        wrapper, _, _ = _build_wrapper(worker_id)
        _stage_t072_remote_tree(wrapper, files)


def test_live_activation_transfer_on_two_workers() -> None:
    """Real two-host activation transfer + forward/loss (opt-in)."""
    import math
    import threading

    plan = load_static_parallel_plan("examples/models/static_parallel_plan.yaml")
    stage0 = next(stage for stage in plan.stages if stage.id == "stage0")
    stage1 = next(stage for stage in plan.stages if stage.id == "stage1")

    w0, ip0, id0 = _build_wrapper("gpu4060")
    w1, ip1, id1 = _build_wrapper("gpu1060")
    assert id0 == "gpu4060" and id1 == "gpu1060"

    iface0 = _discover_interface(w0, ip1)
    iface1 = _discover_interface(w1, ip0)

    files = _t072_remote_files()

    preflight: dict[int, str] = {}

    for rank, wrapper in ((0, w0), (1, w1)):
        preflight[rank] = _stage_t072_remote_tree(wrapper, files)

    results: dict[int, Any] = {}
    output_dir = os.environ.get("SHARDGRID_ENGINE_EVIDENCE_DIR") or (
        "/var/tmp/shardgrid/engines"
    )

    def launch(wrapper: Any, rank: int, iface: str) -> None:
        command = (
            "rm -f /tmp/t072/rank{rank}.pid /tmp/t072/rank{rank}.stdout "
            "/tmp/t072/rank{rank}.stderr; "
            f"RANK={rank} WORLD_SIZE=2 LOCAL_RANK=0 "
            f"MASTER_ADDR={ip0} MASTER_PORT=29500 "
            f"NCCL_SOCKET_IFNAME={iface} GLOO_SOCKET_IFNAME={iface} "
            "NCCL_SOCKET_FAMILY=AF_INET NCCL_IB_DISABLE=1 NCCL_NET=Socket "
            "SHARDGRID_PIPELINE_TASK=t072 SHARDGRID_T072_STAGE1_DIAG=1 "
            "PYTHONUNBUFFERED=1 "
            f"PYTHONPATH=/tmp/t072 /home/shardgrid/miniconda3/envs/shardgrid/bin/python "
            f"-X faulthandler /tmp/t072/examples/models/train_pipeline.py "
            f"> /tmp/t072/rank{rank}.stdout 2> /tmp/t072/rank{rank}.stderr & "
            "pid=$!; echo $pid > /tmp/t072/rank{rank}.pid; wait $pid"
        )
        command = command.format(rank=rank)
        results[rank] = wrapper.run(command, timeout=180)

    threads = [
        threading.Thread(target=launch, args=(w0, 0, iface0)),
        threading.Thread(target=launch, args=(w1, 1, iface1)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    remote_state = {
        0: _collect_remote_state(w0, 0),
        1: _collect_remote_state(w1, 1),
    }
    artifact_path = _write_t072_diagnostic_artifacts(
        output_dir=output_dir,
        results=results,
        preflight=preflight,
        remote_state=remote_state,
    )

    for wrapper in (w0, w1):
        _cleanup_t072_processes(wrapper)

    placement: dict[int, dict[str, Any]] = {}
    forward: dict[int, dict[str, Any]] = {}
    for rank in (0, 1):
        result = results[rank]
        placement_evidence = parse_placement_evidence(result.stdout or "")
        forward_evidence = parse_forward_evidence(result.stdout or "")
        if placement_evidence is None or forward_evidence is None:
            assert False, (
                f"rank {rank} missing T072 evidence; "
                f"stdout tail: {(result.stdout or '')[-1200:]}; "
                f"stderr tail: {(result.stderr or '')[-1200:]}; "
                f"artifact={artifact_path}"
            )
        assert result.exit_code == 0, (
            f"rank {rank} exit_code={result.exit_code}; "
            f"stdout tail: {(result.stdout or '')[-1200:]}; "
            f"stderr tail: {(result.stderr or '')[-1200:]}; "
            f"artifact={artifact_path}"
        )
        placement[rank] = placement_evidence
        forward[rank] = forward_evidence

    rank0_place = placement[0]
    rank1_place = placement[1]
    rank0 = forward[0]
    rank1 = forward[1]

    assert rank0_place["hostname"] != rank1_place["hostname"]
    assert rank0_place["stage_id"] == "stage0"
    assert rank1_place["stage_id"] == "stage1"
    assert rank0_place["parameter_count"] == stage0.parameter_count
    assert rank1_place["parameter_count"] == stage1.parameter_count

    assert rank0["stage_id"] == "stage0"
    assert rank0["parameter_count"] == stage0.parameter_count
    assert "RTX 4060" in rank0["gpu_name"]
    assert rank0["device"] == "cuda:0"
    assert rank0["input_shape"] == [2, 8]
    assert rank0["input_dtype"] == "int64"
    assert rank0["output_shape"] == [2, 8, 128]
    assert rank0["output_dtype"] == "float32"
    assert rank0["stage0_forward_ok"] is True
    assert rank0["lifecycle"]["stage0_forward_begin"] <= rank0["lifecycle"]["stage0_forward_end"]
    assert rank0["lifecycle"]["activation_send_begin"] <= rank0["lifecycle"]["activation_send_end"]

    assert rank1["stage_id"] == "stage1"
    assert rank1["parameter_count"] == stage1.parameter_count
    assert "GTX 1650" in rank1["gpu_name"]
    assert rank1["device"] == "cuda:0"
    assert rank1["input_shape"] == [2, 8, 128]
    assert rank1["input_dtype"] == "float32"
    assert rank1["output_shape"] == [2, 8, 1024]
    assert rank1["output_dtype"] == "float32"
    assert rank1["target_shape"] == [2, 8]
    assert rank1["target_dtype"] == "int64"
    assert rank1["stage1_forward_ok"] is True
    assert rank1["loss_isfinite"] is True
    assert math.isfinite(rank1["loss"])
    assert rank1["lifecycle"]["activation_recv_begin"] <= rank1["lifecycle"]["activation_recv_end"]
    assert rank1["lifecycle"]["stage1_forward_begin"] <= rank1["lifecycle"]["stage1_forward_end"]

    send = rank0["activation_transfer"]
    recv = rank1["activation_transfer"]
    assert send["sender_rank"] == 0 and send["receiver_rank"] == 1
    assert recv["sender_rank"] == 0 and recv["receiver_rank"] == 1
    assert send["shape"] == recv["shape"] == [2, 8, 128]
    assert send["dtype"] == recv["dtype"] == "float32"
    assert send["numel"] == recv["numel"] == 2 * 8 * 128
    assert send["send_begin"] <= send["send_end"]
    assert recv["recv_begin"] <= recv["recv_end"]
    _assert_marker(results[0].stdout or "", "STAGE0_FORWARD_BEGIN")
    _assert_marker(results[0].stdout or "", "STAGE0_FORWARD_END")
    _assert_marker(results[0].stdout or "", "ACTIVATION_SEND_BEGIN")
    _assert_marker(results[0].stdout or "", "ACTIVATION_SEND_END")
    _assert_marker(results[0].stdout or "", "SHUTDOWN_BEGIN")
    _assert_marker(results[0].stdout or "", "SHUTDOWN_END")
    _assert_marker(results[1].stdout or "", "ACTIVATION_RECV_BEGIN")
    _assert_marker(results[1].stdout or "", "ACTIVATION_RECV_END")
    _assert_marker(results[1].stdout or "", "STAGE1_FORWARD_BEGIN")
    _assert_marker(results[1].stdout or "", "STAGE1_FORWARD_END")
    _assert_marker(results[1].stdout or "", "FORWARD_OUTPUT_END")
    _assert_marker(results[1].stdout or "", "SHUTDOWN_BEGIN")
    _assert_marker(results[1].stdout or "", "SHUTDOWN_END")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "activation-transfer-latest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "task": "T072",
                "rank0": {
                    "placement": rank0_place,
                    "forward": rank0,
                },
                "rank1": {
                    "placement": rank1_place,
                    "forward": rank1,
                },
                "cross_host_activation": {
                    "sender_rank": 0,
                    "receiver_rank": 1,
                    "sender_host": rank0_place["hostname"],
                    "receiver_host": rank1_place["hostname"],
                    "shape": send["shape"],
                    "dtype": send["dtype"],
                    "numel": send["numel"],
                    "send_begin": send["send_begin"],
                    "send_end": send["send_end"],
                    "recv_begin": recv["recv_begin"],
                    "recv_end": recv["recv_end"],
                    "distinct_hosts": rank0_place["hostname"] != rank1_place["hostname"],
                },
                "process_lifecycle": {
                    "rank0_exit_code": results[0].exit_code,
                    "rank1_exit_code": results[1].exit_code,
                    "clean_exit": results[0].exit_code == 0 and results[1].exit_code == 0,
                },
                "remote_preflight": {
                    "rank0": preflight[0],
                    "rank1": preflight[1],
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    assert os.path.exists(path)
