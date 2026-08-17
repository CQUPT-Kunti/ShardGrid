"""`shardgrid network-test` CLI command (T045).

Runs the T043 pairwise network probe for the configured GPU Workers and shows
the resulting T044 NetworkState in human or JSON form.  All probe logic lives
in the network modules; this command only wires config, workers, and output.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shardgrid.cli.context import (
    EXIT_CONFIG_ERROR,
    EXIT_RUNTIME_ERROR,
    EXIT_USAGE,
)
from shardgrid.common.config import WorkerConfig
from shardgrid.network.probe import LinkProbeResult, run_pairwise_probe
from shardgrid.network.state import network_state_from_probe_results
from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
from shardgrid.transport.ssh import SSHOptions, SSHTransport

DEFAULT_STATE_DIR = "/var/tmp/shardgrid/network"


def register_network_test_command(
    subparsers: argparse._SubParsersAction[Any],
) -> None:
    parser = subparsers.add_parser(
        "network-test", help="Run pairwise network tests between GPU Workers"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test every pair of enabled GPU Workers in both directions",
    )
    parser.add_argument(
        "--workers",
        nargs=2,
        metavar=("SOURCE", "TARGET"),
        help="Test a specific worker pair (source target)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit structured JSON output",
    )
    parser.set_defaults(handler=run_network_test_command, command_name="network-test")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_wrapper(
    config: Any, worker: WorkerConfig
) -> tuple[WSLRuntimeWrapper, str]:
    address_book = json.loads(Path("tests/address.json").read_text(encoding="utf-8"))
    gpu_label = worker.labels.get("gpu", "").upper().replace(" ", "")
    entry = next(
        item
        for item in address_book
        if gpu_label
        and gpu_label
        in str(item.get("gpu_model") or "").replace(" ", "").upper()
    )
    ip = entry["ip"]
    worker = replace(worker, host=ip, ssh_user=entry["username"])
    transport = SSHTransport(
        SSHOptions.from_ssh_config(
            config.ssh,
            host=ip,
            user=worker.ssh_user,
            port=worker.ssh_port,
        )
    )
    return (
        WSLRuntimeWrapper(
            WSLRuntimeConfig.from_worker_and_runtime(worker, config.runtime), transport
        ),
        ip,
    )


def _human_report(state: Any) -> str:
    lines = [
        f"ShardGrid network-test ({state.network_id})",
        f"created: {state.created_at} | diagnostics: {state.diagnostics_path or 'n/a'}",
        "source -> target | tcp | latency(ms) | bandwidth(mbps) | interface | status",
    ]
    for link in state.links:
        latency = "n/a" if link.latency_ms is None else f"{link.latency_ms:.3f}"
        bandwidth = "n/a" if link.bandwidth_mbps is None else f"{link.bandwidth_mbps:.3f}"
        status = "reachable" if link.tcp_reachable else "unreachable"
        reason = f" | reason: {link.failure_reason}" if link.failure_reason else ""
        lines.append(
            f"{link.source_worker_id} -> {link.target_worker_id} | "
            f"{'yes' if link.tcp_reachable else 'no'} | {latency} | {bandwidth} | "
            f"{link.interface or 'n/a'} | {status}{reason}"
        )
    return "\n".join(lines)


def _resolve_workers(
    config: Any, source_id: str, target_id: str
) -> tuple[WorkerConfig, WorkerConfig]:
    by_id = {str(worker.worker_id): worker for worker in config.workers}
    if source_id not in by_id or target_id not in by_id:
        raise ValueError(
            f"unknown worker id(s): {source_id}, {target_id}; "
            f"known: {', '.join(sorted(by_id))}"
        )
    return by_id[source_id], by_id[target_id]


def _probe_pair(config: Any, source_id: str, target_id: str) -> LinkProbeResult:
    source, target = _resolve_workers(config, source_id, target_id)
    source_wrapper, source_ip = _build_wrapper(config, source)
    target_wrapper, target_ip = _build_wrapper(config, target)
    return run_pairwise_probe(
        source_wrapper,
        target_wrapper,
        source_worker_id=str(source.worker_id),
        target_worker_id=str(target.worker_id),
        target_ip=target_ip,
        tcp_port=config.network.rendezvous_port,
        iperf3_port=config.network.iperf3_port,
    )


def run_network_test_command(args: argparse.Namespace) -> int:
    context = getattr(args, "context", None)
    config = getattr(context, "config", None)
    json_output = bool(getattr(args, "json", False)) or bool(
        getattr(context, "json_output", False)
    )
    if config is None:
        print(
            "network-test requires a cluster config: "
            "shardgrid --config examples/workers.yaml network-test --all"
        )
        return EXIT_CONFIG_ERROR

    workers = [worker for worker in config.workers if worker.enabled]
    if getattr(args, "all", False):
        pairs = [
            (str(workers[0].worker_id), str(workers[1].worker_id)),
            (str(workers[1].worker_id), str(workers[0].worker_id)),
        ]
    elif getattr(args, "workers", None):
        pairs = [(args.workers[0], args.workers[1])]
    else:
        print("network-test requires --all or --workers SOURCE TARGET")
        return EXIT_USAGE

    try:
        results = [_probe_pair(config, source, target) for source, target in pairs]
    except ValueError as error:
        print(f"network-test: {error}")
        return EXIT_USAGE

    state_dir = Path(DEFAULT_STATE_DIR)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"network-state-{_now()}.json"
    state = network_state_from_probe_results(
        results, network_id="lan-a", diagnostics_path=str(state_dir)
    )
    state_path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True))

    if json_output:
        print(json.dumps(state.to_dict(), indent=2, sort_keys=True))
    else:
        print(_human_report(state))
        print(f"network state saved: {state_path}")

    if any(not link.tcp_reachable for link in state.links):
        return EXIT_RUNTIME_ERROR
    return 0