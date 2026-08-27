"""Cluster resource aggregation for planner-ready state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Sequence, cast

from shardgrid.common.enums import Health, SerializableStrEnum
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource

DEFAULT_FRESHNESS = timedelta(hours=24)


class EligibilityStatus(SerializableStrEnum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"


@dataclass(frozen=True)
class WorkerEligibility:
    worker_id: str
    resource: WorkerResource
    status: EligibilityStatus
    eligible: bool
    stale: bool
    observed_at: str | None
    exclusion_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class ClusterState:
    generated_at: str
    workers: tuple[WorkerEligibility, ...]
    eligible_workers: tuple[WorkerEligibility, ...]
    network_state: NetworkState | None
    network_stale: bool
    freshness_threshold_seconds: int
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


class ResourceManager:
    def __init__(self, *, freshness_threshold: timedelta = DEFAULT_FRESHNESS) -> None:
        self.freshness_threshold = freshness_threshold

    def build_cluster_state(
        self,
        workers: Sequence[WorkerResource],
        *,
        network_state: NetworkState | None = None,
        minimum_gpu_memory_mb: int | None = None,
        require_network: bool = False,
        now: datetime | None = None,
    ) -> ClusterState:
        observed_now = now or self._default_now(workers, network_state)
        sorted_workers = sorted(workers, key=lambda item: str(item.worker_id))
        network_stale = require_network and self._is_network_stale(network_state, observed_now)
        worker_entries = tuple(
            self._worker_entry(
                resource,
                workers=sorted_workers,
                network_state=network_state,
                network_stale=network_stale,
                minimum_gpu_memory_mb=minimum_gpu_memory_mb,
                require_network=require_network,
                now=observed_now,
            )
            for resource in sorted_workers
        )
        eligible = tuple(entry for entry in worker_entries if entry.eligible)
        return ClusterState(
            generated_at=observed_now.isoformat(),
            workers=worker_entries,
            eligible_workers=eligible,
            network_state=network_state,
            network_stale=network_stale,
            freshness_threshold_seconds=int(self.freshness_threshold.total_seconds()),
            summary={
                "total_workers": len(worker_entries),
                "eligible_workers": len(eligible),
                "ineligible_workers": len(worker_entries) - len(eligible),
                "network_required": require_network,
                "network_state_present": network_state is not None,
                "network_stale": network_stale,
            },
        )

    def _worker_entry(
        self,
        resource: WorkerResource,
        *,
        workers: Sequence[WorkerResource],
        network_state: NetworkState | None,
        network_stale: bool,
        minimum_gpu_memory_mb: int | None,
        require_network: bool,
        now: datetime,
    ) -> WorkerEligibility:
        reasons: list[str] = []
        stale = self._is_stale(resource.last_probe_at, now)
        if resource.health is Health.UNREACHABLE:
            reasons.append("worker is unreachable")
        elif resource.health is Health.FAILED:
            reasons.append("worker is unhealthy")
        elif resource.health is Health.BLOCKED_MANUAL_ACTION:
            reasons.append("worker is blocked by manual action")
        elif resource.health is Health.DEGRADED:
            reasons.append("worker is degraded")
        elif resource.health is Health.UNKNOWN:
            reasons.append("worker health is unknown")
        if stale:
            reasons.append("worker resource is stale")
        if not resource.gpu_name or not resource.python_executable or not resource.torch_version:
            reasons.append("gpu/runtime evidence missing")
        if minimum_gpu_memory_mb is not None:
            if resource.gpu_free_memory is None or resource.gpu_free_memory < minimum_gpu_memory_mb:
                reasons.append(
                    f"gpu memory below minimum requirement: "
                    f"{resource.gpu_free_memory or 'unknown'} < {minimum_gpu_memory_mb}"
                )
        if require_network:
            reasons.extend(
                self._network_exclusion_reasons(
                    resource=resource,
                    workers=workers,
                    network_state=network_state,
                    network_stale=network_stale,
                )
            )
        eligible = not reasons
        return WorkerEligibility(
            worker_id=str(resource.worker_id),
            resource=resource,
            status=EligibilityStatus.ELIGIBLE if eligible else EligibilityStatus.INELIGIBLE,
            eligible=eligible,
            stale=stale,
            observed_at=resource.last_probe_at,
            exclusion_reasons=tuple(dict.fromkeys(reasons)),
        )

    def _network_exclusion_reasons(
        self,
        *,
        resource: WorkerResource,
        workers: Sequence[WorkerResource],
        network_state: NetworkState | None,
        network_stale: bool,
    ) -> list[str]:
        if network_state is None:
            return ["network state is missing"]
        if network_stale:
            return ["network state is stale"]
        reasons: list[str] = []
        worker_id = str(resource.worker_id)
        for other in workers:
            other_id = str(other.worker_id)
            if other_id == worker_id:
                continue
            forward = self._find_link(network_state, worker_id, other_id)
            reverse = self._find_link(network_state, other_id, worker_id)
            if forward is None or reverse is None:
                reasons.append(f"missing required network link: {worker_id} <-> {other_id}")
                continue
            for link in (forward, reverse):
                if self._is_stale(link.measured_at, self._parse_now(network_state, link)):
                    reasons.append(f"required network link is stale: {worker_id} <-> {other_id}")
                    break
                if not link.tcp_reachable or link.failure_reason:
                    reasons.append(
                        f"required network link failed: {worker_id} <-> {other_id}"
                        + (f" ({link.failure_reason})" if link.failure_reason else "")
                    )
                    break
        return reasons

    def _is_network_stale(self, state: NetworkState | None, now: datetime) -> bool:
        if state is None:
            return True
        if self._is_stale(state.created_at, now):
            return True
        return any(self._is_stale(link.measured_at, now) for link in state.links)

    def _find_link(self, state: NetworkState, source: str, target: str) -> NetworkLink | None:
        for link in sorted(
            state.links,
            key=lambda item: (str(item.source_worker_id), str(item.target_worker_id)),
        ):
            if str(link.source_worker_id) == source and str(link.target_worker_id) == target:
                return link
        return None

    def _is_stale(self, timestamp: str | None, now: datetime) -> bool:
        stamp = self._parse_timestamp(timestamp)
        if stamp is None:
            return True
        return now - stamp > self.freshness_threshold

    def _parse_now(self, state: NetworkState, link: NetworkLink) -> datetime:
        stamp = self._parse_timestamp(state.created_at) or self._parse_timestamp(link.measured_at)
        return stamp or datetime.now(tz=UTC)

    def _default_now(
        self,
        workers: Sequence[WorkerResource],
        network_state: NetworkState | None,
    ) -> datetime:
        stamps = [
            stamp
            for stamp in (
                self._parse_timestamp(worker.last_probe_at) for worker in workers
            )
            if stamp is not None
        ]
        if network_state is not None:
            state_stamp = self._parse_timestamp(network_state.created_at)
            if state_stamp is not None:
                stamps.append(state_stamp)
            stamps.extend(
                stamp
                for stamp in (
                    self._parse_timestamp(link.measured_at) for link in network_state.links
                )
                if stamp is not None
            )
        return max(stamps, default=datetime.now(tz=UTC))

    def _parse_timestamp(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None


def _serialize(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(cast(Any, value)).items()}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value
