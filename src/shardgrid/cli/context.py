"""Shared CLI context and exit policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from shardgrid.common.config import ClusterConfig, load_cluster_config
from shardgrid.common.errors import StageError

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_RUNTIME_ERROR = 1
EXIT_CONFIG_ERROR = 3


@dataclass(frozen=True)
class CLIContext:
    config_path: Path | None
    jobs_root: Path | None
    verbose: bool = False
    json_output: bool = False

    def load_config(self) -> ClusterConfig | None:
        if self.config_path is None:
            return None
        return load_cluster_config(self.config_path)


def resolve_cli_context(
    *,
    config: str | None,
    jobs_root: str | None,
    verbose: bool,
    json_output: bool,
) -> CLIContext:
    return CLIContext(
        config_path=None if config is None else Path(config),
        jobs_root=None if jobs_root is None else Path(jobs_root),
        verbose=verbose,
        json_output=json_output,
    )


def format_cli_error(error: Exception, *, json_output: bool) -> str:
    if json_output:
        payload: dict[str, object] = {"error": type(error).__name__, "message": str(error)}
        if isinstance(error, StageError):
            payload["failure"] = error.failure.to_dict()
        return json.dumps(payload, sort_keys=True)
    return str(error)
