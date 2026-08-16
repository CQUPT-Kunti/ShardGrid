"""Linux platform adapter."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Mapping, Sequence

from shardgrid.common.process import ProcessResult, run_process
from shardgrid.platforms.base import BootstrapStep, ManualActionCheck, PlatformAdapter


class LinuxPlatform(PlatformAdapter):
    platform_name = "linux"
    shell_program = "/bin/bash"

    def detect(self) -> dict[str, str]:
        return {"platform": self.platform_name, "shell": self.shell_program}

    def run(
        self,
        command: Sequence[str] | str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        shell: bool = False,
        secrets: Sequence[str] = (),
    ) -> ProcessResult:
        return run_process(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            shell=shell,
            secrets=secrets,
        )

    def path_join(self, *parts: str) -> str:
        return str(PurePosixPath(*parts))

    def validate_manual_action(self, action: str) -> ManualActionCheck:
        if action.startswith("manual:"):
            return ManualActionCheck(
                allowed=False,
                reason=action.split(":", 1)[1] or "manual action required",
                requires_manual_action=True,
            )
        return ManualActionCheck(allowed=True)

    def bootstrap_step(self, name: str, command: Sequence[str] | str) -> BootstrapStep:
        step_command = tuple(command) if not isinstance(command, str) else command
        return BootstrapStep(name=name, command=step_command, shell=False)
