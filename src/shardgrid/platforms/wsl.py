"""WSL Bash platform adapter."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Mapping, Sequence

from shardgrid.common.process import ProcessResult, run_process
from shardgrid.platforms.base import BootstrapStep, ManualActionCheck, PlatformAdapter


class WSLPlatform(PlatformAdapter):
    platform_name = "wsl2_linux"
    shell_program = "/bin/bash"

    def __init__(self, distro: str | None = None) -> None:
        self.distro = distro

    def detect(self) -> dict[str, str]:
        payload = {"platform": self.platform_name, "shell": self.shell_program}
        if self.distro is not None:
            payload["distro"] = self.distro
        return payload

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
            self.wrap_command(command, shell=shell),
            cwd=cwd,
            env=env,
            timeout=timeout,
            shell=False,
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
        return BootstrapStep(name=name, command=self.wrap_command(command), shell=False)

    def wrap_command(self, command: Sequence[str] | str, *, shell: bool = False) -> tuple[str, ...]:
        if isinstance(command, str):
            payload = command
        else:
            payload = " ".join(command)
        prefix: list[str] = ["wsl"]
        if self.distro is not None:
            prefix.extend(["-d", self.distro])
        prefix.extend([self.shell_program, "-lc", payload])
        return tuple(prefix)
