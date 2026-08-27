"""Immutable local code snapshot creation for a job snapshot."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path, PurePath
from typing import Iterable, Sequence

from shardgrid.jobs.models import JobSnapshot

DEFAULT_CODE_SNAPSHOT_INCLUDES = (
    "src/shardgrid",
    "examples/train-minimal.yaml",
    "examples/models",
)
_MANIFEST_NAME = ".shardgrid-code-snapshot.json"
_TRANSIENT_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".git",
    "jobs",
    "build",
    "dist",
}
_TRANSIENT_PATTERNS = ("*.pyc", "*.pyo", "*.log", "*.tmp", "*.swp")
_SECRET_NAME_PATTERNS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa",
    "id_ed25519",
    "*secret*",
    "*token*",
)


@dataclass(frozen=True)
class CodeSnapshot:
    root_path: str
    checksum: str
    files: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CodeSnapshot":
        return cls(
            root_path=str(data["root_path"]),
            checksum=str(data["checksum"]),
            files=[str(item) for item in data.get("files", [])],
        )


def create_code_snapshot(
    job_snapshot: JobSnapshot,
    *,
    source_root: str | Path,
    includes: Sequence[str] = DEFAULT_CODE_SNAPSHOT_INCLUDES,
    secrets: Sequence[str] = (),
) -> CodeSnapshot:
    root = Path(source_root).resolve()
    code_root = Path(job_snapshot.code_path).resolve()
    snapshot_root = Path(job_snapshot.root_path).resolve()
    _ensure_contained(code_root, snapshot_root)
    manifest_path = code_root / _MANIFEST_NAME

    if manifest_path.exists():
        snapshot = CodeSnapshot.from_dict(json.loads(manifest_path.read_text()))
        if snapshot.checksum != _compute_snapshot_checksum(code_root):
            raise ValueError("existing code snapshot checksum mismatch")
        return snapshot

    if any(code_root.iterdir()):
        raise ValueError("existing code snapshot already present; refusing to overwrite")

    copied: list[str] = []
    for include in includes:
        include_path = _resolve_include(root, include)
        copied.extend(_copy_include(include_path, root=root, code_root=code_root, secrets=secrets))

    snapshot = CodeSnapshot(
        root_path=str(code_root),
        checksum=_compute_snapshot_checksum(code_root),
        files=sorted(copied),
    )
    manifest_path.write_text(json.dumps(snapshot.to_dict(), indent=2, sort_keys=True))
    return snapshot


def _resolve_include(root: Path, include: str) -> Path:
    include_path = PurePath(include)
    if include_path.is_absolute() or any(part == ".." for part in include_path.parts):
        raise ValueError("include path must stay within the source root")
    resolved = (root / include).resolve(strict=True)
    _ensure_contained(resolved, root)
    if resolved.is_symlink():
        raise ValueError("symlink includes are not allowed in code snapshots")
    return resolved


def _copy_include(
    include_path: Path,
    *,
    root: Path,
    code_root: Path,
    secrets: Sequence[str],
) -> list[str]:
    if include_path.is_file():
        relative_path = include_path.relative_to(root)
        if _should_skip(relative_path, include_path, secrets):
            return []
        _copy_file(include_path, code_root / relative_path, code_root)
        return [relative_path.as_posix()]

    copied: list[str] = []
    for path in sorted(_iter_files(include_path)):
        relative_path = path.relative_to(root)
        if _should_skip(relative_path, path, secrets):
            continue
        _copy_file(path, code_root / relative_path, code_root)
        copied.append(relative_path.as_posix())
    return copied


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_symlink():
        raise ValueError("symlink paths are not allowed in code snapshots")
    for child in sorted(path.iterdir()):
        if child.is_symlink():
            raise ValueError("symlink paths are not allowed in code snapshots")
        if child.is_dir():
            if child.name in _TRANSIENT_NAMES:
                continue
            yield from _iter_files(child)
            continue
        yield child


def _should_skip(relative_path: Path, source_path: Path, secrets: Sequence[str]) -> bool:
    if any(part in _TRANSIENT_NAMES for part in relative_path.parts):
        return True
    if any(fnmatch.fnmatch(source_path.name, pattern) for pattern in _TRANSIENT_PATTERNS):
        return True
    if any(fnmatch.fnmatch(source_path.name, pattern) for pattern in _SECRET_NAME_PATTERNS):
        return True
    if _contains_secret(source_path, secrets):
        return True
    return False


def _contains_secret(path: Path, secrets: Sequence[str]) -> bool:
    active_secrets = [secret.encode() for secret in secrets if secret]
    if not active_secrets:
        return False
    data = path.read_bytes()
    return any(secret in data for secret in active_secrets)


def _copy_file(source: Path, destination: Path, code_root: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _ensure_contained(destination, code_root)
    shutil.copy2(source, destination)


def _compute_snapshot_checksum(code_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        candidate
        for candidate in code_root.rglob("*")
        if candidate.is_file() and candidate.name != _MANIFEST_NAME
    ):
        digest.update(path.relative_to(code_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _ensure_contained(path: Path, root: Path) -> None:
    candidate = path.resolve(strict=False)
    base = root.resolve(strict=False)
    if base not in candidate.parents and candidate != base:
        raise ValueError("path escaped snapshot root")
