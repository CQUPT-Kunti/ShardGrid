from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.models import train_pipeline


def test_configure_rank_log_skips_tee_when_launcher_owns_sink(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "launcher-owned.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    train_pipeline._LOG_HANDLES.clear()
    monkeypatch.setenv("SHARDGRID_LOG_PATH", str(path))
    monkeypatch.setenv(train_pipeline.LAUNCHER_OWNS_LOG_ENV, "1")

    resolved = train_pipeline._configure_rank_log(0)

    assert resolved == path
    assert train_pipeline._LOG_HANDLES == []
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr


def test_emit_line_writes_once_per_target_when_tee_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "tee.log"
    console = io.StringIO()
    error_console = io.StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_dunder_stdout = sys.__stdout__
    original_dunder_stderr = sys.__stderr__
    train_pipeline._LOG_HANDLES.clear()
    monkeypatch.setenv("SHARDGRID_LOG_PATH", str(path))
    monkeypatch.delenv(train_pipeline.LAUNCHER_OWNS_LOG_ENV, raising=False)
    monkeypatch.setattr(sys, "__stdout__", console)
    monkeypatch.setattr(sys, "__stderr__", error_console)

    try:
        train_pipeline._configure_rank_log(0)
        train_pipeline._emit_line("ONE_MARKER")
    finally:
        for handle in list(train_pipeline._LOG_HANDLES):
            handle.close()
        train_pipeline._LOG_HANDLES.clear()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        sys.__stdout__ = original_dunder_stdout
        sys.__stderr__ = original_dunder_stderr

    assert console.getvalue().count("ONE_MARKER\n") == 1
    assert path.read_text(encoding="utf-8").count("ONE_MARKER\n") == 1
