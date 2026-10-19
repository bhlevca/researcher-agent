"""Shared test fixtures."""

import os
import sys

# Ensure CrewAI doesn't prompt for tracing during tests
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ.setdefault("SERPER_API_KEY", "test-key-not-real")

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests that require a running server",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--integration"):
        skip_int = pytest.mark.skip(reason="Need --integration flag to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_int)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _isolate_song_engine(tmp_path, monkeypatch):
    """No test may touch the real ACE-Step service, song folder or audit log.

    Written after a test run restarted the real systemd service at 19:14:45 on
    2026-10-18. song.stop() asks the engine whether it holds the GPU and
    restarts it if so, and the broker now probes the engine when it has no
    record — so on a machine where ACE-Step is running, ordinary unit tests
    reached it. They also wrote their prunes into the real audit log.

    Forcing "no systemd" alone would be worse: stop() would then fall back to
    killing stray processes, scan the real /proc, find the service and kill
    it. So the effectors are stubbed — systemctl raises, nothing is killed,
    /health is never called — and every path points into tmp_path.
    """
    try:
        from researcher import song
    except Exception:          # a test run without the app importable
        yield
        return
    root = tmp_path / "songs"
    monkeypatch.setattr(song, "SONG_ROOT", root)
    monkeypatch.setattr(song, "SCRATCH_DIR", root / "scratch")
    monkeypatch.setattr(song, "KEPT_DIR", root / "kept")
    monkeypatch.setattr(song, "AUDIT_PATH", tmp_path / "songs.log")
    monkeypatch.setattr(song, "_systemd_managed", lambda: False)
    monkeypatch.setattr(song, "_health", lambda *a, **k: None)
    monkeypatch.setattr(song, "_stop_orphans", lambda: False)
    monkeypatch.setattr(song, "_orphan_pids", lambda: [])

    def _no_systemctl(*a, **k):
        raise AssertionError("a test tried to control a real systemd unit")

    monkeypatch.setattr(song, "_systemctl", _no_systemctl)
    yield
