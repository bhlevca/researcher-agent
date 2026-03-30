"""Shared test fixtures."""

import os
import sys

# Ensure CrewAI doesn't prompt for tracing during tests
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ.setdefault("SERPER_API_KEY", "test-key-not-real")

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--integration", action="store_true", default=False,
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
