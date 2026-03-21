"""Shared test fixtures."""

import os
import sys

# Ensure CrewAI doesn't prompt for tracing during tests
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ.setdefault("SERPER_API_KEY", "test-key-not-real")

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
