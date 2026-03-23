"""Shared test fixtures."""

import os
import sys

os.environ.setdefault("SERPER_API_KEY", "test-key-not-real")

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
