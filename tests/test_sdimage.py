"""Integration test: Stable Diffusion pipeline via /chat endpoint.

Requires a running server (``uvicorn researcher.main:app``) with
IMAGE_BACKEND=sd in .env and the SD model downloaded.  Run with::

    pytest tests/test_sdimage.py -v -s

The test authenticates, sends an image-generation prompt, and verifies
that the SSE stream completes with a valid image tag in the response.
"""

import os
import re

import pytest
import requests

HOST = os.getenv("TEST_HOST", "http://localhost:8000")
USERNAME = os.getenv("TEST_USER", "bogdan")
PASSWORD = os.getenv("TEST_PASS", "AI_Ralcua@1986")

pytestmark = pytest.mark.integration


def _login() -> str:
    """Log in and return a JWT token."""
    resp = requests.post(
        f"{HOST}/auth/login",
        json={"username": USERNAME, "password": PASSWORD},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["token"]


@pytest.fixture(scope="module")
def jwt_token():
    try:
        return _login()
    except requests.ConnectionError:
        pytest.skip("Server not running")


def test_sdimage_generates_image(jwt_token):
    """POST /chat with an image prompt returns a generated image tag (SD backend)."""
    resp = requests.post(
        f"{HOST}/chat",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jwt_token}",
        },
        json={
            "message": "Draw a photorealistic cat in 8k",
            "history": [],
            "file_ids": [],
        },
        timeout=600,
    )
    assert resp.status_code == 200, f"Unexpected status {resp.status_code}: {resp.text[:300]}"

    body = resp.text

    # The SSE stream should contain a "done" event with the final response
    assert "event: done" in body, "Missing 'done' SSE event"

    # Extract the JSON payload from the done event
    done_match = re.search(r'event: done\ndata: ({.*})', body)
    assert done_match, "Could not parse 'done' event payload"

    import json
    payload = json.loads(done_match.group(1))
    response_text = payload.get("response", "")

    # Should contain a generated image markdown tag
    assert "![generated image]" in response_text, (
        f"No image tag in response: {response_text[:200]}"
    )
    assert "/static/generated/" in response_text, (
        f"No generated path in response: {response_text[:200]}"
    )
    assert response_text.endswith(".png)") or ".png)" in response_text, (
        f"Image path doesn't end with .png: {response_text[:200]}"
    )

    print(f"\n✅ SD image test passed. Response excerpt:\n{response_text[:300]}")
