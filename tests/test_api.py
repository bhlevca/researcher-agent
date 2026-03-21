"""Tests for FastAPI endpoints in main.py.

Uses httpx AsyncClient with mocked heavy dependencies (CrewAI, Ollama, Whisper, SD).
"""

import asyncio
import json
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx
from fastapi.testclient import TestClient

from researcher.main import app, DB_PATH


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_external_services(tmp_path, monkeypatch):
    """Patch CrewAI, Ollama, SD preload so tests don't need real services."""
    # Use temp DB for tests
    test_db = tmp_path / "test_sessions.db"
    monkeypatch.setattr("researcher.main.DB_PATH", test_db)
    test_db.parent.mkdir(parents=True, exist_ok=True)

    # Mock preload_sd so it doesn't load GPU models
    monkeypatch.setattr("researcher.main.preload_sd", lambda: None)

    # Mock ResearchCrew so it doesn't need Ollama
    mock_crew = MagicMock()
    mock_result = MagicMock()
    mock_result.__str__ = MagicMock(return_value="Test response from agent")
    mock_result.token_usage = None
    mock_crew.kickoff.return_value = mock_result

    mock_rc = MagicMock()
    mock_rc.return_value.crew.return_value = mock_crew
    monkeypatch.setattr("researcher.main.ResearchCrew", mock_rc)


@pytest.fixture
def client(_patch_external_services):
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestIndex:
    def test_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# GET /info
# ---------------------------------------------------------------------------

class TestInfo:
    def test_returns_model(self, client):
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model" in data


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

class TestModels:
    def test_returns_models_when_ollama_available(self, client, monkeypatch):
        async def mock_get(self, url, **kw):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "models": [
                    {"name": "qwen3.5:9b", "size": 6_600_000_000,
                     "details": {"family": "qwen2", "parameter_size": "9B"}}
                ]
            }
            return mock_resp

        monkeypatch.setattr("httpx.AsyncClient.get", mock_get)
        resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert len(data["models"]) == 1
        assert data["models"][0]["name"] == "qwen3.5:9b"

    def test_502_when_ollama_down(self, client, monkeypatch):
        async def mock_get(self, url, **kw):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr("httpx.AsyncClient.get", mock_get)
        resp = client.get("/models")
        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# POST /model
# ---------------------------------------------------------------------------

class TestSwitchModel:
    def test_switch_adds_ollama_prefix(self, client):
        resp = client.post("/model", json={"model": "llama3:8b"})
        assert resp.status_code == 200
        assert resp.json()["model"] == "ollama/llama3:8b"

    def test_switch_keeps_existing_prefix(self, client):
        resp = client.post("/model", json={"model": "ollama/llama3:8b"})
        assert resp.status_code == 200
        assert resp.json()["model"] == "ollama/llama3:8b"


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

class TestSessions:
    def test_create_and_list(self, client):
        # Create
        resp = client.post("/sessions", json={
            "name": "Test Session",
            "messages": [{"role": "user", "text": "hello"}]
        })
        assert resp.status_code == 200
        sid = resp.json()["id"]
        assert len(sid) == 8

        # List
        resp = client.get("/sessions")
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        assert any(s["id"] == sid for s in sessions)

    def test_load_session(self, client):
        resp = client.post("/sessions", json={
            "name": "Loadable",
            "messages": [{"role": "user", "text": "test"}]
        })
        sid = resp.json()["id"]

        resp = client.get(f"/sessions/{sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Loadable"
        assert len(data["messages"]) == 1

    def test_update_session(self, client):
        resp = client.post("/sessions", json={
            "name": "Original",
            "messages": []
        })
        sid = resp.json()["id"]

        resp = client.put(f"/sessions/{sid}", json={
            "name": "Updated",
            "messages": [{"role": "user", "text": "updated"}]
        })
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated"

    def test_delete_session(self, client):
        resp = client.post("/sessions", json={
            "name": "ToDelete",
            "messages": []
        })
        sid = resp.json()["id"]

        resp = client.delete(f"/sessions/{sid}")
        assert resp.status_code == 200

        resp = client.get(f"/sessions/{sid}")
        assert resp.status_code == 404

    def test_invalid_session_id(self, client):
        resp = client.get("/sessions/INVALID!")
        assert resp.status_code == 400

    def test_nonexistent_session(self, client):
        resp = client.get("/sessions/deadbeef")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------

class TestAsk:
    def test_returns_structured_response(self, client):
        resp = client.post("/ask", json={"topic": "What is Python?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "response" in data
        assert "reasoning" in data
        assert "model" in data


# ---------------------------------------------------------------------------
# GET /tts/voices
# ---------------------------------------------------------------------------

class TestTTSVoices:
    def test_returns_voice_list(self, client, monkeypatch):
        # Reset the cache
        monkeypatch.setattr("researcher.main._tts_voices_cache", None)

        async def mock_list():
            return [
                {"ShortName": "en-US-AriaNeural", "Gender": "Female", "Locale": "en-US"},
                {"ShortName": "en-US-GuyNeural", "Gender": "Male", "Locale": "en-US"},
            ]

        monkeypatch.setattr("edge_tts.list_voices", mock_list)
        resp = client.get("/tts/voices")
        assert resp.status_code == 200
        voices = resp.json()
        assert len(voices) == 2
        assert voices[0]["name"] == "en-US-AriaNeural"


# ---------------------------------------------------------------------------
# POST /tts/speak
# ---------------------------------------------------------------------------

class TestTTSSpeak:
    def test_returns_audio(self, client, monkeypatch):
        # Pre-populate voice cache so endpoint doesn't call edge_tts.list_voices
        monkeypatch.setattr("researcher.main._tts_voices_cache", [
            {"name": "en-US-AriaNeural", "gender": "Female", "locale": "en-US"},
        ])

        class MockCommunicate:
            def __init__(self, text, voice):
                pass

            async def stream(self):
                yield {"type": "audio", "data": b"\xff\xfb\x90\x00" * 10}

        monkeypatch.setattr("edge_tts.Communicate", MockCommunicate)
        resp = client.post("/tts/speak", json={
            "text": "Hello world",
            "voice": "en-US-AriaNeural"
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/mpeg"
        assert len(resp.content) > 0

    def test_empty_text_rejected(self, client):
        resp = client.post("/tts/speak", json={
            "text": "   ",
            "voice": "en-US-AriaNeural"
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /transcribe
# ---------------------------------------------------------------------------

class TestTranscribe:
    def test_returns_text(self, client, monkeypatch):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Hello from whisper"}
        monkeypatch.setattr("researcher.main._get_whisper", lambda: mock_model)

        # Create a minimal fake audio file
        files = {"file": ("test.webm", b"\x00" * 100, "audio/webm")}
        resp = client.post("/transcribe?language=en", files=files)
        assert resp.status_code == 200
        assert resp.json()["text"] == "Hello from whisper"

    def test_language_param_forwarded(self, client, monkeypatch):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Bonjour"}
        monkeypatch.setattr("researcher.main._get_whisper", lambda: mock_model)

        files = {"file": ("test.webm", b"\x00" * 100, "audio/webm")}
        resp = client.post("/transcribe?language=fr", files=files)
        assert resp.status_code == 200
        # Verify the language was passed through
        mock_model.transcribe.assert_called_once()
        call_kwargs = mock_model.transcribe.call_args
        assert call_kwargs[1]["language"] == "fr" or call_kwargs[0][1] == "fr" or \
               any("fr" in str(a) for a in call_kwargs)
