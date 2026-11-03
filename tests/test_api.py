"""Tests for FastAPI endpoints in main.py.

Uses httpx AsyncClient with mocked heavy dependencies (CrewAI, Ollama, Whisper, SD).
"""

import asyncio
import json
import io
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx
from fastapi.testclient import TestClient

from researcher.main import app
from researcher.config import DB_PATH
from researcher import auth as auth_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_external_services(tmp_path, monkeypatch):
    """Patch CrewAI, Ollama, SD preload so tests don't need real services."""
    # Use temp DB for tests
    test_db = tmp_path / "test_sessions.db"
    monkeypatch.setattr("researcher.config.DB_PATH", test_db)
    monkeypatch.setattr("researcher.main.DB_PATH", test_db)
    test_db.parent.mkdir(parents=True, exist_ok=True)

    # Mock ResearchCrew so it doesn't need Ollama
    mock_crew = MagicMock()
    mock_result = MagicMock()
    mock_result.__str__ = MagicMock(return_value="Test response from agent")
    mock_result.token_usage = None
    mock_crew.kickoff.return_value = mock_result

    mock_rc = MagicMock()
    mock_rc.return_value.crew.return_value = mock_crew
    # Crews are constructed in one place now — user_crews._crew_class,
    # which imports each class from its own module at call time — so this is
    # the single point that keeps a real model out of the tests.
    monkeypatch.setattr("researcher.crew.ResearchCrew", mock_rc)
    monkeypatch.setattr("researcher.routes.models.ResearchCrew", mock_rc)


@pytest.fixture
def client(_patch_external_services):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers(client):
    """Register a test user and return Authorization headers."""
    auth_module._rate_log.clear()
    orig = auth_module.INVITE_CODE
    auth_module.INVITE_CODE = ""
    try:
        resp = client.post("/auth/register", json={
            "username": "testuser", "password": "testpass123"
        })
        token = resp.json()["token"]
        return {"Authorization": f"Bearer {token}"}
    finally:
        auth_module.INVITE_CODE = orig


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
    def test_returns_models_when_ollama_available(self, client, auth_headers, monkeypatch):
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
        resp = client.get("/models", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert len(data["models"]) == 1
        assert data["models"][0]["name"] == "qwen3.5:9b"

    def test_502_when_ollama_down(self, client, auth_headers, monkeypatch):
        async def mock_get(self, url, **kw):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr("httpx.AsyncClient.get", mock_get)
        resp = client.get("/models", headers=auth_headers)
        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# POST /model
# ---------------------------------------------------------------------------

class TestSwitchModel:
    def test_switch_adds_ollama_prefix(self, client, auth_headers):
        resp = client.post("/model", json={"model": "llama3:8b"}, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["model"] == "ollama/llama3:8b"

    def test_switch_keeps_existing_prefix(self, client, auth_headers):
        resp = client.post("/model", json={"model": "ollama/llama3:8b"}, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["model"] == "ollama/llama3:8b"


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

class TestSessions:
    def test_create_and_list(self, client, auth_headers):
        # Create
        resp = client.post("/sessions", json={
            "name": "Test Session",
            "messages": [{"role": "user", "text": "hello"}]
        }, headers=auth_headers)
        assert resp.status_code == 200
        sid = resp.json()["id"]
        assert len(sid) == 8

        # List
        resp = client.get("/sessions", headers=auth_headers)
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        assert any(s["id"] == sid for s in sessions)

    def test_load_session(self, client, auth_headers):
        resp = client.post("/sessions", json={
            "name": "Loadable",
            "messages": [{"role": "user", "text": "test"}]
        }, headers=auth_headers)
        sid = resp.json()["id"]

        resp = client.get(f"/sessions/{sid}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Loadable"
        assert len(data["messages"]) == 1

    def test_update_session(self, client, auth_headers):
        resp = client.post("/sessions", json={
            "name": "Original",
            "messages": []
        }, headers=auth_headers)
        sid = resp.json()["id"]

        resp = client.put(f"/sessions/{sid}", json={
            "name": "Updated",
            "messages": [{"role": "user", "text": "updated"}]
        }, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated"

    def test_delete_session(self, client, auth_headers):
        resp = client.post("/sessions", json={
            "name": "ToDelete",
            "messages": []
        }, headers=auth_headers)
        sid = resp.json()["id"]

        resp = client.delete(f"/sessions/{sid}", headers=auth_headers)
        assert resp.status_code == 200

        resp = client.get(f"/sessions/{sid}", headers=auth_headers)
        assert resp.status_code == 404

    def test_invalid_session_id(self, client, auth_headers):
        resp = client.get("/sessions/INVALID!", headers=auth_headers)
        assert resp.status_code == 400

    def test_nonexistent_session(self, client, auth_headers):
        resp = client.get("/sessions/deadbeef", headers=auth_headers)
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------

class TestAsk:
    def test_returns_structured_response(self, client, auth_headers):
        resp = client.post("/ask", json={"topic": "What is Python?"}, headers=auth_headers)
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
        monkeypatch.setattr("researcher.tts._tts_voices_cache", None)

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
        monkeypatch.setattr("researcher.tts._tts_voices_cache", [
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
    def test_returns_text(self, client, auth_headers, monkeypatch):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Hello from whisper",
            "language": "en",
            "segments": [{"text": "Hello from whisper", "no_speech_prob": 0.05, "avg_logprob": -0.3}],
        }
        monkeypatch.setattr("researcher.routes.speech._get_whisper", lambda: mock_model)

        # Create a minimal fake audio file
        files = {"file": ("test.webm", b"\x00" * 100, "audio/webm")}
        resp = client.post("/transcribe?language=en", files=files, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["text"] == "Hello from whisper"

    def test_language_param_forwarded(self, client, auth_headers, monkeypatch):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Bonjour",
            "language": "fr",
            "segments": [{"text": "Bonjour", "no_speech_prob": 0.1, "avg_logprob": -0.3}],
        }
        monkeypatch.setattr("researcher.routes.speech._get_whisper", lambda: mock_model)

        files = {"file": ("test.webm", b"\x00" * 100, "audio/webm")}
        resp = client.post("/transcribe?language=fr", files=files, headers=auth_headers)
        assert resp.status_code == 200
        kwargs = mock_model.transcribe.call_args[1]
        assert kwargs["language"] == "fr"

    def test_auto_detect_omits_language(self, client, auth_headers, monkeypatch):
        """language=auto lets Whisper detect the spoken language."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Hello",
            "language": "en",
            "segments": [{"text": "Hello", "no_speech_prob": 0.05, "avg_logprob": -0.3}],
        }
        monkeypatch.setattr("researcher.routes.speech._get_whisper", lambda: mock_model)

        files = {"file": ("test.webm", b"\x00" * 100, "audio/webm")}
        resp = client.post("/transcribe?language=auto", files=files, headers=auth_headers)
        assert resp.status_code == 200
        kwargs = mock_model.transcribe.call_args[1]
        assert "language" not in kwargs

    def test_filters_hallucinated_segments(self, client, auth_headers, monkeypatch):
        """Segments with high no_speech_prob are filtered out (anti-hallucination)."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Real speech. Hallucinated noise.",
            "language": "en",
            "segments": [
                {"text": "Real speech.", "no_speech_prob": 0.1, "avg_logprob": -0.3},
                {"text": "Hallucinated noise.", "no_speech_prob": 0.9, "avg_logprob": -0.3},
            ],
        }
        monkeypatch.setattr("researcher.routes.speech._get_whisper", lambda: mock_model)

        files = {"file": ("test.webm", b"\x00" * 100, "audio/webm")}
        resp = client.post("/transcribe?language=en", files=files, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["text"] == "Real speech."

    def test_filters_script_mismatch(self, client, auth_headers, monkeypatch):
        """Segments with wrong Unicode script (e.g. CJK in French) are filtered."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Bonjour \u304b\u3063\u305f",
            "language": "fr",
            "segments": [
                {"text": "Bonjour", "no_speech_prob": 0.1, "avg_logprob": -0.3},
                {"text": "\u304b\u3063\u305f\u3002", "no_speech_prob": 0.1, "avg_logprob": -0.3},
            ],
        }
        monkeypatch.setattr("researcher.routes.speech._get_whisper", lambda: mock_model)

        files = {"file": ("test.webm", b"\x00" * 100, "audio/webm")}
        resp = client.post("/transcribe?language=fr", files=files, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["text"] == "Bonjour"

    def test_filters_low_confidence_segments(self, client, auth_headers, monkeypatch):
        """Segments with very low avg_logprob are filtered out."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Real. Garbage.",
            "language": "en",
            "segments": [
                {"text": "Real.", "no_speech_prob": 0.1, "avg_logprob": -0.5},
                {"text": "Garbage.", "no_speech_prob": 0.1, "avg_logprob": -1.5},
            ],
        }
        monkeypatch.setattr("researcher.routes.speech._get_whisper", lambda: mock_model)

        files = {"file": ("test.webm", b"\x00" * 100, "audio/webm")}
        resp = client.post("/transcribe?language=en", files=files, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["text"] == "Real."

    def test_all_silence_returns_empty(self, client, auth_headers, monkeypatch):
        """When all segments are silence/noise, return empty text."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Thank you for watching.",
            "language": "en",
            "segments": [
                {"text": "Thank you for watching.", "no_speech_prob": 0.95, "avg_logprob": -0.3},
            ],
        }
        monkeypatch.setattr("researcher.routes.speech._get_whisper", lambda: mock_model)

        files = {"file": ("test.webm", b"\x00" * 100, "audio/webm")}
        resp = client.post("/transcribe?language=en", files=files, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["text"] == ""

    def test_anti_hallucination_params(self, client, auth_headers, monkeypatch):
        """Verify condition_on_previous_text=False is passed to prevent loops."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "OK",
            "language": "en",
            "segments": [{"text": "OK", "no_speech_prob": 0.0, "avg_logprob": -0.3}],
        }
        monkeypatch.setattr("researcher.routes.speech._get_whisper", lambda: mock_model)

        files = {"file": ("test.webm", b"\x00" * 100, "audio/webm")}
        client.post("/transcribe?language=en", files=files, headers=auth_headers)

        kwargs = mock_model.transcribe.call_args[1]
        assert kwargs["condition_on_previous_text"] is False
        assert kwargs["no_speech_threshold"] == 0.6
        assert kwargs["compression_ratio_threshold"] == 2.4


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

class TestAuth:
    """Registration, login, rate-limiting, invite code, honeypot."""

    @pytest.fixture(autouse=True)
    def _clear_rate_limit(self, monkeypatch):
        auth_module._rate_log.clear()
        monkeypatch.setattr(auth_module, "INVITE_CODE", "")

    def test_register_and_login(self, client):
        resp = client.post("/auth/register", json={
            "username": "alice", "password": "secret123"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["user"]["username"] == "alice"

        # Login with same credentials
        resp = client.post("/auth/login", json={
            "username": "alice", "password": "secret123"
        })
        assert resp.status_code == 200
        assert "token" in resp.json()

    def test_register_duplicate(self, client):
        client.post("/auth/register", json={
            "username": "dup_user", "password": "secret123"
        })
        resp = client.post("/auth/register", json={
            "username": "dup_user", "password": "other456"
        })
        assert resp.status_code == 409

    def test_login_wrong_password(self, client):
        client.post("/auth/register", json={
            "username": "bob", "password": "correct123"
        })
        resp = client.post("/auth/login", json={
            "username": "bob", "password": "wrong"
        })
        assert resp.status_code == 401

    def test_login_nonexistent_user(self, client):
        resp = client.post("/auth/login", json={
            "username": "nobody", "password": "anything"
        })
        assert resp.status_code == 401

    def test_me_requires_auth(self, client):
        resp = client.get("/auth/me")
        assert resp.status_code == 401

    def test_me_with_valid_token(self, client):
        reg = client.post("/auth/register", json={
            "username": "carol", "password": "secret123"
        })
        token = reg.json()["token"]
        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["username"] == "carol"

    def test_invite_code_required(self, client, monkeypatch):
        monkeypatch.setattr(auth_module, "INVITE_CODE", "my-secret-code")
        # Without invite code
        resp = client.post("/auth/register", json={
            "username": "intruder", "password": "secret123"
        })
        assert resp.status_code == 403
        assert "invite" in resp.json()["detail"].lower()

    def test_invite_code_accepted(self, client, monkeypatch):
        monkeypatch.setattr(auth_module, "INVITE_CODE", "my-secret-code")
        resp = client.post("/auth/register", json={
            "username": "invited_user", "password": "secret123",
            "invite_code": "my-secret-code"
        })
        assert resp.status_code == 200
        assert resp.json()["user"]["username"] == "invited_user"

    def test_honeypot_blocks_bots(self, client):
        resp = client.post("/auth/register", json={
            "username": "botuser", "password": "secret123",
            "website": "http://spam.example.com"
        })
        assert resp.status_code == 400

    def test_rate_limit(self, client, monkeypatch):
        monkeypatch.setattr(auth_module, "_AUTH_RATE_MAX_ATTEMPTS", 3)

        for _ in range(3):
            client.post("/auth/login", json={
                "username": "nobody", "password": "wrong"
            })
        # 4th attempt should be rate-limited
        resp = client.post("/auth/login", json={
            "username": "nobody", "password": "wrong"
        })
        assert resp.status_code == 429

    def test_auth_config_endpoint(self, client, monkeypatch):
        resp = client.get("/auth/config")
        assert resp.status_code == 200
        assert resp.json()["invite_required"] is False

        monkeypatch.setattr(auth_module, "INVITE_CODE", "something")
        resp = client.get("/auth/config")
        assert resp.json()["invite_required"] is True

    def test_short_password_rejected(self, client):
        resp = client.post("/auth/register", json={
            "username": "shortpw", "password": "abc"
        })
        assert resp.status_code == 400

    def test_invalid_username_rejected(self, client):
        resp = client.post("/auth/register", json={
            "username": "a", "password": "secret123"
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Session isolation (authenticated vs guest)
# ---------------------------------------------------------------------------

class TestSessionIsolation:
    """Verify that sessions are scoped per user."""

    @pytest.fixture(autouse=True)
    def _clear_rate_limit(self, monkeypatch):
        auth_module._rate_log.clear()
        monkeypatch.setattr(auth_module, "INVITE_CODE", "")

    def _register(self, client, username):
        resp = client.post("/auth/register", json={
            "username": username, "password": "secret123"
        })
        return resp.json()["token"]

    def test_user_sees_only_own_sessions(self, client, monkeypatch):
        monkeypatch.setattr(auth_module, "INVITE_CODE", "")
        auth_module._rate_log.clear()

        tok_a = self._register(client, "user_a")
        tok_b = self._register(client, "user_b")

        # User A creates a session
        client.post("/sessions", json={"name": "A's session", "messages": []},
                     headers={"Authorization": f"Bearer {tok_a}"})
        # User B creates a session
        client.post("/sessions", json={"name": "B's session", "messages": []},
                     headers={"Authorization": f"Bearer {tok_b}"})

        # User A should only see their session
        resp = client.get("/sessions", headers={"Authorization": f"Bearer {tok_a}"})
        names = [s["name"] for s in resp.json()["sessions"]]
        assert "A's session" in names
        assert "B's session" not in names

    def test_user_cannot_load_others_session(self, client, monkeypatch):
        monkeypatch.setattr(auth_module, "INVITE_CODE", "")
        auth_module._rate_log.clear()

        tok_a = self._register(client, "owner_x")
        tok_b = self._register(client, "other_y")

        resp = client.post("/sessions", json={"name": "Private", "messages": []},
                           headers={"Authorization": f"Bearer {tok_a}"})
        sid = resp.json()["id"]

        # Other user tries to load it
        resp = client.get(f"/sessions/{sid}",
                          headers={"Authorization": f"Bearer {tok_b}"})
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# File upload / ingestion
# ---------------------------------------------------------------------------

class TestFileUpload:
    def test_upload_requires_auth(self, client):
        resp = client.post("/files/upload", files=[("files", ("a.txt", b"hello", "text/plain"))])
        assert resp.status_code == 401

    def test_upload_txt(self, client, auth_headers):
        resp = client.post("/files/upload",
                           headers=auth_headers,
                           files=[("files", ("test.txt", b"Hello world", "text/plain"))])
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["files"]) == 1
        f = data["files"][0]
        assert f["filename"] == "test.txt"
        assert "id" in f
        assert f["size"] == 11

    def test_upload_csv(self, client, auth_headers):
        csv_data = b"name,age\nAlice,30\nBob,25"
        resp = client.post("/files/upload",
                           headers=auth_headers,
                           files=[("files", ("data.csv", csv_data, "text/csv"))])
        assert resp.status_code == 200
        assert resp.json()["files"][0]["filename"] == "data.csv"

    def test_upload_multiple_files(self, client, auth_headers):
        files = [
            ("files", ("a.txt", b"file one", "text/plain")),
            ("files", ("b.txt", b"file two", "text/plain")),
        ]
        resp = client.post("/files/upload", headers=auth_headers, files=files)
        assert resp.status_code == 200
        assert len(resp.json()["files"]) == 2

    def test_upload_unsupported_type(self, client, auth_headers):
        resp = client.post("/files/upload",
                           headers=auth_headers,
                           files=[("files", ("bad.exe", b"\x00\x01", "application/octet-stream"))])
        assert resp.status_code == 200
        f = resp.json()["files"][0]
        assert "error" in f
        assert "Unsupported" in f["error"]

    def test_list_files(self, client, auth_headers):
        client.post("/files/upload", headers=auth_headers,
                     files=[("files", ("t.txt", b"data", "text/plain"))])
        resp = client.get("/files", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()["files"]) >= 1

    def test_get_file(self, client, auth_headers):
        up = client.post("/files/upload", headers=auth_headers,
                          files=[("files", ("g.txt", b"get me", "text/plain"))])
        fid = up.json()["files"][0]["id"]
        resp = client.get(f"/files/{fid}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["text"] == "get me"

    def test_delete_file(self, client, auth_headers):
        up = client.post("/files/upload", headers=auth_headers,
                          files=[("files", ("d.txt", b"delete me", "text/plain"))])
        fid = up.json()["files"][0]["id"]
        resp = client.delete(f"/files/{fid}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["deleted"] == fid
        # Verify gone
        resp = client.get(f"/files/{fid}", headers=auth_headers)
        assert resp.status_code == 404

    def test_get_nonexistent_file(self, client, auth_headers):
        resp = client.get("/files/nonexistent999", headers=auth_headers)
        assert resp.status_code == 404


class TestFileIsolation:
    """Files from one user must not be visible to another."""

    def _register(self, client, username):
        auth_module._rate_log.clear()
        resp = client.post("/auth/register", json={
            "username": username, "password": "pass12345"
        })
        return resp.json()["token"]

    def test_user_cannot_see_others_files(self, client, monkeypatch):
        monkeypatch.setattr(auth_module, "INVITE_CODE", "")
        auth_module._rate_log.clear()

        tok_a = self._register(client, "file_owner")
        tok_b = self._register(client, "file_intruder")

        up = client.post("/files/upload",
                          headers={"Authorization": f"Bearer {tok_a}"},
                          files=[("files", ("secret.txt", b"secret data", "text/plain"))])
        fid = up.json()["files"][0]["id"]

        # Other user cannot get it
        resp = client.get(f"/files/{fid}",
                          headers={"Authorization": f"Bearer {tok_b}"})
        assert resp.status_code == 404

        # Other user's list doesn't include it
        resp = client.get("/files", headers={"Authorization": f"Bearer {tok_b}"})
        ids = [f["id"] for f in resp.json()["files"]]
        assert fid not in ids


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_requires_auth(self, client):
        resp = client.post("/export", json={"content": "hi", "format": "txt"})
        assert resp.status_code == 401

    def test_export_markdown(self, client, auth_headers):
        resp = client.post("/export", headers={**auth_headers, "Content-Type": "application/json"},
                           json={"content": "# Title\nHello", "format": "md", "filename": "test"})
        assert resp.status_code == 200
        assert "markdown" in resp.headers["content-type"]
        assert resp.content == b"# Title\nHello"

    def test_export_txt(self, client, auth_headers):
        resp = client.post("/export", headers={**auth_headers, "Content-Type": "application/json"},
                           json={"content": "**bold** and *italic*", "format": "txt", "filename": "test"})
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        # Markdown formatting should be stripped
        assert b"**" not in resp.content
        assert b"bold" in resp.content

    def test_export_docx(self, client, auth_headers):
        content = "# Report\n\nSome **bold** text.\n\n| A | B |\n|---|---|\n| 1 | 2 |"
        resp = client.post("/export", headers={**auth_headers, "Content-Type": "application/json"},
                           json={"content": content, "format": "docx", "filename": "report"})
        assert resp.status_code == 200
        assert "officedocument.wordprocessingml" in resp.headers["content-type"]
        assert len(resp.content) > 100  # non-trivial DOCX
        # Check it's valid DOCX (starts with PK zip header)
        assert resp.content[:2] == b"PK"

    def test_export_xlsx(self, client, auth_headers):
        content = "| Name | Score |\n|------|-------|\n| Alice | 95 |\n| Bob | 87 |"
        resp = client.post("/export", headers={**auth_headers, "Content-Type": "application/json"},
                           json={"content": content, "format": "xlsx", "filename": "scores"})
        assert resp.status_code == 200
        assert "spreadsheetml" in resp.headers["content-type"]
        assert resp.content[:2] == b"PK"

    def test_export_xlsx_no_tables(self, client, auth_headers):
        resp = client.post("/export", headers={**auth_headers, "Content-Type": "application/json"},
                           json={"content": "Just plain text, no tables.", "format": "xlsx", "filename": "plain"})
        assert resp.status_code == 200
        assert resp.content[:2] == b"PK"

    def test_export_unsupported_format(self, client, auth_headers):
        resp = client.post("/export", headers={**auth_headers, "Content-Type": "application/json"},
                           json={"content": "test", "format": "pdf", "filename": "test"})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Scanning a document. The 10 MB cap was arbitrary; the silent page cap behind
# it was the real bug — a 400-page book came back as 20 pages that merely
# appeared to end.
# ---------------------------------------------------------------------------

class TestScanning:
    def test_the_size_cap_is_configurable_and_not_ten_megabytes(self):
        from researcher import ingestion

        assert ingestion.MAX_FILE_SIZE > 10 * 1024 * 1024
        src = pathlib.Path(ingestion.__file__).read_text()
        assert 'os.getenv("MAX_FILE_SIZE_MB"' in src

    def test_the_error_names_the_actual_size_and_limit(self):
        from researcher import ingestion

        src = pathlib.Path(ingestion.__file__).read_text()
        assert "MAX_FILE_SIZE_MB" in src
        assert '"File too large (max 10 MB)"' not in src

    def test_skipped_pages_are_announced_in_the_text(self):
        """They used to be dropped with a bare `continue`."""
        from researcher import ingestion

        src = pathlib.Path(ingestion.__file__).read_text()
        body = src[src.index("def _extract_pdf("):src.index("def scan_pdf_pages(")]
        assert "skipped += 1" in body
        assert "were not read" in body
        # The wording wraps across f-string lines, so match a fragment.
        assert "do not treat it as the" in body

    def test_a_page_the_model_cannot_read_is_recorded_not_dropped(self):
        from researcher import ingestion

        src = pathlib.Path(ingestion.__file__).read_text()
        body = src[src.index("def scan_pdf_pages("):src.index("def _extract_docx(")]
        assert "could not be read" in body
        assert 'stats["empty"]' in body

    def test_the_scan_has_no_page_cap_of_its_own(self):
        """The caller chose a range and accepts the wait."""
        from researcher import ingestion

        src = pathlib.Path(ingestion.__file__).read_text()
        body = src[src.index("def scan_pdf_pages("):src.index("def _extract_docx(")]
        assert "OCR_MAX_PAGES" not in body

    def test_deepseek_ocr_is_preferred_and_qwen_vl_is_not_called_video(self):
        from researcher import ingestion

        assert ingestion._scan_score("deepseek-ocr:3b")[0] > \
            ingestion._scan_score("qwen3-vl:8b")[0]
        # Both are document models; the ranking is about speed on dense text.
        assert "document model" in ingestion._scan_score("qwen3-vl:8b")[1]
        assert ingestion._scan_score("qwen3-vl:8b")[0] > \
            ingestion._scan_score("llama3:8b")[0]

    def test_an_unreachable_ollama_gives_an_empty_list_not_a_crash(self, monkeypatch):
        import requests

        from researcher import ingestion

        def _boom(*a, **k):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(requests, "get", _boom)
        assert ingestion.scan_models() == []
        # And the configured model is still offered as the fallback.
        assert ingestion.best_scan_model() == ingestion.OCR_MODEL

    def test_the_scan_holds_the_gpu_and_says_so_to_everyone(self):
        from researcher import ingestion

        src = pathlib.Path(ingestion.__file__).read_text()
        body = src[src.index('@app.post("/files/scan")'):]
        assert 'GPU.reserve(Need("ollama", chosen), "scan"' in body
        assert "res.say(" in body, "the queue must show what it is scanning"
        assert "queue_message(" in body

    def test_a_running_job_can_publish_progress_to_the_queue(self):
        import inspect

        from researcher.gpu_broker import Reservation, _Passthrough

        assert "self.entry.note" in inspect.getsource(Reservation.say)
        # The pass-through must honour the same interface or a nested
        # reservation would crash on it.
        assert hasattr(_Passthrough, "say")

    def test_only_pdfs_are_scanned(self):
        from researcher import ingestion

        src = pathlib.Path(ingestion.__file__).read_text()
        body = src[src.index('@app.post("/files/scan")'):]
        assert 'ext != ".pdf"' in body

    def test_a_model_that_is_not_installed_is_refused(self):
        from researcher import ingestion

        src = pathlib.Path(ingestion.__file__).read_text()
        body = src[src.index('@app.post("/files/scan")'):]
        assert "not an installed vision model" in body

    def test_a_book_is_flagged_as_too_long_for_the_context(self):
        """Nothing truncates the text, so attaching a book silently overflows."""
        from researcher import ingestion

        src = pathlib.Path(ingestion.__file__).read_text()
        assert '"too_long_for_context"' in src

    def test_the_page_is_wired_up(self):
        static = pathlib.Path(__file__).resolve().parents[1] / "src" / "researcher" / "static"
        html = (static / "index.html").read_text()
        assert 'id="scan-btn"' in html
        assert 'id="scan-model"' in html
        js = (static / "js" / "files.js").read_text()
        assert "/files/scan/models" in js
        assert "openScanDialog" in js
        # The dialog must say that the GPU is held for the whole scan.
        assert "holds the GPU" in html


class TestAttachingSomethingLong:
    """A scanned book is far more text than the window holds.

    448855-The Human Zoo came to 493,647 characters over 260 pages against a
    num_ctx of 32768 (~131k characters). get_file_context used to send it
    whole under a banner calling it the FULL TEXT: the tail is dropped or the
    call errors inside litellm, and either way the model answers from a
    fragment whose edges it cannot see.
    """

    BOOK = "\n\n".join(
        [f"[Page {n}]\nfiller text about nothing in particular. " * 12
         for n in range(1, 40)]
        + ["[Page 40]\n" + "overcrowding in cities and the status game. " * 12]
    )

    def test_a_short_attachment_is_sent_whole(self):
        from researcher import ingestion

        text = "[Page 1]\nshort enough"
        chunk, info = ingestion.select_pages(text, "anything", 10000)
        assert chunk == text
        assert info["mode"] == "head"

    def test_the_pages_matching_the_question_are_chosen(self):
        from researcher import ingestion

        chunk, info = ingestion.select_pages(
            self.BOOK, "what about overcrowding in cities?", 1200)
        assert info["mode"] == "selected"
        assert 40 in info["numbers"], "the only relevant page must be included"
        assert "overcrowding" in chunk

    def test_the_selection_stays_inside_the_budget(self):
        from researcher import ingestion

        for budget in (600, 1500, 6000):
            chunk, info = ingestion.select_pages(self.BOOK, "status game", budget)
            assert len(chunk) <= budget + 200, (budget, len(chunk))

    def test_chosen_pages_keep_reading_order_and_their_numbers(self):
        """So the model can cite a page, and quotes are not reordered."""
        from researcher import ingestion

        chunk, info = ingestion.select_pages(self.BOOK, "overcrowding status", 4000)
        assert info["numbers"] == sorted(info["numbers"])
        assert "[Page 40]" in chunk

    def test_a_question_matching_nothing_falls_back_to_the_opening(self):
        from researcher import ingestion

        chunk, info = ingestion.select_pages(self.BOOK, "zzzqqq xyzzy", 1200)
        assert info["mode"] == "head"
        assert chunk.startswith("[Page 1]")

    def test_an_empty_question_takes_the_opening(self):
        from researcher import ingestion

        chunk, info = ingestion.select_pages(self.BOOK, "", 1200)
        assert info["mode"] == "head"
        assert chunk.startswith("[Page 1]")

    def test_text_with_no_page_markers_is_still_handled(self):
        """A .docx or .txt has no pages to select."""
        from researcher import ingestion

        chunk, info = ingestion.select_pages("x" * 50000, "anything", 1000)
        assert info["mode"] == "head"
        assert len(chunk) == 1000

    def test_a_single_page_bigger_than_the_budget_is_cut_not_dropped(self):
        from researcher import ingestion

        chunk, info = ingestion.select_pages("[Page 1]\n" + "y" * 9000, "y", 500)
        assert len(chunk) == 500

    def test_stopwords_do_not_drive_the_selection(self):
        """"What does the author say about X" must select on X."""
        from researcher import ingestion

        assert ingestion._keywords("What does the author say about overcrowding?") \
            == ["author", "say", "overcrowding"]
        assert "the" not in ingestion._keywords("the the the")

    def test_the_budget_follows_the_models_context_window(self):
        from researcher import ingestion

        small = ingestion.attach_budget(8192)
        big = ingestion.attach_budget(32768)
        assert small < big
        # A third of the window at ~4 characters a token.
        assert abs(big - 32768 * 4 / 3) < 10
        assert ingestion.attach_budget(0) == ingestion.ATTACH_BUDGET_CHARS

    def test_the_banner_stops_claiming_full_text_when_it_is_not(self):
        """The old wording would have been a lie once anything was dropped."""
        from researcher import ingestion

        src = pathlib.Path(ingestion.__file__).read_text()
        body = src[src.index("async def get_file_context("):]
        assert "TOO LONG to include whole" in body
        assert "cite their page numbers" in body
        # And it must forbid filling in the rest from imagination.
        assert "do NOT guess at the rest" in body

    def test_every_call_site_passes_the_question(self):
        """Without it a book silently degrades to its opening pages."""
        root = pathlib.Path(__file__).resolve().parents[1] / "src" / "researcher"
        for rel in ("routes/chat.py", "routes/ask.py"):
            src = (root / rel).read_text()
            for call in src.split("get_file_context(")[1:]:
                head = call[:220]
                assert "question=" in head, f"{rel}: a call site omits the question"

    def test_get_file_context_actually_reduces_a_long_attachment(self):
        """The behavioural test. Asserting select_pages works in isolation let
        a mutation that sent the whole text again pass every other test here."""
        from researcher import ingestion

        async def go():
            import aiosqlite

            db = await aiosqlite.connect(":memory:")
            await db.execute("CREATE TABLE files (id TEXT, user_id TEXT, "
                             "filename TEXT, extracted_text TEXT)")
            await db.execute("INSERT INTO files VALUES ('f1','u','book.pdf',?)",
                             (self.BOOK,))
            await db.commit()
            block = await ingestion.get_file_context(
                db, "u", ["f1"], question="overcrowding in cities", budget=1200)
            await db.close()
            return block

        block = asyncio.run(go())
        assert len(block) < len(self.BOOK), "the whole document was sent"
        assert len(block) < 1200 + 1500, "the budget was not applied"
        assert "TOO LONG to include whole" in block
        assert "overcrowding" in block
        assert "FULL TEXT" not in block

    def test_a_short_attachment_still_arrives_whole_and_is_called_full(self):
        from researcher import ingestion

        async def go():
            import aiosqlite

            db = await aiosqlite.connect(":memory:")
            await db.execute("CREATE TABLE files (id TEXT, user_id TEXT, "
                             "filename TEXT, extracted_text TEXT)")
            await db.execute("INSERT INTO files VALUES ('f1','u','note.txt',?)",
                             ("a short note about cheese",))
            await db.commit()
            block = await ingestion.get_file_context(db, "u", ["f1"],
                                                     question="cheese")
            await db.close()
            return block

        block = asyncio.run(go())
        assert "a short note about cheese" in block
        assert "FULL TEXT" in block
        assert "TOO LONG" not in block


class TestAnAlreadyOcrdScan:
    """Bogdan doubted that 255 of 260 pages "had text", since the book is an
    image scan. Both were true: it is 260 page images with an OCR text layer
    added by whoever made the file — hence its name, 448855-The Human
    Zoo_text.pdf. That layer measured 3 obvious errors in 16,294 words, nearly
    all on capitalised front matter, so reading it was right. But its quality
    is unknowable in advance, so the choice is now exposed.
    """

    def test_a_text_layer_can_be_ignored_on_request(self):
        from researcher import ingestion

        src = pathlib.Path(ingestion.__file__).read_text()
        body = src[src.index("def scan_pdf_pages("):src.index("def _extract_docx(")]
        assert 'text = "" if force_ocr else page.get_text()' in body

    def test_the_scan_reports_what_the_file_already_has(self):
        """So a long wait is not spent re-reading readable pages."""
        from researcher import ingestion

        src = pathlib.Path(ingestion.__file__).read_text()
        assert "_text_layer_survey" in src
        body = src[src.index('@app.post("/files/scan")'):]
        assert "_text_layer_survey(data)" in body

    def test_the_survey_samples_rather_than_reading_everything(self):
        """It runs before the scan; it must not itself take a minute."""
        import inspect

        from researcher import ingestion

        src = inspect.getsource(ingestion._text_layer_survey)
        assert "step = max(1, total // sample)" in src

    def test_a_corrupt_pdf_survey_does_not_break_the_scan(self):
        from researcher import ingestion

        assert ingestion._text_layer_survey(b"not a pdf") == {}

    def test_the_survey_counts_a_real_text_layer(self):
        import fitz

        from researcher import ingestion

        doc = fitz.open()
        for i in range(6):
            doc.new_page().insert_text((72, 100), f"page {i}")
        data = doc.tobytes()
        doc.close()
        s = ingestion._text_layer_survey(data)
        assert s["pages"] == 6
        assert s["with_text_layer"] == s["sampled"]

    def test_an_image_only_pdf_shows_no_text_layer(self):
        import fitz

        from researcher import ingestion

        src = fitz.open()
        src.new_page().insert_text((72, 100), "hello")
        flat = fitz.open()
        for pg in src:
            pix = pg.get_pixmap(dpi=36)
            flat.new_page(width=pg.rect.width, height=pg.rect.height) \
                .insert_image(pg.rect, pixmap=pix)
        data = flat.tobytes()
        src.close()
        flat.close()
        assert ingestion._text_layer_survey(data)["with_text_layer"] == 0

    def test_forcing_is_recorded_in_the_result(self):
        from researcher import ingestion

        src = pathlib.Path(ingestion.__file__).read_text()
        assert '"forced": bool(force_ocr)' in src

    def test_the_dialog_offers_the_choice_and_explains_the_cost(self):
        static = pathlib.Path(__file__).resolve().parents[1] / "src" / "researcher" / "static"
        html = (static / "index.html").read_text()
        assert 'id="scan-force"' in html
        assert "much slower" in html
        js = (static / "js" / "files.js").read_text()
        assert "force_ocr=${force}" in js
