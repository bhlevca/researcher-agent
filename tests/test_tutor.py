"""Tests for the Language Tutor feature.

Tests cover:
- Tutor session CRUD
- Vocabulary management
- Lesson plan storage
- Quiz generation/grading
- Appraisal endpoint
- TutorCrew helper methods (vocabulary extraction, quiz JSON parsing)
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from researcher.main import app
from researcher import auth as auth_module
from researcher.tutor.crew import TutorCrew


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_external_services(tmp_path, monkeypatch):
    """Patch heavy dependencies so tests don't need real services."""
    test_db = tmp_path / "test_sessions.db"
    monkeypatch.setattr("researcher.config.DB_PATH", test_db)
    monkeypatch.setattr("researcher.main.DB_PATH", test_db)
    test_db.parent.mkdir(parents=True, exist_ok=True)

    # Mock ResearchCrew
    mock_rc = MagicMock()
    monkeypatch.setattr("researcher.main.ResearchCrew", mock_rc)

    # Mock TutorCrew — lightweight stub for non-crew tests
    mock_tc = MagicMock(spec=TutorCrew)
    mock_tc.return_value = mock_tc
    mock_tc._model_name = "ollama/test-model"
    mock_tc.extract_vocabulary_from_response = TutorCrew.extract_vocabulary_from_response
    mock_tc.extract_quiz_json = TutorCrew.extract_quiz_json
    monkeypatch.setattr("researcher.main.TutorCrew", lambda **kw: mock_tc)


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
        resp = client.post(
            "/auth/register",
            json={"username": "tutoruser", "password": "testpass123"},
        )
        token = resp.json()["token"]
        return {"Authorization": f"Bearer {token}"}
    finally:
        auth_module.INVITE_CODE = orig


# ---------------------------------------------------------------------------
# Tutor Sessions CRUD
# ---------------------------------------------------------------------------


class TestTutorSessions:
    def test_create_session(self, client, auth_headers):
        resp = client.post(
            "/tutor/sessions",
            json={
                "name": "French Beginner",
                "target_lang": "French",
                "native_lang": "English",
                "level": "A1",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "French Beginner"
        assert data["target_lang"] == "French"
        assert data["level"] == "A1"
        assert "id" in data

    def test_list_sessions(self, client, auth_headers):
        # Create two sessions
        client.post(
            "/tutor/sessions",
            json={"name": "French", "target_lang": "French"},
            headers=auth_headers,
        )
        client.post(
            "/tutor/sessions",
            json={"name": "Spanish", "target_lang": "Spanish"},
            headers=auth_headers,
        )

        resp = client.get("/tutor/sessions", headers=auth_headers)
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        assert len(sessions) == 2

    def test_get_session(self, client, auth_headers):
        create_resp = client.post(
            "/tutor/sessions",
            json={"name": "German", "target_lang": "German", "level": "B1"},
            headers=auth_headers,
        )
        sid = create_resp.json()["id"]

        resp = client.get(f"/tutor/sessions/{sid}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["target_lang"] == "German"
        assert data["level"] == "B1"
        assert data["messages"] == []

    def test_delete_session(self, client, auth_headers):
        create_resp = client.post(
            "/tutor/sessions",
            json={"name": "Italian", "target_lang": "Italian"},
            headers=auth_headers,
        )
        sid = create_resp.json()["id"]

        resp = client.delete(f"/tutor/sessions/{sid}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["deleted"] == sid

        # Verify it's gone
        resp = client.get(f"/tutor/sessions/{sid}", headers=auth_headers)
        assert resp.status_code == 404

    def test_invalid_session_id(self, client, auth_headers):
        resp = client.get("/tutor/sessions/INVALID!", headers=auth_headers)
        assert resp.status_code == 400

    def test_session_not_found(self, client, auth_headers):
        resp = client.get("/tutor/sessions/deadbeef", headers=auth_headers)
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class TestVocabulary:
    def _create_session(self, client, auth_headers):
        resp = client.post(
            "/tutor/sessions",
            json={"name": "French Vocab", "target_lang": "French"},
            headers=auth_headers,
        )
        return resp.json()["id"]

    def test_add_vocabulary(self, client, auth_headers):
        sid = self._create_session(client, auth_headers)
        resp = client.post(
            "/tutor/vocabulary",
            json={
                "session_id": sid,
                "word": "bonjour",
                "translation": "hello",
                "part_of_speech": "interjection",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["word"] == "bonjour"
        assert data["translation"] == "hello"
        assert data["mastery_level"] == 0

    def test_list_vocabulary(self, client, auth_headers):
        sid = self._create_session(client, auth_headers)
        client.post(
            "/tutor/vocabulary",
            json={"session_id": sid, "word": "chat", "translation": "cat"},
            headers=auth_headers,
        )
        client.post(
            "/tutor/vocabulary",
            json={"session_id": sid, "word": "chien", "translation": "dog"},
            headers=auth_headers,
        )

        resp = client.get(f"/tutor/sessions/{sid}/vocabulary", headers=auth_headers)
        assert resp.status_code == 200
        vocab = resp.json()["vocabulary"]
        assert len(vocab) == 2

    def test_list_vocabulary_by_lang(self, client, auth_headers):
        """Vocabulary listed by target language spans multiple sessions."""
        sid1 = self._create_session(client, auth_headers)
        sid2 = client.post(
            "/tutor/sessions",
            json={"name": "French 2", "target_lang": "French"},
            headers=auth_headers,
        ).json()["id"]
        # Different language session
        sid3 = client.post(
            "/tutor/sessions",
            json={"name": "German Vocab", "target_lang": "German"},
            headers=auth_headers,
        ).json()["id"]

        client.post("/tutor/vocabulary", json={"session_id": sid1, "word": "chat", "translation": "cat"}, headers=auth_headers)
        client.post("/tutor/vocabulary", json={"session_id": sid2, "word": "chien", "translation": "dog"}, headers=auth_headers)
        client.post("/tutor/vocabulary", json={"session_id": sid3, "word": "Hund", "translation": "dog"}, headers=auth_headers)

        # French should return both
        resp = client.get("/tutor/vocabulary?lang=French", headers=auth_headers)
        assert resp.status_code == 200
        vocab = resp.json()["vocabulary"]
        assert len(vocab) == 2
        words = {v["word"] for v in vocab}
        assert words == {"chat", "chien"}

        # German should return one
        resp = client.get("/tutor/vocabulary?lang=German", headers=auth_headers)
        assert len(resp.json()["vocabulary"]) == 1

    def test_delete_vocabulary(self, client, auth_headers):
        sid = self._create_session(client, auth_headers)
        add_resp = client.post(
            "/tutor/vocabulary",
            json={"session_id": sid, "word": "merci", "translation": "thanks"},
            headers=auth_headers,
        )
        vid = add_resp.json()["id"]

        resp = client.delete(f"/tutor/vocabulary/{vid}", headers=auth_headers)
        assert resp.status_code == 200

        # Verify it's gone
        vocab_resp = client.get(f"/tutor/sessions/{sid}/vocabulary", headers=auth_headers)
        assert len(vocab_resp.json()["vocabulary"]) == 0


# ---------------------------------------------------------------------------
# Lesson Plans
# ---------------------------------------------------------------------------


class TestLessonPlans:
    def _create_session(self, client, auth_headers):
        resp = client.post(
            "/tutor/sessions",
            json={"name": "Spanish Lessons", "target_lang": "Spanish"},
            headers=auth_headers,
        )
        return resp.json()["id"]

    def test_save_lesson(self, client, auth_headers):
        sid = self._create_session(client, auth_headers)
        content = (
            "# Lesson: Greetings\n\n"
            "## Vocabulary\n"
            "| Word | Translation | Part of Speech |\n"
            "|------|-------------|----------------|\n"
            "| hola | hello | interjection |\n"
            "| adiós | goodbye | interjection |\n"
        )
        resp = client.post(
            "/tutor/lessons/save",
            json={
                "session_id": sid,
                "title": "Greetings",
                "topic": "Basic greetings",
                "lesson_type": "vocabulary",
                "content": content,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "Greetings"
        assert data["vocabulary_saved"] == 2

    def test_list_lessons(self, client, auth_headers):
        sid = self._create_session(client, auth_headers)
        client.post(
            "/tutor/lessons/save",
            json={
                "session_id": sid,
                "title": "Lesson 1",
                "topic": "greetings",
                "content": "Hello world",
            },
            headers=auth_headers,
        )

        resp = client.get(f"/tutor/sessions/{sid}/lessons", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()["lessons"]) == 1

    def test_get_lesson(self, client, auth_headers):
        sid = self._create_session(client, auth_headers)
        save_resp = client.post(
            "/tutor/lessons/save",
            json={
                "session_id": sid,
                "title": "Past Tense",
                "topic": "past tense verbs",
                "content": "# Past Tense\nConjugations...",
            },
            headers=auth_headers,
        )
        lid = save_resp.json()["id"]

        resp = client.get(f"/tutor/lessons/{lid}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["title"] == "Past Tense"
        assert "markdown" in resp.json()["content"]


# ---------------------------------------------------------------------------
# Quiz Save & Submit (non-streaming)
# ---------------------------------------------------------------------------


class TestQuizzes:
    def _create_session(self, client, auth_headers):
        resp = client.post(
            "/tutor/sessions",
            json={"name": "Quiz Test", "target_lang": "French"},
            headers=auth_headers,
        )
        return resp.json()["id"]

    def test_save_quiz(self, client, auth_headers):
        sid = self._create_session(client, auth_headers)
        quiz_response = (
            '```json\n'
            '[{"id": 0, "type": "vocabulary", "question": "What is \\"chat\\" in English?", '
            '"options": ["cat", "dog", "bird", "fish"], "correct_answer": "cat"}, '
            '{"id": 1, "type": "vocabulary", "question": "What is \\"chien\\" in English?", '
            '"options": ["cat", "dog", "bird", "fish"], "correct_answer": "dog"}]\n'
            '```'
        )
        resp = client.post(
            "/tutor/quiz/save",
            json={
                "session_id": sid,
                "response": quiz_response,
                "quiz_type": "vocabulary",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["questions"]) == 2

    def test_submit_quiz(self, client, auth_headers):
        sid = self._create_session(client, auth_headers)

        # Save quiz first
        quiz_response = (
            '```json\n'
            '[{"id": 0, "type": "vocabulary", "question": "chat?", '
            '"options": ["cat", "dog"], "correct_answer": "cat"}, '
            '{"id": 1, "type": "vocabulary", "question": "chien?", '
            '"options": ["cat", "dog"], "correct_answer": "dog"}]\n'
            '```'
        )
        save_resp = client.post(
            "/tutor/quiz/save",
            json={"session_id": sid, "response": quiz_response, "quiz_type": "vocabulary"},
            headers=auth_headers,
        )
        quiz_id = save_resp.json()["id"]

        # Submit answers
        resp = client.post(
            "/tutor/quiz/submit",
            json={
                "session_id": sid,
                "quiz_id": quiz_id,
                "answers": [
                    {"question_id": 0, "answer": "cat"},   # correct
                    {"question_id": 1, "answer": "cat"},   # wrong
                ],
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] == 1
        assert data["total"] == 2
        assert data["percentage"] == 50.0

    def test_list_quizzes(self, client, auth_headers):
        sid = self._create_session(client, auth_headers)
        quiz_response = '```json\n[{"id": 0, "type": "vocabulary", "question": "q?", "options": null, "correct_answer": "a"}]\n```'
        client.post(
            "/tutor/quiz/save",
            json={"session_id": sid, "response": quiz_response, "quiz_type": "mixed"},
            headers=auth_headers,
        )

        resp = client.get(f"/tutor/sessions/{sid}/quizzes", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()["quizzes"]) == 1

    def test_save_quiz_invalid_json(self, client, auth_headers):
        sid = self._create_session(client, auth_headers)
        resp = client.post(
            "/tutor/quiz/save",
            json={"session_id": sid, "response": "This is not JSON", "quiz_type": "mixed"},
            headers=auth_headers,
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Student Stats
# ---------------------------------------------------------------------------


class TestStudentStats:
    def test_stats_empty_session(self, client, auth_headers):
        resp = client.post(
            "/tutor/sessions",
            json={"name": "Stats Test", "target_lang": "German"},
            headers=auth_headers,
        )
        sid = resp.json()["id"]

        resp = client.get(f"/tutor/sessions/{sid}/stats", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["vocabulary"]["total_words"] == 0
        assert data["quizzes"]["total_taken"] == 0
        assert data["lessons_completed"] == 0
        assert data["session"]["target_lang"] == "German"


# ---------------------------------------------------------------------------
# TutorCrew helper methods (unit tests, no API)
# ---------------------------------------------------------------------------


class TestTutorCrewHelpers:
    def test_extract_vocabulary_basic(self):
        response = (
            "## Key Vocabulary\n"
            "| Word | Translation | Part of Speech |\n"
            "|------|-------------|----------------|\n"
            "| bonjour | hello | interjection |\n"
            "| merci | thank you | interjection |\n"
            "| au revoir | goodbye | phrase |\n"
        )
        entries = TutorCrew.extract_vocabulary_from_response(response)
        assert len(entries) == 3
        assert entries[0]["word"] == "bonjour"
        assert entries[0]["translation"] == "hello"
        assert entries[1]["word"] == "merci"

    def test_extract_vocabulary_skips_headers(self):
        response = (
            "| Word | Translation | Part of Speech |\n"
            "|------|-------------|----------------|\n"
        )
        entries = TutorCrew.extract_vocabulary_from_response(response)
        assert len(entries) == 0

    def test_extract_vocabulary_empty(self):
        entries = TutorCrew.extract_vocabulary_from_response("No tables here")
        assert entries == []

    def test_extract_quiz_json_fenced(self):
        response = (
            "Here is your quiz:\n\n"
            '```json\n'
            '[{"id": 0, "type": "vocabulary", "question": "What is chat?", '
            '"options": ["cat", "dog"], "correct_answer": "cat"}]\n'
            '```\n'
        )
        questions = TutorCrew.extract_quiz_json(response)
        assert questions is not None
        assert len(questions) == 1
        assert questions[0]["correct_answer"] == "cat"

    def test_extract_quiz_json_raw(self):
        response = '[{"id": 0, "type": "grammar", "question": "q", "options": null, "correct_answer": "a"}]'
        questions = TutorCrew.extract_quiz_json(response)
        assert questions is not None
        assert len(questions) == 1

    def test_extract_quiz_json_invalid(self):
        result = TutorCrew.extract_quiz_json("Not JSON at all")
        assert result is None

    def test_extract_quiz_json_not_array(self):
        result = TutorCrew.extract_quiz_json('{"key": "value"}')
        assert result is None


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


class TestTranslate:
    def test_translate_success(self, client, auth_headers):
        with patch("litellm.completion") as mock_completion:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "Bonjour le monde"
            mock_completion.return_value = mock_resp

            resp = client.post(
                "/tutor/translate",
                json={"text": "Hello world", "source_lang": "English", "target_lang": "French"},
                headers=auth_headers,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["translation"] == "Bonjour le monde"
            assert data["source_lang"] == "English"
            assert data["target_lang"] == "French"

    def test_translate_empty_text(self, client, auth_headers):
        resp = client.post(
            "/tutor/translate",
            json={"text": "", "source_lang": "English", "target_lang": "French"},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_translate_missing_langs(self, client, auth_headers):
        resp = client.post(
            "/tutor/translate",
            json={"text": "Hello", "source_lang": "English"},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_translate_strips_think_tags(self, client, auth_headers):
        with patch("litellm.completion") as mock_completion:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = (
                "<think>Let me think about this translation...</think>Bonjour"
            )
            mock_completion.return_value = mock_resp

            resp = client.post(
                "/tutor/translate",
                json={"text": "Hello", "source_lang": "English", "target_lang": "French"},
                headers=auth_headers,
            )
            assert resp.status_code == 200
            assert resp.json()["translation"] == "Bonjour"

    def test_translate_requires_auth(self, client):
        resp = client.post(
            "/tutor/translate",
            json={"text": "Hello", "source_lang": "English", "target_lang": "French"},
        )
        assert resp.status_code == 401
