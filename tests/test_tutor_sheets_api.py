"""Printable language-tutor sheets, end to end against a temporary database.

As with the math sheets, the property worth protecting is that a *seed is a
sheet*: the paper, its answer key, and any later reprint must be the same
exercises. The storage layer keeps only the recipe, so if that stopped holding,
an answer key would quietly stop matching the test it was printed with.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from researcher.main import app
from researcher import auth as auth_module


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    test_db = tmp_path / "tutor_sheets_test.db"
    monkeypatch.setattr("researcher.config.DB_PATH", test_db)
    monkeypatch.setattr("researcher.main.DB_PATH", test_db)
    mock_rc = MagicMock()
    monkeypatch.setattr("researcher.routes.chat.ResearchCrew", mock_rc)
    monkeypatch.setattr("researcher.routes.ask.ResearchCrew", mock_rc)
    monkeypatch.setattr("researcher.routes.models.ResearchCrew", mock_rc)


@pytest.fixture
def client(_temp_db):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def headers(client):
    auth_module._rate_log.clear()
    orig = auth_module.INVITE_CODE
    auth_module.INVITE_CODE = ""
    try:
        r = client.post("/auth/register",
                        json={"username": "sheetparent", "password": "testpass123"})
        return {"Authorization": f"Bearer {r.json()['token']}"}
    finally:
        auth_module.INVITE_CODE = orig


def _body(**kw):
    base = {"language": "French", "level": "A2", "kind": "homework",
            "sections": [{"skill": "conjugation", "count": 4}]}
    base.update(kw)
    return base


class TestSkills:
    def test_requires_login(self, client):
        assert client.get("/tutor/sheets/skills").status_code == 401

    def test_lists_the_three_printable_languages(self, client, headers):
        d = client.get("/tutor/sheets/skills", headers=headers).json()
        assert d["languages"] == ["French", "German", "Romanian"]
        assert d["deterministic"] is True
        assert {"homework", "test", "exam", "challenge"} <= set(d["kinds"])

    def test_german_offers_cases_and_french_does_not(self, client, headers):
        de = client.get("/tutor/sheets/skills?language=German",
                        headers=headers).json()
        fr = client.get("/tutor/sheets/skills?language=French",
                        headers=headers).json()
        assert "german_case" in [s["id"] for s in de["skills"]]
        assert "german_case" not in [s["id"] for s in fr["skills"]]

    def test_a_chat_only_language_reports_no_pack(self, client, headers):
        """The tutor converses in ten languages but prints sheets for three.
        The UI needs to know which, rather than discovering it on failure."""
        d = client.get("/tutor/sheets/skills?language=Japanese",
                       headers=headers).json()
        assert d["deterministic"] is False
        assert d["skills"] == []

    def test_the_skills_route_is_not_shadowed_by_the_id_route(self, client, headers):
        """/tutor/sheets/skills must not be read as a sheet whose id is
        'skills' — a real risk given both are registered."""
        assert client.get("/tutor/sheets/skills",
                          headers=headers).status_code == 200


class TestGenerate:
    def test_generates_the_requested_sections(self, client, headers):
        r = client.post("/tutor/sheets/generate", headers=headers, json=_body(
            sections=[{"skill": "conjugation", "count": 4},
                      {"skill": "plural", "count": 3}]))
        assert r.status_code == 200
        d = r.json()
        assert len(d["sections"]) == 2
        assert len(d["sections"][0]["exercises"]) == 4
        assert len(d["sections"][1]["exercises"]) == 3
        assert d["seed"]

    def test_sections_get_a_default_instruction(self, client, headers):
        """A sheet with no instruction is not usable without the teacher
        writing one."""
        d = client.post("/tutor/sheets/generate", headers=headers,
                        json=_body()).json()
        assert d["sections"][0]["instruction"]

    def test_marks_are_totalled_for_a_test(self, client, headers):
        d = client.post("/tutor/sheets/generate", headers=headers, json=_body(
            kind="test",
            sections=[{"skill": "conjugation", "count": 4, "marks_each": 1},
                      {"skill": "tense_transform", "count": 3,
                       "marks_each": 2}])).json()
        assert d["total_marks"] == 4 * 1 + 3 * 2

    def test_same_seed_same_sheet(self, client, headers):
        b = _body(seed=31)
        a1 = client.post("/tutor/sheets/generate", headers=headers, json=b).json()
        a2 = client.post("/tutor/sheets/generate", headers=headers, json=b).json()
        assert a1["sections"] == a2["sections"]

    def test_unsupported_language_is_rejected_with_a_useful_message(
            self, client, headers):
        r = client.post("/tutor/sheets/generate", headers=headers,
                        json=_body(language="Japanese"))
        assert r.status_code == 400
        assert "French, German, Romanian" in r.json()["detail"]

    def test_unknown_skill_is_rejected(self, client, headers):
        r = client.post("/tutor/sheets/generate", headers=headers,
                        json=_body(sections=[{"skill": "telepathy", "count": 3}]))
        assert r.status_code == 400

    def test_a_german_skill_on_a_french_sheet_is_rejected(self, client, headers):
        r = client.post("/tutor/sheets/generate", headers=headers,
                        json=_body(sections=[{"skill": "german_case", "count": 3}]))
        assert r.status_code == 400

    def test_no_sections_is_rejected(self, client, headers):
        r = client.post("/tutor/sheets/generate", headers=headers,
                        json=_body(sections=[]))
        assert r.status_code == 400

    def test_count_is_capped(self, client, headers):
        d = client.post("/tutor/sheets/generate", headers=headers, json=_body(
            sections=[{"skill": "conjugation", "count": 5000}])).json()
        assert len(d["sections"][0]["exercises"]) <= 40


class TestPdf:
    def test_sheet_and_key_are_pdfs_and_differ(self, client, headers):
        b = _body(seed=12)
        sheet = client.post("/tutor/sheets/pdf", headers=headers, json=b)
        key = client.post("/tutor/sheets/answers", headers=headers, json=b)
        if sheet.status_code == 503:
            pytest.skip("pdflatex not installed")
        assert sheet.content[:4] == b"%PDF"
        assert key.content[:4] == b"%PDF"
        assert sheet.content != key.content

    @pytest.mark.parametrize("language", ["French", "German", "Romanian"])
    def test_every_language_compiles(self, client, headers, language):
        """Accented characters reach LaTeX in all three; this is where a
        missing glyph would surface."""
        skill = "german_case" if language == "German" else "conjugation"
        r = client.post("/tutor/sheets/answers", headers=headers, json=_body(
            language=language, sections=[{"skill": skill, "count": 5}]))
        if r.status_code == 503:
            pytest.skip("pdflatex not installed")
        assert r.status_code == 200 and r.content[:4] == b"%PDF"

    @pytest.mark.parametrize("kind", ["homework", "test", "exam", "challenge"])
    def test_every_sheet_kind_compiles(self, client, headers, kind):
        r = client.post("/tutor/sheets/pdf", headers=headers,
                        json=_body(kind=kind))
        if r.status_code == 503:
            pytest.skip("pdflatex not installed")
        assert r.status_code == 200 and r.content[:4] == b"%PDF"


class TestSaved:
    def _save(self, client, headers, **kw):
        return client.post("/tutor/sheets", headers=headers,
                           json=_body(seed=21, title="Devoir 1",
                                      student="Ana", **kw))

    def test_save_and_list(self, client, headers):
        sid = self._save(client, headers).json()["id"]
        got = client.get("/tutor/sheets", headers=headers).json()["sheets"]
        assert [s["id"] for s in got] == [sid]

    def test_a_saved_sheet_regenerates_identically(self, client, headers):
        """The whole reason only the recipe is stored."""
        sid = self._save(client, headers).json()["id"]
        saved = client.get(f"/tutor/sheets/{sid}", headers=headers).json()
        fresh = client.post("/tutor/sheets/generate", headers=headers,
                            json=_body(seed=21)).json()
        assert (saved["exercises"][0]["exercises"]
                == fresh["sections"][0]["exercises"])

    def test_reprint_from_storage(self, client, headers):
        sid = self._save(client, headers).json()["id"]
        sheet = client.get(f"/tutor/sheets/{sid}/pdf", headers=headers)
        if sheet.status_code == 503:
            pytest.skip("pdflatex not installed")
        assert sheet.content[:4] == b"%PDF"
        key = client.get(f"/tutor/sheets/{sid}/pdf?answers=true", headers=headers)
        assert key.content[:4] == b"%PDF"

    def test_another_user_cannot_read_it(self, client, headers):
        sid = self._save(client, headers).json()["id"]
        auth_module._rate_log.clear()
        orig = auth_module.INVITE_CODE
        auth_module.INVITE_CODE = ""
        try:
            r = client.post("/auth/register",
                            json={"username": "otherparent",
                                  "password": "testpass123"})
        finally:
            auth_module.INVITE_CODE = orig
        other = {"Authorization": f"Bearer {r.json()['token']}"}
        assert client.get(f"/tutor/sheets/{sid}", headers=other).status_code == 404
        assert client.get("/tutor/sheets", headers=other).json()["sheets"] == []

    def test_delete(self, client, headers):
        sid = self._save(client, headers).json()["id"]
        assert client.delete(f"/tutor/sheets/{sid}",
                             headers=headers).status_code == 200
        assert client.get(f"/tutor/sheets/{sid}",
                          headers=headers).status_code == 404

    def test_missing_sheet_is_404(self, client, headers):
        assert client.get("/tutor/sheets/nope",
                          headers=headers).status_code == 404


class TestExistingTutorStillWorks:
    """The sheets are additive: the conversational tutor must be untouched."""

    def test_tutor_endpoints_are_still_registered(self, client, headers):
        assert client.get("/tutor/curriculum", headers=headers).status_code == 200
        assert client.get("/tutor/sessions", headers=headers).status_code == 200
        assert client.get("/tutor/scenarios", headers=headers).status_code == 200
