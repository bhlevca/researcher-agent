"""Math Tutor endpoints, exercised end to end against a temporary database.

The behaviour worth protecting here is that a *seed is a worksheet*: the sheet,
its answer key, and any later reprint all have to be the same questions. The
storage layer keeps only the recipe, so if that stopped holding, an answer key
would silently stop matching the sheet it was printed with.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from researcher.main import app
from researcher import auth as auth_module


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    test_db = tmp_path / "math_test.db"
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
                        json={"username": "mathparent", "password": "testpass123"})
        return {"Authorization": f"Bearer {r.json()['token']}"}
    finally:
        auth_module.INVITE_CODE = orig


class TestCurriculum:
    def test_requires_login(self, client):
        assert client.get("/math/curriculum").status_code == 401

    def test_lists_twelve_grades_with_ib_equivalents(self, client, headers):
        data = client.get("/math/curriculum", headers=headers).json()
        grades = data["grades"]
        assert [g["id"] for g in grades] == [f"grade{i}" for i in range(1, 13)]
        # Both namings must be present, or the equivalence is not visible.
        assert all(g["ib"] and g["ib_label"] for g in grades)
        by_id = {g["id"]: g for g in grades}
        assert by_id["grade8"]["ib"] == "MYP3"
        assert by_id["grade12"]["ib"] == "DP_AA_HL"

    def test_every_offered_topic_is_available(self, client, headers):
        """A topic listed but not generatable is a dead checkbox."""
        data = client.get("/math/curriculum", headers=headers).json()
        dead = [(g["id"], t["id"]) for g in data["grades"]
                for t in g["topics"] if not t["available"]]
        assert not dead, dead


class TestGenerate:
    def test_generates_the_requested_number(self, client, headers):
        r = client.post("/math/generate", headers=headers, json={
            "grade": "grade8", "topics": ["pythagoras", "linear_equations"],
            "count": 10})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 10
        assert len(body["problems"]) == 10
        assert body["seed"]

    def test_accepts_an_ib_code_as_the_level(self, client, headers):
        """The point of carrying both namings."""
        a = client.post("/math/generate", headers=headers, json={
            "grade": "MYP3", "topics": ["pythagoras"], "count": 4, "seed": 7})
        b = client.post("/math/generate", headers=headers, json={
            "grade": "grade8", "topics": ["pythagoras"], "count": 4, "seed": 7})
        assert a.status_code == 200
        assert a.json()["grade"] == "grade8"
        assert a.json()["problems"] == b.json()["problems"]

    def test_label_shows_both_namings(self, client, headers):
        r = client.post("/math/generate", headers=headers, json={
            "grade": "grade8", "topics": ["pythagoras"], "count": 2})
        label = r.json()["label"]
        assert "Grade 8" in label and "MYP" in label

    def test_same_seed_same_sheet(self, client, headers):
        body = {"grade": "grade10", "topics": ["quadratics"], "count": 6, "seed": 42}
        a = client.post("/math/generate", headers=headers, json=body).json()
        b = client.post("/math/generate", headers=headers, json=body).json()
        assert a["problems"] == b["problems"]

    def test_unknown_level_is_rejected(self, client, headers):
        r = client.post("/math/generate", headers=headers,
                        json={"grade": "grade99", "count": 4})
        assert r.status_code == 400

    def test_unknown_topic_is_rejected(self, client, headers):
        r = client.post("/math/generate", headers=headers,
                        json={"grade": "grade8", "topics": ["astrology"]})
        assert r.status_code == 400

    def test_count_is_capped(self, client, headers):
        """A thousand-question sheet would hang the request, not help anyone."""
        r = client.post("/math/generate", headers=headers, json={
            "grade": "grade8", "topics": ["linear_equations"], "count": 5000})
        assert r.status_code == 200
        assert r.json()["count"] <= 60

    def test_empty_topics_uses_the_whole_grade(self, client, headers):
        r = client.post("/math/generate", headers=headers,
                        json={"grade": "grade4", "topics": [], "count": 10})
        assert r.status_code == 200
        assert len({p["topic"] for p in r.json()["problems"]}) > 1


class TestLanguage:
    def test_curriculum_lists_languages_and_pairs(self, client, headers):
        langs = client.get("/math/curriculum", headers=headers).json()["languages"]
        ids = [l["id"] for l in langs]
        assert {"en", "fr", "ro"} <= set(ids)
        assert any(l["bilingual"] for l in langs)

    def test_french_sheet_is_in_french(self, client, headers):
        body = {"grade": "grade10", "topics": ["sectors_arcs"], "count": 3,
                "seed": 5}
        en = client.post("/math/generate", headers=headers, json=body).json()
        fr = client.post("/math/generate", headers=headers,
                         json=dict(body, lang="fr")).json()
        assert en["problems"][0]["question"] != fr["problems"][0]["question"]
        assert "secteur" in fr["problems"][0]["question"]
        # Translation changes the wording and nothing else.
        assert [p["answer"] for p in en["problems"]] == \
               [p["answer"] for p in fr["problems"]]

    def test_bilingual_sheet_carries_both(self, client, headers):
        r = client.post("/math/generate", headers=headers, json={
            "grade": "grade10", "topics": ["circle_theorems"], "count": 2,
            "seed": 5, "lang": "ro+en"})
        p = r.json()["problems"][0]
        assert p["question_alt"], "no second language"
        assert p["question"] != p["question_alt"]

    def test_saved_sheet_reprints_in_its_own_language(self, client, headers):
        """The language is part of the recipe: a French sheet reprinted in
        English would not be the same sheet."""
        wid = client.post("/math/worksheets", headers=headers, json={
            "grade": "grade10", "topics": ["sectors_arcs"], "count": 3,
            "seed": 5, "title": "Devoir", "lang": "fr"}).json()["id"]
        rec = client.get(f"/math/worksheets/{wid}", headers=headers).json()
        assert rec["lang"] == "fr"
        assert "secteur" in rec["problems"][0]["question"]

    def test_unknown_language_falls_back_to_english(self, client, headers):
        r = client.post("/math/generate", headers=headers, json={
            "grade": "grade10", "topics": ["sectors_arcs"], "count": 2,
            "seed": 5, "lang": "klingon"})
        assert r.status_code == 200
        assert "sector" in r.json()["problems"][0]["question"]


class TestPdf:
    def test_worksheet_and_key_are_pdfs(self, client, headers):
        body = {"grade": "grade8", "topics": ["pythagoras"], "count": 4, "seed": 9}
        sheet = client.post("/math/pdf", headers=headers, json=body)
        key = client.post("/math/answers", headers=headers, json=body)
        if sheet.status_code == 503:
            pytest.skip("pdflatex not installed")
        assert sheet.status_code == key.status_code == 200
        assert sheet.content[:4] == b"%PDF"
        assert key.content[:4] == b"%PDF"
        assert sheet.content != key.content

    def test_student_name_is_printed_on_the_sheet_only(self, client, headers):
        """The key goes to the adult; it should not carry the child's name."""
        body = {"grade": "grade4", "topics": ["decimals"], "count": 3,
                "seed": 11, "student": "Ana"}
        sheet = client.post("/math/pdf", headers=headers, json=body)
        if sheet.status_code == 503:
            pytest.skip("pdflatex not installed")
        plain = dict(body, student="")
        assert sheet.content != client.post("/math/pdf", headers=headers,
                                            json=plain).content


class TestSavedWorksheets:
    def _save(self, client, headers, **kw):
        body = {"grade": "grade8", "topics": ["pythagoras"], "count": 5,
                "seed": 21, "title": "Homework 1", "student": "Ana"}
        body.update(kw)
        return client.post("/math/worksheets", headers=headers, json=body)

    def test_save_and_list(self, client, headers):
        r = self._save(client, headers)
        assert r.status_code == 200
        wid = r.json()["id"]
        listing = client.get("/math/worksheets", headers=headers).json()
        assert [w["id"] for w in listing["worksheets"]] == [wid]

    def test_a_saved_sheet_regenerates_identically(self, client, headers):
        """The whole reason only the recipe is stored."""
        wid = self._save(client, headers).json()["id"]
        saved = client.get(f"/math/worksheets/{wid}", headers=headers).json()
        fresh = client.post("/math/generate", headers=headers, json={
            "grade": "grade8", "topics": ["pythagoras"], "count": 5,
            "seed": 21, "title": "Homework 1"}).json()
        assert saved["problems"] == fresh["problems"]

    def test_reprint_from_storage(self, client, headers):
        wid = self._save(client, headers).json()["id"]
        sheet = client.get(f"/math/worksheets/{wid}/pdf", headers=headers)
        if sheet.status_code == 503:
            pytest.skip("pdflatex not installed")
        assert sheet.content[:4] == b"%PDF"
        key = client.get(f"/math/worksheets/{wid}/pdf?answers=true", headers=headers)
        assert key.content[:4] == b"%PDF"

    def test_another_user_cannot_read_it(self, client, headers):
        wid = self._save(client, headers).json()["id"]
        auth_module._rate_log.clear()
        orig = auth_module.INVITE_CODE
        auth_module.INVITE_CODE = ""
        try:
            r = client.post("/auth/register",
                            json={"username": "someoneelse", "password": "testpass123"})
        finally:
            auth_module.INVITE_CODE = orig
        other = {"Authorization": f"Bearer {r.json()['token']}"}
        assert client.get(f"/math/worksheets/{wid}", headers=other).status_code == 404
        assert client.get("/math/worksheets", headers=other).json()["worksheets"] == []

    def test_delete(self, client, headers):
        wid = self._save(client, headers).json()["id"]
        assert client.delete(f"/math/worksheets/{wid}", headers=headers).status_code == 200
        assert client.get(f"/math/worksheets/{wid}", headers=headers).status_code == 404

    def test_missing_worksheet_is_404(self, client, headers):
        assert client.get("/math/worksheets/nope", headers=headers).status_code == 404


class TestMarking:
    def test_results_feed_the_progress_view(self, client, headers):
        wid = client.post("/math/worksheets", headers=headers, json={
            "grade": "grade8", "topics": ["pythagoras", "linear_equations"],
            "count": 4, "seed": 3, "title": "HW"}).json()["id"]
        r = client.post("/math/results", headers=headers, json={
            "worksheet_id": wid, "student": "Ana", "marks": [
                {"position": 1, "topic": "pythagoras", "correct": True},
                {"position": 2, "topic": "pythagoras", "correct": False},
                {"position": 3, "topic": "linear_equations", "correct": True},
                {"position": 4, "topic": "linear_equations", "correct": True},
            ]})
        assert r.json()["recorded"] == 4

        topics = client.get("/math/progress", headers=headers).json()["topics"]
        by = {t["topic"]: t for t in topics}
        assert by["pythagoras"]["accuracy"] == 0.5
        assert by["linear_equations"]["accuracy"] == 1.0
        # Weakest first is the ordering a parent needs.
        assert topics[0]["topic"] == "pythagoras"
        assert by["pythagoras"]["name"] == "Pythagoras' theorem"

    def test_marks_against_an_unknown_sheet_are_rejected(self, client, headers):
        r = client.post("/math/results", headers=headers,
                        json={"worksheet_id": "nope", "marks": []})
        assert r.status_code == 404


class TestPage:
    def test_math_page_is_served(self, client):
        r = client.get("/math")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
