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


_STORY_JSON = '''```json
{"title": "Le chat perdu",
 "story": "Madame Blanc habite près de la forêt.\\n\\nUn jour, le chat ne revient pas.",
 "questions": [{"q": "Où habite Madame Blanc ?", "a": "Près de la forêt."},
               {"q": "Que se passe-t-il ?", "a": "Le chat ne revient pas."}],
 "summary_prompt": "Résumez l'histoire en 40 mots.",
 "summary_answer": "Le chat de Madame Blanc disparaît.",
 "glossary": [{"word": "la forêt", "gloss": "forest"}]}
```'''


class TestStorySheets:
    """The model-written half. Its answers are MODEL answers, and its text is
    stored rather than regenerated — a second call writes a different story."""

    @pytest.fixture
    def mock_story(self, monkeypatch):
        """Stand in for the model, so the test needs no Ollama."""
        crew = MagicMock()
        crew.kickoff.return_value = _STORY_JSON
        tutor = MagicMock()
        tutor.build_story_sheet_crew.return_value = crew
        tutor.get_llm_params.return_value = {"crew_timeout_sec": 30}
        monkeypatch.setattr("researcher.routes.tutor_sheets._tutor_crew",
                            lambda request, uid: tutor)
        return tutor

    def test_generate_parses_a_fenced_response(self, client, headers, mock_story):
        r = client.post("/tutor/sheets/story", headers=headers,
                        json={"language": "French", "level": "A2",
                              "topic": "a lost cat"})
        assert r.status_code == 200
        c = r.json()["content"]
        assert c["title"] == "Le chat perdu"
        assert len(c["questions"]) == 2
        assert c["summary_answer"]

    def test_a_bad_model_response_is_reported_not_printed(
            self, client, headers, mock_story):
        """An empty sheet is worse than an error."""
        mock_story.build_story_sheet_crew.return_value.kickoff.return_value = \
            "I'm sorry, I can't do that."
        r = client.post("/tutor/sheets/story", headers=headers,
                        json={"language": "French", "level": "A2"})
        assert r.status_code == 502

    def test_sheet_and_key_compile(self, client, headers, mock_story):
        content = client.post("/tutor/sheets/story", headers=headers,
                              json={"language": "French", "level": "A2"}
                              ).json()["content"]
        body = {"content": content, "language": "French", "level": "A2",
                "kind": "homework"}
        sheet = client.post("/tutor/sheets/story/pdf", headers=headers, json=body)
        key = client.post("/tutor/sheets/story/answers", headers=headers, json=body)
        if sheet.status_code == 503:
            pytest.skip("pdflatex not installed")
        assert sheet.content[:4] == b"%PDF"
        assert key.content[:4] == b"%PDF"
        assert sheet.content != key.content

    def test_printing_does_not_call_the_model(self, client, headers, mock_story):
        """The client carries the story back; printing must never regenerate."""
        content = client.post("/tutor/sheets/story", headers=headers,
                              json={"language": "French", "level": "A2"}
                              ).json()["content"]
        before = mock_story.build_story_sheet_crew.call_count
        client.post("/tutor/sheets/story/pdf", headers=headers,
                    json={"content": content, "language": "French", "level": "A2"})
        assert mock_story.build_story_sheet_crew.call_count == before

    def test_saved_story_reprints_the_same_text(self, client, headers, mock_story):
        content = client.post("/tutor/sheets/story", headers=headers,
                              json={"language": "French", "level": "A2"}
                              ).json()["content"]
        sid = client.post("/tutor/sheets/story/save", headers=headers, json={
            "content": content, "language": "French", "level": "A2",
            "kind": "homework"}).json()["id"]
        rec = client.get(f"/tutor/sheets/{sid}", headers=headers).json()
        assert rec["content"]["story"] == content["story"]
        assert rec["content"]["questions"] == content["questions"]
        pdf = client.get(f"/tutor/sheets/{sid}/pdf", headers=headers)
        if pdf.status_code == 503:
            pytest.skip("pdflatex not installed")
        assert pdf.content[:4] == b"%PDF"

    def test_empty_story_is_refused(self, client, headers, mock_story):
        r = client.post("/tutor/sheets/story/pdf", headers=headers, json={
            "content": {"title": "", "story": "", "questions": []},
            "language": "French", "level": "A2"})
        assert r.status_code == 400


class TestStoryParsing:
    """parse_story must cope with how models actually reply."""

    @pytest.mark.parametrize("wrapper", [
        "{body}",
        "```json\n{body}\n```",
        "```\n{body}\n```",
        "Here you go:\n\n{body}\n\nHope that helps!",
    ])
    def test_handles_fences_and_surrounding_prose(self, wrapper):
        from researcher.tutor.story import parse_story

        body = ('{"title":"T","story":"S","questions":[{"q":"Q","a":"A"}],'
                '"summary_prompt":"P","summary_answer":"M","glossary":[]}')
        got = parse_story(wrapper.format(body=body))
        assert got["title"] == "T" and got["questions"][0]["q"] == "Q"

    def test_repairs_a_trailing_comma(self):
        from researcher.tutor.story import parse_story

        got = parse_story('{"title":"T","story":"S",'
                          '"questions":[{"q":"Q","a":"A"},],}')
        assert got["questions"][0]["a"] == "A"

    def test_questions_given_as_bare_strings_still_work(self):
        from researcher.tutor.story import parse_story

        got = parse_story('{"title":"T","story":"S","questions":["Q1","Q2"]}')
        assert [q["q"] for q in got["questions"]] == ["Q1", "Q2"]

    def test_a_response_with_no_json_raises(self):
        from researcher.tutor.story import parse_story

        with pytest.raises(ValueError):
            parse_story("I cannot help with that.")

    def test_missing_required_keys_raises(self):
        from researcher.tutor.story import parse_story

        with pytest.raises(ValueError):
            parse_story('{"title": "T"}')


_LESSON_MD = """# Lesson: Daily routines

## 1. Objectives
- Describe a daily routine
- Use reflexive verbs

## 2. Grammar
The pattern is **se + verb**.

| Person | Pronoun | se lever |
|--------|---------|----------|
| je | me | je me lève |
| tu | te | tu te lèves |

## 3. Practice Exercises
1. Complete: Je ______ (se lever).
2. Translate: She wakes up early.

## 4. Solutions
1. Je **me lève**.
2. Elle se réveille tôt.
"""


class TestLessonSheets:
    """A lesson already ends with Solutions, so the sheet and key are the same
    document cut in half — no second model call, and they cannot disagree."""

    def test_split_finds_the_solutions_section(self):
        from researcher.tutor.markdown_tex import split_solutions

        body, sol = split_solutions(_LESSON_MD)
        assert "Objectives" in body and "Solutions" not in body
        assert "me lève" in sol

    @pytest.mark.parametrize("heading", [
        "## Solutions", "## 7. Solutions", "# Answer Key",
        "## Solutions to the exercises", "## Answers", "## Corrigé",
    ])
    def test_split_accepts_the_headings_models_actually_write(self, heading):
        from researcher.tutor.markdown_tex import split_solutions

        body, sol = split_solutions(f"# L\n\nBody text\n\n{heading}\n\n1. A")
        assert "Body text" in body
        assert "1. A" in sol

    def test_a_lesson_without_solutions_yields_no_key(self):
        from researcher.tutor import printing

        md = "# Lesson\n\n## Grammar\nSome text.\n"
        assert not printing.lesson_has_answers(md)
        assert printing.render_lesson_pdf(md, title="T", subtitle="S",
                                          answers=True) is None

    def test_sheet_and_key_compile(self, client, headers):
        body = {"markdown": _LESSON_MD, "title": "Daily routines",
                "language": "French", "level": "A2"}
        sheet = client.post("/tutor/sheets/lesson/pdf", headers=headers, json=body)
        key = client.post("/tutor/sheets/lesson/answers", headers=headers, json=body)
        if sheet.status_code == 503:
            pytest.skip("pdflatex not installed")
        assert sheet.content[:4] == b"%PDF"
        assert key.content[:4] == b"%PDF"
        assert sheet.content != key.content

    def test_asking_for_a_key_that_does_not_exist_says_so(self, client, headers):
        """422 with an explanation beats a PDF that silently repeats the sheet."""
        r = client.post("/tutor/sheets/lesson/answers", headers=headers,
                        json={"markdown": "# Lesson\n\nNo solutions here.",
                              "language": "French", "level": "A2"})
        assert r.status_code == 422
        assert "Solutions" in r.json()["detail"]

    def test_markdown_tables_and_markup_survive(self):
        from researcher.tutor.markdown_tex import to_latex

        tex = to_latex(_LESSON_MD)
        assert r"\begin{tabular}" in tex
        assert tex.count(r"\hline") >= 3      # top, under header, bottom
        assert r"\textbf{se + verb}" in tex
        assert r"\begin{itemize}" in tex and r"\begin{enumerate}" in tex

    def test_latex_specials_in_a_lesson_are_escaped(self):
        """A model writing "50% of verbs" or "a_b" must not break the compile."""
        from researcher.tutor import printing

        md = ("# L\n\n100% of verbs & more_things #1 {braces} ~tilde^ $5\n\n"
              "## Solutions\n\n1. 50% done\n")
        pdf = printing.render_lesson_pdf(md, title="T", subtitle="S")
        if not printing.latex_available():
            pytest.skip("pdflatex not installed")
        assert pdf and pdf[:4] == b"%PDF"

    def test_a_ragged_table_still_prints(self):
        """A model that drops a cell should cost a blank box, not the sheet."""
        from researcher.tutor.markdown_tex import to_latex

        tex = to_latex("| a | b | c |\n|---|---|---|\n| 1 | 2 |\n")
        assert r"\begin{tabular}" in tex

    def test_printing_a_saved_lesson_needs_no_model(self, client, headers,
                                                    monkeypatch):
        """The whole point of printing what was saved earlier."""
        from researcher.tutor import storage

        async def fake_get(db, lesson_id, user_id):
            return {"title": "Daily routines", "target_lang": "French",
                    "level": "A2", "content": {"markdown": _LESSON_MD}}

        monkeypatch.setattr("researcher.routes.tutor_sheets.get_lesson_plan",
                            fake_get)
        r = client.get("/tutor/lessons/abc/sheet", headers=headers)
        if r.status_code == 503:
            pytest.skip("pdflatex not installed")
        assert r.content[:4] == b"%PDF"
        k = client.get("/tutor/lessons/abc/sheet?answers=true", headers=headers)
        assert k.content[:4] == b"%PDF"

    def test_missing_saved_lesson_is_404(self, client, headers, monkeypatch):
        async def fake_get(db, lesson_id, user_id):
            return None

        monkeypatch.setattr("researcher.routes.tutor_sheets.get_lesson_plan",
                            fake_get)
        assert client.get("/tutor/lessons/nope/sheet",
                          headers=headers).status_code == 404
