"""The model a request runs on is the one the person picked.

This exists because of a bug that looked like the agent inventing a model:

    Error: Model huihui_ai/mistral-small-abliterated not found: 404

Nobody had selected that model in months. It was ``MODEL`` in ``.env``, and the
reason it won is that the picker wrote the choice to the database *and* to
``app.state.current_model``, while the crews were built from the latter. That
global is re-read from the environment on every restart, so the picker showed
the saved model and the crews ran the stale one — and because it is a global,
in a household the last person to touch the picker also changed what everyone
else's crews were built with.

The tests here pin the rule: the database decides, ``.env`` is only the
fallback for someone who has never chosen.
"""

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from researcher import auth as auth_module
from researcher.user_crews import get_crew, resolve_model, verify_default_model


class FakeCrew:
    """Records the model it was built with, and nothing else."""

    DEFAULT_LLM_PARAMS = {"temperature": 0.5}

    def __init__(self, model=None):
        self._model_name = model
        self.params = None

    @property
    def model_name(self):
        return self._model_name

    def update_llm_params(self, params):
        self.params = params
        return params


@pytest.fixture
def client(tmp_path, monkeypatch):
    db = tmp_path / "choice.db"
    monkeypatch.setattr("researcher.config.DB_PATH", db)
    import researcher.main as main
    monkeypatch.setattr(main, "DB_PATH", db)
    # No real models: every crew kind resolves to the recorder above.
    monkeypatch.setattr("researcher.crew.ResearchCrew", FakeCrew)
    monkeypatch.setattr("researcher.tutor.crew.TutorCrew", FakeCrew)
    monkeypatch.setattr("researcher.composer.crew.ComposerCrew", FakeCrew)
    monkeypatch.setattr("researcher.routes.models.ResearchCrew", MagicMock())
    monkeypatch.setattr("researcher.routes.models.TutorCrew", MagicMock())
    monkeypatch.setattr("researcher.routes.models.ComposerCrew", MagicMock())
    with TestClient(main.app) as c:
        c.app.state.default_model = "ollama/env-default"
        c.app.state.current_model = "ollama/env-default"
        yield c


def _register(client, username):
    auth_module._rate_log.clear()
    orig = auth_module.INVITE_CODE
    auth_module.INVITE_CODE = ""
    try:
        r = client.post("/auth/register", json={
            "username": username, "password": "pw123456",
            "invite_code": "", "website": ""})
    finally:
        auth_module.INVITE_CODE = orig
    assert r.status_code == 200, r.text
    body = r.json()
    return body["user"]["id"], {"Authorization": f"Bearer {body['token']}"}


class _Req:
    """Just enough of a Request for the resolver."""

    def __init__(self, app):
        self.app = app


class TestTheChoiceWins:
    def test_a_saved_choice_beats_the_env_default(self, client):
        uid, _ = _register(client, "chooser")

        async def go():
            db = client.app.state.db
            await db.execute("UPDATE users SET model = ? WHERE id = ?",
                             ("ollama/picked-in-the-gui", uid))
            await db.commit()
            crew = await get_crew(_Req(client.app), uid, "research")
            return crew.model_name

        assert client.portal.call(go) == "ollama/picked-in-the-gui"

    def test_no_choice_falls_back_to_the_env_default(self, client):
        uid, _ = _register(client, "nochoice")

        async def go():
            crew = await get_crew(_Req(client.app), uid, "research")
            return crew.model_name

        assert client.portal.call(go) == "ollama/env-default"

    def test_a_restart_does_not_resurrect_the_env_model(self, client):
        """The whole bug in one test.

        A restart resets current_model from the environment. The crew must
        still be built from the saved choice.
        """
        uid, _ = _register(client, "restarter")

        async def go():
            db = client.app.state.db
            await db.execute("UPDATE users SET model = ? WHERE id = ?",
                             ("ollama/picked-in-the-gui", uid))
            await db.commit()
            # Simulate the restart: caches empty, globals back to .env.
            client.app.state.research_crews.clear()
            client.app.state.current_model = "ollama/env-default"
            crew = await get_crew(_Req(client.app), uid, "research")
            return crew.model_name

        assert client.portal.call(go) == "ollama/picked-in-the-gui"

    def test_one_persons_choice_does_not_change_anothers(self, client):
        """current_model was process-wide; this is the household consequence."""
        ana, _ = _register(client, "ana")
        bob, _ = _register(client, "bob")

        async def go():
            db = client.app.state.db
            await db.execute("UPDATE users SET model = ? WHERE id = ?",
                             ("ollama/anas-pick", ana))
            await db.commit()
            a = await get_crew(_Req(client.app), ana, "research")
            b = await get_crew(_Req(client.app), bob, "research")
            return a.model_name, b.model_name

        a, b = client.portal.call(go)
        assert a == "ollama/anas-pick"
        assert b == "ollama/env-default", "bob never chose anything"


class TestTheCache:
    def test_an_unchanged_choice_reuses_the_crew(self, client):
        """Building a crew probes the model, so it must not happen per request."""
        uid, _ = _register(client, "cached")

        async def go():
            r = _Req(client.app)
            first = await get_crew(r, uid, "research")
            second = await get_crew(r, uid, "research")
            return first is second

        assert client.portal.call(go) is True

    def test_a_changed_choice_rebuilds(self, client):
        uid, _ = _register(client, "switcher")

        async def go():
            db = client.app.state.db
            r = _Req(client.app)
            first = await get_crew(r, uid, "research")
            await db.execute("UPDATE users SET model = ? WHERE id = ?",
                             ("ollama/something-else", uid))
            await db.commit()
            second = await get_crew(r, uid, "research")
            return first is second, second.model_name

        same, name = client.portal.call(go)
        assert same is False
        assert name == "ollama/something-else"

    def test_saved_params_survive_a_rebuild(self, client):
        """A rebuilt crew starts from class defaults and would lose them."""
        uid, _ = _register(client, "tuned")

        async def go():
            db = client.app.state.db
            await db.execute(
                "UPDATE users SET model = ?, llm_params = ? WHERE id = ?",
                ("ollama/tuned", json.dumps({"temperature": 0.11}), uid))
            await db.commit()
            crew = await get_crew(_Req(client.app), uid, "research")
            return crew.params

        assert client.portal.call(go) == {"temperature": 0.11}

    def test_unreadable_params_do_not_break_the_crew(self, client):
        uid, _ = _register(client, "corrupt")

        async def go():
            db = client.app.state.db
            await db.execute(
                "UPDATE users SET model = ?, llm_params = ? WHERE id = ?",
                ("ollama/fine", "{not json", uid))
            await db.commit()
            crew = await get_crew(_Req(client.app), uid, "research")
            return crew.model_name, crew.params

        name, params = client.portal.call(go)
        assert name == "ollama/fine"
        assert params is None


class TestThePicker:
    def test_switching_saves_the_choice_and_drops_the_cache(self, client):
        uid, headers = _register(client, "picker")

        async def warm():
            return await get_crew(_Req(client.app), uid, "research")

        client.portal.call(warm)
        assert uid in client.app.state.research_crews

        r = client.post("/model", headers=headers, json={"model": "qwen3.5:9b"})
        assert r.status_code == 200
        assert r.json()["model"] == "ollama/qwen3.5:9b"
        assert uid not in client.app.state.research_crews, "cache not dropped"

        async def check():
            db = client.app.state.db
            cur = await db.execute("SELECT model FROM users WHERE id = ?", (uid,))
            row = await cur.fetchone()
            crew = await get_crew(_Req(client.app), uid, "research")
            return row[0], crew.model_name

        saved, built = client.portal.call(check)
        assert saved == "ollama/qwen3.5:9b"
        assert built == "ollama/qwen3.5:9b"

    def test_the_picker_accepts_an_already_prefixed_name(self, client):
        _, headers = _register(client, "prefixed")
        r = client.post("/model", headers=headers,
                        json={"model": "ollama/gemma4:12b"})
        assert r.json()["model"] == "ollama/gemma4:12b"

    def test_the_panel_distinguishes_yours_from_the_default(self, client):
        """Telling these apart is how this bug gets diagnosed next time."""
        uid, headers = _register(client, "panelist")
        client.post("/model", headers=headers, json={"model": "phi4:14b"})
        crew = client.get("/subsystem-params", headers=headers).json()["params"]["crew"]
        assert crew["your_model"] == "ollama/phi4:14b"
        assert crew["default_model"] == "ollama/env-default"


class TestADeletedEnvModel:
    """The trigger: .env naming a model that has since been removed."""

    def test_a_missing_default_is_replaced_with_an_installed_one(
            self, client, monkeypatch):
        installed = {"models": [{"name": "nomic-embed-text:latest"},
                                {"name": "qwen3.5:9b"}]}
        monkeypatch.setattr(client.app.state, "default_model",
                            "ollama/deleted-months-ago", raising=False)
        _fake_tags(monkeypatch, installed)
        chosen = client.portal.call(lambda: verify_default_model(client.app))
        assert chosen == "ollama/qwen3.5:9b"
        assert client.app.state.default_model == "ollama/qwen3.5:9b"

    def test_an_embedding_model_is_not_chosen_as_the_chat_default(
            self, client, monkeypatch):
        monkeypatch.setattr(client.app.state, "default_model",
                            "ollama/deleted", raising=False)
        _fake_tags(monkeypatch, {"models": [{"name": "nomic-embed-text:latest"},
                                            {"name": "phi4:14b"}]})
        assert client.portal.call(
            lambda: verify_default_model(client.app)) == "ollama/phi4:14b"

    def test_an_installed_default_is_left_alone(self, client, monkeypatch):
        monkeypatch.setattr(client.app.state, "default_model",
                            "ollama/qwen3.5:9b", raising=False)
        _fake_tags(monkeypatch, {"models": [{"name": "qwen3.5:9b"}]})
        assert client.portal.call(
            lambda: verify_default_model(client.app)) == "ollama/qwen3.5:9b"

    def test_ollama_being_unreachable_changes_nothing(self, client, monkeypatch):
        """Not an answer either way — do not rewrite config on a network blip."""
        monkeypatch.setattr(client.app.state, "default_model",
                            "ollama/configured", raising=False)
        _fake_tags(monkeypatch, None)
        assert client.portal.call(
            lambda: verify_default_model(client.app)) == "ollama/configured"
        assert client.app.state.default_model == "ollama/configured"


def _fake_tags(monkeypatch, payload):
    """Stub Ollama's /api/tags. *payload* None means unreachable."""
    import httpx

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            if payload is None:
                raise httpx.ConnectError("refused")
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)


class TestResolveModel:
    def test_it_returns_the_saved_params_alongside_the_model(self, client):
        uid, _ = _register(client, "resolver")

        async def go():
            db = client.app.state.db
            await db.execute(
                "UPDATE users SET model = ?, llm_params = ? WHERE id = ?",
                ("ollama/m", '{"temperature": 0.9}', uid))
            await db.commit()
            return await resolve_model(_Req(client.app), uid)

        model, params = client.portal.call(go)
        assert model == "ollama/m"
        assert json.loads(params) == {"temperature": 0.9}

    def test_an_unknown_user_gets_the_default(self, client):
        async def go():
            return await resolve_model(_Req(client.app), "no-such-user")

        model, params = client.portal.call(go)
        assert model == "ollama/env-default"
        assert params is None
