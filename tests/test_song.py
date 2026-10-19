"""Song audio through ACE-Step, and the things that actually went wrong.

Every test corresponds to a failure observed while building or running this,
not a hypothetical. ACE-Step runs in its own process against its own torch, so
the risk sits in the seams: the HTTP contract, who owns the process, and what
is allowed to delete a file.

Measured on this machine (RTX 5070 Ti, the systemd service):

    idle, lazily loaded                0 MiB
    after any song                     ~10 GB held, never released on its own
    warm render, 1 take / 2 takes      9.1 s / 8.4 s — two cost the same as one
    switching engine via /v1/init      1.0-1.3 s to turbo, 9.5 s to base
    cold, after a restart              ~45-58 s
"""

import json
import os
import pathlib
from unittest.mock import MagicMock

import pytest

from researcher import song


class TestTheHttpContract:
    """Read from docs/en/API.md. Each of these was guessed wrong first."""

    def test_the_poll_field_is_task_id_list(self):
        """`task_ids` returns {"data": []} with HTTP 200 — polls for ever."""
        import inspect

        src = inspect.getsource(song._await_task)
        assert '"task_id_list"' in src
        assert '"task_ids"' not in src

    def test_result_is_a_json_string_and_gets_parsed(self):
        entry = {"task_id": "x", "status": 1,
                 "result": json.dumps([{"file": "/v1/audio?path=a.mp3",
                                        "metas": {"bpm": 96}}])}
        got = song._parse_result(entry)
        assert got["file"] == "/v1/audio?path=a.mp3"
        assert got["metas"]["bpm"] == 96

    def test_a_result_that_is_already_a_list_works(self):
        entry = {"result": [{"file": "a.mp3"}]}
        assert song._parse_result(entry)["file"] == "a.mp3"

    def test_a_result_that_is_already_a_dict_works(self):
        assert song._parse_result({"result": {"file": "a.mp3"}})["file"] == "a.mp3"

    def test_unreadable_result_says_so(self):
        with pytest.raises(RuntimeError, match="unreadable"):
            song._parse_result({"result": "{not json"})

    def test_an_empty_result_is_an_error_not_a_silent_pass(self):
        with pytest.raises(RuntimeError):
            song._parse_result({"result": "[]"})


class TestStatusCodes:
    @pytest.mark.parametrize("raw,want", [
        (0, 0), (1, 1), (2, 2),
        ("queued", 0), ("running", 0),
        ("succeeded", 1), ("failed", 2),
        ("SUCCESS", 1),
    ])
    def test_both_shapes_are_understood(self, raw, want):
        """Submit answers with a string, query answers with an int."""
        assert song._status_of({"status": raw}) == want

    def test_an_unknown_status_is_not_mistaken_for_success(self):
        assert song._status_of({"status": "wat"}) is None
        assert song._status_of({}) is None

    def test_a_bool_is_not_read_as_a_status(self):
        """True would otherwise compare equal to 1 and look like success."""
        assert song._status_of({"status": True}) is None


class TestErrorReporting:
    def test_the_real_error_is_dug_out_of_the_result(self):
        """The top level carries only status 2; the reason is nested.

        Reading only the top level produced "ACE-Step reported the generation
        failed" and nothing else, which hid a cuDNN mismatch for a whole run.
        """
        entry = {"task_id": "x", "status": 2, "result": json.dumps([{
            "status": 2, "stage": "failed",
            "error": "CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH"}])}
        assert "CUDNN" in song._error_of(entry)

    def test_a_top_level_error_still_wins(self):
        assert song._error_of({"error": "boom"}) == "boom"

    def test_no_error_anywhere_returns_empty_rather_than_raising(self):
        assert song._error_of({"status": 2}) == ""
        assert song._error_of({"status": 2, "result": "{bad"}) == ""


class TestProgressFromTheLog:
    """A silent hour is indistinguishable from a hang."""

    def test_a_download_line_becomes_a_readable_note(self, monkeypatch):
        monkeypatch.setattr(song, "log_tail", lambda n=40: (
            "Qwen3-Embedding-0.6B/model.safetensors:   3%|2  | "
            "35.1M/1.19G [00:27<10:36, 1.82MB/s]"))
        note = song.progress_note()
        assert "Downloading" in note
        assert "model.safetensors" in note
        assert "3%" in note and "35.1M/1.19G" in note

    def test_the_latest_file_wins(self, monkeypatch):
        monkeypatch.setattr(song, "log_tail", lambda n=40: (
            "first/a.safetensors: 100%| 1.0G/1.0G\n"
            "second/b.safetensors:  12%| 0.4G/3.7G\n"))
        assert "b.safetensors" in song.progress_note()

    def test_loading_is_distinguished_from_composing(self, monkeypatch):
        monkeypatch.setattr(song, "log_tail", lambda n=40: "Loading checkpoint")
        assert song.progress_note() == "Loading the song model"

    def test_an_unremarkable_log_falls_back(self, monkeypatch):
        monkeypatch.setattr(song, "log_tail", lambda n=40: "INFO: serving")
        assert song.progress_note() == "Composing the audio"

    def test_an_unreadable_log_does_not_raise(self, monkeypatch, tmp_path):
        monkeypatch.setattr(song, "LOG_PATH", tmp_path / "nope.log")
        assert song.log_tail() == "(no log)"
        assert song.progress_note() == "Composing the audio"


class TestFetchingTheAudio:
    def test_a_local_file_is_read_directly(self, tmp_path, monkeypatch):
        """ACE-Step runs on this machine, so the path is usually visible."""
        f = tmp_path / "out.mp3"
        f.write_bytes(b"ID3audio")
        assert song._fetch_audio({"path": str(f)}) == b"ID3audio"

    def test_a_server_reference_is_fetched_over_http(self, monkeypatch):
        calls = []

        class _R:
            ok = True
            content = b"mp3bytes"

        import requests
        monkeypatch.setattr(requests, "get",
                            lambda url, **k: (calls.append(url), _R())[1])
        got = song._fetch_audio({"file": "/v1/audio?path=x.mp3"})
        assert got == b"mp3bytes"
        assert calls[0].startswith(song.ACESTEP_BASE)

    def test_no_reference_at_all_is_an_error(self):
        with pytest.raises(RuntimeError, match="no audio"):
            song._fetch_audio({"file": "", "path": ""})


class TestInstallDetection:
    def test_a_missing_checkout_is_reported_not_crashed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(song, "ACESTEP_HOME", tmp_path / "nowhere")
        assert not song.is_installed()
        hint = song.install_hint()
        assert "git clone" in hint and "uv sync" in hint

    def test_a_checkout_without_a_venv_is_not_installed(self, monkeypatch, tmp_path):
        (tmp_path / "acestep").mkdir()
        monkeypatch.setattr(song, "ACESTEP_HOME", tmp_path)
        assert not song.is_installed()



# ---------------------------------------------------------------------------
# Who owns the engine. v3.2: Bogdan runs ACE-Step as a systemd user service,
# and the broker used to find it by command line and kill it — systemd then
# did not restart it, because a SIGTERM is a clean exit. Journal:
# "Shutting down" at 20:21:51 on 2026-10-17, and later songs ran on a stray
# copy this module started instead.
# ---------------------------------------------------------------------------

def _real_source(name: str) -> str:
    """A song.py function's source as written, bypassing any test stub."""
    import ast

    path = pathlib.Path(song.__file__)
    text = path.read_text()
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return "\n".join(text.splitlines()[node.lineno - 1:node.end_lineno])
    raise AssertionError(f"{name} not found in {path}")


def _argv(*parts: str) -> bytes:
    return b"\0".join(p.encode() for p in parts) + b"\0"


class TestTheProcessMatcher:
    """Parsed as an argument vector, never searched as one string."""

    def test_the_systemd_service_command_line_is_recognised(self):
        raw = _argv("/x/.venv/bin/python3", "-m", "uvicorn",
                    "acestep.api_server:app", "--host", "0.0.0.0",
                    "--port", str(song.ACESTEP_PORT))
        assert song._is_acestep_server(raw)

    def test_our_own_subprocess_is_recognised(self):
        raw = _argv("/x/.venv/bin/python", "-m", "acestep.api_server",
                    "--host", "127.0.0.1", "--port", str(song.ACESTEP_PORT),
                    "--no-init")
        assert song._is_acestep_server(raw)

    @pytest.mark.parametrize("raw", [
        # The substring test matched these, and would have killed them. The
        # first is the shape of the very shell that wrote the new matcher: its
        # heredoc mentioned both "acestep.api_server" and "--port 8001".
        _argv("/bin/bash", "-c", "cat > f <<'PY'\nacestep.api_server --port 8001\nPY"),
        _argv("grep", "-rn", "acestep.api_server", "8001"),
        _argv("vim", "acestep/api_server.py"),
        _argv("python3", "-c", "print('acestep.api_server 8001')"),
    ])
    def test_a_process_that_merely_mentions_it_is_not(self, raw):
        assert not song._is_acestep_server(raw)

    def test_another_port_is_not_ours(self):
        raw = _argv("python3", "-m", "acestep.api_server", "--port", "8002")
        assert not song._is_acestep_server(raw)

    def test_systemd_owned_pids_are_never_orphans(self):
        """Read off the source: the cgroup's PIDs are skipped before matching.

        From the file, not the live function — the conftest guard replaces
        _orphan_pids with a stub so no test can scan the real /proc.
        """
        src = _real_source("_orphan_pids")
        assert "owned = _systemd_pids()" in src
        assert "pid in owned" in src


class TestStoppingUnderSystemd:
    def test_it_restarts_the_unit_when_the_gpu_is_held(self, monkeypatch):
        calls = []
        monkeypatch.setattr(song, "_systemd_managed", lambda: True)
        monkeypatch.setattr(song, "holding_gpu", lambda: True)
        monkeypatch.setattr(song, "_systemctl", lambda *a: calls.append(a) or True)
        assert song.stop() is True
        assert calls == [("restart",)]

    def test_nothing_is_restarted_when_nothing_is_held(self, monkeypatch):
        """A restart then would only cost the next song a ~50s reload."""
        calls = []
        monkeypatch.setattr(song, "_systemd_managed", lambda: True)
        monkeypatch.setattr(song, "holding_gpu", lambda: False)
        monkeypatch.setattr(song, "_systemctl", lambda *a: calls.append(a) or True)
        assert song.stop() is False
        assert calls == []

    def test_it_never_kills_processes_when_systemd_owns_the_engine(self, monkeypatch):
        killed = []
        monkeypatch.setattr(song, "_systemd_managed", lambda: True)
        monkeypatch.setattr(song, "holding_gpu", lambda: True)
        monkeypatch.setattr(song, "_systemctl", lambda *a: True)
        monkeypatch.setattr(song, "_stop_orphans", lambda: killed.append(1) or True)
        song.stop()
        assert killed == []

    def test_without_systemd_it_is_safe_when_nothing_runs(self, monkeypatch):
        monkeypatch.setattr(song, "_proc", None)
        assert song.stop() is False

    def test_holding_gpu_reads_models_initialized(self, monkeypatch):
        """Running and holding are different: lazy loading holds nothing."""
        monkeypatch.setattr(song, "_health", lambda *a, **k: {"models_initialized": False})
        assert not song.holding_gpu()
        monkeypatch.setattr(song, "_health", lambda *a, **k: {"models_initialized": True})
        assert song.holding_gpu()
        monkeypatch.setattr(song, "_health", lambda *a, **k: None)
        assert not song.holding_gpu()


class TestTheTestsCannotReachTheRealEngine:
    """The conftest guard, pinned — it exists because a test run restarted
    the real service at 19:14:45 on 2026-10-18."""

    def test_systemctl_is_refused(self):
        with pytest.raises(AssertionError, match="real systemd unit"):
            song._systemctl("restart")

    def test_paths_point_into_a_temporary_folder(self, tmp_path):
        assert str(song.SONG_ROOT).startswith(str(tmp_path))
        assert str(song.AUDIT_PATH).startswith(str(tmp_path))


class TestTheSubprocessEnvironment:
    """For installs with no systemd unit. The environment and output
    handling each cost a failed run when they were wrong."""

    @pytest.mark.parametrize("var", [
        "PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD",
    ])
    def test_our_interpreter_and_library_paths_are_not_inherited(self, var):
        """A mix of two cuDNNs: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH."""
        import inspect

        src = inspect.getsource(song._start_subprocess)
        assert f'"{var}"' in src
        assert f'env["{var}"]' not in src

    def test_the_server_output_never_goes_to_a_pipe(self):
        """With stdout=PIPE the download died of its own progress output."""
        import inspect

        src = inspect.getsource(song._start_subprocess)
        assert "stdout=log" in src
        assert "stdout=subprocess.PIPE" not in src

    def test_it_is_started_in_its_own_session(self):
        import inspect

        assert "start_new_session=True" in inspect.getsource(song._start_subprocess)


# ---------------------------------------------------------------------------
# Switching engines, and what each request actually sends.
# ---------------------------------------------------------------------------

class TestEngines:
    def test_both_engines_are_offered_with_their_real_cost(self):
        by_id = {m["id"]: m for m in song.model_choices()}
        assert by_id["acestep-v15-turbo"]["steps"] == 8
        assert by_id["acestep-v15-base"]["steps"] == 50
        assert by_id["acestep-v15-base"]["guidance"] == 7.0
        assert by_id["acestep-v15-turbo"]["guidance"] is None

    @pytest.mark.parametrize("bad", [
        "", None, "acestep-v15-xl-base", "../../etc/passwd", "; rm -rf /",
    ])
    def test_only_allowlisted_names_get_through(self, bad):
        assert song.resolve_model(bad) in song.MODELS

    def test_switching_is_in_place_through_v1_init(self):
        """Measured 1.0-1.3s to turbo and 9.5s to base, against ~50s cold."""
        import inspect

        src = inspect.getsource(song.ensure_running)
        assert "_switch_model" in src
        assert "stop()" not in src, "switching must not restart the engine"
        assert '"/v1/init"' in inspect.getsource(song._switch_model) or \
            "/v1/init" in inspect.getsource(song._switch_model)

    def test_a_failed_switch_is_an_error_not_a_silent_wrong_model(self, monkeypatch):
        import requests

        class _R:
            ok = True
            status_code = 200
            text = "ok"

            def json(self):
                return {"data": {"loaded_model": "acestep-v15-turbo"}}

        monkeypatch.setattr(requests, "post", lambda *a, **k: _R())
        with pytest.raises(RuntimeError, match="could not switch"):
            song._switch_model("acestep-v15-base")


def _captured_fields(monkeypatch, **kwargs):
    """Run generate() against a fake engine and return what it sent."""
    import asyncio
    import requests

    sent = {}

    async def _no_engine(*a, **k):
        return None

    class _Submit:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": {"task_id": "t1"}}

    def _post(url, json=None, data=None, files=None, timeout=None):
        sent.update(json or data or {})
        return _Submit()

    async def _done(task_id, on_progress):
        return [{"file": "/a.mp3", "seed_value": "11,22", "metas": {"duration": 20}},
                {"file": "/b.mp3", "seed_value": "11,22", "metas": {"duration": 20}}]

    monkeypatch.setattr(song, "ensure_running", _no_engine)
    monkeypatch.setattr(song, "_await_task", _done)
    monkeypatch.setattr(song, "_fetch_audio", lambda res: b"ID3" + res["file"].encode())
    monkeypatch.setattr(requests, "post", _post)
    takes = asyncio.run(song.generate(prompt="a chanson", **kwargs))
    return sent, takes


class TestWhatARenderSends:
    def test_steps_are_sent_explicitly_for_each_engine(self, monkeypatch):
        """Left to the server default (8), "50 steps" ran at 8."""
        sent, _ = _captured_fields(monkeypatch, model="acestep-v15-turbo")
        assert sent["inference_steps"] == "8"
        assert "guidance_scale" not in sent
        sent, _ = _captured_fields(monkeypatch, model="acestep-v15-base")
        assert sent["inference_steps"] == "50"
        assert sent["guidance_scale"] == "7.0"

    def test_takes_become_batch_size_and_every_take_comes_back(self, monkeypatch):
        """The server made two anyway and this used to keep one."""
        sent, takes = _captured_fields(monkeypatch, takes=2)
        assert sent["batch_size"] == "2"
        assert [t["audio"] for t in takes] == [b"ID3/a.mp3", b"ID3/b.mp3"]

    def test_each_take_gets_its_own_seed(self, monkeypatch):
        """ACE-Step reports them as one "a,b" string per entry."""
        _, takes = _captured_fields(monkeypatch, takes=2)
        assert [t["seed"] for t in takes] == [11, 22]

    def test_takes_are_clamped(self, monkeypatch):
        sent, _ = _captured_fields(monkeypatch, takes=99)
        assert sent["batch_size"] == str(song.MAX_TAKES)
        sent, _ = _captured_fields(monkeypatch, takes=0)
        assert sent["batch_size"] == "1"

    def test_a_seed_reproduces_rather_than_randomising(self, monkeypatch):
        sent, _ = _captured_fields(monkeypatch, seed=424242)
        assert sent["use_random_seed"] == "false"
        assert sent["seed"] == "424242"
        sent, _ = _captured_fields(monkeypatch)
        assert "seed" not in sent

    def test_instrumental_uses_the_documented_convention(self, monkeypatch):
        sent, _ = _captured_fields(monkeypatch, lyrics="la la", instrumental=True)
        assert sent["instrumental"] == "true"
        assert sent["lyrics"] == "[Instrumental]"

    def test_time_signature_is_sent_as_the_beat_count(self, monkeypatch):
        """The session stores "3/4"; the API wants "3"."""
        sent, _ = _captured_fields(monkeypatch, time_signature="3/4")
        assert sent["time_signature"] == "3"

    def test_shift_is_explicit(self, monkeypatch):
        """Equal to the REST default today; explicit so a new default cannot
        silently change every song."""
        sent, _ = _captured_fields(monkeypatch)
        assert sent["shift"] == "3.0"


class TestTheVocalLanguage:
    """ACE-Step uses it for pronunciation and never reads the description."""

    def test_french_lyrics_are_detected(self):
        lyrics = ("[Verse]\nSous le ciel gris de Paris\nJe marche seule dans la nuit\n"
                  "[Chorus]\nLa vie passe comme un rêve\nEt mon cœur chante sans trêve")
        assert song.detect_language(lyrics) == "fr"

    def test_english_lyrics_are_detected(self):
        lyrics = ("[Verse]\nWalking through the empty streets\nThinking of your gentle touch\n"
                  "[Chorus]\nWe rise together into the light")
        assert song.detect_language(lyrics) == "en"

    def test_tags_alone_fall_back(self):
        assert song.detect_language("[Intro] [Verse] [Chorus] [Outro]") == "en"

    def test_too_little_text_falls_back_rather_than_guessing(self):
        assert song.detect_language("oui") == "en"
        assert song.detect_language("") == "en"

    def test_auto_is_resolved_before_sending(self, monkeypatch):
        sent, _ = _captured_fields(monkeypatch, vocal_language="auto",
                                   lyrics="Sous le ciel gris de Paris je marche seule dans la nuit")
        assert sent["vocal_language"] == "fr"

    def test_an_explicit_choice_wins(self, monkeypatch):
        sent, _ = _captured_fields(monkeypatch, vocal_language="ro",
                                   lyrics="Sous le ciel gris de Paris je marche seule dans la nuit")
        assert sent["vocal_language"] == "ro"


# ---------------------------------------------------------------------------
# Where songs live. Saved songs move out of scratch/, so the cleanup cannot
# reach them by construction — the shared folder plus a database lookup was
# defeated by test servers with their own databases.
# ---------------------------------------------------------------------------

class TestStorage:
    def test_a_render_goes_to_scratch(self):
        rel = song.write_scratch(b"ID3")
        assert rel.startswith("scratch/song-") and rel.endswith(".mp3")
        assert (song.SONG_ROOT / rel).read_bytes() == b"ID3"
        assert song.url_for(rel) == f"/songs/{rel}"

    def test_keeping_moves_it_out_of_scratch(self):
        rel = song.write_scratch(b"ID3")
        kept = song.keep(rel)
        assert kept.startswith("kept/")
        assert not (song.SONG_ROOT / rel).exists()
        assert (song.SONG_ROOT / kept).read_bytes() == b"ID3"

    def test_keeping_twice_is_harmless(self):
        kept = song.keep(song.write_scratch(b"ID3"))
        assert song.keep(kept) == kept

    def test_keeping_something_gone_says_so(self):
        with pytest.raises(FileNotFoundError):
            song.keep("scratch/song-000000000000.mp3")

    @pytest.mark.parametrize("bad", [
        "../../etc/passwd", "kept/../../x.mp3", "scratch/x.mp3", "other/song-0123456789ab.mp3",
        "/etc/passwd", "kept/song-0123456789ab.mp3/..", "",
    ])
    def test_paths_from_the_browser_are_refused_unless_ours(self, bad):
        with pytest.raises(ValueError):
            song.resolve(bad)

    def test_a_url_resolves_like_a_path(self):
        p = song.resolve("/songs/kept/song-0123456789ab.mp3")
        assert p == song.KEPT_DIR / "song-0123456789ab.mp3"

    def test_the_cleanup_never_looks_in_kept(self):
        import os
        import time

        kept = song.SONG_ROOT / song.keep(song.write_scratch(b"ID3"))
        old = time.time() - 10_000_000
        os.utime(kept, (old, old))
        assert song.prune_unsaved() == []
        assert kept.exists()

    def test_the_cleanup_records_what_it_deletes(self):
        import os
        import time

        f = song.SONG_ROOT / song.write_scratch(b"ID3")
        old = time.time() - 10_000
        os.utime(f, (old, old))
        assert song.prune_unsaved() == [f.name]
        assert f"prune file={f.name}" in song.AUDIT_PATH.read_text()


class TestTheMigration:
    """Songs saved before v3.2 sat in static/generated beside scratch files."""

    def _db(self, tmp_path, rows):
        import asyncio
        import aiosqlite

        db_path = tmp_path / "m.db"

        async def setup():
            db = await aiosqlite.connect(db_path)
            await db.execute("CREATE TABLE compositions (id TEXT, user_id TEXT, "
                             "title TEXT, musicxml TEXT DEFAULT '', audio_path TEXT)")
            for rid, path in rows:
                await db.execute("INSERT INTO compositions (id, user_id, title, audio_path) "
                                 "VALUES (?, 'u', 't', ?)", (rid, path))
            await db.commit()
            return db

        return asyncio.run(setup()), db_path

    def _run(self, db, legacy):
        import asyncio

        async def go():
            n = await song.migrate_legacy(db, legacy)
            cur = await db.execute("SELECT id, audio_path FROM compositions ORDER BY id")
            rows = await cur.fetchall()
            await db.close()
            return n, rows

        return asyncio.run(go())

    def test_a_saved_song_moves_and_its_rows_follow(self, tmp_path):
        legacy = tmp_path / "generated"
        legacy.mkdir()
        (legacy / "song-0123456789ab.mp3").write_bytes(b"ID3song")
        # Two rows on one file: the rename duplicates in the real database.
        db, _ = self._db(tmp_path, [("a", "song-0123456789ab.mp3"),
                                    ("b", "song-0123456789ab.mp3")])
        n, rows = self._run(db, legacy)
        assert n == 1
        assert rows == [("a", "kept/song-0123456789ab.mp3"),
                        ("b", "kept/song-0123456789ab.mp3")]
        assert (song.KEPT_DIR / "song-0123456789ab.mp3").read_bytes() == b"ID3song"
        assert not (legacy / "song-0123456789ab.mp3").exists()

    def test_a_missing_file_is_recorded_and_its_row_left_alone(self, tmp_path):
        legacy = tmp_path / "generated"
        legacy.mkdir()
        db, _ = self._db(tmp_path, [("a", "song-0123456789ab.mp3")])
        n, rows = self._run(db, legacy)
        assert n == 0
        assert rows == [("a", "song-0123456789ab.mp3")]
        assert "migrate-missing" in song.AUDIT_PATH.read_text()

    def test_it_is_idempotent(self, tmp_path):
        legacy = tmp_path / "generated"
        legacy.mkdir()
        db, _ = self._db(tmp_path, [("a", "kept/song-0123456789ab.mp3")])
        n, rows = self._run(db, legacy)
        assert n == 0
        assert rows == [("a", "kept/song-0123456789ab.mp3")]

    def test_unrelated_files_in_the_old_folder_are_untouched(self, tmp_path):
        """Images, video and unsaved songs there are not this code's to judge."""
        legacy = tmp_path / "generated"
        legacy.mkdir()
        for name in ("picture.png", "clip.mp4", "song-ffffffffffff.mp3"):
            (legacy / name).write_bytes(b"x")
        db, _ = self._db(tmp_path, [])
        self._run(db, legacy)
        for name in ("picture.png", "clip.mp4", "song-ffffffffffff.mp3"):
            assert (legacy / name).exists()


# ---------------------------------------------------------------------------
# The routes.
# ---------------------------------------------------------------------------

def _route(path: str, method: str = "post") -> str:
    src = (pathlib.Path(__file__).resolve().parents[1] / "src" / "researcher"
           / "routes" / "composer.py").read_text()
    start = src.index(f'@app.{method}("{path}")')
    rest = src[start + 10:]
    return rest[:rest.index("    @app.")] if "    @app." in rest else rest


class TestTheRoutes:
    def test_the_song_route_holds_the_gpu(self):
        body = _route("/composer/song")
        assert 'GPU.reserve(Need("song", "acestep")' in body
        assert "queue_message(" in body

    def test_it_is_text_to_audio_only(self):
        """Performing and composing scores here produced unrelated scores."""
        body = _route("/composer/song")
        for gone in ("composition_id", "musicxml_to_midi", "src_audio=", "cover_strength"):
            assert gone not in body, gone

    def test_every_take_is_returned(self):
        body = _route("/composer/song")
        assert '"takes": rendered' in body
        assert "song.write_scratch" in body

    def test_saving_is_an_upsert_on_the_audio(self):
        """Saving twice used to insert "french chanson , type cafe" twice."""
        body = _route("/composer/song/save")
        assert "find_composition_by_audio" in body
        assert "rename_composition" in body
        assert "song.keep(req.file)" in body

    def test_there_is_a_rename(self):
        assert "rename_composition" in _route("/composer/compositions/{composition_id}", "put")

    def test_delete_removes_audio_nobody_else_points_at(self):
        body = _route("/composer/compositions/{composition_id}", "delete")
        assert 'stored.startswith("kept/")' in body
        assert "saved_audio_paths(db)" in body

    def test_freeing_the_gpu_is_refused_mid_render(self):
        """Under systemd it is a restart, which would throw the render away."""
        assert 'running.get("label") == "song"' in _route("/composer/song/stop")

    def test_status_says_who_owns_the_engine_and_whether_it_holds_the_gpu(self):
        body = _route("/composer/song/status", "get")
        assert '"managed_by"' in body and '"holding_gpu"' in body

    def test_songs_are_served_from_their_own_folder(self):
        src = (pathlib.Path(__file__).resolve().parents[1] / "src" / "researcher"
               / "main.py").read_text()
        assert "app.mount(_song_storage.SONG_URL_PREFIX" in src
        assert "migrate_legacy" in src


class TestTheBroker:
    def test_it_asks_the_engine_when_it_has_no_record(self):
        """After an agent restart the engine may still hold ~10 GB."""
        import inspect

        from researcher.gpu_broker import GpuBroker

        src = inspect.getsource(GpuBroker._make_resident)
        assert "_song.holding_gpu" in src
        assert 'current.kind == "song"' in src


def test_the_apps_own_log_lines_reach_the_journal():
    """Nothing configured researcher.* loggers, so INFO was dropped."""
    import logging

    import researcher.main  # noqa: F401  (configures on import)

    log = logging.getLogger("researcher")
    assert log.handlers, "researcher.* has no handler"
    assert log.level <= logging.INFO


# ---------------------------------------------------------------------------
# The page.
# ---------------------------------------------------------------------------

STATIC = pathlib.Path(__file__).resolve().parents[1] / "src" / "researcher" / "static"


def _song_panel() -> str:
    html = (STATIC / "composer.html").read_text()
    start = html.index('id="cpanel-song"')
    return html[start:html.index("COMPOSITIONS panel", start)]


class TestTheSongPage:
    def test_no_score_controls_remain(self):
        panel = _song_panel()
        for gone in ("song-source", "song-compose", "song-instruments",
                     "song-measures", "song-strength", "Faithfulness", "Perform"):
            assert gone not in panel, gone

    def test_the_controls_that_remain(self):
        panel = _song_panel()
        for want in ('id="song-prompt"', 'id="song-lyrics"', 'id="song-instrumental"',
                     'id="song-language"', 'id="song-takes"', 'id="song-model"',
                     'id="song-seed"', 'value="auto"'):
            assert want in panel, want

    def test_the_lyrics_tags_are_explained(self):
        panel = _song_panel()
        assert "[Chorus - anthemic]" in panel
        assert "UPPERCASE" in panel

    def test_saving_offers_a_real_dialog_and_falls_back(self):
        js = (STATIC / "js" / "composer-song.js").read_text()
        assert "showSaveFilePicker" in js
        assert "a.download = suggestedName" in js
        assert "AbortError" in js, "cancelling the dialog must not save"

    def test_a_take_has_its_own_player(self):
        js = (STATIC / "js" / "composer-song.js").read_text()
        body = js[js.index("function buildTake("):]
        assert "document.createElement('audio')" in body
        assert "audio.src = take.url" in body

    def test_saving_also_keeps_it_out_of_the_cleanup(self):
        js = (STATIC / "js" / "composer-song.js").read_text()
        body = js[js.index("async function saveTake("):]
        assert "/composer/song/save" in body
        assert "file: take.file" in body

    def test_the_compositions_tab_plays_songs(self):
        js = (STATIC / "js" / "composer-score.js").read_text()
        body = js[js.index("async function loadComposition(id)"):]
        assert "data.audio_url" in body
        assert "document.createElement('audio')" in body
        # Delete used to sit inside the score branch.
        assert body.index("'🗑️ Delete'") > body.index("if (data.musicxml) {\n            const dlBtn")
