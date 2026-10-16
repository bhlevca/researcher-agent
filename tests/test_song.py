"""Song audio through ACE-Step, and the things that actually went wrong.

Every test here corresponds to a failure observed while building this, not to
a hypothetical. ACE-Step runs in its own process against its own torch, so
most of the risk is in the seam: the HTTP contract, the environment handed to
the subprocess, and who is allowed to stop it.

Measured on this machine, for the record:

    server start (--no-init)        4.0s, holding no VRAM
    first song (model load)         141s
    cover from a rendered score     42.7s
    VRAM during generation          ~9.8 GiB held
    VRAM after stop()               fully reclaimed

The chain the cover path depends on, also measured:

    ABC -> MusicXML 7923 chars -> MIDI 336 B -> fluidsynth WAV 3.9 MB
        -> ACE-Step cover -> MP3 353 KB, at the score's own tempo and key
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


class TestTheSubprocessEnvironment:
    """The cuDNN bug, pinned.

    ACE-Step runs torch 2.10+cu128; we run 2.13+cu130, and both virtualenvs
    ship their own nvidia/cudnn. Handing it our PYTHONPATH and VIRTUAL_ENV
    made it load a mix of the two, which surfaced as
    CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH inside a convolution — only on
    the cover path, because that is the one that runs convolutions over the
    source audio. Scrubbing those variables fixed it.
    """

    @pytest.mark.parametrize("var", [
        "PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME",
        "LD_LIBRARY_PATH", "LD_PRELOAD",
    ])
    def test_our_interpreter_and_library_paths_are_not_inherited(self, var):
        import inspect

        src = inspect.getsource(song.ensure_running)
        assert var in src, f"{var} must be named in the scrub list"
        # It appears inside the exclusion set, not as an assignment.
        assert f'env["{var}"]' not in src

    def test_the_server_output_never_goes_to_a_pipe(self):
        """A pipe nobody drains kills the download.

        Measured with stdout=PIPE and the parent gone: "Failed to download
        main model: Both HuggingFace and ModelScope downloads failed.
        HF: [Errno 32] Broken pipe". The download was killed by its own
        progress output.
        """
        import inspect

        src = inspect.getsource(song.ensure_running)
        assert "stdout=log" in src
        assert "stdout=subprocess.PIPE" not in src

    def test_it_is_started_in_its_own_session(self):
        """So a signal to our process group cannot take it down mid-song."""
        import inspect

        assert "start_new_session=True" in inspect.getsource(song.ensure_running)


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


class TestStoppingIsReliable:
    def test_an_orphan_is_matched_narrowly(self, monkeypatch, tmp_path):
        """uvicorn --reload orphans the server on every code change.

        Matching has to be narrow enough that it cannot reach anything else
        the person is running.
        """
        proc = tmp_path / "4242"
        proc.mkdir()
        (proc / "cmdline").write_bytes(
            b"python\0-m\0acestep.api_server\0--port\0"
            + str(song.ACESTEP_PORT).encode())
        other = tmp_path / "4243"
        other.mkdir()
        (other / "cmdline").write_bytes(b"python\0-m\0something.else\0")
        monkeypatch.setattr(song.pathlib, "Path", lambda p: tmp_path
                            if str(p) == "/proc" else __import__("pathlib").Path(p))
        pids = song._orphan_pids()
        assert 4242 in pids
        assert 4243 not in pids

    def test_stop_is_safe_when_nothing_runs(self, monkeypatch):
        monkeypatch.setattr(song, "_proc", None)
        monkeypatch.setattr(song, "_orphan_pids", lambda: [])
        song.stop()          # must not raise

    def test_idle_seconds_is_zero_when_stopped(self, monkeypatch):
        monkeypatch.setattr(song, "running", lambda: False)
        assert song.idle_seconds() == 0.0


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


class TestTheRouteHoldsTheGpu:
    """song.py's gpu-queue-exempt comments point here. This is the backing."""

    def test_the_song_route_reserves_the_card(self):
        import pathlib as _p

        src = (_p.Path(__file__).resolve().parents[1] / "src" / "researcher"
               / "routes" / "composer.py").read_text()
        body = src[src.index('@app.post("/composer/song")'):]
        body = body[:body.index('@app.post("/composer/song/stop")')]
        assert 'GPU.reserve(Need("song", "acestep")' in body, (
            "the song route must hold the GPU for the whole generation")
        assert "queue_message(" in body, "a queued song must say so"

    def test_the_broker_stops_the_engine_when_leaving_it(self):
        """There is no unload endpoint, so the process has to go."""
        import inspect

        from researcher.gpu_broker import GpuBroker

        src = inspect.getsource(GpuBroker._make_resident)
        assert 'current.kind == "song"' in src
        assert "_song.stop" in src


class TestTurboConfiguration:
    """v3.1.1. Everything before it ran outside the model's own spec.

    INFERENCE.md: guidance_scale is auto-corrected for turbo models but
    `shift` is **not** — the default 1.0 is applied as-is, and "the only
    timestep/guidance parameter you need to set manually for turbo is
    shift=3.0".
    """

    def test_turbo_is_recognised(self, monkeypatch):
        monkeypatch.setattr(song, "ACESTEP_MODEL", "acestep-v15-turbo")
        assert song._is_turbo()
        monkeypatch.setattr(song, "ACESTEP_MODEL", "acestep-v15-xl-turbo")
        assert song._is_turbo()

    def test_base_models_are_not_turbo(self, monkeypatch):
        monkeypatch.setattr(song, "ACESTEP_MODEL", "acestep-v15-base")
        assert not song._is_turbo()
        monkeypatch.setattr(song, "ACESTEP_MODEL", "acestep-v15-sft")
        assert not song._is_turbo()

    def test_shift_is_sent_for_turbo_and_only_for_turbo(self):
        """Read off the source: the request builder is inside generate()."""
        import inspect

        src = inspect.getsource(song.generate)
        assert 'fields["shift"] = str(TURBO_SHIFT)' in src
        # Keyed off the model actually chosen, not the configured default —
        # v3.1.3 made the engine model a per-request toggle.
        assert "if _is_turbo(want):" in src

    def test_the_documented_value_is_the_default(self):
        assert song.TURBO_SHIFT == 3.0


class TestInstrumentalSwitch:
    """The model has a switch; inferring it from empty lyrics was a guess."""

    def test_it_is_derived_from_the_lyrics_and_sent_explicitly(self):
        import inspect

        src = inspect.getsource(song.generate)
        assert 'instrumental = not (lyrics or "").strip()' in src
        assert '"instrumental": "true" if instrumental else "false"' in src
        # ACE-Step's documented convention for the lyrics field.
        assert '"[Instrumental]"' in src


class TestKeepingOnlyWhatWasSaved:
    """A render is scratch until saved. v3.1.1."""

    def _song(self, d, name, age_seconds=0.0):
        import os
        import time

        f = d / name
        f.write_bytes(b"ID3")
        if age_seconds:
            old = time.time() - age_seconds
            os.utime(f, (old, old))
        return f

    def test_an_unsaved_old_render_is_deleted(self, tmp_path):
        f = self._song(tmp_path, "song-aaa.mp3", age_seconds=10_000)
        removed = song.prune_unsaved(tmp_path, keep=set())
        assert removed == ["song-aaa.mp3"]
        assert not f.exists()

    def test_a_saved_render_is_kept_however_old(self, tmp_path):
        f = self._song(tmp_path, "song-bbb.mp3", age_seconds=10_000_000)
        removed = song.prune_unsaved(tmp_path, keep={"song-bbb.mp3"})
        assert removed == []
        assert f.exists()

    def test_the_keep_set_may_hold_urls_rather_than_names(self, tmp_path):
        """The database could store either; matching is on the file name."""
        f = self._song(tmp_path, "song-ccc.mp3", age_seconds=10_000)
        song.prune_unsaved(tmp_path, keep={"/static/generated/song-ccc.mp3"})
        assert f.exists()

    def test_a_fresh_unsaved_render_survives_the_grace_window(self, tmp_path):
        """You must be able to listen, think, and come back to save it."""
        f = self._song(tmp_path, "song-ddd.mp3", age_seconds=5)
        assert song.prune_unsaved(tmp_path, keep=set()) == []
        assert f.exists()

    def test_images_and_video_are_never_touched(self, tmp_path):
        """They are referenced from chat history this cannot read."""
        import os
        import time

        old = time.time() - 10_000_000
        for name in ("picture.png", "clip.mp4", "fe7a33d45136.mp4",
                     "notasong.mp3"):
            f = tmp_path / name
            f.write_bytes(b"x")
            os.utime(f, (old, old))
        song.prune_unsaved(tmp_path, keep=set())
        for name in ("picture.png", "clip.mp4", "fe7a33d45136.mp4",
                     "notasong.mp3"):
            assert (tmp_path / name).exists(), f"{name} must be left alone"

    def test_an_undeletable_file_does_not_abort_the_sweep(self, tmp_path, monkeypatch):
        self._song(tmp_path, "song-eee.mp3", age_seconds=10_000)
        good = self._song(tmp_path, "song-fff.mp3", age_seconds=10_000)
        real = pathlib.Path.unlink

        def _unlink(self, *a, **k):
            if self.name == "song-eee.mp3":
                raise OSError("busy")
            return real(self, *a, **k)

        monkeypatch.setattr(pathlib.Path, "unlink", _unlink)
        removed = song.prune_unsaved(tmp_path, keep=set())
        assert removed == ["song-fff.mp3"]
        assert not good.exists()

    def test_an_empty_directory_is_fine(self, tmp_path):
        assert song.prune_unsaved(tmp_path, keep=set()) == []


class TestTheSaveEndpoint:
    def test_it_refuses_anything_that_is_not_a_render(self):
        """The name is used to build a path, so it has to be constrained."""
        import pathlib as _p

        src = (_p.Path(__file__).resolve().parents[1] / "src" / "researcher"
               / "routes" / "composer.py").read_text()
        body = src[src.index('@app.post("/composer/song/save")'):]
        body = body[:body.index('@app.post("/composer/song/stop")')]
        assert 'name.startswith("song-")' in body
        assert "PurePath(audio_url).name" in body, (
            "take only the file name, never a caller-supplied path")
        assert "save_composition(" in body
        assert "audio_path=name" in body

    def test_the_route_prunes_after_a_render(self):
        import pathlib as _p

        src = (_p.Path(__file__).resolve().parents[1] / "src" / "researcher"
               / "routes" / "composer.py").read_text()
        assert "song.prune_unsaved" in src
        assert "saved_audio_paths(db)" in src

    def test_compositions_gained_an_audio_path_column(self):
        import pathlib as _p

        src = (_p.Path(__file__).resolve().parents[1] / "src" / "researcher"
               / "composer" / "storage.py").read_text()
        assert "audio_path      TEXT NOT NULL DEFAULT ''" in src, "fresh schema"
        assert "ALTER TABLE compositions ADD COLUMN audio_path" in src, (
            "an existing database must be migrated, not broken")


def _composer_route(name: str) -> str:
    """The body of one composer endpoint, for checking what it does."""
    src = (pathlib.Path(__file__).resolve().parents[1] / "src" / "researcher"
           / "routes" / "composer.py").read_text()
    start = src.index(f'@app.post("/composer/song{name}")'
                      if name else '@app.post("/composer/song")')
    rest = src[start + 10:]
    nxt = rest.index("    @app.")
    return rest[:nxt]


class TestWhereTheScoreComesFrom:
    """v3.1.2. The Song tab renders audio; it does not write notation.

    Leaving that implicit made it look like a composer: someone described a
    piece in the Song tab, pressed render, and silently got a description-only
    generation with "There is no score to go with it". The source is now an
    explicit choice, and a score written in Chat counts as much as one from the
    Score tab.
    """

    def test_a_saved_composition_can_be_performed(self):
        body = _composer_route("")
        assert "req.composition_id" in body
        assert "get_composition(db, req.composition_id, uid)" in body

    def test_a_saved_composition_is_used_verbatim(self):
        """Its MusicXML is already exact; re-extracting would be a worse copy."""
        body = _composer_route("")
        i = body.index("if req.composition_id:")
        j = body.index("elif req.response.strip():")
        assert 'comp.get("musicxml")' in body[i:j]
        assert "extract_musicxml" not in body[i:j]

    def test_has_score_follows_the_notation_not_the_render(self):
        """A score that exists but fails to render is still a score to save."""
        body = _composer_route("")
        assert '"has_score": bool(musicxml)' in body

    def test_the_source_is_named_back_to_the_caller(self):
        body = _composer_route("")
        assert '"source": src_label' in body

    def test_a_missing_composition_is_404_not_a_silent_fallback(self):
        body = _composer_route("")
        assert "That saved composition is gone" in body

    def test_the_client_offers_the_choice_explicitly(self):
        js = (SONG_JS := pathlib.Path(__file__).resolve().parents[1]
              / "src" / "researcher" / "static" / "js"
              / "composer-song.js").read_text()
        assert "refreshSongSources" in js
        assert "Description only — no score" in js
        assert "composition_id: compositionId" in js

    def test_a_score_written_in_chat_counts(self):
        """Only the Score tab used to record one, which was the whole bug."""
        js = (pathlib.Path(__file__).resolve().parents[1] / "src" / "researcher"
              / "static" / "js" / "composer-chat.js").read_text()
        assert "composerScoreRawResponse = text" in js
        assert "_hasScore(text" in js

    def test_both_tabs_refresh_the_source_list(self):
        root = (pathlib.Path(__file__).resolve().parents[1] / "src"
                / "researcher" / "static" / "js")
        for name in ("composer-chat.js", "composer-score.js"):
            assert "refreshSongSources" in (root / name).read_text(), name


class TestTheInstrumentalSwitchIsVisible:
    """It existed only as an inference from empty lyrics, with no control."""

    def test_the_panel_has_a_checkbox(self):
        html = (pathlib.Path(__file__).resolve().parents[1] / "src"
                / "researcher" / "static" / "composer.html").read_text()
        assert 'id="song-instrumental"' in html
        assert "Instrumental (no vocals)" in html

    def test_the_request_carries_it(self):
        models = (pathlib.Path(__file__).resolve().parents[1] / "src"
                  / "researcher" / "composer" / "models.py").read_text()
        assert "instrumental: bool = False" in models
        body = _composer_route("")
        assert 'lyrics="" if req.instrumental else req.lyrics' in body


class TestTheSavedNameIsEditable:
    def test_the_save_row_has_a_text_field_not_a_prompt(self):
        js = (pathlib.Path(__file__).resolve().parents[1] / "src" / "researcher"
              / "static" / "js" / "composer-song.js").read_text()
        assert "id = 'song-save-name'" in js
        assert "prompt('Name this song" not in js, "a JS prompt is not editable"

    def test_an_empty_name_is_refused(self):
        js = (pathlib.Path(__file__).resolve().parents[1] / "src" / "researcher"
              / "static" / "js" / "composer-song.js").read_text()
        assert "Give the song a name" in js

    def test_the_download_uses_the_title(self):
        """The file on disk is song-<hash>.mp3; the download is the title."""
        src = (pathlib.Path(__file__).resolve().parents[1] / "src" / "researcher"
               / "routes" / "composer.py").read_text()
        body = src[src.index('@app.get("/composer/compositions/{composition_id}/audio")'):]
        body = body[:body.index('@app.get("/composer/compositions/{composition_id}/musicxml")')]
        assert "content_disposition(f\"{comp['title']}.mp3\")" in body
        assert "audio/mpeg" in body

    def test_a_non_ascii_title_does_not_break_the_header(self):
        """An em dash in a filename crashed Starlette once already."""
        from researcher.downloads import content_disposition

        header = content_disposition("Study in C — trio.mp3")["Content-Disposition"]
        header.encode("latin-1")        # must not raise
        assert "filename*=UTF-8''" in header


SONG_JS_PATH = (pathlib.Path(__file__).resolve().parents[1] / "src" / "researcher"
                / "static" / "js" / "composer-song.js")
COMPOSER_HTML = (pathlib.Path(__file__).resolve().parents[1] / "src" / "researcher"
                 / "static" / "composer.html")


class TestTheEngineModelToggle:
    """v3.1.3. Only `base` can separate stems; turbo is quicker for everything
    else. The Model Zoo marks extract/lego/complete ❌ for turbo and sft.

    The model belongs to the engine *process* — one slot, chosen at startup —
    so switching restarts it. That is cheap precisely because the broker
    already starts and stops the engine on demand.
    """

    def test_both_models_are_offered_with_an_honest_difference(self):
        ids = {m["id"] for m in song.model_choices()}
        assert ids == {"acestep-v15-turbo", "acestep-v15-base"}
        by_id = {m["id"]: m for m in song.model_choices()}
        assert by_id["acestep-v15-base"]["stems"] is True
        assert by_id["acestep-v15-turbo"]["stems"] is False
        # The step counts are the real cost difference, so they are stated.
        assert by_id["acestep-v15-turbo"]["steps"] == 8
        assert by_id["acestep-v15-base"]["steps"] == 50

    @pytest.mark.parametrize("bad", [
        "", None, "acestep-v15-xl-base", "../../etc/passwd",
        "; rm -rf /", "acestep-v15-turbo-shift3",
    ])
    def test_only_allowlisted_names_get_through(self, bad):
        """The name becomes an env var for a subprocess and a download path."""
        assert song.resolve_model(bad) in song.MODELS

    def test_a_valid_choice_is_honoured(self):
        assert song.resolve_model("acestep-v15-base") == "acestep-v15-base"

    def test_switching_restarts_the_engine(self):
        import inspect

        src = inspect.getsource(song.ensure_running)
        assert "if _loaded_model in (\"\", want):" in src
        assert "stop()" in src, "a different model means a restart"

    def test_an_engine_we_did_not_start_is_left_alone(self):
        """Restarting someone else's work to satisfy a preference is rude."""
        import inspect

        src = inspect.getsource(song.ensure_running)
        assert "cannot ask" in src or "did not start" in src

    def test_stopping_forgets_the_model(self):
        import inspect

        assert "_loaded_model = \"\"" in inspect.getsource(song.stop)

    def test_loaded_model_is_empty_when_stopped(self, monkeypatch):
        monkeypatch.setattr(song, "running", lambda: False)
        monkeypatch.setattr(song, "_loaded_model", "acestep-v15-base")
        assert song.loaded_model() == ""

    def test_turbo_tuning_follows_the_chosen_model_not_the_default(self):
        """shift=3.0 is a turbo correction; base must not get it."""
        assert song._is_turbo("acestep-v15-turbo")
        assert not song._is_turbo("acestep-v15-base")
        import inspect

        assert "_is_turbo(want)" in inspect.getsource(song.generate)

    def test_the_panel_has_the_toggle(self):
        assert 'id="song-model"' in COMPOSER_HTML.read_text()

    def test_the_request_carries_the_choice(self):
        models = (pathlib.Path(__file__).resolve().parents[1] / "src"
                  / "researcher" / "composer" / "models.py").read_text()
        assert "model: str = \"\"" in models
        assert "model=req.model" in _composer_route("")

    def test_status_reports_what_is_loaded_and_what_is_available(self):
        src = (pathlib.Path(__file__).resolve().parents[1] / "src" / "researcher"
               / "routes" / "composer.py").read_text()
        assert '"loaded_model": song.loaded_model()' in src
        assert '"models": song.model_choices()' in src


class TestComposingInTheSongTab:
    """v3.1.3. Sending people to another tab to write notation, with the
    description already typed in this one, was ceremony. The Score tab keeps
    the fuller form — instrument picker, bar count, multi-minute assembly."""

    def test_it_composes_in_place_rather_than_switching_tabs(self):
        js = SONG_JS_PATH.read_text()
        assert "fetch('/composer/score'" in js
        assert "switchComposerTab('score')" not in js, "no more redirect"

    def test_the_result_becomes_the_song_source(self):
        js = SONG_JS_PATH.read_text()
        body = js[js.index("async function composeScoreForSong()"):]
        body = body[:body.index("function songComposeInstruments()")]
        assert "composerScoreRawResponse = result.response" in body
        assert "refreshSongSources()" in body
        assert "sel.value = 'latest'" in body

    def test_it_streams_and_shows_the_queue(self):
        """It runs the LLM, so it waits in the same queue as everything else."""
        js = SONG_JS_PATH.read_text()
        body = js[js.index("async function composeScoreForSong()"):]
        body = body[:body.index("function songComposeInstruments()")]
        assert "gpuShowQueueNote" in body
        assert "getReader()" in body

    def test_instruments_default_to_piano_when_left_blank(self):
        js = SONG_JS_PATH.read_text()
        assert "return ['Piano']" in js

    def test_the_inline_controls_exist(self):
        html = COMPOSER_HTML.read_text()
        assert 'id="song-instruments"' in html
        assert 'id="song-measures"' in html

    def test_an_empty_description_is_refused_before_the_model_runs(self):
        js = SONG_JS_PATH.read_text()
        body = js[js.index("async function composeScoreForSong()"):]
        assert "Describe what to compose first" in body
