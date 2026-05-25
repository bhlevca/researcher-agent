# TODO_Claude.md — Improvement Plan for Researcher, Tutor & Composer

Generated: 2026-05-23 | Project: ai-agent v2.2.0 | Branch: developer

---

## Executive Summary

This is a well-built, feature-rich local AI platform. The architecture is sound and the separation between researcher, tutor, and composer is clean. The main risks are:

1. **~300 lines of monkey-patches** in `crew.py` that will silently break on any CrewAI/litellm update
2. **Single-user concurrency** — a semaphore serializes all crew runs
3. **Fragmented error handling** — no consistent format across the three agents
4. **Tutor and composer UIs** are incomplete relative to the backend capabilities

The sections below are ordered by priority within each component.

---

## RESEARCHER

### R1 — Monkey-Patch Isolation (HIGH)
**Files**: `src/researcher/crew.py` lines 36–330

The patches for CrewAI/litellm/Ollama compatibility are embedded directly at module import time with no version guards. A single `pip install --upgrade crewai` can break the whole app silently.

**Actions**:
- [ ] Move all monkey-patches into `src/researcher/patches.py`
- [ ] Add version guards: `assert crewai.__version__ == "1.14.5", "Patches untested on this version"`
- [ ] Add a comment block at the top of each patch group explaining *what* it fixes and *which upstream issue* it works around
- [ ] Create `tests/test_patches.py` that verifies each patch is still necessary (if the patched behavior is fixed upstream, the test fails and tells you to remove the patch)

### R2 — Multi-User Crew Isolation (HIGH)
**Files**: `src/researcher/main.py` line 65, `src/researcher/routes/chat.py` lines 155, 173, 208

Authentication is correctly implemented — each request is tied to an authenticated `user_id`. However, the *crew execution* state is stored as a single global on `app.state`:
- `app.state.crew_semaphore` — one Semaphore(1) blocks **all authenticated users**, not just the same user
- `app.state.research_crew` — a single crew instance; user B's request at line 155 overwrites user A's in-flight crew
- `app.state.cancel_event` — a single cancel flag; any user hitting `/chat/cancel` cancels every other user's active run

**Actions**:
- [ ] Replace `app.state.research_crew` with a `dict[user_id, ResearchCrew]` (lazy-init per user on first request)
- [ ] Replace `app.state.crew_semaphore` with a `dict[user_id, asyncio.Semaphore]` so only the *same user* with two tabs blocks themselves, not other users
- [ ] Replace `app.state.cancel_event` with a `dict[user_id, asyncio.Event]` so `/chat/cancel` only cancels the requesting user's run
- [ ] Apply the same isolation pattern to tutor and composer (they share the same global-state anti-pattern)
- [ ] Return HTTP 409 Conflict (not 429) if the same user already has an active crew run

### R3 — Request Timeouts & Circuit Breaker (HIGH)
**Files**: `src/researcher/routes/chat.py`, `src/researcher/routes/ask.py`

Long-running crew kickoffs can hang indefinitely. There is no timeout, no cancellation path surfaced to the client on errors, and no circuit breaker.

**Actions**:
- [ ] Wrap `crew.kickoff()` in `asyncio.wait_for(…, timeout=180)` (configurable via `CREW_TIMEOUT_SEC` env var)
- [ ] On timeout, cancel the task, yield a `data: [TIMEOUT]` SSE event, and clean up the cancel token
- [ ] The `/chat/cancel` endpoint already exists — ensure it actually cancels the underlying asyncio task (verify with a test)

### R4 — Consistent Error Responses (MEDIUM)
**Files**: `src/researcher/routes/chat.py`, `routes/ask.py`, `routes/sessions.py`, `routes/models.py`

Some routes return `{"detail": "..."}` (FastAPI default), others return `{"error": "..."}`, others raise bare exceptions.

**Actions**:
- [ ] Define a single `ErrorResponse` Pydantic model: `{"error": str, "code": str, "detail": str | None}`
- [ ] Add a global FastAPI exception handler in `main.py` that converts all unhandled exceptions to this format
- [ ] Update all `HTTPException` raises to use consistent `code` strings (e.g., `"not_found"`, `"rate_limited"`, `"crew_error"`)

### R5 — Structured Logging (MEDIUM)
**Files**: All modules (currently mix of `print()`, `logging.info()`, `logger.debug()`)

**Actions**:
- [ ] Add `structlog` or configure stdlib logging with a JSON formatter in `main.py`
- [ ] Define standard fields: `user_id`, `session_id`, `model`, `duration_ms`, `tool_calls`
- [ ] Replace all bare `print()` calls with proper logger calls
- [ ] Add request/response logging middleware (log method, path, status, duration)

### R6 — VRAM Spike After Image Generation (MEDIUM)
**Files**: `src/researcher/config.py` line 201

After image generation, Ollama is unloaded from VRAM to free space. This causes a ~10s reload penalty on the next message.

**Actions**:
- [ ] Add `KEEP_LLM_LOADED=true` env var option (for users with enough VRAM, e.g., 24GB+)
- [ ] If `KEEP_LLM_LOADED=false` (default), pre-warm Ollama in a background task after image gen completes so the reload happens while the user reads the image, not while they wait for the next response
- [ ] Add a `/memory` endpoint that reports current Ollama VRAM usage

### R7 — Input Sanitization (MEDIUM)
**Files**: `src/researcher/routes/chat.py`, `routes/ask.py`

User prompts are passed directly to the LLM and injected into the system prompt context without escaping.

**Actions**:
- [ ] Strip leading/trailing control characters and zero-width spaces from user input
- [ ] Enforce a max prompt length (e.g., 8192 chars) with a clear error message
- [ ] For file attachment context, truncate injected text at a configurable token budget (e.g., 6144 tokens)

### R8 — Static File Caching (LOW)
**Files**: `src/researcher/main.py` (StaticFiles mount)

No cache headers, no ETags, no compression. Browser re-downloads all assets on every page load.

**Actions**:
- [ ] Add `Cache-Control: max-age=3600` headers for static assets via a custom `StaticFiles` subclass or middleware
- [ ] Enable gzip compression for HTML/CSS/JS responses (FastAPI middleware or nginx)

---

## TUTOR

### T1 — Situational Dialogs / Roleplay Scenarios (HIGH)
**Files**: `src/researcher/tutor/crew.py`, `src/researcher/static/tutor.html`

- [x] Add `scenario` mode to `TutorCrew` task builder (in `tutor/crew.py`)
- [x] Create `tutor/config/scenarios.yaml` with starter situations
- [x] Add `/tutor/scenarios` endpoint (list) and `/tutor/chat` scenario mode
- [x] Add scenario picker to `tutor.html` UI (🎭 Scenarios tab)
- [x] Reasoning now collapsible instead of overlapping the reply

**Remaining — Scenario quality improvements**:
- [x] Tutor must conduct the entire dialog in the target language — strengthened prompt with NON-NEGOTIABLE language rule and first-person framing
- [x] Make scenarios more pedagogically driven — PEDAGOGY RULES added: always end with a question, correct errors briefly, prompt if student is silent
- [x] Save scenario transcript to `lesson_plans` table — `save_transcript: true` on each real turn saves via `on_complete` callback
- [x] Opening line auto-generated in target language — replaced static English `starter` with LLM-generated first turn (`__start__` trigger)
- [x] Fixed `liveBox` reference bug in `tutor-scenarios.js`

### T2 — Progress Visualization (HIGH) — DONE
**Files**: `src/researcher/routes/tutor.py` (`/tutor/sessions/{id}/stats`), `src/researcher/static/tutor.html`

- [x] Chart.js 4.4 loaded in `tutor.html`
- [x] Stats tab with mastery histogram and quiz score trend charts (`tutor-stats.js`)
- [x] Stats grid shows vocab count, avg mastery, quiz count, avg score

### T3 — Lesson Export (MEDIUM) — DONE
**Files**: `src/researcher/routes/tutor.py`, `src/researcher/static/js/tutor-lessons.js`

- [x] `GET /tutor/lessons/{id}/export?format=md|docx` endpoint
- [x] Markdown export — proper `.md` file from stored content
- [x] DOCX export — line-by-line markdown parsing into python-docx
- [x] Download buttons use `fetch` + blob (auth headers, no JSON download bug)

### T4 — Pronunciation Scoring (MEDIUM) — DONE
**Files**: `src/researcher/routes/speech.py`, `src/researcher/static/js/tutor-quiz.js`

- [x] `pronunciation` quiz type implemented end-to-end
- [x] WER utility in `speech.py` `_pronunciation_score()`
- [x] `/tutor/pronunciation/score` endpoint (Whisper transcribe + WER)
- [x] Quiz UI renders pronunciation questions with Play + Record buttons
- [x] Pre-computed scores bypass LLM grader in quiz submit

### T5 — CEFR Curriculum Structure (MEDIUM) — DONE
**Files**: `src/researcher/tutor/storage.py`, `src/researcher/static/tutor.html`

- [x] `curriculum.yaml` with per-CEFR-level topic sequences
- [x] `topic_index` column in `tutor_sessions`
- [x] `/tutor/curriculum-progress` and `/tutor/sessions/{id}/advance-topic` endpoints
- [x] Curriculum bar in UI: clickable topic name pre-fills lesson generator, Prev/Next navigation
- [x] `/tutor/sessions/{id}/prev-topic` endpoint added

### T6 — SM-2 Edge Cases (LOW)
**Files**: `src/researcher/tutor/storage.py`, `src/researcher/static/js/tutor-vocab.js`

- [x] `SRS_INITIAL_EASE` env var (default 2.5)
- [x] Lapse handling: grade < 3 resets interval to 1, decrements ease (floor 1.3)
- [x] `lapses` column in `vocabulary` table (with migration)
- [x] Words with `lapses >= 3` flagged as "difficult" (⚠ hard badge + red row tint) in vocab UI

---

## COMPOSER

### C1 — Modulation & Key Change Support (HIGH)
**Files**: `src/researcher/composer/musicxml_builder.py`, `src/researcher/composer/knowledge/theory.yaml`

The MusicXML builder handles a single key signature. Real compositions frequently modulate (change key mid-piece). The theory.yaml has no modulation tables.

**Actions**:
- [ ] Add `key_changes: list[{measure: int, fifths: int, mode: str}]` to the composition JSON schema
- [ ] Update `musicxml_builder.py` to emit `<key>` elements mid-score at the correct measure numbers
- [ ] Add modulation section to `theory.yaml`: common modulations per key (relative major/minor, dominant, subdominant), pivot chords
- [ ] Update `agents.yaml` composer rules: "You may modulate by specifying key_changes in the JSON"
- [ ] Add test cases in `test_musicxml_builder.py` for mid-score key changes

### C2 — Dynamics & Articulation in MusicXML (HIGH)
**Files**: `src/researcher/composer/musicxml_builder.py`

The builder has partial dynamics support (mentioned in file comments) but articulation marks (staccato, legato slurs, accents, fermata) are not emitted.

**Actions**:
- [ ] Add `articulation` field to note JSON schema: `"staccato" | "accent" | "tenuto" | "fermata" | null`
- [ ] Add `slur_start` / `slur_end` boolean fields to note JSON (for legato phrasing)
- [ ] Emit `<articulations>` and `<slur>` elements in `musicxml_builder.py`
- [ ] Update `agents.yaml` composer rules with articulation vocabulary
- [ ] Test: verify MuseScore can import and display articulations correctly

### C3 — Multi-Instrument / Orchestration Support (HIGH)
**Files**: `src/researcher/composer/musicxml_builder.py`, `src/researcher/composer/tools.py`

Currently the builder supports piano (grand staff) and single-instrument scores. Orchestration for multiple instruments is not implemented.

**Actions**:
- [ ] Extend JSON schema: `parts: [{instrument: str, clef: str, notes: [...]}]`
- [ ] Update `musicxml_builder.py` to emit `<part-list>` with multiple `<score-part>` elements
- [ ] The existing `instrument_range` tool already has range data — link it to part validation
- [ ] Add orchestration templates to `genres.yaml` (string quartet, jazz combo, brass quintet)
- [ ] Update `agents.yaml` rules: "For multi-instrument scores, output a parts array"

### C4 — Score Playback in Browser (MEDIUM)
**Files**: `src/researcher/static/composer.html`

MusicXML is downloadable but can only be played in external apps (MuseScore, etc.). In-browser playback would close the feedback loop dramatically.

**Actions**:
- [ ] Integrate [OSMD (OpenSheetMusicDisplay)](https://github.com/opensheetmusicdisplay/opensheetmusicdisplay) for browser MusicXML rendering
- [ ] Integrate [midi.js](https://github.com/mudcube/MIDI.js) or Web MIDI API for playback
- [ ] Add "View Score" button that opens rendered notation in a modal
- [ ] Add "Play" button that triggers MIDI playback in browser
- [ ] Keep "Download MusicXML" button for users who want MuseScore

### C5 — Robust JSON Repair Hardening (MEDIUM)
**Files**: `src/researcher/composer/musicxml_fix.py`

There are 19 regex + 8 structural repairs. This file is a patch on top of a patch — the LLM produces broken JSON and it gets repaired downstream. The correct fix is better prompting.

**Actions**:
- [ ] In `agents.yaml`, add explicit few-shot JSON output examples showing correct note format
- [ ] Add a JSON schema validation step *before* calling `musicxml_fix.py` — if the LLM output is already valid JSON, skip the 19 regex passes entirely (speed improvement)
- [ ] Log which repairs fire on each composition (helps identify which LLM failures are most common so prompting can target them)
- [ ] If repair fails completely, return a structured error `{"error": "json_repair_failed", "raw": "..."}` instead of crashing
- [ ] Add `test_musicxml_fix.py` cases for the top-5 most-fired repairs

### C6 — Harmonization Feedback UI (MEDIUM)
**Files**: `src/researcher/static/composer.html`, `src/researcher/routes/composer.py` (`/composer/harmonize`)

The harmonize endpoint exists and works, but the UI has no way to visually compare the original melody with the harmonized version side by side.

**Actions**:
- [ ] After harmonize response, render both original and harmonized MusicXML snippets in the UI (using OSMD from C4, or as text diff)
- [ ] Highlight changed notes / added chords in a different color
- [ ] Allow user to accept or reject individual harmonic suggestions

### C7 — Ear Training Mode (LOW — v3.0 prep)
**Files**: `src/researcher/composer/crew.py`, `src/researcher/static/composer.html`

The TODO mentions "Music Tutor" as v3.0. Ear training is the natural bridge between composer and tutor.

**Actions**:
- [ ] Add `ear_training` mode to `ComposerCrew`: generate interval/chord identification exercises
- [ ] Use Web Audio API (already present for mic) to play generated tones in browser
- [ ] Store results in `quiz_results` table (reuse tutor quiz infrastructure)
- [ ] This is a research spike — do a 2-day prototype before committing to full implementation

---

## CROSS-CUTTING

### X1 — Shared LLM Initialization Utility (MEDIUM)
**Files**: `crew.py`, `tutor/crew.py`, `composer/crew.py`

All three crew classes repeat the same LLM initialization, capability probing, and planning LLM creation code (~80 lines each).

**Actions**:
- [ ] Create `src/researcher/llm_factory.py` with `make_llm(model, params) -> LLM` and `make_planning_llm(model) -> LLM`
- [ ] Replace the three duplicated blocks with calls to this factory
- [ ] Move `probe_model_capabilities()` from `config.py` into `llm_factory.py` (it belongs there)

### X2 — Central Configuration (MEDIUM)
**Files**: Many (magic numbers scattered throughout)

Context window (32768), predict tokens (4096), image size (512), SM-2 ease (2.5), rate limit window (300s) are all hardcoded.

**Actions**:
- [ ] Add a `src/researcher/settings.py` using Pydantic `BaseSettings`
- [ ] Load all constants from env vars with documented defaults
- [ ] Replace all magic numbers with `settings.X` references
- [ ] Update `.env.example` with all new variables and descriptions

### X3 — Per-User Concurrency Model (MEDIUM)
**Files**: `main.py` (researcher semaphore), tutor routes, composer routes

Tutor and composer have no concurrency protection at all — two tabs can corrupt state. Researcher has a global semaphore.

**Actions**:
- [ ] Implement a shared `UserLockManager` in `main.py` (dict of `user_id → asyncio.Lock`)
- [ ] Apply it to researcher, tutor, and composer crew kickoffs consistently
- [ ] Return 409 Conflict if the same user tries to start two concurrent crew runs

### X4 — Database Migrations (LOW)
**Files**: `main.py` (SQLite init), `tutor/storage.py`, `composer/storage.py`

Tables are created with `CREATE TABLE IF NOT EXISTS`. If a column is added to an existing install, the old DB silently has the old schema.

**Actions**:
- [ ] Add [yoyo-migrations](https://ollycope.com/software/yoyo/latest/) or hand-rolled migration runner
- [ ] Write `migrations/001_initial.sql` (extract current CREATE TABLE statements)
- [ ] Any new column addition gets a `migrations/00N_add_X.sql`
- [ ] Run migrations on server startup in `main.py` lifespan

### X5 — Documentation (LOW)
**Files**: `README.md`, `docs/`

The README is good but there is no architecture diagram and no per-component API reference.

**Actions**:
- [ ] Add a Mermaid architecture diagram to README (shows researcher/tutor/composer → CrewAI → Ollama → tools)
- [ ] Auto-generate API docs are available at `/docs` (FastAPI) — add a note in README
- [ ] Add docstrings to the three `crew.py` classes and `musicxml_builder.py` (these are the hardest files to understand cold)

---

## Priority Order (Suggested Sequence)

| # | Item | Component | Effort | Impact |
|---|------|-----------|--------|--------|
| 1 | R2 — Per-user concurrency | Researcher | M | HIGH |
| 2 | R3 — Request timeouts | Researcher | S | HIGH |
| 3 | R1 — Monkey-patch isolation | Researcher | M | HIGH (maintenance) |
| 4 | X1 — Shared LLM factory | All | S | MEDIUM |
| 5 | X2 — Central config/settings | All | S | MEDIUM |
| 6 | T1 — Situational dialogs | Tutor | L | HIGH (UX) |
| 7 | T2 — Progress visualization | Tutor | M | HIGH (UX) |
| 8 | C1 — Modulation support | Composer | M | HIGH (quality) |
| 9 | C2 — Dynamics/articulation | Composer | S | HIGH (quality) |
| 10 | C3 — Multi-instrument | Composer | L | HIGH (feature) |
| 11 | R4 — Consistent error responses | Researcher | S | MEDIUM |
| 12 | T3 — Lesson export | Tutor | S | MEDIUM |
| 13 | C4 — Browser score playback | Composer | L | HIGH (UX) |
| 14 | T4 — Pronunciation scoring | Tutor | M | MEDIUM |
| 15 | T5 — CEFR curriculum | Tutor | M | MEDIUM |
| 16 | C5 — JSON repair hardening | Composer | S | MEDIUM |
| 17 | R5 — Structured logging | Researcher | S | LOW-MEDIUM |
| 18 | R6 — VRAM spike fix | Researcher | S | MEDIUM |
| 19 | X3 — Cross-component locking | All | S | MEDIUM |
| 20 | X4 — DB migrations | All | M | LOW (maintenance) |
| 21 | C6 — Harmonization feedback UI | Composer | M | LOW-MEDIUM |
| 22 | T6 — SM-2 edge cases | Tutor | S | LOW |
| 23 | C7 — Ear training (spike) | Composer | L | FUTURE |
| 24 | R7 — Input sanitization | Researcher | S | LOW (single-user) |
| 25 | R8 — Static file caching | Researcher | S | LOW |
| 26 | X5 — Documentation | All | M | LOW |

**Effort key**: S = < 1 day, M = 1–3 days, L = 3–7 days
