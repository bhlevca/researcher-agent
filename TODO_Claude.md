# TODO_Claude.md — Improvement Plan for Researcher, Tutor & Composer

Generated: 2026-05-23 | Project: ai-agent v2.2.0 | Branch: developer
Updated: 2026-08-10 — statuses below re-verified against the code, not against the
previous checkboxes. Several entries were stale in both directions.

---

## Status at a glance (verified 2026-08-10)

| ID | Item | Status |
|----|------|--------|
| R1 | Monkey-patch isolation | ✅ done (v2.4.0) |
| R2 | Multi-user crew isolation | ✅ done — except 409 Conflict |
| R3 | Request timeouts & cancel | ⚠️ timeout works; **cancel does not actually cancel** |
| R4 | Consistent error responses | ❌ not started |
| R5 | Structured logging | ❌ not started — 94 `stderr.write` vs 82 logger calls |
| R6 | VRAM spike after image gen | ❌ not started |
| R7 | Input sanitization | ⚠️ attachments capped; prompts unvalidated |
| R8 | Static file caching | ❌ not started |
| T1–T6 | Tutor feature set | ✅ done (v2.3.0) |
| T-A | FSRS replaces SM-2 | ✅ done (v2.6.0-dev) |
| T-B | Comprehensible input | ✅ done (v2.6.0) |
| T-C | Feedback loop / phoneme scoring | ❌ not started |
| T-D | TTS first-reply language bug | ❌ open |
| C1 | Modulation / key changes | ❌ not started |
| C2 | Dynamics & articulation | ❌ not started |
| C3 | Multi-instrument | ✅ done — entry was stale |
| C4 | Browser score playback | ❌ not started — gates all composer work |
| C5 | JSON repair hardening | ⚠️ partial — likely deleted by the ABC switch |
| C6 | Harmonization feedback UI | ❌ endpoint exists, no comparison view |
| C7 | Ear training | ❌ not started |
| X1 | Shared LLM factory | ❌ not started — duplication has grown since |
| X2 | Central settings | ❌ not started — `pydantic-settings` installed but unused |
| X3 | Per-user concurrency | ✅ done — entry was stale |
| X4 | DB migrations | ❌ hand-rolled `PRAGMA` checks |
| X5 | Documentation / diagram | ❌ no architecture diagram |

**Delivered since this file was written, and not in the list above:** faster-whisper STT,
PDF/image OCR with automatic DOCX/ODT export, the extraction fast path, the CrewAI executor
pin, and `max_iter` as a tunable parameter. See the session notes below.

---

## Session notes — 2026-08-09 (v2.4.0)

### What broke, and why (post-mortem)

A `uv lock --upgrade` moved the venv off a coherent dependency set and took the whole app
down. Root causes, in the order they bite:

| Symptom | Actual cause |
|---|---|
| App would not import at all | `crewai` 1.14.5 → 1.15.14 tripped an exact-equality `assert` in `patches.py` |
| Dialog mode ran to the 600 s timeout | CrewAI 1.15.14 changed Agent's **default executor** from `CrewAgentExecutor` to the Flow-based `crewai.experimental.AgentExecutor` |
| P3 context-length patch silently dead | Same executor change — the patched `_invoke_loop_*` methods still exist but are never called |
| Every image backend dead | `tokenizers` 0.23.1 vs `transformers` 5.8.1 (requires ≤ 0.23.0) |
| Whisper dead | `numpy` 2.5 vs `numba` (requires ≤ 2.4) |
| Voice dead *before* the upgrade | `speech.py` used `os.getenv` without `import os` (introduced in `c826f3e`) |

**Lesson for the next upgrade.** The structural guard in `patches.py` checks that patched
symbols *exist*. That is necessary but not sufficient — the executor swap kept every symbol
alive while removing them from the execution path. When upgrading CrewAI, check
**reachability** too: which executor `Agent` actually instantiates, and whether the patched
methods are on it. `patches._check_executor_reachability()` now logs this at startup.

### Deferred — pick these up next

- [ ] **TTS reads the first reply in the wrong language.** Observed in the tutor: the opening
      line `"Bonjour ! Ça va très bien, merci ! Et toi ? Je m'appelle [Assistant], je suis ton
      tuteur de français."` was spoken with an English voice, mangling the French. Every
      sentence *after* the first was detected correctly. Suspect the language sniff in
      `tts.py` runs before the target language is known, or `langdetect` is being fed too
      short a sample and losing to the bracketed `[Assistant]` placeholder. Fix: pass the
      session's target language explicitly to `/tts/speak` instead of sniffing, and fall back
      to detection only when no target language is set.
- [ ] **Port P3 to the new executor, then unpin.** Agents currently pin
      `executor_class=CrewAgentExecutor` via `patches.executor_kwargs()` because all five
      patches target it, and it is what the app was tested against. `CrewAgentExecutor` is
      deprecated upstream and will be removed. Port the context-length handling to
      `crewai.experimental.AgentExecutor`, verify with `CREWAI_EXECUTOR=experimental`, then
      drop the pin.
- [ ] **`tests/test_titi.py` hits a live Ollama model at collection time**, so it fails the
      whole run when `llama3.1:8b` is absent. All other suites are excluded from this problem.
      Mark it `@pytest.mark.integration` and move the model call inside the test body.
- [ ] Pytest exits 0 but aborts at interpreter teardown (`terminate called without an active
      exception`) once the app is imported. Cosmetic, pre-existing, not STT-related.

### Corrections to the older sections below

- **R2 (per-user concurrency) is DONE** — `main.py` holds per-user dicts for crews,
  semaphores and cancel events.
- **C3 (multi-instrument) is DONE** — `musicxml_builder.py` already emits a full `part-list`
  with per-part `score-part` and `midi-instrument`.
- **R1 (monkey-patch isolation) is DONE** — patches live in `patches.py` with tests.
- **R3 is still open and still real**: `/chat/cancel` sets an event that `_run_crew` never
  reads, so the worker thread runs to completion holding the GPU, and the early-exit `return`
  releases the semaphore while that thread is still alive.

---

## COMPOSER — why the music is weak, and what would actually fix it

*Revised 2026-08-09 after a literature scan. The headline finding below supersedes the
music21-first plan I originally wrote: the **representation** is the primary problem, and
fixing it is cheaper than anything else here.*

The composer is *functional* but musically naive. This is not a bug in
`musicxml_builder.py` (which is fine) and not a prompting problem that more instructions will
solve. It is structural, and there are two compounding causes.

### Cause 1 — we picked the worst possible format for an LLM (fix this first)

We ask the model to emit **note-by-note JSON**. Research on symbolic music generation is
blunt about this: LLMs are *inherently* more compatible with **ABC notation** than with
MIDI-style event lists, because ABC is text that aligns with how language models actually
work. Note-by-note JSON is arguably worse than MIDI — it is the most verbose option and
carries no musical structure at all.

The measured consequences of ABC:

- ~288 tokens per song, roughly **38% of the token count** of MIDI-style representations.
  Shorter sequences mean the model can hold a whole piece in context instead of losing the
  thread after four bars.
- ABC **intrinsically encodes repetition and structure** (repeats, sections, phrase marks).
  Structure is carried by the notation rather than something the model must remember.
- MuPT, trained on ABC, produced music that listeners **preferred over GPT-4**, with high
  intra-texture similarity and repetition scores.

This explains the "kindergarten" quality directly: a general model emitting one JSON note at a
time cannot hear bar 4 by the time it writes bar 12, so it produces locally plausible,
globally incoherent sequences — no motivic development, no phrase structure, no harmonic
direction.

**Action:**
- [ ] Switch the composer's output format from JSON to **ABC notation**.
- [ ] Parse it with `music21` (`converter.parse(abc_string)`), which reads ABC natively and
      converts to MusicXML — so `musicxml_builder.py` keeps working downstream.
- [ ] This **deletes most of `musicxml_fix.py` (506 lines) and `_repair_json` in
      `composer/crew.py`**. Those exist only because JSON from an LLM arrives broken; ABC is
      far more forgiving and music21 validates it properly.
- [ ] Keep grammar-constrained decoding as a backstop where useful, but the format change
      does most of the work.

### Cause 2 — a general chat model has no musical training

Even in ABC, `mistral-small` has no idea what a good melody is. Two specialist models run
locally and both speak ABC natively:

| Model | Size | Notes |
|---|---|---|
| **ChatMusician** (`m-a-p/ChatMusician`) | LLaMA2-based | Continually pre-trained on ABC. Composes well-structured, full-length music conditioned on **text, chords, melodies, motifs and musical forms**. Pure text tokenizer — no multimodal stack, so it can run under Ollama/llama.cpp like any other model. |
| **NotaGen** (`ElectricAlexis/NotaGen`) | **516M params** | Pre-trained on 1.6M ABC sheets. Conditioned on **period, composer and instrumentation**. Tiny — fits alongside Ollama and the image pipelines on the 16 GB card with room to spare. |

Note the symmetry with what we just did for OCR: a small specialist model beat the big general
one decisively. `deepseek-ocr:3b` was 5× faster and produced 4× more text than `qwen3-vl:8b`.
The same pattern very likely holds here.

**Action:**
- [ ] Add `COMPOSER_MODEL`, independent of `MODEL`, exactly as `OCR_MODEL` is today.
- [ ] Evaluate NotaGen first — 516M is cheap to try and it is purpose-built for notation.

### Still worth doing, after the above

1. **`music21`** for what code does better than any model: voice-leading and range validation,
   realising Roman-numeral progressions into voiced chords, and motivic development
   (transposition, inversion, retrograde, augmentation) — one-liners that are exactly what
   makes music sound *composed* rather than sampled.
2. **Two-stage generation.** LLM returns a plan (form, key plan, a 2-bar motif, harmonic
   rhythm); code expands it. The model never emits a long note list.
3. **In-browser playback (OSMD + Tone.js).** Still the gate on everything: until a change can
   be heard in two seconds instead of a MuseScore round-trip, tuning musical quality is
   impractical.
4. Missing MusicXML expression: dynamics, articulations, slurs, and mid-score key changes
   (`key_changes`) — see C1/C2 below.

**Sources:** [MuPT](https://arxiv.org/abs/2404.06393) ·
[ChatMusician](https://huggingface.co/papers/2402.16153) ·
[NotaGen](https://www.ijcai.org/proceedings/2025/1134.pdf) ·
[MMM](https://arxiv.org/pdf/2008.06048)

---

## TUTOR — competitive scan and what to steal

*Added 2026-08-09. The tutor is feature-rich but built almost entirely around **output**
(quizzes, vocabulary drills, scenarios). The research consensus is that this is the smaller
half of language acquisition.*

### T-A — Replace SM-2 with FSRS — ✅ DONE (v2.6.0-dev)

`tutor/storage.py` implements SM-2, the 1987 algorithm with a single ease factor per card.
Anki replaced it with **FSRS**, which models three variables per card — difficulty, stability
and retrievability — fitted to **700M+ real reviews**.

- **~20–30% fewer reviews for the same retention.**
- More accurate recall prediction in **~99.6%** of collections.
- Free, open-source, available as a Python library from the Open Spaced Repetition project.

This is the highest value-per-effort item in the tutor: a drop-in scheduler swap behind the
existing `vocabulary` table. The `lapses` column added in T6 is already one of the inputs FSRS
wants.

- [x] Added `fsrs` 6.3.2 (MIT); `vocabulary` gained `fsrs_stability`, `fsrs_difficulty`,
      `fsrs_state`, `fsrs_step`, `last_review`
- [x] SM-2 kept behind `SRS_ALGO=sm2`; `schedule_review()` dispatches
- [x] Existing cards are seeded from their SM-2 history rather than reset —
      `stability ← interval_days`, `difficulty ←` inverted ease factor, nudged up by lapses
- [x] Added `SRS_MAX_INTERVAL` (default 365). FSRS's own default is 36500 days, which is
      meaningless for language study.
- [x] 9 tests in `tests/test_tutor.py::TestFsrsScheduler`

**Bug found and fixed during implementation** — worth remembering, because it is silent:
seeding a migrated card with `last_review = NULL` makes FSRS compute zero elapsed time, so a
long-interval card looks perfectly retained. A 90-day card rated *Easy* was scheduled
**4994 days** out. Setting `last_review` to one interval ago (which is simply true) gives 447
days. Pinned by `test_migration_does_not_explode_the_interval`.

*Not yet verifiable:* the ~20–30% review reduction is a property of scheduling accuracy over
weeks of real reviews. The tests prove correctness and continuity, not that benefit.

### T-B — Add comprehensible input — ✅ DONE (v2.6.0)

Current design is ~100% output and review. The research points the other way:

- Learners spending **80% of study time on comprehensible input** (real content at their level
  plus ~10% unknown words — Krashen's *i+1*) and 20% on active review progress significantly
  faster.
- Authentic content + SRS gives **3.2× higher retention** than drill-based apps, and reaches
  intermediate comprehension **40% faster**.
- App study combined with authentic media improved **listening comprehension 47% faster** than
  apps alone.

We already have every piece needed to build this and have not connected them:

- [x] **Levelled reading passages** — new 📚 Reading tab. `passage_task` in `tasks.yaml`
      explicitly forbids grammar tables and exercises: this is prose to be read, not a lesson.
      Defaults to the session's current curriculum topic when none is given.
- [x] **Seeded with due vocabulary** — `list_vocabulary_due()` feeds the prompt, which
      instructs the model to use each word naturally and *not* draw attention to it.
- [x] **Read aloud** via the existing edge-tts. `_passageProseOnly()` strips the glossary and
      comprehension questions first, so only the prose is spoken.
- [x] **Unknown-word ratio** computed server-side per passage, with an i+1 verdict in the UI.
- [x] Endpoints: `POST /tutor/passages` (SSE), `/tutor/passages/save`, list / get / delete;
      new `passages` table.

**Two corrections made during implementation — both worth remembering:**

1. *The first ratio was useless.* It reported **95.7% unknown** on a clean A1 passage because
   `le`, `la`, `il`, `est`, `de` counted as unknown vocabulary. Function-word stop lists for
   French, Spanish, German, Italian, Portuguese, English and Romanian now exclude them.
2. *The ratio is measured against **saved** vocabulary*, which is only a proxy for what the
   learner knows. Below 50 saved words the UI now says "save more vocabulary to gauge
   difficulty (n/50)" rather than always declaring a passage "hard". Expect the metric to look
   unhelpful on day one and to become meaningful as the word list grows.

**Known limitations:** proper nouns count as new words (*Léo*), and there is no lemmatisation,
so *dort* and *dormir* are counted separately. Both would need a per-language NLP dependency
for modest gain.

### T-C — Tighten the conversation feedback loop (MEDIUM)

Scenario roleplay (T1) is the right shape; the market leaders differ in the *feedback*, not
the roleplay:

- Praktika's core loop is explicitly **speak → get feedback → improve**, with user-selectable
  strictness: **soft / balanced / strict**.
- Leaders **diagnose pronunciation at phoneme level** and adapt to the learner's actual
  errors rather than a preset script.
- Persistent **contextual memory** across sessions, so the tutor remembers earlier mistakes.

Against that, ours corrects errors inline but has no strictness control, no error history and
only word-level scoring:

- [ ] Add a **feedback strictness** setting to scenario and chat modes.
- [ ] Upgrade pronunciation scoring from WER to **phoneme-level**. Current `_wer()` scores
      *word recognition*, so a heavy accent Whisper still decodes scores 1.0 — see T3 note
      above. Phoneme alignment gives genuinely actionable feedback ("your *u* is fronted").
- [ ] Persist a **per-learner error profile** (recurring grammar and phoneme errors) and feed
      it into scenario and quiz generation, so practice targets real weak spots.

### T-D — Deferred bug (carried from the v2.4.0 notes)

- [ ] TTS speaks the **first** reply of a session in the wrong language — see the deferred
      item in the session notes above.

**Sources:** [FSRS vs SM-2](https://www.antiagent.io/blog/fsrs-vs-sm-2) ·
[Anki scheduler FAQ](https://faqs.ankiweb.net/what-spaced-repetition-algorithm) ·
[Research-backed app comparison](https://www.sciencebasedlearning.com/blog/best-language-learning-apps-2026) ·
[Praktika review](https://languatalk.com/blog/praktika-review/) ·
[AI speaking apps compared](https://www.talkio.ai/blog/best-ai-language-speaking-practice-apps-in-2026)

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

### R1 — Monkey-Patch Isolation (HIGH) — ✅ DONE (v2.4.0)

**Actions**:
- [x] Move all monkey-patches into `src/researcher/patches.py`
- [x] Add version guards — implemented as a *range* plus a structural check, not an
      exact-equality assert. The original suggestion here was actively harmful: an exact
      `==` assert turned a routine `uv lock --upgrade` into a hard startup crash. See the
      v2.4.0 session notes.
- [x] Comment block per patch group explaining what it fixes
- [x] `tests/test_patches.py` verifies each patch is still necessary
- [x] **Beyond the original plan**: `_check_executor_reachability()` also catches patches
      that are still *present* but no longer *called* — which is exactly how P3 died silently.

### R2 — Multi-User Crew Isolation (HIGH)
**Files**: `src/researcher/main.py` line 65, `src/researcher/routes/chat.py` lines 155, 173, 208

Authentication is correctly implemented — each request is tied to an authenticated `user_id`. However, the *crew execution* state is stored as a single global on `app.state`:
- `app.state.crew_semaphore` — one Semaphore(1) blocks **all authenticated users**, not just the same user
- `app.state.research_crew` — a single crew instance; user B's request at line 155 overwrites user A's in-flight crew
- `app.state.cancel_event` — a single cancel flag; any user hitting `/chat/cancel` cancels every other user's active run

**Status**: ✅ mostly DONE — one sub-item outstanding.

**Actions**:
- [x] `app.state.research_crews` is a `dict[user_id, ResearchCrew]`
- [x] `app.state.crew_semaphores` is a `dict[user_id, asyncio.Semaphore]`
- [x] `app.state.cancel_events` is a `dict[user_id, threading.Event]`
- [x] Tutor and composer use the same per-user semaphores (see X3)
- [ ] Return HTTP 409 Conflict if the same user already has an active crew run —
      currently the second request simply blocks on the semaphore instead of being rejected

### R3 — Request Timeouts & Circuit Breaker (HIGH)
**Files**: `src/researcher/routes/chat.py`, `src/researcher/routes/ask.py`

Long-running crew kickoffs can hang indefinitely. There is no timeout, no cancellation path surfaced to the client on errors, and no circuit breaker.

**Status**: ⚠️ PARTIAL — the timeout fires, but nothing is actually cancelled.

**Actions**:
- [x] Timeout implemented, configurable via `CREW_TIMEOUT_SEC` and the per-user
      `crew_timeout_sec` LLM param
- [x] On timeout the stream yields an `error` SSE event and stops relaying
- [ ] **Cancel does not cancel.** `/chat/cancel` sets the event and the SSE loop checks it,
      but `_run_crew` never reads it — the worker thread runs to completion, still holding
      the GPU. The user sees the request stop; the machine does not.
- [ ] **The semaphore is released early.** The early-exit `return` leaves the
      `async with semaphore` block while the worker thread is still alive, so a second
      request can start a concurrent crew on a single GPU.
- [ ] Verify both with a test

*Practical note:* `max_iter` (added v2.4.0, tunable from the params panel) is the real
lever on worst-case latency — each iteration is a full LLM call.

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

**Status**: ⚠️ PARTIAL.

**Actions**:
- [ ] Strip leading/trailing control characters and zero-width spaces from user input
- [ ] Enforce a max prompt length with a clear error message
- [x] Attachment text is capped — `MAX_TEXT_CHARS = 50_000` in `ingestion.py`, applied at
      extraction time with a visible truncation marker. Still a character budget rather than
      a token budget, and 50k chars is generous against a 32k-token context.

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

**Status**: ✅ largely DONE — this entry was stale.

**Actions**:
- [x] JSON schema has a `parts` array
- [x] `musicxml_builder.py` emits `<part-list>` with per-part `<score-part>` and
      `<midi-instrument>`; also handles grand staff vs single staff and merges split parts
- [ ] Link the `instrument_range` tool to part validation (music21 would do this better)
- [ ] Orchestration templates in `genres.yaml` (string quartet, jazz combo, brass quintet)
- [x] `agents.yaml` documents the parts array

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

**Status**: ⚠️ PARTIAL — and **likely to be deleted rather than finished**. Switching the
composer's output format to ABC notation removes the reason this layer exists. Do not invest
here before reading the revised COMPOSER section near the top of this file.

**Actions**:
- [ ] Few-shot JSON examples in `agents.yaml`
- [x] Valid JSON already short-circuits — `_extract_json_score` parses as-is first and only
      falls back to repairs on `JSONDecodeError`
- [ ] Log which repairs fire (would identify the most common LLM failures)
- [x] Repair failure returns `None` and logs, rather than crashing
- [x] `tests/test_musicxml_fix.py` exists (419 lines)

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

**Status**: ❌ NOT STARTED — still three near-identical copies. The duplication has since
grown: `executor_kwargs()` and the `max_iter` param had to be added to all three by hand.

**Actions**:
- [ ] Create `src/researcher/llm_factory.py` with `make_llm(model, params) -> LLM` and `make_planning_llm(model) -> LLM`
- [ ] Replace the three duplicated blocks with calls to this factory
- [ ] Move `probe_model_capabilities()` from `config.py` into `llm_factory.py`

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

**Status**: ✅ mostly DONE — this entry was stale. Tutor and composer *do* acquire the
per-user semaphore (`routes/tutor.py:125`, `routes/composer.py:105` with
`async with semaphore`). An earlier audit of mine claimed otherwise; that was wrong.

**Actions**:
- [x] Per-user semaphores live on `app.state.crew_semaphores`, shared by all three components
- [x] Applied to researcher, tutor and composer kickoffs
- [ ] Return 409 Conflict instead of silently queueing behind the semaphore (same open
      sub-item as R2)

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

> **Superseded — kept for reference.** This ordering is from 2026-05-23 and several entries
> are complete. Use the "Status at a glance" table at the top of this file for current state,
> and the recommended order below for what to do next.

### Recommended order now (2026-08-10)

| # | Item | Why now |
|---|------|---------|
| 1 | **C4 — Browser score playback** | Gates all composer work. Until a change can be heard in seconds rather than via MuseScore, musical quality cannot be tuned. |
| 2 | **Composer: ABC notation switch** | Cheapest large win available; also deletes most of C5. See the revised COMPOSER section. |
| 3 | **T-C — Feedback loop, phoneme scoring** | The remaining tutor gap now that T-A and T-B have landed. |
| 4 | **R3 — Make cancel actually cancel** | A "cancelled" run still holds the GPU on a single-GPU machine. |
| 5 | **X1 + X2 — LLM factory, central settings** | Duplication keeps growing; every new param must be added in three places by hand. |
| 6 | T-D — TTS first-reply language bug | Small, user-visible. |
| 7 | C1/C2 — Dynamics, articulation, key changes | Worth doing once you can *hear* the difference (after 1). |
| 8 | Tutor: lemmatisation / proper-noun handling | Would sharpen the i+1 metric; needs a per-language NLP dependency. |

### Original 2026-05-23 ordering (historical)

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
