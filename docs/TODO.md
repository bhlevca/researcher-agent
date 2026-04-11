# TODO

## v1.1.0 — Code Refactoring (DONE)

- [x] Split `crew.py` into smaller modules:
  - [x] `image.py` — Image generation backends (SD, ZImage, VRAM management)
  - [x] `tools.py` — Tool definitions (search wrappers, image tools, system tools)
  - [x] `postprocess.py` — Response postprocessing, hallucination rescue, context builder
  - [x] `crew.py` — Crew/Agent/Task orchestration only

## v1.3.1 — Image Generation Fixes (DONE)

- [x] Fix monkey-patch target: `OpenAICompatibleCompletion` (not `crewai.llm.LLM`)
- [x] Fix ZImage scheduler crash: 4 steps, guidance 0.0 for FLUX-distilled turbo model
- [x] Fix post-tool LLM loop: `result_as_answer = True` on `generate_ai_image`
- [x] Fix planning LLM: dedicated `_make_planning_llm()` with `_allow_fc = True` bypass

## v1.4.0 — main.py Refactoring (DONE)

- [x] Split `main.py` into route modules:
  - [x] `routes/chat.py` — `/chat`, `/chat/cancel`, `/chat/continue` (SSE streaming)
  - [x] `routes/ask.py` — `/ask` programmatic API
  - [x] `routes/sessions.py` — Sessions CRUD endpoints
  - [x] `routes/models.py` — `/info`, `/models`, `/model`, `/memory-depth`, `/llm-params`
  - [x] `routes/speech.py` — `/transcribe` Whisper endpoint
- [x] Extract `config.py` — shared constants, Pydantic models, utilities
- [x] Slim `main.py` to app factory + lifespan + route registration (1030 → 160 lines)
- [x] Auth (`auth.py`), files (`ingestion.py`), TTS (`tts.py`) already extracted
- [x] All 200 tests pass

## v1.5.0 — High-Resolution Tiled Image Generation

- [ ] Research tiled diffusion approaches:
  - [ ] MultiDiffusion / Mixture of Diffusers (overlapping tiles with blended seams)
  - [ ] Check if diffusers `StableDiffusionPanoramaPipeline` is compatible with FLUX/Z-Image
  - [ ] img2img upscaling: generate 1024x1024, then tile-upscale to 4K with overlap
- [ ] Implement tiled generation in `image.py`:
  - [ ] Add `target_width` / `target_height` params to `generate_ai_image`
  - [ ] Calculate grid layout: tile count, overlap regions (~128px per edge)
  - [ ] Generate each tile sequentially (fits in 5070 Ti VRAM)
  - [ ] Blend overlapping regions (cosine/gaussian alpha feathering) to eliminate seams
  - [ ] Stitch into final high-res image (Pillow compositing)
- [ ] Coherence strategy: shared seed + positional sub-prompts, or MultiDiffusion if FLUX-compatible
- [ ] UI: aspect ratio picker or width/height sliders, preset sizes (portrait, landscape, panorama)
- [ ] Add `ZIMAGE_HIRES` env var or API param to opt into large generation
- [ ] Add tests for tile stitching and overlap blending

## v1.6.0 — Music Composer (DONE)

- [x] Implement composer module (`src/researcher/composer/`):
  - [x] `crew.py` — ComposerCrew orchestrator (compose, arrange, harmonize, analyze modes)
  - [x] `models.py` — Pydantic request models for all composer operations
  - [x] `storage.py` — SQLite persistence for composer sessions and saved compositions
  - [x] `tools.py` — CrewAI tools: scale_reference, instrument_range, chord_progression_builder, musicxml_template
  - [x] `config/agents.yaml` — Music composer agent with MusicXML reference and 13 mandatory rules
  - [x] `config/tasks.yaml` — Tasks for compose_chat, compose_score, harmonize, analyze
- [x] Implement MusicXML postprocessor (`musicxml_fix.py`):
  - [x] Phase 1 — 16 regex pre-fixes (note-attribute stripping, garbage tag removal, truncation recovery, etc.)
  - [x] Phase 2 — 8 ElementTree structural repairs (ensure children, deduplicate attributes, insert backups, reorder, etc.)
  - [x] Handles qwen3.5:9b broken output patterns (attributes on `<note>`, repeated `<attributes>`, missing `<backup>`)
- [x] Add API endpoints (`routes/composer.py`):
  - [x] Session CRUD: GET/POST/PUT/DELETE `/composer/sessions`
  - [x] SSE streaming: POST `/composer/chat`, `/composer/score`, `/composer/harmonize`, `/composer/analyze`
  - [x] Composition storage: save, list, get, download MusicXML
- [x] Register composer in `main.py` (lifespan, DB init, routes, `/composer` page)
- [x] Implement frontend:
  - [x] `composer.html` — Tab-based UI (Chat, Score, Harmonize, Analyze, Compositions)
  - [x] `js/composer-chat.js`, `composer-score.js`, `composer-sessions.js`, `composer-state.js`
  - [x] Copy/Save/CopyAll/SaveAll/Download buttons, reasoning collapsible
- [x] Add MusicXML postprocessor tests (`tests/test_musicxml_fix.py`) — 21 tests, all passing
- [x] Grand staff support for piano (treble+bass with `<staves>`, `<backup>`, `<staff>`, `<voice>`)


## v2.0.0 — Language Tutor Agent

- [x] Design tutor agent architecture:
  - [x] Conversational mode — read/write/speak in target language
  - [x] Lesson creation — structured lessons with exercises
  - [x] Appraisal — evaluate student responses and tests
  - [x] TTS integration — spoken replies via existing edge-tts
- [x] Implement tutor agent and tasks in CrewAI:
  - [x] `tutor/crew.py` — TutorCrew orchestrator (conversation, lesson, quiz, appraisal crews)
  - [x] `tutor/config/agents.yaml` — Language tutor agent with 4 operating modes
  - [x] `tutor/config/tasks.yaml` — Task templates for conversation, lesson, quiz, appraisal
- [x] Add tutor-specific tools (`tutor/tools.py`):
  - [x] GrammarCheck — inline grammar analysis
  - [x] VocabularyLookup — detailed word entries with conjugation/IPA
  - [x] PronunciationGuide — IPA transcription and sound approximations
  - [x] ConjugationTable — full verb conjugation tables
- [x] Add tutor data layer (`tutor/storage.py`):
  - [x] Separate `tutor_sessions` table (not regular sessions)
  - [x] `vocabulary` table — word tracking with mastery levels
  - [x] `lesson_plans` table — saved lesson content
  - [x] `quiz_results` table — quiz questions, answers, scores
  - [x] Student statistics aggregation for appraisal
- [x] Add tutor API endpoints (`routes/tutor.py`):
  - [x] `/tutor/sessions` CRUD — separate from regular chat sessions
  - [x] `/tutor/chat` — SSE streaming conversation with tutor agent
  - [x] `/tutor/lessons` — generate + save lesson plans with auto vocabulary extraction
  - [x] `/tutor/vocabulary` — manual + auto vocabulary management
  - [x] `/tutor/quiz/generate` — SSE streaming quiz generation
  - [x] `/tutor/quiz/save` + `/tutor/quiz/submit` — save and grade quizzes
  - [x] `/tutor/appraisal` — SSE streaming student progress evaluation
  - [x] `/tutor/sessions/{id}/stats` — aggregated learning statistics
- [x] Add tutor Pydantic models (`tutor/models.py`)
- [x] Register tutor in `main.py` (lifespan, routes)
- [x] Add tutor tests (`tests/test_tutor.py`) — 24 tests, all passing

## v2.1.0 — LLM Parameter Panels (DONE)

- [x] Add configurable LLM params to TutorCrew and ComposerCrew (get/update methods)
- [x] Add DB migration for `tutor_llm_params` and `composer_llm_params` columns
- [x] Add backend routes: GET/POST/reset for `/tutor/llm-params` and `/composer/llm-params`
- [x] Create reusable `page-settings.js` settings panel module
- [x] Add settings panels to tutor.html and composer.html
- [x] Model switch recreates ComposerCrew and TutorCrew with saved params

## v2.2.0 — Composer Quality Improvements

- [ ] Longer scores: increase minimum measure count, encourage 16-32 measure compositions
- [ ] Better musical variety: richer chord progressions (beyond I-IV-V-I), more progression styles
- [ ] Bass staff reliability: ensure two-voice (treble+bass) output consistently
- [ ] Context window optimization: reduce token overhead so model can output more notes
- [ ] Include music theory knowledge in LLM context (counterpoint rules, voice leading)

## v3.0.0 — Music Tutor

- [ ] Design music theory tutor agent:
  - [ ] Teach scales, intervals, chords, progressions, form
  - [ ] Interactive exercises: identify intervals, build chords, harmonize melodies
  - [ ] Grade student answers with detailed explanations
  - [ ] Progressive curriculum from beginner to advanced
- [ ] Integrate with composer: tutor can generate example scores
- [ ] Add ear training exercises with audio playback
