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


## Known Issues (Quiz Features)

- [x] Reorder quiz: Does not work reliably, often fails to render or grade correctly. Fixed: words now always derived from correct_answer; accent-fold grading.
- [x] Translation quiz: Grading is unreliable, too strict or too lenient, and feedback is unclear. Fixed: LLM grader hardened; correct answer shown once; feedback separated.
- [x] Matching quiz: Often fails to build the answer grid or serialize student pair selections correctly. Fixed: pair normalization + correct_answer rebuild.

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

## v2.2.0 — Competent Composer: Music Knowledge & Creativity

### Phase 1 — Genre Knowledge Database
- [ ] Build structured YAML knowledge files (`src/researcher/composer/knowledge/`):
  - [ ] Per-genre composition templates (23 genres) with:
    - [ ] Rhythmic patterns as beat grids (e.g. bossa: `bass [1, ., 2+, .] | comping [., 1+, 2, 2+]`)
    - [ ] Typical melodic intervals and contour shapes per style
    - [ ] Voicing templates (jazz: rootless/drop-2; classical: close/open position)
    - [ ] Form templates (AABA, 12-bar, verse-chorus, binary, ternary, rondo)
    - [ ] Harmonic rhythm rules (chords per bar, where changes accelerate)
  - [ ] Universal counterpoint & voice leading rules:
    - [ ] Interval consonance/dissonance tables
    - [ ] Resolution rules (7ths down, leading tones up, tendency tones)
    - [ ] Motion types with do/don't examples
    - [ ] Cadence patterns with note-by-note examples
  - [ ] Diatonic chord maps for ALL modes (dorian, phrygian, lydian, mixolydian, etc.)
  - [ ] Extended chord voicings (9th, 11th, 13th, altered dominants)

### Phase 2 — Motif-First Composition Workflow
- [ ] Change agent instructions from "compose 16 bars" to structured workflow:
  - [ ] Step 1: Create a 2-bar motif (melodic seed)
  - [ ] Step 2: Develop motif using sequence, inversion, variation, fragmentation
  - [ ] Step 3: Build phrase structure (antecedent/consequent, tension/release arc)
  - [ ] Step 4: Add bass voice as independent counterpoint (not chord roots)
- [ ] Inject bar-by-bar structural directives from form templates:
  - [ ] e.g. "Bars 1-4: expose motif (ascending). Bars 5-8: answer (descending mirror).
         Bar 8: half cadence. Bars 9-12: develop. Bars 13-16: climax + authentic cadence."

### Phase 3 — Few-Shot Examples & Creativity Fuel
- [ ] Add 2-4 bars of genre-authentic example measures in JSON format per style
- [ ] Include motif development techniques as concrete note examples:
  - [ ] Repetition, sequence (step up/down), inversion, retrograde
  - [ ] Fragmentation, augmentation, diminution
  - [ ] Call-and-response patterns
- [ ] Inject curated examples into CompositionPrep output as "EXAMPLE" section

### Phase 4 — Model Evaluation
- [ ] Test gemma3:12b as composer (12B, 8.1GB, fits easily in 16GB VRAM)
- [ ] Test mistral-small3.2:24b if gemma insufficient (24B, 15GB, tight fit)
- [ ] A/B compare outputs: same prompt, different models, score musical quality
- [ ] Document which model produces best composition results

### Phase 5 — Advanced (Future)
- [ ] Multi-pass composition (melody skeleton → rhythm fill → bass → dynamics)
- [ ] Dynamics and articulation support in JSON schema + MusicXML builder
- [ ] Modulation support (pivot chords, secondary dominants, key changes)
- [ ] Orchestration rules (doubling, balance, register-specific colors)

## v2.3.0 — Language Tutor Enhancements

### Phase 1 — Spaced Repetition System (SRS)
- [x] Implement SM-2 algorithm in `tutor/storage.py`:
  - [x] Add `next_review`, `interval_days`, `ease_factor` columns to `vocabulary` table (with migration)
  - [x] Schedule reviews based on mastery level and past performance (`sm2_review()`, `_sm2_compute()`)
  - [x] Update intervals on correct/incorrect quiz answers (via `update_vocabulary_mastery`)
- [x] Add `/tutor/vocabulary/due` endpoint — returns words due for review (today or past-due)
- [x] Add "Review Due (N)" button in Vocabulary tab that launches a focused flashcard quiz
- [x] Auto-schedule newly extracted vocabulary for first review (1 day)

## v2.4.0 — OpenAI-compatible /v1 Proxy (ollama launch layer)
- [x] Add `routes/proxy.py` with OpenAI-compatible endpoints:
  - [x] `GET  /v1/models` — list Ollama models in OpenAI format
  - [x] `GET  /v1/models/{model_id}` — single model record
  - [x] `POST /v1/chat/completions` — streaming + non-streaming chat proxy with reasoning hint injection
  - [x] `POST /v1/completions` — legacy text completions pass-through
  - [x] `POST /v1/embeddings` — embeddings pass-through
- [x] Optional `PROXY_API_KEY` env var — if set, requires `Authorization: Bearer <key>`
- [x] Auto-inject reasoning system prompt for thinking models (deepseek-r1, qwq, qwen3)
- [x] Documented in README.md and .env.example
- [x] Usage: set `OPENAI_BASE_URL=http://localhost:8000/v1` in any OpenAI-compatible tool

### Phase 2 — Semantic Answer Grading
- [x] Replace exact string match in quiz grading with LLM-based evaluation:
  - [x] Accept synonyms, alternate spellings, minor typos
  - [x] Partial credit scoring (0.0–1.0 instead of binary)
  - [x] Detailed feedback explaining why an answer is wrong
- [x] Update `quiz_results` schema to store float scores
- [x] Add grading prompt template to `tutor/config/tasks.yaml`

### Phase 3 — Richer Quiz Types
- [x] Matching quiz: pair words with translations (numbered dropdowns)
- [x] Sentence reordering: shuffle word chips, student reconstructs correct order
- [x] Listening comprehension: TTS plays phrase, student types what they heard
- [x] Cloze passages: fill gaps in a paragraph (context-dependent answers)
- [x] Update quiz UI to handle new question formats
- [x] Update quiz generation prompt to produce new formats

### Phase 4 — Situational Dialogs
- [ ] Add "situation" dialog generator:
  - [ ] Pre-defined scenarios: restaurant, airport, hotel, doctor, shopping, directions
  - [ ] Multi-turn roleplay with the tutor as conversation partner
  - [ ] Inline corrections and vocabulary extraction during dialog
- [ ] Optional scene illustration via Z-Image pipeline
- [ ] Save dialog transcripts to lesson_plans table
- [x] Translation quiz: Input deleted, correct version listed twice, too strict marking
- [x] Cloze quiz: Giving 1/2 point for perfect answers (should give full points)
- [x] Reorder quiz: French vocabulary errors (e.g., "heures" instead of "heureux"), missing articles/prepositions
- [x] Listening quiz: Incorrect marking for minor differences (e.g., "cinema" vs "cinéma")
- [x] UI: Need vertical scroll for all windows bigger than visible area


### Phase 5 — Lesson Export
- [ ] Add `/tutor/lessons/{id}/export` endpoint:
  - [ ] Markdown export (`.md`) — direct from stored content
  - [ ] DOCX export via `python-docx` with proper formatting
  - [ ] Include vocabulary tables and exercise sections
- [ ] Add Download button (MD / DOCX) to Lessons tab

### Phase 6 — Progress Visualization
- [ ] Add Chart.js to tutor frontend:
  - [ ] Vocabulary growth over time (words learned per week)
  - [ ] Quiz score trends (line chart)
  - [ ] Mastery distribution (pie/bar: how many words at each level 0-5)
  - [ ] Weak areas identification (lowest-scoring topics)
- [ ] Add `/tutor/stats/history` endpoint for time-series data
- [ ] Integrate charts into Stats tab

### Phase 7 — Structured Curriculum
- [ ] Define topic sequences per CEFR level in YAML:
  - [ ] A1: greetings → numbers → food → directions → time → weather
  - [ ] A2: past tense → travel → shopping → health → hobbies
  - [ ] B1: opinions → work → news → conditional → subjunctive
  - [ ] B2+: idioms → formal writing → debate → literature
- [ ] Add "Next Lesson" button that follows the curriculum path
- [ ] Track completed topics per user per language
- [ ] Show curriculum progress bar in Stats tab

### Phase 8 — Pronunciation Scoring
- [ ] Use Whisper transcription of student speech recordings
- [ ] Compare transcription against expected text (word error rate / character accuracy)
- [ ] Display per-word accuracy highlighting (correct/incorrect/missing)
- [ ] Add pronunciation exercise mode: tutor speaks → student repeats → score

## v3.0.0 — Music Tutor (Future)

- [ ] Design music theory tutor agent:
  - [ ] Teach scales, intervals, chords, progressions, form
  - [ ] Interactive exercises: identify intervals, build chords, harmonize melodies
  - [ ] Grade student answers with detailed explanations
  - [ ] Progressive curriculum from beginner to advanced
- [ ] Integrate with composer: tutor can generate example scores
- [ ] Add ear training exercises with audio playback
