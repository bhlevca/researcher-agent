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

## v1.6.0 — Music Composer

- [ ] Research music generation models:
  - [ ] Meta MusicGen (text-to-music, MIT license, runs on GPU)
  - [ ] Riffusion (Stable Diffusion fine-tuned on spectrograms)
  - [ ] AudioCraft / MusicGen-Melody (melody-conditioned generation)
- [ ] Implement music generation backend:
  - [ ] Lazy-load model with same VRAM management pattern as ZImage
  - [ ] Ollama unload/reload around inference
  - [ ] Generate audio from text prompt
  - [ ] Save as WAV/MP3, serve from `/static/generated/`
- [ ] Add CrewAI tool: `GenerateMusic` (prompt → audio file)
- [ ] Add audio player support in frontend
- [ ] Add music generation tests
- [ ] Score generation and compatibility with MuseScore that will allow also midi audio play


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
