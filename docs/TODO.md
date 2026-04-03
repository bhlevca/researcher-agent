# TODO

## v1.1.0 — Code Refactoring

- [ ] Split `crew.py` into smaller modules:
  - [ ] `llm.py` — LLM construction (`_make_llm`, model config)
  - [ ] `image.py` — Image generation backends (SD, ZImage, VRAM management)
  - [ ] `tools.py` — Tool definitions (search wrappers, image tools, system tools)
  - [ ] `crew.py` — Crew/Agent/Task orchestration only
- [ ] Split `main.py` into smaller modules:
  - [ ] `routes/chat.py` — `/chat`, `/chat/continue` endpoints
  - [ ] `routes/ask.py` — `/ask` endpoint
  - [ ] `routes/auth.py` — Authentication endpoints
  - [ ] `routes/files.py` — File upload/management endpoints
  - [ ] `postprocess.py` — Response postprocessing, hallucination rescue
  - [ ] `context.py` — Conversation context builder (3-tier)
  - [ ] `main.py` — App factory, startup, middleware only
- [ ] Extract shared constants/config into `config.py`
- [ ] Add type hints where missing
- [ ] Ensure tests pass after refactoring

## v2.0.0 — Language Tutor Agent

- [ ] Design tutor agent architecture:
  - [ ] Conversational mode — read/write/speak in target language
  - [ ] Lesson creation — structured lessons with exercises
  - [ ] Appraisal — evaluate student responses and tests
  - [ ] TTS integration — spoken replies for pronunciation practice
- [ ] Implement tutor agent and tasks in CrewAI
- [ ] Add tutor-specific tools:
  - [ ] Grammar checker
  - [ ] Vocabulary builder
  - [ ] Pronunciation guide (TTS)
  - [ ] Lesson generator (structured output)
  - [ ] Test/quiz generator and grader
- [ ] Add tutor endpoints to API
- [ ] Add tutor UI components
- [ ] Add tutor tests
