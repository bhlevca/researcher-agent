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

## v1.4.0 — High-Resolution Tiled Image Generation

- [ ] Research tiled diffusion approaches:
  - [ ] MultiDiffusion / Mixture of Diffusers (overlapping tiles with blended seams)
  - [ ] diffusers `StableDiffusionPanoramaPipeline` or equivalent for ZImage
  - [ ] img2img upscaling: generate 512x512, then tile-upscale to 4K with overlap
- [ ] Implement tiled generation:
  - [ ] Split target canvas (e.g. 2048x2048 or 3840x2160) into overlapping tiles
  - [ ] Generate each tile sequentially (fits in VRAM)
  - [ ] Blend overlapping regions (linear/gaussian feathering) to eliminate seams
  - [ ] Stitch into final high-res image
- [ ] Shared prompt conditioning: ensure all tiles use same latent noise schedule / global composition prompt
- [ ] Add `ZIMAGE_HIRES` env var or API param to opt into 4K generation
- [ ] Add tests for tile stitching and overlap blending

## v1.5.0 — Music Composer

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
