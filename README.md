# Researcher Agent

A CrewAI-powered research agent with a browser-based chat UI, voice dialog, image generation, and session management. Runs locally using Ollama — the **client only needs a web browser**.

**Version 2.16.0**

## Architecture

| Component | Runs on | What it does |
|-----------|---------|--------------|
| **Ollama + LLM** | Server | Local LLM inference (qwen3.5:9b recommended) |
| **CrewAI + FastAPI** | Server | Agent orchestration, API endpoints, SSE streaming |
| **faster-whisper** | Server | Speech-to-text (voice input from browser mic) |
| **edge-tts** | Server | Text-to-speech (audio streamed back to browser) |
| **Stable Diffusion** | Server | AI image generation (optional, needs NVIDIA GPU) |
| **ZImage (Z-Image-Turbo)** | Server | Fast AI image generation via diffusers (optional, needs NVIDIA GPU) |
| **Vision critic** | Server | Inspects each generated image and triggers a re-render if it does not match the request |
| **Wan 2.2 / LTX-Video** | Server | Text-to-video generation (optional, needs NVIDIA GPU) |
| **SymPy + matplotlib** | Server | Math Tutor: computes every answer and draws every figure — no LLM involved |
| **LaTeX (pdflatex)** | Server | Typesets printable worksheets and answer keys |
| **Browser UI** | Client | Just a modern web browser — no plugins, no installs |

> **The client computer needs nothing except a web browser with microphone access** (for voice features). All AI processing, TTS, and STT happen on the server.

## Server Requirements

### System Dependencies

Install these on the **server** machine:

| Package | Required | Purpose | Install (example) |
|---------|----------|---------|-------------------|
| [Ollama](https://ollama.com/) | Yes | Local LLM runtime | See ollama.com |
| Python 3.11+ | Yes | Application runtime | `pyenv install 3.12` |
| ffmpeg | No | No longer needed — faster-whisper decodes in-process via PyAV, which bundles its own ffmpeg libraries | — |
| NVIDIA GPU + CUDA | Recommended | GPU inference for Ollama & Stable Diffusion | Driver + CUDA toolkit |
| TeX Live (`pdflatex`) | For Math Tutor | Typesets printable worksheets; `extsizes` gives the large type primary grades need | `zypper in texlive-latex texlive-extsizes` |

### Python Dependencies

All managed automatically via `pip install -e .`:

| Package | Purpose |
|---------|---------|
| crewai[tools] | Agent framework with search tools |
| fastapi | HTTP API server |
| uvicorn | ASGI server |
| python-dotenv | Environment variable loading |
| langchain-community | LangChain integrations |
| httpx | Async HTTP client (Ollama API) |
| Pillow | Image generation (text-to-image tool) |
| diffusers | Stable Diffusion pipeline |
| transformers | Model loading for SD |
| accelerate | GPU offload for SD |
| safetensors | Fast model weight loading |
| aiosqlite | Async SQLite for session storage |
| faster-whisper | Speech-to-text (CTranslate2; int8 on CPU by default) |
| sentencepiece | T5 tokenizer for LTX/Wan — without it transformers misroutes `spiece.model` |
| imageio-ffmpeg | H.264 MP4 export (OpenCV's mp4v is not browser-playable) |
| fsrs | Spaced-repetition scheduler for the tutor |
| odfpy | ODT export |
| edge-tts | Text-to-speech via Microsoft Edge voices (server-side, streams audio to browser) |
| PyJWT | JWT token generation and validation |
| bcrypt | Secure password hashing |
| pymupdf | PDF text extraction (PyMuPDF/fitz) |
| python-docx | DOCX read/write (upload + export) |
| openpyxl | XLSX read/write (upload + export) |

## Client Requirements

**A modern web browser.** That's it.

- Chrome, Firefox, Edge, Safari, or any Chromium-based browser
- Microphone access (optional, for voice input)
- No software installation, no plugins, no system dependencies on the client

## Setup

```bash
# Clone the repository
git clone https://github.com/bhlevca/researcher-agent.git
cd researcher-agent

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install all Python dependencies
pip install -e .

# Pull a model in Ollama
ollama pull qwen3.5:9b

# Copy and configure environment
cp .env.example .env
# Edit .env — generate real secrets (see below)
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL` | Ollama model (litellm format) | `ollama/qwen3.5:9b` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `SERPER_API_KEY` | Google Serper API key for web search | *(required)* |
| `JWT_SECRET` | Secret key for signing JWT tokens | *(required)* |
| `INVITE_CODE` | Invite code required to register | *(required)* |
| `IMAGE_BACKEND` | Image generation backend: `zimage`, `sd`, `ernie`, `flux`, or `none` | `sd` |
| `ZIMAGE_MODEL` | HuggingFace repo for ZImage pipeline | `mrfakename/Z-Image-Turbo` |
| `ZIMAGE_WIDTH` | Generated image width (ZImage) | `512` |
| `ZIMAGE_HEIGHT` | Generated image height (ZImage) | `512` |
| `SD_MODEL` | HuggingFace repo for Stable Diffusion | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| `HUGGINGFACE_TOKEN` | HuggingFace token for gated model downloads | *(optional)* |
| `TRANSFORMERS_OFFLINE` | Set to `1` after first model download for offline use | `0` |
| `PROXY_API_KEY` | Bearer token required for `/v1/*` proxy endpoints (empty = open) | *(optional)* |
| `IMAGE_REFINE_ROUNDS` | Image critique-and-retry rounds; 0 disables | `2` |
| `IMAGE_CRITIC_MODEL` | Vision model that inspects generated images | `qwen3-vl:8b` |
| `VIDEO_BACKEND` | Video generator: `wan`, `ltx`, or `none` | `wan` |
| `OCR_MODEL` | Vision model used to read scans | `deepseek-ocr:3b` |
| `WHISPER_MODEL` | faster-whisper size | `small` |
| `SRS_ALGO` | Tutor scheduler: `fsrs` or `sm2` | `fsrs` |
| `COMPOSER_MODEL` | Music-specialist model for composing (empty = general model) | *(unset)* |

> This table lists the settings people change most often. **`.env.example` documents
> all 64**, commented out and showing each actual default — it is generated against
> the `os.getenv()` calls in `src/`, so it cannot drift out of date silently.

### Authentication Setup

All functional endpoints require authentication. Users must register and log in to use the agent.

Generate secrets for your `.env` file:

```bash
# Generate JWT_SECRET
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate INVITE_CODE
python3 -c "import secrets; print(secrets.token_urlsafe(12))"
```

Add them to `.env`:

```
JWT_SECRET=<paste-generated-secret>
INVITE_CODE=<paste-generated-code>
```

#### What is `JWT_SECRET`?

`JWT_SECRET` is the cryptographic key used to **sign and verify login tokens** (JWTs). When a user logs in, the server creates a token signed with this secret. On every subsequent API request, the server verifies the token's signature — proving it wasn't tampered with.

- **Used in** `src/researcher/auth.py`: `_create_token()` signs tokens, `_decode_token()` verifies them.
- **If it disappears from `.env`**: the code falls back to a weak built-in default. All previously-issued tokens become invalid (every user is logged out), and new tokens are signed with the weak default — a security risk.
- **If you lose it**: generate a new one (see above), add it to `.env`, and restart the server. Users will need to log in again, but their **accounts and sessions in the database are unaffected** — only the login tokens expire, not passwords or saved data.
- **Bottom line**: the secret only affects active login sessions, not stored data. Losing it forces a re-login but is not destructive.

Security features:
- **Invite code gate** — only people who know the code can register
- **Per-IP rate limiting** — max 5 login/register attempts per 60 seconds
- **Honeypot field** — blocks automated bot submissions
- **JWT Bearer tokens** — stateless session management
- **bcrypt password hashing** — passwords are never stored in plaintext
- **Session isolation** — each user can only access their own sessions

### Inviting Users

To allow someone to use your agent:

1. Share the **invite code** from your `.env` file with them (via email, message, etc.)
2. Share the **server URL** (e.g. `http://your-server:8000`)
3. They open the URL in a browser, click **Register**, enter a username, password, and the invite code
4. Once registered, they log in with their username and password — the invite code is only needed once

To rotate the invite code (invalidate the old one while keeping existing users):

```bash
# Generate a new code
python3 -c "import secrets; print(secrets.token_urlsafe(12))"
# Update INVITE_CODE in .env and restart the server
```

### Managing Users

There is no admin UI yet. Manage users directly via SQLite:

```bash
# List all registered users
sqlite3 src/researcher/data/sessions.db "SELECT id, username, created_at FROM users;"

# Delete a user and all their sessions
sqlite3 src/researcher/data/sessions.db \
  "DELETE FROM sessions WHERE user_id='<user-id>'; DELETE FROM files WHERE user_id='<user-id>'; DELETE FROM users WHERE id='<user-id>';"

# Count sessions per user
sqlite3 src/researcher/data/sessions.db \
  "SELECT u.username, COUNT(s.id) FROM users u LEFT JOIN sessions s ON s.user_id = u.id GROUP BY u.id;"
```

## Usage

```bash
# Start the server
uvicorn researcher.main:app --host 0.0.0.0 --port 8000

# Open in browser
# http://localhost:8000
```

### Voice Dialog Mode

1. Click the **🗣️ Dialog** button to enable dialog mode
2. Click **🎤** to record your question — it auto-sends when you stop
3. The server generates speech audio and streams it back to the browser
4. After the response is spoken, the mic auto-activates for the next turn

All voice processing happens server-side: Whisper (STT) and edge-tts (TTS). The browser just records audio and plays it back.

### Generating Images and Video

Just ask — the agent picks the right tool. For direct control, prefix the message
and the agent is bypassed entirely:

| Prefix | Effect |
|--------|--------|
| `image:` `img:` `picture:` `photo:` `draw:` | Prompt goes straight to the image model, verbatim |
| `video:` `clip:` `animation:` | Prompt goes straight to the video model, verbatim |
| *(no prefix)* | Normal agent path — it plans, writes the prompt, and calls the tool |

The prefix skips planning (~30 s) and stops the agent rewording your prompt, which
matters when you are iterating on exact phrasing. It is opt-in: without a prefix
nothing changes.

#### Image self-refinement

A diffusion model renders once and never looks at the result. Here, a vision model
inspects each image against the request and re-renders if it is wrong:

```
prompt → render → critic inspects → wrong? rewrite prompt → render again (max N)
```

This runs for **every** image, whether the agent or a prefix triggered it — the loop
lives in the tool, not the agent. It costs roughly 2–3x the time of a single shot;
set `IMAGE_REFINE_ROUNDS=0` to disable.

The critic (`IMAGE_CRITIC_MODEL`, default `qwen3-vl:8b`) must be a **vision** model
pulled in Ollama. If it is missing, refinement silently does nothing — the server
logs `[IMAGE] WARNING: …` at startup to make that visible. It catches gross errors
(wrong subject, missing elements) reliably; fine-grained counting is less reliable.

#### Video

```bash
# .env
VIDEO_BACKEND=wan            # wan | ltx | none
```

**Wan 2.2 TI2V-5B** is the default. Weights are ~32 GiB and are **never** fetched
during a chat request — download once:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --include 'text_encoder/*' --include 'transformer/*' --include 'vae/*' \
  --include 'tokenizer/*' --include 'scheduler/*' --include '*.json'
```

Measured on a 16 GB RTX 5070 Ti, same prompt through both models:

| | LTX-Video 2B | Wan 2.2 TI2V-5B |
|---|---|---|
| Output | 704x512, 2.3 s | 832x480, 2.7 s |
| Motion (mean inter-frame change) | 8.0 | 12.2 |
| Detail (Laplacian variance) | 27.7 | **424.1** |
| Time per clip | ~99 s | ~208 s |

Roughly 2x the time for ~15x the detail. LTX is kept as a faster fallback.

**Why not something bigger?** Evaluated and rejected for a 16 GB / 64 GB machine:

| Model | Why not |
|---|---|
| Wan 2.2 A14B | Mixture-of-experts: **two** 26.6 GiB transformers, ~58.5 GiB of weights. Exceeds 64 GB RAM once CUDA and the app are resident. |
| LTX-2 | 19B transformer (25 GiB even in fp8) plus a ~55 GiB Gemma3 text encoder. "Runs on 16 GB" refers to ComfyUI with GGUF quantisation, not diffusers. |
| HunyuanVideo 1.5 | No `model_index.json` — not a diffusers pipeline at all. |
| lightx2v 4-step LoRAs | Exist only for A14B; no TI2V-5B variant. |

CPU offload raises the ceiling but not indefinitely: diffusers keeps weights in
system RAM and **streams** them to the GPU each step — unlike llama.cpp, it never
computes on the CPU. So the limit becomes RAM, and the cost is PCIe traffic
repeated every denoising step. Wan 5B avoids that entirely: at ~9.3 GiB in bf16
the transformer sits wholly in VRAM.

##### Settings are per backend

Each model has its own trained resolution and frame rate, and using one model's
numbers on the other visibly degrades motion — generating LTX clips at 16 fps
while it conditions on 25 made walking animals appear to limp.

| | Wan | LTX |
|---|---|---|
| Resolution | 1280x704 | 704x512 |
| Frame rate | 24 | 25 (conditioning, not just playback) |
| Guidance | 5.0 | 3.0 |

Switching backend re-derives anything you have not pinned in `.env`. Setting
`VIDEO_WIDTH` and friends pins them across **both** backends, so leave them
commented unless you mean it.

Memory scales with the **frame count** — frames are denoised jointly — so raising
fps shortens the clip that fits. On OOM the generator retries with fewer frames,
then smaller ones, rather than failing outright.

##### Longer videos: clip stitching

A single clip is ~3 seconds, and pushing duration further is the wrong lever —
at 5 seconds the model runs out of coherent movement and drifts toward static
(measured: mean inter-frame motion fell from 12.2 to 4.2). Length comes from
**stitching** instead.

Set **⚙️ Params → Video → Clips** (1–24). Each clip is generated from the
**last frame of the previous one**, because Wan 2.2 is a text-*and-image* model.
That is what makes the joins continuous — concatenating independently generated
clips gives a hard cut to an unrelated scene.

Measured at 720p: 3 clips → 193 frames → 8.0 s in 22.6 minutes, with inter-frame
change at the joins (2.2, 3.7) at or below the clip average of 3.4 — no visible seam.

| Target | Clips | Time at 720p |
|--------|-------|--------------|
| ~8 s | 3 | ~23 min |
| ~30 s | 11 | ~83 min |
| ~60 s | 22 | ~2.8 hours |

Two caveats. Cost is linear and **time is the only real limit** — frames live in
system RAM (~3 GiB for a minute of 720p), not VRAM. And quality **drifts** over a
long chain: each clip is conditioned only on the previous frame, so the scene
wanders. Over 11 clips expect the background to change character.

Progress streams to the reasoning panel, since an 80-minute request that reports
nothing is indistinguishable from a hang:

```
🎬 Clip 2/11 (33% — 128 frames so far, ~5.3s of video)
🎬 33% — step 50/150 — about 42m 10s left
```

### Tuning at runtime

**⚙️ Params** in the header exposes every runtime-tunable setting, grouped:
LLM generation, crew limits, image generation, image self-refinement, OCR,
speech-to-text, tutor spaced repetition, and video. LLM and image settings persist
per user; the rest are process-wide and reset to `.env` on restart.

Every setting the code reads is documented in `.env.example`, commented out and
showing its actual default.

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | No | Chat UI |
| GET | `/info` | No | Current model info |
| GET | `/auth/config` | No | Auth configuration (invite required?) |
| POST | `/auth/register` | No | Register a new account (invite code required) |
| POST | `/auth/login` | No | Log in and receive JWT token |
| GET | `/auth/me` | Yes | Current user info |
| GET | `/models` | Yes | List available Ollama models |
| POST | `/model` | Yes | Switch active model |
| POST | `/chat` | Yes | SSE-streaming chat with the agent |
| POST | `/ask` | Yes | Synchronous single-question endpoint |
| GET | `/sessions` | Yes | List saved sessions (own only) |
| POST | `/sessions` | Yes | Create a session |
| GET | `/sessions/{id}` | Yes | Load a session (own only) |
| PUT | `/sessions/{id}` | Yes | Update a session (own only) |
| DELETE | `/sessions/{id}` | Yes | Delete a session (own only) |
| POST | `/transcribe` | Yes | Speech-to-text (Whisper, server-side) |
| GET | `/tts/voices` | No | List available TTS voices (edge-tts, server-side) |
| POST | `/tts/speak` | No | Text-to-speech audio (edge-tts, server-side) |
| POST | `/files/upload` | Yes | Upload files (TXT, CSV, PDF, DOCX, XLSX, MD, JSON) |
| GET | `/files` | Yes | List uploaded files (own only) |
| GET | `/files/{id}` | Yes | Get file details + extracted text (own only) |
| DELETE | `/files/{id}` | Yes | Delete an uploaded file (own only) |
| POST | `/export` | Yes | Export content to MD, TXT, DOCX, or XLSX |
| GET | `/math` | No | Math Tutor UI |
| GET | `/math/curriculum` | Yes | Grades 1–12 with their IB equivalents and topics |
| POST | `/math/generate` | Yes | Generate problems + HTML preview (nothing stored) |
| POST | `/math/pdf` | Yes | Printable worksheet (PDF) |
| POST | `/math/answers` | Yes | Answer key for the same sheet (PDF) |
| POST | `/math/worksheets` | Yes | Save a worksheet recipe |
| GET | `/math/worksheets` | Yes | List saved worksheets (own only) |
| GET | `/math/worksheets/{id}` | Yes | Load a saved worksheet, regenerated from its seed |
| GET | `/math/worksheets/{id}/pdf` | Yes | Reprint it (`?answers=true` for the key) |
| DELETE | `/math/worksheets/{id}` | Yes | Delete a saved worksheet |
| POST | `/math/results` | Yes | Record a marked sheet |
| GET | `/math/progress` | Yes | Accuracy per topic, weakest first |

## Math Tutor

Open `/math`. Generates printable mathematics worksheets with matching answer
keys, Grade 1 to Grade 12, at International Baccalaureate level.

**No language model is involved in producing a question or an answer.** Every
answer is computed with SymPy and every figure is drawn with matplotlib from the
same parameters that produced the answer, so a diagram cannot contradict its
question. The reason is simple: a wrong note in a composition is a bad listen,
but a wrong answer on a worksheet teaches a child something false, and whoever
marks it then reinforces the error.

### Levels

Grades carry their IB equivalent and **either naming can be used to choose a
level** — `grade8`, `MYP3`, `Grade 8` and `8` all select the same level. The
equivalence is shown in the selector and printed on the sheet, because the two
schemes do not line up the way people assume: PYP and MYP number their years
from the start of the *programme*, so Grade 8 is MYP 3, not MYP 8.

| Grade | IB | Grade | IB |
|-------|-----|-------|-----|
| 1–5 | PYP Years 2–6 | 9 | MYP Year 4 |
| 6 | MYP Year 1 | 10 | MYP Year 5 |
| 7 | MYP Year 2 | 11 | DP 1 — Analysis & Approaches SL |
| 8 | MYP Year 3 | 12 | DP 2 — Analysis & Approaches HL |

The sequence is IB, not North American: algebra starts in Grade 6, Pythagoras
and circle geometry land in Grade 8, exact form (surds, fractions) is expected
throughout rather than decimals, and Grades 11–12 cover calculus, complex
numbers, proof by induction, differential equations and Maclaurin series.

### Languages

Sheets print in **English, French or Romanian**, or in a **pair** — the first
language leads and the second follows in italics, which is what a child learning
in a second language actually needs.

Nothing is translated at generation time. The wording comes from
[`phrases.yaml`](src/researcher/mathtutor/phrases.yaml), a catalogue of ~150
whole sentence frames written once and checked in. That matters for two reasons:

- **A seed stays a worksheet.** Run the text through a model each time and the
  French wording drifts between runs, so a reprint would be a different sheet.
- **Terminology.** French maths says *coefficient directeur*, not "gradient";
  *médiatrice*, not "bissectrice perpendiculaire". Romanian says *mediatoarea*,
  *rația*, *patrulater inscriptibil*. A general translator gets these plausibly
  wrong, which is the worst kind of wrong on a worksheet.

Frames are whole sentences rather than words, so a translation can reorder the
placeholders; gluing translated words together produces "Trouver le aire de ce
cercle". Decimals print with a comma in French and Romanian, as they should.

The file is data — edit it and the next sheet changes. A test fails if any
phrase is missing a language, or if an English sentence reaches a French sheet.

### Geometry

53 topics, 32 of which draw a figure: right triangles for Pythagoras, circles,
compound areas, coordinate grids, polygons, SAS triangles for the sine and
cosine rules. Figures render to SVG for the browser and PDF for printing, in
black line work only — worksheets get photocopied, and colour turns to grey mush.

### Printing

PDF is the real output; the on-screen preview *is* the PDF, so what is seen is
what prints. Type size and working space scale with the grade — 14pt with short
gaps for Grade 1, 11pt with half a page for a Grade 12 integration question.
A sheet can carry the student's name; the answer key never does.

### A seed is a worksheet

Generation is deterministic, so only the recipe (grade, topics, count, seed) is
stored, never the problems. A sheet printed in March reprints identically in
June, and its answer key cannot drift from it. Saving a worksheet stores about a
hundred bytes.

### Marking and progress

Marked sheets can be recorded per question, and `/math/progress` reports
accuracy per topic with the weakest first — so "which topics is she actually
getting wrong" is a query rather than a memory.

## Language Tutor — printable sheets

Open `/tutor` and press **🖨️ Sheets**. Generates homework, tests, exams and
quick challenges with matching answer keys, for **French, German and Romanian**.

Everything that already existed is untouched: chat, lessons, passages, quizzes,
vocabulary with FSRS, scenarios, translation, appraisal, pronunciation scoring,
voice and TTS. The sheets are additive.

### Why the answers are trustworthy

Grammar is rule-governed, so **no model produces an exercise or its answer**.
Every answer is read out of a checked-in language pack
([`tutor/langpacks/*.yaml`](src/researcher/tutor/langpacks)) or derived from a
table in one. A child marked wrong on a correct answer learns something false
and is then told it twice — once by the sheet and once by whoever marks it.

The packs are data, so a form that reads wrong to a speaker is fixed by editing
YAML, and a language is extended by adding entries.

### What each language drills

The packs are shaped by the language, not copied from one another:

| | |
|---|---|
| **French** | Conjugation across four tenses, elision (`l'ami`, `j'ai`), prenominal adjectives (`le petit livre`), the `bel`/`vieil`/`nouvel` pre-vowel forms, and *être* agreement in the compound past (`nous sommes partis`) |
| **German** | Four cases, plurals, and adjective endings **derived from the weak/mixed/strong tables** rather than transcribed — so none of the 48 endings can be mistyped into a key |
| **Romanian** | The enclitic definite article (`masă → masa`, `câine → câinele`), three genders including neuter, and the compound past and future |

### Vocabulary

Each pack now carries roughly 60 verbs, 60 nouns and 30 adjectives across
A1–B2 — enough that a 20-question sheet does not repeat itself.

Regular verbs are **one line**: `{inf: donner, gloss: to give, class: er}`. The
paradigm is derived by [`conjugator.py`](src/researcher/tutor/conjugator.py),
which handles the subclasses that trip people up — `commençons`/`commencions`,
`mangeons`/`mangions`, `achète`/`achetons`/`achèterai`, German's missing `ge-`
on `besucht` and `studiert`, the linking `-e-` in `gearbeitet`. Irregular verbs
keep explicit tables.

That derivation is not taken on trust: a test expands the verbs the packs *also*
spell out by hand and requires the two to agree, so the rules are checked
against known-good data. Adding a verb is one line; adding a language is a new
YAML file.

### Stories and summaries

The one part that genuinely needs a model — a story cannot be derived from a
table. Press **Story + summary**, give a topic, and the model writes a levelled
story with comprehension questions, a summary task and a short glossary.

Two consequences, both deliberate rather than compromises:

- **The key prints "Model answer"**, not "Answer". A comprehension answer is
  one acceptable answer; printing it as *the* answer invites an adult to mark a
  better one wrong.
- **The story is stored verbatim, not as a seed.** For grammar a seed rebuilds
  the sheet exactly; a second model call writes a *different* story, so a
  reprint would no longer match the paper the child was given.

A response the model mangles is reported rather than printed — an empty
question is worse than an error.

### Lesson sheets

A lesson is a full teaching document — objectives, grammar explanation, a
vocabulary table, dialogues and practice exercises. These were previously
exportable only as Markdown or DOCX, which is something a teacher reads rather
than something a child writes on.

In the **Lesson** tab you can either write a new one or print a lesson saved
earlier. Printing a saved lesson makes no model call at all.

The answer key costs nothing extra: the model is already told to end a lesson
with a **Solutions** section, so the sheet is everything before it and the key
is the section itself. They are the same document cut in half, which is why
they cannot disagree. A lesson with no Solutions section reports that there is
no key rather than printing the sheet again under an "Answer key" heading.

Markdown renders to LaTeX directly — headings, bold, italic, bullet and
numbered lists, and tables with wrapped cells — and numbered practice exercises
get writing space beneath them.

### Sheet types

`homework` and `challenge` print questions with writing space. `test` and `exam`
add marks in the right margin, a per-section subtotal and a total box — which is
what makes a test a test. The math tutor will borrow the mark scheme later.

Typesetting is shared with the math tutor
([`researcher/latex.py`](src/researcher/latex.py)), so both produce the same
paper and a fix lands in both. The preview is the real PDF in an iframe: what
you see is what prints.

### A seed is a sheet

Only the recipe is stored — language, level, kind, sections, seed — so a test
printed in March reprints identically in June and its key cannot drift.

### Not covered

The tutor converses in ten languages; sheets exist for three. Asking for a sheet
in another language returns a clear refusal rather than a worse sheet.
Pronunciation and diction stay on screen, where the markings are the point.

## File Ingestion & Export

### Uploading Files

Click the **📎** button (or drag files) to attach files to your message. Supported formats:

| Format | Extensions | Extraction |
|--------|-----------|------------|
| Plain text | `.txt`, `.md`, `.csv`, `.json` | Direct read |
| PDF | `.pdf` | Page-by-page text extraction (PyMuPDF), with OCR fallback for scanned pages |
| Images | `.png`, `.jpg`, `.jpeg`, `.webp`, `.tif`, `.tiff`, `.bmp`, `.gif` | OCR via an Ollama vision model |
| Word | `.docx` | Paragraphs + tables |
| Excel | `.xlsx` | All sheets, rows as tab-separated values |

#### Scanned PDFs and images (OCR)

Upload a scanned PDF or a photo and the rest is automatic:

```
upload → OCR at ingest → transcription downloads as a DOCX
```

No agent, no chat turn, no LLM reasoning — the transcription is written straight
to a document. Asking the agent to "OCR this document" instead makes it
*re-generate* text that already exists, which is far slower and can reword the
content. Set `OCR_AUTO_EXPORT=none` if you would rather download it manually from
the Files panel, which also offers ODT.

A PDF with no text layer extracts to nothing, so pages without text are rendered
to PNG and transcribed by an Ollama **vision** model. Ollama models cannot read a
PDF directly — they accept images only — so the app renders and downscales each
page itself. A PDF that already has a text layer is never sent to OCR.

```bash
ollama pull deepseek-ocr:3b     # then, in .env:
OCR_MODEL=deepseek-ocr:3b
```

Measured on a one-page scanned report: `deepseek-ocr:3b` returned a complete
transcription in ~3 s; `qwen3-vl:8b` took ~16 s and truncated. Note that
`OCR_MAX_DIM` defaults to 1024 because `qwen3-vl` returns an **empty** string on
noticeably larger images — raise it only after testing your own model.

- **Max file size**: 10 MB per file
- **Max files per user**: 50
- Files are stored per-user with full isolation
- Extracted text is passed to the model as context alongside your message

### Managing Files

Click **📁 Files** in the header to open the files panel — view and delete your uploaded files.

### Exporting Responses

Every assistant response has an **⬇️ Export** dropdown with four formats:

| Format | What you get |
|--------|--------------|
| **MD** | Raw markdown as-is |
| **TXT** | Markdown stripped (no `**`, `*`, backticks, heading markers) |
| **DOCX** | Word document with headings, bold/italic, tables, bullet lists, code blocks, math blocks |
| **XLSX** | Markdown tables extracted into separate Excel sheets with bold headers |

## Project Structure

```
src/researcher/
├── main.py              # FastAPI app — all endpoints
├── auth.py              # Authentication (JWT, bcrypt, invite code, rate limiting)
├── ingestion.py         # File upload, text extraction, DOCX/XLSX export
├── crew.py              # CrewAI agent, tasks, tools (search, image gen)
├── tts.py               # Text-to-speech (edge-tts, multilingual voice selection)
├── static/
│   └── index.html       # Browser chat UI (served to client)
├── data/
│   ├── sessions.db      # SQLite storage (sessions, users, files — auto-created)
│   └── uploads/         # Uploaded file originals (per-user directories)
└── config/
    ├── agents.yaml      # Agent configuration
    └── tasks.yaml       # Task configuration
docs/
└── HF_zimage_clean_download.txt  # ZImage model setup & troubleshooting
tests/
├── conftest.py          # Shared fixtures, --integration flag
├── test_api.py          # Unit tests (mocked dependencies)
├── test_crew.py         # Unit tests for crew logic
├── test_zimage.py       # Integration test: ZImage image generation
├── test_sdimage.py      # Integration test: Stable Diffusion image generation
└── ...
```

## Image Generation

The agent can generate AI images via the `GenerateAIImage` tool. Two backends are supported:

### ZImage (recommended for quality)

Uses the [Z-Image-Turbo](https://huggingface.co/mrfakename/Z-Image-Turbo) pipeline from diffusers.

```bash
# .env
IMAGE_BACKEND=zimage
ZIMAGE_MODEL=mrfakename/Z-Image-Turbo
HUGGINGFACE_TOKEN=hf_your_token_here
TRANSFORMERS_OFFLINE=1          # after first download
ZIMAGE_WIDTH=512                # optional, default 512
ZIMAGE_HEIGHT=512               # optional, default 512
```

### Stable Diffusion

Uses Stable Diffusion v1.5 with CPU offload.

```bash
# .env
IMAGE_BACKEND=sd
SD_MODEL=stable-diffusion-v1-5/stable-diffusion-v1-5
```

### GPU Memory Management

With a 16 GB GPU shared between the LLM (Ollama) and image generation:

1. **Before image generation**: The Ollama model is unloaded (`keep_alive=0`) and the system waits for VRAM to free
2. **During generation**: The image pipeline loads onto GPU, generates the image, saves to disk
3. **After generation**: The pipeline is destroyed and moved off GPU, then the Ollama model is reloaded
4. **After request completes**: If image generation was used, the Ollama model is unloaded to fully free VRAM

For text-only queries, the Ollama model stays resident in VRAM for fast responses.

See [docs/HF_zimage_clean_download.txt](docs/HF_zimage_clean_download.txt) for detailed model download and troubleshooting instructions.

## Music Composer

The composer writes in **ABC notation**, which music21 converts to MusicXML.

### Why ABC and not JSON

The composer originally emitted note-by-note JSON and needed ~500 lines of regex
repair to survive its own output. ABC replaced it for two reasons, the second
mattering more:

- **It parses.** The same local model that produced broken JSON writes valid ABC
  first time.
- **It carries structure.** JSON encodes only pitches and durations, so the model
  had no way to express repeats, sections or phrasing. ABC does, is ~38% of the
  tokens, and appears in training data as *music* rather than as arbitrary data.

### The specialist model

`COMPOSER_MODEL` is independent of `MODEL`, exactly as `OCR_MODEL` is, and is
called **directly** rather than through CrewAI: it has no tools and is not an
instruction-following chat model, so a planning loop would add minutes and a
format it does not speak.

```bash
ollama pull SeekerThe3rd/chatmusician
# .env
COMPOSER_MODEL=SeekerThe3rd/chatmusician:latest
```

Measured against the general chat model on the same jazz task:

| | ChatMusician | general model |
|---|---|---|
| distinct rhythms | **7** | 2 |
| distinct pitches | **14** | 8 |
| pitch classes used | **11/12** | 7/12 |

Two rhythmic values is what the ear hears as an exercise. 11 of 12 pitch classes
is real jazz vocabulary — blue notes and approach tones the model has actually
heard, rather than a prompt asking for them.

Leave `COMPOSER_MODEL` empty to compose with the general model instead.

### Length: short form vs assembled

A single model call holds together for 8–16 bars and then repeats or drifts, so
longer pieces are **assembled, not requested**.

| Score tab setting | Behaviour |
|---|---|
| **Length: short (bars)** | The model writes **Measures** bars directly |
| **Length: 1–3 min** | An 8-bar seed is written in the model's idiom, then developed across a recurring form with extended harmony |

Assembled pieces use genre-appropriate chromatic harmony — `IMaj7 vi7 ii7 V9`,
`V7/ii`, `iv7 bVII7`, `ii65 V7 I V7/vi`, `IV Ger+65 V I` — every figure realised
by music21 rather than guessed.

Measured: 64 bars, 2.6 minutes, 8 sections, in about 5 seconds.

Recurring material is deliberate: three minutes of entirely *new* material is
three minutes of mush. But see `TODO_Claude.md` for the measured limits — only
**3 distinct rhythms** currently reach the long form, which is why extended
pieces still sound repetitive.

### Output

| Action | Result |
|---|---|
| Download MusicXML | Opens in MuseScore and any notation editor |
| MIDI | Plays in the browser or any DAW |
| Save | Stored per session with its score, listed in the **Compositions** tab |

### How a composition is made

Style comes from the interface — genre, key, time signature and tempo — plus the
matching block from `knowledge/genres.yaml` (form, structure, rhythm, voicing,
melody style; ~1.6 KB of real guidance for jazz alone).

The prompt then enforces a **method** rather than asking for notes: invent a
two-bar motif, develop it by sequence/inversion/fragmentation/augmentation, build
antecedent–consequent phrases, place one high point about two thirds through, and
cadence on the tonic.

That alone is not enough, so each composition goes through:

1. **Generate** ABC.
2. **Validate** — `validate_abc()` measures bar lengths, the longest run in one
   direction, and the proportion of stepwise motion. These catch faults music21
   accepts silently: a 3/4 piece whose bars each hold 2 beats parses fine and
   plays wrong.
3. **Correct** — faults are fed back *specifically* ("a 12-note unbroken run in
   one direction — that is a scale, not a melody"), up to
   `COMPOSER_REFINE_ROUNDS` times.
4. **Keep the best attempt**, not the last — a correction pass can make things
   worse, and does.
5. **Re-bar deterministically.** The model never counted bar durations correctly
   in any attempt, even when given the arithmetic. So the code discards its
   barlines, re-bars at the correct metre and ties notes across barlines.

Measured on the same prompt, before and after this pipeline:

| | Before | After |
|---|---|---|
| Longest scale run | 12 notes | 6 |
| Stepwise motion | 90% | 61% |
| Bar lengths | wrong, silently | every bar correct |
| Output | often unparseable | valid MusicXML every time |

### Known limits

- Quality is bounded by the local model. It writes *structured* music, not good
  music; the validator enforces a floor, not excellence.
- `report["ok"]` reflects the **pre-repair ABC**, so it can read `false` even
  though the delivered MusicXML has correct bars.
- Dynamics, articulation and mid-score key changes are still unimplemented.

## Testing

### Unit Tests

Run with mocked dependencies (no server required):

```bash
pytest tests/ -v
```

### Integration Tests

Require a running server:

```bash
# Start the server first
uvicorn researcher.main:app --host 0.0.0.0 --port 8000

# In another terminal — test ZImage backend (IMAGE_BACKEND=zimage)
pytest tests/test_zimage.py -v -s --integration

# Test Stable Diffusion backend (IMAGE_BACKEND=sd)
pytest tests/test_sdimage.py -v -s --integration
```

Integration tests authenticate against the running server, send image-generation prompts via `/chat`, and verify the SSE response contains valid image tags.

Test credentials default to environment variables `TEST_USER` / `TEST_PASS`, or can be overridden:

```bash
TEST_USER=myuser TEST_PASS=mypass pytest tests/test_zimage.py -v -s --integration
```
