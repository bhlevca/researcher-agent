# Researcher Agent

A CrewAI-powered research agent with a browser-based chat UI, voice dialog, image generation, and session management. Runs locally using Ollama — the **client only needs a web browser**.

**Version 2.7.0**

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
| **LTX-Video / Wan** | Server | Text-to-video generation (optional, needs NVIDIA GPU) |
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
| `IMAGE_BACKEND` | Image generation backend: `sd`, `zimage`, or `none` | `sd` |
| `ZIMAGE_MODEL` | HuggingFace repo for ZImage pipeline | `mrfakename/Z-Image-Turbo` |
| `ZIMAGE_WIDTH` | Generated image width (ZImage) | `512` |
| `ZIMAGE_HEIGHT` | Generated image height (ZImage) | `512` |
| `SD_MODEL` | HuggingFace repo for Stable Diffusion | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| `HUGGINGFACE_TOKEN` | HuggingFace token for gated model downloads | *(optional)* |
| `TRANSFORMERS_OFFLINE` | Set to `1` after first model download for offline use | `0` |
| `PROXY_API_KEY` | Bearer token required for `/v1/*` proxy endpoints (empty = open) | *(optional)* |
| `IMAGE_REFINE_ROUNDS` | Image critique-and-retry rounds; 0 disables | `2` |
| `IMAGE_CRITIC_MODEL` | Vision model that inspects generated images | `qwen3-vl:8b` |
| `VIDEO_BACKEND` | Video generator: `ltx`, `wan`, or `none` | `ltx` |
| `OCR_MODEL` | Vision model used to read scans | `deepseek-ocr:3b` |
| `WHISPER_MODEL` | faster-whisper size | `small` |
| `SRS_ALGO` | Tutor scheduler: `fsrs` or `sm2` | `fsrs` |

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
VIDEO_BACKEND=ltx            # ltx | wan | none
```

Weights are ~27 GiB and are **never** downloaded during a chat request — fetch them
once, taking only the diffusers components (the LTX repo also hosts 208 GiB of
standalone checkpoints for other tools):

```bash
hf download Lightricks/LTX-Video \
  --include 'text_encoder/*' --include 'transformer/*' --include 'vae/*' \
  --include 'tokenizer/*' --include 'scheduler/*' --include '*.json'
```

Measured on a 16 GB RTX 5070 Ti: **~100 s** for 704x512, 57 frames at 25 fps.
Peak VRAM is the largest single component (~9.5 GiB for the T5 text encoder), not
the sum, because components load one at a time.

`VIDEO_FPS` is **conditioning, not playback speed** — LTX uses it to decide how much
movement belongs between frames. Generating at one rate and exporting at another
stretches the motion. 25 is LTX's native rate. Memory scales with the *frame count*,
so raising fps shortens the clip that fits.

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
