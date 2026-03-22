# Researcher Agent

A CrewAI-powered research agent with a browser-based chat UI, voice dialog, image generation, and session management. Runs locally using Ollama — the **client only needs a web browser**.

## Architecture

| Component | Runs on | What it does |
|-----------|---------|--------------|
| **Ollama + LLM** | Server | Local LLM inference (qwen3.5:9b recommended) |
| **CrewAI + FastAPI** | Server | Agent orchestration, API endpoints, SSE streaming |
| **Whisper** | Server | Speech-to-text (voice input from browser mic) |
| **edge-tts** | Server | Text-to-speech (audio streamed back to browser) |
| **Stable Diffusion** | Server | AI image generation (optional, needs NVIDIA GPU) |
| **Browser UI** | Client | Just a modern web browser — no plugins, no installs |

> **The client computer needs nothing except a web browser with microphone access** (for voice features). All AI processing, TTS, and STT happen on the server.

## Server Requirements

### System Dependencies

Install these on the **server** machine:

| Package | Required | Purpose | Install (example) |
|---------|----------|---------|-------------------|
| [Ollama](https://ollama.com/) | Yes | Local LLM runtime | See ollama.com |
| Python 3.11+ | Yes | Application runtime | `pyenv install 3.12` |
| ffmpeg | Yes | Audio format conversion (Whisper) | `sudo zypper install ffmpeg` / `sudo apt install ffmpeg` |
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
| openai-whisper | Speech-to-text (runs on CPU) |
| edge-tts | Text-to-speech via Microsoft Edge voices (server-side, streams audio to browser) |
| PyJWT | JWT token generation and validation |
| bcrypt | Secure password hashing |

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
  "DELETE FROM sessions WHERE user_id='<user-id>'; DELETE FROM users WHERE id='<user-id>';"

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

## Project Structure

```
src/researcher/
├── main.py              # FastAPI app — all endpoints
├── auth.py              # Authentication (JWT, bcrypt, invite code, rate limiting)
├── crew.py              # CrewAI agent, tasks, tools (search, image gen)
├── tts.py               # Text-to-speech (edge-tts, multilingual voice selection)
├── static/
│   └── index.html       # Browser chat UI (served to client)
├── data/
│   └── sessions.db      # SQLite session storage (auto-created)
└── config/
    ├── agents.yaml      # Agent configuration
    └── tasks.yaml       # Task configuration
```
