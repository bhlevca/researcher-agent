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
# Edit .env with your API key and model choice
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL` | Ollama model (litellm format) | `ollama/qwen3.5:9b` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `SERPER_API_KEY` | Google Serper API key for web search | *(required)* |

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

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Chat UI |
| GET | `/info` | Current model info |
| GET | `/models` | List available Ollama models |
| POST | `/model` | Switch active model |
| POST | `/chat` | SSE-streaming chat with the agent |
| POST | `/ask` | Synchronous single-question endpoint |
| GET | `/sessions` | List saved sessions |
| POST | `/sessions` | Create a session |
| GET | `/sessions/{id}` | Load a session |
| PUT | `/sessions/{id}` | Update a session |
| DELETE | `/sessions/{id}` | Delete a session |
| POST | `/transcribe` | Speech-to-text (Whisper, server-side) |
| GET | `/tts/voices` | List available TTS voices (edge-tts, server-side) |
| POST | `/tts/speak` | Text-to-speech audio (edge-tts, server-side) |

## Project Structure

```
src/researcher/
├── main.py              # FastAPI app — all endpoints
├── crew.py              # CrewAI agent, tasks, tools (search, image gen)
├── static/
│   └── index.html       # Browser chat UI (served to client)
├── data/
│   └── sessions.db      # SQLite session storage (auto-created)
└── config/
    ├── agents.yaml      # Agent configuration
    └── tasks.yaml       # Task configuration
```
