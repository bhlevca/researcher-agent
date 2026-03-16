# Researcher Agent

A CrewAI-powered research agent that runs locally using Ollama. It uses a reasoning agent with Google (Serper) and DuckDuckGo search to answer questions, exposed via a FastAPI endpoint.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) running locally with `qwen3.5:27b` (or another model)
- A [Serper](https://serper.dev/) API key for Google search

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Copy and configure environment
cp .env.example .env
# Edit .env with your API key and model choice
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL` | Ollama model to use (litellm format) | `ollama/qwen3.5:27b` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `SERPER_API_KEY` | Google Serper API key | *(required)* |

## Usage

```bash
# Start the server
uvicorn researcher.main:app --host 0.0.0.0 --port 8000

# Query the agent
curl "http://localhost:8000/ask?q=What+is+the+weather+in+London"
```

## Project Structure

```
src/researcher/
├── main.py              # FastAPI app
├── crew.py              # CrewAI agent/task/crew definitions
└── config/
    ├── agents.yaml      # Agent configuration
    └── tasks.yaml       # Task configuration
```
