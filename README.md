# Qwen3 Dockerized Chat Inference

This repository provides a minimal FastAPI service for chatting with a locally hosted Qwen model through an OpenAI-compatible chat format. The service downloads a Hugging Face model at startup and serves a simple `/chat` endpoint.

## Requirements
- Docker
- Access to download the selected Qwen model from Hugging Face

## Environment variables
- `QWEN_MODEL` (optional): Model name or local path. Defaults to `Qwen/Qwen2.5-0.5B-Instruct`.

## Build and run with Docker
```bash
docker build -t qwen3-chat .
docker run -it --rm -p 8000:8000 -e QWEN_MODEL="Qwen/Qwen2.5-0.5B-Instruct" qwen3-chat
```

The service exposes:
- `GET /health` for a basic health check
- `POST /chat` for chat completions

## Example request
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Hello, Qwen3!"}
        ],
        "max_tokens": 128,
        "temperature": 0.6
      }'
```

The response contains the assistant reply and the model name used. Set `QWEN_MODEL` to point at any compatible Qwen chat model (local path or Hugging Face repository). If you need GPU acceleration, ensure Docker is configured with GPU support and that the model fits in GPU memory.

## Local development (optional)
If you prefer running without Docker, install dependencies and start the server:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
