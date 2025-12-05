# Qwen3 Dockerized Chat Inference

This repository provides a minimal FastAPI service for chatting with a Qwen3 model through an OpenAI-compatible API. The service is containerized with Docker for easy deployment.

## Requirements
- Docker
- A Qwen-compatible API endpoint (DashScope compatible) and API key

## Environment variables
- `QWEN_API_KEY` (**required**): API key for the Qwen endpoint.
- `QWEN_BASE_URL` (optional): Base URL for the compatible endpoint. Defaults to `https://dashscope.aliyuncs.com/compatible-mode/v1`.
- `QWEN_MODEL` (optional): Default model name. Defaults to `qwen3-instruct` when not provided in the request body.

## Build and run with Docker
```bash
docker build -t qwen3-chat .
docker run -it --rm -p 8000:8000 -e QWEN_API_KEY="<your_key>" qwen3-chat
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

The response will contain the assistant reply and the model name used.

## Local development (optional)
If you prefer running without Docker, install dependencies and start the server:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
