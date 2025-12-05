import os
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: float = 0.7
    model: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    model: str


app = FastAPI(title="Qwen3 Chat Inference", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    api_key = os.environ.get("QWEN_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="QWEN_API_KEY is not configured")

    base_url = os.environ.get(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model_name = request.model or os.environ.get("QWEN_MODEL", "qwen3-instruct")

    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[message.model_dump() for message in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    choice = completion.choices[0].message
    content = choice.content if isinstance(choice.content, str) else "".join(
        block["text"] if isinstance(block, dict) else str(block)
        for block in choice.content
    )

    return ChatResponse(reply=content, model=completion.model)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "qwen3-chat",
        "health": "/health",
        "chat": "/chat",
    }
