import os
from typing import List, Literal, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def _load_model(model_name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
    except Exception as exc:  # pragma: no cover - startup failure
        raise RuntimeError(f"Failed to load model '{model_name}': {exc}")

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.pad_token_id

    return tokenizer, model


MODEL_NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
TOKENIZER, MODEL = _load_model(MODEL_NAME)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    model_name = request.model or MODEL_NAME
    if model_name != MODEL_NAME:
        raise HTTPException(status_code=400, detail="Dynamic model selection is not supported")

    chat_messages = [message.model_dump() for message in request.messages]
    input_ids = TOKENIZER.apply_chat_template(
        chat_messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(MODEL.device)

    max_new_tokens = request.max_tokens or 512

    try:
        with torch.no_grad():
            output = MODEL.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=request.temperature,
                do_sample=request.temperature > 0,
                pad_token_id=TOKENIZER.eos_token_id,
            )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Generation failed: {exc}")

    generated_tokens = output[0][input_ids.shape[-1] :]
    reply = TOKENIZER.decode(generated_tokens, skip_special_tokens=True)

    return ChatResponse(reply=reply, model=model_name)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "qwen3-chat",
        "health": "/health",
        "chat": "/chat",
    }
