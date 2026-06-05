"""FastAPI HTTP server for LLaiSys chatbot with OpenAI-compatible API."""

import argparse
import time
import uuid
from typing import Optional, List, Dict, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="LLaiSys Chat API", version="1.0.0")

# Global model reference
_model = None
_tokenizer = None


# --- Request/Response Models ---

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "llaisys-qwen2"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


# --- API Endpoints ---

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt_tokens = 0  # Approximate

    if req.stream:
        return StreamingResponse(
            _stream_response(req, messages, prompt_tokens),
            media_type="text/event-stream",
        )
    else:
        return _sync_response(req, messages, prompt_tokens)


def _sync_response(req: ChatCompletionRequest, messages: List[dict], prompt_tokens: int):
    """Generate a non-streaming chat completion response."""
    start_time = time.time()

    output_text = _model.chat(
        messages,
        _tokenizer,
        max_new_tokens=req.max_tokens,
        top_k=req.top_k,
        top_p=req.top_p,
        temperature=req.temperature,
        stream=False,
    )

    completion_tokens = len(_tokenizer.encode(output_text)) if output_text else 0
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    return ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=output_text),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def _stream_response(req: ChatCompletionRequest, messages: List[dict], prompt_tokens: int):
    """Generate a streaming chat completion response (SSE format)."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # Send role first
    first_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=req.model,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content=""),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # Generate tokens
    full_text = ""
    for text in _model.chat(
        messages,
        _tokenizer,
        max_new_tokens=req.max_tokens,
        top_k=req.top_k,
        top_p=req.top_p,
        temperature=req.temperature,
        stream=True,
    ):
        new_content = text[len(full_text):]
        full_text = text

        if new_content:
            chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=req.model,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=DeltaMessage(content=new_content),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

    # Send final chunk with finish_reason
    final_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=req.model,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaMessage(content=""),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


def init_model(model_path: str, device_name: str = "cpu"):
    """Initialize the model and tokenizer globally."""
    global _model, _tokenizer

    import llaisys
    from transformers import AutoTokenizer
    from llaisys.libllaisys import DeviceType

    device = DeviceType.CPU if device_name == "cpu" else DeviceType.NVIDIA
    _model = llaisys.models.Qwen2(model_path, device)
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def main():
    parser = argparse.ArgumentParser(description="LLaiSys Chat API Server")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    print(f"Loading model from {args.model} on {args.device}...")
    init_model(args.model, args.device)
    print("Model loaded. Starting server...")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()