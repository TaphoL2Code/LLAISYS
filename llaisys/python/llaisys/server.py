"""Multi-user inference server with OpenAI-compatible API.

Supports:
- Async request submission with immediate return
- Request status polling
- SSE streaming responses
- Concurrent request processing via model pool
- Health and stats endpoints

Usage:
    python -m llaisys.server --model ./Qwen2-0.5B --tokenizer ./Qwen2-0.5B --port 8080 --pool-size 4
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
from typing import Optional, List, Dict, Union

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from .request_queue import RequestQueue, InferenceRequest, RequestState
from .model_pool import ModelPool
from .batch_scheduler import BatchScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLaiSys Multi-User Chat API", version="2.0.0")

# Global references
_scheduler: Optional[BatchScheduler] = None
_queue: Optional[RequestQueue] = None
_pool: Optional[ModelPool] = None
_tokenizer = None
_model_path: str = ""
_model_class: str = "qwen2"


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "llaisys"
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


class RequestStatus(BaseModel):
    request_id: str
    state: str
    created_at: float
    elapsed_seconds: float
    generated_text: str = ""
    generated_tokens: int = 0
    error_message: Optional[str] = None


class QueueStats(BaseModel):
    pending: int
    running: int
    total_queued: int
    total_finished: int
    pool_total: int
    pool_active: int
    pool_idle: int


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def _create_model_factory(model_path: str, model_class: str, device):
    """Create a factory function that produces new model instances."""

    def factory():
        if model_class == "llama":
            from .models.llama import Llama
            return Llama(model_path, device)
        else:
            from .models.qwen2 import Qwen2
            return Qwen2(model_path, device)

    return factory


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check endpoint."""
    if _scheduler is None:
        return {"status": "starting", "model_loaded": False}
    return {
        "status": "ok",
        "model_loaded": True,
        "scheduler_running": _scheduler.is_running,
    }


@app.get("/v1/stats")
async def stats() -> QueueStats:
    """Get queue and pool statistics."""
    if _scheduler is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    s = _scheduler.stats()
    return QueueStats(
        pending=s["queue"]["pending"],
        running=s["queue"]["running"],
        total_queued=s["queue"]["total_queued"],
        total_finished=s["queue"]["total_finished"],
        pool_total=s["pool"]["total_instances"],
        pool_active=s["pool"]["active_instances"],
        pool_idle=s["pool"]["idle_instances"],
    )


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """Submit a chat completion request.

    - If stream=True: returns SSE stream (blocks until completion)
    - If stream=False: returns immediately with request_id, use polling
    """
    if _scheduler is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    request = InferenceRequest(
        messages=messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
    )

    if req.stream:
        return StreamingResponse(
            _stream_response(request),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming: enqueue and return request_id immediately
        if not _queue.enqueue(request):
            raise HTTPException(status_code=429, detail="Request queue full")

        return JSONResponse(
            status_code=202,
            content={
                "request_id": request.request_id,
                "state": request.state.value,
                "message": "Request accepted. Poll /v1/requests/{request_id} for results.",
            },
        )


@app.get("/v1/requests/{request_id}")
async def get_request_status(request_id: str):
    """Poll for request status and results."""
    if _queue is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    req = _queue.get_status(request_id)
    if req is None:
        raise HTTPException(status_code=404, detail="Request not found")

    response = RequestStatus(
        request_id=req.request_id,
        state=req.state.value,
        created_at=req.created_at,
        elapsed_seconds=req.elapsed_seconds,
        generated_text=req.generated_text if req.state == RequestState.DONE else "",
        generated_tokens=len(req.generated_tokens),
        error_message=req.error_message,
    )

    if req.state == RequestState.DONE:
        return JSONResponse(
            status_code=200,
            content={
                **response.model_dump(),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": req.generated_text},
                        "finish_reason": "stop",
                    }
                ] if req.generated_text else [],
                "usage": {
                    "prompt_tokens": req.prompt_tokens,
                    "completion_tokens": req.completion_tokens,
                    "total_tokens": req.prompt_tokens + req.completion_tokens,
                },
            },
        )
    elif req.state == RequestState.ERROR:
        return JSONResponse(status_code=500, content=response.model_dump())
    else:
        return JSONResponse(status_code=202, content=response.model_dump())


@app.delete("/v1/requests/{request_id}")
async def cancel_request(request_id: str):
    """Cancel a pending request."""
    if _queue is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    if _queue.cancel(request_id):
        return {"status": "cancelled", "request_id": request_id}
    raise HTTPException(status_code=404, detail="Request not found or already processing")


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "llaisys",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llaisys",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Streaming Response Generator
# ---------------------------------------------------------------------------

async def _stream_response(request: InferenceRequest):
    """Generate SSE streaming response for a request.

    The request is enqueued and processed by the scheduler.
    A callback is used to stream tokens back to the client.
    """
    import asyncio

    completion_id = f"chatcmpl-{request.request_id}"
    created = int(time.time())

    # Queue to collect tokens from the callback
    token_queue: asyncio.Queue = asyncio.Queue()

    def on_text(text: str):
        """Callback called by scheduler for each new token."""
        try:
            # Schedule the token to be sent via the async queue
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(token_queue.put_nowait, text)
        except Exception:
            pass

    request._text_callback = on_text

    # Enqueue for processing
    if not _queue.enqueue(request):
        error_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "llaisys",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "[Error: Queue full]"},
                    "finish_reason": "error",
                }
            ],
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Send role first
    first_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model="llaisys",
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content=""),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # Stream tokens
    previous_text = ""
    try:
        while request.state not in (RequestState.DONE, RequestState.ERROR):
            try:
                # Wait for next token with timeout
                text = await asyncio.wait_for(token_queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Check if request is done
                if request.state in (RequestState.DONE, RequestState.ERROR):
                    break
                continue

            new_content = text[len(previous_text):]
            previous_text = text

            if new_content:
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model="llaisys",
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(content=new_content),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

        # Send final chunk
        finish_reason = "stop" if request.state == RequestState.DONE else "error"
        final_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model="llaisys",
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(content=""),
                    finish_reason=finish_reason,
                )
            ],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"

    except asyncio.CancelledError:
        _queue.cancel(request.request_id)

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Server Lifecycle
# ---------------------------------------------------------------------------

def init_server(
    model_path: str,
    tokenizer_path: str,
    device: int = 0,  # 0=CPU, 1=NVIDIA
    pool_size: int = 4,
    model_class: str = "qwen2",
):
    """Initialize the multi-user inference server.

    Args:
        model_path: Path to model directory (with config.json and .safetensors).
        tokenizer_path: Path to tokenizer directory.
        device: Device type (0=CPU, 1=NVIDIA/GPU).
        pool_size: Number of model instances in the pool (max concurrent requests).
        model_class: Model class name ("qwen2" or "llama").
    """
    global _scheduler, _queue, _pool, _tokenizer, _model_path, _model_class

    from .libllaisys import DeviceType

    device_type = DeviceType.NVIDIA if device == 1 else DeviceType.CPU

    logger.info("Loading tokenizer from %s", tokenizer_path)
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    logger.info("Creating model pool (size=%d) from %s", pool_size, model_path)
    _model_path = model_path
    _model_class = model_class

    factory = _create_model_factory(model_path, model_class, device_type)
    _pool = ModelPool(factory, pool_size=pool_size)

    logger.info("Creating request queue")
    _queue = RequestQueue(max_size=100)

    logger.info("Starting batch scheduler")
    _scheduler = BatchScheduler(
        request_queue=_queue,
        model_pool=_pool,
        tokenizer=_tokenizer,
        poll_interval=0.1,
    )
    _scheduler.start()

    logger.info("Server initialized successfully")


def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    model_path: str = "",
    tokenizer_path: str = "",
    device: int = 0,
    pool_size: int = 4,
    model_class: str = "qwen2",
):
    """Run the multi-user inference server.

    Args:
        host: Host to bind to.
        port: Port to listen on.
        model_path: Path to model directory.
        tokenizer_path: Path to tokenizer directory.
        device: 0=CPU, 1=NVIDIA/GPU.
        pool_size: Max concurrent requests.
        model_class: "qwen2" or "llama".
    """
    import uvicorn

    init_server(
        model_path=model_path,
        tokenizer_path=tokenizer_path or model_path,
        device=device,
        pool_size=pool_size,
        model_class=model_class,
    )

    logger.info("Starting HTTP server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaiSys Multi-User Inference Server")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer (defaults to model path)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--device", type=int, default=0, choices=[0, 1], help="0=CPU, 1=NVIDIA")
    parser.add_argument("--pool-size", type=int, default=4, help="Max concurrent requests")
    parser.add_argument("--model-class", type=str, default="qwen2", choices=["qwen2", "llama"], help="Model class")
    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        model_path=args.model,
        tokenizer_path=args.tokenizer or args.model,
        device=args.device,
        pool_size=args.pool_size,
        model_class=args.model_class,
    )