"""Request queue and data structures for multi-user inference."""

import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class RequestState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class InferenceRequest:
    """A single inference request from a user."""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: List[Dict[str, str]] = field(default_factory=list)
    max_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 1.0
    top_k: int = 0

    # State tracking
    state: RequestState = RequestState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    # Results
    generated_tokens: List[int] = field(default_factory=list)
    generated_text: str = ""
    error_message: Optional[str] = None

    # Internal: for streaming
    _text_callback: Optional[callable] = field(default=None, repr=False)
    _done_event: Optional[threading.Event] = field(default=None, repr=False)

    @property
    def prompt_tokens(self) -> int:
        """Estimate prompt token count from messages."""
        return sum(len(m["content"]) // 3 for m in self.messages)

    @property
    def completion_tokens(self) -> int:
        return len(self.generated_tokens)

    @property
    def elapsed_seconds(self) -> float:
        if self.finished_at:
            return self.finished_at - self.created_at
        return time.time() - self.created_at

    @property
    def ttft(self) -> Optional[float]:
        """Time to first token."""
        if self.started_at:
            return self.started_at - self.created_at
        return None


class RequestQueue:
    """Thread-safe request queue for multi-user inference.

    Supports:
    - Enqueue new requests
    - Dequeue for processing
    - Query request status
    - Cancel pending requests
    """

    def __init__(self, max_size: int = 100):
        self._queue: List[InferenceRequest] = []
        self._lock = threading.Lock()
        self._max_size = max_size
        self._finished: Dict[str, InferenceRequest] = {}  # Keep finished requests for polling

    def enqueue(self, request: InferenceRequest) -> bool:
        """Add a request to the queue. Returns False if queue is full."""
        with self._lock:
            if len(self._queue) >= self._max_size:
                return False
            self._queue.append(request)
            return True

    def dequeue(self) -> Optional[InferenceRequest]:
        """Get the next pending request."""
        with self._lock:
            for i, req in enumerate(self._queue):
                if req.state == RequestState.PENDING:
                    req.state = RequestState.RUNNING
                    req.started_at = time.time()
                    return req
            return None

    def mark_done(self, request: InferenceRequest):
        """Mark a request as completed."""
        with self._lock:
            request.state = RequestState.DONE
            request.finished_at = time.time()
            if request in self._queue:
                self._queue.remove(request)
            self._finished[request.request_id] = request

    def mark_error(self, request: InferenceRequest, error: str):
        """Mark a request as failed."""
        with self._lock:
            request.state = RequestState.ERROR
            request.error_message = error
            request.finished_at = time.time()
            if request in self._queue:
                self._queue.remove(request)
            self._finished[request.request_id] = request

    def get_status(self, request_id: str) -> Optional[InferenceRequest]:
        """Get the status of a request by ID."""
        with self._lock:
            for req in self._queue:
                if req.request_id == request_id:
                    return req
            return self._finished.get(request_id)

    def cancel(self, request_id: str) -> bool:
        """Cancel a pending request."""
        with self._lock:
            for i, req in enumerate(self._queue):
                if req.request_id == request_id and req.state == RequestState.PENDING:
                    self._queue.pop(i)
                    return True
            return False

    @property
    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._queue if r.state == RequestState.PENDING)

    @property
    def running_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._queue if r.state == RequestState.RUNNING)

    @property
    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    def stats(self) -> dict:
        """Get queue statistics."""
        with self._lock:
            return {
                "pending": sum(1 for r in self._queue if r.state == RequestState.PENDING),
                "running": sum(1 for r in self._queue if r.state == RequestState.RUNNING),
                "total_queued": len(self._queue),
                "total_finished": len(self._finished),
            }