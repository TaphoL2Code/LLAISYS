"""Continuous batching scheduler for multi-user inference.

Runs a background worker thread that:
1. Dequeues pending requests from the request queue
2. Acquires model instances from the pool
3. Processes requests (generate tokens)
4. Returns model instances to the pool
5. Marks requests as done

Supports continuous batching: the scheduler can process multiple requests
concurrently by assigning each to a separate model instance.
"""

import threading
import time
import logging
from typing import Optional

from .request_queue import RequestQueue, InferenceRequest, RequestState
from .model_pool import ModelPool

logger = logging.getLogger(__name__)


class BatchScheduler:
    """Continuous batching scheduler for multi-user inference.

    Architecture:
        ┌──────────────┐     ┌──────────────┐
        │ RequestQueue │────▶│   Scheduler  │
        └──────────────┘     └──────┬───────┘
                                    │
                           ┌────────▼───────┐
                           │   ModelPool    │
                           │  (N instances) │
                           └────────────────┘

    Each request gets its own model instance from the pool.
    The scheduler runs in a background thread, continuously:
    - Pulling new requests from the queue
    - Assigning model instances
    - Processing inference
    - Returning results
    """

    def __init__(
        self,
        request_queue: RequestQueue,
        model_pool: ModelPool,
        tokenizer,
        poll_interval: float = 0.1,
    ):
        """
        Args:
            request_queue: Queue of pending inference requests.
            model_pool: Pool of model instances for concurrent processing.
            tokenizer: Tokenizer for encoding/decoding.
            poll_interval: Seconds between polls for new requests.
        """
        self._queue = request_queue
        self._pool = model_pool
        self._tokenizer = tokenizer
        self._poll_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._active_requests: dict = {}  # request_id → (model_instance, thread)

        # Statistics
        self._total_requests_processed = 0
        self._total_tokens_generated = 0
        self._stats_lock = threading.Lock()

    def start(self):
        """Start the scheduler background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="batch-scheduler")
        self._thread.start()
        logger.info("BatchScheduler started (pool_size=%d)", self._pool._pool_size)

    def stop(self, timeout: float = 10.0):
        """Stop the scheduler and wait for active requests to finish."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
            logger.info("BatchScheduler stopped")

    def _run_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                # Try to process pending requests
                self._process_pending()
            except Exception as e:
                logger.error("Scheduler error: %s", e, exc_info=True)

            # Wait before next poll
            time.sleep(self._poll_interval)

    def _process_pending(self):
        """Process pending requests from the queue."""
        while True:
            request = self._queue.dequeue()
            if request is None:
                break  # No more pending requests

            # Process in a new thread so we can handle multiple concurrently
            thread = threading.Thread(
                target=self._process_request,
                args=(request,),
                daemon=True,
                name=f"req-{request.request_id}",
            )
            thread.start()
            self._active_requests[request.request_id] = thread

    def _process_request(self, request: InferenceRequest):
        """Process a single inference request."""
        instance = None
        try:
            # Acquire a model instance from the pool
            instance = self._pool.acquire(request.request_id)
            model = instance.model
            model.reset_kv()  # Fresh KV-cache for this request

            # Encode the prompt
            from .models.chat_format import format_chat_prompt
            prompt = format_chat_prompt(request.messages, add_generation_prompt=True)
            inputs = self._tokenizer.encode(prompt)

            # Generate tokens
            if request._text_callback is not None:
                # Streaming mode: yield tokens via callback
                self._stream_generate(model, inputs, request)
            else:
                # Non-streaming mode: collect all tokens
                self._batch_generate(model, inputs, request)

            self._queue.mark_done(request)

            with self._stats_lock:
                self._total_requests_processed += 1
                self._total_tokens_generated += len(request.generated_tokens)

        except Exception as e:
            logger.error("Request %s failed: %s", request.request_id, e, exc_info=True)
            self._queue.mark_error(request, str(e))

        finally:
            if instance is not None:
                self._pool.release(instance)
            self._active_requests.pop(request.request_id, None)

    def _stream_generate(self, model, inputs: list, request: InferenceRequest):
        """Generate tokens in streaming mode, calling callback for each token."""
        generated_ids = list(inputs)
        for token_id in model.generate_stream(
            inputs,
            max_new_tokens=request.max_tokens,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
        ):
            generated_ids.append(token_id)
            request.generated_tokens.append(token_id)

            # Decode and call callback
            text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            request.generated_text = text
            if request._text_callback:
                request._text_callback(text)

    def _batch_generate(self, model, inputs: list, request: InferenceRequest):
        """Generate all tokens at once (non-streaming)."""
        output_ids = model.generate(
            inputs,
            max_new_tokens=request.max_tokens,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
        )
        request.generated_tokens = output_ids[len(inputs):]
        request.generated_text = self._tokenizer.decode(output_ids, skip_special_tokens=True)

    @property
    def is_running(self) -> bool:
        return self._running

    def stats(self) -> dict:
        with self._stats_lock:
            return {
                "total_requests_processed": self._total_requests_processed,
                "total_tokens_generated": self._total_tokens_generated,
                "active_requests": len(self._active_requests),
                "queue": self._queue.stats(),
                "pool": self._pool.stats(),
            }