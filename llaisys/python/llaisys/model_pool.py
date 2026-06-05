"""Model instance pool for multi-user inference.

Provides per-request model instances with independent KV-caches.
Multiple requests can share the same loaded weights while maintaining
separate inference state (KV-cache, sequence length).
"""

import threading
import time
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class ModelInstance:
    """A model instance with its own KV-cache state."""

    model: object  # Qwen2 or Llama model instance
    in_use: bool = False
    last_used: float = 0.0
    request_id: Optional[str] = None


class ModelPool:
    """Pool of model instances for concurrent request processing.

    Each instance has its own KV-cache, allowing multiple requests
    to maintain independent conversation state.

    Usage:
        pool = ModelPool(model_factory, pool_size=4)
        with pool.acquire() as model:
            output = model.generate(inputs, ...)
    """

    def __init__(
        self,
        model_factory: callable,
        pool_size: int = 4,
        idle_timeout: float = 300.0,
    ):
        """
        Args:
            model_factory: Callable that creates a new model instance.
                           Should return a model with generate()/generate_stream() methods.
            pool_size: Maximum number of concurrent model instances.
            idle_timeout: Seconds before an idle instance's KV-cache is reset.
        """
        self._model_factory = model_factory
        self._pool_size = pool_size
        self._idle_timeout = idle_timeout
        self._instances: List[ModelInstance] = []
        self._lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(pool_size)

        # Pre-create one warm instance
        self._warm_up()

    def _warm_up(self):
        """Pre-create the first model instance to avoid cold start."""
        try:
            instance = self._create_instance()
            with self._lock:
                self._instances.append(instance)
        except Exception:
            pass  # Warm-up failure is non-fatal

    def _create_instance(self) -> ModelInstance:
        """Create a new model instance."""
        model = self._model_factory()
        return ModelInstance(model=model)

    def _find_idle_instance(self) -> Optional[ModelInstance]:
        """Find an idle instance, resetting stale ones."""
        now = time.time()
        for inst in self._instances:
            if not inst.in_use:
                # Reset KV-cache if idle too long
                if now - inst.last_used > self._idle_timeout:
                    inst.model.reset_kv()
                return inst
        return None

    def _find_or_create_instance(self) -> ModelInstance:
        """Find an idle instance or create a new one."""
        # Try to find an idle instance
        inst = self._find_idle_instance()
        if inst is not None:
            inst.in_use = True
            inst.last_used = time.time()
            return inst

        # Create new instance if pool not full
        if len(self._instances) < self._pool_size:
            inst = self._create_instance()
            inst.in_use = True
            inst.last_used = time.time()
            self._instances.append(inst)
            return inst

        # Pool is full, should not happen if semaphore is used correctly
        raise RuntimeError("Model pool exhausted")

    def acquire(self, request_id: str) -> ModelInstance:
        """Acquire a model instance for a request. Blocks if pool is full.

        Args:
            request_id: ID of the requesting inference request.

        Returns:
            ModelInstance with the model ready for inference.
        """
        self._semaphore.acquire()
        with self._lock:
            inst = self._find_or_create_instance()
            inst.request_id = request_id
            return inst

    def release(self, instance: ModelInstance):
        """Release a model instance back to the pool.

        Args:
            instance: The ModelInstance to release.
        """
        with self._lock:
            instance.in_use = False
            instance.last_used = time.time()
            instance.request_id = None
        self._semaphore.release()

    def reset_all(self):
        """Reset KV-cache for all instances (e.g., on server restart)."""
        with self._lock:
            for inst in self._instances:
                if not inst.in_use:
                    inst.model.reset_kv()

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(1 for inst in self._instances if inst.in_use)

    @property
    def total_count(self) -> int:
        with self._lock:
            return len(self._instances)

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_instances": len(self._instances),
                "active_instances": sum(1 for i in self._instances if i.in_use),
                "idle_instances": sum(1 for i in self._instances if not i.in_use),
                "pool_size": self._pool_size,
            }