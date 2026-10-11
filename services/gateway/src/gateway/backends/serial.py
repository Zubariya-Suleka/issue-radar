import asyncio
import time

from gateway.backends.base import GenerationRequest, GenerationResult, ModelBackend
from gateway.request_log import annotate, elapsed_ms


class BackendBusyError(Exception):
    """A request waited longer than the queue timeout for its turn on the model."""


class SerialBackend:
    """Wraps any backend so only one generation runs at a time (D-011).

    On a 4-core CPU, two concurrent generations each run at roughly half speed, so
    concurrency adds latency without adding throughput. Requests queue for the lock and
    give up after `queue_timeout_s`.

    The timeout covers *waiting* only. Once a generation starts it runs to completion:
    CPU-bound work in a worker thread cannot be cancelled midway.
    """

    def __init__(self, inner: ModelBackend, queue_timeout_s: float) -> None:
        self.name = inner.name
        self._inner = inner
        self._queue_timeout_s = queue_timeout_s
        self._lock = asyncio.Lock()

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        # Queue and generation time are logged separately: "slow model" and "long queue"
        # need different fixes, and total latency alone can't tell them apart.
        waiting = time.perf_counter()
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=self._queue_timeout_s)
        except TimeoutError:
            annotate(queue_ms=elapsed_ms(waiting))
            raise BackendBusyError(
                f"model busy: waited {self._queue_timeout_s}s for a free slot"
            ) from None
        annotate(queue_ms=elapsed_ms(waiting))
        generating = time.perf_counter()
        try:
            return await self._inner.generate(request)
        finally:
            annotate(generate_ms=elapsed_ms(generating))
            self._lock.release()
