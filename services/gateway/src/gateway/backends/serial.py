import asyncio

from gateway.backends.base import GenerationRequest, GenerationResult, ModelBackend


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
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=self._queue_timeout_s)
        except TimeoutError:
            raise BackendBusyError(
                f"model busy: waited {self._queue_timeout_s}s for a free slot"
            ) from None
        try:
            return await self._inner.generate(request)
        finally:
            self._lock.release()
