from collections import OrderedDict

from gateway.backends.base import GenerationRequest, GenerationResult, ModelBackend
from gateway.request_log import annotate


class CachedBackend:
    """Wraps any backend with an in-memory LRU cache of greedy generations (D-015).

    Only temperature 0 is cached. Greedy decoding is deterministic, so a stored answer is
    exactly what the model would produce again. A sampled answer is one random draw;
    replaying it would silently turn sampling off for every repeat of that prompt.

    Sits outside SerialBackend, so a hit is answered at once instead of queueing for the
    model. Failed generations raise and are not stored.
    """

    def __init__(self, inner: ModelBackend, max_entries: int) -> None:
        self.name = inner.name
        self._inner = inner
        self._max_entries = max_entries
        # GenerationRequest is frozen and hashable (D-010): it is the key, no hashing needed.
        # One backend per cache, so the model name is implied.
        self._entries: OrderedDict[GenerationRequest, GenerationResult] = OrderedDict()

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if request.temperature != 0.0:
            annotate(cache="bypass")
            return await self._inner.generate(request)

        cached = self._entries.get(request)
        if cached is not None:
            self._entries.move_to_end(request)  # most recently used
            annotate(cache="hit")
            return cached

        annotate(cache="miss")
        result = await self._inner.generate(request)
        self._entries[request] = result
        if len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)  # evict the least recently used
        return result

    def __len__(self) -> int:
        return len(self._entries)
