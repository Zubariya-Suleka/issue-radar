import pytest
from fastapi.testclient import TestClient

from gateway.backends.base import GenerationRequest, GenerationResult, Message
from gateway.backends.cached import CachedBackend
from gateway.main import create_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def request(text: str = "hi", temperature: float = 0.0, max_tokens: int = 8) -> GenerationRequest:
    return GenerationRequest((Message("user", text),), max_tokens, temperature)


class Counting:
    """Returns a distinct result per call, so a cached answer is distinguishable from a new one."""

    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        return GenerationResult(f"answer {self.calls}", 1, 1, "stop")


async def test_repeat_greedy_request_is_served_from_cache():
    inner = Counting()
    cache = CachedBackend(inner, max_entries=10)

    first = await cache.generate(request())
    second = await cache.generate(request())

    assert second == first
    assert inner.calls == 1


async def test_any_difference_in_the_request_is_a_different_entry():
    inner = Counting()
    cache = CachedBackend(inner, max_entries=10)

    for r in (request("hi"), request("hi", max_tokens=9), request("other")):
        await cache.generate(r)

    assert inner.calls == 3


async def test_sampled_requests_are_never_cached():
    inner = Counting()
    cache = CachedBackend(inner, max_entries=10)

    await cache.generate(request(temperature=0.7))
    await cache.generate(request(temperature=0.7))

    assert inner.calls == 2
    assert len(cache) == 0


async def test_least_recently_used_entry_is_evicted():
    inner = Counting()
    cache = CachedBackend(inner, max_entries=2)

    await cache.generate(request("a"))
    await cache.generate(request("b"))
    await cache.generate(request("a"))  # touch "a": "b" is now least recently used
    await cache.generate(request("c"))  # evicts "b"
    calls_before = inner.calls
    await cache.generate(request("a"))
    await cache.generate(request("b"))

    assert inner.calls == calls_before + 1  # "a" was a hit, "b" had to be regenerated


async def test_failures_are_not_cached():
    class FailsOnce:
        name = "flaky"

        def __init__(self) -> None:
            self.failed = False

        async def generate(self, request):
            if not self.failed:
                self.failed = True
                raise RuntimeError("boom")
            return GenerationResult("ok", 1, 1, "stop")

    cache = CachedBackend(FailsOnce(), max_entries=10)

    with pytest.raises(RuntimeError):
        await cache.generate(request())
    assert (await cache.generate(request())).text == "ok"


def test_hits_skip_the_queue_and_are_logged(settings, read_log):
    body = {"model": "x", "messages": [{"role": "user", "content": "hello"}]}
    with TestClient(create_app(settings, backend=Counting())) as client:
        first = client.post("/v1/chat/completions", json=body).json()
        second = client.post("/v1/chat/completions", json=body).json()

    assert second["choices"] == first["choices"]
    miss, hit = read_log()
    assert (miss["cache"], hit["cache"]) == ("miss", "hit")
    assert "queue_ms" in miss
    assert "queue_ms" not in hit  # never reached SerialBackend


def test_cache_can_be_disabled(settings):
    settings.cache_max_entries = 0
    inner = Counting()
    body = {"model": "x", "messages": [{"role": "user", "content": "hello"}]}
    with TestClient(create_app(settings, backend=inner)) as client:
        client.post("/v1/chat/completions", json=body)
        client.post("/v1/chat/completions", json=body)

    assert inner.calls == 2
