import asyncio

import pytest
from fastapi.testclient import TestClient

from gateway.backends.base import GenerationRequest, GenerationResult, Message
from gateway.backends.fake import FakeBackend
from gateway.backends.serial import BackendBusyError, SerialBackend
from gateway.main import create_app

pytestmark = pytest.mark.anyio

REQUEST = GenerationRequest(messages=(Message("user", "hi"),), max_tokens=8, temperature=0.0)


@pytest.fixture
def anyio_backend():
    return "asyncio"


class ConcurrencyProbe:
    """Records the highest number of generations that were ever running at once."""

    name = "probe"

    def __init__(self) -> None:
        self.running = 0
        self.peak = 0

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0.01)
        self.running -= 1
        return GenerationResult("ok", 1, 1, "stop")


async def test_generations_run_one_at_a_time():
    probe = ConcurrencyProbe()
    backend = SerialBackend(probe, queue_timeout_s=1.0)

    await asyncio.gather(*(backend.generate(REQUEST) for _ in range(5)))

    assert probe.peak == 1


async def test_waiting_past_the_timeout_raises_busy():
    backend = SerialBackend(FakeBackend(delay_s=0.2), queue_timeout_s=0.05)

    results = await asyncio.gather(
        backend.generate(REQUEST), backend.generate(REQUEST), return_exceptions=True
    )

    assert isinstance(results[0], GenerationResult)
    assert isinstance(results[1], BackendBusyError)


async def test_lock_is_released_when_the_inner_backend_fails():
    class Exploding:
        name = "exploding"

        async def generate(self, request):
            raise RuntimeError("boom")

    backend = SerialBackend(Exploding(), queue_timeout_s=0.05)

    for _ in range(2):  # a leaked lock would turn the second call into BackendBusyError
        with pytest.raises(RuntimeError):
            await backend.generate(REQUEST)


def test_busy_backend_maps_to_503_with_openai_error_body(settings):
    class AlwaysBusy:
        name = "busy"

        async def generate(self, request):
            raise BackendBusyError("model busy")

    with TestClient(create_app(settings, backend=AlwaysBusy())) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    assert response.json()["error"]["type"] == "server_busy"
