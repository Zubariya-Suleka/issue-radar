import asyncio

from gateway.backends.base import GenerationRequest, GenerationResult


class FakeBackend:
    """Deterministic backend for tests and local development. Never loads a model.

    Echoes the last user message. "Tokens" are whitespace-separated words, which is
    enough to exercise max_tokens truncation and usage accounting.
    """

    def __init__(self, name: str = "fake", delay_s: float = 0.0) -> None:
        self.name = name
        self.delay_s = delay_s

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if self.delay_s:
            await asyncio.sleep(self.delay_s)

        last_user = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        words = f"echo: {last_user}".split()
        kept = words[: request.max_tokens]

        return GenerationResult(
            text=" ".join(kept),
            prompt_tokens=sum(len(m.content.split()) for m in request.messages),
            completion_tokens=len(kept),
            finish_reason="length" if len(words) > request.max_tokens else "stop",
        )
