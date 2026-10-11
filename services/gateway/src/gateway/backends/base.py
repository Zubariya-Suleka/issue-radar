"""The contract between the gateway and whatever produces text (D-010).

These types are internal. The OpenAI request/response format lives only in gateway.api;
backends never see it. Frozen dataclasses with tuples are immutable and hashable, which
the cache will rely on later.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class GenerationRequest:
    messages: tuple[Message, ...]
    max_tokens: int
    temperature: float


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: FinishReason  # "length" means generation hit max_tokens


class ModelBackend(Protocol):
    """Anything with a `name` and an async `generate` is a backend. No inheritance needed."""

    name: str

    async def generate(self, request: GenerationRequest) -> GenerationResult: ...
