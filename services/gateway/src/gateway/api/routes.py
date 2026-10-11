import asyncio
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from gateway.api.deps import client_name, enforce_rate_limit
from gateway.api.errors import ApiError
from gateway.api.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Usage,
)
from gateway.backends.base import GenerationRequest, Message
from gateway.request_log import annotate, prompt_hash

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    return {"status": "ok", "backend": request.app.state.backend.name}


def allowed_client(request: Request, client: Annotated[str, Depends(client_name)]) -> str:
    enforce_rate_limit(request, client)
    return client


# Dependencies run before the body is validated: a caller without a valid key gets 401,
# and an over-limit caller 429, before anything else.
@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    response: Response,
    client: Annotated[str, Depends(allowed_client)],
) -> ChatCompletionResponse:
    settings = request.app.state.settings
    backend = request.app.state.backend

    max_tokens = body.max_tokens or settings.max_tokens_default
    if max_tokens > settings.max_tokens_cap:
        raise ApiError(
            400, f"max_tokens must be <= {settings.max_tokens_cap}", "invalid_request_error"
        )

    # Translate the outer (OpenAI) format into the internal contract.
    gen_request = GenerationRequest(
        messages=tuple(Message(m.role, m.content) for m in body.messages),
        max_tokens=max_tokens,
        temperature=(
            body.temperature if body.temperature is not None else settings.default_temperature
        ),
    )
    # Logged before generating, so a 503 or a crash still records what was asked.
    annotate(
        model=backend.name,
        n_messages=len(gen_request.messages),
        max_tokens=gen_request.max_tokens,
        temperature=gen_request.temperature,
        prompt_sha256=prompt_hash((m.role, m.content) for m in gen_request.messages),
    )
    if settings.log_content:
        annotate(messages=[{"role": m.role, "content": m.content} for m in gen_request.messages])

    # Before the cache: a blocked request must not be answered from it either.
    check = request.app.state.injection_check
    if check is not None:
        score = await asyncio.to_thread(check.score, gen_request.messages)
        flagged = score >= check.threshold
        annotate(injection_score=round(score, 3), injection_flagged=flagged)
        if flagged and check.mode == "block":
            raise ApiError(
                400,
                "Request blocked: possible prompt injection in a user message",
                "invalid_request_error",
                "prompt_injection_detected",
            )
        if flagged:
            response.headers["X-Injection-Flagged"] = "true"

    result = await backend.generate(gen_request)

    annotate(
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        finish_reason=result.finish_reason,
    )
    if settings.log_content:
        annotate(response_text=result.text)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=backend.name,
        choices=[
            Choice(
                index=0,
                message=AssistantMessage(content=result.text),
                finish_reason=result.finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )
