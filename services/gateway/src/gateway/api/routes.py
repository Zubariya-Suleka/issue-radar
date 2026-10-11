import time
import uuid

from fastapi import APIRouter, HTTPException, Request

from gateway.api.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Usage,
)
from gateway.backends.base import GenerationRequest, Message

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    return {"status": "ok", "backend": request.app.state.backend.name}


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request) -> ChatCompletionResponse:
    settings = request.app.state.settings
    backend = request.app.state.backend

    max_tokens = body.max_tokens or settings.max_tokens_default
    if max_tokens > settings.max_tokens_cap:
        raise HTTPException(400, f"max_tokens must be <= {settings.max_tokens_cap}")

    # Translate the outer (OpenAI) format into the internal contract.
    gen_request = GenerationRequest(
        messages=tuple(Message(m.role, m.content) for m in body.messages),
        max_tokens=max_tokens,
        temperature=(
            body.temperature if body.temperature is not None else settings.default_temperature
        ),
    )
    result = await backend.generate(gen_request)

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
