"""App factory. Run with: uv run uvicorn gateway.main:create_app --factory"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from gateway.api.routes import router
from gateway.backends.base import ModelBackend
from gateway.backends.factory import build_backend
from gateway.backends.serial import BackendBusyError, SerialBackend
from gateway.config import Settings


async def backend_busy_handler(request: Request, exc: Exception) -> JSONResponse:
    # OpenAI-style error body so OpenAI clients surface the message and can retry.
    return JSONResponse(
        status_code=503,
        content={"error": {"message": str(exc), "type": "server_busy"}},
        headers={"Retry-After": "5"},
    )


def create_app(settings: Settings | None = None, backend: ModelBackend | None = None) -> FastAPI:
    """`backend` lets tests inject a specific backend instead of building one from settings."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Runs once at startup, before the first request: load the model here, never per request.
        app.state.settings = settings
        app.state.backend = SerialBackend(
            backend or build_backend(settings), queue_timeout_s=settings.queue_timeout_s
        )
        yield

    app = FastAPI(title="issue-radar gateway", lifespan=lifespan)
    app.include_router(router)
    app.add_exception_handler(BackendBusyError, backend_busy_handler)
    return app
