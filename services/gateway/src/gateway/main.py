"""App factory. Run with: uv run uvicorn gateway.main:create_app --factory"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from gateway.api.errors import ApiError, api_error_handler
from gateway.api.routes import router
from gateway.backends.base import GenerationRequest, Message, ModelBackend
from gateway.backends.cached import CachedBackend
from gateway.backends.factory import build_backend
from gateway.backends.serial import BackendBusyError, SerialBackend
from gateway.config import Settings
from gateway.rate_limit import RateLimiter
from gateway.request_log import close_request_log, configure_request_log, log_requests
from gateway.safety.check import build_injection_check


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
        configure_request_log(settings.log_file)
        app.state.settings = settings
        inner = backend or build_backend(settings)
        if settings.warmup:
            # One token touches every layer, so all weights are paged in now rather than
            # during the first real request (46 s in Docker, where weights are read lazily
            # from the G: mount; D-018). Startup and /health wait for it.
            await inner.generate(
                GenerationRequest((Message("user", "hi"),), max_tokens=1, temperature=0.0)
            )
        # Cache outside the queue: a hit never waits for the model.
        model: ModelBackend = SerialBackend(inner, queue_timeout_s=settings.queue_timeout_s)
        if settings.cache_max_entries > 0:
            model = CachedBackend(model, max_entries=settings.cache_max_entries)
        app.state.backend = model
        app.state.rate_limiter = (
            RateLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst)
            if settings.rate_limit_per_minute > 0
            else None
        )
        app.state.injection_check = build_injection_check(settings)
        yield
        close_request_log()

    app = FastAPI(title="issue-radar gateway", lifespan=lifespan)
    app.middleware("http")(log_requests)
    app.include_router(router)
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(BackendBusyError, backend_busy_handler)
    return app
