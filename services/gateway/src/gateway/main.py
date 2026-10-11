"""App factory. Run with: uv run uvicorn gateway.main:create_app --factory"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from gateway.api.routes import router
from gateway.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Runs once at startup, before the first request; anything expensive goes here.
        app.state.settings = settings
        yield

    app = FastAPI(title="issue-radar gateway", lifespan=lifespan)
    app.include_router(router)
    return app
