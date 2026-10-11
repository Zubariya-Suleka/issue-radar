from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from gateway.config import Settings
from gateway.main import create_app


@pytest.fixture
def settings() -> Settings:
    # _env_file=None: tests must not depend on a developer's local .env
    return Settings(_env_file=None, backend="fake")


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    # Using TestClient as a context manager runs the app's startup/shutdown (lifespan).
    with TestClient(create_app(settings)) as c:
        yield c
