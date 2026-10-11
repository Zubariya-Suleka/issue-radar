import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gateway.config import Settings
from gateway.main import create_app


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # _env_file=None: tests must not depend on a developer's local .env
    return Settings(_env_file=None, backend="fake", log_file=tmp_path / "gateway.jsonl")


@pytest.fixture
def read_log(settings: Settings) -> Callable[[], list[dict]]:
    """Returns the request-log lines written so far, parsed."""

    def read() -> list[dict]:
        text = settings.log_file.read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines()]

    return read


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    # Using TestClient as a context manager runs the app's startup/shutdown (lifespan).
    with TestClient(create_app(settings)) as c:
        yield c
