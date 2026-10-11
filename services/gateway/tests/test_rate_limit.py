import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from gateway.config import Settings
from gateway.main import create_app
from gateway.rate_limit import RateLimiter

BODY = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_burst_is_allowed_then_requests_are_refused():
    limiter = RateLimiter(per_minute=60, burst=3, clock=Clock())

    assert [limiter.check("a") for _ in range(3)] == [0, 0, 0]
    assert limiter.check("a") == pytest.approx(1.0)  # 60/min: next token in 1 s


def test_tokens_refill_with_time_up_to_the_burst():
    clock = Clock()
    limiter = RateLimiter(per_minute=60, burst=2, clock=clock)
    for _ in range(2):
        limiter.check("a")

    clock.now = 1.0  # one token back
    assert limiter.check("a") == 0
    assert limiter.check("a") > 0

    clock.now = 100.0  # long idle: refills to the burst, not beyond
    assert [limiter.check("a") for _ in range(3)][-1] > 0


def test_clients_have_separate_buckets():
    limiter = RateLimiter(per_minute=60, burst=1, clock=Clock())

    assert limiter.check("a") == 0
    assert limiter.check("b") == 0
    assert limiter.check("a") > 0


@pytest.fixture
def keyed_settings(settings: Settings) -> Settings:
    settings.api_keys = {"rag": "key-rag", "agent": "key-agent"}
    settings.rate_limit_per_minute = 60
    settings.rate_limit_burst = 2
    return settings


def post(client: TestClient, key: str | None = None, body: dict = BODY):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return client.post("/v1/chat/completions", json=body, headers=headers)


@pytest.mark.parametrize("key", [None, "wrong", "key-rag-and-more"])
def test_missing_or_wrong_key_is_401(keyed_settings, key):
    with TestClient(create_app(keyed_settings)) as client:
        response = post(client, key)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_auth_is_checked_before_the_body(keyed_settings):
    with TestClient(create_app(keyed_settings)) as client:
        assert post(client, None, body={"nonsense": True}).status_code == 401


def test_valid_key_is_logged_as_the_client(keyed_settings, read_log):
    with TestClient(create_app(keyed_settings)) as client:
        assert post(client, "key-agent").status_code == 200

    assert read_log()[0]["client"] == "agent"


def test_over_limit_client_gets_429_with_retry_after(keyed_settings, read_log):
    with TestClient(create_app(keyed_settings)) as client:
        statuses = [post(client, "key-rag").status_code for _ in range(3)]
        limited = post(client, "key-rag")
        other_client = post(client, "key-agent")

    assert statuses == [200, 200, 429]
    assert limited.json()["error"]["type"] == "rate_limit_error"
    assert int(limited.headers["retry-after"]) >= 1
    assert other_client.status_code == 200  # one client's burst doesn't block another
    assert read_log()[2]["rate_limited"] is True


def test_without_keys_everyone_is_one_anonymous_client(settings, read_log):
    settings.rate_limit_burst = 1
    with TestClient(create_app(settings)) as client:
        statuses = [post(client).status_code for _ in range(2)]

    assert statuses == [200, 429]
    assert read_log()[0]["client"] == "anonymous"


def test_limit_can_be_disabled(settings):
    settings.rate_limit_per_minute = 0
    settings.rate_limit_burst = 1
    with TestClient(create_app(settings)) as client:
        assert {post(client).status_code for _ in range(5)} == {200}


def test_health_is_not_authenticated_or_limited(keyed_settings):
    with TestClient(create_app(keyed_settings)) as client:
        assert {client.get("/health").status_code for _ in range(5)} == {200}


def test_duplicate_keys_are_rejected_at_startup():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, api_keys={"rag": "same", "agent": "same"})
