from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from gateway.backends.fake import FakeBackend
from gateway.main import create_app
from gateway.request_log import annotate

BODY = {"model": "x", "messages": [{"role": "user", "content": "hello there"}]}


def test_one_line_per_request_with_usage_and_no_content(client, read_log):
    response = client.post("/v1/chat/completions", json=BODY)

    [line] = read_log()
    assert line["request_id"] == response.headers["x-request-id"]
    assert line["status"] == 200
    assert line["path"] == "/v1/chat/completions"
    assert line["model"] == "fake"
    assert line["prompt_tokens"] == response.json()["usage"]["prompt_tokens"]
    assert line["finish_reason"] == "stop"
    assert line["total_ms"] >= 0
    assert "hello there" not in str(line)


def test_identical_prompts_share_a_hash(client, read_log):
    client.post("/v1/chat/completions", json=BODY)
    client.post("/v1/chat/completions", json=BODY)
    client.post(
        "/v1/chat/completions", json={**BODY, "messages": [{"role": "user", "content": "other"}]}
    )

    first, second, third = (line["prompt_sha256"] for line in read_log())
    assert first == second != third


def test_caller_request_id_is_kept(client, read_log):
    response = client.post("/v1/chat/completions", json=BODY, headers={"X-Request-ID": "rag-42"})

    assert response.headers["x-request-id"] == "rag-42"
    assert read_log()[0]["request_id"] == "rag-42"


def test_unsafe_caller_request_id_is_replaced(client, read_log):
    unsafe = 'evil"\n{"status":200}'
    response = client.post("/v1/chat/completions", json=BODY, headers={"X-Request-ID": unsafe})

    assert response.headers["x-request-id"] != unsafe
    assert len(read_log()) == 1


def test_health_checks_are_not_logged_but_get_a_request_id(client, settings):
    response = client.get("/health")

    assert response.headers["x-request-id"]
    assert not settings.log_file.exists() or settings.log_file.read_text() == ""


def test_validation_errors_are_logged(client, read_log):
    client.post("/v1/chat/completions", json={**BODY, "messages": []})

    [line] = read_log()
    assert line["status"] == 422
    assert "prompt_tokens" not in line


def test_crashes_are_logged_as_500(settings, read_log):
    class Exploding:
        name = "exploding"

        async def generate(self, request):
            raise RuntimeError("boom")

    app = create_app(settings, backend=Exploding())
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/chat/completions", json=BODY)

    assert response.status_code == 500
    [line] = read_log()
    assert line["status"] == 500
    assert line["error"] == "RuntimeError"
    assert line["prompt_sha256"]  # request fields were logged before the crash


def test_content_is_logged_only_when_enabled(settings, read_log):
    settings.log_content = True
    with TestClient(create_app(settings)) as client:
        client.post("/v1/chat/completions", json=BODY)

    [line] = read_log()
    assert line["messages"] == BODY["messages"]
    assert line["response_text"] == "echo: hello there"


def test_queue_and_generation_time_are_logged_separately(settings, read_log):
    with TestClient(create_app(settings, backend=FakeBackend(delay_s=0.05))) as client:
        client.post("/v1/chat/completions", json=BODY)

    [line] = read_log()
    # Margins, not exact values: Windows timers tick every ~15.6 ms, so a 50 ms sleep can
    # measure as 48.
    assert line["queue_ms"] < 20
    assert line["generate_ms"] >= 35
    assert line["total_ms"] >= line["queue_ms"] + line["generate_ms"]


def test_busy_rejection_logs_the_time_spent_waiting(settings, read_log):
    settings.queue_timeout_s = 0.05
    app = create_app(settings, backend=FakeBackend(delay_s=0.3))
    with TestClient(app) as client:
        with ThreadPoolExecutor(2) as pool:
            statuses = sorted(
                f.result().status_code
                for f in [
                    pool.submit(client.post, "/v1/chat/completions", json=BODY) for _ in range(2)
                ]
            )

    assert statuses == [200, 503]
    busy = next(line for line in read_log() if line["status"] == 503)
    assert busy["queue_ms"] >= 50
    assert "generate_ms" not in busy


def test_annotate_outside_a_request_is_a_no_op():
    annotate(anything=1)  # must not raise
