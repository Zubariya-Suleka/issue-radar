from fastapi.testclient import TestClient

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
    response = client.get("/health", headers={"X-Request-ID": "rag-42"})

    assert response.headers["x-request-id"] == "rag-42"
    assert read_log()[0]["request_id"] == "rag-42"


def test_unsafe_caller_request_id_is_replaced(client, read_log):
    response = client.get("/health", headers={"X-Request-ID": 'evil"\n{"status":200}'})

    assert response.headers["x-request-id"] != 'evil"\n{"status":200}'
    assert len(read_log()) == 1


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


def test_annotate_outside_a_request_is_a_no_op():
    annotate(anything=1)  # must not raise
