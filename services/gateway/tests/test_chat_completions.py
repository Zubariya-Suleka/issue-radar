def chat(client, **overrides):
    body = {"model": "anything", "messages": [{"role": "user", "content": "hello there"}]}
    body.update(overrides)
    return client.post("/v1/chat/completions", json=body)


def test_returns_openai_shaped_completion(client):
    response = chat(client)

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert data["model"] == "fake"
    assert data["choices"][0]["message"] == {"role": "assistant", "content": "echo: hello there"}
    assert data["choices"][0]["finish_reason"] == "stop"


def test_usage_totals_add_up(client):
    usage = chat(client).json()["usage"]

    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_max_tokens_truncates_and_reports_length(client):
    data = chat(client, max_tokens=2).json()

    assert data["choices"][0]["message"]["content"] == "echo: hello"
    assert data["choices"][0]["finish_reason"] == "length"


def test_max_tokens_above_cap_is_rejected(client, settings):
    response = chat(client, max_tokens=settings.max_tokens_cap + 1)

    assert response.status_code == 400


def test_streaming_is_rejected(client):
    assert chat(client, stream=True).status_code == 422


def test_unsupported_fields_are_rejected_not_ignored(client):
    assert chat(client, tools=[{"type": "function"}]).status_code == 422


def test_empty_messages_are_rejected(client):
    assert chat(client, messages=[]).status_code == 422


def test_unknown_role_is_rejected(client):
    assert chat(client, messages=[{"role": "tool", "content": "x"}]).status_code == 422
