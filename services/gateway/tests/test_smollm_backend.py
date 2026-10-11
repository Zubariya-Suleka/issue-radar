"""Tests against the real SmolLM2 model.

Marked slow and deselected by default (see pyproject addopts); run with `uv run pytest -m slow`.
Skipped if torch is not installed or the model is not in the Hugging Face cache, so these
never trigger a download. Fetch the model with (weights only, not the repo's ~5 GB of ONNX):
    uv run hf download HuggingFaceTB/SmolLM2-360M-Instruct model.safetensors config.json
        generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json
        vocab.json merges.txt
"""

import asyncio

import pytest

from gateway.backends.base import GenerationRequest, Message
from gateway.config import Settings

pytestmark = [pytest.mark.slow, pytest.mark.anyio]

SETTINGS = Settings(_env_file=None)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def backend():
    # Heavy imports live here, not at module level: pytest imports this file even when the
    # slow tests are deselected, and importing torch alone takes seconds.
    pytest.importorskip("torch")
    from huggingface_hub import try_to_load_from_cache

    from gateway.backends.smollm import SmolLMBackend

    cached = try_to_load_from_cache(
        SETTINGS.model_id, "model.safetensors", cache_dir=SETTINGS.model_cache_dir
    )
    if not isinstance(cached, str):
        pytest.skip(f"{SETTINGS.model_id} is not in the Hugging Face cache")
    return SmolLMBackend(SETTINGS.model_id, cache_dir=SETTINGS.model_cache_dir)


def request(text: str, max_tokens: int = 32, temperature: float = 0.0) -> GenerationRequest:
    return GenerationRequest((Message("user", text),), max_tokens, temperature)


async def test_short_answer_stops_naturally(backend):
    result = await backend.generate(request("Reply with only the word: yes"))

    assert result.finish_reason == "stop"
    assert "yes" in result.text.lower()
    assert 0 < result.completion_tokens < 32


async def test_hitting_max_tokens_reports_length(backend):
    result = await backend.generate(request("Write a long essay about rivers.", max_tokens=3))

    assert result.finish_reason == "length"
    assert result.completion_tokens == 3


async def test_text_contains_no_chat_markers(backend):
    result = await backend.generate(request("Say hello."))

    assert "<|im_start|>" not in result.text
    assert "<|im_end|>" not in result.text
    assert "assistant" not in result.text[:12].lower()  # the prompt was sliced off


async def test_prompt_tokens_match_the_chat_template(backend):
    from transformers import AutoTokenizer

    messages = (Message("system", "Be brief."), Message("user", "What is 2 + 2?"))
    tokenizer = AutoTokenizer.from_pretrained(SETTINGS.model_id, cache_dir=SETTINGS.model_cache_dir)
    expected = tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in messages],
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]

    result = await backend.generate(GenerationRequest(messages, 8, 0.0))

    assert result.prompt_tokens == len(expected)


async def test_greedy_decoding_is_repeatable(backend):
    first = await backend.generate(request("Name three fruits."))
    second = await backend.generate(request("Name three fruits."))

    assert first == second


async def test_sampling_runs(backend):
    result = await backend.generate(request("Name a colour.", max_tokens=8, temperature=0.8))

    assert result.completion_tokens > 0


async def test_generation_does_not_block_the_event_loop(backend):
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    await backend.generate(request("Count from 1 to 20.", max_tokens=16))
    task.cancel()

    # 16 tokens take ~1.5 s on CPU. A blocked loop would tick once or not at all.
    assert ticks > 10
