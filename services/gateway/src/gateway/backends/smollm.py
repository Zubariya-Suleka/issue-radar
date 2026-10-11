"""Local SmolLM2 backend via Hugging Face transformers.

YOUR PART: implement this class so it satisfies ModelBackend (see base.py).
Install the heavy dependencies first (from the repo root):  uv sync --all-packages --extra local

Questions your implementation has to answer (decide, then note anything non-obvious
for DECISIONS.md):

1. Loading: what happens once in __init__, and what must never happen per request?
2. Prompt: the model is chat-tuned. How do `request.messages` become input ids so the
   model sees the format it was trained on? (Look at the tokenizer's chat template, and
   at what `add_generation_prompt` does.)
3. Decoding: `generate()` returns prompt + completion ids. How do you return only the
   completion text, and count prompt_tokens / completion_tokens correctly?
4. finish_reason: how can you tell "stopped naturally" from "hit max_tokens"?
5. temperature: what should temperature == 0.0 map to? (Sampling with temperature 0
   is not the same thing as greedy decoding in transformers.)
6. Blocking: model.generate() is CPU-bound and blocks for seconds. `generate` here is
   async; how do you keep the event loop free while it runs? (D-011 explains why.)
7. Memory/speed on CPU: gradients are not needed at inference; how do you tell torch?

Tests for this backend should be marked slow and skipped unless the model is available,
so the default test run stays fast.
"""

from pathlib import Path

from gateway.backends.base import GenerationRequest, GenerationResult


class SmolLMBackend:
    def __init__(self, model_id: str, cache_dir: Path | None = None) -> None:
        self.name = model_id
        raise NotImplementedError("Load the tokenizer and model here, once.")

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError
