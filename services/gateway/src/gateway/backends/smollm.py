"""Local SmolLM2 backend via Hugging Face transformers (D-013).

Install the heavy dependencies first (from the repo root):  uv sync --all-packages --extra local
"""

import asyncio
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gateway.backends.base import FinishReason, GenerationRequest, GenerationResult


class SmolLMBackend:
    def __init__(self, model_id: str, cache_dir: Path | None = None) -> None:
        self.name = model_id
        # Loading takes seconds and ~0.7 GB: once per process, never per request.
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        # bf16 is how the weights ship; on this CPU it measured as fast as fp32 at half the RAM.
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=cache_dir, dtype=torch.bfloat16
        )
        self._model.eval()
        eos = self._model.generation_config.eos_token_id
        self._eos_ids = frozenset(eos if isinstance(eos, list) else [eos])

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        # model.generate() blocks for seconds; a worker thread keeps the event loop free (D-011).
        return await asyncio.to_thread(self._generate_blocking, request)

    def _generate_blocking(self, request: GenerationRequest) -> GenerationResult:
        # The chat template wraps messages in the ChatML markers the model was tuned on.
        # add_generation_prompt appends the assistant header so the model answers rather than
        # continuing the user's turn. With no system message, the template adds SmolLM's
        # default one; callers override it by sending their own (D-013).
        inputs = self._tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in request.messages],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        prompt_tokens = inputs["input_ids"].shape[1]

        # Temperature 0 means greedy. Sampling at temperature 0 would divide logits by zero.
        if request.temperature == 0.0:
            sampling = {"do_sample": False}
        else:
            sampling = {"do_sample": True, "temperature": request.temperature}

        # inference_mode is thread-local, so it must be entered here, on the worker thread.
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                **sampling,
            )

        # generate() returns prompt + completion; keep only the new ids.
        completion = output[0, prompt_tokens:].tolist()
        # Check the last token, not the count: EOS can land exactly on the max_tokens-th token.
        finish_reason: FinishReason = (
            "stop" if completion and completion[-1] in self._eos_ids else "length"
        )
        return GenerationResult(
            text=self._tokenizer.decode(completion, skip_special_tokens=True),
            prompt_tokens=prompt_tokens,
            completion_tokens=len(completion),
            finish_reason=finish_reason,
        )
