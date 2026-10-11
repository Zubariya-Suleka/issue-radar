from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Gateway configuration, read from environment variables prefixed GATEWAY_ (or .env)."""

    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_", env_file=".env", extra="ignore", protected_namespaces=()
    )

    backend: Literal["fake", "smollm"] = "fake"
    model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    # None means the Hugging Face default (HF_HOME), which we point at G: outside the repo.
    model_cache_dir: Path | None = None

    max_tokens_default: int = 256
    max_tokens_cap: int = 512
    # Provisional: revisit when designing the cache (greedy decoding makes outputs repeatable).
    default_temperature: float = 0.0

    # How long a request may wait for its turn on the model before getting a 503 (D-011).
    queue_timeout_s: float = 30.0
