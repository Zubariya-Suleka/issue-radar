from pathlib import Path
from typing import Literal

from pydantic import field_validator
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
    # Greedy by default: repeatable outputs, which the cache relies on (D-015).
    default_temperature: float = 0.0

    # How long a request may wait for its turn on the model before getting a 503 (D-011).
    queue_timeout_s: float = 30.0

    # In-memory LRU of greedy generations (D-015). 0 disables the cache.
    cache_max_entries: int = 1024

    # Client name -> API key, as JSON: GATEWAY_API_KEYS='{"rag": "...", "agent": "..."}'.
    # Empty means auth is off and every caller is one "anonymous" client (D-016).
    api_keys: dict[str, str] = {}
    # Per-client token bucket (D-016). rate_limit_per_minute = 0 disables limiting.
    rate_limit_per_minute: float = 30.0
    rate_limit_burst: int = 10

    @field_validator("api_keys")
    @classmethod
    def keys_are_distinct_and_non_empty(cls, keys: dict[str, str]) -> dict[str, str]:
        if any(not key for key in keys.values()):
            raise ValueError("API keys must not be empty")
        if len(set(keys.values())) != len(keys):
            raise ValueError("each client needs its own API key")
        return keys

    # Request log (D-014): JSON lines to stdout, and to this file unless None.
    log_file: Path | None = Path("logs/gateway.jsonl")
    # Debug only: also log prompt and response text. Issue text can contain personal data (D-006).
    log_content: bool = False
