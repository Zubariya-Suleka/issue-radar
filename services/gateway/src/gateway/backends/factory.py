from gateway.backends.base import ModelBackend
from gateway.backends.fake import FakeBackend
from gateway.config import Settings


def build_backend(settings: Settings) -> ModelBackend:
    """Pick the backend from config. Adding a backend means one new class and one branch here."""
    if settings.backend == "fake":
        return FakeBackend()
    if settings.backend == "smollm":
        # Imported lazily: torch is heavy and only installed with the 'local' extra.
        from gateway.backends.smollm import SmolLMBackend

        return SmolLMBackend(model_id=settings.model_id, cache_dir=settings.model_cache_dir)
    raise ValueError(f"Unknown backend: {settings.backend}")
