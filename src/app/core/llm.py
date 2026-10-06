"""The single Consciousnx client used by every feature that calls a language model."""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI, OpenAIError

from app.core.config import get_settings
from app.core.errors import ConfigurationError, DependencyUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Return the shared Consciousnx client, built from settings on first use."""
    settings = get_settings()
    if not settings.has_provider_key:
        raise ConfigurationError(
            "CONSCIOUSNX_API_KEY is not set. Open .env and replace the placeholder "
            "with a real key."
        )

    return OpenAI(
        api_key=settings.consciousnx_api_key,
        base_url=settings.consciousnx_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def complete(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.1,
    model: str | None = None,
) -> str:
    """Send a single-turn completion request and return the assistant text."""
    settings = get_settings()
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = get_client().chat.completions.create(
            model=model or settings.consciousnx_model,
            messages=messages,
            temperature=temperature,
        )
    except OpenAIError as exc:
        logger.warning("Consciousnx request failed: %s", exc)
        raise DependencyUnavailableError(f"Consciousnx request failed: {exc}") from exc

    content = response.choices[0].message.content if response.choices else None
    return (content or "").strip()


def is_configured() -> bool:
    """Report whether a client can be built. Performs no network call."""
    try:
        get_client()
    except (ConfigurationError, OpenAIError):
        return False
    return True
