"""Core LLM configuration base and lazy re-exports for provider configs.

Provider-specific configurations live in provider modules (e.g.
`providers.google_genai_provider`). This module exposes a minimal base
class `LLMConfig` and lazily returns provider config classes when accessed
as attributes to preserve backward-compatible import paths.
"""

from abc import ABC, abstractmethod
from typing import Optional


class LLMConfig(ABC):
    """Base configuration class for LLM providers.

    Subclasses should implement `from_env` to construct an instance from
    environment variables.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        thinking_enabled: bool = False,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_enabled = thinking_enabled

    @classmethod
    @abstractmethod
    def from_env(cls) -> "LLMConfig":
        raise NotImplementedError()


__all__ = ["LLMConfig", "GeminiConfig", "OpenAIConfig", "OllamaConfig", "OpenAIResponsesConfig"]


def __getattr__(name: str):
    if name == "GeminiConfig":
        from .providers.google_genai_provider import GeminiConfig  # type: ignore

        return GeminiConfig
    if name == "OpenAIConfig":
        from .providers.openai_provider import OpenAIConfig  # type: ignore

        return OpenAIConfig
    if name == "OllamaConfig":
        from .providers.ollama_provider import OllamaConfig  # type: ignore

        return OllamaConfig
    if name == "OpenAIResponsesConfig":
        from .providers.openai_responses_provider import OpenAIResponsesConfig  # type: ignore

        return OpenAIResponsesConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
