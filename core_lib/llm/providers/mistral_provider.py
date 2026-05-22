"""Mistral AI provider using the OpenAI-compatible API.

Mistral's /v1/chat/completions endpoint is fully OpenAI-compatible.
Reuses OpenAIProvider for all features (chat, vision, tools, structured output)
and only overrides thinking mode, which uses ``reasoning_effort`` instead of
``enable_thinking``.

Thinking / reasoning:
    Only ``magistral-*`` models support reasoning.  Pass ``thinking_enabled=True``
    (or call ``chat(thinking_enabled=True)``) to activate it.  Non-magistral
    models silently ignore the flag.

Supported env vars:
    MISTRAL_API_KEY      API key (required)
    MISTRAL_MODEL        Default model (default: mistral-small-latest)
    MISTRAL_TEMPERATURE  Sampling temperature (default: 0.7)
    MISTRAL_MAX_TOKENS   Maximum output tokens (optional)
    MISTRAL_TIMEOUT      HTTP timeout in seconds (default: 60)

Docs: https://docs.mistral.ai/api
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .openai_provider import OpenAIConfig, OpenAIProvider

MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
MISTRAL_DEFAULT_MODEL = "mistral-small-latest"


class MistralConfig(OpenAIConfig):
    """Configuration for the Mistral AI provider."""

    def __init__(
        self,
        api_key: str,
        model: str = MISTRAL_DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        thinking_enabled: bool = False,
        timeout: int = 60,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_enabled=thinking_enabled,
            base_url=MISTRAL_BASE_URL,
            timeout=timeout,
        )
        # OpenAIConfig.__init__ sets provider="openai"; override to "mistral".
        self.provider = "mistral"

    @property
    def is_local_compatible(self) -> bool:
        """Mistral is a cloud provider — never treat it as a local-compatible endpoint."""
        return False

    @classmethod
    def from_env(cls) -> "MistralConfig":
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        model = os.environ.get("MISTRAL_MODEL", MISTRAL_DEFAULT_MODEL)
        temperature = float(os.environ.get("MISTRAL_TEMPERATURE", "0.7"))
        max_tokens_env = os.environ.get("MISTRAL_MAX_TOKENS")
        max_tokens = int(max_tokens_env) if max_tokens_env else None
        timeout = int(os.environ.get("MISTRAL_TIMEOUT", "60"))
        return cls(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )


class MistralProvider(OpenAIProvider):
    """Provider for Mistral AI via the OpenAI-compatible Chat Completions API.

    Inherits all functionality from :class:`OpenAIProvider` (vision, tool
    calling, structured output, tracing) and only overrides:

    * ``_provider_tracing_name`` — reports ``"mistral"`` in traces/logs
    * ``_apply_thinking_mode`` — injects ``reasoning_effort`` for magistral-*
      models instead of the OpenAI/Alibaba ``enable_thinking`` mechanism
    """

    def __init__(self, config: MistralConfig) -> None:
        super().__init__(config)
        self.config: MistralConfig = config

    @property
    def _provider_tracing_name(self) -> str:
        return "mistral"

    def _is_magistral_model(self) -> bool:
        """Return True when the configured model is a magistral reasoning model."""
        return "magistral" in (self.config.model or "").lower()

    def _apply_thinking_mode(self, create_kwargs: Dict[str, Any], use_thinking: bool) -> None:
        """Apply Mistral reasoning_effort for magistral-* models.

        Non-magistral models do not support ``reasoning_effort``; the flag is
        silently ignored for them.
        """
        if self._is_magistral_model():
            create_kwargs["extra_body"] = {"reasoning_effort": "high" if use_thinking else "none"}
        # For all other Mistral models, thinking is unsupported — no-op.
