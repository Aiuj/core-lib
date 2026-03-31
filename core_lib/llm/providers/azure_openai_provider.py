"""Azure OpenAI provider — first-class "azure" provider backed by AzureOpenAI SDK.

This module wraps :class:`~core_lib.llm.providers.openai_provider.OpenAIProvider`
to produce an explicit ``provider = "azure"`` identity and enforce that
``azure_endpoint`` is always set before making any API call.

All chat logic (structured output, tool calling, thinking mode, tracing …) is
fully inherited from :class:`OpenAIProvider` — no duplication.

Configuration via environment variables
---------------------------------------
+-------------------------------+----------------------------------------------+
| Variable                      | Purpose                                      |
+===============================+==============================================+
| ``AZURE_OPENAI_API_KEY``      | API key (required)                           |
+-------------------------------+----------------------------------------------+
| ``AZURE_OPENAI_ENDPOINT``     | Service endpoint URL (required)              |
|                               | e.g. ``https://<resource>.openai.azure.com`` |
+-------------------------------+----------------------------------------------+
| ``AZURE_OPENAI_DEPLOYMENT``   | Deployment / model name (default: gpt-4o-mini) |
+-------------------------------+----------------------------------------------+
| ``AZURE_OPENAI_API_VERSION``  | REST API date version (default: 2024-08-01-preview) |
+-------------------------------+----------------------------------------------+
| ``AZURE_OPENAI_TEMPERATURE``  | Sampling temperature (falls back to         |
|                               | ``OPENAI_TEMPERATURE``)                      |
+-------------------------------+----------------------------------------------+
| ``AZURE_OPENAI_MAX_TOKENS``   | Max output tokens (falls back to            |
|                               | ``OPENAI_MAX_TOKENS``)                       |
+-------------------------------+----------------------------------------------+
| ``AZURE_OPENAI_ORG``          | Organisation ID (optional)                   |
+-------------------------------+----------------------------------------------+

Programmatic usage
------------------
::

    from core_lib.llm import create_azure_openai_client, AzureOpenAIConfig

    # From environment variables
    client = create_azure_openai_client()

    # Explicit configuration
    client = create_azure_openai_client(
        api_key="<key>",
        azure_endpoint="https://my-resource.openai.azure.com",
        deployment="gpt-4o",
    )

    # Direct config object
    config = AzureOpenAIConfig(
        api_key="<key>",
        azure_endpoint="https://my-resource.openai.azure.com",
        model="gpt-4o",
        azure_api_version="2024-08-01-preview",
    )
    from core_lib.llm import LLMClient
    client = LLMClient(config)
    response = client.chat("Hello!")
    print(response["content"])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core_lib import get_module_logger

from .openai_provider import OpenAIConfig, OpenAIProvider

logger = get_module_logger()

_DEFAULT_API_VERSION = "2024-08-01-preview"
_DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class AzureOpenAIConfig(OpenAIConfig):
    """Configuration for Azure OpenAI.

    Inherits all fields from :class:`OpenAIConfig` and forces
    ``provider = "azure"``.  ``azure_endpoint`` **must** be provided either
    at construction time or through the ``AZURE_OPENAI_ENDPOINT`` environment
    variable before :class:`AzureOpenAIProvider` is instantiated.

    Args:
        api_key:           Azure OpenAI API key.
        azure_endpoint:    Full service endpoint URL,
                           e.g. ``https://<resource>.openai.azure.com``.
        model:             Deployment name / model identifier.
        azure_api_version: REST API date version string.
        temperature:       Sampling temperature (0–2).
        max_tokens:        Maximum number of tokens to generate.
        thinking_enabled:  Enable chain-of-thought / reasoning mode.
        thinking_budget:   Token budget for reasoning (provider-specific).
        organization:      Azure / OpenAI organisation ID (optional).
        project:           OpenAI project ID (optional).
    """

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        model: str = _DEFAULT_MODEL,
        azure_api_version: str = _DEFAULT_API_VERSION,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        thinking_enabled: bool = False,
        thinking_budget: Optional[int] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
    ) -> None:
        # Call OpenAIConfig.__init__ — sets self.provider = "openai" first …
        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_enabled=thinking_enabled,
            thinking_budget=thinking_budget,
            base_url=None,
            organization=organization,
            project=project,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
        )
        # … then override to make the provider identity explicit.
        self.provider = "azure"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_azure(self) -> bool:
        """Always ``True`` for this config class."""
        return True

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "AzureOpenAIConfig":
        """Build an :class:`AzureOpenAIConfig` from environment variables.

        Reads (in priority order):

        * ``AZURE_OPENAI_API_KEY``
        * ``AZURE_OPENAI_ENDPOINT``
        * ``AZURE_OPENAI_DEPLOYMENT`` / ``OPENAI_MODEL``
        * ``AZURE_OPENAI_API_VERSION``
        * ``AZURE_OPENAI_TEMPERATURE`` / ``OPENAI_TEMPERATURE``
        * ``AZURE_OPENAI_MAX_TOKENS`` / ``OPENAI_MAX_TOKENS``
        * ``AZURE_OPENAI_ORG``

        Returns:
            Fully populated :class:`AzureOpenAIConfig`.

        Raises:
            ValueError: If ``AZURE_OPENAI_ENDPOINT`` is not set.
        """
        import os

        def _getenv(*names: str, default: Optional[str] = None) -> Optional[str]:
            for name in names:
                val = os.getenv(name)
                if val:
                    return val
            return default

        api_key = _getenv("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY", default="") or ""
        azure_endpoint = _getenv("AZURE_OPENAI_ENDPOINT") or ""
        model = _getenv("AZURE_OPENAI_DEPLOYMENT", "OPENAI_MODEL", default=_DEFAULT_MODEL) or _DEFAULT_MODEL
        azure_api_version = (
            _getenv("AZURE_OPENAI_API_VERSION", default=_DEFAULT_API_VERSION) or _DEFAULT_API_VERSION
        )
        temperature = float(
            _getenv("AZURE_OPENAI_TEMPERATURE", "OPENAI_TEMPERATURE", default="0.7") or 0.7
        )
        max_tokens_raw = _getenv("AZURE_OPENAI_MAX_TOKENS", "OPENAI_MAX_TOKENS")
        max_tokens = int(max_tokens_raw) if max_tokens_raw else None
        organization = _getenv("AZURE_OPENAI_ORG", "OPENAI_ORG", "OPENAI_ORGANIZATION")

        if not azure_endpoint:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT is required when using the Azure provider. "
                "Set it to your Azure OpenAI service endpoint URL, e.g. "
                "https://<resource-name>.openai.azure.com"
            )

        return cls(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            model=model,
            azure_api_version=azure_api_version,
            temperature=temperature,
            max_tokens=max_tokens,
            organization=organization,
        )


class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI provider.

    Thin subclass of :class:`OpenAIProvider` that enforces the Azure path
    (``azure_endpoint`` **must** be set) and reports ``"azure"`` as the
    provider name in logs and traces.

    All chat logic — structured output, tool calling, thinking mode, usage
    tracing — is fully inherited.

    Args:
        config: An :class:`AzureOpenAIConfig` instance.

    Raises:
        ValueError: If ``config.azure_endpoint`` is empty or ``None``.
    """

    def __init__(self, config: AzureOpenAIConfig) -> None:  # type: ignore[override]
        if not config.azure_endpoint:
            raise ValueError(
                "AzureOpenAIProvider requires azure_endpoint to be set. "
                "Pass it to AzureOpenAIConfig or set AZURE_OPENAI_ENDPOINT."
            )
        super().__init__(config)
        logger.debug(
            "azure_openai.init",
            extra={
                "llm_provider": "azure",
                "model": config.model,
                "azure_endpoint": config.azure_endpoint,
                "azure_api_version": config.azure_api_version,
            },
        )
