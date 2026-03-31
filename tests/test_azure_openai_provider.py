"""Tests for the Azure OpenAI provider (AzureOpenAIConfig / AzureOpenAIProvider)."""

from __future__ import annotations

import pytest
from typing import Optional
from unittest.mock import MagicMock, patch, call
from pydantic import BaseModel

from core_lib.llm.providers.azure_openai_provider import (
    AzureOpenAIConfig,
    AzureOpenAIProvider,
)
from core_lib.llm.providers.openai_provider import OpenAIConfig, OpenAIProvider
from core_lib.llm import LLMClient, create_azure_openai_client, AzureOpenAIConfig as PublicConfig
from core_lib.llm.factory import LLMFactory
from core_lib.llm.provider_registry import ProviderConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENDPOINT = "https://my-resource.openai.azure.com"
_API_KEY = "azure-test-key"
_DEPLOYMENT = "gpt-4o-mini"
_API_VERSION = "2024-08-01-preview"


def _make_provider(
    model: str = _DEPLOYMENT,
    api_key: str = _API_KEY,
    azure_endpoint: str = _ENDPOINT,
    azure_api_version: str = _API_VERSION,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    thinking_enabled: bool = False,
    thinking_budget: Optional[int] = None,
) -> tuple[AzureOpenAIProvider, MagicMock]:
    """Return (provider, mock_azure_client) with AzureOpenAI SDK patched."""
    config = AzureOpenAIConfig(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        model=model,
        azure_api_version=azure_api_version,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking_enabled=thinking_enabled,
        thinking_budget=thinking_budget,
    )
    with patch("core_lib.llm.providers.openai_provider.AzureOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        provider = AzureOpenAIProvider(config)
        provider._client = mock_client
    return provider, mock_client


def _make_completion(text: str = "Hello!", input_tokens: int = 10, output_tokens: int = 20) -> MagicMock:
    """Build a minimal mock ChatCompletion object."""
    completion = MagicMock()
    choice = MagicMock()
    message = MagicMock()
    message.content = text
    message.tool_calls = []
    choice.message = message
    completion.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens
    usage.total_tokens = input_tokens + output_tokens
    completion.usage = usage
    return completion


# ---------------------------------------------------------------------------
# AzureOpenAIConfig: creation
# ---------------------------------------------------------------------------

class TestAzureOpenAIConfigCreation:
    def test_provider_identity(self):
        cfg = AzureOpenAIConfig(api_key=_API_KEY, azure_endpoint=_ENDPOINT)
        assert cfg.provider == "azure"

    def test_required_fields(self):
        cfg = AzureOpenAIConfig(api_key=_API_KEY, azure_endpoint=_ENDPOINT, model="gpt-4o")
        assert cfg.api_key == _API_KEY
        assert cfg.azure_endpoint == _ENDPOINT
        assert cfg.model == "gpt-4o"

    def test_default_model(self):
        cfg = AzureOpenAIConfig(api_key=_API_KEY, azure_endpoint=_ENDPOINT)
        assert cfg.model == "gpt-4o-mini"

    def test_default_api_version(self):
        cfg = AzureOpenAIConfig(api_key=_API_KEY, azure_endpoint=_ENDPOINT)
        assert cfg.azure_api_version == _API_VERSION

    def test_custom_api_version(self):
        cfg = AzureOpenAIConfig(api_key=_API_KEY, azure_endpoint=_ENDPOINT, azure_api_version="2025-01-01-preview")
        assert cfg.azure_api_version == "2025-01-01-preview"

    def test_is_azure_property(self):
        cfg = AzureOpenAIConfig(api_key=_API_KEY, azure_endpoint=_ENDPOINT)
        assert cfg.is_azure is True

    def test_is_subclass_of_openai_config(self):
        cfg = AzureOpenAIConfig(api_key=_API_KEY, azure_endpoint=_ENDPOINT)
        assert isinstance(cfg, OpenAIConfig)

    def test_temperature_and_max_tokens(self):
        cfg = AzureOpenAIConfig(
            api_key=_API_KEY, azure_endpoint=_ENDPOINT,
            temperature=0.2, max_tokens=256,
        )
        assert cfg.temperature == 0.2
        assert cfg.max_tokens == 256

    def test_thinking_fields(self):
        cfg = AzureOpenAIConfig(
            api_key=_API_KEY, azure_endpoint=_ENDPOINT,
            thinking_enabled=True, thinking_budget=1024,
        )
        assert cfg.thinking_enabled is True
        assert cfg.thinking_budget == 1024

    def test_organization_and_project(self):
        cfg = AzureOpenAIConfig(
            api_key=_API_KEY, azure_endpoint=_ENDPOINT,
            organization="my-org", project="my-proj",
        )
        assert cfg.organization == "my-org"
        assert cfg.project == "my-proj"

    # base_url should remain None (Azure doesn't use it)
    def test_base_url_is_none(self):
        cfg = AzureOpenAIConfig(api_key=_API_KEY, azure_endpoint=_ENDPOINT)
        assert cfg.base_url is None


# ---------------------------------------------------------------------------
# AzureOpenAIConfig: from_env()
# ---------------------------------------------------------------------------

class TestAzureOpenAIConfigFromEnv:
    @patch.dict("os.environ", {
        "AZURE_OPENAI_API_KEY": "env-key",
        "AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com",
    })
    def test_basic_from_env(self):
        cfg = AzureOpenAIConfig.from_env()
        assert cfg.api_key == "env-key"
        assert cfg.azure_endpoint == "https://env.openai.azure.com"
        assert cfg.provider == "azure"

    @patch.dict("os.environ", {
        "AZURE_OPENAI_API_KEY": "env-key",
        "AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com",
        "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
        "AZURE_OPENAI_API_VERSION": "2025-01-01-preview",
        "AZURE_OPENAI_TEMPERATURE": "0.3",
        "AZURE_OPENAI_MAX_TOKENS": "512",
        "AZURE_OPENAI_ORG": "org-xyz",
    })
    def test_all_env_vars(self):
        cfg = AzureOpenAIConfig.from_env()
        assert cfg.model == "gpt-4o"
        assert cfg.azure_api_version == "2025-01-01-preview"
        assert cfg.temperature == pytest.approx(0.3)
        assert cfg.max_tokens == 512
        assert cfg.organization == "org-xyz"

    @patch.dict("os.environ", {
        "OPENAI_API_KEY": "fallback-key",
        "AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com",
    }, clear=True)
    def test_api_key_fallback_to_openai(self):
        """Falls back to OPENAI_API_KEY when AZURE_OPENAI_API_KEY is absent."""
        cfg = AzureOpenAIConfig.from_env()
        assert cfg.api_key == "fallback-key"

    @patch.dict("os.environ", {
        "AZURE_OPENAI_API_KEY": "env-key",
        "AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com",
        "OPENAI_MODEL": "gpt-4o",  # secondary fallback for deployment
    })
    def test_deployment_fallback_to_openai_model(self):
        cfg = AzureOpenAIConfig.from_env()
        assert cfg.model == "gpt-4o"

    @patch.dict("os.environ", {
        "AZURE_OPENAI_API_KEY": "env-key",
        # AZURE_OPENAI_ENDPOINT intentionally absent
    }, clear=True)
    def test_raises_when_endpoint_missing(self):
        with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
            AzureOpenAIConfig.from_env()

    @patch.dict("os.environ", {
        "AZURE_OPENAI_API_KEY": "env-key",
        "AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com",
    })
    def test_default_api_version_from_env(self):
        cfg = AzureOpenAIConfig.from_env()
        assert cfg.azure_api_version == _API_VERSION


# ---------------------------------------------------------------------------
# AzureOpenAIProvider: instantiation
# ---------------------------------------------------------------------------

class TestAzureOpenAIProviderInit:
    def test_is_subclass_of_openai_provider(self):
        cfg = AzureOpenAIConfig(api_key=_API_KEY, azure_endpoint=_ENDPOINT)
        with patch("core_lib.llm.providers.openai_provider.AzureOpenAI"):
            provider = AzureOpenAIProvider(cfg)
        assert isinstance(provider, OpenAIProvider)

    def test_uses_azure_openai_sdk_client(self):
        cfg = AzureOpenAIConfig(api_key=_API_KEY, azure_endpoint=_ENDPOINT)
        with patch("core_lib.llm.providers.openai_provider.AzureOpenAI") as mock_azure_cls:
            with patch("core_lib.llm.providers.openai_provider.OpenAI") as mock_openai_cls:
                AzureOpenAIProvider(cfg)
        # AzureOpenAI must be called, plain OpenAI must NOT
        mock_azure_cls.assert_called_once()
        mock_openai_cls.assert_not_called()

    def test_azure_sdk_receives_correct_kwargs(self):
        cfg = AzureOpenAIConfig(
            api_key=_API_KEY,
            azure_endpoint=_ENDPOINT,
            azure_api_version="2025-01-01-preview",
        )
        with patch("core_lib.llm.providers.openai_provider.AzureOpenAI") as mock_azure_cls:
            AzureOpenAIProvider(cfg)
        mock_azure_cls.assert_called_once_with(
            api_key=_API_KEY,
            azure_endpoint=_ENDPOINT,
            api_version="2025-01-01-preview",
        )

    def test_raises_when_no_endpoint(self):
        # Build a config object but then clear the endpoint to simulate misconfiguration
        cfg = AzureOpenAIConfig(api_key=_API_KEY, azure_endpoint=_ENDPOINT)
        cfg.azure_endpoint = ""  # clear after construction
        with pytest.raises(ValueError, match="azure_endpoint"):
            AzureOpenAIProvider(cfg)


# ---------------------------------------------------------------------------
# AzureOpenAIProvider: chat()
# ---------------------------------------------------------------------------

class TestAzureOpenAIProviderChat:
    def test_simple_text_response(self):
        provider, mock_client = _make_provider()
        mock_client.chat.completions.create.return_value = _make_completion("Hi from Azure!")
        result = provider.chat(messages=[{"role": "user", "content": "Hello"}])
        assert result["content"] == "Hi from Azure!"
        assert result["structured"] is False
        assert result["tool_calls"] == [] or result.get("tool_calls") is not None

    def test_system_message_prepended(self):
        provider, mock_client = _make_provider()
        mock_client.chat.completions.create.return_value = _make_completion("ok")
        provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            system_message="You are helpful.",
        )
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        msgs = call_kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful."

    def test_model_and_temperature_forwarded(self):
        provider, mock_client = _make_provider(model="gpt-4o", temperature=0.3)
        mock_client.chat.completions.create.return_value = _make_completion("ok")
        provider.chat(messages=[{"role": "user", "content": "Hi"}])
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == pytest.approx(0.3)

    def test_max_tokens_forwarded(self):
        provider, mock_client = _make_provider(max_tokens=128)
        mock_client.chat.completions.create.return_value = _make_completion("ok")
        provider.chat(messages=[{"role": "user", "content": "Hi"}])
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 128

    def test_max_tokens_omitted_when_none(self):
        provider, mock_client = _make_provider()
        mock_client.chat.completions.create.return_value = _make_completion("ok")
        provider.chat(messages=[{"role": "user", "content": "Hi"}])
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "max_tokens" not in call_kwargs

    def test_structured_output_returned(self):
        class Answer(BaseModel):
            value: int

        provider, mock_client = _make_provider()
        import json
        mock_client.chat.completions.create.return_value = _make_completion('{"value": 42}')
        result = provider.chat(
            messages=[{"role": "user", "content": "Give me 42"}],
            structured_output=Answer,
        )
        assert result["structured"] is True
        assert result["content"]["value"] == 42

    def test_tool_calls_returned(self):
        provider, mock_client = _make_provider()
        completion = MagicMock()
        choice = MagicMock()
        message = MagicMock()
        message.content = None
        tc = MagicMock()
        tc.id = "call_1"
        tc.type = "function"
        tc.function.name = "get_weather"
        tc.function.arguments = '{"city": "Paris"}'
        message.tool_calls = [tc]
        choice.message = message
        completion.choices = [choice]
        completion.usage = MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        mock_client.chat.completions.create.return_value = completion

        result = provider.chat(
            messages=[{"role": "user", "content": "Weather?"}],
            tools=[{
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}},
            }],
        )
        assert len(result["tool_calls"]) == 1

    def test_error_returns_error_key(self):
        provider, mock_client = _make_provider()
        mock_client.chat.completions.create.side_effect = RuntimeError("network timeout")
        result = provider.chat(messages=[{"role": "user", "content": "Hi"}])
        assert "error" in result
        assert result["content"] is None


# ---------------------------------------------------------------------------
# LLMClient routing
# ---------------------------------------------------------------------------

class TestLLMClientRouting:
    def test_azure_config_routes_to_azure_provider(self):
        cfg = AzureOpenAIConfig(api_key=_API_KEY, azure_endpoint=_ENDPOINT)
        with patch("core_lib.llm.llm_client.AzureOpenAIProvider") as mock_cls:
            with patch("core_lib.llm.providers.openai_provider.AzureOpenAI"):
                client = LLMClient(cfg)
        mock_cls.assert_called_once_with(cfg)

    def test_openai_config_still_routes_to_openai_provider(self):
        """A plain OpenAIConfig must not be routed to AzureOpenAIProvider."""
        cfg = OpenAIConfig(api_key="sk-test", model="gpt-4o-mini")
        with patch("core_lib.llm.llm_client.OpenAIProvider") as mock_cls:
            with patch("core_lib.llm.llm_client.AzureOpenAIProvider") as mock_azure_cls:
                with patch("core_lib.llm.providers.openai_provider.OpenAI"):
                    client = LLMClient(cfg)
        mock_cls.assert_called_once()
        mock_azure_cls.assert_not_called()

    def test_public_import_is_same_class(self):
        from core_lib.llm.providers.azure_openai_provider import AzureOpenAIConfig as PrivateCls
        assert PublicConfig is PrivateCls


# ---------------------------------------------------------------------------
# ProviderConfig / registry integration
# ---------------------------------------------------------------------------

class TestProviderConfigIntegration:
    def test_azure_alias_normalizes(self):
        pc = ProviderConfig(provider="azure", api_key=_API_KEY, azure_endpoint=_ENDPOINT)
        assert pc.provider == "azure-openai"

    def test_azure_openai_alias_normalizes(self):
        pc = ProviderConfig(provider="azure_openai", api_key=_API_KEY, azure_endpoint=_ENDPOINT)
        assert pc.provider == "azure-openai"

    def test_to_llm_config_returns_azure_config(self):
        pc = ProviderConfig(provider="azure", api_key=_API_KEY, azure_endpoint=_ENDPOINT, model="gpt-4o")
        cfg = pc.to_llm_config()
        assert isinstance(cfg, AzureOpenAIConfig)
        assert cfg.provider == "azure"
        assert cfg.model == "gpt-4o"

    def test_to_llm_config_api_key_propagated(self):
        pc = ProviderConfig(provider="azure", api_key=_API_KEY, azure_endpoint=_ENDPOINT)
        cfg = pc.to_llm_config()
        assert cfg.api_key == _API_KEY

    def test_to_llm_config_endpoint_propagated(self):
        pc = ProviderConfig(provider="azure", api_key=_API_KEY, azure_endpoint=_ENDPOINT)
        cfg = pc.to_llm_config()
        assert cfg.azure_endpoint == _ENDPOINT

    def test_to_llm_config_default_api_version(self):
        pc = ProviderConfig(provider="azure", api_key=_API_KEY, azure_endpoint=_ENDPOINT)
        cfg = pc.to_llm_config()
        assert cfg.azure_api_version == _API_VERSION

    def test_to_llm_config_custom_api_version(self):
        pc = ProviderConfig(
            provider="azure", api_key=_API_KEY, azure_endpoint=_ENDPOINT,
            azure_api_version="2025-01-01-preview",
        )
        cfg = pc.to_llm_config()
        assert cfg.azure_api_version == "2025-01-01-preview"

    def test_is_configured_true_with_key_and_endpoint(self):
        pc = ProviderConfig(provider="azure", api_key=_API_KEY, azure_endpoint=_ENDPOINT)
        assert pc.is_configured() is True

    def test_is_configured_false_missing_endpoint(self):
        pc = ProviderConfig(provider="azure", api_key=_API_KEY)
        assert pc.is_configured() is False

    def test_is_configured_false_missing_key(self):
        pc = ProviderConfig(provider="azure", azure_endpoint=_ENDPOINT)
        assert pc.is_configured() is False

    @patch.dict("os.environ", {
        "AZURE_OPENAI_API_KEY": "env-key",
        "AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com",
    })
    def test_is_configured_true_via_env(self):
        pc = ProviderConfig(provider="azure")
        assert pc.is_configured() is True

    @patch.dict("os.environ", {
        "AZURE_OPENAI_API_KEY": "env-key",
        "AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com",
    })
    def test_to_llm_config_reads_endpoint_from_env(self):
        pc = ProviderConfig(provider="azure")
        cfg = pc.to_llm_config()
        assert cfg.azure_endpoint == "https://env.openai.azure.com"
        assert cfg.api_key == "env-key"

    def test_from_dict_azure_provider(self):
        pc = ProviderConfig.from_dict({
            "provider": "azure",
            "api_key": _API_KEY,
            "azure_endpoint": _ENDPOINT,
            "model": "gpt-4o",
            "azure_api_version": "2025-01-01-preview",
        })
        assert pc.provider == "azure-openai"
        cfg = pc.to_llm_config()
        assert isinstance(cfg, AzureOpenAIConfig)
        assert cfg.azure_api_version == "2025-01-01-preview"

    def test_thinking_budget_propagated(self):
        pc = ProviderConfig.from_dict({
            "provider": "azure",
            "api_key": _API_KEY,
            "azure_endpoint": _ENDPOINT,
            "thinking_enabled": True,
            "thinking_budget": 2048,
        })
        cfg = pc.to_llm_config()
        assert cfg.thinking_enabled is True
        assert cfg.thinking_budget == 2048


# ---------------------------------------------------------------------------
# LLMFactory integration
# ---------------------------------------------------------------------------

class TestLLMFactoryAzure:
    @patch("core_lib.llm.factory.LLMClient")
    def test_factory_azure_openai_explicit_params(self, mock_client_cls):
        LLMFactory.azure_openai(
            api_key=_API_KEY,
            azure_endpoint=_ENDPOINT,
            deployment="gpt-4o",
            temperature=0.2,
        )
        call_args = mock_client_cls.call_args[0][0]
        assert isinstance(call_args, AzureOpenAIConfig)
        assert call_args.api_key == _API_KEY
        assert call_args.azure_endpoint == _ENDPOINT
        assert call_args.model == "gpt-4o"
        assert call_args.temperature == pytest.approx(0.2)

    @patch.dict("os.environ", {
        "AZURE_OPENAI_API_KEY": "env-key",
        "AZURE_OPENAI_ENDPOINT": _ENDPOINT,
        "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
    })
    @patch("core_lib.llm.factory.LLMClient")
    def test_factory_azure_openai_from_env(self, mock_client_cls):
        LLMFactory.azure_openai()
        call_args = mock_client_cls.call_args[0][0]
        assert isinstance(call_args, AzureOpenAIConfig)
        assert call_args.api_key == "env-key"

    @patch.dict("os.environ", {
        "AZURE_OPENAI_API_KEY": "env-key",
        "AZURE_OPENAI_ENDPOINT": _ENDPOINT,
    })
    @patch("core_lib.llm.factory.LLMClient")
    def test_factory_create_with_azure_provider(self, mock_client_cls):
        from core_lib.llm import create_llm_client
        create_llm_client(provider="azure")
        call_args = mock_client_cls.call_args[0][0]
        assert isinstance(call_args, AzureOpenAIConfig)

    @patch.dict("os.environ", {
        "AZURE_OPENAI_API_KEY": "env-key",
        "AZURE_OPENAI_ENDPOINT": _ENDPOINT,
    })
    @patch("core_lib.llm.factory.LLMClient")
    def test_create_azure_openai_client_convenience(self, mock_client_cls):
        create_azure_openai_client(
            api_key=_API_KEY,
            azure_endpoint=_ENDPOINT,
            deployment="gpt-4o",
        )
        call_args = mock_client_cls.call_args[0][0]
        assert isinstance(call_args, AzureOpenAIConfig)
        assert call_args.model == "gpt-4o"

    @patch("core_lib.llm.factory.LLMClient")
    def test_factory_azure_openai_deployment_override(self, mock_client_cls):
        """deployment= kwarg should override the model from environment."""
        with patch.dict("os.environ", {
            "AZURE_OPENAI_API_KEY": "env-key",
            "AZURE_OPENAI_ENDPOINT": _ENDPOINT,
            "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
        }):
            LLMFactory.azure_openai(deployment="gpt-4o-mini")
        call_args = mock_client_cls.call_args[0][0]
        assert call_args.model == "gpt-4o-mini"
