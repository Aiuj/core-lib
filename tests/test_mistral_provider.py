"""Unit tests for the Mistral AI provider."""

import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel
from typing import Optional

from core_lib.llm.providers.mistral_provider import MistralConfig, MistralProvider, MISTRAL_BASE_URL


# ---------------------------------------------------------------------------
# MistralConfig tests
# ---------------------------------------------------------------------------

class TestMistralConfig:

    def test_defaults(self):
        config = MistralConfig(api_key="sk-test")
        assert config.provider == "mistral"
        assert config.model == "mistral-small-latest"
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.thinking_enabled is False
        assert config.base_url == MISTRAL_BASE_URL

    def test_custom_values(self):
        config = MistralConfig(
            api_key="sk-test",
            model="mistral-large-latest",
            temperature=0.3,
            max_tokens=2048,
            thinking_enabled=False,
            timeout=120,
        )
        assert config.model == "mistral-large-latest"
        assert config.temperature == 0.3
        assert config.max_tokens == 2048
        assert config.timeout == 120

    def test_is_local_compatible_always_false(self):
        """Mistral is a cloud provider — should never be treated as local-compatible."""
        config = MistralConfig(api_key="sk-test")
        assert config.is_local_compatible is False

    def test_is_alibaba_false(self):
        config = MistralConfig(api_key="sk-test")
        assert config.is_alibaba is False

    def test_is_ovh_false(self):
        config = MistralConfig(api_key="sk-test")
        assert config.is_ovh is False

    def test_is_openrouter_false(self):
        config = MistralConfig(api_key="sk-test")
        assert config.is_openrouter is False

    def test_from_env_defaults(self):
        with patch.dict("os.environ", {
            "MISTRAL_API_KEY": "sk-env-key",
        }, clear=False):
            config = MistralConfig.from_env()
            assert config.api_key == "sk-env-key"
            assert config.model == "mistral-small-latest"
            assert config.temperature == 0.7
            assert config.max_tokens is None
            assert config.timeout == 60

    def test_from_env_full(self):
        with patch.dict("os.environ", {
            "MISTRAL_API_KEY": "sk-abc",
            "MISTRAL_MODEL": "mistral-large-latest",
            "MISTRAL_TEMPERATURE": "0.1",
            "MISTRAL_MAX_TOKENS": "1024",
            "MISTRAL_TIMEOUT": "30",
        }, clear=False):
            config = MistralConfig.from_env()
            assert config.api_key == "sk-abc"
            assert config.model == "mistral-large-latest"
            assert config.temperature == 0.1
            assert config.max_tokens == 1024
            assert config.timeout == 30

    def test_is_subclass_of_openai_config(self):
        from core_lib.llm.providers.openai_provider import OpenAIConfig
        config = MistralConfig(api_key="key")
        assert isinstance(config, OpenAIConfig)


# ---------------------------------------------------------------------------
# MistralProvider thinking mode tests
# ---------------------------------------------------------------------------

class TestMistralProviderThinkingMode:

    def _make_provider(self, model: str = "mistral-small-latest", thinking_enabled: bool = False):
        config = MistralConfig(api_key="sk-test", model=model, thinking_enabled=thinking_enabled)
        with patch("core_lib.llm.providers.openai_provider.OpenAIProvider.__init__"):
            provider = MistralProvider.__new__(MistralProvider)
            provider.config = config
        return provider

    def test_provider_tracing_name(self):
        p = self._make_provider()
        assert p._provider_tracing_name == "mistral"

    def test_is_magistral_model_true(self):
        p = self._make_provider(model="magistral-medium-latest")
        assert p._is_magistral_model() is True

    def test_is_magistral_model_false(self):
        p = self._make_provider(model="mistral-large-latest")
        assert p._is_magistral_model() is False

    def test_apply_thinking_mode_magistral_enabled(self):
        p = self._make_provider(model="magistral-medium-latest")
        create_kwargs: dict = {}
        p._apply_thinking_mode(create_kwargs, use_thinking=True)
        assert create_kwargs.get("extra_body") == {"reasoning_effort": "high"}

    def test_apply_thinking_mode_magistral_disabled(self):
        p = self._make_provider(model="magistral-medium-latest")
        create_kwargs: dict = {}
        p._apply_thinking_mode(create_kwargs, use_thinking=False)
        assert create_kwargs.get("extra_body") == {"reasoning_effort": "none"}

    def test_apply_thinking_mode_non_magistral_noop(self):
        """Non-magistral models: thinking flag should have no effect."""
        p = self._make_provider(model="mistral-large-latest")
        create_kwargs: dict = {}
        p._apply_thinking_mode(create_kwargs, use_thinking=True)
        assert "extra_body" not in create_kwargs

    def test_apply_thinking_mode_small_model_noop(self):
        p = self._make_provider(model="mistral-small-latest")
        create_kwargs: dict = {}
        p._apply_thinking_mode(create_kwargs, use_thinking=True)
        assert "extra_body" not in create_kwargs


# ---------------------------------------------------------------------------
# MistralProvider full chat (mocked API call)
# ---------------------------------------------------------------------------

def _fake_completion(content: str = "Hello from Mistral", tool_calls=None):
    """Build a minimal mock OpenAI completion object."""
    message = MagicMock()
    message.content = content
    message.reasoning = None
    message.tool_calls = tool_calls or []
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15
    completion.usage = usage
    return completion


class TestMistralProviderChat:
    """Test MistralProvider.chat() with mocked API calls."""

    def _make_provider(self, model: str = "mistral-small-latest", thinking_enabled: bool = False):
        config = MistralConfig(api_key="sk-test", model=model, thinking_enabled=thinking_enabled)
        with patch("core_lib.llm.providers.openai_provider.OpenAIProvider.__init__", return_value=None):
            provider = MistralProvider.__new__(MistralProvider)
            provider.config = config
            provider._wake_on_lan = MagicMock()
            provider._wake_on_lan.is_in_warmup.return_value = False
            provider._wake_on_lan.maybe_get_initial_timeout.return_value = 60.0
            provider._wake_on_lan.maybe_wake.return_value = MagicMock(succeeded=False)
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _fake_completion()
            provider._client = mock_client
        return provider

    def test_basic_chat(self):
        provider = self._make_provider()
        result = provider.chat(messages=[{"role": "user", "content": "Hi"}])
        assert result["content"] == "Hello from Mistral"
        assert result["structured"] is False
        assert result["tool_calls"] == []

    def test_tracing_uses_mistral_provider_name(self):
        """Verify add_trace_metadata receives gen_ai.system='mistral'."""
        provider = self._make_provider()
        captured_meta = {}
        with patch("core_lib.llm.providers.openai_provider.add_trace_metadata") as mock_trace, \
             patch("core_lib.llm.providers.openai_provider.log_llm_usage") as mock_log:
            mock_trace.side_effect = lambda m: captured_meta.update(m)
            provider.chat(messages=[{"role": "user", "content": "Hi"}])
            assert captured_meta.get("gen_ai.system") == "mistral"
            # log_llm_usage should also use 'mistral'
            assert mock_log.call_args.kwargs.get("provider") == "mistral"

    def test_thinking_extra_body_injected_for_magistral(self):
        """reasoning_effort should be added to API call for magistral models."""
        provider = self._make_provider(model="magistral-medium-latest", thinking_enabled=True)
        provider.chat(messages=[{"role": "user", "content": "Think hard"}])
        call_kwargs = provider._client.chat.completions.create.call_args.kwargs
        # extra_body should have been passed
        extra_body = call_kwargs.get("extra_body", {})
        assert extra_body.get("reasoning_effort") == "high"

    def test_no_extra_body_for_non_magistral(self):
        provider = self._make_provider(model="mistral-large-latest", thinking_enabled=True)
        provider.chat(messages=[{"role": "user", "content": "Hi"}])
        call_kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert "extra_body" not in call_kwargs


# ---------------------------------------------------------------------------
# LLMClient integration (dispatch to MistralProvider)
# ---------------------------------------------------------------------------

class TestLLMClientMistralDispatch:
    """Verify LLMClient creates a MistralProvider for MistralConfig."""

    def test_dispatch_creates_mistral_provider(self):
        from core_lib.llm.llm_client import LLMClient
        config = MistralConfig(api_key="sk-test")
        with patch("core_lib.llm.llm_client.MistralProvider") as mock_mistral:
            mock_mistral.return_value = MagicMock()
            client = LLMClient(config)
            mock_mistral.assert_called_once_with(config)

    def test_mistral_not_dispatched_to_openai_provider(self):
        """MistralConfig should NOT create an OpenAIProvider."""
        from core_lib.llm.llm_client import LLMClient
        config = MistralConfig(api_key="sk-test")
        with patch("core_lib.llm.llm_client.MistralProvider", return_value=MagicMock()), \
             patch("core_lib.llm.llm_client.OpenAIProvider") as mock_openai:
            LLMClient(config)
            mock_openai.assert_not_called()


# ---------------------------------------------------------------------------
# LLMFactory
# ---------------------------------------------------------------------------

class TestLLMFactoryMistral:

    def test_factory_mistral_with_key(self):
        from core_lib.llm import LLMFactory
        with patch("core_lib.llm.llm_client.MistralProvider") as mock_cls:
            mock_cls.return_value = MagicMock()
            client = LLMFactory.mistral(
                api_key="sk-test",
                model="mistral-large-latest",
                temperature=0.2,
            )
            mock_cls.assert_called_once()
            config = mock_cls.call_args[0][0]
            assert isinstance(config, MistralConfig)
            assert config.api_key == "sk-test"
            assert config.model == "mistral-large-latest"
            assert config.temperature == 0.2

    def test_factory_mistral_from_env(self):
        from core_lib.llm import LLMFactory
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "sk-env"}), \
             patch("core_lib.llm.llm_client.MistralProvider") as mock_cls:
            mock_cls.return_value = MagicMock()
            client = LLMFactory.mistral()
            config = mock_cls.call_args[0][0]
            assert config.api_key == "sk-env"

    def test_create_mistral_client_convenience(self):
        from core_lib.llm import create_mistral_client
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "sk-abc"}), \
             patch("core_lib.llm.llm_client.MistralProvider") as mock_cls:
            mock_cls.return_value = MagicMock()
            create_mistral_client(api_key="sk-abc", model="ministral-8b-latest")
            config = mock_cls.call_args[0][0]
            assert isinstance(config, MistralConfig)
            assert config.model == "ministral-8b-latest"

    def test_from_env_mistral_provider(self):
        from core_lib.llm import LLMFactory
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "mistral",
            "MISTRAL_API_KEY": "sk-test",
        }), patch("core_lib.llm.llm_client.MistralProvider") as mock_cls:
            mock_cls.return_value = MagicMock()
            LLMFactory.from_env()
            assert mock_cls.called

    @patch.dict("os.environ", {"MISTRAL_API_KEY": "sk-auto"})
    def test_detect_provider_from_env_prefers_mistral(self):
        from core_lib.llm import LLMFactory
        provider = LLMFactory._detect_provider_from_env()
        assert provider == "mistral"


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

class TestMistralPricing:
    """Verify Mistral models are registered in the pricing table."""

    def test_ministral_3b_pricing(self):
        from core_lib.tracing.service_pricing import get_llm_pricing
        pricing = get_llm_pricing("ministral-3b-latest")
        assert pricing is not None
        assert pricing["input"] == pytest.approx(0.0001)
        assert pricing["output"] == pytest.approx(0.0001)

    def test_ministral_8b_pricing(self):
        from core_lib.tracing.service_pricing import get_llm_pricing
        pricing = get_llm_pricing("ministral-8b-latest")
        assert pricing is not None
        assert pricing["input"] == pytest.approx(0.00015)

    def test_mistral_small_pricing(self):
        from core_lib.tracing.service_pricing import get_llm_pricing
        pricing = get_llm_pricing("mistral-small-latest")
        assert pricing is not None
        assert pricing["input"] == pytest.approx(0.00015)
        assert pricing["output"] == pytest.approx(0.0006)

    def test_mistral_medium_pricing(self):
        from core_lib.tracing.service_pricing import get_llm_pricing
        pricing = get_llm_pricing("mistral-medium-latest")
        assert pricing is not None
        assert pricing["input"] == pytest.approx(0.0015)
        assert pricing["output"] == pytest.approx(0.0075)

    def test_mistral_large_pricing(self):
        from core_lib.tracing.service_pricing import get_llm_pricing
        pricing = get_llm_pricing("mistral-large-latest")
        assert pricing is not None
        assert pricing["input"] == pytest.approx(0.0005)

    def test_magistral_medium_pricing(self):
        from core_lib.tracing.service_pricing import get_llm_pricing
        pricing = get_llm_pricing("magistral-medium-latest")
        assert pricing is not None
        assert pricing["input"] == pytest.approx(0.002)
        assert pricing["output"] == pytest.approx(0.005)

    def test_magistral_small_pricing(self):
        from core_lib.tracing.service_pricing import get_llm_pricing
        pricing = get_llm_pricing("magistral-small-latest")
        assert pricing is not None
        assert pricing["input"] == pytest.approx(0.0005)

    def test_ministral_14b_pricing(self):
        from core_lib.tracing.service_pricing import get_llm_pricing
        pricing = get_llm_pricing("ministral-14b-latest")
        assert pricing is not None
        assert pricing["input"] == pytest.approx(0.0002)
        assert pricing["output"] == pytest.approx(0.0002)


# ---------------------------------------------------------------------------
# ProviderConfig / ProviderRegistry integration
# ---------------------------------------------------------------------------

class TestProviderConfigMistral:
    """Verify ProviderConfig recognises the 'mistral' provider."""

    def test_to_llm_config_returns_mistral_config(self):
        from core_lib.llm.provider_registry import ProviderConfig
        cfg = ProviderConfig(
            provider="mistral",
            api_key="sk-test",
            model="mistral-small-latest",
        )
        llm_cfg = cfg.to_llm_config()
        assert isinstance(llm_cfg, MistralConfig)
        assert llm_cfg.api_key == "sk-test"
        assert llm_cfg.model == "mistral-small-latest"
        assert llm_cfg.provider == "mistral"

    def test_is_configured_with_api_key(self):
        from core_lib.llm.provider_registry import ProviderConfig
        cfg = ProviderConfig(provider="mistral", api_key="sk-test")
        assert cfg.is_configured() is True

    def test_is_configured_from_env(self):
        from core_lib.llm.provider_registry import ProviderConfig
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "sk-env"}):
            cfg = ProviderConfig(provider="mistral")
            assert cfg.is_configured() is True

    def test_is_configured_without_key_false(self):
        from core_lib.llm.provider_registry import ProviderConfig
        cfg = ProviderConfig(provider="mistral")
        # No api_key, no env var
        with patch.dict("os.environ", {}, clear=True):
            assert cfg.is_configured() is False

    def test_default_model(self):
        from core_lib.llm.provider_registry import ProviderConfig
        cfg = ProviderConfig(provider="mistral", api_key="sk-test")
        assert cfg.model == "mistral-small-latest"

    def test_from_dict_to_llm_config(self):
        from core_lib.llm.provider_registry import ProviderConfig
        cfg = ProviderConfig.from_dict({
            "provider": "mistral",
            "api_key": "sk-test",
            "model": "ministral-8b-latest",
            "temperature": 0.1,
        })
        assert cfg.provider == "mistral"
        llm_cfg = cfg.to_llm_config()
        assert isinstance(llm_cfg, MistralConfig)
        assert llm_cfg.model == "ministral-8b-latest"
        assert llm_cfg.temperature == pytest.approx(0.1)
