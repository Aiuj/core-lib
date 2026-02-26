"""Tests for FallbackLLMClient with automatic provider failover."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from core_lib.llm.fallback_client import (
    FallbackLLMClient,
    FallbackResult,
    create_fallback_llm_client,
)
from core_lib.llm.provider_registry import ProviderConfig, ProviderRegistry
from core_lib.llm.provider_health import ProviderHealthTracker, reset_health_tracker


@pytest.fixture(autouse=True)
def reset_health():
    """Reset health tracker before each test."""
    reset_health_tracker()
    yield
    reset_health_tracker()


@pytest.fixture
def mock_registry():
    """Create a mock registry with two providers."""
    registry = ProviderRegistry()
    registry.add(ProviderConfig(
        provider="gemini",
        model="gemini-2.0-flash",
        api_key="test-key-1",
        priority=1,
    ))
    registry.add(ProviderConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="test-key-2",
        priority=2,
    ))
    return registry


@pytest.fixture
def mock_health_tracker():
    """Create a health tracker with in-memory tracking."""
    return ProviderHealthTracker(cache_client=None)


def create_mock_client(should_succeed=True, response_content="Test response"):
    """Create a mock LLMClient."""
    mock = MagicMock()
    if should_succeed:
        mock.chat.return_value = {
            "content": response_content,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "tool_calls": [],
            "structured": False,
        }
    else:
        mock.chat.side_effect = RuntimeError("Provider unavailable")
    mock.close.return_value = None
    return mock


class TestFallbackLLMClientInit:
    """Tests for FallbackLLMClient initialization."""
    
    def test_init_requires_providers(self):
        """Test that initialization fails without providers."""
        empty_registry = ProviderRegistry()
        with pytest.raises(ValueError, match="at least one configured provider"):
            FallbackLLMClient(registry=empty_registry)
    
    def test_init_with_registry(self, mock_registry, mock_health_tracker):
        """Test initialization with a valid registry."""
        client = FallbackLLMClient(
            registry=mock_registry,
            health_tracker=mock_health_tracker,
        )
        assert client._registry == mock_registry
        assert len(mock_registry.providers) == 2
    
    def test_init_logs_providers(self, mock_registry, mock_health_tracker, caplog):
        """Test that initialization logs configured providers."""
        with caplog.at_level("INFO"):
            FallbackLLMClient(
                registry=mock_registry,
                health_tracker=mock_health_tracker,
            )
        assert "gemini:gemini-2.0-flash" in caplog.text
        assert "openai:gpt-4o-mini" in caplog.text


class TestFallbackLLMClientChat:
    """Tests for the chat method with fallback behavior."""
    
    def test_chat_success_primary(self, mock_registry, mock_health_tracker):
        """Test successful chat with primary provider."""
        client = FallbackLLMClient(
            registry=mock_registry,
            health_tracker=mock_health_tracker,
        )
        
        mock_llm = create_mock_client(should_succeed=True, response_content="Hello!")
        
        with patch.object(client, '_get_client', return_value=mock_llm):
            response = client.chat("Hi there")
        
        assert response["content"] == "Hello!"
        assert client.last_used_provider == "gemini"
        assert client.last_was_fallback is False
        assert client.last_attempts == 1
    
    def test_chat_fallback_on_primary_failure(self, mock_registry, mock_health_tracker):
        """Test fallback to secondary when primary fails."""
        client = FallbackLLMClient(
            registry=mock_registry,
            health_tracker=mock_health_tracker,
        )
        
        # First call fails, second succeeds
        call_count = [0]
        
        def mock_get_client(config):
            call_count[0] += 1
            if config.provider == "gemini":
                return create_mock_client(should_succeed=False)
            return create_mock_client(should_succeed=True, response_content="Fallback response")
        
        with patch.object(client, '_get_client', side_effect=mock_get_client):
            response = client.chat("Hi there")
        
        assert response["content"] == "Fallback response"
        assert client.last_used_provider == "openai"
        assert client.last_was_fallback is True
        assert client.last_attempts == 2
    
    def test_chat_all_providers_fail(self, mock_registry, mock_health_tracker):
        """Test RuntimeError when all providers fail."""
        client = FallbackLLMClient(
            registry=mock_registry,
            health_tracker=mock_health_tracker,
        )
        
        mock_llm = create_mock_client(should_succeed=False)
        
        with patch.object(client, '_get_client', return_value=mock_llm):
            with pytest.raises(RuntimeError, match="All.*providers failed"):
                client.chat("Hi there")
    
    def test_chat_return_fallback_result(self, mock_registry, mock_health_tracker):
        """Test returning FallbackResult with metadata."""
        client = FallbackLLMClient(
            registry=mock_registry,
            health_tracker=mock_health_tracker,
        )
        
        mock_llm = create_mock_client(should_succeed=True)
        
        with patch.object(client, '_get_client', return_value=mock_llm):
            result = client.chat("Hi", return_fallback_result=True)
        
        assert isinstance(result, FallbackResult)
        assert result.provider == "gemini"
        assert result.was_fallback is False
        assert result.attempts == 1
    
    def test_chat_all_fail_returns_fallback_result(self, mock_registry, mock_health_tracker):
        """Test FallbackResult on all failures."""
        client = FallbackLLMClient(
            registry=mock_registry,
            health_tracker=mock_health_tracker,
        )
        
        mock_llm = create_mock_client(should_succeed=False)
        
        with patch.object(client, '_get_client', return_value=mock_llm):
            result = client.chat("Hi", return_fallback_result=True)
        
        assert isinstance(result, FallbackResult)
        assert result.content is None
        assert result.error is not None
        assert "failed" in result.error.lower()


class TestFallbackLLMClientHealthTracking:
    """Tests for health tracking integration."""
    
    def test_marks_provider_healthy_on_success(self, mock_registry, mock_health_tracker):
        """Test that successful requests mark provider as healthy."""
        client = FallbackLLMClient(
            registry=mock_registry,
            health_tracker=mock_health_tracker,
        )
        
        # Pre-mark as unhealthy
        mock_health_tracker.mark_unhealthy("gemini", "gemini-2.0-flash", reason="test")
        assert not mock_health_tracker.is_healthy("gemini", "gemini-2.0-flash")
        
        mock_llm = create_mock_client(should_succeed=True)
        
        with patch.object(client, '_get_client', return_value=mock_llm):
            client.chat("Hi")
        
        # Should now be healthy
        assert mock_health_tracker.is_healthy("gemini", "gemini-2.0-flash")
    
    def test_marks_provider_unhealthy_on_failure(self, mock_registry, mock_health_tracker):
        """Test that failed requests mark provider as unhealthy."""
        client = FallbackLLMClient(
            registry=mock_registry,
            health_tracker=mock_health_tracker,
            max_retries_per_provider=1,
        )
        
        def mock_get_client(config):
            if config.provider == "gemini":
                return create_mock_client(should_succeed=False)
            return create_mock_client(should_succeed=True)
        
        with patch.object(client, '_get_client', side_effect=mock_get_client):
            client.chat("Hi")
        
        # Gemini should be unhealthy now
        assert not mock_health_tracker.is_healthy("gemini", "gemini-2.0-flash")
        # OpenAI should be healthy
        assert mock_health_tracker.is_healthy("openai", "gpt-4o-mini")
    
    def test_skips_unhealthy_providers(self, mock_registry, mock_health_tracker):
        """Test that unhealthy providers are tried last."""
        client = FallbackLLMClient(
            registry=mock_registry,
            health_tracker=mock_health_tracker,
        )
        
        # Mark primary as unhealthy
        mock_health_tracker.mark_unhealthy("gemini", "gemini-2.0-flash", reason="test")
        
        providers_tried = []
        
        def mock_get_client(config):
            providers_tried.append(config.provider)
            return create_mock_client(should_succeed=True)
        
        with patch.object(client, '_get_client', side_effect=mock_get_client):
            client.chat("Hi")
        
        # OpenAI should be tried first since gemini is unhealthy
        assert providers_tried[0] == "openai"


class TestFallbackLLMClientRetries:
    """Tests for retry behavior."""
    
    def test_retries_before_fallback(self, mock_registry, mock_health_tracker):
        """Test that provider is retried before moving to fallback."""
        client = FallbackLLMClient(
            registry=mock_registry,
            health_tracker=mock_health_tracker,
            max_retries_per_provider=3,
        )
        
        call_count = [0]
        
        def mock_chat(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:  # Fail twice, succeed on third
                raise RuntimeError("Temporary failure")
            return {
                "content": "Success on retry",
                "usage": {},
                "tool_calls": [],
                "structured": False,
            }
        
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = mock_chat
        
        with patch.object(client, '_get_client', return_value=mock_llm):
            response = client.chat("Hi")
        
        assert response["content"] == "Success on retry"
        assert call_count[0] == 3
        assert client.last_was_fallback is False


class TestFallbackLLMClientFactoryMethods:
    """Tests for factory methods."""
    
    def test_from_config(self):
        """Test creating from config dictionaries."""
        client = FallbackLLMClient.from_config([
            {"provider": "gemini", "api_key": "key1", "model": "gemini-2.0-flash"},
            {"provider": "ollama", "host": "http://localhost:11434", "model": "llama3.2"},
        ])
        
        assert len(client._registry.providers) == 2
        assert client._registry.providers[0].provider == "gemini"
    
    @patch('core_lib.llm.provider_registry.ProviderRegistry.from_env')
    def test_from_env(self, mock_from_env):
        """Test creating from environment."""
        mock_registry = ProviderRegistry()
        mock_registry.add(ProviderConfig(provider="gemini", api_key="test", model="test-model"))
        mock_from_env.return_value = mock_registry
        
        client = FallbackLLMClient.from_env()
        
        assert len(client._registry.providers) == 1
        mock_from_env.assert_called_once()
    
    def test_from_registry(self, mock_registry):
        """Test creating from existing registry."""
        client = FallbackLLMClient.from_registry(
            registry=mock_registry,
            intelligence_level=5,
        )
        
        assert client._registry == mock_registry
        assert client._default_intelligence_level == 5


class TestFallbackResult:
    """Tests for FallbackResult dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dict format."""
        result = FallbackResult(
            content="Hello",
            provider="gemini",
            model="gemini-2.0-flash",
            was_fallback=False,
            attempts=1,
            usage={"tokens": 100},
        )
        
        d = result.to_dict()
        
        assert d["content"] == "Hello"
        assert d["_fallback_metadata"]["provider"] == "gemini"
        assert d["_fallback_metadata"]["was_fallback"] is False
    
    def test_to_dict_with_error(self):
        """Test dict conversion with error."""
        result = FallbackResult(
            content=None,
            provider="",
            model="",
            was_fallback=True,
            attempts=3,
            usage={},
            error="All providers failed",
        )
        
        d = result.to_dict()
        
        assert d["content"] is None
        assert d["error"] == "All providers failed"


class TestCreateFallbackLLMClient:
    """Tests for the create_fallback_llm_client convenience function."""
    
    def test_with_providers(self):
        """Test creation with explicit providers."""
        client = create_fallback_llm_client(
            providers=[
                {"provider": "gemini", "api_key": "key1"},
                {"provider": "openai", "api_key": "key2"},
            ]
        )
        
        assert isinstance(client, FallbackLLMClient)
        assert len(client._registry.providers) == 2
    
    @patch('core_lib.llm.fallback_client.FallbackLLMClient.from_env')
    def test_without_providers_uses_env(self, mock_from_env):
        """Test that missing providers falls back to environment."""
        mock_from_env.return_value = MagicMock(spec=FallbackLLMClient)
        
        create_fallback_llm_client()
        
        mock_from_env.assert_called_once()


class TestFallbackLLMClientContextManager:
    """Tests for context manager support."""
    
    def test_context_manager(self, mock_registry, mock_health_tracker):
        """Test using client as context manager."""
        with FallbackLLMClient(
            registry=mock_registry,
            health_tracker=mock_health_tracker,
        ) as client:
            assert client is not None
        
        # close() should have been called
        assert client._client_cache == {}  # Cache cleared
    
    def test_close_clears_cache(self, mock_registry, mock_health_tracker):
        """Test that close() clears the client cache."""
        client = FallbackLLMClient(
            registry=mock_registry,
            health_tracker=mock_health_tracker,
        )
        
        # Populate cache
        client._client_cache["test"] = MagicMock()
        
        client.close()
        
        assert client._client_cache == {}


# ---------------------------------------------------------------------------
# WoL non-blocking warmup integration with FallbackLLMClient
# ---------------------------------------------------------------------------

class TestFallbackLLMClientWoLWarmup:
    """Tests for non-blocking Wake-on-LAN warmup handling in FallbackLLMClient."""

    def _make_registry_with_ollama_secondary(self):
        """Two-provider registry: Ollama (priority 1) + OpenAI (priority 2)."""
        from core_lib.llm.provider_registry import ProviderConfig, ProviderRegistry
        registry = ProviderRegistry()
        registry.add(ProviderConfig(
            provider="ollama",
            model="qwen3:1.7b",
            host="http://powerspec:11434",
            priority=1,
            wake_on_lan={
                "enabled": True,
                "warmup_seconds": 30,
                "targets": [{"mac_address": "FC:34:97:9E:C8:AF", "wait_seconds": 0}],
            },
        ))
        registry.add(ProviderConfig(
            provider="openai",
            model="gpt-4o-mini",
            api_key="sk-test",
            priority=2,
        ))
        return registry

    def _make_client_mock(self, in_warmup=False, chat_side_effect=None, chat_return=None):
        """Return a MagicMock that mimics LLMClient with is_in_warmup support."""
        mock = MagicMock()
        mock.is_in_warmup.return_value = in_warmup
        if chat_side_effect is not None:
            mock.chat.side_effect = chat_side_effect
        else:
            mock.chat.return_value = chat_return or {
                "content": "ok",
                "usage": {},
                "tool_calls": [],
                "structured": False,
            }
        return mock

    def test_skips_provider_in_warmup(self, mock_health_tracker):
        """FallbackLLMClient must skip a provider whose is_in_warmup() is True."""
        registry = self._make_registry_with_ollama_secondary()
        fallback = FallbackLLMClient(
            registry=registry,
            health_tracker=mock_health_tracker,
        )

        primary_mock = self._make_client_mock(in_warmup=True)
        secondary_mock = self._make_client_mock(chat_return={
            "content": "secondary answer",
            "usage": {},
            "tool_calls": [],
            "structured": False,
        })

        def fake_get_client(config):
            if config.provider == "ollama":
                return primary_mock
            return secondary_mock

        fallback._get_client = fake_get_client

        response = fallback.chat(messages=[{"role": "user", "content": "hello"}])

        assert response["content"] == "secondary answer"
        primary_mock.chat.assert_not_called()
        secondary_mock.chat.assert_called_once()

    def test_does_not_mark_unhealthy_during_warmup(self, mock_health_tracker):
        """Provider should NOT be marked unhealthy when WoL warmup is active."""
        registry = self._make_registry_with_ollama_secondary()
        fallback = FallbackLLMClient(
            registry=registry,
            health_tracker=mock_health_tracker,
        )

        # Primary raises a connection error, then immediately enters warmup
        primary_call_count = {"n": 0}

        def primary_chat(**_kw):
            primary_call_count["n"] += 1
            raise RuntimeError("connection refused")

        primary_mock = self._make_client_mock()
        primary_mock.chat.side_effect = primary_chat

        secondary_mock = self._make_client_mock(chat_return={
            "content": "from secondary",
            "usage": {},
            "tool_calls": [],
            "structured": False,
        })

        # After the exception the provider transitions to warmup
        warmup_states = [False, True]  # first call returns False (not yet in warmup), then True

        primary_mock.is_in_warmup.side_effect = warmup_states.__iter__().__next__

        def fake_get_client(config):
            if config.provider == "ollama":
                return primary_mock
            return secondary_mock

        fallback._get_client = fake_get_client

        response = fallback.chat(messages=[{"role": "user", "content": "hello"}])

        assert response["content"] == "from secondary"
        # Provider should NOT be marked unhealthy (warmup was active)
        health = mock_health_tracker.get_status("ollama", "qwen3:1.7b")
        assert health.is_healthy, (
            "Ollama provider must stay healthy in tracker when warmup handles the failure"
        )

    def test_returns_to_main_after_warmup(self, mock_health_tracker):
        """After warmup expires, FallbackLLMClient must try main provider first."""
        registry = self._make_registry_with_ollama_secondary()
        fallback = FallbackLLMClient(
            registry=registry,
            health_tracker=mock_health_tracker,
        )

        primary_answer = {
            "content": "main server answer",
            "usage": {},
            "tool_calls": [],
            "structured": False,
        }
        # Warmup has already expired (is_in_warmup returns False)
        primary_mock = self._make_client_mock(in_warmup=False, chat_return=primary_answer)
        secondary_mock = self._make_client_mock()

        def fake_get_client(config):
            if config.provider == "ollama":
                return primary_mock
            return secondary_mock

        fallback._get_client = fake_get_client

        response = fallback.chat(messages=[{"role": "user", "content": "hello"}])

        assert response["content"] == "main server answer"
        primary_mock.chat.assert_called_once()
        secondary_mock.chat.assert_not_called()

