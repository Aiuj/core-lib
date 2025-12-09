"""Tests for provider health tracking and YAML configuration loading."""

import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from core_lib.llm.provider_health import (
    ProviderHealthTracker,
    HealthStatus,
    classify_error,
    get_health_tracker,
    reset_health_tracker,
    DEFAULT_UNHEALTHY_TTL,
    FAILURE_TTL_MAP,
)
from core_lib.llm.provider_registry import (
    ProviderConfig,
    ProviderRegistry,
    substitute_env_vars,
)


class TestSubstituteEnvVars:
    """Tests for environment variable substitution."""
    
    def test_substitute_simple_var(self, monkeypatch):
        """Test substitution of a simple ${VAR} pattern."""
        monkeypatch.setenv("TEST_API_KEY", "secret123")
        result = substitute_env_vars("${TEST_API_KEY}")
        assert result == "secret123"
    
    def test_substitute_with_default(self, monkeypatch):
        """Test substitution with default value when var is missing."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        result = substitute_env_vars("${MISSING_VAR:-default_value}")
        assert result == "default_value"
    
    def test_substitute_with_default_when_var_exists(self, monkeypatch):
        """Test that default is not used when var exists."""
        monkeypatch.setenv("EXISTING_VAR", "actual_value")
        result = substitute_env_vars("${EXISTING_VAR:-default_value}")
        assert result == "actual_value"
    
    def test_substitute_in_dict(self, monkeypatch):
        """Test substitution in nested dict."""
        monkeypatch.setenv("API_KEY", "key123")
        monkeypatch.setenv("MODEL", "gpt-4")
        
        data = {
            "api_key": "${API_KEY}",
            "model": "${MODEL}",
            "host": "${HOST:-localhost}",
        }
        result = substitute_env_vars(data)
        
        assert result["api_key"] == "key123"
        assert result["model"] == "gpt-4"
        assert result["host"] == "localhost"
    
    def test_substitute_in_list(self, monkeypatch):
        """Test substitution in list items."""
        monkeypatch.setenv("ITEM1", "first")
        monkeypatch.setenv("ITEM2", "second")
        
        data = ["${ITEM1}", "${ITEM2}", "${ITEM3:-third}"]
        result = substitute_env_vars(data)
        
        assert result == ["first", "second", "third"]
    
    def test_substitute_nested_structure(self, monkeypatch):
        """Test substitution in nested dict/list structure."""
        monkeypatch.setenv("GEMINI_KEY", "gem-key")
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
        
        data = [
            {
                "provider": "gemini",
                "api_key": "${GEMINI_KEY}",
                "model": "${GEMINI_MODEL:-gemini-2.0-flash}",
            },
            {
                "provider": "ollama",
                "host": "${OLLAMA_HOST}",
                "model": "llama3.2",
            },
        ]
        result = substitute_env_vars(data)
        
        assert result[0]["api_key"] == "gem-key"
        assert result[0]["model"] == "gemini-2.0-flash"
        assert result[1]["host"] == "http://localhost:11434"
    
    def test_non_string_values_unchanged(self):
        """Test that non-string values are not modified."""
        data = {
            "number": 42,
            "boolean": True,
            "none_value": None,
        }
        result = substitute_env_vars(data)
        
        assert result["number"] == 42
        assert result["boolean"] is True
        assert result["none_value"] is None


class TestProviderRegistryFromFile:
    """Tests for loading provider config from files."""
    
    def test_from_yaml_file(self, monkeypatch, tmp_path):
        """Test loading providers from YAML file."""
        monkeypatch.setenv("TEST_API_KEY", "yaml-api-key")
        
        yaml_content = """
providers:
  - provider: gemini
    api_key: ${TEST_API_KEY}
    model: gemini-2.0-flash
    priority: 1
  - provider: ollama
    host: ${OLLAMA_HOST:-http://localhost:11434}
    model: llama3.2
    priority: 2
"""
        yaml_file = tmp_path / "providers.yaml"
        yaml_file.write_text(yaml_content)
        
        registry = ProviderRegistry.from_file(str(yaml_file), substitute_env=True)
        
        assert len(registry) == 2
        providers = registry.providers
        assert providers[0].provider == "gemini"
        assert providers[0].api_key == "yaml-api-key"
        assert providers[1].provider == "ollama"
        assert providers[1].host == "http://localhost:11434"
    
    def test_from_json_file(self, monkeypatch, tmp_path):
        """Test loading providers from JSON file."""
        monkeypatch.setenv("JSON_API_KEY", "json-key-123")
        
        json_content = """[
            {"provider": "gemini", "api_key": "${JSON_API_KEY}", "model": "gemini-2.0-flash"}
        ]"""
        json_file = tmp_path / "providers.json"
        json_file.write_text(json_content)
        
        registry = ProviderRegistry.from_file(str(json_file), substitute_env=True)
        
        assert len(registry) == 1
        assert registry.providers[0].api_key == "json-key-123"
    
    def test_from_env_with_file(self, monkeypatch, tmp_path):
        """Test from_env loads from LLM_PROVIDERS_FILE."""
        monkeypatch.setenv("FILE_API_KEY", "from-file")
        
        yaml_content = """
providers:
  - provider: openai
    api_key: ${FILE_API_KEY}
    model: gpt-4o-mini
"""
        yaml_file = tmp_path / "llm.yaml"
        yaml_file.write_text(yaml_content)
        
        monkeypatch.setenv("LLM_PROVIDERS_FILE", str(yaml_file))
        monkeypatch.delenv("LLM_PROVIDERS", raising=False)
        
        registry = ProviderRegistry.from_env()
        
        assert len(registry) == 1
        assert registry.providers[0].provider == "openai"
        assert registry.providers[0].api_key == "from-file"


class TestProviderHealthTracker:
    """Tests for provider health tracking."""
    
    @pytest.fixture
    def tracker(self):
        """Create a fresh health tracker for each test."""
        reset_health_tracker()
        return ProviderHealthTracker(unhealthy_ttl=60, cache_client=None)
    
    def test_new_provider_is_healthy(self, tracker):
        """Test that a provider with no status is considered healthy."""
        assert tracker.is_healthy("gemini", "gemini-2.0-flash") is True
    
    def test_mark_unhealthy(self, tracker):
        """Test marking a provider as unhealthy."""
        tracker.mark_unhealthy("gemini", "gemini-2.0-flash", reason="rate_limit")
        
        assert tracker.is_healthy("gemini", "gemini-2.0-flash") is False
    
    def test_mark_healthy_clears_status(self, tracker):
        """Test that marking healthy clears unhealthy status."""
        tracker.mark_unhealthy("gemini", "gemini-2.0-flash")
        assert tracker.is_healthy("gemini", "gemini-2.0-flash") is False
        
        tracker.mark_healthy("gemini", "gemini-2.0-flash")
        assert tracker.is_healthy("gemini", "gemini-2.0-flash") is True
    
    def test_get_status_healthy(self, tracker):
        """Test getting status for healthy provider."""
        status = tracker.get_status("gemini", "gemini-2.0-flash")
        
        assert status.is_healthy is True
        assert status.failure_reason is None
    
    def test_get_status_unhealthy(self, tracker):
        """Test getting status for unhealthy provider."""
        tracker.mark_unhealthy("gemini", "gemini-2.0-flash", reason="timeout")
        
        status = tracker.get_status("gemini", "gemini-2.0-flash")
        
        assert status.is_healthy is False
        assert status.failure_reason == "timeout"
        assert status.recovery_at is not None
    
    def test_filter_healthy(self, tracker):
        """Test filtering providers to only healthy ones."""
        providers = [
            ProviderConfig(provider="gemini", model="gemini-2.0-flash", api_key="k1"),
            ProviderConfig(provider="openai", model="gpt-4o-mini", api_key="k2"),
            ProviderConfig(provider="ollama", model="llama3.2"),
        ]
        
        # Mark one as unhealthy
        tracker.mark_unhealthy("openai", "gpt-4o-mini", reason="rate_limit")
        
        healthy = tracker.filter_healthy(providers)
        
        assert len(healthy) == 2
        assert all(p.provider != "openai" for p in healthy)
    
    def test_get_first_healthy(self, tracker):
        """Test getting first healthy provider."""
        providers = [
            ProviderConfig(provider="gemini", model="flash", api_key="k1", priority=1),
            ProviderConfig(provider="openai", model="mini", api_key="k2", priority=2),
        ]
        
        # Mark primary as unhealthy
        tracker.mark_unhealthy("gemini", "flash")
        
        first = tracker.get_first_healthy(providers)
        
        assert first is not None
        assert first.provider == "openai"


class TestClassifyError:
    """Tests for error classification."""
    
    def test_rate_limit_detection(self):
        """Test detection of rate limit errors."""
        error = Exception("Rate limit exceeded: 429 Too Many Requests")
        assert classify_error(error) == "rate_limit"
    
    def test_quota_exceeded_detection(self):
        """Test detection of quota exceeded errors."""
        error = Exception("API quota exceeded for this billing period")
        assert classify_error(error) == "quota_exceeded"
    
    def test_timeout_detection(self):
        """Test detection of timeout errors."""
        error = Exception("Request timed out after 30 seconds")
        assert classify_error(error) == "timeout"
    
    def test_connection_error_detection(self):
        """Test detection of connection errors."""
        error = Exception("Connection refused: server unreachable")
        assert classify_error(error) == "connection_error"
    
    def test_auth_error_detection(self):
        """Test detection of auth errors."""
        error = Exception("401 Unauthorized: Invalid API key")
        assert classify_error(error) == "auth_error"
    
    def test_server_error_detection(self):
        """Test detection of server errors."""
        error = Exception("500 Internal Server Error")
        assert classify_error(error) == "server_error"
    
    def test_unknown_error(self):
        """Test that unknown errors are classified as unknown."""
        error = Exception("Something completely unexpected happened")
        assert classify_error(error) == "unknown"


class TestProviderRegistryHealthAware:
    """Tests for health-aware provider selection in ProviderRegistry."""
    
    @pytest.fixture
    def registry(self):
        """Create a registry with multiple providers."""
        reset_health_tracker()
        registry = ProviderRegistry()
        registry.add(ProviderConfig(provider="gemini", model="flash", api_key="k1", priority=1))
        registry.add(ProviderConfig(provider="openai", model="mini", api_key="k2", priority=2))
        registry.add(ProviderConfig(provider="ollama", model="llama3.2", priority=3))
        return registry
    
    def test_get_healthy_providers_all_healthy(self, registry):
        """Test that all providers are returned when all healthy."""
        healthy = registry.get_healthy_providers()
        assert len(healthy) == 3
    
    def test_get_healthy_providers_excludes_unhealthy(self, registry):
        """Test that unhealthy providers are excluded."""
        registry.mark_unhealthy("gemini", "flash", reason="rate_limit")
        
        healthy = registry.get_healthy_providers()
        
        assert len(healthy) == 2
        assert all(p.provider != "gemini" for p in healthy)
    
    def test_iter_clients_with_fallback_healthy_first(self, registry):
        """Test that healthy clients are yielded first."""
        # Mark primary as unhealthy
        registry.mark_unhealthy("gemini", "flash")
        
        # Get the order of providers
        provider_order = []
        for client, is_fallback, provider_config in registry.iter_clients_with_fallback():
            provider_order.append(provider_config.provider)
        
        # Healthy providers should come first
        assert provider_order[0] == "openai"  # First healthy
        assert "gemini" in provider_order  # Unhealthy still included as fallback
        assert provider_order.index("gemini") > 0  # But after healthy ones
    
    def test_mark_healthy_via_registry(self, registry):
        """Test marking healthy via registry method."""
        registry.mark_unhealthy("gemini", "flash")
        assert len(registry.get_healthy_providers()) == 2
        
        registry.mark_healthy("gemini", "flash")
        assert len(registry.get_healthy_providers()) == 3
    
    def test_mark_unhealthy_with_exception(self, registry):
        """Test marking unhealthy with exception classification."""
        error = Exception("429 Rate limit exceeded")
        registry.mark_unhealthy("gemini", "flash", error=error)
        
        tracker = registry._get_health_tracker()
        status = tracker.get_status("gemini", "flash")
        
        assert status.failure_reason == "rate_limit"
