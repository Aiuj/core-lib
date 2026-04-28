"""
Tests for overload detection and automatic recovery in FallbackEmbeddingClient.

Verifies that:
- HTTP 503/429 and timeout errors are classified as temporary overload
- Overloaded providers get shorter TTL (30s) for faster recovery
- Primary server automatically becomes preferred again when it recovers
- Overload and failure counters are tracked separately
"""

import pytest
import time
from unittest.mock import patch, Mock, MagicMock
import requests
from core_lib.embeddings.fallback_client import FallbackEmbeddingClient


class MockCache:
    """Simple in-memory cache for testing."""
    
    def __init__(self):
        self.data = {}
    
    def get(self, key):
        """Get value from cache."""
        item = self.data.get(key)
        if item is None:
            return None
        value, expiry = item
        if expiry and time.time() > expiry:
            del self.data[key]
            return None
        return value
    
    def set(self, key, value, ttl=None):
        """Set value in cache with optional TTL."""
        expiry = time.time() + ttl if ttl else None
        self.data[key] = (value, expiry)
    
    def delete(self, key):
        """Delete key from cache."""
        self.data.pop(key, None)


def create_mock_provider(name="mock", should_succeed=True, embedding_dim=384):
    """Create a mock embedding provider."""
    provider = Mock()
    provider.model = f"model-{name}"
    provider.embedding_dim = embedding_dim
    provider.base_url = f"http://{name}:7997"
    
    if should_succeed:
        provider._generate_embedding_raw.return_value = [[0.1] * embedding_dim]
        provider.health_check.return_value = True
    else:
        provider._generate_embedding_raw.side_effect = Exception(f"{name} failed")
        provider.health_check.return_value = False
    
    return provider


def test_http_503_triggers_overload():
    """HTTP 503 error should mark provider as overloaded, not failed."""
    mock_cache = MockCache()
    provider1 = create_mock_provider("primary")
    provider2 = create_mock_provider("secondary")
    
    # Simulate 503 error from primary
    error_503 = requests.exceptions.HTTPError(response=MagicMock(status_code=503))
    provider1._generate_embedding_raw.side_effect = error_503
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient([provider1, provider2], use_health_cache=True)
        
        # Should fallback to secondary
        result = client.generate_embedding("test")
        assert result is not None
        
        # Primary should be marked as overloaded, not failed
        stats = client.get_provider_stats()
        assert stats["providers"][0]["overloads"] == 1
        assert stats["providers"][0]["failures"] == 0
        assert stats["providers"][0]["cached_overloaded"] is True


def test_http_429_triggers_overload():
    """HTTP 429 error should mark provider as overloaded."""
    mock_cache = MockCache()
    provider1 = create_mock_provider("primary")
    provider2 = create_mock_provider("secondary")
    
    error_429 = requests.exceptions.HTTPError(response=MagicMock(status_code=429))
    provider1._generate_embedding_raw.side_effect = error_429
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient([provider1, provider2], use_health_cache=True)
        result = client.generate_embedding("test")
        assert result is not None
        
        stats = client.get_provider_stats()
        assert stats["providers"][0]["overloads"] == 1
        assert stats["providers"][0]["failures"] == 0


def test_timeout_triggers_overload():
    """Timeout error should mark provider as overloaded."""
    mock_cache = MockCache()
    provider1 = create_mock_provider("primary")
    provider2 = create_mock_provider("secondary")
    
    provider1._generate_embedding_raw.side_effect = requests.exceptions.Timeout("Connection timed out")
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient([provider1, provider2], use_health_cache=True)
        result = client.generate_embedding("test")
        assert result is not None
        
        stats = client.get_provider_stats()
        assert stats["providers"][0]["overloads"] == 1
        assert stats["providers"][0]["failures"] == 0


def test_success_clears_stale_overload_status():
    """A recovered provider should not remain cached as overloaded."""
    mock_cache = MockCache()
    provider1 = create_mock_provider("primary")
    provider2 = create_mock_provider("secondary")

    provider1._generate_embedding_raw.return_value = [[0.2] * 384]

    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient([provider1, provider2], use_health_cache=True)
        client._mark_provider_overloaded(0)
        client._last_health_check[0] = time.time() - 31
        assert client.get_provider_stats()["providers"][0]["cached_overloaded"] is True

        assert client.generate_embedding("test recovered") is not None
        assert client.get_provider_stats()["providers"][0]["cached_overloaded"] is False


def test_connection_pool_exhausted_triggers_overload():
    """Connection pool exhausted error should mark provider as overloaded."""
    mock_cache = MockCache()
    provider1 = create_mock_provider("primary")
    provider2 = create_mock_provider("secondary")
    
    error = Exception("Connection pool is exhausted")
    provider1._generate_embedding_raw.side_effect = error
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient([provider1, provider2], use_health_cache=True)
        result = client.generate_embedding("test")
        assert result is not None
        
        stats = client.get_provider_stats()
        assert stats["providers"][0]["overloads"] == 1
        assert stats["providers"][0]["failures"] == 0


def test_non_overload_error_triggers_failure():
    """Non-overload errors (500, 404, etc.) should mark as failed."""
    mock_cache = MockCache()
    provider1 = create_mock_provider("primary")
    provider2 = create_mock_provider("secondary")
    
    error_500 = requests.exceptions.HTTPError(response=MagicMock(status_code=500))
    provider1._generate_embedding_raw.side_effect = error_500
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient([provider1, provider2], use_health_cache=True)
        result = client.generate_embedding("test")
        assert result is not None
        
        stats = client.get_provider_stats()
        assert stats["providers"][0]["failures"] == 1
        assert stats["providers"][0]["overloads"] == 0


def test_overload_and_failure_tracked_separately():
    """Overload and failure counters should be independent."""
    mock_cache = MockCache()
    provider1 = create_mock_provider("primary")
    provider2 = create_mock_provider("secondary")
    
    # First: overload (503), then failure (500)
    provider1._generate_embedding_raw.side_effect = [
        requests.exceptions.HTTPError(response=MagicMock(status_code=503)),
        requests.exceptions.HTTPError(response=MagicMock(status_code=500)),
    ]
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient([provider1, provider2], use_health_cache=True)
        
        # First call: overload
        client.generate_embedding("test")
        stats = client.get_provider_stats()
        assert stats["providers"][0]["overloads"] == 1
        assert stats["providers"][0]["failures"] == 0
        
        # Second call: failure
        client.generate_embedding("test")
        stats = client.get_provider_stats()
        assert stats["providers"][0]["overloads"] == 1
        assert stats["providers"][0]["failures"] == 1


def test_multiple_overloads_increments_counter():
    """Multiple overload events should increment counter."""
    mock_cache = MockCache()
    provider1 = create_mock_provider("primary")
    provider2 = create_mock_provider("secondary")
    
    # Simulate 3 overload events
    provider1._generate_embedding_raw.side_effect = requests.exceptions.HTTPError(
        response=MagicMock(status_code=503)
    )
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient([provider1, provider2], use_health_cache=True)
        
        for _ in range(3):
            client.generate_embedding("test")
        
        stats = client.get_provider_stats()
        assert stats["providers"][0]["overloads"] == 3
        assert stats["providers"][0]["failures"] == 0


def test_reset_clears_overload_status():
    """reset_failures() should clear overload counters and cache."""
    mock_cache = MockCache()
    provider1 = create_mock_provider("primary")
    provider2 = create_mock_provider("secondary")
    
    provider1._generate_embedding_raw.side_effect = requests.exceptions.HTTPError(
        response=MagicMock(status_code=503)
    )
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient([provider1, provider2], use_health_cache=True)
        
        client.generate_embedding("test")
        
        stats = client.get_provider_stats()
        assert stats["providers"][0]["overloads"] == 1
        assert stats["providers"][0]["cached_overloaded"] is True
        
        # Reset
        client.reset_failures()
        
        # Should be cleared
        stats = client.get_provider_stats()
        assert stats["providers"][0]["overloads"] == 0
        assert stats["providers"][0]["cached_overloaded"] is False


def test_all_providers_overloaded_returns_none():
    """When all providers are overloaded, should return None gracefully."""
    mock_cache = MockCache()
    provider1 = create_mock_provider("primary")
    provider2 = create_mock_provider("secondary")
    
    # Both providers overloaded
    provider1._generate_embedding_raw.side_effect = requests.exceptions.HTTPError(
        response=MagicMock(status_code=503)
    )
    provider2._generate_embedding_raw.side_effect = requests.exceptions.HTTPError(
        response=MagicMock(status_code=503)
    )
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient([provider1, provider2], use_health_cache=True)
        
        result = client.generate_embedding("test")
        assert result is None
        
        # Both should be marked as overloaded
        stats = client.get_provider_stats()
        assert stats["providers"][0]["overloads"] == 1
        assert stats["providers"][1]["overloads"] == 1
