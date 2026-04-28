"""Tests for FallbackEmbeddingClient health status caching.

Tests that the fallback client properly caches provider health status,
avoids repeatedly trying failed providers, and gracefully handles all-provider
failure scenarios.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time

from core_lib.embeddings.fallback_client import FallbackEmbeddingClient
from core_lib.embeddings.base import EmbeddingGenerationError


class MockCache:
    """Mock cache for testing health status tracking."""
    
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


def test_health_cache_remembers_successful_provider():
    """Test that successful provider is remembered in cache."""
    mock_cache = MockCache()
    
    provider1 = create_mock_provider("provider1", should_succeed=False)
    provider2 = create_mock_provider("provider2", should_succeed=True)
    provider3 = create_mock_provider("provider3", should_succeed=True)
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient(
            providers=[provider1, provider2, provider3],
            use_health_cache=True,
            health_check_interval=60,
        )
        
        # First request - should try provider1 (fails), then provider2 (succeeds)
        result = client.generate_embedding("test text")
        assert result is not None
        assert provider1._generate_embedding_raw.call_count == 1
        assert provider2._generate_embedding_raw.call_count == 1
        assert provider3._generate_embedding_raw.call_count == 0  # Not reached
        
        # Check cache - provider2 should be marked healthy and preferred
        preferred_key = f"embedding:fallback:{client._client_id}:preferred_provider"
        assert mock_cache.get(preferred_key) == "1"  # Provider index 1
        
        provider2_health_key = f"embedding:fallback:{client._client_id}:provider:1:healthy"
        assert mock_cache.get(provider2_health_key) == "1"
        
        # Second request - should go directly to provider2 (cached as healthy)
        provider1._generate_embedding_raw.reset_mock()
        provider2._generate_embedding_raw.reset_mock()
        
        result = client.generate_embedding("test text 2")
        assert result is not None
        assert provider1._generate_embedding_raw.call_count == 0  # Skipped!
        assert provider2._generate_embedding_raw.call_count == 1  # Used directly


def test_health_cache_skips_known_failed_providers():
    """Test that providers marked as failed are skipped temporarily."""
    mock_cache = MockCache()
    
    provider1 = create_mock_provider("provider1", should_succeed=False)
    provider2 = create_mock_provider("provider2", should_succeed=False)
    provider3 = create_mock_provider("provider3", should_succeed=True)
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient(
            providers=[provider1, provider2, provider3],
            use_health_cache=True,
            health_check_interval=60,
        )
        
        # First request - tries all until success
        result = client.generate_embedding("test")
        assert result is not None
        
        # Provider 1 and 2 should be marked temporarily unhealthy in cache
        provider1_health_key = f"embedding:fallback:{client._client_id}:provider:0:healthy"
        provider2_health_key = f"embedding:fallback:{client._client_id}:provider:1:healthy"
        assert mock_cache.get(provider1_health_key) == "0"
        assert mock_cache.get(provider2_health_key) == "0"
        
        # Provider 3 should be marked healthy
        provider3_health_key = f"embedding:fallback:{client._client_id}:provider:2:healthy"
        assert mock_cache.get(provider3_health_key) == "1"
        
        # Second request - should skip providers 1 and 2 (recently failed)
        # and go directly to provider 3
        provider1._generate_embedding_raw.reset_mock()
        provider2._generate_embedding_raw.reset_mock()
        provider3._generate_embedding_raw.reset_mock()
        
        result = client.generate_embedding("test 2")
        assert result is not None
        
        # Providers 1 and 2 should NOT have been tried (within health_check_interval)
        assert provider1._generate_embedding_raw.call_count == 0
        assert provider2._generate_embedding_raw.call_count == 0
        assert provider3._generate_embedding_raw.call_count == 1


def test_all_providers_fail_gracefully():
    """Test that when all providers fail, error is handled gracefully."""
    mock_cache = MockCache()
    
    provider1 = create_mock_provider("provider1", should_succeed=False)
    provider2 = create_mock_provider("provider2", should_succeed=False)
    provider3 = create_mock_provider("provider3", should_succeed=False)
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        # Test with fail_on_all_providers=True (default)
        client = FallbackEmbeddingClient(
            providers=[provider1, provider2, provider3],
            use_health_cache=True,
            fail_on_all_providers=True,
        )
        
        with pytest.raises(EmbeddingGenerationError) as exc_info:
            client.generate_embedding("test")
        
        assert "All 3 embedding providers failed" in str(exc_info.value)
        assert "tried 3 providers" in str(exc_info.value)
        
        # Each provider should have been tried exactly once
        assert provider1._generate_embedding_raw.call_count == 1
        assert provider2._generate_embedding_raw.call_count == 1
        assert provider3._generate_embedding_raw.call_count == 1


def test_all_providers_fail_returns_none():
    """Test that with fail_on_all_providers=False, None is returned."""
    mock_cache = MockCache()
    
    provider1 = create_mock_provider("provider1", should_succeed=False)
    provider2 = create_mock_provider("provider2", should_succeed=False)
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient(
            providers=[provider1, provider2],
            use_health_cache=True,
            fail_on_all_providers=False,  # Return None instead of raising
        )
        
        result = client.generate_embedding("test")
        assert result is None


def test_no_infinite_loop_on_all_failures():
    """Test that we don't loop infinitely when all providers fail."""
    mock_cache = MockCache()
    
    provider1 = create_mock_provider("provider1", should_succeed=False)
    provider2 = create_mock_provider("provider2", should_succeed=False)
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient(
            providers=[provider1, provider2],
            use_health_cache=True,
            max_retries_per_provider=2,
        )
        
        with pytest.raises(EmbeddingGenerationError):
            client.generate_embedding("test")
        
        # Each provider should be tried exactly max_retries_per_provider times
        assert provider1._generate_embedding_raw.call_count == 2
        assert provider2._generate_embedding_raw.call_count == 2


def test_health_check_interval_respected():
    """Test that failed providers are retried after health_check_interval."""
    mock_cache = MockCache()
    
    provider1 = create_mock_provider("provider1", should_succeed=False)
    provider2 = create_mock_provider("provider2", should_succeed=True)
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient(
            providers=[provider1, provider2],
            use_health_cache=True,
            health_check_interval=1,  # 1 second
        )
        
        # First request - provider1 fails, provider2 succeeds
        result = client.generate_embedding("test")
        assert result is not None
        assert provider1._generate_embedding_raw.call_count == 1
        
        # Second request immediately - provider1 should be skipped
        provider1._generate_embedding_raw.reset_mock()
        provider2._generate_embedding_raw.reset_mock()
        
        result = client.generate_embedding("test 2")
        assert provider1._generate_embedding_raw.call_count == 0  # Skipped
        assert provider2._generate_embedding_raw.call_count == 1
        
        # Wait for health check interval to pass
        time.sleep(1.1)
        
        # Now provider1 can be tried again (but will still fail)
        provider1._generate_embedding_raw.reset_mock()
        provider2._generate_embedding_raw.reset_mock()
        
        result = client.generate_embedding("test 3")
        assert provider1._generate_embedding_raw.call_count == 1  # Tried again!
        assert provider2._generate_embedding_raw.call_count == 1


def test_without_cache_works_normally():
    """Test that fallback works correctly when cache is not available."""
    provider1 = create_mock_provider("provider1", should_succeed=False)
    provider2 = create_mock_provider("provider2", should_succeed=True)
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=None):
        client = FallbackEmbeddingClient(
            providers=[provider1, provider2],
            use_health_cache=True,  # Enabled but cache not available
        )
        
        # Should still work, just without caching optimization
        result = client.generate_embedding("test")
        assert result is not None
        assert provider1._generate_embedding_raw.call_count == 1
        assert provider2._generate_embedding_raw.call_count == 1


def test_health_cache_disabled():
    """Test that health caching can be disabled."""
    mock_cache = MockCache()
    
    provider1 = create_mock_provider("provider1", should_succeed=False)
    provider2 = create_mock_provider("provider2", should_succeed=True)
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient(
            providers=[provider1, provider2],
            use_health_cache=False,  # Explicitly disabled
        )
        
        result = client.generate_embedding("test")
        assert result is not None
        
        # Cache should be empty (health caching disabled)
        assert len(mock_cache.data) == 0


def test_force_provider():
    """Test forcing use of a specific provider."""
    mock_cache = MockCache()
    
    provider1 = create_mock_provider("provider1", should_succeed=True)
    provider2 = create_mock_provider("provider2", should_succeed=True)
    provider3 = create_mock_provider("provider3", should_succeed=True)
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient(
            providers=[provider1, provider2, provider3],
            use_health_cache=True,
        )
        
        # Force provider 2
        client.force_provider(2)
        
        # Should use provider 2 directly
        result = client.generate_embedding("test")
        assert result is not None
        assert provider1._generate_embedding_raw.call_count == 0
        assert provider2._generate_embedding_raw.call_count == 0
        assert provider3._generate_embedding_raw.call_count == 1
        
        # Provider 2 should be marked healthy in cache
        provider2_health_key = f"embedding:fallback:{client._client_id}:provider:2:healthy"
        assert mock_cache.get(provider2_health_key) == "1"


def test_reset_failures_clears_cache():
    """Test that reset_failures clears health cache."""
    mock_cache = MockCache()
    
    provider1 = create_mock_provider("provider1", should_succeed=False)
    provider2 = create_mock_provider("provider2", should_succeed=True)
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient(
            providers=[provider1, provider2],
            use_health_cache=True,
        )
        
        # Generate embedding - marks provider2 as healthy
        result = client.generate_embedding("test")
        assert result is not None
        
        # Cache should have entries
        assert len(mock_cache.data) > 0
        
        # Reset failures
        client.reset_failures()
        
        # Cache should be cleared
        assert len(mock_cache.data) == 0
        assert client.provider_failures == {0: 0, 1: 0}


def test_get_provider_stats_includes_health():
    """Test that provider stats include cached health status."""
    mock_cache = MockCache()
    
    provider1 = create_mock_provider("provider1", should_succeed=False)
    provider2 = create_mock_provider("provider2", should_succeed=True)
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient(
            providers=[provider1, provider2],
            use_health_cache=True,
        )
        
        # Generate embedding
        client.generate_embedding("test")
        
        # Get stats
        stats = client.get_provider_stats()
        
        assert stats["health_cache_enabled"] is True
        assert stats["total_providers"] == 2
        assert "preferred_provider" in stats
        assert stats["preferred_provider"] == 1  # Provider 2
        
        # Check provider-specific stats
        provider0_stats = stats["providers"][0]
        assert provider0_stats["cached_healthy"] is None  # Failed, not cached as healthy
        assert provider0_stats["failures"] > 0
        
        provider1_stats = stats["providers"][1]
        assert provider1_stats["cached_healthy"] is True  # Succeeded, cached as healthy
        assert provider1_stats["failures"] == 0


def test_provider_recovery_after_failure():
    """Test that a provider can recover after failing."""
    mock_cache = MockCache()
    
    # Provider that initially fails but then recovers
    provider1 = create_mock_provider("provider1", should_succeed=False)
    provider2 = create_mock_provider("provider2", should_succeed=True)
    
    with patch('core_lib.embeddings.fallback_client.get_cache', return_value=mock_cache):
        client = FallbackEmbeddingClient(
            providers=[provider1, provider2],
            use_health_cache=True,
            health_check_interval=1,
        )
        
        # First request - provider1 fails, provider2 succeeds
        result = client.generate_embedding("test")
        assert result is not None
        
        # Now "fix" provider1
        provider1._generate_embedding_raw.side_effect = None
        provider1._generate_embedding_raw.return_value = [[0.1] * 384]
        provider1.health_check.return_value = True
        
        # Wait for health check interval
        time.sleep(1.1)
        
        # Provider1 should now work when retried
        provider1._generate_embedding_raw.reset_mock()
        provider2._generate_embedding_raw.reset_mock()
        
        result = client.generate_embedding("test 2")
        assert result is not None
        
        # Provider1 should have been tried and succeeded
        assert provider1._generate_embedding_raw.call_count == 1
