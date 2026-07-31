"""
Tests for embedding factory integration with model-aware normalization.
"""

import pytest
from unittest.mock import Mock, patch

from core_lib.embeddings import (
    EmbeddingFactory,
    create_embedding_client,
    get_model_dimension,
    get_best_normalization_method,
)
from core_lib.embeddings.ollama import OllamaEmbeddingClient


class TestEmbeddingFactoryNormalization:
    """Tests for factory integration with intelligent normalization."""
    
    def test_ollama_client_auto_normalization_matryoshka(self):
        """Test Ollama client with Matryoshka model auto-detects truncate_or_pad."""
        client = EmbeddingFactory.ollama(
            model="nomic-embed-text-v1.5",
            embedding_dim=512,
        )
        
        assert client.model == "nomic-embed-text-v1.5"
        assert client.embedding_dim == 512
        assert client.model_native_dim == 768  # From database
        assert client.norm_method == "truncate_or_pad"  # Matryoshka model
    
    def test_ollama_client_auto_normalization_non_matryoshka(self):
        """Test Ollama client with non-Matryoshka model uses interpolate."""
        client = EmbeddingFactory.ollama(
            model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=256,
        )
        
        assert client.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert client.embedding_dim == 256
        assert client.model_native_dim == 384  # From database
        # Large reduction without Matryoshka -> interpolate or pca_approximate
        assert client.norm_method in ["interpolate", "pca_approximate"]
    
    def test_ollama_client_explicit_normalization_method(self):
        """Test explicit normalization method overrides auto-detection."""
        client = EmbeddingFactory.ollama(
            model="nomic-embed-text-v1.5",
            embedding_dim=512,
            norm_method="interpolate",  # Override auto-detection
        )
        
        assert client.norm_method == "interpolate"  # Explicit value used
    
    def test_ollama_client_unknown_model(self):
        """Test client with unknown model falls back to default."""
        client = EmbeddingFactory.ollama(
            model="unknown-custom-model",
            embedding_dim=768,
        )
        
        assert client.model == "unknown-custom-model"
        assert client.model_native_dim is None  # Not in database
        assert client.norm_method == "interpolate"  # Default
    
    def test_client_without_target_dimension(self):
        """Test client without explicit target dimension."""
        client = EmbeddingFactory.ollama(
            model="nomic-embed-text-v1.5",
            embedding_dim=None,  # No target dimension
        )
        
        # Should still work, just no dimension normalization
        assert client.embedding_dim is None
        assert client.model_native_dim == 768


class TestModelDatabase:
    """Tests for model database queries."""
    
    def test_get_model_dimension_openai(self):
        """Test getting dimensions for OpenAI models."""
        assert get_model_dimension("text-embedding-3-small") == 1536
        assert get_model_dimension("text-embedding-3-large") == 3072
        assert get_model_dimension("text-embedding-ada-002") == 1536
    
    def test_get_model_dimension_nomic(self):
        """Test getting dimensions for Nomic models."""
        assert get_model_dimension("nomic-embed-text") == 768
        assert get_model_dimension("nomic-embed-text-v1.5") == 768
    
    def test_get_model_dimension_fuzzy_match(self):
        """Test fuzzy matching for model names."""
        # Should match with case-insensitive
        assert get_model_dimension("NOMIC-EMBED-TEXT") == 768
        assert get_model_dimension("Text-Embedding-3-Small") == 1536
    
    def test_get_model_dimension_unknown(self):
        """Test unknown model returns None."""
        assert get_model_dimension("unknown-model") is None
        assert get_model_dimension("") is None


class TestEndToEndNormalization:
    """End-to-end tests for normalization pipeline."""
    
    @patch('core_lib.embeddings.ollama.ollama')
    def test_embedding_generation_with_normalization(self, mock_ollama):
        """Test complete embedding generation with dimension normalization."""
        # Mock ollama.embed to return 768-dim embedding
        mock_response = {
            'embeddings': [[0.1] * 768]
        }
        mock_ollama.Client.return_value.embed.return_value = mock_response
        
        # Create client that should normalize to 512 dimensions
        client = EmbeddingFactory.ollama(
            model="nomic-embed-text-v1.5",
            embedding_dim=512,
            use_l2_norm=False,  # Disable L2 for simpler testing
        )
        client.cache_duration_seconds = 0
        
        # Generate embedding
        result = client.generate_embedding("test text")
        
        # Should be normalized to target dimension
        assert len(result) == 512
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)
    
    @patch('core_lib.embeddings.ollama.ollama')
    def test_embedding_with_l2_and_dimension_norm(self, mock_ollama):
        """Test embedding generation with both L2 and dimension normalization."""
        import numpy as np
        
        # Mock ollama.embed to return 768-dim embedding
        mock_response = {
            'embeddings': [[0.5] * 768]
        }
        mock_ollama.Client.return_value.embed.return_value = mock_response
        
        # Create client with both normalizations enabled
        client = EmbeddingFactory.ollama(
            model="nomic-embed-text-v1.5",
            embedding_dim=512,
            use_l2_norm=True,  # Enable L2 normalization
        )
        client.cache_duration_seconds = 0
        
        # Generate embedding
        result = client.generate_embedding("test text")
        
        # Should be normalized to target dimension
        assert len(result) == 512
        
        # Should be L2 normalized (unit vector)
        norm = np.linalg.norm(result)
        assert 0.99 <= norm <= 1.01  # Allow small floating point error


class TestConvenienceFunctions:
    """Test convenience functions with normalization."""
    
    def test_create_embedding_client_with_norm_method(self):
        """Test create_embedding_client with explicit norm_method."""
        client = create_embedding_client(
            provider="ollama",
            model="nomic-embed-text-v1.5",
            embedding_dim=512,
            norm_method="interpolate",
        )
        
        assert client.norm_method == "interpolate"
    
    def test_create_embedding_client_auto_norm(self):
        """Test create_embedding_client with auto-detection."""
        client = create_embedding_client(
            provider="ollama",
            model="text-embedding-3-large",
            embedding_dim=1024,
        )
        
        # Should auto-detect Matryoshka and use truncate_or_pad
        assert client.norm_method == "truncate_or_pad"


class TestCacheKeyIntegration:
    """Test that cache keys include normalization method."""
    
    def test_cache_key_includes_norm_method(self):
        """Test that different norm methods generate different cache keys."""
        client1 = EmbeddingFactory.ollama(
            model="nomic-embed-text",
            embedding_dim=512,
            norm_method="truncate_or_pad",
        )
        
        client2 = EmbeddingFactory.ollama(
            model="nomic-embed-text",
            embedding_dim=512,
            norm_method="interpolate",
        )
        
        # Same text but different norm methods should have different cache keys
        key1 = client1._generate_cache_key("test text")
        key2 = client2._generate_cache_key("test text")
        
        assert key1 != key2
    
    def test_cache_key_same_for_identical_config(self):
        """Test that identical configurations generate same cache key."""
        client1 = EmbeddingFactory.ollama(
            model="nomic-embed-text",
            embedding_dim=512,
            norm_method="truncate_or_pad",
        )
        
        client2 = EmbeddingFactory.ollama(
            model="nomic-embed-text",
            embedding_dim=512,
            norm_method="truncate_or_pad",
        )
        
        key1 = client1._generate_cache_key("test text")
        key2 = client2._generate_cache_key("test text")
        
        assert key1 == key2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
