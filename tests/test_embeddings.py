"""Tests for embedding functionality."""

import pytest
from unittest.mock import patch, MagicMock
from typing import List

from core_lib.embeddings import (
    BaseEmbeddingClient,
    EmbeddingGenerationError,
    EmbeddingsConfig,
    TaskType,
    EmbeddingFactory,
    create_embedding_client,
    create_client_from_env,
    create_openai_client,
    create_google_genai_client,
    create_ollama_client,
    create_local_client,
)


class TestEmbeddingsConfig:
    """Test embedding configuration."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = EmbeddingsConfig()
        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"
        assert config.embedding_dimension is None
        assert config.task_type is None
    
    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict('os.environ', {
            'EMBEDDING_PROVIDER': 'google_genai',
            'EMBEDDING_MODEL': 'text-embedding-004',
            'EMBEDDING_DIMENSION': '1024',
            'EMBEDDING_TASK_TYPE': 'SEMANTIC_SIMILARITY',
            'GOOGLE_GENAI_API_KEY': 'test-key',
        }):
            config = EmbeddingsConfig.from_env()
            assert config.provider == "google_genai"
            assert config.model == "text-embedding-004"
            assert config.embedding_dimension == 1024
            assert config.task_type == "SEMANTIC_SIMILARITY"
            assert config.google_api_key == "test-key"

    def test_task_type_enum(self):
        """Test TaskType enum values."""
        assert TaskType.SEMANTIC_SIMILARITY == "SEMANTIC_SIMILARITY"
        assert TaskType.CLASSIFICATION == "CLASSIFICATION"
        assert TaskType.CLUSTERING == "CLUSTERING"


class MockOpenAIEmbeddingClient(BaseEmbeddingClient):
    """Mock OpenAI embedding client for testing."""
    
    def __init__(self, model=None, embedding_dim=None, use_l2_norm=True, **kwargs):
        super().__init__(model=model, embedding_dim=embedding_dim, use_l2_norm=use_l2_norm)
        # Store other parameters for testing but don't pass to parent
        self.api_key = kwargs.get('api_key')
        self.base_url = kwargs.get('base_url')
        self.organization = kwargs.get('organization')
        self.project = kwargs.get('project')
        self.mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    
    def _generate_embedding_raw(self, texts: List[str]) -> List[List[float]]:
        # Return mock embeddings based on number of input texts
        return self.mock_embeddings[:len(texts)]


class MockGoogleGenAIEmbeddingClient(BaseEmbeddingClient):
    """Mock Google GenAI embedding client for testing."""
    
    def __init__(self, model=None, embedding_dim=None, use_l2_norm=True, task_type=None, **kwargs):
        super().__init__(model=model, embedding_dim=embedding_dim, use_l2_norm=use_l2_norm)
        # Store other parameters for testing but don't pass to parent
        self.task_type = task_type
        self.api_key = kwargs.get('api_key')
        self.title = kwargs.get('title')
        self.mock_embeddings = [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
    
    def _generate_embedding_raw(self, texts: List[str]) -> List[List[float]]:
        return self.mock_embeddings[:len(texts)]


class MockLocalEmbeddingClient(BaseEmbeddingClient):
    """Mock local embedding client for testing."""
    
    def __init__(self, model=None, embedding_dim=None, use_l2_norm=True, device="cpu", **kwargs):
        super().__init__(model=model, embedding_dim=embedding_dim, use_l2_norm=use_l2_norm)
        # Store other parameters for testing but don't pass to parent
        self.device = device
        self.cache_dir = kwargs.get('cache_dir')
        self.trust_remote_code = kwargs.get('trust_remote_code', False)
        self.use_sentence_transformers = kwargs.get('use_sentence_transformers', True)
        self.mock_embeddings = [[1.3, 1.4, 1.5], [1.6, 1.7, 1.8]]
    
    def _generate_embedding_raw(self, texts: List[str]) -> List[List[float]]:
        return self.mock_embeddings[:len(texts)]


class TestEmbeddingFactory:
    """Test embedding factory functionality."""
    
    def test_factory_create_with_provider(self):
        """Test factory create method with explicit provider."""
        with patch('core_lib.embeddings.factory.OpenAIEmbeddingClient', MockOpenAIEmbeddingClient):
            client = EmbeddingFactory.create(provider="openai", model="test-model")
            assert isinstance(client, MockOpenAIEmbeddingClient)
            assert client.model == "test-model"
    
    def test_factory_create_unknown_provider(self):
        """Test factory with unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            EmbeddingFactory.create(provider="unknown")
    
    @patch('core_lib.embeddings.factory.OpenAIEmbeddingClient', MockOpenAIEmbeddingClient)
    def test_factory_openai(self):
        """Test OpenAI factory method."""
        client = EmbeddingFactory.openai(
            model="text-embedding-3-small",
            api_key="test-key"
        )
        assert isinstance(client, MockOpenAIEmbeddingClient)
        assert client.model == "text-embedding-3-small"
    
    @patch('core_lib.embeddings.factory.GoogleGenAIEmbeddingClient', MockGoogleGenAIEmbeddingClient)
    def test_factory_google_genai(self):
        """Test Google GenAI factory method."""
        client = EmbeddingFactory.google_genai(
            model="text-embedding-004",
            task_type="SEMANTIC_SIMILARITY"
        )
        assert isinstance(client, MockGoogleGenAIEmbeddingClient)
        assert client.model == "text-embedding-004"
        assert client.task_type == "SEMANTIC_SIMILARITY"
    
    @patch('core_lib.embeddings.factory.LocalEmbeddingClient', MockLocalEmbeddingClient)
    def test_factory_local(self):
        """Test local factory method."""
        client = EmbeddingFactory.local(
            model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        assert isinstance(client, MockLocalEmbeddingClient)
        assert client.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert client.device == "cpu"
    
    def test_factory_from_config(self):
        """Test factory from configuration."""
        config = EmbeddingsConfig(
            provider="openai",
            model="test-model",
            api_key="test-key"
        )
        
        with patch('core_lib.embeddings.factory.OpenAIEmbeddingClient', MockOpenAIEmbeddingClient):
            client = EmbeddingFactory.from_config(config)
            assert isinstance(client, MockOpenAIEmbeddingClient)
            assert client.model == "test-model"


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch('core_lib.embeddings.factory.OpenAIEmbeddingClient', MockOpenAIEmbeddingClient)
    def test_create_embedding_client(self):
        """Test create_embedding_client function."""
        client = create_embedding_client(provider="openai", model="test-model")
        assert isinstance(client, MockOpenAIEmbeddingClient)
        assert client.model == "test-model"
    
    @patch('core_lib.embeddings.factory.OpenAIEmbeddingClient', MockOpenAIEmbeddingClient)
    def test_create_openai_client(self):
        """Test create_openai_client function."""
        client = create_openai_client(model="text-embedding-3-large")
        assert isinstance(client, MockOpenAIEmbeddingClient)
        assert client.model == "text-embedding-3-large"
    
    @patch('core_lib.embeddings.factory.GoogleGenAIEmbeddingClient', MockGoogleGenAIEmbeddingClient)
    def test_create_google_genai_client(self):
        """Test create_google_genai_client function."""
        client = create_google_genai_client(
            model="text-embedding-004",
            task_type="CLASSIFICATION"
        )
        assert isinstance(client, MockGoogleGenAIEmbeddingClient)
        assert client.model == "text-embedding-004"
        assert client.task_type == "CLASSIFICATION"
    
    @patch('core_lib.embeddings.factory.LocalEmbeddingClient', MockLocalEmbeddingClient)
    def test_create_local_client(self):
        """Test create_local_client function."""
        client = create_local_client(
            model="test-model",
            device="cuda"
        )
        assert isinstance(client, MockLocalEmbeddingClient)
        assert client.model == "test-model"
        assert client.device == "cuda"


class TestBaseEmbeddingClient:
    """Test base embedding client functionality."""
    
    def test_base_client_single_embedding(self):
        """Test single embedding generation."""
        client = MockOpenAIEmbeddingClient(use_l2_norm=False)  # Disable normalization for exact comparison
        embedding = client.generate_embedding_single("test text")
        assert isinstance(embedding, list)
        assert len(embedding) == 3
        assert embedding == [0.1, 0.2, 0.3]
    
    def test_base_client_batch_embeddings(self):
        """Test batch embedding generation."""
        client = MockOpenAIEmbeddingClient(use_l2_norm=False)  # Disable normalization for exact comparison
        embeddings = client.generate_embedding_batch(["text1", "text2"])
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
    
    def test_base_client_generate_embedding_single_input(self):
        """Test generate_embedding with single string."""
        client = MockOpenAIEmbeddingClient()
        result = client.generate_embedding("test text")
        assert isinstance(result, list)
        assert len(result) == 3
    
    def test_base_client_generate_embedding_batch_input(self):
        """Test generate_embedding with list of strings."""
        client = MockOpenAIEmbeddingClient()
        result = client.generate_embedding(["text1", "text2"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]) == 3
    
    def test_base_client_invalid_input(self):
        """Test generate_embedding with invalid input."""
        client = MockOpenAIEmbeddingClient()
        with pytest.raises(ValueError, match="Input must be a string or list of strings"):
            client.generate_embedding(123)
    
    def test_normalize_vector(self):
        """Test vector normalization."""
        client = MockOpenAIEmbeddingClient(embedding_dim=5)
        
        # Test exact size
        vec = [1, 2, 3, 4, 5]
        normalized = client.normalize(vec)
        assert normalized == vec
        
        # Test truncation
        vec = [1, 2, 3, 4, 5, 6, 7]
        normalized = client.normalize(vec)
        assert normalized == [1, 2, 3, 4, 5]
        
        # Test padding
        vec = [1, 2, 3]
        normalized = client.normalize(vec)
        assert normalized == [1, 2, 3, 0.0, 0.0]
    
    def test_l2_normalization(self):
        """Test L2 normalization."""
        client = MockOpenAIEmbeddingClient(use_l2_norm=True)
        embeddings = [[3.0, 4.0, 0.0]]  # Magnitude = 5
        normalized = client._l2_normalize(embeddings)
        
        # Check that the normalized vector has unit length
        import math
        magnitude = math.sqrt(sum(x**2 for x in normalized[0]))
        assert abs(magnitude - 1.0) < 1e-6
    
    def test_health_check_default(self):
        """Test default health check."""
        client = MockOpenAIEmbeddingClient()
        assert client.health_check() is True
    
    def test_get_embedding_time(self):
        """Test embedding time tracking."""
        client = MockOpenAIEmbeddingClient()
        client.embedding_time_ms = 150.5
        assert client.get_embedding_time_ms() == 150.5
    
    @patch('core_lib.embeddings.base.cache_get_many')
    @patch('core_lib.embeddings.base.cache_set_many')
    @patch('core_lib.embeddings.base.cache_get')
    @patch('core_lib.embeddings.base.cache_set')
    def test_cache_disabled_with_zero_duration(
        self, mock_cache_set, mock_cache_get, mock_cache_set_many, mock_cache_get_many
    ):
        """Test that cache is completely bypassed when cache_duration_seconds=0."""
        # Create client with cache disabled
        client = MockOpenAIEmbeddingClient(
            use_l2_norm=False, 
            cache_duration_seconds=0
        )
        # The lightweight test double does not forward arbitrary constructor
        # kwargs to BaseEmbeddingClient, so set the cache policy explicitly.
        client.cache_duration_seconds = 0
        
        # Test single embedding - cache should not be checked or set
        embedding = client.generate_embedding_single("test text")
        assert isinstance(embedding, list)
        assert len(embedding) == 3
        mock_cache_get.assert_not_called()
        mock_cache_set.assert_not_called()
        mock_cache_get_many.assert_not_called()
        mock_cache_set_many.assert_not_called()
        
        # Reset mocks for batch test
        mock_cache_get.reset_mock()
        mock_cache_set.reset_mock()
        
        # Test batch embeddings - cache should not be checked or set
        embeddings = client.generate_embedding_batch(["text1", "text2"])
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        mock_cache_get.assert_not_called()
        mock_cache_set.assert_not_called()
    
    @patch('core_lib.embeddings.base.cache_get_many')
    @patch('core_lib.embeddings.base.cache_set_many')
    @patch('core_lib.embeddings.base.cache_get')
    @patch('core_lib.embeddings.base.cache_set')
    def test_cache_enabled_with_positive_duration(
        self, mock_cache_set, mock_cache_get, mock_cache_set_many, mock_cache_get_many
    ):
        """Test that cache is used when cache_duration_seconds > 0."""
        # Mock cache_get to return None (cache miss)
        mock_cache_get.return_value = None
        mock_cache_get_many.return_value = [None, None]
        
        # Create client with cache enabled
        client = MockOpenAIEmbeddingClient(
            use_l2_norm=False, 
            cache_duration_seconds=3600
        )
        
        # Test single embedding - cache should be checked and set
        embedding = client.generate_embedding_single("test text")
        assert isinstance(embedding, list)
        assert len(embedding) == 3
        assert mock_cache_get.call_count == 1
        assert mock_cache_set.call_count == 1
        
        # Reset mocks for batch test
        mock_cache_get.reset_mock()
        mock_cache_set.reset_mock()
        mock_cache_get.return_value = None
        
        # Test batch embeddings - cache should be checked and set
        embeddings = client.generate_embedding_batch(["text1", "text2"])
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        mock_cache_get.assert_not_called()
        mock_cache_set.assert_not_called()
        mock_cache_get_many.assert_called_once()
        mock_cache_set_many.assert_called_once()


class TestProviderAvailability:
    """Test provider availability and import handling."""
    
    def test_openai_not_available(self):
        """Test OpenAI provider when library not available."""
        with patch('core_lib.embeddings.factory._openai_available', False):
            with pytest.raises(ImportError, match="OpenAI provider not available"):
                EmbeddingFactory.openai()
    
    def test_google_genai_not_available(self):
        """Test Google GenAI provider when library not available."""
        with patch('core_lib.embeddings.factory._google_genai_available', False):
            with pytest.raises(ImportError, match="Google GenAI provider not available"):
                EmbeddingFactory.google_genai()
    
    def test_local_not_available(self):
        """Test local provider when libraries not available."""
        with patch('core_lib.embeddings.factory._local_available', False):
            with pytest.raises(ImportError, match="Local provider not available"):
                EmbeddingFactory.local()


class TestIntegration:
    """Integration tests (mocked to avoid network calls)."""
    
    @patch('core_lib.embeddings.factory.OpenAIEmbeddingClient', MockOpenAIEmbeddingClient)
    def test_end_to_end_openai(self):
        """Test end-to-end OpenAI embedding generation."""
        client = create_openai_client(model="text-embedding-3-small")
        
        # Test single embedding
        embedding = client.generate_embedding("Hello world")
        assert isinstance(embedding, list)
        assert len(embedding) == 3
        
        # Test batch embeddings
        embeddings = client.generate_embedding(["Hello", "World"])
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
    
    @patch('core_lib.embeddings.factory.GoogleGenAIEmbeddingClient', MockGoogleGenAIEmbeddingClient)
    def test_end_to_end_google_genai(self):
        """Test end-to-end Google GenAI embedding generation."""
        client = create_google_genai_client(
            model="text-embedding-004",
            task_type="SEMANTIC_SIMILARITY"
        )
        
        embedding = client.generate_embedding("Test text")
        assert isinstance(embedding, list)
        assert len(embedding) == 3
    
    @patch.dict('os.environ', {
        'EMBEDDING_PROVIDER': 'openai',
        'EMBEDDING_MODEL': 'text-embedding-3-small',
        'OPENAI_API_KEY': 'test-key'
    })
    @patch('core_lib.embeddings.factory.OpenAIEmbeddingClient', MockOpenAIEmbeddingClient)
    def test_auto_detection_from_env(self):
        """Test automatic provider detection from environment."""
        # Force reload of embeddings_settings
        from core_lib.embeddings.embeddings_config import EmbeddingsConfig
        config = EmbeddingsConfig.from_env()
        
        with patch('core_lib.embeddings.factory.embeddings_settings', config):
            client = create_client_from_env()
            assert isinstance(client, MockOpenAIEmbeddingClient)
            assert client.model == "text-embedding-3-small"


if __name__ == "__main__":
    pytest.main([__file__])
