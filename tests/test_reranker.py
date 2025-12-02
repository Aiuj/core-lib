"""Tests for the reranker module."""
import pytest
from unittest.mock import Mock, patch, MagicMock

from core_lib.reranker import (
    BaseRerankerClient,
    RerankerError,
    RerankResult,
    RerankerSettings,
    reranker_settings,
    RerankerFactory,
    create_reranker_client,
)


class TestRerankResult:
    """Tests for RerankResult dataclass."""
    
    def test_rerank_result_creation(self):
        """Test creating a RerankResult."""
        result = RerankResult(index=0, score=0.95, document="Test document")
        assert result.index == 0
        assert result.score == 0.95
        assert result.document == "Test document"
    
    def test_rerank_result_without_document(self):
        """Test RerankResult without document."""
        result = RerankResult(index=1, score=0.8)
        assert result.index == 1
        assert result.score == 0.8
        assert result.document is None
    
    def test_rerank_result_repr(self):
        """Test RerankResult string representation."""
        result = RerankResult(index=0, score=0.95, document="Test document")
        repr_str = repr(result)
        assert "index=0" in repr_str
        assert "score=0.95" in repr_str


class TestRerankerSettings:
    """Tests for RerankerSettings configuration."""
    
    def test_default_settings(self):
        """Test default reranker settings."""
        settings = RerankerSettings()
        assert settings.provider == "infinity"
        assert settings.model == "BAAI/bge-reranker-v2-m3"
        assert settings.timeout == 30
        assert settings.cache_duration_seconds == 3600
        assert settings.default_top_k == 10
    
    def test_settings_from_env(self):
        """Test creating settings from environment."""
        with patch.dict('os.environ', {
            'RERANKER_PROVIDER': 'cohere',
            'RERANKER_MODEL': 'rerank-english-v3.0',
            'RERANKER_TIMEOUT': '60',
        }):
            settings = RerankerSettings.from_env(load_dotenv=False)
            assert settings.provider == "cohere"
            assert settings.model == "rerank-english-v3.0"
            assert settings.timeout == 60
    
    def test_settings_as_dict(self):
        """Test converting settings to dictionary."""
        settings = RerankerSettings()
        settings_dict = settings.as_dict()
        assert "provider" in settings_dict
        assert "model" in settings_dict
        assert settings_dict["provider"] == "infinity"


class TestBaseRerankerClient:
    """Tests for BaseRerankerClient."""
    
    def test_abstract_rerank_raw(self):
        """Test that _rerank_raw raises NotImplementedError."""
        client = BaseRerankerClient(model="test-model")
        with pytest.raises(NotImplementedError):
            client._rerank_raw("query", ["doc1", "doc2"], 2)
    
    def test_rerank_empty_documents(self):
        """Test reranking with empty document list."""
        client = BaseRerankerClient(model="test-model")
        results = client.rerank("query", [])
        assert results == []
    
    def test_health_check_default(self):
        """Test default health check returns True."""
        client = BaseRerankerClient(model="test-model")
        assert client.health_check() is True


class TestRerankerFactory:
    """Tests for RerankerFactory."""
    
    def test_create_with_invalid_provider(self):
        """Test factory raises error for invalid provider."""
        with pytest.raises(ValueError, match="Unknown reranker provider"):
            RerankerFactory.create(provider="invalid_provider")
    
    @patch('core_lib.reranker.factory._infinity_available', False)
    def test_infinity_not_available(self):
        """Test error when infinity provider not available."""
        with pytest.raises(ImportError, match="Infinity provider not available"):
            RerankerFactory.infinity()
    
    @patch('core_lib.reranker.factory._cohere_available', False)
    def test_cohere_not_available(self):
        """Test error when cohere provider not available."""
        with pytest.raises(ImportError, match="Cohere provider not available"):
            RerankerFactory.cohere()
    
    @patch('core_lib.reranker.factory._local_available', False)
    def test_local_not_available(self):
        """Test error when local provider not available."""
        with pytest.raises(ImportError, match="Local provider not available"):
            RerankerFactory.local()


class TestInfinityRerankerClient:
    """Tests for InfinityRerankerClient."""
    
    @pytest.fixture
    def mock_requests(self):
        """Mock requests module."""
        with patch('core_lib.reranker.infinity_provider.requests') as mock:
            yield mock
    
    def test_infinity_client_initialization(self, mock_requests):
        """Test Infinity client initialization."""
        from core_lib.reranker.infinity_provider import InfinityRerankerClient
        
        client = InfinityRerankerClient(
            model="BAAI/bge-reranker-v2-m3",
            base_url="http://localhost:7997",
        )
        assert client.model == "BAAI/bge-reranker-v2-m3"
        assert client.base_url == "http://localhost:7997"
    
    def test_infinity_rerank_success(self, mock_requests):
        """Test successful reranking with Infinity."""
        from core_lib.reranker.infinity_provider import InfinityRerankerClient
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.95, "document": {"text": "Doc 2"}},
                {"index": 0, "relevance_score": 0.85, "document": {"text": "Doc 1"}},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_requests.post.return_value = mock_response
        
        client = InfinityRerankerClient(model="test-model")
        results = client.rerank("What is AI?", ["Doc 1", "Doc 2"], top_k=2)
        
        assert len(results) == 2
        assert results[0].score == 0.95
        assert results[0].index == 1
    
    def test_infinity_timeout_error(self, mock_requests):
        """Test timeout error handling."""
        from core_lib.reranker.infinity_provider import InfinityRerankerClient
        import requests as real_requests
        
        mock_requests.post.side_effect = real_requests.exceptions.Timeout()
        mock_requests.exceptions = real_requests.exceptions
        
        client = InfinityRerankerClient(model="test-model", timeout=5)
        
        with pytest.raises(RerankerError, match="timed out"):
            client.rerank("query", ["doc1"])
    
    def test_infinity_health_check(self, mock_requests):
        """Test Infinity health check."""
        from core_lib.reranker.infinity_provider import InfinityRerankerClient
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests.get.return_value = mock_response
        
        client = InfinityRerankerClient(model="test-model")
        assert client.health_check() is True


class TestCacheIntegration:
    """Tests for reranker caching."""
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        client = BaseRerankerClient(model="test-model")
        
        key1 = client._generate_cache_key("query1", ["doc1", "doc2"], 2)
        key2 = client._generate_cache_key("query1", ["doc1", "doc2"], 2)
        key3 = client._generate_cache_key("query2", ["doc1", "doc2"], 2)
        
        # Same inputs should generate same key
        assert key1 == key2
        # Different query should generate different key
        assert key1 != key3
    
    def test_cache_key_includes_model(self):
        """Test that cache key includes model name."""
        client1 = BaseRerankerClient(model="model1")
        client2 = BaseRerankerClient(model="model2")
        
        key1 = client1._generate_cache_key("query", ["doc"], 1)
        key2 = client2._generate_cache_key("query", ["doc"], 1)
        
        # Different models should generate different keys
        assert key1 != key2
