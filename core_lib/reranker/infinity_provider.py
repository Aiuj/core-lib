"""Infinity reranker client implementation.

Infinity is a high-throughput, low-latency REST API for serving reranking models.
It supports cross-encoder models like BGE-reranker and provides an efficient
local reranking server.

Documentation: https://github.com/michaelfeil/infinity
"""
import time
from typing import List, Optional

try:
    import requests
except ImportError:
    requests = None

from .reranker_config import reranker_settings
from .base import BaseRerankerClient, RerankerError, RerankResult
from core_lib.tracing.logger import get_module_logger

logger = get_module_logger()


class InfinityRerankerClient(BaseRerankerClient):
    """Client for reranking using Infinity server.
    
    Infinity provides a reranking API running locally with cross-encoder models.
    It supports various reranking models with high throughput and low latency.
    
    Default model: BAAI/bge-reranker-v2-m3 (multilingual, high quality)
    
    Other recommended models:
    - BAAI/bge-reranker-large: English, high quality
    - BAAI/bge-reranker-base: English, balanced
    - cross-encoder/ms-marco-MiniLM-L-6-v2: English, fast
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        token: Optional[str] = None,
        cache_duration_seconds: Optional[int] = None,
        return_documents: bool = True,
        **kwargs
    ):
        """Initialize Infinity reranker client.
        
        Args:
            model: Model name (e.g., 'BAAI/bge-reranker-v2-m3')
            base_url: Base URL of the Infinity server (default: http://localhost:7997)
            timeout: Request timeout in seconds (default: 30)
            token: Authentication token for secured Infinity servers
            cache_duration_seconds: How long to cache results
            return_documents: Whether to include document text in results
            **kwargs: Additional parameters
        """
        if requests is None:
            raise ImportError(
                "requests is required for InfinityRerankerClient. "
                "Install with: pip install requests"
            )

        super().__init__(
            model=model,
            cache_duration_seconds=cache_duration_seconds,
            return_documents=return_documents,
        )
        
        # Set base URL with sensible defaults
        # Priority: explicit param > INFINITY_RERANK_URL > INFINITY_BASE_URL > fallback to localhost
        self.base_url = (
            base_url 
            or reranker_settings.infinity_url 
            or "http://localhost:7997"
        )
        self.base_url = self.base_url.rstrip('/')
        
        # Set token for authentication
        self.token = token or reranker_settings.infinity_token
        
        # Set timeout
        self.timeout = timeout or reranker_settings.infinity_timeout or reranker_settings.timeout or 30
        
        # Set default model if not provided
        if not self.model:
            self.model = "BAAI/bge-reranker-v2-m3"
        
        logger.debug(
            f"Initialized Infinity reranker: model={self.model}, "
            f"base_url={self.base_url}, timeout={self.timeout}"
        )

    def _rerank_raw(
        self,
        query: str,
        documents: List[str],
        top_k: int,
    ) -> List[RerankResult]:
        """Perform reranking using Infinity API.
        
        Infinity uses a reranking endpoint that accepts query-document pairs.
        """
        start_time = time.time()
        
        try:
            # Prepare request body for Infinity rerank endpoint
            # Infinity follows the Cohere-compatible rerank API format
            request_body = {
                'model': self.model,
                'query': query,
                'documents': documents,
                'top_n': top_k,
                'return_documents': self.return_documents,
            }
            
            # Prepare headers
            headers = {'Content-Type': 'application/json'}
            if self.token:
                headers['Authorization'] = f'Bearer {self.token}'
            
            # Make request to Infinity server
            response = requests.post(
                f"{self.base_url}/rerank",
                json=request_body,
                headers=headers,
                timeout=self.timeout
            )
            
            # Raise exception for HTTP errors
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Extract results
            # Response format: {"results": [{"index": 0, "relevance_score": 0.95, "document": {...}}, ...]}
            results = []
            for item in data.get('results', []):
                result = RerankResult(
                    index=item['index'],
                    score=item.get('relevance_score', item.get('score', 0.0)),
                    document=item.get('document', {}).get('text') if isinstance(item.get('document'), dict) else item.get('document'),
                )
                results.append(result)
            
            self.rerank_time_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"Reranked {len(documents)} documents in {self.rerank_time_ms:.2f}ms "
                f"using Infinity ({self.model})"
            )
            
            return results
            
        except requests.exceptions.Timeout:
            self.rerank_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Infinity rerank request timed out after {self.timeout}s"
            logger.error(error_msg)
            raise RerankerError(error_msg)
            
        except requests.exceptions.ConnectionError as e:
            self.rerank_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Failed to connect to Infinity server at {self.base_url}: {e}"
            logger.error(error_msg)
            raise RerankerError(error_msg)
            
        except requests.exceptions.HTTPError as e:
            self.rerank_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Infinity server returned HTTP error: {e}"
            logger.error(error_msg)
            try:
                error_detail = response.json()
                logger.error(f"Error detail: {error_detail}")
            except:
                pass
            raise RerankerError(error_msg)
            
        except Exception as e:
            self.rerank_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Unexpected error reranking with Infinity: {e}"
            logger.error(error_msg)
            raise RerankerError(error_msg)

    def health_check(self) -> bool:
        """Check if the Infinity service is accessible and healthy."""
        try:
            # Try the health endpoint first
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            if response.status_code == 200:
                return True
            
            # Fallback: try the models endpoint
            response = requests.get(
                f"{self.base_url}/models",
                timeout=5
            )
            return response.status_code == 200
            
        except Exception as e:
            logger.warning(f"Infinity health check failed: {e}")
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available reranking models from Infinity server."""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                timeout=5
            )
            response.raise_for_status()
            
            data = response.json()
            # Filter for reranking models (those with rerank capability)
            if 'data' in data and isinstance(data['data'], list):
                rerank_models = []
                for model in data['data']:
                    # Check if model has reranking capability
                    capabilities = model.get('capabilities', [])
                    if 'rerank' in capabilities or 'reranking' in capabilities:
                        rerank_models.append(model['id'])
                    # Also include if model name suggests it's a reranker
                    elif 'reranker' in model.get('id', '').lower():
                        rerank_models.append(model['id'])
                return rerank_models
            
            return []
            
        except Exception as e:
            logger.warning(f"Failed to get available models from Infinity: {e}")
            return []
