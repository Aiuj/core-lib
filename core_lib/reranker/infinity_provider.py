"""Infinity reranker client implementation.

Infinity is a high-throughput, low-latency REST API for serving reranking models.
It supports cross-encoder models like BGE-reranker and provides an efficient
local reranking server.

Supports multi-server failover via comma-separated URLs for high availability.

Documentation: https://github.com/michaelfeil/infinity
"""
import time
from typing import List, Optional, Tuple, Dict

from core_lib.api_utils import InfinityAPIClient, InfinityAPIError
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
    
    Supports multi-server failover by providing comma-separated URLs:
        INFINITY_BASE_URL=http://server1:7997,http://server2:7997,http://server3:7997
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        token: Optional[str] = None,
        wake_on_lan: Optional[dict] = None,
        cache_duration_seconds: Optional[int] = None,
        return_documents: bool = True,
        **kwargs
    ):
        """Initialize Infinity reranker client.
        
        Args:
            model: Model name (e.g., 'BAAI/bge-reranker-v2-m3')
            base_url: Base URL(s) - single or comma-separated (default: http://localhost:7997)
            timeout: Request timeout in seconds (default: 30)
            token: Authentication token for secured Infinity servers
            wake_on_lan: Optional Wake-on-LAN config for sleeping hosts
            cache_duration_seconds: How long to cache results
            return_documents: Whether to include document text in results
            **kwargs: Additional parameters
        """
        super().__init__(
            model=model,
            cache_duration_seconds=cache_duration_seconds,
            return_documents=return_documents,
        )
        
        # Set base URL with sensible defaults
        # Priority: explicit param > INFINITY_RERANK_URL > INFINITY_BASE_URL > fallback to localhost
        base_url = (
            base_url 
            or reranker_settings.infinity_url 
            or "http://localhost:7997"
        )
        
        # Set timeout
        timeout = timeout or reranker_settings.infinity_timeout or reranker_settings.timeout or 30
        
        # Set token for authentication
        token = token or reranker_settings.infinity_token
        
        # Create shared API client with multi-URL failover support
        self._api_client = InfinityAPIClient(
            base_urls=base_url,
            timeout=timeout,
            token=token,
            wake_on_lan=wake_on_lan,
        )
        
        # Set default model if not provided
        if not self.model:
            self.model = "BAAI/bge-reranker-v2-m3"
        
        logger.debug(
            f"Initialized Infinity reranker: model={self.model}, "
            f"servers={len(self._api_client.base_urls)}"
        )

    def _rerank_raw(
        self,
        query: str,
        documents: List[str],
        top_k: int,
    ) -> Tuple[List[RerankResult], Optional[Dict[str, int]]]:
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
            
            # Make request via shared API client with automatic failover
            data, used_url = self._api_client.post('/rerank', json=request_body)
            
            # Extract usage
            usage = None
            if 'usage' in data:
                usage = {
                    'input_tokens': data['usage'].get('prompt_tokens', 0),
                    'output_tokens': data['usage'].get('completion_tokens', 0) or data['usage'].get('total_tokens', 0) - data['usage'].get('prompt_tokens', 0)
                }
            elif 'meta' in data and 'billed_units' in data['meta']:
                billed = data['meta']['billed_units']
                usage = {
                    'input_tokens': billed.get('input_tokens', 0),
                    'output_tokens': billed.get('output_tokens', 0)
                }

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
                f"using Infinity ({self.model}) @ {used_url}"
            )
            
            return results, usage
            
        except InfinityAPIError as e:
            self.rerank_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Infinity reranking failed: {e}"
            logger.error(error_msg)
            raise RerankerError(error_msg)
            
        except Exception as e:
            self.rerank_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Unexpected error reranking with Infinity: {e}"
            logger.error(error_msg)
            raise RerankerError(error_msg)

    def health_check(self) -> bool:
        """Check if the Infinity service is accessible and healthy."""
        return self._api_client.health_check()

    def get_available_models(self) -> List[str]:
        """Get list of available reranking models from Infinity server."""
        try:
            data, _ = self._api_client.get('/models')
            
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
