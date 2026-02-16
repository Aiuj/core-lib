"""Infinity embedding client implementation.

Infinity is a high-throughput, low-latency REST API for serving text embeddings
using the OpenAI-compatible API format. It supports multiple models and provides
an efficient local embedding server.

Supports multi-server failover via comma-separated URLs for high availability.

Documentation: https://github.com/michaelfeil/infinity
"""
import time
from typing import List, Optional

from core_lib.api_utils import InfinityAPIClient, InfinityAPIError
from .embeddings_config import embeddings_settings
from .base import BaseEmbeddingClient, EmbeddingGenerationError
from core_lib.tracing.logger import get_module_logger
from core_lib.tracing.service_usage import log_embedding_usage

logger = get_module_logger()


class InfinityEmbeddingClient(BaseEmbeddingClient):
    """Client for generating embeddings using Infinity server.
    
    Infinity provides an OpenAI-compatible embedding API running locally.
    It supports various embedding models with high throughput and low latency.
    
    Supports multi-server failover by providing comma-separated URLs:
        INFINITY_BASE_URL=http://server1:7997,http://server2:7997,http://server3:7997
    """

    def __init__(
        self,
        model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        use_l2_norm: bool = True,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        token: Optional[str] = None,
        wake_on_lan: Optional[dict] = None,
        **kwargs
    ):
        """Initialize Infinity embedding client.
        
        Args:
            model: Model name (e.g., 'BAAI/bge-small-en-v1.5', 'sentence-transformers/all-MiniLM-L6-v2')
            embedding_dim: Target embedding dimension
            use_l2_norm: Whether to apply L2 normalization
            base_url: Base URL(s) - single or comma-separated (default: http://localhost:7997)
            timeout: Request timeout in seconds (default: 30)
            token: Authentication token for secured Infinity servers
            wake_on_lan: Optional Wake-on-LAN config for sleeping hosts
            **kwargs: Additional parameters
        """
        super().__init__(model=model, embedding_dim=embedding_dim, use_l2_norm=use_l2_norm)
        
        # Set base URL with sensible defaults
        # Priority: explicit param > INFINITY_BASE_URL > EMBEDDING_BASE_URL > fallback to localhost
        base_url = (
            base_url 
            or embeddings_settings.infinity_url 
            or embeddings_settings.base_url
            or "http://localhost:7997"
        )
        
        # Set timeout
        # Priority: explicit param > INFINITY_TIMEOUT > EMBEDDING_TIMEOUT > OLLAMA_TIMEOUT > default 30s
        timeout = timeout or embeddings_settings.infinity_timeout or embeddings_settings.ollama_timeout or 30
        
        # Set token for authentication
        # Priority: explicit param > INFINITY_TOKEN > EMBEDDING_TOKEN
        token = (
            token
            or embeddings_settings.infinity_token
        )
        
        # Create shared API client with multi-URL failover support
        self._api_client = InfinityAPIClient(
            base_urls=base_url,
            timeout=timeout,
            token=token,
            wake_on_lan=wake_on_lan,
        )
        
        # Set default model if not provided
        if not self.model:
            self.model = "BAAI/bge-small-en-v1.5"
        
        logger.debug(
            f"Initialized Infinity embedding client: model={self.model}, "
            f"servers={len(self._api_client.base_urls)}"
        )

    def _generate_embedding_raw(self, texts: List[str]) -> List[List[float]]:
        """Generate raw embeddings using Infinity API.
        
        Infinity uses the OpenAI-compatible embeddings endpoint format.
        """
        start_time = time.time()
        
        try:
            # Prepare request body (OpenAI-compatible format)
            request_body = {
                'model': self.model,
                'input': texts,
                'encoding_format': 'float',  # Infinity supports 'float' and 'base64'
            }
            
            # Add dimensions parameter if specified and supported
            if self.embedding_dim:
                request_body['dimensions'] = self.embedding_dim
            
            # Make request via shared API client with automatic failover
            data, used_url = self._api_client.post('/embeddings', json=request_body)
            
            # Extract embeddings from response
            # Response format: {"object": "list", "data": [{"object": "embedding", "embedding": [...], "index": 0}, ...]}
            embeddings = [item['embedding'] for item in sorted(data['data'], key=lambda x: x['index'])]
            
            self.embedding_time_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"Generated {len(embeddings)} embeddings in {self.embedding_time_ms:.2f}ms "
                f"using Infinity @ {used_url}"
            )
            
            # Log service usage to OpenTelemetry/OpenSearch
            try:
                # Infinity doesn't return token counts directly, estimate from text length
                # Rough estimate: ~4 chars per token for English text
                estimated_tokens = sum(len(text) // 4 for text in texts)
                
                log_embedding_usage(
                    provider="infinity",
                    model=self.model,
                    input_tokens=estimated_tokens,
                    num_texts=len(texts),
                    embedding_dim=self.embedding_dim or len(embeddings[0]) if embeddings else None,
                    latency_ms=self.embedding_time_ms,
                    host=used_url,
                )
            except Exception as e:
                logger.warning(f"Failed to log embedding usage: {e}")
            
            return embeddings
            
        except InfinityAPIError as e:
            self.embedding_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Infinity embedding failed: {e}"
            logger.error(error_msg)
            
            # Log error to OpenTelemetry/OpenSearch
            try:
                log_embedding_usage(
                    provider="infinity",
                    model=self.model,
                    num_texts=len(texts),
                    embedding_dim=self.embedding_dim,
                    latency_ms=self.embedding_time_ms,
                    error=str(e),
                )
            except Exception:
                pass
            
            raise EmbeddingGenerationError(error_msg)
            
        except Exception as e:
            self.embedding_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Unexpected error generating embeddings with Infinity: {e}"
            logger.error(error_msg)
            raise EmbeddingGenerationError(error_msg)

    def health_check(self) -> bool:
        """Check if the Infinity service is accessible and healthy."""
        return self._api_client.health_check()

    def get_available_models(self) -> List[str]:
        """Get list of available models from Infinity server."""
        try:
            data, _ = self._api_client.get('/models')
            
            # Response format: {"object": "list", "data": [{"id": "model_name", ...}, ...]}
            if 'data' in data and isinstance(data['data'], list):
                return [model['id'] for model in data['data'] if 'id' in model]
            
            return []
            
        except Exception as e:
            logger.warning(f"Failed to get available models from Infinity: {e}")
            return []

    def get_model_info(self) -> dict:
        """Get information about the current model from Infinity server."""
        try:
            data, _ = self._api_client.get('/models')
            
            if 'data' in data and isinstance(data['data'], list):
                for model_info in data['data']:
                    if model_info.get('id') == self.model:
                        return {
                            'id': model_info.get('id'),
                            'backend': model_info.get('backend', 'unknown'),
                            'capabilities': model_info.get('capabilities', []),
                            'created': model_info.get('created'),
                            'stats': model_info.get('stats', {}),
                        }
            
            # Return default info if not found
            return {
                'id': self.model,
                'backend': 'unknown',
                'capabilities': [],
            }
            
        except Exception as e:
            logger.warning(f"Failed to get model info from Infinity: {e}")
            return {
                'id': self.model,
                'backend': 'unknown',
                'capabilities': [],
            }
