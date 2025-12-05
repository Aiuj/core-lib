"""Infinity embedding client implementation.

Infinity is a high-throughput, low-latency REST API for serving text embeddings
using the OpenAI-compatible API format. It supports multiple models and provides
an efficient local embedding server.

Documentation: https://github.com/michaelfeil/infinity
"""
import time
from typing import List, Optional

try:
    import requests
except ImportError:
    requests = None

from .embeddings_config import embeddings_settings
from .base import BaseEmbeddingClient, EmbeddingGenerationError
from core_lib.tracing.logger import get_module_logger
from core_lib.tracing.service_usage import log_embedding_usage

logger = get_module_logger()


class InfinityEmbeddingClient(BaseEmbeddingClient):
    """Client for generating embeddings using Infinity server.
    
    Infinity provides an OpenAI-compatible embedding API running locally.
    It supports various embedding models with high throughput and low latency.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        use_l2_norm: bool = True,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        token: Optional[str] = None,
        **kwargs
    ):
        """Initialize Infinity embedding client.
        
        Args:
            model: Model name (e.g., 'BAAI/bge-small-en-v1.5', 'sentence-transformers/all-MiniLM-L6-v2')
            embedding_dim: Target embedding dimension
            use_l2_norm: Whether to apply L2 normalization
            base_url: Base URL of the Infinity server (default: http://localhost:7997)
            timeout: Request timeout in seconds (default: 30)
            token: Authentication token for secured Infinity servers
            **kwargs: Additional parameters
        """
        if requests is None:
            raise ImportError(
                "requests is required for InfinityEmbeddingClient. "
                "Install with: pip install requests"
            )

        super().__init__(model=model, embedding_dim=embedding_dim, use_l2_norm=use_l2_norm)
        
        # Set base URL with sensible defaults
        # Priority: explicit param > INFINITY_BASE_URL > EMBEDDING_BASE_URL > fallback to localhost
        self.base_url = (
            base_url 
            or embeddings_settings.infinity_url 
            or embeddings_settings.base_url
            or "http://localhost:7997"
        )
        self.base_url = self.base_url.rstrip('/')
        
        # Set token for authentication
        # Priority: explicit param > INFINITY_TOKEN > EMBEDDING_TOKEN
        self.token = (
            token
            or embeddings_settings.infinity_token
        )
        
        # Set timeout
        # Priority: explicit param > INFINITY_TIMEOUT > EMBEDDING_TIMEOUT > OLLAMA_TIMEOUT > default 30s
        self.timeout = timeout or embeddings_settings.infinity_timeout or embeddings_settings.ollama_timeout or 30
        
        # Set default model if not provided
        if not self.model:
            self.model = "BAAI/bge-small-en-v1.5"
        
        # Log which URL source was used for debugging
        url_source = "parameter" if base_url else (
            "INFINITY_BASE_URL" if embeddings_settings.infinity_url else (
                "EMBEDDING_BASE_URL" if embeddings_settings.base_url else "default"
            )
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
            
            # Prepare headers
            headers = {'Content-Type': 'application/json'}
            if self.token:
                headers['Authorization'] = f'Bearer {self.token}'
            
            # Make request to Infinity server
            response = requests.post(
                f"{self.base_url}/embeddings",
                json=request_body,
                headers=headers,
                timeout=self.timeout
            )
            
            # Raise exception for HTTP errors
            response.raise_for_status()
            
            # Parse response (OpenAI-compatible format)
            data = response.json()
            
            # Extract embeddings from response
            # Response format: {"object": "list", "data": [{"object": "embedding", "embedding": [...], "index": 0}, ...]}
            embeddings = [item['embedding'] for item in sorted(data['data'], key=lambda x: x['index'])]
            
            self.embedding_time_ms = (time.time() - start_time) * 1000
            logger.debug(f"Generated {len(embeddings)} embeddings in {self.embedding_time_ms:.2f}ms using Infinity")
            
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
                    host=self.base_url,
                )
            except Exception as e:
                logger.warning(f"Failed to log embedding usage: {e}")
            
            return embeddings
            
        except requests.exceptions.Timeout:
            self.embedding_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Infinity request timed out after {self.timeout}s"
            logger.error(error_msg)
            
            # Log error to OpenTelemetry/OpenSearch
            try:
                log_embedding_usage(
                    provider="infinity",
                    model=self.model,
                    num_texts=len(texts),
                    embedding_dim=self.embedding_dim,
                    latency_ms=self.embedding_time_ms,
                    error=error_msg,
                    host=self.base_url,
                )
            except Exception:
                pass
            
            raise EmbeddingGenerationError(error_msg)
            
        except requests.exceptions.ConnectionError as e:
            self.embedding_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Failed to connect to Infinity server at {self.base_url}: {e}"
            logger.error(error_msg)
            raise EmbeddingGenerationError(error_msg)
            
        except requests.exceptions.HTTPError as e:
            self.embedding_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Infinity server returned HTTP error: {e}"
            logger.error(error_msg)
            try:
                error_detail = response.json()
                logger.error(f"Error detail: {error_detail}")
            except:
                pass
            raise EmbeddingGenerationError(error_msg)
            
        except Exception as e:
            self.embedding_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Unexpected error generating embeddings with Infinity: {e}"
            logger.error(error_msg)
            raise EmbeddingGenerationError(error_msg)

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
            
            # Fallback: try to generate a simple embedding
            response = requests.post(
                f"{self.base_url}/embeddings",
                json={
                    'model': self.model,
                    'input': ["test"],
                    'encoding_format': 'float'
                },
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            return response.status_code == 200
            
        except Exception as e:
            logger.warning(f"Infinity health check failed: {e}")
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available models from Infinity server."""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                timeout=5
            )
            response.raise_for_status()
            
            data = response.json()
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
            response = requests.get(
                f"{self.base_url}/models",
                timeout=5
            )
            response.raise_for_status()
            
            data = response.json()
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
