"""OpenAI embedding client implementation."""
import time
from typing import List, Optional, Union

try:
    import openai
except ImportError:
    openai = None

from .embeddings_config import embeddings_settings
from .base import BaseEmbeddingClient, EmbeddingGenerationError
from .models import EmbeddingResponse
from core_lib.tracing.logger import get_module_logger
from core_lib.tracing.service_usage import log_embedding_usage

logger = get_module_logger()


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """Client for generating embeddings using OpenAI API."""

    def __init__(
        self,
        model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        use_l2_norm: bool = True,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
    ):
        """Initialize OpenAI embedding client.
        
        Args:
            model: Model name (e.g., 'text-embedding-3-small', 'text-embedding-3-large')
            embedding_dim: Target embedding dimension (for models that support it)
            use_l2_norm: Whether to apply L2 normalization
            api_key: OpenAI API key
            base_url: Custom base URL for OpenAI-compatible APIs
            organization: OpenAI organization ID
            project: OpenAI project ID
        """
        if openai is None:
            raise ImportError(
                "openai is required for OpenAIEmbeddingClient. "
                "Install with: pip install openai"
            )

        super().__init__(model=model, embedding_dim=embedding_dim, use_l2_norm=use_l2_norm)
        
        # Use provided API key or fall back to config
        self.api_key = api_key or embeddings_settings.api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Set default model if not provided
        if not self.model:
            self.model = "text-embedding-3-small"
        
        # Initialize OpenAI client
        client_kwargs = {
            'api_key': self.api_key,
        }
        
        if base_url or embeddings_settings.base_url:
            client_kwargs['base_url'] = base_url or embeddings_settings.base_url
            
        if organization:
            client_kwargs['organization'] = organization
            
        if project:
            client_kwargs['project'] = project
        
        self.client = openai.OpenAI(**client_kwargs)
        
        logger.debug(f"Initialized OpenAIEmbeddingClient with model={self.model}")

    def _generate_embedding_raw(self, texts: List[str]) -> List[List[float]]:
        """Generate raw embeddings using OpenAI API."""
        start_time = time.time()
        
        try:
            # Prepare embedding request parameters
            embed_kwargs = {
                'model': self.model,
                'input': texts,
            }
            
            # Add dimensions parameter for models that support it
            if self.embedding_dim and self._supports_dimensions():
                embed_kwargs['dimensions'] = self.embedding_dim
            
            # Generate embeddings
            response = self.client.embeddings.create(**embed_kwargs)
            
            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            
            self.embedding_time_ms = (time.time() - start_time) * 1000
            logger.debug(f"Generated {len(embeddings)} embeddings in {self.embedding_time_ms:.2f}ms")
            
            # Log service usage to OpenTelemetry/OpenSearch
            try:
                usage = getattr(response, 'usage', None)
                input_tokens = getattr(usage, 'total_tokens', None) if usage else None
                
                log_embedding_usage(
                    provider="openai",
                    model=self.model,
                    input_tokens=input_tokens,
                    num_texts=len(texts),
                    embedding_dim=self.embedding_dim or len(embeddings[0]) if embeddings else None,
                    latency_ms=self.embedding_time_ms,
                    host=getattr(self.client, '_base_url', None) or "https://api.openai.com",
                )
            except Exception as e:
                logger.warning(f"Failed to log embedding usage: {e}")
            
            return embeddings
            
        except Exception as e:
            self.embedding_time_ms = (time.time() - start_time) * 1000
            
            # Log error to OpenTelemetry/OpenSearch
            try:
                log_embedding_usage(
                    provider="openai",
                    model=self.model,
                    num_texts=len(texts),
                    embedding_dim=self.embedding_dim,
                    latency_ms=self.embedding_time_ms,
                    error=str(e),
                    host=getattr(self.client, '_base_url', None) or "https://api.openai.com",
                )
            except Exception:
                pass
            
            logger.error(f"Error generating embeddings with OpenAI: {e}")
            raise EmbeddingGenerationError(f"OpenAI embedding generation failed: {e}")

    def _supports_dimensions(self) -> bool:
        """Check if the current model supports the dimensions parameter."""
        # Models that support custom dimensions
        dimension_supported_models = [
            'text-embedding-3-small',
            'text-embedding-3-large',
        ]
        return self.model in dimension_supported_models

    def health_check(self) -> bool:
        """Check if the OpenAI service is accessible."""
        try:
            # Try to generate a simple embedding to test connectivity
            response = self.client.embeddings.create(
                model=self.model,
                input=["test"]
            )
            return len(response.data) > 0
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI embedding models."""
        return [
            'text-embedding-3-small',
            'text-embedding-3-large',
            'text-embedding-ada-002',  # Legacy model
        ]

    def get_model_info(self) -> dict:
        """Get information about the current model."""
        model_info = {
            'text-embedding-3-small': {
                'max_input_tokens': 8191,
                'dimensions': 1536,
                'supports_custom_dimensions': True,
                'price_per_1k_tokens': 0.00002,
            },
            'text-embedding-3-large': {
                'max_input_tokens': 8191,
                'dimensions': 3072,
                'supports_custom_dimensions': True,
                'price_per_1k_tokens': 0.00013,
            },
            'text-embedding-ada-002': {
                'max_input_tokens': 8191,
                'dimensions': 1536,
                'supports_custom_dimensions': False,
                'price_per_1k_tokens': 0.0001,
            }
        }
        
        return model_info.get(self.model, {
            'max_input_tokens': 8191,
            'dimensions': 1536,
            'supports_custom_dimensions': False,
        })