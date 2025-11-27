"""Google GenAI embedding client implementation."""
import time
from typing import List, Optional, Union, Any

try:
    import google.genai as genai
    from google.genai.types import TaskType
except ImportError:
    genai = None
    TaskType = None

from .embeddings_config import embeddings_settings
from .base import BaseEmbeddingClient, EmbeddingGenerationError
from .models import EmbeddingResponse
from core_lib.tracing.logger import get_module_logger
from core_lib.tracing.service_usage import log_embedding_usage

logger = get_module_logger()


class GoogleGenAIEmbeddingClient(BaseEmbeddingClient):
    """Client for generating embeddings using Google GenAI (latest library)."""

    def __init__(
        self,
        model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        use_l2_norm: bool = True,
        api_key: Optional[str] = None,
        task_type: Optional[str] = None,
        title: Optional[str] = None,
    ):
        """Initialize Google GenAI embedding client.
        
        Args:
            model: Model name (e.g., 'text-embedding-004')
            embedding_dim: Target embedding dimension
            use_l2_norm: Whether to apply L2 normalization
            api_key: Google API key
            task_type: Task type for embeddings (SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING, etc.)
            title: Optional title for the embedding task
        """
        if genai is None:
            raise ImportError(
                "google-genai is required for GoogleGenAIEmbeddingClient. "
                "Install with: pip install google-genai"
            )

        super().__init__(model=model, embedding_dim=embedding_dim, use_l2_norm=use_l2_norm)
        
        # Use provided API key or fall back to config
        self.api_key = api_key or embeddings_settings.google_api_key or embeddings_settings.api_key
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_GENAI_API_KEY or GEMINI_API_KEY environment variable.")
        
        # Configure the client
        genai.configure(api_key=self.api_key)
        
        # Set default model if not provided
        if not self.model:
            self.model = "text-embedding-004"
        
        # Task type configuration
        self.task_type = self._get_task_type(task_type)
        self.title = title
        
        logger.debug(f"Initialized GoogleGenAIEmbeddingClient with model={self.model}, task_type={task_type}")

    def _get_task_type(self, task_type: Optional[str]) -> Optional[Any]:
        """Convert string task type to Google GenAI TaskType enum."""
        if not task_type or TaskType is None:
            return None
            
        # Map common task type names to Google GenAI TaskType
        task_type = task_type.upper()
        task_type_mapping = {
            'SEMANTIC_SIMILARITY': TaskType.SEMANTIC_SIMILARITY,
            'CLASSIFICATION': TaskType.CLASSIFICATION,
            'CLUSTERING': TaskType.CLUSTERING,
            'RETRIEVAL_DOCUMENT': TaskType.RETRIEVAL_DOCUMENT,
            'RETRIEVAL_QUERY': TaskType.RETRIEVAL_QUERY,
            'CODE_RETRIEVAL_QUERY': TaskType.CODE_RETRIEVAL_QUERY,
            'QUESTION_ANSWERING': TaskType.QUESTION_ANSWERING,
            'FACT_VERIFICATION': TaskType.FACT_VERIFICATION,
            # Legacy compatibility
            'DOCUMENT_RETRIEVAL': TaskType.RETRIEVAL_DOCUMENT,
        }
        
        return task_type_mapping.get(task_type)

    def _generate_embedding_raw(self, texts: List[str]) -> List[List[float]]:
        """Generate raw embeddings using Google GenAI."""
        start_time = time.time()
        
        try:
            # Prepare embedding request parameters
            embed_kwargs = {
                'model': self.model,
                'content': texts,
            }
            
            # Add task type if specified
            if self.task_type:
                embed_kwargs['task_type'] = self.task_type
                
            # Add title if specified
            if self.title:
                embed_kwargs['title'] = self.title
            
            # Generate embeddings
            result = genai.embed(**embed_kwargs)
            
            # Extract embeddings from response
            if hasattr(result, 'embeddings'):
                embeddings = [embedding.values for embedding in result.embeddings]
            elif hasattr(result, 'embedding'):
                # Single embedding case
                embeddings = [result.embedding.values] if hasattr(result.embedding, 'values') else [result.embedding]
            else:
                raise EmbeddingGenerationError("Unexpected response format from Google GenAI")
            
            self.embedding_time_ms = (time.time() - start_time) * 1000
            logger.debug(f"Generated {len(embeddings)} embeddings in {self.embedding_time_ms:.2f}ms")
            
            # Log service usage to OpenTelemetry/OpenSearch
            try:
                # Google GenAI doesn't provide token counts directly, estimate from text length
                estimated_tokens = sum(len(text) // 4 for text in texts)
                
                log_embedding_usage(
                    provider="google-genai",
                    model=self.model,
                    input_tokens=estimated_tokens,
                    num_texts=len(texts),
                    embedding_dim=self.embedding_dim or len(embeddings[0]) if embeddings else None,
                    latency_ms=self.embedding_time_ms,
                    metadata={"task_type": str(self.task_type) if self.task_type else None},
                )
            except Exception as e:
                logger.warning(f"Failed to log embedding usage: {e}")
            
            return embeddings
            
        except Exception as e:
            self.embedding_time_ms = (time.time() - start_time) * 1000
            
            # Log error to OpenTelemetry/OpenSearch
            try:
                log_embedding_usage(
                    provider="google-genai",
                    model=self.model,
                    num_texts=len(texts),
                    embedding_dim=self.embedding_dim,
                    latency_ms=self.embedding_time_ms,
                    error=str(e),
                )
            except Exception:
                pass
            
            logger.error(f"Error generating embeddings with Google GenAI: {e}")
            raise EmbeddingGenerationError(f"Google GenAI embedding generation failed: {e}")

    def health_check(self) -> bool:
        """Check if the Google GenAI service is accessible."""
        try:
            # Try to generate a simple embedding to test connectivity
            test_result = genai.embed(
                model=self.model,
                content=["test"]
            )
            return True
        except Exception as e:
            logger.warning(f"Google GenAI health check failed: {e}")
            return False

    def get_supported_task_types(self) -> List[str]:
        """Get list of supported task types."""
        return [
            'SEMANTIC_SIMILARITY',
            'CLASSIFICATION', 
            'CLUSTERING',
            'RETRIEVAL_DOCUMENT',
            'RETRIEVAL_QUERY',
            'CODE_RETRIEVAL_QUERY',
            'QUESTION_ANSWERING',
            'FACT_VERIFICATION'
        ]