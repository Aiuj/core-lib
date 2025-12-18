"""Ollama embedding client implementation."""
import time
import math
import ollama
from typing import List, Union, cast

from .embeddings_config import embeddings_settings
from .base import BaseEmbeddingClient, EmbeddingGenerationError
from .models import EmbeddingResponse
from core_lib.tracing.logger import get_module_logger
from core_lib.tracing.service_usage import log_embedding_usage

logger = get_module_logger()


class OllamaEmbeddingClient(BaseEmbeddingClient):
    """Client for generating embeddings using a local Ollama HTTP API."""

    def __init__(
        self,
        model: str | None = None,
        embedding_dim: int | None = None,
        use_l2_norm: bool = False,
        norm_method: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None
    ):
        # If model/embedding_dim are not provided, BaseEmbeddingClient will
        # fall back to values from `settings`.
        super().__init__(
            model=model,
            embedding_dim=embedding_dim,
            use_l2_norm=use_l2_norm,
            norm_method=norm_method,
        )
        # Priority: explicit param > OLLAMA_URL > EMBEDDING_BASE_URL > default
        self.base_url = base_url or embeddings_settings.ollama_url or "http://localhost:11434"
        # Priority: explicit param > OLLAMA_TIMEOUT > EMBEDDING_TIMEOUT > None
        self.timeout = timeout or embeddings_settings.ollama_timeout
        """Initialize Ollama client with configuration from settings."""
        self.client = ollama.Client(host=self.base_url)  # Use Ollama library client

    def _generate_embedding_raw(self, texts: List[str]) -> List[List[float]]:
        """Generate raw embeddings for a list of texts without L2 normalization.

        Args:
            texts: List of text strings to generate embeddings for.

        Returns:
            A list of embedding vectors (List[List[float]]), one per input text.

        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        if not isinstance(texts, list) or not texts:
            raise EmbeddingGenerationError("Input must be a non-empty list of strings")

        # Validate and strip inputs
        if any(not t or not isinstance(t, str) or not t.strip() for t in texts):
            raise EmbeddingGenerationError("All texts must be non-empty strings")

        input_data = [t.strip() for t in texts]

        try:
            # Use Ollama library's embed function
            start_time = time.perf_counter()
            response = self.client.embed(model=self.model, input=input_data)
            end_time = time.perf_counter()
            self.embedding_time_ms = (end_time - start_time) * 1000

            # Parse response using EmbeddingResponse. The ollama client may return
            # a library response object rather than a plain dict, so coerce to dict
            # Coerce the library response to a plain dict with expected keys.
            response_data: dict
            if isinstance(response, dict):
                response_data = response
            else:
                # Try common conversion paths
                if hasattr(response, "dict") and callable(getattr(response, "dict")):
                    try:
                        response_data = response.dict()
                    except Exception:
                        response_data = {}
                elif hasattr(response, "__dict__"):
                    response_data = dict(getattr(response, "__dict__", {}))
                else:
                    # Last resort: try to convert to dict
                    try:
                        response_data = dict(response)
                    except Exception:
                        response_data = {}

            # Ensure keys are in the shape EmbeddingResponse expects
            if 'embeddings' not in response_data and 'embedding' not in response_data:
                # Try to extract plausible attributes
                possible_embedding = getattr(response, 'embedding', None) or getattr(response, 'embeddings', None)
                if possible_embedding is not None:
                    response_data = {'embedding': possible_embedding, 'model': getattr(response, 'model', self.model)}

            embedding_response = EmbeddingResponse.from_dict(response_data)

            # Normalize each returned embedding using BaseEmbeddingClient.normalize
            embeddings_raw = embedding_response.embedding

            # If the API returned a single vector for the batch, wrap it into a list
            if isinstance(embeddings_raw, list) and embeddings_raw and not isinstance(embeddings_raw[0], list):
                embeddings_raw = [embeddings_raw]

            if not isinstance(embeddings_raw, list):
                raise EmbeddingGenerationError("Unexpected embedding format from Ollama")

            embeddings_list = cast(List[List[float]], embeddings_raw)
            
            # Sanitize embeddings: replace any NaN or Inf values with 0
            # This can happen with some Ollama models under certain conditions
            embeddings_list = self._sanitize_embeddings(embeddings_list)
            
            # Log service usage to OpenTelemetry/OpenSearch
            try:
                # Ollama doesn't return token counts, estimate from text length
                # Rough estimate: ~4 chars per token for English text
                estimated_tokens = sum(len(text) // 4 for text in input_data)
                
                log_embedding_usage(
                    provider="ollama",
                    model=self.model,
                    input_tokens=estimated_tokens,
                    num_texts=len(input_data),
                    embedding_dim=self.embedding_dim or len(embeddings_list[0]) if embeddings_list else None,
                    latency_ms=self.embedding_time_ms,
                    host=self.base_url,
                )
            except Exception as e:
                logger.warning(f"Failed to log embedding usage: {e}")
            
            return embeddings_list
        
        except ollama.ResponseError as e:  # Use library's exception
            error_msg = f"Ollama API error: {e}"
            logger.error(error_msg)
            raise EmbeddingGenerationError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during embedding generation: {e}"
            logger.error(error_msg)
            raise EmbeddingGenerationError(error_msg)

    def health_check(self) -> bool:
        """Verify Ollama server is reachable and responsive.

        Attempts a lightweight call to the client. Returns True when the server
        responds without raising an exception, otherwise False.
        """
        try:
            # Prefer a non-destructive models() call if available
            if hasattr(self.client, "show"):
                show = self.client.show(model=self.model)
            else:
                # Fallback: tiny embed call (should be safe) to detect reachability
                _ = self.client.embed(model=self.model, input="healthcheck")
            return True
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    def _sanitize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Sanitize embeddings by replacing NaN and Inf values with 0.
        
        Ollama can occasionally return NaN or Inf values due to numerical issues
        in certain models or configurations. This method detects and replaces such
        values to prevent JSON serialization errors.
        
        Args:
            embeddings: List of embedding vectors potentially containing NaN/Inf values
            
        Returns:
            Sanitized embeddings with NaN/Inf values replaced by 0
        """
        sanitized = []
        had_invalid = False
        
        for vec in embeddings:
            sanitized_vec = []
            for val in vec:
                if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                    sanitized_vec.append(0.0)
                    had_invalid = True
                else:
                    sanitized_vec.append(val)
            sanitized.append(sanitized_vec)
        
        if had_invalid:
            logger.warning(
                f"Detected and sanitized NaN/Inf values in {len(embeddings)} embeddings. "
                "This may indicate numerical instability in the Ollama model. "
                "Consider restarting Ollama or using a different model."
            )
        
        return sanitized
