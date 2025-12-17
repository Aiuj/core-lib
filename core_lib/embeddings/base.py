"""Base embedding client interface and helpers."""
from typing import List, Union, cast, Optional
import numpy as np
import hashlib
import json

from .embeddings_config import embeddings_settings
from ..cache.cache_manager import cache_get, cache_set
from .models_database import (
    get_model_dimension,
    supports_matryoshka,
    get_model_prefixes,
)
from .embedding_utils import (
    normalize_embedding_dimension,
    get_best_normalization_method,
)
from core_lib.tracing.logger import get_module_logger

logger = get_module_logger()


class EmbeddingGenerationError(Exception):
    """Raised when embedding generation fails."""
    pass


class BaseEmbeddingClient:
    """Abstract base client for embedding providers.

    Concrete implementations should implement `_generate_embedding_raw` (which takes List[str] and returns List[List[float]]) and may
    override `normalize`, `_l2_normalize`, and `health_check` as needed.
    
    Caching behavior:
        - Cache is enabled by default with cache_duration_seconds > 0
        - Set cache_duration_seconds to 0 to disable caching entirely
        - When disabled, all embedding requests bypass the cache
    
    Prefix behavior:
        - Some models (E5, BGE) require prefixes for optimal performance
        - Prefixes are auto-detected from the model database by default
        - Can be overridden via settings or constructor parameters
        - Use query_prefix for search queries, passage_prefix for documents
    """

    def __init__(
        self,
        model: str | None = None,
        embedding_dim: int | None = None,
        use_l2_norm: bool = True,
        cache_duration_seconds: int | None = None,
        norm_method: str | None = None,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
        auto_detect_prefixes: bool | None = None,
    ):
        # Use provided values, otherwise fall back to settings defaults
        self.model = model if model is not None else embeddings_settings.model
        self.embedding_dim = embedding_dim if embedding_dim is not None else embeddings_settings.embedding_dimension
        self.cache_duration_seconds = cache_duration_seconds if cache_duration_seconds is not None else embeddings_settings.cache_duration_seconds
        self.embedding_time_ms = 0
        self.use_l2_norm = use_l2_norm
        
        # Get model's native dimension from database
        self.model_native_dim = get_model_dimension(self.model) if self.model else None
        
        # Determine normalization method
        if norm_method is not None:
            # User explicitly specified a method
            self.norm_method = norm_method
        else:
            # Auto-detect best method based on model and dimensions
            self.norm_method = get_best_normalization_method(
                model_name=self.model,
                current_dimension=self.model_native_dim,
                target_dimension=self.embedding_dim,
            )
        
        # Initialize prefix settings
        self._init_prefixes(query_prefix, passage_prefix, auto_detect_prefixes)
        
        logger.debug(
            f"Initialized embedding client: model={self.model}, "
            f"native_dim={self.model_native_dim}, target_dim={self.embedding_dim}, "
            f"norm_method={self.norm_method}, use_l2_norm={self.use_l2_norm}, "
            f"cache_enabled={self.cache_duration_seconds > 0}, "
            f"query_prefix='{self.query_prefix}', passage_prefix='{self.passage_prefix}'"
        )

    def _init_prefixes(
        self,
        query_prefix: str | None,
        passage_prefix: str | None,
        auto_detect: bool | None,
    ) -> None:
        """Initialize query and passage prefixes for the embedding model.
        
        Priority:
        1. Explicit constructor parameter (if not None)
        2. Environment variable / settings (if not None)  
        3. Auto-detect from model database (if auto_detect is True)
        4. Empty string (no prefix)
        
        Args:
            query_prefix: Explicit query prefix from constructor
            passage_prefix: Explicit passage prefix from constructor
            auto_detect: Whether to auto-detect from model database
        """
        # Determine if we should auto-detect
        should_auto_detect = auto_detect if auto_detect is not None else embeddings_settings.auto_detect_prefixes
        
        # Get auto-detected prefixes from model database
        auto_query_prefix, auto_passage_prefix = ("", "")
        if should_auto_detect and self.model:
            auto_query_prefix, auto_passage_prefix = get_model_prefixes(self.model)
        
        # Priority: constructor param > settings > auto-detect > empty
        # For query_prefix
        if query_prefix is not None:
            self.query_prefix = query_prefix
        elif embeddings_settings.query_prefix is not None:
            self.query_prefix = embeddings_settings.query_prefix
        else:
            self.query_prefix = auto_query_prefix
        
        # For passage_prefix
        if passage_prefix is not None:
            self.passage_prefix = passage_prefix
        elif embeddings_settings.passage_prefix is not None:
            self.passage_prefix = embeddings_settings.passage_prefix
        else:
            self.passage_prefix = auto_passage_prefix

    def _apply_prefix(self, text: str, prefix: str) -> str:
        """Apply a prefix to text if not already present."""
        if prefix and not text.startswith(prefix):
            return f"{prefix}{text}"
        return text

    def _generate_cache_key(self, text: str) -> str:
        """Generate a cache key for the given text and model configuration."""
        cache_data = {
            "text": text,
            "model": self.model,
            "embedding_dim": self.embedding_dim,
            "use_l2_norm": self.use_l2_norm,
            "norm_method": self.norm_method,
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"embedding:{hashlib.sha256(cache_string.encode()).hexdigest()}"

    def generate_embedding(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for the given text(s), applying L2 normalization if enabled."""
        if isinstance(text, str):
            return self.generate_embedding_single(text)
        elif isinstance(text, list):
            return self.generate_embedding_batch(text)
        else:
            raise ValueError("Input must be a string or list of strings")

    def generate_embedding_single(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        # Check cache first (only if caching is enabled)
        if self.cache_duration_seconds > 0:
            cache_key = self._generate_cache_key(text)
            cached_result = cache_get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for embedding: {cache_key}")
                return cached_result
        
        # Generate new embedding
        embeddings = self._generate_embedding_raw([text])
        
        # Apply dimension normalization first (before L2 norm)
        if self.embedding_dim is not None:
            embeddings = [
                normalize_embedding_dimension(
                    emb, self.embedding_dim, method=self.norm_method
                )
                for emb in embeddings
            ]
        
        # Then apply L2 normalization if enabled
        if self.use_l2_norm:
            embeddings = self._l2_normalize(embeddings)
        
        result = embeddings[0] if embeddings else []
        
        # Cache the result (only if caching is enabled)
        if self.cache_duration_seconds > 0:
            cache_key = self._generate_cache_key(text)
            cache_set(cache_key, result, ttl=self.cache_duration_seconds)
            logger.debug(f"Cached embedding result: {cache_key}")
        
        return result

    def generate_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text strings."""
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text (only if caching is enabled)
        if self.cache_duration_seconds > 0:
            for i, text in enumerate(texts):
                cache_key = self._generate_cache_key(text)
                cached_result = cache_get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for embedding: {cache_key}")
                    results.append((i, cached_result))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            # Caching disabled, all texts need to be processed
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            embeddings = self._generate_embedding_raw(uncached_texts)
            
            # Apply dimension normalization first (before L2 norm)
            if self.embedding_dim is not None:
                embeddings = [
                    normalize_embedding_dimension(
                        emb, self.embedding_dim, method=self.norm_method
                    )
                    for emb in embeddings
                ]
            
            # Then apply L2 normalization if enabled
            if self.use_l2_norm:
                embeddings = self._l2_normalize(embeddings)
            
            # Cache and collect the new embeddings
            for j, (text, embedding) in enumerate(zip(uncached_texts, embeddings)):
                if self.cache_duration_seconds > 0:
                    cache_key = self._generate_cache_key(text)
                    cache_set(cache_key, embedding, ttl=self.cache_duration_seconds)
                    logger.debug(f"Cached embedding result: {cache_key}")
                results.append((uncached_indices[j], embedding))
        
        # Sort results by original index and return embeddings in order
        results.sort(key=lambda x: x[0])
        return [embedding for _, embedding in results]

    def generate_query_embedding(self, query: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for search queries with appropriate prefix.
        
        Use this method when embedding search queries. For models like E5 or BGE,
        this will automatically apply the query prefix (e.g., "query: ").
        
        Args:
            query: Query text or list of query texts
            
        Returns:
            Embedding vector(s) for the query/queries
        """
        if isinstance(query, str):
            prefixed_query = self._apply_prefix(query, self.query_prefix)
            return self.generate_embedding_single(prefixed_query)
        elif isinstance(query, list):
            prefixed_queries = [self._apply_prefix(q, self.query_prefix) for q in query]
            return self.generate_embedding_batch(prefixed_queries)
        else:
            raise ValueError("Input must be a string or list of strings")

    def generate_passage_embedding(self, passage: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for documents/passages with appropriate prefix.
        
        Use this method when embedding documents, passages, or any content that
        will be retrieved. For models like E5, this will automatically apply
        the passage prefix (e.g., "passage: ").
        
        Args:
            passage: Document/passage text or list of texts
            
        Returns:
            Embedding vector(s) for the passage(s)
        """
        if isinstance(passage, str):
            prefixed_passage = self._apply_prefix(passage, self.passage_prefix)
            return self.generate_embedding_single(prefixed_passage)
        elif isinstance(passage, list):
            prefixed_passages = [self._apply_prefix(p, self.passage_prefix) for p in passage]
            return self.generate_embedding_batch(prefixed_passages)
        else:
            raise ValueError("Input must be a string or list of strings")

    def has_prefixes(self) -> bool:
        """Check if this client has query/passage prefixes configured.
        
        Returns:
            True if either query_prefix or passage_prefix is non-empty
        """
        return bool(self.query_prefix or self.passage_prefix)

    def _generate_embedding_raw(self, texts: List[str]) -> List[List[float]]:
        """Abstract method for generating raw embeddings without normalization.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors, one for each input text.
        """
        raise NotImplementedError()

    def normalize(self, vec: list) -> list:
        """Normalize the embedding vector to the expected dimension.

        Args:
            vec: The embedding vector to normalize.

        Returns:
            The normalized vector (padded, truncated, or as-is).
        """
        dim = self.embedding_dim
        if not isinstance(vec, list):
            return [0.0] * dim
        if len(vec) == dim:
            return vec
        if len(vec) > dim:
            return vec[:dim]
        # pad
        return vec + [0.0] * (dim - len(vec))

    def _l2_normalize(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Apply L2 normalization to a list of embedding vectors using numpy.

        Args:
            embeddings: List of embedding vectors.

        Returns:
            List of L2 normalized embedding vectors.
        """
        normalized = []
        for vec in embeddings:
            vec_np = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(vec_np)
            if norm > 0:
                normalized_vec = (vec_np / norm).tolist()
            else:
                normalized_vec = vec_np.tolist()  # Avoid division by zero
            normalized.append(normalized_vec)
        return normalized

    def health_check(self) -> bool:
        """Optional health check. Return True if service reachable."""
        return True

    def get_embedding_time_ms(self) -> float:
        return self.embedding_time_ms
