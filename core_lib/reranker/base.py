"""Base reranker client interface and helpers."""
from dataclasses import dataclass
from typing import List, Optional, Union, Any, Tuple, Dict
import hashlib
import json
import math

from .reranker_config import reranker_settings
from ..cache.cache_manager import cache_get, cache_set
from core_lib.tracing.service_usage import log_reranker_usage
from core_lib.tracing.logger import get_module_logger
import time

logger = get_module_logger()


class RerankerError(Exception):
    """Raised when reranking fails."""
    pass


@dataclass
class RerankResult:
    """Result from reranking a single document.
    
    Attributes:
        index: Original index of the document in the input list
        score: Relevance score from the reranker (higher = more relevant)
        document: The document text (optional, for convenience)
    """
    index: int
    score: float
    document: Optional[str] = None
    
    def __repr__(self) -> str:
        doc_preview = self.document[:50] + "..." if self.document and len(self.document) > 50 else self.document
        return f"RerankResult(index={self.index}, score={self.score:.4f}, document='{doc_preview}')"


class BaseRerankerClient:
    """Abstract base client for reranker providers.

    Concrete implementations should implement `_rerank_raw` which takes a query
    and list of documents and returns scored results.
    
    Caching behavior:
        - Cache is enabled by default with cache_duration_seconds > 0
        - Set cache_duration_seconds to 0 to disable caching entirely
        - When disabled, all rerank requests bypass the cache
    """

    def __init__(
        self,
        model: Optional[str] = None,
        cache_duration_seconds: Optional[int] = None,
        return_documents: bool = True,
    ):
        """Initialize reranker client.
        
        Args:
            model: Model name to use for reranking
            cache_duration_seconds: How long to cache results (0 to disable)
            return_documents: Whether to include document text in results
        """
        # Use provided values, otherwise fall back to settings defaults
        self.model = model if model is not None else reranker_settings.model
        self.cache_duration_seconds = (
            cache_duration_seconds 
            if cache_duration_seconds is not None 
            else reranker_settings.cache_duration_seconds
        )
        self.return_documents = return_documents
        self.rerank_time_ms = 0
        
        logger.debug(
            f"Initialized reranker client: model={self.model}, "
            f"cache_enabled={self.cache_duration_seconds > 0}"
        )

    @property
    def host(self) -> Optional[str]:
        """Service host URL used for requests. Override in concrete providers."""
        return None

    def _telemetry_provider_name(self) -> str:
        """Return the provider name recorded in usage telemetry."""
        return self.__class__.__name__.replace("RerankerClient", "").lower()

    def _generate_cache_key(self, query: str, documents: List[str], top_k: Optional[int]) -> str:
        """Generate a cache key for the given query and documents."""
        cache_data = {
            "query": query,
            "documents": documents,
            "top_k": top_k,
            "model": self.model,
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"rerank:{hashlib.sha256(cache_string.encode()).hexdigest()}"

    @staticmethod
    def _estimate_input_tokens(query: str, documents: List[str]) -> int:
        """Estimate submitted cross-encoder tokens when a provider omits usage.

        A reranker receives one ``(query, document)`` pair per candidate, so
        the query is intentionally counted once for every document.  TEI does
        not currently return token usage; byte-based estimation is stable and
        multilingual-friendly, but is explicitly marked as an estimate in
        telemetry rather than represented as provider-reported usage.
        """
        return sum(
            math.ceil(len(f"{query}\n{document}".encode("utf-8")) / 4)
            for document in documents
        )

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank documents by relevance to the query.
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (None = all documents)
            
        Returns:
            List of RerankResult objects sorted by score (highest first)
        """
        if not documents:
            return []
        
        if top_k is None:
            top_k = len(documents)
        
        # Check cache first (only if caching is enabled)
        if self.cache_duration_seconds > 0:
            cache_key = self._generate_cache_key(query, documents, top_k)
            cached_result = cache_get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for rerank: {cache_key[:20]}...")
                # Reconstruct RerankResult objects from cached data
                return [
                    RerankResult(
                        index=r["index"],
                        score=r["score"],
                        document=r.get("document")
                    )
                    for r in cached_result
                ]
        
        # Generate new reranking
        start_time = time.time()
        results, usage = self._rerank_raw(query, documents, top_k)
        latency_ms = (time.time() - start_time) * 1000
        
        # TEI and several local rerankers return scores without token usage.
        # Preserve provider-reported input usage where available; otherwise
        # expose an explicit estimate of the submitted query/document pairs.
        raw_usage = usage or {}
        reported_input_tokens = raw_usage.get("input_tokens")
        input_tokens_estimated = reported_input_tokens is None
        input_tokens = (
            int(reported_input_tokens)
            if reported_input_tokens is not None
            else self._estimate_input_tokens(query, documents)
        )

        # Cross-encoders produce relevance scores, not generated text.  A
        # zero output token count is therefore an actual metric, not missing
        # telemetry.  Retain a provider value if one is supplied for billing.
        output_tokens = int(raw_usage.get("output_tokens") or 0)

        # Log usage
        log_reranker_usage(
            provider=self._telemetry_provider_name(),
            model=self.model,
            num_documents=len(documents),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            host=self.host,
            metadata={"input_tokens_estimated": input_tokens_estimated},
        )
        
        # Add document text if requested
        if self.return_documents:
            for result in results:
                if result.document is None and 0 <= result.index < len(documents):
                    result.document = documents[result.index]
        
        # Sort by score (highest first) and limit to top_k
        results = sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
        
        # Cache the result (only if caching is enabled)
        if self.cache_duration_seconds > 0:
            cache_key = self._generate_cache_key(query, documents, top_k)
            # Convert to dict for JSON serialization
            cache_data = [
                {"index": r.index, "score": r.score, "document": r.document}
                for r in results
            ]
            cache_set(cache_key, cache_data, ttl=self.cache_duration_seconds)
            logger.debug(f"Cached rerank result: {cache_key[:20]}...")
        
        return results

    def rerank_with_scores(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[tuple]:
        """Rerank documents and return (index, score, document) tuples.
        
        Convenience method for cases where tuple unpacking is preferred.
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return
            
        Returns:
            List of (index, score, document) tuples sorted by score
        """
        results = self.rerank(query, documents, top_k)
        return [(r.index, r.score, r.document) for r in results]

    def _rerank_raw(
        self,
        query: str,
        documents: List[str],
        top_k: int,
    ) -> Tuple[List[RerankResult], Optional[Dict[str, int]]]:
        """Abstract method for performing the actual reranking.
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return
            
        Returns:
            Tuple of (List[RerankResult], usage_dict)
        """
        raise NotImplementedError()

    def health_check(self) -> bool:
        """Optional health check. Return True if service reachable."""
        return True

    def is_in_warmup(self) -> bool:
        """Return True while this provider is in a WoL warmup window."""
        return False

    def get_rerank_time_ms(self) -> float:
        """Get the time taken for the last rerank operation in milliseconds."""
        return self.rerank_time_ms
