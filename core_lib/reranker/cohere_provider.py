"""Cohere reranker client implementation.

Cohere provides cloud-based reranking API with high-quality models.
Documentation: https://docs.cohere.com/reference/rerank
"""
import time
from typing import List, Optional

try:
    import cohere
    _cohere_available = True
except ImportError:
    cohere = None
    _cohere_available = False

from .reranker_config import reranker_settings
from .base import BaseRerankerClient, RerankerError, RerankResult
from core_lib.tracing.logger import get_module_logger

logger = get_module_logger()


class CohereRerankerClient(BaseRerankerClient):
    """Client for reranking using Cohere API.
    
    Cohere provides high-quality reranking models through their cloud API.
    
    Models:
    - rerank-english-v3.0: Best for English (default)
    - rerank-multilingual-v3.0: Best for multilingual
    - rerank-english-v2.0: Previous generation
    - rerank-multilingual-v2.0: Previous generation multilingual
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_duration_seconds: Optional[int] = None,
        return_documents: bool = True,
        **kwargs
    ):
        """Initialize Cohere reranker client.
        
        Args:
            model: Model name (e.g., 'rerank-english-v3.0')
            api_key: Cohere API key
            cache_duration_seconds: How long to cache results
            return_documents: Whether to include document text in results
            **kwargs: Additional parameters
        """
        if not _cohere_available or cohere is None:
            raise ImportError(
                "cohere is required for CohereRerankerClient. "
                "Install with: pip install cohere"
            )

        # Set model before calling super().__init__
        if model is None:
            model = "rerank-multilingual-v3.0"  # Good default for multilingual
        
        super().__init__(
            model=model,
            cache_duration_seconds=cache_duration_seconds,
            return_documents=return_documents,
        )
        
        # Set API key
        self.api_key = api_key or reranker_settings.api_key
        if not self.api_key:
            raise ValueError(
                "Cohere API key is required. Set COHERE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize Cohere client
        self.client = cohere.Client(api_key=self.api_key)
        
        logger.debug(f"Initialized Cohere reranker: model={self.model}")

    def _rerank_raw(
        self,
        query: str,
        documents: List[str],
        top_k: int,
    ) -> List[RerankResult]:
        """Perform reranking using Cohere API."""
        start_time = time.time()
        
        try:
            # Call Cohere rerank API
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_k,
                return_documents=self.return_documents,
            )
            
            # Extract results
            results = []
            for item in response.results:
                result = RerankResult(
                    index=item.index,
                    score=item.relevance_score,
                    document=item.document.text if hasattr(item, 'document') and item.document else None,
                )
                results.append(result)
            
            self.rerank_time_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"Reranked {len(documents)} documents in {self.rerank_time_ms:.2f}ms "
                f"using Cohere ({self.model})"
            )
            
            return results
            
        except Exception as e:
            self.rerank_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Error reranking with Cohere: {e}"
            logger.error(error_msg)
            raise RerankerError(error_msg)

    def health_check(self) -> bool:
        """Check if Cohere API is accessible."""
        try:
            # Try a minimal rerank call
            self.client.rerank(
                model=self.model,
                query="test",
                documents=["test document"],
                top_n=1,
            )
            return True
        except Exception as e:
            logger.warning(f"Cohere health check failed: {e}")
            return False
