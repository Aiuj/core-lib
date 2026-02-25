"""Local reranker client using cross-encoder models.

Uses sentence-transformers or transformers library to run cross-encoder
reranking models locally.
"""
import time
from typing import List, Optional, Tuple, Dict

from .reranker_config import reranker_settings
from .base import BaseRerankerClient, RerankerError, RerankResult
from core_lib.tracing.logger import get_module_logger

logger = get_module_logger()

# Try to import sentence-transformers
try:
    from sentence_transformers import CrossEncoder
    _cross_encoder_available = True
except ImportError:
    CrossEncoder = None
    _cross_encoder_available = False


class LocalRerankerClient(BaseRerankerClient):
    """Client for reranking using local cross-encoder models.
    
    Uses sentence-transformers CrossEncoder for efficient local reranking.
    
    Recommended models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2: Fast, English
    - cross-encoder/ms-marco-TinyBERT-L-2-v2: Very fast, English
    - BAAI/bge-reranker-base: Balanced, English
    - BAAI/bge-reranker-large: High quality, English
    - BAAI/bge-reranker-v2-m3: Multilingual
    """

    def __init__(
        self,
        model: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        cache_duration_seconds: Optional[int] = None,
        return_documents: bool = True,
        **kwargs
    ):
        """Initialize local reranker client.
        
        Args:
            model: Model name from HuggingFace
            device: Device to run on ('cpu', 'cuda', 'auto')
            cache_dir: Directory to cache downloaded models
            trust_remote_code: Whether to trust remote code
            cache_duration_seconds: How long to cache results
            return_documents: Whether to include document text in results
            **kwargs: Additional parameters
        """
        if not _cross_encoder_available or CrossEncoder is None:
            raise ImportError(
                "sentence-transformers is required for LocalRerankerClient. "
                "Install with: pip install sentence-transformers"
            )

        # Set model before calling super().__init__
        if model is None:
            model = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast default
        
        super().__init__(
            model=model,
            cache_duration_seconds=cache_duration_seconds,
            return_documents=return_documents,
        )
        
        # Set device
        self.device = device or reranker_settings.device or "auto"
        if self.device == "auto":
            self.device = None  # Let CrossEncoder auto-detect
        
        # Set cache directory
        self.cache_dir = cache_dir or reranker_settings.cache_dir
        
        # Initialize cross-encoder
        try:
            self.cross_encoder = CrossEncoder(
                self.model,
                device=self.device,
                trust_remote_code=trust_remote_code or reranker_settings.trust_remote_code,
            )
            logger.debug(
                f"Initialized local reranker: model={self.model}, device={self.device}"
            )
        except Exception as e:
            error_msg = f"Failed to load cross-encoder model {self.model}: {e}"
            logger.error(error_msg)
            raise RerankerError(error_msg)

    @property
    def host(self) -> Optional[str]:
        """Local in-process model, not a remote service."""
        return "local"

    def _rerank_raw(
        self,
        query: str,
        documents: List[str],
        top_k: int,
    ) -> Tuple[List[RerankResult], Optional[Dict[str, int]]]:
        """Perform reranking using local cross-encoder."""
        start_time = time.time()
        
        try:
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]
            
            # Get scores from cross-encoder
            scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
            
            # Create results
            results = []
            for i, score in enumerate(scores):
                result = RerankResult(
                    index=i,
                    score=float(score),
                    document=documents[i] if self.return_documents else None,
                )
                results.append(result)
            
            self.rerank_time_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"Reranked {len(documents)} documents in {self.rerank_time_ms:.2f}ms "
                f"using local model ({self.model})"
            )
            
            return results, None
            
        except Exception as e:
            self.rerank_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Error reranking with local model: {e}"
            logger.error(error_msg)
            raise RerankerError(error_msg)

    def health_check(self) -> bool:
        """Check if the local model is loaded and working."""
        try:
            # Try a minimal rerank call
            scores = self.cross_encoder.predict([["test query", "test document"]])
            return len(scores) == 1
        except Exception as e:
            logger.warning(f"Local reranker health check failed: {e}")
            return False
