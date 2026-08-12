"""Hugging Face Text Embeddings Inference (TEI) reranker client."""

import time
from typing import Dict, List, Optional, Tuple

from core_lib.api_utils import InfinityAPIClient, InfinityAPIError
from core_lib.tracing.logger import get_module_logger

from .base import BaseRerankerClient, RerankerError, RerankResult
from .reranker_config import reranker_settings

logger = get_module_logger()


class TEIRerankerClient(BaseRerankerClient):
    """Rerank documents through TEI's native ``POST /rerank`` API."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        token: Optional[str] = None,
        wake_on_lan: Optional[dict] = None,
        wakeup_service: Optional[dict] = None,
        cache_duration_seconds: Optional[int] = None,
        return_documents: bool = True,
        **kwargs,
    ):
        super().__init__(
            model=model,
            cache_duration_seconds=cache_duration_seconds,
            return_documents=return_documents,
        )
        resolved_url = (
            base_url or reranker_settings.infinity_url or "http://localhost:8080"
        )
        resolved_timeout = timeout or reranker_settings.timeout or 30
        resolved_token = token or reranker_settings.infinity_token
        self._api_client = InfinityAPIClient(
            base_urls=resolved_url,
            timeout=resolved_timeout,
            token=resolved_token,
            wake_on_lan=wake_on_lan,
            wakeup_service=wakeup_service,
        )
        self._last_used_url = self._api_client.base_urls[0]

    @property
    def host(self) -> Optional[str]:
        return self._last_used_url

    @property
    def base_url(self) -> Optional[str]:
        return self._last_used_url

    def is_in_warmup(self) -> bool:
        """Return whether non-blocking WoL warmup is active."""
        return self._api_client.is_in_warmup()

    def _rerank_raw(
        self,
        query: str,
        documents: List[str],
        top_k: int,
    ) -> Tuple[List[RerankResult], Optional[Dict[str, int]]]:
        start_time = time.time()
        request_body = {
            "query": query,
            "texts": documents,
            "return_text": self.return_documents,
        }
        try:
            data, used_url = self._api_client.post("/rerank", json=request_body)
            self._last_used_url = used_url
            if not isinstance(data, list):
                raise ValueError("TEI rerank response must be a JSON array")
            results = [
                RerankResult(
                    index=int(item["index"]),
                    score=float(item["score"]),
                    document=item.get("text"),
                )
                for item in data
            ]
            self.rerank_time_ms = (time.time() - start_time) * 1000
            return (
                sorted(results, key=lambda item: item.score, reverse=True)[:top_k],
                None,
            )
        except InfinityAPIError as exc:
            self.rerank_time_ms = (time.time() - start_time) * 1000
            message = f"TEI reranking failed: {exc}"
            logger.error(message)
            raise RerankerError(message) from exc
        except Exception as exc:
            self.rerank_time_ms = (time.time() - start_time) * 1000
            message = f"Unexpected error reranking with TEI: {exc}"
            logger.error(message)
            raise RerankerError(message) from exc

    def health_check(self) -> bool:
        return self._api_client.health_check()

    def get_available_models(self) -> List[str]:
        try:
            data, _ = self._api_client.get("/info")
            model_id = data.get("served_model_name") or data.get("model_id")
            return [model_id] if model_id else []
        except Exception as exc:
            logger.warning("Failed to get TEI model info: %s", exc)
            return []
