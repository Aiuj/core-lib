"""DeepInfra native reranker client."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import requests

from .base import BaseRerankerClient, RerankResult, RerankerError
from .reranker_config import reranker_settings
from core_lib.tracing.logger import get_module_logger

logger = get_module_logger()


class DeepInfraRerankerClient(BaseRerankerClient):
    """Reranker client for DeepInfra's native inference endpoint.

    DeepInfra expects ``POST /v1/inference/{model}`` with ``queries`` and
    ``documents`` arrays and returns a parallel ``scores`` array.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[int] = None,
        cache_duration_seconds: Optional[int] = None,
        return_documents: bool = True,
        **kwargs,
    ):
        super().__init__(
            model=model or "Qwen/Qwen3-Reranker-0.6B",
            cache_duration_seconds=cache_duration_seconds,
            return_documents=return_documents,
        )
        self.base_url = (base_url or "https://api.deepinfra.com/v1/inference").rstrip("/")
        self.token = token or getattr(reranker_settings, "infinity_token", None)
        self.timeout = timeout or getattr(reranker_settings, "timeout", 30)
        self._session = requests.Session()

    @property
    def host(self) -> str:
        return self.base_url

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/{self.model}"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _post(self, payload: dict) -> dict:
        response = self._session.post(
            self.endpoint,
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RerankerError("DeepInfra returned a non-object response")
        return data

    def _rerank_raw(
        self, query: str, documents: List[str], top_k: int
    ) -> Tuple[List[RerankResult], Optional[Dict[str, int]]]:
        start = time.time()
        try:
            data = self._post({"queries": [query], "documents": documents})
            scores = data.get("scores")
            if not isinstance(scores, list) or len(scores) != len(documents):
                raise RerankerError(
                    f"DeepInfra response contained {len(scores) if isinstance(scores, list) else 0} "
                    f"scores for {len(documents)} documents"
                )
            results = [
                RerankResult(index=i, score=float(score))
                for i, score in enumerate(scores)
            ]
            usage = None
            if data.get("input_tokens") is not None:
                usage = {"input_tokens": int(data["input_tokens"]), "output_tokens": 0}
            self.rerank_time_ms = (time.time() - start) * 1000
            return results, usage
        except RerankerError:
            self.rerank_time_ms = (time.time() - start) * 1000
            raise
        except Exception as exc:
            self.rerank_time_ms = (time.time() - start) * 1000
            raise RerankerError(f"DeepInfra reranking failed: {exc}") from exc

    def health_check(self) -> bool:
        try:
            data = self._post({"queries": ["health check"], "documents": ["health check"]})
            scores = data.get("scores")
            return isinstance(scores, list) and len(scores) == 1
        except Exception as exc:
            logger.warning("DeepInfra reranker health check failed: %s", exc)
            return False

    def close(self) -> None:
        self._session.close()

