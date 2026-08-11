"""Cloudflare Workers AI reranker client."""
from __future__ import annotations
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
import requests
from .base import BaseRerankerClient, RerankResult, RerankerError
from .reranker_config import reranker_settings
from core_lib.tracing.logger import get_module_logger
logger = get_module_logger()


class CloudflareRerankerClient(BaseRerankerClient):
    """Rerank with Cloudflare's ``@cf/baai/bge-reranker-base`` model."""
    def __init__(self, model: Optional[str] = None, account_id: Optional[str] = None,
                 api_token: Optional[str] = None, base_url: Optional[str] = None,
                 timeout: Optional[int] = None, **kwargs):
        super().__init__(model=model or "@cf/baai/bge-reranker-base", **kwargs)
        self.account_id = account_id or getattr(reranker_settings, "cloudflare_account_id", None)
        self.api_token = api_token or getattr(reranker_settings, "cloudflare_api_token", None)
        if not self.account_id or not self.api_token:
            raise ValueError("Cloudflare account_id and api_token are required")
        root = (base_url or "https://api.cloudflare.com/client/v4").rstrip("/")
        self.endpoint = f"{root}/accounts/{self.account_id}/ai/run/{quote(self.model, safe='@/') }"
        self.timeout = timeout or getattr(reranker_settings, "timeout", 30)

    @property
    def host(self) -> str:
        return self.endpoint

    def _post(self, payload: dict) -> dict:
        response = requests.post(self.endpoint, headers={"Authorization": f"Bearer {self.api_token}"},
                                 json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _rerank_raw(self, query: str, documents: List[str], top_k: int) -> Tuple[List[RerankResult], Optional[Dict[str, int]]]:
        start = time.time()
        try:
            payload = self._post({"query": query, "contexts": [{"text": d} for d in documents], "top_k": top_k})
            items = payload.get("result", payload.get("response", []))
            if not isinstance(items, list):
                raise RerankerError("Cloudflare response did not contain a result list")
            results = []
            for position, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                index = item.get("id", item.get("index", position))
                score = item.get("score", item.get("relevance_score"))
                if score is not None:
                    results.append(RerankResult(index=int(index), score=float(score)))
            self.rerank_time_ms = (time.time() - start) * 1000
            return results, None
        except RerankerError:
            raise
        except Exception as exc:
            self.rerank_time_ms = (time.time() - start) * 1000
            raise RerankerError(f"Cloudflare reranking failed: {exc}") from exc

    def health_check(self) -> bool:
        try:
            return bool(self._post({"query": "health check", "contexts": [{"text": "health check"}], "top_k": 1}).get("result"))
        except Exception as exc:
            logger.warning("Cloudflare reranker health check failed: %s", exc)
            return False

