"""Cloudflare Workers AI embeddings client."""
from __future__ import annotations

import time
from typing import List, Optional
from urllib.parse import quote

import requests

from .base import BaseEmbeddingClient, EmbeddingGenerationError
from .embeddings_config import embeddings_settings
from core_lib.tracing.logger import get_module_logger

logger = get_module_logger()


class CloudflareEmbeddingClient(BaseEmbeddingClient):
    """Generate embeddings through the Cloudflare Workers AI REST API."""

    def __init__(self, model: Optional[str] = None, embedding_dim: Optional[int] = None,
                 use_l2_norm: bool = True, norm_method: Optional[str] = None,
                 account_id: Optional[str] = None, api_token: Optional[str] = None,
                 base_url: Optional[str] = None, timeout: Optional[int] = None, **kwargs):
        super().__init__(model=model or "@cf/qwen/qwen3-embedding-0.6b",
                         embedding_dim=embedding_dim, use_l2_norm=use_l2_norm,
                         norm_method=norm_method)
        self.account_id = account_id or getattr(embeddings_settings, "cloudflare_account_id", None)
        self.api_token = api_token or getattr(embeddings_settings, "cloudflare_api_token", None)
        if not self.account_id or not self.api_token:
            raise ValueError("Cloudflare account_id and api_token are required")
        root = (base_url or "https://api.cloudflare.com/client/v4").rstrip("/")
        self.endpoint = f"{root}/accounts/{self.account_id}/ai/run/{quote(self.model, safe='@/') }"
        self.timeout = timeout or getattr(embeddings_settings, "timeout", 30)

    def _generate_embedding_raw(self, texts: List[str]) -> List[List[float]]:
        start = time.time()
        try:
            response = requests.post(self.endpoint, headers={"Authorization": f"Bearer {self.api_token}"},
                                     json={"text": texts}, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            result = payload.get("result", payload)
            embeddings = result.get("data") if isinstance(result, dict) else None
            if not isinstance(embeddings, list):
                raise ValueError("Cloudflare response did not contain result.data")
            return embeddings
        except Exception as exc:
            self.embedding_time_ms = (time.time() - start) * 1000
            raise EmbeddingGenerationError(f"Cloudflare embedding generation failed: {exc}") from exc

    def health_check(self) -> bool:
        try:
            return bool(self._generate_embedding_raw(["health check"]))
        except Exception as exc:
            logger.warning("Cloudflare embedding health check failed: %s", exc)
            return False

