"""Hugging Face Text Embeddings Inference (TEI) embedding client."""

import time
from typing import List, Optional

from core_lib.api_utils import InfinityAPIClient, InfinityAPIError
from core_lib.tracing.logger import get_module_logger
from core_lib.tracing.service_usage import log_embedding_usage

from .base import BaseEmbeddingClient, EmbeddingGenerationError
from .embeddings_config import embeddings_settings

logger = get_module_logger()


class TEIEmbeddingClient(BaseEmbeddingClient):
    """Generate embeddings through a Hugging Face TEI server.

    TEI exposes an OpenAI-compatible endpoint at ``POST /v1/embeddings``.
    The shared HTTP transport is reused for authentication, Wake-on-LAN, and
    comma-separated URL failover support.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        use_l2_norm: bool = True,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        token: Optional[str] = None,
        wake_on_lan: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            embedding_dim=embedding_dim,
            use_l2_norm=use_l2_norm,
            **kwargs,
        )
        resolved_url = (
            base_url or embeddings_settings.base_url or "http://localhost:8080"
        )
        resolved_timeout = timeout or embeddings_settings.timeout or 30
        resolved_token = token or embeddings_settings.infinity_token
        self._api_client = InfinityAPIClient(
            base_urls=resolved_url,
            timeout=resolved_timeout,
            token=resolved_token,
            wake_on_lan=wake_on_lan,
        )
        self._last_used_url = self._api_client.base_urls[0]

    @property
    def host(self) -> Optional[str]:
        return self._last_used_url

    @property
    def base_url(self) -> Optional[str]:
        return self._last_used_url

    def is_in_warmup(self) -> bool:
        return self._api_client.is_in_warmup()

    def _generate_embedding_raw(self, texts: List[str]) -> List[List[float]]:
        start_time = time.time()
        request_body = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }
        if self.embedding_dim is not None:
            request_body["dimensions"] = self.embedding_dim

        try:
            data, used_url = self._api_client.post("/v1/embeddings", json=request_body)
            self._last_used_url = used_url
            embeddings = [
                item["embedding"]
                for item in sorted(data["data"], key=lambda item: item["index"])
            ]
            self.embedding_time_ms = (time.time() - start_time) * 1000
            usage = data.get("usage", {})
            log_embedding_usage(
                provider="tei",
                model=self.model,
                input_tokens=usage.get("prompt_tokens"),
                num_texts=len(texts),
                embedding_dim=self.embedding_dim
                or (len(embeddings[0]) if embeddings else None),
                latency_ms=self.embedding_time_ms,
                host=used_url,
            )
            return embeddings
        except InfinityAPIError as exc:
            self.embedding_time_ms = (time.time() - start_time) * 1000
            message = f"TEI embedding failed: {exc}"
            if getattr(exc, "is_warmup", False):
                logger.debug(message)
            else:
                logger.error(message)
            raise EmbeddingGenerationError(message) from exc
        except Exception as exc:
            self.embedding_time_ms = (time.time() - start_time) * 1000
            message = f"Unexpected error generating embeddings with TEI: {exc}"
            logger.error(message)
            raise EmbeddingGenerationError(message) from exc

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

    def get_model_info(self) -> dict:
        try:
            data, _ = self._api_client.get("/info")
            return data
        except Exception as exc:
            logger.warning("Failed to get TEI model info: %s", exc)
            return {"model_id": self.model, "model_type": "embedding"}
