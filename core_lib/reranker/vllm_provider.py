"""vLLM reranker client using the standard Cohere-compatible ``/rerank`` API."""

from typing import List

from .infinity_provider import InfinityRerankerClient
from core_lib.tracing.logger import get_module_logger


logger = get_module_logger()


class VLLMRerankerClient(InfinityRerankerClient):
    """Rerank through a vLLM pooling server.

    vLLM and Infinity expose the same Cohere-compatible request/response shape
    at ``POST /rerank``. The subclass keeps provider identity accurate and uses
    vLLM's OpenAI-compatible ``GET /v1/models`` discovery endpoint.
    """

    def get_available_models(self) -> List[str]:
        try:
            data, _ = self._api_client.get("/v1/models")
            return [
                item["id"]
                for item in data.get("data", [])
                if isinstance(item, dict) and item.get("id")
            ]
        except Exception as exc:
            logger.warning("Failed to get available vLLM models: %s", exc)
            return []
