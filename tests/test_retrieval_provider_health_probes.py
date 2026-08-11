"""Tests for live embedding and reranker provider health probes."""

from unittest.mock import Mock, patch

from core_lib.embeddings.health_probe import check_embedding_providers_health
from core_lib.reranker.health_probe import check_reranker_providers_health
from core_lib.reranker.base import RerankResult


def test_embedding_probe_uses_public_generation_path() -> None:
    client = Mock()
    client.generate_embedding_single.return_value = [0.1, 0.2]

    with patch("core_lib.embeddings.health_probe.EmbeddingFactory.create", return_value=client):
        result = check_embedding_providers_health([
            {"provider": "deepinfra", "model": "Qwen/Qwen3-Embedding-0.6B"}
        ])[0]

    assert result.healthy is True
    client.generate_embedding_single.assert_called_once_with("embedding health check")
    client.health_check.assert_not_called()
    assert client.cache_duration_seconds == 0


def test_reranker_probe_uses_public_rerank_path() -> None:
    client = Mock()
    client.rerank.return_value = [RerankResult(index=0, score=0.9)]

    with patch("core_lib.reranker.health_probe.RerankerFactory.create", return_value=client):
        result = check_reranker_providers_health([
            {"provider": "deepinfra", "model": "Qwen/Qwen3-Reranker-0.6B"}
        ])[0]

    assert result.healthy is True
    client.rerank.assert_called_once_with(
        "health check",
        ["health check relevant document", "health check alternate document"],
        top_k=2,
    )
    client.health_check.assert_not_called()
    assert client.cache_duration_seconds == 0
