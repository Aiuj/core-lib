"""Tests for live embedding and reranker provider health probes."""

from unittest.mock import Mock, patch

from core_lib.embeddings.health_probe import check_embedding_providers_health
from core_lib.reranker.health_probe import check_reranker_providers_health
from core_lib.reranker.base import RerankResult
from core_lib.llm.provider_registry import ProviderConfig
from core_lib.llm.startup_preflight import check_llm_providers_health


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


def test_embedding_probe_disables_wol_when_requested() -> None:
    client = Mock()
    client.generate_embedding_single.return_value = [0.1]

    with patch("core_lib.embeddings.health_probe.EmbeddingFactory.create", return_value=client) as create:
        check_embedding_providers_health(
            [{"provider": "tei", "model": "qwen", "wake_on_lan": {"enabled": True}}],
            allow_wol=False,
        )

    assert create.call_args.kwargs["wake_on_lan"] == {"enabled": False}


def test_reranker_probe_disables_wol_and_wakeup_service_when_requested() -> None:
    client = Mock()
    client.rerank.return_value = [RerankResult(index=0, score=0.9)]

    with patch("core_lib.reranker.health_probe.RerankerFactory.create", return_value=client) as create:
        check_reranker_providers_health(
            [{
                "provider": "tei",
                "model": "qwen",
                "wake_on_lan": {"enabled": True},
                "wakeup_service": {"enabled": True},
            }],
            allow_wol=False,
        )

    assert create.call_args.kwargs["wake_on_lan"] == {"enabled": False}
    assert create.call_args.kwargs["wakeup_service"] == {"enabled": False}


def test_llm_probe_disables_wol_when_requested() -> None:
    provider = ProviderConfig(
        provider="openai",
        model="qwen",
        host="http://example.invalid/v1",
        wake_on_lan={"enabled": True},
    )
    client = Mock()
    client.chat.return_value = {"choices": [{"message": {"content": "OK"}}]}
    observed = {}

    def fake_to_client(config):
        observed["wake_on_lan"] = config.wake_on_lan
        return client

    with patch.object(ProviderConfig, "to_client", fake_to_client):
        result = check_llm_providers_health([provider], enable_wol=False)

    assert result[0].healthy is True
    assert observed["wake_on_lan"] == {"enabled": False}
