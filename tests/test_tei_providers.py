"""Tests for Hugging Face Text Embeddings Inference providers."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from core_lib.api_utils import InfinityAPIClient, InfinityAPIError
from core_lib.config.embeddings_settings import EmbeddingsSettings
from core_lib.embeddings.base import BaseEmbeddingClient
from core_lib.embeddings.factory import EmbeddingFactory
from core_lib.embeddings.fallback_client import FallbackEmbeddingClient
from core_lib.embeddings.tei_provider import TEIEmbeddingClient
from core_lib.reranker.factory import RerankerFactory
from core_lib.reranker.base import BaseRerankerClient, RerankResult
from core_lib.reranker.fallback_client import FallbackRerankerClient
from core_lib.reranker.reranker_config import RerankerSettings
from core_lib.reranker.tei_provider import TEIRerankerClient


def test_tei_embedding_uses_openai_compatible_endpoint() -> None:
    client = TEIEmbeddingClient(
        model="Qwen/Qwen3-Embedding-0.6B",
        base_url="http://tei-embed:8080",
        use_l2_norm=False,
        cache_duration_seconds=0,
    )
    response = {
        "data": [
            {"index": 1, "embedding": [0.3, 0.4]},
            {"index": 0, "embedding": [0.1, 0.2]},
        ],
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }

    with (
        patch.object(
            client._api_client, "post", return_value=(response, "http://tei-embed:8080")
        ) as post,
        patch("core_lib.embeddings.tei_provider.log_embedding_usage"),
    ):
        embeddings = client.generate_embedding_batch(["first", "second"])

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    (endpoint,) = post.call_args.args
    assert endpoint == "/v1/embeddings"
    assert post.call_args.kwargs["json"]["input"] == ["first", "second"]


def test_tei_embedding_splits_oversized_batches_and_preserves_order() -> None:
    client = TEIEmbeddingClient(
        model="Qwen/Qwen3-Embedding-0.6B",
        base_url="http://tei-embed:8080",
        max_batch_size=128,
        use_l2_norm=False,
        cache_duration_seconds=0,
    )

    def respond(_endpoint, *, json):
        return (
            {
                "data": [
                    {"index": index, "embedding": [float(text)]}
                    for index, text in reversed(list(enumerate(json["input"])))
                ],
                "usage": {"prompt_tokens": len(json["input"])},
            },
            "http://tei-embed:8080",
        )

    with (
        patch.object(client._api_client, "post", side_effect=respond) as post,
        patch("core_lib.embeddings.tei_provider.log_embedding_usage"),
    ):
        embeddings = client.generate_embedding_batch([str(i) for i in range(306)])

    assert post.call_count == 3
    assert [len(call.kwargs["json"]["input"]) for call in post.call_args_list] == [
        128,
        128,
        50,
    ]
    assert embeddings == [[float(i)] for i in range(306)]


def test_infinity_error_includes_top_level_validation_message() -> None:
    response = Mock(status_code=422)
    response.json.return_value = {
        "message": "batch size 306 > maximum allowed batch size 256",
        "code": 422,
        "type": "Validation",
    }
    response.raise_for_status.side_effect = requests.HTTPError(response=response)
    client = InfinityAPIClient("http://tei-embed:8080")

    with (
        patch("requests.post", return_value=response),
        pytest.raises(InfinityAPIError, match="batch size 306"),
    ):
        client.post("/v1/embeddings", json={"input": ["x"] * 306})


def test_tei_reranker_uses_native_request_and_response_shapes() -> None:
    client = TEIRerankerClient(
        model="Alibaba-NLP/gte-multilingual-reranker-base",
        base_url="http://tei-rerank:8080",
        cache_duration_seconds=0,
    )
    response = [
        {"index": 1, "score": 0.2, "text": "second"},
        {"index": 0, "score": 0.9, "text": "first"},
    ]

    with patch.object(
        client._api_client,
        "post",
        return_value=(response, "http://tei-rerank:8080"),
    ) as post:
        results = client.rerank("query", ["first", "second"], top_k=1)

    assert [(result.index, result.score, result.document) for result in results] == [
        (0, 0.9, "first")
    ]
    (endpoint,) = post.call_args.args
    assert endpoint == "/rerank"
    assert post.call_args.kwargs["json"] == {
        "query": "query",
        "texts": ["first", "second"],
        "return_text": True,
    }


def test_factories_register_tei() -> None:
    embedding = EmbeddingFactory.create(
        provider="tei",
        model="Qwen/Qwen3-Embedding-0.6B",
        base_url="http://tei-embed:8080",
    )
    reranker = RerankerFactory.create(
        provider="tei",
        model="Alibaba-NLP/gte-multilingual-reranker-base",
        base_url="http://tei-rerank:8080",
    )

    assert isinstance(embedding, TEIEmbeddingClient)
    assert isinstance(reranker, TEIRerankerClient)


def test_yaml_config_loads_tei_embedding_and_reranker(
    tmp_path: Path, monkeypatch
) -> None:
    config_file = tmp_path / "providers.yaml"
    config_file.write_text(
        """
embedding_providers:
  - provider: tei
    model: Qwen/Qwen3-Embedding-0.6B
    base_url: http://tei-embed:8110
    max_batch_size: 96
    priority: 1
reranker_providers:
  - provider: tei
    model: Alibaba-NLP/gte-multilingual-reranker-base
    base_url: http://tei-rerank:8111
    priority: 1
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_PROVIDERS_FILE", str(config_file))

    embedding_settings = EmbeddingsSettings.from_env(load_dotenv=False)
    reranker_settings = RerankerSettings.from_env(load_dotenv=False)

    assert embedding_settings.provider == "tei"
    assert embedding_settings.base_url == "http://tei-embed:8110"
    assert embedding_settings.provider_configs[0]["max_batch_size"] == 96
    assert reranker_settings.provider == "tei"
    assert reranker_settings.infinity_url == "http://tei-rerank:8111"


class _DeepInfraStub(BaseEmbeddingClient):
    def __init__(self) -> None:
        super().__init__(
            model="Qwen/Qwen3-Embedding-0.6B",
            use_l2_norm=False,
            cache_duration_seconds=0,
        )
        self.calls = 0

    def _generate_embedding_raw(self, texts):
        self.calls += 1
        return [[0.7, 0.8] for _ in texts]


def test_tei_wol_immediately_falls_through_to_deepinfra(monkeypatch) -> None:
    primary = TEIEmbeddingClient(
        model="Qwen/Qwen3-Embedding-0.6B",
        base_url="http://sleeping-tei:8110",
        timeout=30,
        use_l2_norm=False,
        cache_duration_seconds=0,
        wake_on_lan={
            "enabled": True,
            "initial_timeout_seconds": 0.1,
            "warmup_seconds": 90,
            "mac_address": "FC:34:97:9E:C8:AF",
            "broadcast_ip": "192.168.1.255",
            "port": 9,
            "wait_seconds": 30,
        },
    )
    secondary = _DeepInfraStub()
    fallback = FallbackEmbeddingClient(
        providers=[primary, secondary],
        use_l2_norm=False,
        cache_duration_seconds=0,
        max_retries_per_provider=1,
        use_health_cache=False,
    )
    post = Mock(side_effect=requests.exceptions.ConnectionError("host is sleeping"))
    monkeypatch.setattr("core_lib.api_utils.infinity_api.requests.post", post)
    wake = Mock()
    monkeypatch.setattr(primary._api_client.wake_on_lan, "_send_magic_packet", wake)
    sleep = Mock()
    monkeypatch.setattr("core_lib.api_utils.wake_on_lan.time.sleep", sleep)

    result = fallback.generate_embedding_single("serve this without waiting")

    assert result == [0.7, 0.8]
    assert secondary.calls == 1
    assert post.call_count == 1
    assert post.call_args.kwargs["timeout"] == 0.1
    wake.assert_called_once()
    sleep.assert_not_called()
    assert primary.is_in_warmup() is True

    # Requests during warmup bypass TEI without another connection attempt.
    second = fallback.generate_embedding_single("second request during warmup")
    assert second == [0.7, 0.8]
    assert post.call_count == 1
    assert secondary.calls == 2

    # Once warmup expires, the priority-ordered TEI primary is tried again.
    primary._api_client.wake_on_lan._waking_timestamps[
        "http://sleeping-tei:8110"
    ] -= 91
    tei_response = Mock()
    tei_response.raise_for_status.return_value = None
    tei_response.json.return_value = {
        "data": [{"index": 0, "embedding": [0.9, 1.0]}],
        "usage": {"prompt_tokens": 3, "total_tokens": 3},
    }
    post.side_effect = None
    post.return_value = tei_response

    third = fallback.generate_embedding_single("request after warmup")
    assert third == [0.9, 1.0]
    assert post.call_count == 2
    assert secondary.calls == 2


class _DeepInfraRerankerStub(BaseRerankerClient):
    def __init__(self) -> None:
        super().__init__(model="fallback-reranker", cache_duration_seconds=0)
        self.calls = 0

    def _rerank_raw(self, query, documents, top_k):
        self.calls += 1
        return [RerankResult(index=1, score=0.8)], None


def test_tei_reranker_wol_immediately_falls_through_and_recovers(monkeypatch) -> None:
    primary = TEIRerankerClient(
        model="Alibaba-NLP/gte-multilingual-reranker-base",
        base_url="http://sleeping-tei:8111",
        timeout=30,
        cache_duration_seconds=0,
        wake_on_lan={
            "enabled": True,
            "initial_timeout_seconds": 0.1,
            "warmup_seconds": 90,
            "mac_address": "FC:34:97:9E:C8:AF",
            "broadcast_ip": "192.168.1.255",
            "port": 9,
            "wait_seconds": 60,
            "retry_timeout_seconds": 30,
        },
    )
    secondary = _DeepInfraRerankerStub()
    fallback = FallbackRerankerClient(
        providers=[primary, secondary],
        cache_duration_seconds=0,
        max_retries_per_provider=1,
        use_health_cache=False,
    )
    post = Mock(side_effect=requests.exceptions.ConnectionError("host is sleeping"))
    monkeypatch.setattr("core_lib.api_utils.infinity_api.requests.post", post)
    wake = Mock()
    monkeypatch.setattr(primary._api_client.wake_on_lan, "_send_magic_packet", wake)
    sleep = Mock()
    monkeypatch.setattr("core_lib.api_utils.wake_on_lan.time.sleep", sleep)

    first = fallback.rerank("query", ["irrelevant", "relevant"], top_k=1)
    assert first[0].index == 1
    assert post.call_count == 1
    assert post.call_args.kwargs["timeout"] == 0.1
    wake.assert_called_once()
    sleep.assert_not_called()

    second = fallback.rerank("query 2", ["irrelevant", "relevant"], top_k=1)
    assert second[0].index == 1
    assert post.call_count == 1
    assert secondary.calls == 2

    primary._api_client.wake_on_lan._waking_timestamps[
        "http://sleeping-tei:8111"
    ] -= 91
    tei_response = Mock()
    tei_response.raise_for_status.return_value = None
    tei_response.json.return_value = [
        {"index": 0, "score": 0.95, "text": "primary result"}
    ]
    post.side_effect = None
    post.return_value = tei_response

    third = fallback.rerank("query 3", ["primary result", "other"], top_k=1)
    assert third[0].index == 0
    assert post.call_count == 2
    assert secondary.calls == 2
