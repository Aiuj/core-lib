"""Tests for YAML-driven embeddings/reranker provider routing and failover."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from core_lib.config.embeddings_settings import EmbeddingsSettings
from core_lib.reranker.reranker_config import RerankerSettings
from core_lib.embeddings.factory import create_embedding_client
from core_lib.embeddings.fallback_client import FallbackEmbeddingClient
from core_lib.reranker.factory import create_reranker_client
from core_lib.reranker.fallback_client import FallbackRerankerClient


def _write_yaml(tmp_path: Path, content: str) -> Path:
    config_file = tmp_path / "providers.yaml"
    config_file.write_text(content, encoding="utf-8")
    return config_file


def test_embeddings_settings_loads_priority_chain_from_yaml(tmp_path, monkeypatch):
    config_file = _write_yaml(
        tmp_path,
        """
embedding_providers:
  - provider: infinity
    base_url: http://emb-low:7997
    model: low-model
    embedding_dimension: 256
    wake_on_lan:
      enabled: true
      initial_timeout_seconds: 2
      targets:
        - host: emb-low
          mac_address: FC:34:97:9E:C8:AF
          port: 7777
          wait_seconds: 0
          retry_timeout_seconds: 8
    priority: 2
    min_level: 0
    max_level: 10
    usage: search

  - provider: infinity
    base_url: http://emb-high:7997
    model: high-model
    embedding_dimension: 512
    priority: 1
    min_level: 0
    max_level: 10
    usage: search
""",
    )

    monkeypatch.setenv("LLM_PROVIDERS_FILE", str(config_file))

    settings = EmbeddingsSettings.from_env(
        load_dotenv=False,
        intelligence_level=5,
        usage="search",
    )

    assert settings.provider == "infinity"
    assert settings.model == "high-model"
    assert settings.embedding_dimension == 512
    assert settings.infinity_url == "http://emb-high:7997"
    assert len(settings.provider_configs) == 2
    assert settings.provider_configs[0]["model"] == "high-model"
    assert settings.provider_configs[1]["wake_on_lan"]["targets"][0]["host"] == "emb-low"


def test_create_embedding_client_uses_yaml_fallback_chain(tmp_path, monkeypatch):
    config_file = _write_yaml(
        tmp_path,
        """
embedding_providers:
  - provider: infinity
    base_url: http://emb1:7997
    model: emb-model
    priority: 1

  - provider: infinity
    base_url: http://emb2:7997
    model: emb-model
    priority: 2
""",
    )

    monkeypatch.setenv("LLM_PROVIDERS_FILE", str(config_file))

    with patch("core_lib.embeddings.fallback_client.EmbeddingFactory.create") as mock_create:
        provider = Mock()
        provider.model = "emb-model"
        provider.embedding_dim = 384
        mock_create.return_value = provider

        client = create_embedding_client(intelligence_level=5, usage="search")

        assert isinstance(client, FallbackEmbeddingClient)
        assert len(client.providers) == 2


def test_reranker_settings_loads_priority_chain_from_yaml(tmp_path, monkeypatch):
    config_file = _write_yaml(
        tmp_path,
        """
reranker_providers:
  - provider: infinity
    base_url: http://rerank-low:7997
    model: rerank-low
    wake_on_lan:
      enabled: true
      targets:
        - host: rerank-low
          mac_address: FC:34:97:9E:C8:AF
          port: 7777
          wait_seconds: 0
    priority: 2
    min_level: 0
    max_level: 10

  - provider: infinity
    base_url: http://rerank-high:7997
    model: rerank-high
    priority: 1
    min_level: 0
    max_level: 10
""",
    )

    monkeypatch.setenv("LLM_PROVIDERS_FILE", str(config_file))

    settings = RerankerSettings.from_env(load_dotenv=False, intelligence_level=8)

    assert settings.provider == "infinity"
    assert settings.model == "rerank-high"
    assert settings.infinity_url == "http://rerank-high:7997"
    assert len(settings.provider_configs) == 2
    assert settings.provider_configs[0]["model"] == "rerank-high"
    assert settings.provider_configs[1]["wake_on_lan"]["targets"][0]["host"] == "rerank-low"


def test_create_reranker_client_uses_yaml_fallback_chain(tmp_path, monkeypatch):
    config_file = _write_yaml(
        tmp_path,
        """
reranker_providers:
  - provider: infinity
    base_url: http://rerank1:7997
    model: rerank-model
    priority: 1

  - provider: infinity
    base_url: http://rerank2:7997
    model: rerank-model
    priority: 2
""",
    )

    monkeypatch.setenv("LLM_PROVIDERS_FILE", str(config_file))

    with patch("core_lib.reranker.fallback_client.RerankerFactory.create") as mock_create:
        provider = Mock()
        provider.model = "rerank-model"
        mock_create.return_value = provider

        client = create_reranker_client(intelligence_level=6)

        assert isinstance(client, FallbackRerankerClient)
        assert len(client.providers) == 2
