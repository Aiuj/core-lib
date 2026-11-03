from dataclasses import replace
from unittest.mock import MagicMock

from core_lib.llm.fallback_client import FallbackLLMClient
from core_lib.llm.provider_registry import ProviderConfig, ProviderRegistry


def test_registry_keeps_same_model_variants_distinct_across_access_paths(monkeypatch):
    off = ProviderConfig(provider="openai", model="same-model", api_key="secret", priority=1)
    on = replace(off, thinking_enabled=True, thinking_config={"enabled": True})
    made = []

    def create(config):
        client = MagicMock()
        client.configured_thinking = config.thinking_enabled
        made.append(client)
        return client

    monkeypatch.setattr(ProviderConfig, "to_client", create)
    registry = ProviderRegistry([off, on])
    tracker = MagicMock()
    tracker.filter_healthy.side_effect = lambda providers: providers
    monkeypatch.setattr(registry, "_get_health_tracker", lambda: tracker)

    first, second = registry.get_client(0), registry.get_client(1)
    assert first is not second
    assert (first.configured_thinking, second.configured_thinking) == (False, True)
    assert [c for c, _, _ in registry.iter_clients_for_level(5)] == [first, second]
    assert [c for c, _, _ in registry.iter_clients_with_fallback()] == [first, second]
    assert registry.get_client_for_level(5) is first
    assert registry.get_healthy_client() is first
    off.priority = 2
    assert registry.get_client(0).configured_thinking is True
    registry.clear_cache()
    assert all(client.close.call_count == 1 for client in made)


def test_fallback_cache_distinguishes_thinking_credentials_and_bounded_copies(monkeypatch):
    off = ProviderConfig(provider="openai", model="same-model", api_key="secret")
    on = replace(off, thinking_enabled=True, thinking_config={"enabled": True, "level": "medium"})
    monkeypatch.setattr(ProviderConfig, "to_client", lambda config: MagicMock(configured_thinking=config.thinking_enabled))
    fallback = FallbackLLMClient(ProviderRegistry([off, on]))
    first, second = fallback._get_client(off), fallback._get_client(on)
    assert first is not second
    assert (first.configured_thinking, second.configured_thinking) == (False, True)
    assert fallback._get_client(replace(on)) is second
    assert fallback._get_client(replace(on, max_tokens=100)) is not second
    assert fallback._get_client(replace(on, api_key="other-secret")) is not second
    assert "secret" not in fallback._build_cache_key(on)
    fallback.close()
