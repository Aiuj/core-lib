"""Tests for Wake-on-LAN behavior in Ollama provider."""

import sys
import types

from core_lib.api_utils.wake_on_lan import WakeResult
from core_lib.llm.provider_registry import ProviderConfig
from core_lib.llm.providers.ollama_provider import OllamaConfig, OllamaProvider


def test_provider_registry_passes_wol_config_to_ollama():
    cfg = ProviderConfig.from_dict(
        {
            "provider": "ollama",
            "model": "qwen3:1.7b",
            "host": "http://powerspec:11434",
            "wake_on_lan": {
                "enabled": True,
                "mac_address": "FC:34:97:9E:C8:AF",
                "port": 7777,
            },
        }
    )

    llm_cfg = cfg.to_llm_config()

    assert isinstance(llm_cfg, OllamaConfig)
    assert llm_cfg.wake_on_lan is not None
    assert llm_cfg.wake_on_lan["enabled"] is True
    assert llm_cfg.wake_on_lan["port"] == 7777


def test_ollama_provider_retries_after_wol(monkeypatch):
    call_timeouts = []

    class FakeClient:
        _call_count = 0

        def __init__(self, host=None, **kwargs):
            self.host = host
            self.kwargs = kwargs

        def chat(self, **payload):
            call_timeouts.append(self.kwargs.get("timeout"))
            FakeClient._call_count += 1
            if FakeClient._call_count == 1:
                raise RuntimeError("connection refused")
            return {
                "message": {"content": "ok"},
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    fake_module = types.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "ollama", fake_module)

    config = OllamaConfig(
        model="qwen3:1.7b",
        base_url="http://powerspec:11434",
        timeout=30,
        wake_on_lan={"enabled": True, "mac_address": "FC:34:97:9E:C8:AF"},
    )
    provider = OllamaProvider(config)

    class _WakeStub:
        def maybe_get_initial_timeout(self, _base_url, _default_timeout):
            return 2

        def maybe_wake(self, _base_url, _error):
            return WakeResult(attempted=True, succeeded=True, retry_timeout_seconds=8)

    monkeypatch.setattr(provider, "_wake_on_lan", _WakeStub())

    result = provider.chat(messages=[{"role": "user", "content": "hello"}])

    assert result["content"] == "ok"
    assert call_timeouts == [2, 8]
