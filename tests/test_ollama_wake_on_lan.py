"""Tests for Wake-on-LAN behavior in Ollama provider."""

import sys
import time
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


# ---------------------------------------------------------------------------
# Non-blocking warmup mode
# ---------------------------------------------------------------------------

def _make_ollama_provider(monkeypatch, wol_config: dict) -> OllamaProvider:
    """Return an OllamaProvider with a fake Ollama module in sys.modules."""
    class FakeClient:
        def __init__(self, host=None, **kwargs):
            self.host = host

        def chat(self, **payload):
            raise RuntimeError("connection refused")

    fake_module = types.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "ollama", fake_module)

    config = OllamaConfig(
        model="qwen3:1.7b",
        base_url="http://powerspec:11434",
        timeout=30,
        wake_on_lan=wol_config,
    )
    return OllamaProvider(config)


def test_ollama_is_in_warmup_false_by_default(monkeypatch):
    """is_in_warmup() must be False before any WoL has been fired."""
    provider = _make_ollama_provider(
        monkeypatch,
        {"enabled": True, "warmup_seconds": 30, "mac_address": "FC:34:97:9E:C8:AF"},
    )
    assert provider.is_in_warmup() is False


def test_ollama_is_in_warmup_true_after_wake(monkeypatch):
    """is_in_warmup() must be True immediately after a WoL packet is sent."""
    provider = _make_ollama_provider(
        monkeypatch,
        {
            "enabled": True,
            "warmup_seconds": 30,
            "targets": [{"mac_address": "FC:34:97:9E:C8:AF", "wait_seconds": 0}],
        },
    )
    monkeypatch.setattr(provider._wake_on_lan, "_send_magic_packet", lambda _t: None)

    # chat() should raise (connection refused) after firing WoL
    result = provider.chat(messages=[{"role": "user", "content": "hello"}])
    assert "error" in result  # OllamaProvider wraps exceptions in error dict

    assert provider.is_in_warmup() is True


def test_ollama_chat_reraises_in_nonblocking_warmup_mode(monkeypatch):
    """In non-blocking warmup mode chat() must NOT retry on the same host after WoL."""
    call_count = {"n": 0}

    class FakeClient:
        def __init__(self, host=None, **kwargs):
            pass

        def chat(self, **payload):
            call_count["n"] += 1
            raise RuntimeError("connection refused")

    fake_module = types.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "ollama", fake_module)

    config = OllamaConfig(
        model="qwen3:1.7b",
        base_url="http://powerspec:11434",
        timeout=30,
        wake_on_lan={
            "enabled": True,
            "warmup_seconds": 30,
            "targets": [{"mac_address": "FC:34:97:9E:C8:AF", "wait_seconds": 0}],
        },
    )
    provider = OllamaProvider(config)
    monkeypatch.setattr(provider._wake_on_lan, "_send_magic_packet", lambda _t: None)

    result = provider.chat(messages=[{"role": "user", "content": "hello"}])
    # The provider must NOT attempt a second call on the same server
    assert call_count["n"] == 1, "Expected exactly one call (no blocking retry)"
    assert "error" in result


def test_ollama_is_in_warmup_false_after_window_expires(monkeypatch):
    """is_in_warmup() must revert to False once the warmup window has elapsed."""
    provider = _make_ollama_provider(
        monkeypatch,
        {
            "enabled": True,
            "warmup_seconds": 5,
            "targets": [{"mac_address": "FC:34:97:9E:C8:AF", "wait_seconds": 0}],
        },
    )
    monkeypatch.setattr(provider._wake_on_lan, "_send_magic_packet", lambda _t: None)

    url = "http://powerspec:11434"
    provider._wake_on_lan.maybe_wake(url, RuntimeError("connection refused"))

    # Back-date the wake timestamp past the warmup window
    provider._wake_on_lan._waking_timestamps[url] = time.time() - 6

    assert provider.is_in_warmup() is False
