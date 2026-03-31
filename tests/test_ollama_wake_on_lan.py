"""Tests for Wake-on-LAN behavior in Ollama provider."""

import logging
import sys
import time
import types

import pytest

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


def test_ollama_model_not_found_is_handled_without_exception_log(monkeypatch, caplog):
    class FakeResponseError(Exception):
        def __init__(self, text: str, status_code: int):
            super().__init__(text)
            self.status_code = status_code

    class FakeClient:
        def __init__(self, host=None, **kwargs):
            self.host = host

        def chat(self, **payload):
            raise FakeResponseError("model 'qwen3.5:8b' not found (status code: 404)", 404)

    fake_module = types.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "ollama", fake_module)

    provider = OllamaProvider(
        OllamaConfig(
            model="qwen3.5:8b",
            base_url="http://localhost:11434",
            timeout=30,
        )
    )

    with caplog.at_level(logging.WARNING):
        result = provider.chat(messages=[{"role": "user", "content": "hello"}])

    assert result["error_code"] == "model_not_found"
    assert "model not available (handled)" in caplog.text
    assert "ollama.chat failed" not in caplog.text
    assert "Traceback" not in caplog.text


# ---------------------------------------------------------------------------
# Cloud API key authentication
# ---------------------------------------------------------------------------


def test_ollama_cloud_api_key_sent_as_bearer_header(monkeypatch):
    """When api_key is set, the Authorization header must carry a Bearer token."""
    captured_kwargs: dict = {}

    class FakeClient:
        def __init__(self, host=None, **kwargs):
            captured_kwargs.update({"host": host, **kwargs})

        def chat(self, **payload):
            return {
                "message": {"content": "hello from the cloud"},
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            }

    fake_module = types.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "ollama", fake_module)

    provider = OllamaProvider(
        OllamaConfig(
            model="qwen3.5:2b",
            base_url="https://api.ollama.com",
            api_key="ollama-sk-test-1234",
            timeout=30,
        )
    )

    result = provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert result["content"] == "hello from the cloud"
    assert captured_kwargs.get("host") == "https://api.ollama.com"
    headers = captured_kwargs.get("headers", {})
    assert headers.get("Authorization") == "Bearer ollama-sk-test-1234"


def test_ollama_no_api_key_omits_auth_header(monkeypatch):
    """When api_key is not set, no Authorization header should be added."""
    captured_kwargs: dict = {}

    class FakeClient:
        def __init__(self, host=None, **kwargs):
            captured_kwargs.update({"host": host, **kwargs})

        def chat(self, **payload):
            return {
                "message": {"content": "local response"},
                "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
            }

    fake_module = types.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "ollama", fake_module)

    provider = OllamaProvider(
        OllamaConfig(
            model="qwen3:1.7b",
            base_url="http://localhost:11434",
            timeout=30,
        )
    )

    provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert "headers" not in captured_kwargs


def test_provider_registry_passes_api_key_to_ollama_config(monkeypatch):
    """ProviderConfig.to_llm_config() must propagate api_key into OllamaConfig."""
    from core_lib.llm.provider_registry import ProviderConfig

    cfg = ProviderConfig.from_dict(
        {
            "provider": "ollama",
            "model": "qwen3.5:2b",
            "host": "https://api.ollama.com",
            "api_key": "sk-cloud-key",
        }
    )

    llm_cfg = cfg.to_llm_config()

    assert isinstance(llm_cfg, OllamaConfig)
    assert llm_cfg.api_key == "sk-cloud-key"
    assert llm_cfg.base_url == "https://api.ollama.com"


def test_ollama_config_from_env_reads_api_key(monkeypatch):
    """OllamaConfig.from_env() must pick up OLLAMA_API_KEY."""
    monkeypatch.setenv("OLLAMA_API_KEY", "env-api-key-xyz")
    monkeypatch.setenv("OLLAMA_HOST", "https://api.ollama.com")

    cfg = OllamaConfig.from_env()

    assert cfg.api_key == "env-api-key-xyz"
    assert "ollama.com" in cfg.base_url


# ---------------------------------------------------------------------------
# Thinking suppression for non-hinted models (e.g. ministral-3)
# ---------------------------------------------------------------------------


def test_think_false_sent_for_non_hinted_model_with_thinking_disabled(monkeypatch):
    """think:false must be sent even when the model is not in _THINKING_MODEL_HINTS.

    Regression test: ministral-3 is not in the hint list but supports thinking.
    If the user configures thinking: {enabled: false, level: off} and we omit
    think:false from the payload, Ollama may enable thinking by default, causing
    unexpectedly high token usage during health checks.
    """
    captured_payload: dict = {}

    class FakeClient:
        def __init__(self, host=None, **kwargs):
            pass

        def chat(self, **payload):
            captured_payload.update(payload)
            return {
                "message": {"content": "OK"},
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            }

    fake_module = types.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "ollama", fake_module)

    provider = OllamaProvider(
        OllamaConfig(
            model="ministral-3",
            base_url="http://127.0.0.1:11434",
            timeout=30,
            thinking_config={"enabled": False, "level": "off"},
        )
    )

    # Verify the model is not in the thinking hints list
    assert not provider._supports_thinking()

    provider.chat(messages=[{"role": "user", "content": "Reply with OK."}])

    # think:false must be explicitly sent so Ollama suppresses thinking
    assert captured_payload.get("think") is False


def test_think_true_not_sent_for_non_hinted_model(monkeypatch):
    """think:true must NOT be sent for models not in _THINKING_MODEL_HINTS."""
    captured_payload: dict = {}

    class FakeClient:
        def __init__(self, host=None, **kwargs):
            pass

        def chat(self, **payload):
            captured_payload.update(payload)
            return {
                "message": {"content": "OK"},
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            }

    fake_module = types.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "ollama", fake_module)

    provider = OllamaProvider(
        OllamaConfig(
            model="ministral-3",
            base_url="http://127.0.0.1:11434",
            timeout=30,
            thinking_enabled=True,
        )
    )

    provider.chat(messages=[{"role": "user", "content": "hello"}])

    # Should NOT propagate think:true for an unknown model
    assert "think" not in captured_payload


# ---------------------------------------------------------------------------
# Multimodal / vision message conversion
# ---------------------------------------------------------------------------

def test_convert_messages_plain_text_unchanged():
    """Plain text messages must pass through without modification."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]
    result = OllamaProvider._convert_messages_to_ollama_format(messages)
    assert result == messages


def test_convert_messages_multimodal_extracts_images_and_text():
    """OpenAI-style multimodal messages are converted to Ollama format."""
    import base64

    raw_b64 = base64.b64encode(b"fake-image-bytes").decode("ascii")
    data_url = f"data:image/png;base64,{raw_b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    result = OllamaProvider._convert_messages_to_ollama_format(messages)

    assert len(result) == 1
    msg = result[0]
    assert msg["role"] == "user"
    assert msg["content"] == "Describe this image."
    assert "images" in msg
    assert msg["images"] == [raw_b64]


def test_convert_messages_multimodal_multiple_text_parts_joined():
    """Multiple text parts are joined with a space."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Part one."},
                {"type": "text", "text": "Part two."},
            ],
        }
    ]

    result = OllamaProvider._convert_messages_to_ollama_format(messages)

    assert result[0]["content"] == "Part one. Part two."
    assert "images" not in result[0]


def test_convert_messages_mixed_plain_and_multimodal():
    """Only multimodal messages are converted; plain-text ones are left alone."""
    import base64

    raw_b64 = base64.b64encode(b"img").decode("ascii")
    data_url = f"data:image/jpeg;base64,{raw_b64}"

    messages = [
        {"role": "system", "content": "Be concise."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "What is this?"},
            ],
        },
    ]

    result = OllamaProvider._convert_messages_to_ollama_format(messages)

    assert result[0] == {"role": "system", "content": "Be concise."}
    assert result[1]["content"] == "What is this?"
    assert result[1]["images"] == [raw_b64]
