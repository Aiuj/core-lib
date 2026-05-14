"""Tests for Wake-on-LAN behavior in OpenAI provider."""

import time

import pytest

from core_lib.api_utils.wake_on_lan import WakeResult
from core_lib.llm.provider_registry import ProviderConfig
from core_lib.llm.providers.openai_provider import OpenAIConfig, OpenAIProvider


# ---------------------------------------------------------------------------
# ProviderConfig / ProviderRegistry passthrough
# ---------------------------------------------------------------------------

def test_provider_registry_passes_wol_config_to_openai():
    cfg = ProviderConfig.from_dict(
        {
            "provider": "openai",
            "model": "some-local-model",
            "host": "http://192.168.1.204:8100/v1",
            "api_key": "fake-key",
            "timeout": 120,
            "wake_on_lan": {
                "enabled": True,
                "mac_address": "FC:34:97:9E:C8:AF",
                "port": 7777,
            },
        }
    )

    llm_cfg = cfg.to_llm_config()

    assert isinstance(llm_cfg, OpenAIConfig)
    assert llm_cfg.wake_on_lan is not None
    assert llm_cfg.wake_on_lan["enabled"] is True
    assert llm_cfg.wake_on_lan["port"] == 7777
    assert llm_cfg.timeout == 120


# ---------------------------------------------------------------------------
# Helper: build an OpenAIProvider using a stub OpenAI client
# ---------------------------------------------------------------------------

class _FakeCompletion:
    """Minimal stand-in for openai.types.chat.ChatCompletion."""

    class _Choice:
        class _Message:
            content = "hello"
            tool_calls = None

        message = _Message()
        finish_reason = "stop"

    choices = [_Choice()]
    usage = None


class _FakeCompletionReasoningToolCall:
    """Completion payload where vLLM places a tool call in message.reasoning."""

    class _Choice:
        class _Message:
            content = None
            tool_calls = None
            reasoning = (
                "<tool_call>\n"
                '{"name": "search_kb", "arguments": {"query": "largest wind project"}}\n'
                "</tool_call>"
            )

        message = _Message()
        finish_reason = "stop"

    choices = [_Choice()]
    usage = None


class _FakeOpenAIClient:
    """Stub for openai.OpenAI whose chat.completions.create() we control."""

    def __init__(self):
        self.calls = []  # list of kwargs passed to create()

        class _Completions:
            pass

        completions = _Completions()
        completions.create = self._create

        class _Chat:
            pass

        chat = _Chat()
        chat.completions = completions
        self.chat = chat

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        raise RuntimeError("connection refused")


class _FakeOpenAIClientSuccess:
    """Stub that always returns a successful completion."""

    def __init__(self):
        self.calls = []
        self.call_count = 0

        class _Completions:
            pass

        completions = _Completions()
        completions.create = self._create

        class _Chat:
            pass

        chat = _Chat()
        chat.completions = completions
        self.chat = chat

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeCompletion()


def _make_openai_provider(monkeypatch, wol_config: dict) -> OpenAIProvider:
    """Build an OpenAIProvider with a stubbed openai module and given WoL config."""
    import types
    import sys

    fake_openai = types.SimpleNamespace(
        OpenAI=lambda **kw: _FakeOpenAIClient(),
        AzureOpenAI=lambda **kw: _FakeOpenAIClient(),
        APITimeoutError=TimeoutError,
        APIConnectionError=ConnectionError,
        RateLimitError=type("RateLimitError", (Exception,), {}),
        NotFoundError=type("NotFoundError", (Exception,), {}),
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
        APIStatusError=type("APIStatusError", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    config = OpenAIConfig(
        api_key="fake-key",
        model="local-model",
        base_url="http://powerspec:8100/v1",
        timeout=30,
        wake_on_lan=wol_config,
    )
    return OpenAIProvider(config)


# ---------------------------------------------------------------------------
# is_in_warmup() state machine
# ---------------------------------------------------------------------------

def test_openai_is_in_warmup_false_by_default(monkeypatch):
    """is_in_warmup() must be False before any WoL has been fired."""
    provider = _make_openai_provider(
        monkeypatch,
        {"enabled": True, "warmup_seconds": 30, "mac_address": "FC:34:97:9E:C8:AF"},
    )
    assert provider.is_in_warmup() is False


def test_openai_is_in_warmup_true_after_wake(monkeypatch):
    """is_in_warmup() must be True immediately after a WoL packet is sent."""
    provider = _make_openai_provider(
        monkeypatch,
        {
            "enabled": True,
            "warmup_seconds": 30,
            "targets": [{"mac_address": "FC:34:97:9E:C8:AF", "wait_seconds": 0}],
        },
    )
    monkeypatch.setattr(provider._wake_on_lan, "_send_magic_packet", lambda _t: None)

    # Non-blocking WoL re-raises so FallbackLLMClient can route to secondary
    with pytest.raises(RuntimeError, match="connection refused"):
        provider.chat(messages=[{"role": "user", "content": "hello"}])

    assert provider.is_in_warmup() is True


def test_openai_is_in_warmup_false_after_window_expires(monkeypatch):
    """is_in_warmup() must revert to False once the warmup window has elapsed."""
    provider = _make_openai_provider(
        monkeypatch,
        {
            "enabled": True,
            "warmup_seconds": 5,
            "targets": [{"mac_address": "FC:34:97:9E:C8:AF", "wait_seconds": 0}],
        },
    )
    monkeypatch.setattr(provider._wake_on_lan, "_send_magic_packet", lambda _t: None)

    url = "http://powerspec:8100/v1"
    provider._wake_on_lan.maybe_wake(url, RuntimeError("connection refused"))

    # Back-date the wake timestamp past the warmup window
    provider._wake_on_lan._waking_timestamps[url] = time.time() - 6

    assert provider.is_in_warmup() is False


# ---------------------------------------------------------------------------
# Blocking retry after WoL
# ---------------------------------------------------------------------------

def test_openai_provider_retries_after_wol(monkeypatch):
    """Provider must retry the API call after a successful blocking WoL wake."""
    import types
    import sys

    call_count = 0
    captured_timeouts = []

    class _Completions:
        def create(self, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_timeouts.append(kwargs.get("timeout"))
            if call_count == 1:
                raise TimeoutError("connection timed out")
            return _FakeCompletion()

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        chat = _Chat()

    fake_openai = types.SimpleNamespace(
        OpenAI=lambda **kw: _FakeClient(),
        AzureOpenAI=lambda **kw: _FakeClient(),
        APITimeoutError=TimeoutError,
        APIConnectionError=ConnectionError,
        RateLimitError=type("RateLimitError", (Exception,), {}),
        NotFoundError=type("NotFoundError", (Exception,), {}),
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
        APIStatusError=type("APIStatusError", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    config = OpenAIConfig(
        api_key="fake-key",
        model="local-model",
        base_url="http://powerspec:8100/v1",
        timeout=30,
        wake_on_lan={"enabled": True, "mac_address": "FC:34:97:9E:C8:AF"},
    )
    provider = OpenAIProvider(config)

    class _WakeStub:
        def maybe_get_initial_timeout(self, url, default_timeout):
            return 2.0  # short initial probe timeout

        def maybe_wake(self, url, error):
            return WakeResult(succeeded=True, warmup_seconds=None, retry_timeout_seconds=8)

        def is_in_warmup(self, url):
            return False

    monkeypatch.setattr(provider, "_wake_on_lan", _WakeStub())

    result = provider.chat(messages=[{"role": "user", "content": "hello"}])

    assert result["content"] == "hello"
    assert call_count == 2
    assert captured_timeouts == [2.0, 8]


# ---------------------------------------------------------------------------
# Non-blocking mode: re-raises so FallbackLLMClient can route to secondary
# ---------------------------------------------------------------------------

def test_openai_provider_reraises_in_nonblocking_wol_mode(monkeypatch):
    """Non-blocking WoL should re-raise cleanly past the outer except block."""
    import types
    import sys

    class _Completions:
        def create(self, **kwargs):
            raise TimeoutError("connection timed out")

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        chat = _Chat()

    fake_openai = types.SimpleNamespace(
        OpenAI=lambda **kw: _FakeClient(),
        AzureOpenAI=lambda **kw: _FakeClient(),
        APITimeoutError=TimeoutError,
        APIConnectionError=ConnectionError,
        RateLimitError=type("RateLimitError", (Exception,), {}),
        NotFoundError=type("NotFoundError", (Exception,), {}),
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
        APIStatusError=type("APIStatusError", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    config = OpenAIConfig(
        api_key="fake-key",
        model="local-model",
        base_url="http://powerspec:8100/v1",
        timeout=30,
        wake_on_lan={
            "enabled": True,
            "warmup_seconds": 30,
            "targets": [{"mac_address": "FC:34:97:9E:C8:AF", "wait_seconds": 0}],
        },
    )
    provider = OpenAIProvider(config)

    class _WakeStubNonBlocking:
        def maybe_get_initial_timeout(self, url, default_timeout):
            return 2.0

        def maybe_wake(self, url, error):
            return WakeResult(succeeded=True, warmup_seconds=30, retry_timeout_seconds=None)

        def is_in_warmup(self, url):
            return True

    monkeypatch.setattr(provider, "_wake_on_lan", _WakeStubNonBlocking())

    # The call must raise — NOT return an error dict — so FallbackLLMClient
    # can route to a secondary provider cleanly (no spurious warning log).
    with pytest.raises(TimeoutError):
        provider.chat(messages=[{"role": "user", "content": "hello"}])


# ---------------------------------------------------------------------------
# No WoL configured — normal path unchanged
# ---------------------------------------------------------------------------

def test_openai_provider_without_wol_no_retry(monkeypatch):
    """Without WoL config, a connection error must be raised (not swallowed)."""
    import types
    import sys

    class _Completions:
        def create(self, **kwargs):
            raise TimeoutError("connection timed out")

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        chat = _Chat()

    fake_openai = types.SimpleNamespace(
        OpenAI=lambda **kw: _FakeClient(),
        AzureOpenAI=lambda **kw: _FakeClient(),
        APITimeoutError=TimeoutError,
        APIConnectionError=ConnectionError,
        RateLimitError=type("RateLimitError", (Exception,), {}),
        NotFoundError=type("NotFoundError", (Exception,), {}),
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
        APIStatusError=type("APIStatusError", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    config = OpenAIConfig(
        api_key="fake-key",
        model="gpt-4o-mini",
        timeout=30,
        # no wake_on_lan
    )
    provider = OpenAIProvider(config)

    # With no WoL configured, classifiable errors (timeout, connection) are
    # re-raised so FallbackLLMClient can classify and route to the next provider.
    with pytest.raises(TimeoutError):
        provider.chat(messages=[{"role": "user", "content": "hello"}])


def test_openai_provider_rate_limit_raises(monkeypatch):
    """RateLimitError must be re-raised directly so FallbackLLMClient handles it.

    The provider must NOT swallow it into an error dict. This avoids double-logging
    and ensures FallbackLLMClient can classify it and mark the provider unhealthy.
    """
    import types
    import sys

    RateLimitError = type("RateLimitError", (Exception,), {})

    class _Completions:
        def create(self, **kwargs):
            raise RateLimitError("429 rate limited")

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        chat = _Chat()

    fake_openai = types.SimpleNamespace(
        OpenAI=lambda **kw: _FakeClient(),
        AzureOpenAI=lambda **kw: _FakeClient(),
        APITimeoutError=type("APITimeoutError", (Exception,), {}),
        APIConnectionError=type("APIConnectionError", (Exception,), {}),
        NotFoundError=type("NotFoundError", (Exception,), {}),
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
        APIStatusError=type("APIStatusError", (Exception,), {}),
        RateLimitError=RateLimitError,
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    config = OpenAIConfig(
        api_key="fake-key",
        model="openrouter/model",
        base_url="https://openrouter.ai/api/v1",
        timeout=30,
    )
    provider = OpenAIProvider(config)

    # RateLimitError must propagate so FallbackLLMClient can route to the next provider.
    with pytest.raises(RateLimitError):
        provider.chat(messages=[{"role": "user", "content": "hello"}])


def test_openai_provider_parses_tool_call_from_reasoning(monkeypatch):
    """Recover tool calls when vLLM returns them in message.reasoning with empty content."""
    import types
    import sys

    class _Completions:
        def create(self, **kwargs):
            return _FakeCompletionReasoningToolCall()

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        chat = _Chat()

    fake_openai = types.SimpleNamespace(
        OpenAI=lambda **kw: _FakeClient(),
        AzureOpenAI=lambda **kw: _FakeClient(),
        APITimeoutError=type("APITimeoutError", (Exception,), {}),
        APIConnectionError=type("APIConnectionError", (Exception,), {}),
        RateLimitError=type("RateLimitError", (Exception,), {}),
        NotFoundError=type("NotFoundError", (Exception,), {}),
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
        APIStatusError=type("APIStatusError", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    provider = OpenAIProvider(
        OpenAIConfig(
            api_key="fake-key",
            model="Qwen/Qwen3-4B-FP8",
            base_url="http://localhost:8101/v1",
            thinking_enabled=False,
            thinking_config={"enabled": False, "level": "off"},
            timeout=30,
        )
    )

    result = provider.chat(
        messages=[{"role": "user", "content": "find largest wind project"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search_kb",
                    "description": "Search the KB",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
    )

    assert result["content"] == ""
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "search_kb"
