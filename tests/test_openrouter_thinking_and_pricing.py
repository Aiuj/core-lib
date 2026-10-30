"""OpenRouter honours the YAML thinking setting, and GPT-6 Luna is priced."""
from core_lib.llm.provider_registry import ProviderConfig
from core_lib.llm.providers.openai_provider import OpenAIProvider
from core_lib.tracing.service_pricing import get_llm_pricing


def _provider(thinking):
    config = ProviderConfig.from_dict({
        "provider": "openrouter", "model": "openai/gpt-6-luna", "api_key": "sk-test", "thinking": thinking,
    }).to_llm_config()
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.config = config
    return provider


def test_openrouter_config_carries_the_thinking_setting():
    config = _provider(False).config

    assert config.thinking_enabled is False
    assert config.thinking_config == {"enabled": False}
    assert config.is_openrouter and not config.is_local_compatible


def test_thinking_false_sends_reasoning_effort_none():
    create_kwargs: dict = {"messages": []}
    _provider(False)._apply_thinking_mode(create_kwargs, use_thinking=False)

    assert create_kwargs["reasoning_effort"] == "none"


def test_thinking_level_is_forwarded():
    create_kwargs: dict = {"messages": []}
    _provider({"level": "low"})._apply_thinking_mode(create_kwargs, use_thinking=True)

    assert create_kwargs["reasoning_effort"] == "low"


def test_chat_returns_usage_as_a_plain_dict(monkeypatch):
    """The SDK's CompletionUsage object broke callers reading usage.get(...)."""
    import sys
    import types
    from openai.types.completion_usage import CompletionUsage, CompletionTokensDetails
    from core_lib.llm.providers.openai_provider import OpenAIConfig

    completion = types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="hello", tool_calls=None), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=36, completion_tokens=56, total_tokens=92,
                              completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0)),
    )
    client = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=lambda **kwargs: completion)))
    real_openai = sys.modules["openai"]
    stub = {name: getattr(real_openai, name) for name in dir(real_openai) if not name.startswith("_")}
    stub["OpenAI"] = lambda **kwargs: client
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(**stub))
    provider = OpenAIProvider(OpenAIConfig(api_key="sk-test", model="openai/gpt-6-luna",
                                           base_url="https://openrouter.ai/api/v1"))

    usage = provider.chat(messages=[{"role": "user", "content": "hi"}])["usage"]

    assert isinstance(usage, dict)
    assert usage.get("prompt_tokens") == 36 and usage.get("completion_tokens") == 56
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 0


def test_gpt_6_luna_pricing_under_both_names():
    for name in ("gpt-6-luna", "openai/gpt-6-luna", "OpenAI/GPT-6-Luna"):
        assert get_llm_pricing(name) == {"input": 0.0001, "output": 0.0005}
