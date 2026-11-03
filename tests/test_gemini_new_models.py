from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core_lib.llm.providers.google_genai_provider import GeminiConfig, GoogleGenAIProvider
from core_lib.tracing.service_pricing import get_llm_pricing
from core_lib.tracing.service_usage import calculate_llm_cost


@pytest.mark.parametrize("model,input_rate,output_rate", [
    ("gemini-2.5-flash", 0.0003, 0.0025),
    ("gemini-3.1-flash-lite", 0.00025, 0.0015),
    ("gemini-3.5-flash-lite", 0.0003, 0.0025),
    ("gemini-3.8-flash", 0.00075, 0.00375),
])
def test_vertex_global_text_prices(model, input_rate, output_rate):
    assert get_llm_pricing(model) == {"input": input_rate, "output": output_rate}
    assert calculate_llm_cost("vertex", model, 1000, 1000) == pytest.approx(input_rate + output_rate)


@pytest.mark.parametrize("model,expected", [
    ("gemini-3.1-flash-lite", "MINIMAL"),
    ("gemini-3.5-flash-lite", "MINIMAL"),
    ("gemini-3.8-flash", "LOW"),
])
def test_gemini_3_disable_request_uses_lowest_supported_level(model, expected):
    config = GeminiConfig(api_key="test", model=model, thinking_config={"enabled": False})
    with patch("google.genai.Client"), patch("openinference.instrumentation.google_genai.GoogleGenAIInstrumentor"):
        provider = GoogleGenAIProvider(config)
    meta = {}
    cfg = provider._build_config(structured_output=None, tools=None, system_message=None,
                                 use_search_grounding=False, thinking_enabled_override=None,
                                 cached_content=None, thinking_meta=meta)["config"]
    assert cfg.thinking_config.thinking_level.value == expected
    assert cfg.thinking_config.thinking_budget is None
    assert meta == {"enabled": True, "level": expected.lower()}


def test_google_usage_includes_billed_thinking_tokens():
    config = GeminiConfig(api_key="test", model="gemini-3.8-flash")
    with patch("google.genai.Client") as client_class, patch("openinference.instrumentation.google_genai.GoogleGenAIInstrumentor"), patch("core_lib.llm.providers.google_genai_provider.log_llm_usage") as log_usage:
        provider = GoogleGenAIProvider(config)
        part = SimpleNamespace(text="answer", thought=False)
        response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))], function_calls=None, parsed=None,
                                   usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5, thoughts_token_count=7, total_token_count=22))
        client_class.return_value.models.generate_content.return_value = response
        result = provider.chat(messages=[{"role": "user", "content": "test"}])
    assert result["usage"]["completion_tokens"] == 12
    assert result["usage"]["thinking_tokens"] == 7
    assert log_usage.call_args.kwargs["output_tokens"] == 12


def test_gemini_3_explicit_medium_uses_level_not_budget():
    config = GeminiConfig(api_key="test", model="gemini-3.5-flash-lite", thinking_enabled=True,
                          thinking_config={"enabled": True, "level": "medium"})
    with patch("google.genai.Client"), patch("openinference.instrumentation.google_genai.GoogleGenAIInstrumentor"):
        provider = GoogleGenAIProvider(config)
    cfg = provider._build_config(structured_output=None, tools=None, system_message=None,
                                 use_search_grounding=False, thinking_enabled_override=None,
                                 cached_content=None)["config"]
    assert cfg.thinking_config.thinking_level.value == "MEDIUM"
    assert cfg.thinking_config.thinking_budget is None


def test_gemini_3_thinking_enabled_defaults_to_medium():
    config = GeminiConfig(api_key="test", model="gemini-3.5-flash-lite", thinking_enabled=True,
                          thinking_config={"enabled": True})
    with patch("google.genai.Client"), patch("openinference.instrumentation.google_genai.GoogleGenAIInstrumentor"):
        provider = GoogleGenAIProvider(config)
    cfg = provider._build_config(structured_output=None, tools=None, system_message=None,
                                 use_search_grounding=False, thinking_enabled_override=None,
                                 cached_content=None)["config"]
    assert cfg.thinking_config.thinking_level.value == "MEDIUM"


def test_gemini_38_rejects_unsupported_minimal_level():
    config = GeminiConfig(api_key="test", model="gemini-3.8-flash", thinking_enabled=True,
                          thinking_config={"enabled": True, "level": "minimal"})
    with patch("google.genai.Client"), patch("openinference.instrumentation.google_genai.GoogleGenAIInstrumentor"):
        provider = GoogleGenAIProvider(config)
    with pytest.raises(ValueError, match="Unsupported thinking level"):
        provider._build_config(structured_output=None, tools=None, system_message=None,
                               use_search_grounding=False, thinking_enabled_override=None,
                               cached_content=None)
