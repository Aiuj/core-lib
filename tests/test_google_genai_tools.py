"""Tests for Gemini/OpenAI tool schema compatibility in GoogleGenAIProvider."""

from unittest.mock import patch

from google.genai import types

from core_lib.llm.providers.google_genai_provider import GoogleGenAIProvider, GeminiConfig


OPENAI_KB_TOOL = {
    "type": "function",
    "function": {
        "name": "kb_search",
        "description": "Search the KB.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
}


def _create_provider() -> GoogleGenAIProvider:
    config = GeminiConfig(api_key="test-key", model="gemini-2.5-flash")
    with patch("google.genai.Client"), patch(
        "openinference.instrumentation.google_genai.GoogleGenAIInstrumentor"
    ):
        return GoogleGenAIProvider(config)


def test_build_config_converts_openai_function_tool_to_gemini_tool():
    provider = _create_provider()

    result = provider._build_config(
        structured_output=None,
        tools=[OPENAI_KB_TOOL],
        system_message=None,
        use_search_grounding=False,
        thinking_enabled_override=None,
        cached_content=None,
    )

    cfg = result["config"]
    assert cfg.tools is not None
    assert len(cfg.tools) == 1
    assert isinstance(cfg.tools[0], types.Tool)
    assert cfg.tools[0].function_declarations[0].name == "kb_search"
    assert cfg.tools[0].function_declarations[0].parameters_json_schema["type"] == "object"


def test_build_config_keeps_native_gemini_tool():
    provider = _create_provider()
    native_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="native_tool",
                parameters_json_schema={"type": "object", "properties": {}},
            )
        ]
    )

    result = provider._build_config(
        structured_output=None,
        tools=[native_tool],
        system_message=None,
        use_search_grounding=False,
        thinking_enabled_override=None,
        cached_content=None,
    )

    cfg = result["config"]
    assert cfg.tools is not None
    assert len(cfg.tools) == 1
    assert cfg.tools[0].function_declarations[0].name == "native_tool"


def test_build_config_tools_with_grounding_remain_valid():
    provider = _create_provider()

    result = provider._build_config(
        structured_output=None,
        tools=[OPENAI_KB_TOOL],
        system_message=None,
        use_search_grounding=True,
        thinking_enabled_override=None,
        cached_content=None,
    )

    cfg = result["config"]
    assert cfg.tools is not None
    assert len(cfg.tools) >= 2
    assert all(isinstance(t, types.Tool) for t in cfg.tools)


def test_build_config_with_invalid_openai_tool_name_is_skipped():
    provider = _create_provider()
    invalid_tool = {
        "type": "function",
        "function": {
            # missing "name"
            "description": "Bad tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    result = provider._build_config(
        structured_output=None,
        tools=[invalid_tool],
        system_message=None,
        use_search_grounding=False,
        thinking_enabled_override=None,
        cached_content=None,
    )

    cfg = result["config"]
    # Invalid tool should be dropped rather than causing GenerateContentConfig validation failure.
    assert not cfg.tools
