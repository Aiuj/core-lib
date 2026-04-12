"""Tests for core_lib.llm.providers.base.normalize_tool_calls."""

from types import SimpleNamespace

from core_lib.llm.providers.base import normalize_tool_calls


class TestNormalizeToolCalls:
    def test_empty_list(self):
        assert normalize_tool_calls([]) == []

    def test_none_returns_empty(self):
        assert normalize_tool_calls(None) == []

    def test_plain_dict_passthrough(self):
        tc = {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Paris"}',
            },
        }
        result = normalize_tool_calls([tc])
        assert len(result) == 1
        assert result[0]["id"] == "call_1"
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["arguments"] == '{"city": "Paris"}'

    def test_dict_arguments_coerced_to_json_string(self):
        """Ollama/Google-style: arguments is a dict, not a JSON string."""
        tc = {
            "function": {
                "name": "search",
                "arguments": {"query": "hello"},
            },
        }
        result = normalize_tool_calls([tc])
        assert result[0]["function"]["arguments"] == '{"query": "hello"}'
        assert result[0]["id"] == ""
        assert result[0]["type"] == "function"

    def test_sdk_object_style(self):
        """OpenAI SDK returns attribute-based objects."""
        tc = SimpleNamespace(
            id="call_abc",
            type="function",
            function=SimpleNamespace(
                name="kb_search",
                arguments='{"q": "CEO"}',
            ),
        )
        result = normalize_tool_calls([tc])
        assert result[0]["id"] == "call_abc"
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "kb_search"
        assert result[0]["function"]["arguments"] == '{"q": "CEO"}'

    def test_sdk_object_with_dict_arguments(self):
        """Edge case: SDK object where arguments is already a dict."""
        tc = SimpleNamespace(
            id="call_x",
            type="function",
            function=SimpleNamespace(
                name="tool",
                arguments={"key": "val"},
            ),
        )
        result = normalize_tool_calls([tc])
        assert result[0]["function"]["arguments"] == '{"key": "val"}'

    def test_missing_fields_get_defaults(self):
        """Minimal dict with just a function name."""
        tc = {"function": {"name": "ping"}}
        result = normalize_tool_calls([tc])
        assert result[0]["id"] == ""
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "ping"
        assert result[0]["function"]["arguments"] == "{}"

    def test_multiple_tool_calls(self):
        tcs = [
            {"id": "1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
            SimpleNamespace(id="2", type="function", function=SimpleNamespace(name="b", arguments="{}")),
        ]
        result = normalize_tool_calls(tcs)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"
