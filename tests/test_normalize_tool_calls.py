"""Tests for core_lib.llm.providers.base.normalize_tool_calls."""

import json
from types import SimpleNamespace

from core_lib.llm.providers.base import normalize_tool_calls, parse_text_tool_calls


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


class TestParseTextToolCalls:
    """Tests for XML-style tool call parsing from content text."""

    def test_empty_string(self):
        calls, remaining = parse_text_tool_calls("")
        assert calls == []
        assert remaining == ""

    def test_none_input(self):
        calls, remaining = parse_text_tool_calls(None)
        assert calls == []
        assert remaining is None

    def test_no_tool_calls(self):
        text = "Here is a normal response with no tool calls."
        calls, remaining = parse_text_tool_calls(text)
        assert calls == []
        assert remaining == text

    def test_single_tool_call(self):
        text = (
            "<tool_call>\n"
            "<function=kb_search>\n"
            "<parameter=query>When was the company created?</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls, remaining = parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["type"] == "function"
        assert calls[0]["function"]["name"] == "kb_search"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["query"] == "When was the company created?"
        assert calls[0]["id"].startswith("call_")
        assert remaining == ""

    def test_tool_call_with_surrounding_text(self):
        text = (
            "Let me search for that.\n"
            "<tool_call>\n"
            "<function=kb_search>\n"
            "<parameter=query>CEO name</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "I'll get back to you."
        )
        calls, remaining = parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "kb_search"
        assert "Let me search" in remaining
        assert "I'll get back" in remaining
        assert "<tool_call>" not in remaining

    def test_multiple_parameters(self):
        text = (
            "<tool_call>\n"
            "<function=search>\n"
            "<parameter=query>test query</parameter>\n"
            "<parameter=limit>10</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls, remaining = parse_text_tool_calls(text)
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["query"] == "test query"
        assert args["limit"] == "10"

    def test_multiple_tool_calls(self):
        text = (
            "<tool_call>\n"
            "<function=search>\n"
            "<parameter=query>first</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=search>\n"
            "<parameter=query>second</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls, remaining = parse_text_tool_calls(text)
        assert len(calls) == 2
        args0 = json.loads(calls[0]["function"]["arguments"])
        args1 = json.loads(calls[1]["function"]["arguments"])
        assert args0["query"] == "first"
        assert args1["query"] == "second"
        # Each gets a unique ID
        assert calls[0]["id"] != calls[1]["id"]

    def test_json_style_tool_call_block(self):
        text = (
            "<tool_call>\n"
            '{"name": "search_kb", "arguments": {"query": "largest wind project"}}\n'
            "</tool_call>"
        )
        calls, remaining = parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "search_kb"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["query"] == "largest wind project"
        assert remaining == ""

    def test_json_style_tool_call_malformed_payload_is_ignored(self):
        text = (
            "prefix\n"
            "<tool_call>\n"
            '{"name": "search_kb", "arguments": {"query": "oops"}\n'
            "</tool_call>\n"
            "suffix"
        )
        calls, remaining = parse_text_tool_calls(text)
        assert calls == []
        assert "prefix" in remaining
        assert "suffix" in remaining
