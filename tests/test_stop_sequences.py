"""Per-request stop sequences and closing of a JSON object ended by one.

A local model under schema-constrained decoding can emit whitespace where the
closing brace belongs until it reaches its output limit. A caller stops on a
whitespace-only line instead; the output then ends between JSON tokens and is
closed before validation.
"""

import json
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from core_lib.llm.fallback_client import FallbackLLMClient
from core_lib.llm.json_parser import close_truncated_json
from core_lib.llm.provider_registry import ProviderConfig, ProviderRegistry
from core_lib.llm.providers.openai_provider import OpenAIConfig, OpenAIProvider


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"a": "x", "b": []', {"a": "x", "b": []}),
        ('{"a": "x", "b": [],', {"a": "x", "b": []}),
        ('{"a": "x", "b"', {"a": "x"}),
        ('{"a": "x", "b": ', {"a": "x"}),
        ('{"a": {"b": [1, {"c": "d"}', {"a": {"b": [1, {"c": "d"}]}}),
        ('{"a": "brace } and \\" quote", "b": true', {"a": 'brace } and " quote', "b": True}),
        ('{"a": "x"}', {"a": "x"}),
        ("[1, 2", [1, 2]),
    ],
)
def test_close_truncated_json(text, expected):
    assert json.loads(close_truncated_json(text)) == expected


@pytest.mark.parametrize("text", ['{"a": "unterminated', "plain text", "", '{"a": 1}}'])
def test_close_truncated_json_refuses_what_it_cannot_close(text):
    assert close_truncated_json(text) is None


def test_stop_sequences_reach_a_bounded_copy_of_the_provider_config():
    registry = ProviderRegistry()
    original = ProviderConfig(provider="openai", model="m", api_key="k")
    registry.add(original)
    client = FallbackLLMClient(registry)
    configs = []
    llm = MagicMock()
    llm.chat.return_value = {"content": "ok", "usage": {}, "tool_calls": [], "structured": False}

    def get_client(config):
        configs.append(config)
        return llm

    with patch.object(client, "_get_client", side_effect=get_client):
        client.chat("x", stop_sequences=["\n\n", " \n"])
        stopped = configs[-1]
        client.chat("x")
    assert stopped.stop_sequences == ("\n\n", " \n")
    assert configs[-1].stop_sequences is None
    assert original.stop_sequences is None
    assert stopped.to_llm_config().stop_sequences == ("\n\n", " \n")
    assert client._build_cache_key(stopped) != client._build_cache_key(original)


class Answer(BaseModel):
    answer: str
    claims: List[str] = []
    coverage: Optional[List[str]] = None


def _completion(content):
    message = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _provider(stop):
    provider = OpenAIProvider(
        OpenAIConfig(api_key="k", model="m", base_url="http://127.0.0.1:1/v1", stop_sequences=stop)
    )
    provider._client = MagicMock()
    return provider


def test_openai_sends_stop_and_closes_the_stopped_object():
    provider = _provider(("\n\n", " \n"))
    provider._client.chat.completions.create.return_value = _completion(
        '{"answer": "Paris", "claims": ["Paris is the capital"]'
    )
    result = provider.chat(messages=[{"role": "user", "content": "q"}], structured_output=Answer)
    kwargs = provider._client.chat.completions.create.call_args.kwargs
    assert kwargs["stop"] == ["\n\n", " \n"]
    assert result["structured"] is True
    assert result["content"]["claims"] == ["Paris is the capital"]


def test_without_stop_sequences_nothing_is_sent_or_closed():
    provider = _provider(None)
    provider._client.chat.completions.create.return_value = _completion('{"answer": "Paris", "claims": [')
    result = provider.chat(messages=[{"role": "user", "content": "q"}], structured_output=Answer)
    assert "stop" not in provider._client.chat.completions.create.call_args.kwargs
    assert result["structured"] is False
