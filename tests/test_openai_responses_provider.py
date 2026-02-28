"""Tests for the OpenAI Responses API provider."""

import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel
from typing import Optional

from core_lib.llm.providers.openai_responses_provider import (
    OpenAIResponsesConfig,
    OpenAIResponsesProvider,
)
from core_lib.llm.providers.openai_provider import OpenAIConfig, OpenAIProvider
from core_lib.llm import LLMClient, create_openai_responses_client, create_alibaba_client
from core_lib.llm.factory import LLMFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Chat Completions endpoint (active Alibaba path)
ALIBABA_INTL_BASE_URL = (
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
ALIBABA_CN_BASE_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def _make_mock_response(
    output_text: str = "Hello there!",
    response_id: str = "resp_abc123",
    input_tokens: int = 10,
    output_tokens: int = 20,
    tool_calls: Optional[list] = None,
) -> MagicMock:
    """Build a mock OpenAI Responses API response object."""
    response = MagicMock()
    response.id = response_id
    response.output_text = output_text

    # Build output items
    output_items = []
    msg_item = MagicMock()
    msg_item.type = "message"
    msg_item.role = "assistant"
    content_part = MagicMock()
    content_part.type = "output_text"
    content_part.text = output_text
    msg_item.content = [content_part]
    output_items.append(msg_item)

    if tool_calls:
        for tc in tool_calls:
            fc_item = MagicMock()
            fc_item.type = "function_call"
            fc_item.call_id = tc["call_id"]
            fc_item.name = tc["name"]
            fc_item.arguments = tc["arguments"]
            output_items.append(fc_item)

    response.output = output_items

    # Usage
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.total_tokens = input_tokens + output_tokens
    response.usage = usage

    return response


def _make_provider(
    model: str = "gpt-4.1",
    is_alibaba: bool = False,
    thinking_enabled: bool = False,
    thinking_budget: Optional[int] = None,
    previous_response_id: Optional[str] = None,
) -> tuple[OpenAIResponsesProvider, MagicMock]:
    """Create a provider with a mocked OpenAI client. Returns (provider, mock_client)."""
    config = OpenAIResponsesConfig(
        api_key="sk-test",
        model=model,
        is_alibaba=is_alibaba,
        thinking_enabled=thinking_enabled,
        thinking_budget=thinking_budget,
        previous_response_id=previous_response_id,
    )
    with patch("core_lib.llm.providers.openai_responses_provider.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        provider = OpenAIResponsesProvider(config)
        # replace with reference for assertions
        provider._client = mock_client
    return provider, mock_client


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestOpenAIResponsesConfig:
    def test_basic_creation(self):
        cfg = OpenAIResponsesConfig(api_key="sk-test", model="gpt-5")
        assert cfg.api_key == "sk-test"
        assert cfg.model == "gpt-5"
        assert cfg.provider == "openai-responses"
        assert cfg.is_alibaba is False

    def test_alibaba_auto_detected_from_base_url(self):
        cfg = OpenAIResponsesConfig(
            api_key="sk-test",
            base_url=ALIBABA_INTL_BASE_URL,
        )
        assert cfg.is_alibaba is True

    def test_dashscope_base_url_auto_detected(self):
        cfg = OpenAIResponsesConfig(
            api_key="sk-test",
            base_url="https://dashscope.aliyuncs.com/api/v2/something",
        )
        assert cfg.is_alibaba is True

    def test_is_alibaba_explicit_override(self):
        # Even with a dashscope URL, explicit False should win
        cfg = OpenAIResponsesConfig(
            api_key="sk-test",
            base_url=ALIBABA_INTL_BASE_URL,
            is_alibaba=False,
        )
        assert cfg.is_alibaba is False

    @patch.dict("os.environ", {
        "OPENAI_API_KEY": "env-key",
        "OPENAI_RESPONSES_MODEL": "gpt-5",
        "OPENAI_TEMPERATURE": "0.3",
        "OPENAI_MAX_TOKENS": "1024",
        "OPENAI_REASONING_EFFORT": "high",
    })
    def test_from_env(self):
        cfg = OpenAIResponsesConfig.from_env()
        assert cfg.api_key == "env-key"
        assert cfg.model == "gpt-5"
        assert cfg.temperature == 0.3
        assert cfg.max_tokens == 1024
        assert cfg.reasoning_effort == "high"

    @patch.dict("os.environ", {
        "DASHSCOPE_API_KEY": "dash-key",
        "OPENAI_API_KEY": "openai-key",
    })
    def test_from_env_prefers_openai_key(self):
        cfg = OpenAIResponsesConfig.from_env()
        assert cfg.api_key == "openai-key"

    def test_for_alibaba_international(self):
        cfg = OpenAIResponsesConfig.for_alibaba(
            api_key="dash-key",
            model="qwen-plus",
        )
        assert cfg.is_alibaba is True
        assert "dashscope-intl" in cfg.base_url
        assert cfg.model == "qwen-plus"
        assert cfg.api_key == "dash-key"

    def test_for_alibaba_china(self):
        cfg = OpenAIResponsesConfig.for_alibaba(
            api_key="dash-key",
            model="qwen-plus",
            region="china",
        )
        assert cfg.is_alibaba is True
        assert "dashscope.aliyuncs.com" in cfg.base_url
        assert "intl" not in cfg.base_url


# ---------------------------------------------------------------------------
# Provider instantiation
# ---------------------------------------------------------------------------

class TestOpenAIResponsesProviderInit:
    def test_initialises_openai_client(self):
        config = OpenAIResponsesConfig(api_key="sk-test")
        with patch("core_lib.llm.providers.openai_responses_provider.OpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            provider = OpenAIResponsesProvider(config)
            mock_cls.assert_called_once_with(api_key="sk-test")

    def test_passes_base_url(self):
        config = OpenAIResponsesConfig(api_key="sk-test", base_url="https://example.com/v1")
        with patch("core_lib.llm.providers.openai_responses_provider.OpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            OpenAIResponsesProvider(config)
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["base_url"] == "https://example.com/v1"

    def test_passes_organization(self):
        config = OpenAIResponsesConfig(api_key="sk-test", organization="org-123")
        with patch("core_lib.llm.providers.openai_responses_provider.OpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            OpenAIResponsesProvider(config)
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["organization"] == "org-123"

    def test_last_response_id_starts_none(self):
        provider, _ = _make_provider()
        assert provider.last_response_id is None


# ---------------------------------------------------------------------------
# Basic chat call
# ---------------------------------------------------------------------------

class TestOpenAIResponsesProviderChat:
    def test_basic_text_response(self):
        provider, mock_client = _make_provider()
        mock_client.responses.create.return_value = _make_mock_response("Paris is the capital.")

        result = provider.chat(messages=[{"role": "user", "content": "What is the capital of France?"}])

        assert result["content"] == "Paris is the capital."
        assert result["structured"] is False
        assert result["tool_calls"] == []
        assert result["error"] is not None or "error" not in result

    def test_response_id_stored(self):
        provider, mock_client = _make_provider()
        mock_client.responses.create.return_value = _make_mock_response(response_id="resp_xyz789")

        provider.chat(messages=[{"role": "user", "content": "Hello"}])

        assert provider.last_response_id == "resp_xyz789"

    def test_response_id_in_result(self):
        provider, mock_client = _make_provider()
        mock_client.responses.create.return_value = _make_mock_response(response_id="resp_abc")

        result = provider.chat(messages=[{"role": "user", "content": "Hello"}])

        assert result["response_id"] == "resp_abc"

    def test_usage_normalised(self):
        provider, mock_client = _make_provider()
        mock_client.responses.create.return_value = _make_mock_response(
            input_tokens=15, output_tokens=25
        )

        result = provider.chat(messages=[{"role": "user", "content": "Hi"}])

        assert result["usage"]["prompt_tokens"] == 15
        assert result["usage"]["completion_tokens"] == 25
        assert result["usage"]["total_tokens"] == 40

    def test_model_and_input_in_create_call(self):
        provider, mock_client = _make_provider(model="gpt-5")
        mock_client.responses.create.return_value = _make_mock_response()

        messages = [{"role": "user", "content": "Hello"}]
        provider.chat(messages=messages)

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5"
        assert call_kwargs["input"] == messages

    def test_temperature_passed(self):
        config = OpenAIResponsesConfig(api_key="sk-test", temperature=0.3)
        with patch("core_lib.llm.providers.openai_responses_provider.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            provider = OpenAIResponsesProvider(config)
            provider._client = mock_client
            mock_client.responses.create.return_value = _make_mock_response()

            provider.chat(messages=[{"role": "user", "content": "Hello"}])

            call_kwargs = mock_client.responses.create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.3

    def test_max_tokens_mapped_to_max_output_tokens(self):
        config = OpenAIResponsesConfig(api_key="sk-test", max_tokens=512)
        with patch("core_lib.llm.providers.openai_responses_provider.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            provider = OpenAIResponsesProvider(config)
            provider._client = mock_client
            mock_client.responses.create.return_value = _make_mock_response()

            provider.chat(messages=[{"role": "user", "content": "Hello"}])

            call_kwargs = mock_client.responses.create.call_args.kwargs
            assert call_kwargs["max_output_tokens"] == 512
            assert "max_tokens" not in call_kwargs


# ---------------------------------------------------------------------------
# System message handling
# ---------------------------------------------------------------------------

class TestSystemMessageHandling:
    def test_system_message_param_prepended_as_first_input(self):
        provider, mock_client = _make_provider()
        mock_client.responses.create.return_value = _make_mock_response()

        provider.chat(
            messages=[{"role": "user", "content": "Hello"}],
            system_message="You are a helpful assistant.",
        )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        input_msgs = call_kwargs["input"]
        assert input_msgs[0]["role"] == "system"
        assert input_msgs[0]["content"] == "You are a helpful assistant."
        assert input_msgs[1]["role"] == "user"

    def test_instructions_param_not_used(self):
        """Provider must NOT use the `instructions` param (Alibaba doesn't support it)."""
        provider, mock_client = _make_provider()
        mock_client.responses.create.return_value = _make_mock_response()

        provider.chat(
            messages=[{"role": "user", "content": "Hello"}],
            system_message="Be concise.",
        )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert "instructions" not in call_kwargs

    def test_existing_system_role_not_duplicated(self):
        provider, mock_client = _make_provider()
        mock_client.responses.create.return_value = _make_mock_response()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        provider.chat(messages=messages)

        call_kwargs = mock_client.responses.create.call_args.kwargs
        system_msgs = [m for m in call_kwargs["input"] if m.get("role") == "system"]
        assert len(system_msgs) == 1

    def test_system_message_param_replaces_existing_system_message(self):
        provider, mock_client = _make_provider()
        mock_client.responses.create.return_value = _make_mock_response()

        messages = [
            {"role": "system", "content": "Old system message."},
            {"role": "user", "content": "Hello"},
        ]
        provider.chat(messages=messages, system_message="New system message.")

        call_kwargs = mock_client.responses.create.call_args.kwargs
        system_msgs = [m for m in call_kwargs["input"] if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "New system message."


# ---------------------------------------------------------------------------
# Tool calling
# ---------------------------------------------------------------------------

class TestToolCalling:
    def test_tools_passed_to_api(self):
        provider, mock_client = _make_provider()
        mock_client.responses.create.return_value = _make_mock_response()

        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        provider.chat(messages=[{"role": "user", "content": "Weather?"}], tools=tools)

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs["tools"] == tools

    def test_tool_calls_extracted_from_response(self):
        provider, mock_client = _make_provider()
        mock_client.responses.create.return_value = _make_mock_response(
            tool_calls=[
                {"call_id": "call_1", "name": "get_weather", "arguments": '{"city": "Paris"}'}
            ]
        )

        result = provider.chat(
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        )

        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city": "Paris"}'
        assert tc["id"] == "call_1"

    def test_no_tools_key_absent_from_create_call(self):
        provider, mock_client = _make_provider()
        mock_client.responses.create.return_value = _make_mock_response()

        provider.chat(messages=[{"role": "user", "content": "Hello"}])

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert "tools" not in call_kwargs


# ---------------------------------------------------------------------------
# Search grounding
# ---------------------------------------------------------------------------

class TestSearchGrounding:
    def test_web_search_preview_added(self):
        provider, mock_client = _make_provider()
        mock_client.responses.create.return_value = _make_mock_response()

        provider.chat(
            messages=[{"role": "user", "content": "Latest news?"}],
            use_search_grounding=True,
        )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        tools = call_kwargs.get("tools", [])
        assert any(t.get("type") == "web_search_preview" for t in tools)

    def test_grounding_merged_with_existing_tools(self):
        provider, mock_client = _make_provider()
        mock_client.responses.create.return_value = _make_mock_response()

        function_tool = {"type": "function", "function": {"name": "fn", "parameters": {}}}
        provider.chat(
            messages=[{"role": "user", "content": "Go!"}],
            tools=[function_tool],
            use_search_grounding=True,
        )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        tools = call_kwargs["tools"]
        assert len(tools) == 2
        assert any(t.get("type") == "web_search_preview" for t in tools)
        assert any(t.get("type") == "function" for t in tools)


# ---------------------------------------------------------------------------
# Thinking / reasoning mode
# ---------------------------------------------------------------------------

class TestThinkingMode:
    def test_openai_thinking_sets_reasoning_param(self):
        provider, mock_client = _make_provider(is_alibaba=False)
        mock_client.responses.create.return_value = _make_mock_response()

        provider.chat(
            messages=[{"role": "user", "content": "Think hard."}],
            thinking_enabled=True,
        )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs.get("reasoning") == {"effort": "medium"}
        assert "extra_body" not in call_kwargs

    def test_openai_thinking_respects_reasoning_effort(self):
        config = OpenAIResponsesConfig(
            api_key="sk-test", is_alibaba=False, reasoning_effort="high"
        )
        with patch("core_lib.llm.providers.openai_responses_provider.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            provider = OpenAIResponsesProvider(config)
            provider._client = mock_client
            mock_client.responses.create.return_value = _make_mock_response()

            provider.chat(
                messages=[{"role": "user", "content": "Think."}],
                thinking_enabled=True,
            )

            call_kwargs = mock_client.responses.create.call_args.kwargs
            assert call_kwargs["reasoning"] == {"effort": "high"}

    def test_alibaba_thinking_sets_extra_body(self):
        provider, mock_client = _make_provider(is_alibaba=True)
        mock_client.responses.create.return_value = _make_mock_response()

        provider.chat(
            messages=[{"role": "user", "content": "Think hard."}],
            thinking_enabled=True,
        )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs.get("extra_body") == {"enable_thinking": True}
        assert "reasoning" not in call_kwargs

    def test_alibaba_thinking_with_budget(self):
        provider, mock_client = _make_provider(is_alibaba=True, thinking_budget=4000)
        mock_client.responses.create.return_value = _make_mock_response()

        provider.chat(
            messages=[{"role": "user", "content": "Think hard."}],
            thinking_enabled=True,
        )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs.get("extra_body") == {"enable_thinking": True, "thinking_budget": 4000}

    def test_alibaba_thinking_budget_ignored_when_thinking_disabled(self):
        """thinking_budget must not appear in extra_body when thinking is off."""
        provider, mock_client = _make_provider(is_alibaba=True, thinking_budget=4000)
        mock_client.responses.create.return_value = _make_mock_response()

        provider.chat(
            messages=[{"role": "user", "content": "Hello"}],
            thinking_enabled=False,
        )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert "extra_body" not in call_kwargs

    def test_no_thinking_by_default(self):
        provider, mock_client = _make_provider()
        mock_client.responses.create.return_value = _make_mock_response()

        provider.chat(messages=[{"role": "user", "content": "Hello"}])

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert "reasoning" not in call_kwargs
        assert "extra_body" not in call_kwargs

    def test_thinking_enabled_from_config(self):
        provider, mock_client = _make_provider(thinking_enabled=True, is_alibaba=False)
        mock_client.responses.create.return_value = _make_mock_response()

        provider.chat(messages=[{"role": "user", "content": "Hello"}])

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert "reasoning" in call_kwargs

    def test_thinking_override_false_suppresses_config(self):
        provider, mock_client = _make_provider(thinking_enabled=True)
        mock_client.responses.create.return_value = _make_mock_response()

        provider.chat(
            messages=[{"role": "user", "content": "Hello"}],
            thinking_enabled=False,
        )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert "reasoning" not in call_kwargs
        assert "extra_body" not in call_kwargs


# ---------------------------------------------------------------------------
# Stateful multi-turn (previous_response_id)
# ---------------------------------------------------------------------------

class TestStatefulMultiTurn:
    def test_previous_response_id_passed(self):
        provider, mock_client = _make_provider(previous_response_id="resp_prev_123")
        mock_client.responses.create.return_value = _make_mock_response()

        provider.chat(messages=[{"role": "user", "content": "Follow-up"}])

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs["previous_response_id"] == "resp_prev_123"

    def test_no_previous_response_id_by_default(self):
        provider, mock_client = _make_provider()
        mock_client.responses.create.return_value = _make_mock_response()

        provider.chat(messages=[{"role": "user", "content": "First message"}])

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert "previous_response_id" not in call_kwargs


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

class WeatherResponse(BaseModel):
    city: str
    temperature: float
    condition: str


class TestStructuredOutput:
    def test_text_format_sent_in_request(self):
        provider, mock_client = _make_provider()
        mock_client.responses.create.return_value = _make_mock_response(
            output_text='{"city": "Paris", "temperature": 20.0, "condition": "sunny"}'
        )

        provider.chat(
            messages=[{"role": "user", "content": "Weather in Paris?"}],
            structured_output=WeatherResponse,
        )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        text_format = call_kwargs.get("text", {}).get("format", {})
        assert text_format["type"] == "json_schema"
        assert text_format["name"] == "WeatherResponse"
        assert "schema" in text_format
        assert text_format["strict"] is True

    def test_structured_response_parsed(self):
        provider, mock_client = _make_provider()
        mock_client.responses.create.return_value = _make_mock_response(
            output_text='{"city": "Paris", "temperature": 20.0, "condition": "sunny"}'
        )

        result = provider.chat(
            messages=[{"role": "user", "content": "Weather?"}],
            structured_output=WeatherResponse,
        )

        assert result["structured"] is True
        assert result["content"]["city"] == "Paris"
        assert result["content"]["temperature"] == 20.0

    def test_structured_response_fallback_on_invalid_json(self):
        provider, mock_client = _make_provider()
        mock_client.responses.create.return_value = _make_mock_response(
            output_text="This is not JSON at all."
        )

        result = provider.chat(
            messages=[{"role": "user", "content": "Weather?"}],
            structured_output=WeatherResponse,
        )

        # Should return unstructured with the raw text
        assert result["structured"] is False
        assert result["content"] == "This is not JSON at all."


# ---------------------------------------------------------------------------
# LLMClient integration
# ---------------------------------------------------------------------------

class TestLLMClientIntegration:
    def test_llmclient_instantiates_responses_provider(self):
        config = OpenAIResponsesConfig(api_key="sk-test")
        with patch("core_lib.llm.llm_client.OpenAIResponsesProvider") as mock_provider_cls:
            mock_provider_cls.return_value = MagicMock()
            client = LLMClient(config)
            mock_provider_cls.assert_called_once_with(config)

    def test_llmclient_does_not_instantiate_openai_provider_for_responses_config(self):
        config = OpenAIResponsesConfig(api_key="sk-test")
        with (
            patch("core_lib.llm.llm_client.OpenAIResponsesProvider") as mock_responses,
            patch("core_lib.llm.llm_client.OpenAIProvider") as mock_chat,
        ):
            mock_responses.return_value = MagicMock()
            mock_chat.return_value = MagicMock()
            LLMClient(config)
            mock_responses.assert_called_once()
            mock_chat.assert_not_called()


# ---------------------------------------------------------------------------
# ProviderConfig → thinking_budget propagation
# ---------------------------------------------------------------------------

class TestProviderConfigThinkingBudget:
    """Test that thinking_budget flows from llm_providers.yaml / ProviderConfig → OpenAIConfig (alibaba) or OpenAIResponsesConfig (openai-responses)."""

    def _make_registry_config(self, data: dict):
        from core_lib.llm.provider_registry import ProviderConfig
        return ProviderConfig.from_dict(data)

    def test_thinking_budget_parsed_from_yaml_dict(self):
        cfg = self._make_registry_config({
            "provider": "alibaba",
            "api_key": "sk-test",
            "model": "qwen3-flash",
            "thinking": True,
            "thinking_budget": 4000,
        })
        assert cfg.thinking_enabled is True
        assert cfg.thinking_config is not None
        assert cfg.thinking_config["budget"] == 4000

    def test_thinking_budget_propagates_to_llm_config(self):
        cfg = self._make_registry_config({
            "provider": "alibaba",
            "api_key": "sk-test",
            "model": "qwen3-flash",
            "thinking": True,
            "thinking_budget": 4000,
        })
        llm_cfg = cfg.to_llm_config()
        # alibaba routes to OpenAIConfig (Chat Completions), not OpenAIResponsesConfig
        assert isinstance(llm_cfg, OpenAIConfig)
        assert llm_cfg.thinking_enabled is True
        assert llm_cfg.thinking_budget == 4000

    def test_no_thinking_budget_when_not_set(self):
        cfg = self._make_registry_config({
            "provider": "alibaba",
            "api_key": "sk-test",
            "model": "qwen-plus",
            "thinking": True,
        })
        llm_cfg = cfg.to_llm_config()
        assert isinstance(llm_cfg, OpenAIConfig)
        assert llm_cfg.thinking_budget is None

    def test_thinking_disabled_no_budget(self):
        cfg = self._make_registry_config({
            "provider": "alibaba",
            "api_key": "sk-test",
            "model": "qwen-plus",
        })
        llm_cfg = cfg.to_llm_config()
        assert isinstance(llm_cfg, OpenAIConfig)
        assert llm_cfg.thinking_enabled is False
        assert llm_cfg.thinking_budget is None

    def test_int_thinking_value_sets_budget(self):
        """thinking: 4000 (integer) means budget=4000 and thinking_enabled=True."""
        cfg = self._make_registry_config({
            "provider": "openai-responses",
            "api_key": "sk-test",
            "model": "gpt-4.1",
            "thinking": 4000,
        })
        assert cfg.thinking_enabled is True
        llm_cfg = cfg.to_llm_config()
        assert llm_cfg.thinking_budget == 4000

    def test_budget_sent_in_extra_body_via_provider(self):
        """End-to-end: registry config budget → extra_body in Chat Completions API call."""
        cfg = self._make_registry_config({
            "provider": "alibaba",
            "api_key": "sk-test",
            "model": "qwen3-flash",
            "thinking": True,
            "thinking_budget": 2000,
        })
        llm_cfg = cfg.to_llm_config()
        assert isinstance(llm_cfg, OpenAIConfig), "alibaba must produce OpenAIConfig"

        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            # Build a minimal Chat Completions response mock
            mock_choice = MagicMock()
            mock_choice.message.content = "ok"
            mock_choice.message.tool_calls = None
            mock_completion = MagicMock()
            mock_completion.choices = [mock_choice]
            mock_completion.usage.prompt_tokens = 5
            mock_completion.usage.completion_tokens = 10
            mock_completion.usage.total_tokens = 15
            mock_client.chat.completions.create.return_value = mock_completion

            provider = OpenAIProvider(llm_cfg)
            provider._client = mock_client

            provider.chat(messages=[{"role": "user", "content": "Hello"}])

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["extra_body"] == {"enable_thinking": True, "thinking_budget": 2000}


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

class TestFactoryFunctions:
    def test_create_openai_responses_client(self):
        with patch("core_lib.llm.factory.LLMClient") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()
            create_openai_responses_client(api_key="sk-test", model="gpt-5")
            call_config = mock_client_cls.call_args[0][0]
            assert isinstance(call_config, OpenAIResponsesConfig)
            assert call_config.model == "gpt-5"

    def test_create_alibaba_client_sets_alibaba_url(self):
        with patch("core_lib.llm.factory.LLMClient") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()
            create_alibaba_client(api_key="dash-key", model="qwen-plus")
            call_config = mock_client_cls.call_args[0][0]
            # alibaba factory now produces OpenAIConfig (Chat Completions), not OpenAIResponsesConfig
            assert isinstance(call_config, OpenAIConfig)
            assert call_config.is_alibaba is True
            assert "dashscope-intl" in call_config.base_url
            assert "compatible-mode/v1" in call_config.base_url
            assert call_config.model == "qwen-plus"

    def test_create_alibaba_client_china_region(self):
        with patch("core_lib.llm.factory.LLMClient") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()
            create_alibaba_client(api_key="dash-key", model="qwen-plus", region="china")
            call_config = mock_client_cls.call_args[0][0]
            assert "intl" not in call_config.base_url
            assert "dashscope.aliyuncs.com" in call_config.base_url

    @patch.dict("os.environ", {"DASHSCOPE_API_KEY": "env-dash-key"})
    def test_create_alibaba_client_reads_dashscope_key_from_env(self):
        with patch("core_lib.llm.factory.LLMClient") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()
            create_alibaba_client()  # No api_key arg
            call_config = mock_client_cls.call_args[0][0]
            assert call_config.api_key == "env-dash-key"

    def test_llmfactory_openai_responses_method(self):
        with patch("core_lib.llm.factory.LLMClient") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()
            LLMFactory.openai_responses(api_key="sk-test", model="gpt-5", reasoning_effort="low")
            call_config = mock_client_cls.call_args[0][0]
            assert isinstance(call_config, OpenAIResponsesConfig)
            assert call_config.reasoning_effort == "low"

    def test_llmfactory_alibaba_method(self):
        with patch("core_lib.llm.factory.LLMClient") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()
            LLMFactory.alibaba(api_key="sk-test", model="qwen3-max")
            call_config = mock_client_cls.call_args[0][0]
            # LLMFactory.alibaba() now produces OpenAIConfig (Chat Completions)
            assert isinstance(call_config, OpenAIConfig)
            assert call_config.is_alibaba is True
            assert call_config.model == "qwen3-max"

    def test_from_env_routes_openai_responses(self):
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
            patch("core_lib.llm.factory.LLMClient") as mock_client_cls,
        ):
            mock_client_cls.return_value = MagicMock()
            from core_lib.llm.factory import LLMFactory as _F
            _F.from_env(provider="openai-responses")
            call_config = mock_client_cls.call_args[0][0]
            assert isinstance(call_config, OpenAIResponsesConfig)
