"""OpenAI Responses API provider.

Uses ``client.responses.create()`` — the newer OpenAI API that is now the
recommended interface for all new projects and supports stateful multi-turn,
built-in tools (web search, code interpreter), and a cleaner structured output
format.

Compatible endpoints
--------------------
* OpenAI:         ``https://api.openai.com/v1`` (default, no ``base_url`` needed)
* Alibaba Cloud:  ``https://dashscope-intl.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1``
* Any provider that exposes a ``/responses`` endpoint via the OpenAI SDK

Alibaba compatibility notes
----------------------------
The ``instructions`` parameter is **not supported** by Alibaba's Responses
endpoint.  This provider therefore always places the system/instructions text
as the first message in the ``input`` array (``{"role": "system", "content":
...}``) rather than using the ``instructions`` parameter, guaranteeing
identical behaviour on both OpenAI and Alibaba endpoints.

Thinking / reasoning mode
--------------------------
For OpenAI reasoning models (o3, o4-mini, …) pass ``thinking_enabled=True``
which sets ``reasoning={"effort": "medium"}``.  The effort level can be
overridden via ``OpenAIResponsesConfig.reasoning_effort``.

For Alibaba/Qwen hybrid thinking models, set ``thinking_enabled=True`` which
adds ``enable_thinking=True`` via the ``extra_body`` parameter in the SDK call.

Stateful multi-turn
--------------------
Set ``config.previous_response_id`` to the ``id`` returned in the previous
response to let the server manage context automatically.  After every
successful call the provider stores the latest response id in
``provider.last_response_id``.
"""

from __future__ import annotations

import json as _json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from .base import BaseProvider
from ..llm_config import LLMConfig
from core_lib import get_module_logger
from core_lib.tracing.tracing import add_trace_metadata
from core_lib.tracing.service_usage import log_llm_usage

logger = get_module_logger()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OpenAIResponsesConfig(LLMConfig):
    """Configuration for the OpenAI Responses API provider.

    Attributes:
        api_key:              API key (OpenAI or Alibaba DashScope key).
        base_url:             Override base URL.  Set to the Alibaba endpoint to
                              use Qwen models via the Responses API.
        organization:         OpenAI organisation ID (ignored by Alibaba).
        project:              OpenAI project ID (ignored by Alibaba).
        previous_response_id: When set the server uses this as context for
                              stateful multi-turn conversation.  Update between
                              calls to ``provider.last_response_id``.
        reasoning_effort:     "low" | "medium" | "high" — only meaningful for
                              OpenAI reasoning models when ``thinking_enabled``
                              is True.  Alibaba will use ``enable_thinking``
                              instead.
        thinking_budget:      Maximum number of tokens the model may use for
                              its internal reasoning/thinking step (Alibaba/Qwen
                              only).  Maps to ``extra_body["thinking_budget"]``.
                              Has no effect when targeting plain OpenAI.
        is_alibaba:           Set to True when targeting Alibaba's Responses
                              endpoint so that provider-specific quirks are
                              applied automatically.  Inferred from ``base_url``
                              when not set explicitly.
    """

    api_key: str
    base_url: Optional[str] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    previous_response_id: Optional[str] = None
    reasoning_effort: str = "medium"
    thinking_budget: Optional[int] = None
    is_alibaba: Optional[bool] = None

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        thinking_enabled: bool = False,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        reasoning_effort: str = "medium",
        thinking_budget: Optional[int] = None,
        is_alibaba: Optional[bool] = None,
    ):
        super().__init__("openai-responses", model, temperature, max_tokens, thinking_enabled)
        self.api_key = api_key
        self.base_url = base_url
        self.organization = organization
        self.project = project
        self.previous_response_id = previous_response_id
        self.reasoning_effort = reasoning_effort
        self.thinking_budget = thinking_budget
        # Auto-detect Alibaba from base_url if not explicitly set
        if is_alibaba is None:
            self.is_alibaba = bool(base_url and "alibaba" in base_url.lower() or
                                   base_url and "dashscope" in base_url.lower() or
                                   base_url and "aliyun" in base_url.lower())
        else:
            self.is_alibaba = is_alibaba

    @classmethod
    def from_env(cls) -> "OpenAIResponsesConfig":
        import os

        def getenv(*names: str, default: Optional[str] = None) -> Optional[str]:
            for n in names:
                v = os.getenv(n)
                if v is not None:
                    return v
            return default

        api_key = getenv("OPENAI_API_KEY", "DASHSCOPE_API_KEY", default="") or ""
        model = getenv("OPENAI_RESPONSES_MODEL", "OPENAI_MODEL", default="gpt-4.1") or "gpt-4.1"
        temperature = float(getenv("OPENAI_TEMPERATURE", default="0.7") or 0.7)
        max_tokens_env = getenv("OPENAI_MAX_TOKENS")
        max_tokens = int(max_tokens_env) if max_tokens_env else None
        base_url = getenv("OPENAI_RESPONSES_BASE_URL", "OPENAI_BASE_URL")
        organization = getenv("OPENAI_ORG", "OPENAI_ORGANIZATION")
        project = getenv("OPENAI_PROJECT")
        reasoning_effort = getenv("OPENAI_REASONING_EFFORT", default="medium") or "medium"
        thinking_env = getenv("OPENAI_THINKING_ENABLED", default="false") or "false"
        thinking_enabled = thinking_env.lower() in ("1", "true", "yes")
        thinking_budget_env = getenv("OPENAI_THINKING_BUDGET")
        thinking_budget = int(thinking_budget_env) if thinking_budget_env else None

        return cls(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            organization=organization,
            project=project,
            reasoning_effort=reasoning_effort,
            thinking_enabled=thinking_enabled,
            thinking_budget=thinking_budget,
        )

    @classmethod
    def for_alibaba(
        cls,
        api_key: str,
        model: str = "qwen-plus",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        thinking_enabled: bool = False,
        thinking_budget: Optional[int] = None,
        region: str = "international",
    ) -> "OpenAIResponsesConfig":
        """Create a config pre-configured for Alibaba Cloud Model Studio.

        Args:
            api_key:          DashScope API key (``sk-...``).
            model:            Qwen model name, e.g. ``qwen-plus``, ``qwen3-max``.
            temperature:      Sampling temperature.
            max_tokens:       Maximum output tokens.
            thinking_enabled: Enable Qwen thinking/chain-of-thought mode.
            thinking_budget:  Token budget for the thinking step (e.g. 4000).
                              Ignored when ``thinking_enabled`` is False.
            region:           ``"international"`` (Singapore/Virginia, default)
                              or ``"china"`` (Beijing).
        """
        if region == "china":
            base_url = "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1"
        else:
            base_url = "https://dashscope-intl.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1"

        return cls(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_enabled=thinking_enabled,
            thinking_budget=thinking_budget,
            base_url=base_url,
            is_alibaba=True,
        )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class OpenAIResponsesProvider(BaseProvider):
    """Provider implementation for the OpenAI Responses API.

    Uses ``client.responses.create()`` rather than
    ``client.chat.completions.create()`` — the recommended API for all new
    OpenAI (and compatible) projects.
    """

    def __init__(self, config: OpenAIResponsesConfig) -> None:  # type: ignore[override]
        super().__init__(config)
        from openai import OpenAI as _OpenAI  # type: ignore

        kwargs: Dict[str, Any] = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.organization:
            kwargs["organization"] = config.organization
        if config.project:
            kwargs["project"] = config.project
        self._client = _OpenAI(**kwargs)
        # Stores the last response id for stateful multi-turn
        self.last_response_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_tool_param(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert OpenAI function schema tools to Responses API format.

        The Responses API uses the same tool schema as Chat Completions for
        user-defined functions, so no conversion is needed.
        """
        return tools or None

    def _build_text_format(
        self, structured_output: Type[BaseModel]
    ) -> Dict[str, Any]:
        """Build the ``text.format`` dict for structured JSON output."""
        try:
            schema = structured_output.model_json_schema()  # type: ignore[attr-defined]
        except Exception:
            schema = structured_output.schema()  # type: ignore[attr-defined]

        return {
            "type": "json_schema",
            "name": structured_output.__name__,
            "schema": schema,
            "strict": True,
        }

    def _extract_text(self, response: Any) -> str:
        """Extract the plain text content from a Responses API response object."""
        # The SDK provides a convenience .output_text property that concatenates
        # all text outputs — use it when available.
        output_text = getattr(response, "output_text", None)
        if output_text is not None:
            return output_text

        # Fallback: manually walk output items
        parts: List[str] = []
        for item in getattr(response, "output", []) or []:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for part in getattr(item, "content", []) or []:
                    part_type = getattr(part, "type", None)
                    if part_type == "output_text":
                        parts.append(getattr(part, "text", "") or "")
            elif item_type == "reasoning":
                # Skip reasoning/thinking tokens from the text output
                pass
        return "".join(parts)

    def _extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract function call items from response output into OpenAI-style tool_calls."""
        tool_calls: List[Dict[str, Any]] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "function_call":
                tool_calls.append(
                    {
                        "id": getattr(item, "call_id", None),
                        "type": "function",
                        "function": {
                            "name": getattr(item, "name", ""),
                            "arguments": getattr(item, "arguments", "{}"),
                        },
                    }
                )
        return tool_calls

    def _extract_usage(self, response: Any) -> Dict[str, Any]:
        """Normalise Responses API usage to ChatCompletion-style keys."""
        usage_obj = getattr(response, "usage", None)
        if usage_obj is None:
            return {}
        # Responses API uses input_tokens / output_tokens (not prompt_tokens)
        input_tokens = getattr(usage_obj, "input_tokens", None)
        output_tokens = getattr(usage_obj, "output_tokens", None)
        total_tokens = getattr(usage_obj, "total_tokens", None)
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    # ------------------------------------------------------------------
    # BaseProvider.chat implementation
    # ------------------------------------------------------------------

    def chat(
        self,
        *,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        structured_output: Optional[Type[BaseModel]] = None,
        system_message: Optional[str] = None,
        use_search_grounding: bool = False,
        thinking_enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        cfg: OpenAIResponsesConfig = self.config  # type: ignore[assignment]

        # -----------------------------------------------------------------
        # Build input messages
        # Alibaba does NOT support the `instructions` parameter, so we always
        # inject the system message as the first input item instead.
        # -----------------------------------------------------------------
        input_messages: List[Dict[str, Any]] = list(messages)

        if system_message:
            # Override: place as first message (replace existing system if any)
            if input_messages and input_messages[0].get("role") == "system":
                input_messages = [{"role": "system", "content": system_message}] + input_messages[1:]
            else:
                input_messages = [{"role": "system", "content": system_message}] + input_messages
        # (if messages already contains a system role at index 0 we leave it as-is)

        # -----------------------------------------------------------------
        # Determine effective thinking mode
        # -----------------------------------------------------------------
        use_thinking = thinking_enabled if thinking_enabled is not None else cfg.thinking_enabled

        try:
            logger.debug(
                "openai_responses.chat start",
                extra={
                    "llm_provider": "openai-responses",
                    "model": cfg.model,
                    "msg_count": len(input_messages),
                    "has_tools": bool(tools),
                    "structured": bool(structured_output),
                    "search_grounding": use_search_grounding,
                    "thinking": use_thinking,
                    "is_alibaba": cfg.is_alibaba,
                },
            )

            # -----------------------------------------------------------------
            # Assemble create_kwargs
            # -----------------------------------------------------------------
            create_kwargs: Dict[str, Any] = {
                "model": cfg.model,
                "input": input_messages,
            }

            # Temperature / generation parameters
            if cfg.temperature is not None:
                create_kwargs["temperature"] = cfg.temperature
            if cfg.max_tokens is not None:
                create_kwargs["max_output_tokens"] = cfg.max_tokens

            # Stateful multi-turn
            if cfg.previous_response_id:
                create_kwargs["previous_response_id"] = cfg.previous_response_id

            # Tools (function calling)
            tool_param = self._build_tool_param(tools)
            if tool_param:
                create_kwargs["tools"] = tool_param

            # Web search grounding
            if use_search_grounding:
                grounding_tool: Dict[str, Any] = {"type": "web_search_preview"}
                if "tools" in create_kwargs and isinstance(create_kwargs["tools"], list):
                    create_kwargs["tools"] = [*create_kwargs["tools"], grounding_tool]
                else:
                    create_kwargs["tools"] = [grounding_tool]

            # Structured output
            if structured_output is not None:
                create_kwargs["text"] = {"format": self._build_text_format(structured_output)}

            # Thinking / reasoning mode
            if use_thinking:
                if cfg.is_alibaba:
                    # Alibaba/Qwen: enable_thinking + optional budget via extra_body
                    extra_body: Dict[str, Any] = {"enable_thinking": True}
                    if cfg.thinking_budget is not None:
                        extra_body["thinking_budget"] = cfg.thinking_budget
                    create_kwargs["extra_body"] = extra_body
                else:
                    # OpenAI reasoning models use the reasoning parameter
                    create_kwargs["reasoning"] = {"effort": cfg.reasoning_effort}

            # -----------------------------------------------------------------
            # Call API
            # -----------------------------------------------------------------
            start = time.perf_counter()
            response = self._client.responses.create(**create_kwargs)
            latency_ms = (time.perf_counter() - start) * 1000

            # Store response id for potential next turn
            self.last_response_id = getattr(response, "id", None)

            # -----------------------------------------------------------------
            # Parse response
            # -----------------------------------------------------------------
            content_text = self._extract_text(response)
            tool_calls = self._extract_tool_calls(response)
            usage = self._extract_usage(response)

            # -----------------------------------------------------------------
            # Observability
            # -----------------------------------------------------------------
            try:
                input_tokens = usage.get("prompt_tokens")
                output_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")
                tokens_per_second = (
                    (total_tokens / latency_ms) * 1000
                    if total_tokens and latency_ms > 0
                    else None
                )
                usage["latency_ms"] = latency_ms
                if tokens_per_second is not None:
                    usage["tokens_per_second"] = tokens_per_second

                add_trace_metadata(
                    {k: v for k, v in {
                        "gen_ai.system": "openai-responses",
                        "gen_ai.request.model": cfg.model,
                        "gen_ai.usage.input_tokens": input_tokens,
                        "gen_ai.usage.output_tokens": output_tokens,
                        "tokens.total": total_tokens,
                        "gen_ai.response.latency_ms": latency_ms,
                        "latency_ms": latency_ms,
                        "tokens_per_second": tokens_per_second,
                        "features.structured_output": str(bool(structured_output)).lower(),
                        "features.tools": str(bool(tools)).lower(),
                        "features.search_grounding": str(use_search_grounding).lower(),
                        "features.thinking": str(use_thinking).lower(),
                    }.items() if v is not None}
                )

                log_llm_usage(
                    provider="openai-responses",
                    model=cfg.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    structured=bool(structured_output),
                    has_tools=bool(tools),
                    search_grounding=use_search_grounding,
                    host=cfg.base_url or "https://api.openai.com",
                )
            except Exception as e:
                logger.warning(f"Failed to log LLM usage: {e}")

            # -----------------------------------------------------------------
            # Structured output parsing
            # -----------------------------------------------------------------
            if structured_output is not None:
                from ..json_parser import parse_structured_output as _parse_structured_output

                parsed_dict: Any = None
                # Attempt 1: native Pydantic validation
                try:
                    validated = structured_output.model_validate_json(content_text)  # type: ignore[attr-defined]
                    parsed_dict = validated.model_dump()
                except Exception:
                    pass

                # Attempt 2: fuzzy recovery (handles fences, schema echo, etc.)
                if parsed_dict is None and content_text:
                    parsed_dict = _parse_structured_output(content_text, structured_output)

                if parsed_dict is not None:
                    return {
                        "content": parsed_dict,
                        "structured": True,
                        "tool_calls": tool_calls,
                        "usage": usage,
                        "text": content_text,
                        "content_json": _json.dumps(parsed_dict, ensure_ascii=False),
                        "response_id": self.last_response_id,
                    }

                logger.warning(
                    "openai_responses: structured output could not be validated against %s; "
                    "falling back to unstructured text",
                    structured_output.__name__,
                )
                return {
                    "content": content_text,
                    "structured": False,
                    "tool_calls": tool_calls,
                    "usage": usage,
                    "text": content_text,
                    "response_id": self.last_response_id,
                }

            # Plain text response
            return {
                "content": content_text,
                "structured": False,
                "tool_calls": tool_calls,
                "usage": usage,
                "response_id": self.last_response_id,
            }

        except Exception as e:  # pragma: no cover - network errors
            logger.exception("openai_responses.chat failed")
            return {
                "error": str(e),
                "content": None,
                "structured": structured_output is not None,
                "tool_calls": [],
                "usage": {},
                "response_id": self.last_response_id,
            }
