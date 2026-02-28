"""OpenAI-compatible provider using the official OpenAI SDK with Langfuse.

Supports:
- OpenAI (standard)
- Azure OpenAI (via AzureOpenAI client)
- OpenAI-compatible endpoints (Ollama, LiteLLM, vLLM) via base_url

Features:
- Structured output via response_format (Pydantic schema supported)
- Tool calling (OpenAI function tool schema)
- Optional "grounding" equivalent using OpenAI file search / web search tools when requested

Tracing:
Uses Langfuse's drop-in OpenAI wrapper for full observability.
Docs: https://langfuse.com/integrations/model-providers/openai-py
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type
import time

from pydantic import BaseModel

from .base import BaseProvider
from ..llm_config import LLMConfig
from dataclasses import dataclass
from typing import Optional
from core_lib import get_module_logger
from core_lib.tracing.tracing import add_trace_metadata
from core_lib.tracing.service_usage import log_llm_usage

logger = get_module_logger()



@dataclass
class OpenAIConfig(LLMConfig):
    api_key: str
    base_url: Optional[str] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_api_version: Optional[str] = None
    thinking_budget: Optional[int] = None

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.2-preview",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        thinking_enabled: bool = False,
        thinking_budget: Optional[int] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
    ):
        super().__init__("openai", model, temperature, max_tokens, thinking_enabled)
        self.api_key = api_key
        self.thinking_budget = thinking_budget
        self.base_url = base_url
        self.organization = organization
        self.project = project
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version

    @property
    def is_alibaba(self) -> bool:
        """True when base_url points to an Alibaba DashScope endpoint."""
        return bool(
            self.base_url and (
                "dashscope" in self.base_url.lower()
                or "aliyun" in self.base_url.lower()
                or "alibaba" in self.base_url.lower()
            )
        )

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        import os

        def getenv(*names: str, default: Optional[str] = None) -> Optional[str]:
            for n in names:
                v = os.getenv(n)
                if v is not None:
                    return v
            return default

        azure_endpoint = getenv("AZURE_OPENAI_ENDPOINT")
        api_key = getenv("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY", default="") or ""
        model = getenv("AZURE_OPENAI_DEPLOYMENT", "OPENAI_MODEL", default="gpt-5.2-preview") or "gpt-5.2-preview"
        temperature = float(getenv("OPENAI_TEMPERATURE", default="0.7") or 0.7)
        max_tokens_env = getenv("OPENAI_MAX_TOKENS")
        max_tokens = int(max_tokens_env) if max_tokens_env else None
        base_url = getenv("OPENAI_BASE_URL")
        organization = getenv("OPENAI_ORG", "OPENAI_ORGANIZATION")
        project = getenv("OPENAI_PROJECT")
        azure_api_version = getenv("AZURE_OPENAI_API_VERSION", default="2024-08-01-preview")

        return cls(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            organization=organization,
            project=project,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
        )

class OpenAIProvider(BaseProvider):
    """Provider implementation for OpenAI-compatible APIs."""

    def __init__(self, config: OpenAIConfig) -> None:  # type: ignore[override]
        super().__init__(config)
        from openai import OpenAI as _OpenAI, AzureOpenAI as _AzureOpenAI  # type: ignore

        # Instantiate client based on mode using official OpenAI SDK only
        if config.azure_endpoint:
            self._client = _AzureOpenAI(
                api_key=config.api_key,
                azure_endpoint=config.azure_endpoint,
                api_version=config.azure_api_version,
            )
        else:
            kwargs: Dict[str, Any] = {"api_key": config.api_key}
            if config.base_url:
                kwargs["base_url"] = config.base_url
            if config.organization:
                kwargs["organization"] = config.organization
            if config.project:
                kwargs["project"] = config.project
            self._client = _OpenAI(**kwargs)

    def _build_tool_param(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        # Assume tools are already in OpenAI format: {type: "function", function: {name, parameters}}
        return tools

    def _build_response_format(self, structured_output: Optional[Type[BaseModel]]) -> Optional[Dict[str, Any]]:
        if structured_output is None:
            return None
        try:
            # Use OpenAI SDK helper if available to convert Pydantic to response_format
            from openai.lib._parsing._completions import type_to_response_format_param  # type: ignore

            return type_to_response_format_param(structured_output)
        except Exception:
            # Fallback: JSON schema from Pydantic
            try:
                schema = structured_output.model_json_schema()  # type: ignore[attr-defined]
            except Exception:
                schema = structured_output.schema()  # type: ignore[attr-defined]
            return {"type": "json_schema", "json_schema": {"name": structured_output.__name__, "schema": schema}}

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
        # Normalize system message by inserting/updating first system role
        if system_message:
            if messages and messages[0].get("role") == "system":
                messages = [{"role": "system", "content": system_message}] + messages[1:]
            else:
                messages = [{"role": "system", "content": system_message}] + messages

        try:
            logger.debug(
                "openai.chat start",
                extra={
                    "llm_provider": "openai",
                    "model": self.config.model,
                    "msg_count": len(messages),
                    "has_tools": bool(tools),
                    "structured": bool(structured_output),
                    "search_grounding": use_search_grounding,
                },
            )

            create_kwargs: Dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
            }
            if self.config.max_tokens is not None:
                create_kwargs["max_tokens"] = self.config.max_tokens

            # Tools (function calling)
            tool_param = self._build_tool_param(tools)
            if tool_param:
                create_kwargs["tools"] = tool_param

            # Grounding-equivalent: enable web/file search tools when requested
            # Note: Requires account access; we pass the tool spec per OpenAI docs when flag is set
            if use_search_grounding:
                grounding_tools: List[Dict[str, Any]] = []
                try:
                    grounding_tools.append({"type": "web_search"})
                except Exception:
                    pass
                try:
                    grounding_tools.append({"type": "file_search"})
                except Exception:
                    pass
                if grounding_tools:
                    # Merge with provided tools
                    if "tools" in create_kwargs and isinstance(create_kwargs["tools"], list):
                        create_kwargs["tools"] = [*create_kwargs["tools"], *grounding_tools]
                    else:
                        create_kwargs["tools"] = grounding_tools

            # Structured output via response_format
            resp_format = self._build_response_format(structured_output)
            if resp_format is not None:
                # Alibaba/DashScope has two quirks with structured output:
                # 1. Requires the word "json" in the messages when response_format is set
                # 2. Does not enforce json_schema strictly — use json_object + schema instruction
                if self.config.is_alibaba:
                    # Use simpler json_object format (Alibaba doesn't enforce json_schema)
                    create_kwargs["response_format"] = {"type": "json_object"}
                    # Build a schema hint to guide the model
                    schema_hint = ""
                    if structured_output is not None:
                        try:
                            schema = structured_output.model_json_schema()  # type: ignore[attr-defined]
                            props = schema.get("properties", {})
                            required = schema.get("required", list(props.keys()))
                            field_descs = ", ".join(
                                f'"{k}": {v.get("type", "string")}' for k, v in props.items()
                            )
                            schema_hint = (
                                f' Respond ONLY with a JSON object matching this schema: '
                                f'{{{field_descs}}}. Required fields: {required}.'
                            )
                        except Exception:
                            schema_hint = " Respond ONLY with a valid JSON object."
                    msgs = create_kwargs["messages"]
                    combined_text = " ".join(
                        (m.get("content") or "") for m in msgs if isinstance(m, dict)
                    ).lower()
                    if "json" not in combined_text or schema_hint:
                        instruction = f"Respond in JSON.{schema_hint}"
                        if msgs and msgs[0].get("role") == "system":
                            msgs = [
                                {"role": "system", "content": msgs[0]["content"] + " " + instruction},
                                *msgs[1:],
                            ]
                        else:
                            msgs = [{"role": "system", "content": instruction}] + msgs
                        create_kwargs["messages"] = msgs
                else:
                    create_kwargs["response_format"] = resp_format

            # Thinking mode — Alibaba/Qwen Chat Completions uses extra_body.
            # We always send enable_thinking explicitly so that thinking: false
            # actually disables CoT on models (e.g. qwen3.5-flash) that default
            # to thinking mode ON.
            use_thinking = thinking_enabled if thinking_enabled is not None else self.config.thinking_enabled
            if self.config.is_alibaba:
                extra_body: Dict[str, Any] = {"enable_thinking": bool(use_thinking)}
                if use_thinking and self.config.thinking_budget is not None:
                    extra_body["thinking_budget"] = self.config.thinking_budget
                create_kwargs["extra_body"] = extra_body

            # Call API
            start = time.perf_counter()
            completion = self._client.chat.completions.create(**create_kwargs)
            latency_ms = (time.perf_counter() - start) * 1000

            # Extract message
            choice = completion.choices[0] if getattr(completion, "choices", []) else None
            message = getattr(choice, "message", {}) if choice else {}
            content_text = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else None) or ""
            tool_calls = getattr(message, "tool_calls", None) or (message.get("tool_calls") if isinstance(message, dict) else None) or []
            usage = getattr(completion, "usage", {}) or {}

            # Log service usage to OpenTelemetry/OpenSearch (replaces Langfuse tracing)
            try:
                input_tokens = getattr(usage, "prompt_tokens", None) or (usage.get("prompt_tokens") if isinstance(usage, dict) else None)
                output_tokens = getattr(usage, "completion_tokens", None) or (usage.get("completion_tokens") if isinstance(usage, dict) else None)
                total_tokens = getattr(usage, "total_tokens", None) or (usage.get("total_tokens") if isinstance(usage, dict) else None)

                if total_tokens is None and input_tokens is not None and output_tokens is not None:
                    total_tokens = input_tokens + output_tokens

                tokens_per_second = None
                if total_tokens is not None and latency_ms > 0:
                    tokens_per_second = (total_tokens / latency_ms) * 1000

                if isinstance(usage, dict):
                    usage["latency_ms"] = latency_ms
                    if tokens_per_second is not None:
                        usage["tokens_per_second"] = tokens_per_second

                trace_metadata = {
                    "gen_ai.system": "openai",
                    "gen_ai.request.model": self.config.model,
                    "gen_ai.usage.input_tokens": input_tokens,
                    "gen_ai.usage.output_tokens": output_tokens,
                    "tokens.total": total_tokens,
                    "gen_ai.response.latency_ms": latency_ms,
                    "latency_ms": latency_ms,
                    "tokens_per_second": tokens_per_second,
                    "features.structured_output": str(bool(structured_output)).lower(),
                    "features.tools": str(bool(tools)).lower(),
                    "features.search_grounding": str(use_search_grounding).lower(),
                }
                add_trace_metadata({k: v for k, v in trace_metadata.items() if v is not None})
                
                log_llm_usage(
                    provider="openai",
                    model=self.config.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    structured=bool(structured_output),
                    has_tools=bool(tools),
                    search_grounding=use_search_grounding,
                    host=self.config.azure_endpoint or self.config.base_url or "https://api.openai.com",
                )
            except Exception as e:
                # Service usage logging should never break the call
                logger.warning(f"Failed to log LLM usage: {e}")

            # If structured_output requested, attempt to validate
            if resp_format is not None and structured_output is not None:
                import json as _json
                from ..json_parser import parse_structured_output as _parse_structured_output

                # Attempt 1: native SDK validation
                parsed_dict: Any = None
                try:
                    validated = structured_output.model_validate_json(content_text)  # type: ignore[attr-defined]
                    parsed_dict = validated.model_dump()
                except Exception:
                    pass

                # Attempt 2: comprehensive fuzzy recovery (handles markdown fences,
                # schema-echo, schema-as-instance, key normalisation, nested dicts,
                # Literal/enum coercion).
                if parsed_dict is None:
                    parsed_dict = _parse_structured_output(content_text, structured_output) if content_text else None

                if parsed_dict is not None:
                    return {
                        "content": parsed_dict,
                        "structured": True,
                        "tool_calls": tool_calls or [],
                        "usage": usage,
                        "text": content_text,
                        "content_json": _json.dumps(parsed_dict, ensure_ascii=False),
                    }

                # All recovery failed – return unstructured so callers can decide.
                logger.warning(
                    "openai: structured output could not be validated against %s; "
                    "falling back to unstructured text",
                    structured_output.__name__,
                )
                return {
                    "content": content_text,
                    "structured": False,
                    "tool_calls": tool_calls or [],
                    "usage": usage,
                    "text": content_text,
                }

            # Plain text
            return {
                "content": content_text,
                "structured": False,
                "tool_calls": tool_calls or [],
                "usage": usage,
            }
        except Exception as e:  # pragma: no cover - network errors
            logger.exception("openai.chat failed")
            return {
                "error": str(e),
                "content": None,
                "structured": structured_output is not None,
                "tool_calls": [],
                "usage": {},
            }
