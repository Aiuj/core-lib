"""Ollama provider using the official ollama Python library.

Supports native tools (function calling) and simple structured outputs via
format='json'.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from .base import BaseProvider
from ..llm_config import LLMConfig
from dataclasses import dataclass
from typing import Optional

from core_lib import get_module_logger
from core_lib.tracing.service_usage import log_llm_usage

logger = get_module_logger()

@dataclass
class OllamaConfig(LLMConfig):
    base_url: str = "http://localhost:11434"
    timeout: int = 60
    num_ctx: Optional[int] = None
    num_predict: Optional[int] = None
    repeat_penalty: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    def __init__(
        self,
        model: str = "qwen3:1.7b",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        thinking_enabled: bool = False,
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        super().__init__("ollama", model, temperature, max_tokens, thinking_enabled)
        self.base_url = base_url
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.repeat_penalty = repeat_penalty
        self.top_k = top_k
        self.top_p = top_p

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        import os

        return cls(
            model=os.getenv("OLLAMA_MODEL", "qwen3:1.7b"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS")) if os.getenv("OLLAMA_MAX_TOKENS") else None,
            thinking_enabled=os.getenv("OLLAMA_THINKING_ENABLED", "false").lower() == "true",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "60")),
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX")) if os.getenv("OLLAMA_NUM_CTX") else None,
            num_predict=int(os.getenv("OLLAMA_NUM_PREDICT")) if os.getenv("OLLAMA_NUM_PREDICT") else None,
            repeat_penalty=float(os.getenv("OLLAMA_REPEAT_PENALTY")) if os.getenv("OLLAMA_REPEAT_PENALTY") else None,
            top_k=int(os.getenv("OLLAMA_TOP_K")) if os.getenv("OLLAMA_TOP_K") else None,
            top_p=float(os.getenv("OLLAMA_TOP_P")) if os.getenv("OLLAMA_TOP_P") else None,
        )

class OllamaProvider(BaseProvider):
    """Provider implementation for Ollama (local models)."""

    def __init__(self, config: OllamaConfig) -> None:  # type: ignore[override]
        super().__init__(config)
        import ollama  # type: ignore

        self._ollama = ollama

    def _build_options(self) -> Dict[str, Any]:
        # Map config to ollama options when available
        options: Dict[str, Any] = {
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            options["num_predict"] = self.config.max_tokens
        if self.config.num_ctx is not None:
            options["num_ctx"] = self.config.num_ctx
        if self.config.num_predict is not None:
            options["num_predict"] = self.config.num_predict
        if self.config.repeat_penalty is not None:
            options["repeat_penalty"] = self.config.repeat_penalty
        if self.config.top_k is not None:
            options["top_k"] = self.config.top_k
        if self.config.top_p is not None:
            options["top_p"] = self.config.top_p
        return options

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
        try:
            logger.debug(
                "ollama.chat start",
                extra={
                    "llm_provider": "ollama",
                    "model": self.config.model,
                    "msg_count": len(messages),
                    "has_tools": bool(tools),
                    "structured": bool(structured_output),
                    "search_grounding": use_search_grounding,
                },
            )
            payload: Dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "options": self._build_options(),
            }
            if tools:
                payload["tools"] = tools

            resp_format: Optional[str] = None
            if structured_output is not None:
                # Ollama supports format='json'. We'll validate with Pydantic if provided.
                resp_format = "json"
                # Some Ollama models accept a JSON schema directly; others use format='json'.
                try:
                    payload["format"] = structured_output.model_json_schema()
                except Exception:
                    payload["format"] = "json"

            # Thinking support per https://ollama.com/blog/thinking
            # Two mechanisms:
            # 1) If the selected model is a "think" model (e.g., llama3.1:8b-instruct-fp16-think), thoughts may be produced automatically.
            # 2) When supported, pass options.thinking: { type: "enabled" } to enable chain-of-thought style output.
            if thinking_enabled is True:
                try:
                    opts = payload.get("options", {}) or {}
                    # Newer API supports a nested thinking config
                    if isinstance(opts, dict):
                        # Conservative defaults: we only signal that thinking is enabled; providers decide budget
                        opts["thinking"] = opts.get("thinking", {"type": "enabled"})
                        payload["options"] = opts
                except Exception:
                    pass

            # Configure host via custom client if base_url differs
            if getattr(self.config, "base_url", None):
                from ollama import Client  # type: ignore

                client = Client(host=self.config.base_url)
                resp = client.chat(**payload)
            else:
                resp = self._ollama.chat(**payload)

            message = resp.get("message", {})
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", []) or []
            usage = resp.get("usage", {}) or {}
            
            # Log service usage to OpenTelemetry/OpenSearch
            try:
                input_tokens = usage.get("prompt_tokens") or usage.get("prompt_eval_count")
                output_tokens = usage.get("completion_tokens") or usage.get("eval_count")
                total_tokens = usage.get("total_tokens")
                if total_tokens is None and input_tokens and output_tokens:
                    total_tokens = input_tokens + output_tokens
                
                log_llm_usage(
                    provider="ollama",
                    model=self.config.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    structured=bool(structured_output),
                    has_tools=bool(tools),
                    search_grounding=use_search_grounding,
                )
            except Exception as e:
                logger.warning(f"Failed to log LLM usage: {e}")

            # If structured_output requested, attempt to validate
            if resp_format is not None and structured_output is not None:
                try:
                    data = structured_output.model_validate_json(content_text)  # type: ignore[attr-defined]
                    content: Any = data.model_dump()
                except Exception:
                    import json as _json

                    try:
                        content = _json.loads(content_text) if content_text else {}
                    except Exception:
                        content = {"_raw": content_text}
                import json as _json
                return {
                    "content": content,
                    "structured": True,
                    "tool_calls": tool_calls or [],
                    "usage": usage,
                    "text": content_text,
                    "content_json": _json.dumps(content, ensure_ascii=False),
                }

            return {
                "content": content_text,
                "structured": False,
                "tool_calls": tool_calls,
                "usage": usage,
            }
        except Exception as e:  # pragma: no cover - runtime connectivity
            logger.exception("ollama.chat failed")
            
            # Log error to OpenTelemetry/OpenSearch
            try:
                log_llm_usage(
                    provider="ollama",
                    model=self.config.model,
                    structured=bool(structured_output),
                    has_tools=bool(tools),
                    error=str(e),
                )
            except Exception:
                pass
            
            return {
                "error": str(e),
                "content": None,
                "structured": structured_output is not None,
                "tool_calls": [],
                "usage": {},
            }
