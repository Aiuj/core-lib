"""Ollama provider using the official ollama Python library.

Supports native tools (function calling) and simple structured outputs via
format='json'.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type
import time

from pydantic import BaseModel

from .base import BaseProvider
from ..llm_config import LLMConfig
from dataclasses import dataclass

from core_lib import get_module_logger
from core_lib.api_utils.wake_on_lan import WakeOnLanStrategy
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
    thinking_config: Optional[Dict[str, Any]] = None
    wake_on_lan: Optional[Dict[str, Any]] = None

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
        thinking_config: Optional[Dict[str, Any]] = None,
        wake_on_lan: Optional[Dict[str, Any]] = None,
    ):
        super().__init__("ollama", model, temperature, max_tokens, thinking_enabled)
        self.base_url = base_url
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.repeat_penalty = repeat_penalty
        self.top_k = top_k
        self.top_p = top_p
        self.thinking_config = dict(thinking_config) if thinking_config else None
        self.wake_on_lan = dict(wake_on_lan) if wake_on_lan else None

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        import os

        max_tokens_env = os.getenv("OLLAMA_MAX_TOKENS")
        num_ctx_env = os.getenv("OLLAMA_NUM_CTX")
        num_predict_env = os.getenv("OLLAMA_NUM_PREDICT")
        repeat_penalty_env = os.getenv("OLLAMA_REPEAT_PENALTY")
        top_k_env = os.getenv("OLLAMA_TOP_K")
        top_p_env = os.getenv("OLLAMA_TOP_P")
        thinking_level = os.getenv("OLLAMA_THINKING_LEVEL")
        thinking_budget_env = os.getenv("OLLAMA_THINKING_BUDGET")

        thinking_config: Optional[Dict[str, Any]] = None
        if thinking_level is not None or thinking_budget_env is not None:
            thinking_config = {}
            if thinking_level is not None:
                thinking_config["level"] = str(thinking_level).lower()
            if thinking_budget_env is not None:
                thinking_config["budget"] = int(thinking_budget_env)

        return cls(
            model=os.getenv("OLLAMA_MODEL", "qwen3:1.7b"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
            max_tokens=int(max_tokens_env) if max_tokens_env is not None else None,
            thinking_enabled=os.getenv("OLLAMA_THINKING_ENABLED", "false").lower() == "true",
            thinking_config=thinking_config,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "60")),
            num_ctx=int(num_ctx_env) if num_ctx_env is not None else None,
            num_predict=int(num_predict_env) if num_predict_env is not None else None,
            repeat_penalty=float(repeat_penalty_env) if repeat_penalty_env is not None else None,
            top_k=int(top_k_env) if top_k_env is not None else None,
            top_p=float(top_p_env) if top_p_env is not None else None,
        )

class OllamaProvider(BaseProvider):
    """Provider implementation for Ollama (local models)."""

    _THINKING_MODEL_HINTS = (
        "deepseek-r1",
        "qwen3",
    )

    def __init__(self, config: OllamaConfig) -> None:  # type: ignore[override]
        super().__init__(config)
        # Narrow the config type so mypy can see Ollama-specific fields like base_url.
        self.config: OllamaConfig = config
        import ollama  # type: ignore

        self._ollama = ollama
        self._wake_on_lan = WakeOnLanStrategy(self.config.wake_on_lan)

    def _is_connection_or_timeout_error(self, error: Exception) -> bool:
        """Return True when error indicates host may be sleeping/unreachable."""
        error_type = type(error).__name__.lower()
        if any(name in error_type for name in ("timeout", "connecterror", "connectionerror")):
            return True

        error_str = str(error).lower()
        indicators = (
            "connection refused",
            "failed to connect",
            "could not connect",
            "network is unreachable",
            "timed out",
            "timeout",
            "name or service not known",
            "nodename nor servname",
        )
        return any(token in error_str for token in indicators)

    def _chat_once(self, payload: Dict[str, Any], timeout: Optional[float]) -> Dict[str, Any]:
        """Execute one Ollama chat call with optional timeout override."""
        if getattr(self.config, "base_url", None):
            from ollama import Client  # type: ignore

            client_kwargs: Dict[str, Any] = {"host": self.config.base_url}
            if timeout is not None:
                client_kwargs["timeout"] = timeout

            client = Client(**client_kwargs)
            return client.chat(**payload)

        # Fallback path when no host is configured
        if timeout is not None:
            from ollama import Client  # type: ignore
            client = Client(timeout=timeout)
            return client.chat(**payload)

        return self._ollama.chat(**payload)

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

    def _supports_thinking(self) -> bool:
        model_lc = (self.config.model or "").lower()
        return any(hint in model_lc for hint in self._THINKING_MODEL_HINTS)

    def _resolve_think_flag(self, thinking_enabled_override: Optional[bool]) -> Optional[bool]:
        cfg_thinking = getattr(self.config, "thinking_config", None) or {}
        if not isinstance(cfg_thinking, dict):
            cfg_thinking = {}

        disable_levels = {"off", "none", "disabled", "disable", "0"}
        level_raw = cfg_thinking.get("level")
        level = str(level_raw).lower().strip() if level_raw is not None else None

        budget_raw = cfg_thinking.get("budget")
        budget: Optional[int] = None
        if budget_raw is not None:
            try:
                budget = int(budget_raw)
            except Exception:
                budget = None

        if thinking_enabled_override is not None:
            return bool(thinking_enabled_override)
        if "enabled" in cfg_thinking:
            return bool(cfg_thinking.get("enabled"))
        if level in disable_levels:
            return False
        if budget is not None:
            return budget > 0
        if level is not None:
            return True
        return bool(getattr(self.config, "thinking_enabled", False))

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
                resp_format = "json"
                try:
                    payload["format"] = structured_output.model_json_schema()
                except Exception:
                    payload["format"] = "json"

            # Thinking support per https://ollama.com/blog/thinking
            # Ollama Python/HTTP uses top-level `think: bool`.
            think_flag = self._resolve_think_flag(thinking_enabled)
            if think_flag is not None and self._supports_thinking():
                try:
                    payload["think"] = bool(think_flag)
                except Exception:
                    pass

            # Execute API call with latency measurement
            start = time.perf_counter()
            base_url_for_wol = self.config.base_url or ""
            default_timeout = float(self.config.timeout)
            effective_timeout = self._wake_on_lan.maybe_get_initial_timeout(
                base_url_for_wol,
                default_timeout,
            )

            try:
                resp = self._chat_once(payload, effective_timeout)
            except Exception as first_error:
                if self._is_connection_or_timeout_error(first_error):
                    wake_result = self._wake_on_lan.maybe_wake(base_url_for_wol, first_error)
                    if wake_result.succeeded:
                        retry_timeout = wake_result.retry_timeout_seconds or default_timeout
                        logger.info(
                            f"Retrying Ollama request after WoL wake with timeout={retry_timeout}s"
                        )
                        resp = self._chat_once(payload, retry_timeout)
                    else:
                        raise
                else:
                    raise
            latency_ms = (time.perf_counter() - start) * 1000

            message = resp.get("message", {})
            content_text = message.get("content", "")
            thinking_text = message.get("thinking")
            tool_calls = message.get("tool_calls", []) or []

            usage = resp.get("usage", {}) or {}
            if not usage:
                usage = {
                    "prompt_tokens": resp.get("prompt_eval_count"),
                    "completion_tokens": resp.get("eval_count"),
                    "total_tokens": resp.get("total_tokens"),
                }

            input_tokens = usage.get("prompt_tokens") or usage.get("prompt_eval_count")
            output_tokens = usage.get("completion_tokens") or usage.get("eval_count")
            total_tokens = usage.get("total_tokens")
            if total_tokens is None and input_tokens and output_tokens:
                total_tokens = input_tokens + output_tokens

            tokens_per_second = None
            if total_tokens is not None and latency_ms > 0:
                tokens_per_second = (total_tokens / latency_ms) * 1000

            if isinstance(usage, dict):
                usage.setdefault("latency_ms", latency_ms)
                if tokens_per_second is not None:
                    usage.setdefault("tokens_per_second", tokens_per_second)

            try:
                log_llm_usage(
                    provider="ollama",
                    model=self.config.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    structured=bool(structured_output),
                    has_tools=bool(tools),
                    search_grounding=use_search_grounding,
                    host=self.config.base_url,
                )
            except Exception as e:
                logger.warning(f"Failed to log LLM usage: {e}")

            if resp_format is not None and structured_output is not None:
                import json as _json
                from ..json_parser import parse_structured_output, _strip_markdown_code_block

                # Use parse_structured_output which handles:
                # 1. Markdown code-block wrappers (```json ... ```)
                # 2. Schema-as-instance: model returns JSON Schema structure with
                #    actual values inside "properties" instead of a flat dict
                clean_text = _strip_markdown_code_block(content_text) if content_text else ""
                parsed = parse_structured_output(clean_text, structured_output) if clean_text else None
                if parsed is not None:
                    content: Any = parsed
                else:
                    # Last-resort raw fallback so the caller always gets something
                    try:
                        content = _json.loads(clean_text) if clean_text else {}
                    except Exception:
                        content = {"_raw": content_text}
                    logger.warning(
                        "ollama structured output could not be validated against %s; "
                        "returning raw parse result",
                        structured_output.__name__,
                    )
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
                "thinking": thinking_text,
            }
        except Exception as e:  # pragma: no cover - runtime connectivity
            if self._is_connection_or_timeout_error(e):
                logger.warning(
                    "ollama.chat connectivity failure (handled): %s",
                    e,
                )
            else:
                logger.exception("ollama.chat failed")

            try:
                log_llm_usage(
                    provider="ollama",
                    model=self.config.model,
                    structured=bool(structured_output),
                    has_tools=bool(tools),
                    error=str(e),
                    host=self.config.base_url,
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
