"""Ollama provider using the official ollama Python library.

Supports native tools (function calling) and simple structured outputs via
format='json'.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Union
import time
import re
import json

from pydantic import BaseModel

from .base import BaseProvider, normalize_tool_calls, parse_text_tool_calls
from ..llm_config import LLMConfig
from ..json_parser import augment_prompt_for_json
from dataclasses import dataclass

from core_lib import get_module_logger
from core_lib.api_utils.wake_on_lan import WakeOnLanStrategy
from core_lib.llm.provider_health import classify_error
from core_lib.tracing.service_usage import log_llm_usage
from core_lib.tracing.payload_capture import capture_llm_payload

logger = get_module_logger()

# Conservative, provider-agnostic estimate (~4 chars/token) used only to size
# the dynamic timeout below -- not an exact count.
_CHARS_PER_TOKEN_ESTIMATE = 4

# Local Ollama models run on modest hardware (CPU or a small GPU) and are far
# slower than hosted APIs. These throughput assumptions are deliberately
# conservative so the computed timeout has headroom rather than clipping a
# slow-but-successful generation.
_PREFILL_TOKENS_PER_SECOND = 200.0
_GENERATION_TOKENS_PER_SECOND = 15.0
_DEFAULT_EXPECTED_OUTPUT_TOKENS = 512
_DEFAULT_THINKING_BUDGET_TOKENS = 1024
_TIMEOUT_FLOOR_SECONDS = 30.0
_TIMEOUT_SAFETY_MARGIN_SECONDS = 15.0

# Levels Ollama's `think` field accepts as a string (in addition to a plain
# bool) for models that support graduated reasoning effort -- see
# https://docs.ollama.com/capabilities/thinking. Sent as-is when configured;
# retried as a boolean if the server rejects the string for this model.
_THINK_LEVEL_STRINGS = {"low", "medium", "high", "max"}

@dataclass
class OllamaConfig(LLMConfig):
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
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
        api_key: Optional[str] = None,
        timeout: int = 60,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        thinking_config: Optional[Dict[str, Any]] = None,
        wake_on_lan: Optional[Dict[str, Any]] = None,
        payload_capture: Optional[bool] = None,
    ):
        super().__init__("ollama", model, temperature, max_tokens, thinking_enabled, payload_capture)
        self.base_url = base_url
        self.api_key = api_key or None
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
            base_url=os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434")),
            api_key=os.getenv("OLLAMA_API_KEY") or None,
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
        "granite-4.2",
        "granite4.2",
    )

    def __init__(self, config: OllamaConfig) -> None:  # type: ignore[override]
        super().__init__(config)
        # Narrow the config type so mypy can see Ollama-specific fields like base_url.
        self.config: OllamaConfig = config
        import ollama  # type: ignore

        self._ollama = ollama
        self._wake_on_lan = WakeOnLanStrategy(self.config.wake_on_lan)

    def is_in_warmup(self) -> bool:
        """Return True while a non-blocking WoL warmup window is active."""
        return self._wake_on_lan.is_in_warmup(self.config.base_url or "")

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

    def _is_model_not_found_error(self, error: Exception) -> bool:
        """Return True when Ollama reports a missing model (typically HTTP 404)."""
        status_code = getattr(error, "status_code", None)
        if status_code is None:
            response = getattr(error, "response", None)
            status_code = getattr(response, "status_code", None)

        error_type = type(error).__name__.lower()
        error_str = str(error).lower()

        has_missing_model_text = "model" in error_str and "not found" in error_str
        if status_code == 404 and has_missing_model_text:
            return True

        return "responseerror" in error_type and has_missing_model_text

    @staticmethod
    def _is_schema_grammar_error(error: Exception) -> bool:
        """Return True when Ollama cannot compile a JSON Schema into grammar.

        Some Ollama/model combinations reject otherwise valid Pydantic JSON
        Schemas (for example schemas containing nested references).  Retrying
        with Ollama's JSON-only format still leaves the normal Pydantic output
        validation in place, without disabling structured output globally.
        """
        error_text = str(error).lower()
        return (
            "failed to initialize samplers" in error_text
            and "failed to parse grammar" in error_text
        )

    def _extract_missing_model_name(self, error: Exception) -> str:
        """Best-effort extraction of missing model name from Ollama error text."""
        message = str(error)
        match = re.search(r"model\s+['\"]([^'\"]+)['\"]\s+not\s+found", message, re.IGNORECASE)
        if match:
            return match.group(1)
        return self.config.model

    @staticmethod
    def _normalize_tool_calls_for_ollama(
        tool_calls: Any,
    ) -> Any:
        """Normalize outbound tool calls to Ollama's expected schema.

        Ollama's Pydantic `Message` model expects:
          tool_calls[].function.arguments -> dict

        Some upstream callers (OpenAI-style) store arguments as a JSON string.
        Convert those strings to dicts when possible to avoid payload validation
        errors in ollama.Client.chat().
        """
        if not isinstance(tool_calls, list):
            return tool_calls

        normalized: List[Dict[str, Any]] = []
        for entry in tool_calls:
            if not isinstance(entry, dict):
                normalized.append(entry)
                continue

            entry_copy: Dict[str, Any] = dict(entry)
            function_obj = entry_copy.get("function")
            if isinstance(function_obj, dict):
                function_copy = dict(function_obj)
                arguments = function_copy.get("arguments")
                if isinstance(arguments, str):
                    stripped = arguments.strip()
                    if stripped:
                        try:
                            parsed = json.loads(stripped)
                            if isinstance(parsed, dict):
                                function_copy["arguments"] = parsed
                        except Exception:
                            # Leave as-is when parsing fails; better to preserve
                            # original value than silently mutate to wrong shape.
                            pass
                entry_copy["function"] = function_copy

            normalized.append(entry_copy)

        return normalized

    @staticmethod
    def _convert_messages_to_ollama_format(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI-style multimodal messages to Ollama's native format.

        Ollama's Python client (and HTTP API) requires:
        - ``content``: plain string (text only)
        - ``images``: list of base64-encoded image data (no data-URL prefix)

        OpenAI-style multimodal messages use a ``content`` list with typed parts
        (``{"type": "image_url", "image_url": {"url": "data:...;base64,..."}}``,
        ``{"type": "text", "text": "..."}``).  This method normalises the latter
        into the former so that multimodal calls work with any Ollama vision model.
        """
        import base64 as _base64

        result: List[Dict[str, Any]] = []
        for msg in messages:
            msg_for_ollama: Dict[str, Any] = dict(msg)
            if "tool_calls" in msg_for_ollama:
                msg_for_ollama["tool_calls"] = OllamaProvider._normalize_tool_calls_for_ollama(
                    msg_for_ollama.get("tool_calls")
                )

            content = msg.get("content")
            if not isinstance(content, list):
                result.append(msg_for_ollama)
                continue

            # OpenAI multimodal format — extract text parts and image parts
            text_parts: List[str] = []
            images: List[str] = []

            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text":
                    text_parts.append(part.get("text", ""))
                elif part_type == "image_url":
                    url: str = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:"):
                        # Strip "data:<mime>;base64," prefix — Ollama only wants raw b64
                        try:
                            b64_data = url.split(",", 1)[1]
                        except IndexError:
                            b64_data = url
                    else:
                        # For security and latency reasons, we do not fetch remote URLs here.
                        # Only data: URLs are supported; skip any other image URLs.
                        logger.warning(
                            "Skipping non-data image URL in Ollama provider: %s",
                            url,
                        )
                        continue
                    images.append(b64_data)

            new_msg: Dict[str, Any] = {
                **{k: v for k, v in msg_for_ollama.items() if k != "content"},
                "content": " ".join(text_parts),
            }
            if images:
                new_msg["images"] = images
            result.append(new_msg)

        return result

    def _chat_once(self, payload: Dict[str, Any], timeout: Optional[float]) -> Dict[str, Any]:
        """Execute one Ollama chat call with optional timeout override."""
        from ollama import Client  # type: ignore

        host = getattr(self.config, "base_url", None) or "http://localhost:11434"
        client_kwargs: Dict[str, Any] = {"host": host}
        if timeout is not None:
            client_kwargs["timeout"] = timeout

        api_key = getattr(self.config, "api_key", None)
        if api_key:
            client_kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}

        client = Client(**client_kwargs)
        return client.chat(**payload)

    def _build_options(
        self,
        think_value: Optional[Union[bool, str]] = None,
        expected_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
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

        # When thinking is enabled and nothing already caps generation length
        # (max_tokens/num_predict), bound it to the configured thinking budget
        # plus expected output so a model can't reason indefinitely and blow
        # past the client timeout. Thinking tokens share the same num_predict
        # budget as the final answer in Ollama.
        if think_value and "num_predict" not in options:
            cfg_thinking = getattr(self.config, "thinking_config", None) or {}
            budget = cfg_thinking.get("budget") if isinstance(cfg_thinking, dict) else None
            try:
                thinking_budget = int(budget) if budget is not None else _DEFAULT_THINKING_BUDGET_TOKENS
            except Exception:
                thinking_budget = _DEFAULT_THINKING_BUDGET_TOKENS
            output_budget = expected_output_tokens or _DEFAULT_EXPECTED_OUTPUT_TOKENS
            options["num_predict"] = thinking_budget + output_budget

        return options

    def _supports_thinking(self) -> bool:
        model_lc = (self.config.model or "").lower()
        return any(hint in model_lc for hint in self._THINKING_MODEL_HINTS)

    def _resolve_think_value(
        self, thinking_enabled_override: Optional[bool]
    ) -> Optional[Union[bool, str]]:
        """Resolve the value to send as Ollama's `think` field.

        Returns ``True``/``False`` for on/off, or one of ``_THINK_LEVEL_STRINGS``
        when the config specifies a graduated level (e.g. "low") and thinking is
        enabled -- some models (gpt-oss requires it; Qwen3/DeepSeek/Granite
        accept it optionally) use this to bound reasoning-trace length instead
        of just an on/off switch. Previously this level was parsed only to
        decide the boolean on/off state and then discarded, so a configured
        "low" budget had no effect on Ollama and the model reasoned at full,
        unbounded depth -- a likely source of local-model timeouts.
        """
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
            enabled = bool(thinking_enabled_override)
        elif "enabled" in cfg_thinking:
            enabled = bool(cfg_thinking.get("enabled"))
        elif level in disable_levels:
            enabled = False
        elif budget is not None:
            enabled = budget > 0
        elif level is not None:
            enabled = True
        else:
            enabled = bool(getattr(self.config, "thinking_enabled", False))

        if not enabled:
            return False
        if level in _THINK_LEVEL_STRINGS:
            return level
        return True

    @staticmethod
    def _is_unsupported_think_value_error(error: Exception) -> bool:
        """Return True when Ollama rejected a string `think` level for this model."""
        error_text = str(error).lower()
        return "think" in error_text and (
            "invalid" in error_text or "unsupported" in error_text or "unknown" in error_text
        )

    def _estimate_prompt_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total_chars = 0
        for message in messages or []:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        total_chars += len(part["text"])
        return total_chars // _CHARS_PER_TOKEN_ESTIMATE

    def _compute_dynamic_timeout(
        self,
        messages: List[Dict[str, Any]],
        think_value: Optional[Union[bool, str]],
        expected_output_tokens: Optional[int],
    ) -> float:
        """Size the request timeout to the work being asked of a local model.

        The configured `timeout` is treated as a floor, not a ceiling: a large
        prompt and/or thinking mode can legitimately take much longer than a
        short interactive chat, so we scale up from conservative local
        throughput assumptions rather than failing a slow-but-successful call.
        """
        prompt_tokens = self._estimate_prompt_tokens(messages)
        prefill_seconds = prompt_tokens / _PREFILL_TOKENS_PER_SECOND

        output_tokens = expected_output_tokens
        if output_tokens is None:
            output_tokens = self.config.num_predict or self.config.max_tokens or _DEFAULT_EXPECTED_OUTPUT_TOKENS

        thinking_tokens = 0
        if think_value:
            cfg_thinking = getattr(self.config, "thinking_config", None) or {}
            budget = cfg_thinking.get("budget") if isinstance(cfg_thinking, dict) else None
            try:
                thinking_tokens = int(budget) if budget is not None else _DEFAULT_THINKING_BUDGET_TOKENS
            except Exception:
                thinking_tokens = _DEFAULT_THINKING_BUDGET_TOKENS

        generation_seconds = (output_tokens + thinking_tokens) / _GENERATION_TOKENS_PER_SECOND

        computed = (
            _TIMEOUT_FLOOR_SECONDS
            + prefill_seconds
            + generation_seconds
            + _TIMEOUT_SAFETY_MARGIN_SECONDS
        )
        return max(float(self.config.timeout), computed)

    def chat(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        structured_output: Optional[Type[BaseModel]] = None,
        system_message: Optional[str] = None,
        use_search_grounding: bool = False,
        thinking_enabled: Optional[bool] = None,
        expected_output_tokens: Optional[int] = None,
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

            # Thinking support per https://docs.ollama.com/capabilities/thinking
            # `think` is a bool for most models; gpt-oss requires (and
            # Qwen3/DeepSeek/Granite optionally accept) a "low"/"medium"/"high"
            # level string to bound reasoning-trace length instead of just
            # on/off. When think_value is truthy, only set it if the model is
            # known to support thinking (avoid sending unsupported params to
            # non-thinking models). When False, always send it explicitly so
            # that models capable of thinking (but not in our hints list) are
            # told to disable it.
            think_value = self._resolve_think_value(thinking_enabled)
            model_supports_thinking = self._supports_thinking()

            payload: Dict[str, Any] = {
                "model": self.config.model,
                "messages": self._convert_messages_to_ollama_format(messages),
                "options": self._build_options(think_value, expected_output_tokens),
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

                # Augment the last user message with a compact JSON template so
                # the model knows *what* to produce.  The Ollama `format` param
                # constrains syntax but many models still need the expected
                # structure described in the prompt to reliably fill it in.
                # See https://docs.ollama.com/capabilities/structured-outputs
                augmented_messages = list(payload["messages"])
                for i in range(len(augmented_messages) - 1, -1, -1):
                    if augmented_messages[i].get("role") == "user":
                        augmented_messages[i] = {
                            **augmented_messages[i],
                            "content": augment_prompt_for_json(
                                augmented_messages[i].get("content", ""),
                                structured_output,
                            ),
                        }
                        break
                payload["messages"] = augmented_messages

            if think_value and model_supports_thinking:
                payload["think"] = think_value
            elif think_value is False:
                payload["think"] = False

            # Execute API call with latency measurement
            start = time.perf_counter()
            base_url_for_wol = self.config.base_url or ""
            default_timeout = self._compute_dynamic_timeout(
                payload["messages"], think_value, expected_output_tokens
            )
            effective_timeout = self._wake_on_lan.maybe_get_initial_timeout(
                base_url_for_wol,
                default_timeout,
            )

            try:
                resp = self._chat_once(payload, effective_timeout)
            except Exception as first_error:
                if (
                    isinstance(payload.get("think"), str)
                    and self._is_unsupported_think_value_error(first_error)
                ):
                    # This model doesn't accept a string think level -- fall
                    # back to a plain boolean and retry once.
                    logger.warning(
                        "ollama rejected think level %r for %s; retrying with think=True",
                        payload.get("think"),
                        self.config.model,
                    )
                    bool_payload = {**payload, "think": True}
                    resp = self._chat_once(bool_payload, effective_timeout)
                elif (
                    structured_output is not None
                    and isinstance(payload.get("format"), dict)
                    and self._is_schema_grammar_error(first_error)
                ):
                    # Keep structured parsing/validation below, but use the
                    # broadly supported JSON mode when this server cannot
                    # compile the Pydantic schema into a grammar.
                    logger.warning(
                        "ollama rejected structured-output grammar; retrying with format='json'"
                    )
                    json_only_payload = {**payload, "format": "json"}
                    resp = self._chat_once(json_only_payload, effective_timeout)
                elif self._is_connection_or_timeout_error(first_error):
                    wake_result = self._wake_on_lan.maybe_wake(base_url_for_wol, first_error)
                    if wake_result.succeeded:
                        if wake_result.warmup_seconds:
                            # Non-blocking mode: WoL packet was sent, don't wait here.
                            # Re-raise so FallbackLLMClient can route to a secondary
                            # provider immediately while the main server powers on.
                            logger.info(
                                f"WoL sent to {base_url_for_wol} (non-blocking) — "
                                f"routing to secondary for {wake_result.warmup_seconds:.0f}s warmup"
                            )
                            raise
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
            tool_calls = normalize_tool_calls(message.get("tool_calls", []) or [])

            # Debug: log raw Ollama response to diagnose tool-call recovery
            if tools:
                logger.debug(
                    "ollama chat response: content_len=%d tool_calls=%d has_markup=%s",
                    len(content_text or ""),
                    len(tool_calls),
                    "<tool_call>" in (content_text or ""),
                )
                if content_text and len(content_text) < 500:
                    logger.debug("ollama raw content: %s", content_text[:200])

            # Fallback: some models emit tool calls as text blocks in content
            # instead of populating message.tool_calls. Recover these when tools
            # were provided and structured tool calls are absent.
            if not tool_calls and tools and isinstance(content_text, str) and "<tool_call>" in content_text:
                logger.info("ollama: attempting to parse text-based tool calls from content")
                text_tool_calls, content_text = parse_text_tool_calls(content_text)
                if text_tool_calls:
                    tool_calls = text_tool_calls
                    logger.info(
                        "ollama: successfully parsed %d text-based tool call(s) from content",
                        len(tool_calls),
                    )
                else:
                    logger.warning("ollama: found <tool_call> markup but parse_text_tool_calls returned empty")

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
                call_id = log_llm_usage(
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
                    thinking_enabled=bool(think_value) if think_value is not None else None,
                    thinking_level=think_value if isinstance(think_value, str) else None,
                    response_format="structured" if structured_output is not None else "text",
                )
                capture_llm_payload(
                    call_id=call_id,
                    provider="ollama",
                    model=self.config.model,
                    messages=messages,
                    response_text=content_text,
                    force_enabled=self.config.payload_capture,
                )
            except Exception as e:
                logger.warning(f"Failed to log LLM usage: {e}")

            if resp_format is not None and structured_output is not None:
                import json as _json
                from ..json_parser import parse_structured_output, _strip_markdown_code_block

                # Use parse_structured_output which handles:
                # 1. Markdown code-block wrappers (```json ... ```)
                # 2. Schema-as-instance: model echoes JSON Schema structure with
                #    actual values inside "properties" instead of a plain instance
                clean_text = _strip_markdown_code_block(content_text) if content_text else ""
                parsed = parse_structured_output(clean_text, structured_output) if clean_text else None

                if parsed is not None:
                    # Successfully extracted a valid structured instance.
                    # Return structured=True so callers can trust content is a dict.
                    return {
                        "content": parsed,
                        "structured": True,
                        "tool_calls": tool_calls or [],
                        "usage": usage,
                        "text": content_text,
                        "content_json": _json.dumps(parsed, ensure_ascii=False, default=str),
                    }
                else:
                    # Could not validate the model output against the schema even
                    # after all recovery attempts.  Check if the model returned its
                    # own schema definition — if so, clear the text so callers do not
                    # surface the schema JSON as an answer.
                    from ..json_parser import extract_json_from_text, _is_pydantic_schema_echo
                    try:
                        raw_json = extract_json_from_text(content_text) if content_text else None
                        if isinstance(raw_json, dict) and _is_pydantic_schema_echo(
                            raw_json, structured_output
                        ):
                            logger.warning(
                                "ollama: LLM returned its own schema definition instead of an "
                                "answer; clearing response text",
                            )
                            content_text = ""
                        else:
                            logger.warning(
                                "ollama structured output could not be validated against %s; "
                                "falling back to unstructured text response",
                                structured_output.__name__,
                            )
                    except Exception:
                        logger.warning(
                            "ollama structured output could not be validated against %s; "
                            "falling back to unstructured text response",
                            structured_output.__name__,
                        )
                    return {
                        "content": content_text,
                        "structured": False,
                        "tool_calls": tool_calls or [],
                        "usage": usage,
                        "text": content_text,
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
            elif self._is_model_not_found_error(e):
                missing_model = self._extract_missing_model_name(e)
                logger.warning(
                    "ollama model not available (handled): %s",
                    missing_model,
                )
            elif classify_error(e) != "unknown":
                # A recognized, expected failure category (config/auth/rate-limit/
                # server error, etc.) — the fallback layer will classify and
                # route around it; a full traceback here is just noise.
                logger.warning("ollama.chat failed (classified as: %s): %s", classify_error(e), e)
            else:
                logger.exception("ollama.chat failed")

            try:
                _think_value = locals().get("think_value")
                log_llm_usage(
                    provider="ollama",
                    model=self.config.model,
                    structured=bool(structured_output),
                    has_tools=bool(tools),
                    error=str(e),
                    host=self.config.base_url,
                    thinking_enabled=bool(_think_value) if _think_value is not None else None,
                    thinking_level=_think_value if isinstance(_think_value, str) else None,
                    response_format="structured" if structured_output is not None else "text",
                )
            except Exception:
                pass

            return {
                "error": str(e),
                "error_code": "model_not_found" if self._is_model_not_found_error(e) else "provider_error",
                "_usage_error_logged": True,
                "content": None,
                "structured": structured_output is not None,
                "tool_calls": [],
                "usage": {},
            }
