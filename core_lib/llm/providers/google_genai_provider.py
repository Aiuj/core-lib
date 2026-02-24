"""Google GenAI provider using the official google.genai SDK.

Docs consulted via Context7 show structured outputs with Pydantic are supported
by passing response_mime_type='application/json' and response_schema=MyModel to
GenerateContentConfig, and chat via client.chats.create(...).send_message(...).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type
import os
import time
import json


from pydantic import BaseModel

from .base import BaseProvider
from ..llm_config import LLMConfig
from dataclasses import dataclass
from typing import Optional, Dict, Any
from core_lib import get_module_logger
from core_lib.tracing.tracing import add_trace_metadata
from core_lib.tracing.service_usage import log_llm_usage
from core_lib.llm.rate_limiter import RateLimitConfig, RateLimiter
from core_lib.llm.retry import RetryConfig, retry_handler
from core_lib.llm.json_parser import (
    parse_structured_output,
    augment_prompt_for_json,
    extract_json_from_text,
    _is_pydantic_schema_echo,
)
from core_lib.llm.provider_health import classify_error

logger = get_module_logger()

# Global flag to ensure instrumentation happens only once
_instrumentation_initialized = False


def _get_http_timeout_ms() -> int:
    """Get Google GenAI HTTP timeout in milliseconds.

    Uses GOOGLE_GENAI_HTTP_TIMEOUT_MS when set; defaults to 60000ms.
    """
    raw = os.getenv("GOOGLE_GENAI_HTTP_TIMEOUT_MS", "").strip()
    if not raw:
        return 60_000
    try:
        timeout = int(raw)
        if timeout > 0:
            return timeout
    except ValueError:
        pass
    logger.warning("Invalid GOOGLE_GENAI_HTTP_TIMEOUT_MS=%r, falling back to 60000", raw)
    return 60_000


def _clean_schema_for_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Remove additionalProperties from JSON schema recursively and add propertyOrdering.
    
    The Gemini API does not support the 'additionalProperties' field in JSON schemas.
    When Pydantic models have extra='forbid', they generate additionalProperties: false,
    which causes Gemini API errors. This function recursively removes that field.
    
    Additionally, Gemini 2.0 requires an explicit 'propertyOrdering' list to define
    the preferred structure. We derive this from the keys of the 'properties' dict.
    
    Args:
        schema: JSON schema dictionary (from Pydantic's model_json_schema())
        
    Returns:
        Cleaned schema without additionalProperties and with propertyOrdering
    """
    if not isinstance(schema, dict):
        return schema
    
    cleaned = {}
    for key, value in schema.items():
        # Skip additionalProperties entirely
        if key == "additionalProperties":
            continue
        # Recursively clean nested dicts
        if isinstance(value, dict):
            cleaned[key] = _clean_schema_for_gemini(value)
        # Recursively clean lists of dicts
        elif isinstance(value, list):
            cleaned[key] = [
                _clean_schema_for_gemini(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
            
    # Add propertyOrdering if this is an object with properties
    if "properties" in cleaned and isinstance(cleaned["properties"], dict):
        cleaned["propertyOrdering"] = list(cleaned["properties"].keys())
        
    return cleaned


@dataclass
class GeminiConfig(LLMConfig):
    api_key: Optional[str]
    base_url: str = "https://generativelanguage.googleapis.com"
    safety_settings: Optional[Dict[str, Any]] = None
    project: Optional[str] = None
    location: Optional[str] = None
    service_account_file: Optional[str] = None
    thinking_config: Optional[Dict[str, Any]] = None
    http_timeout_ms: Optional[int] = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        thinking_enabled: bool = False,
        thinking_config: Optional[Dict[str, Any]] = None,
        base_url: str = "https://generativelanguage.googleapis.com",
        safety_settings: Optional[Dict[str, Any]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        service_account_file: Optional[str] = None,
        http_timeout_ms: Optional[int] = None,
    ):
        super().__init__("gemini", model, temperature, max_tokens, thinking_enabled)
        self.api_key = api_key
        self.base_url = base_url
        self.safety_settings = safety_settings or {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",
        }
        self.project = project
        self.location = location
        self.service_account_file = service_account_file
        self.thinking_config = dict(thinking_config) if thinking_config else None
        self.http_timeout_ms = http_timeout_ms

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        import os

        def get_env(*names, default=None):
            for name in names:
                val = os.getenv(name)
                if val is not None:
                    return val
            return default

        api_key = get_env("GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY", default=None)
        model = get_env("GEMINI_MODEL", "GOOGLE_GENAI_MODEL", "GOOGLE_GENAI_MODEL_DEFAULT", default="gemini-1.5-flash")
        temperature = float(get_env("GEMINI_TEMPERATURE", "GOOGLE_GENAI_TEMPERATURE", default="0.1"))
        max_tokens_env = get_env("GEMINI_MAX_TOKENS", "GOOGLE_GENAI_MAX_TOKENS")
        max_tokens = int(max_tokens_env) if max_tokens_env is not None else None
        thinking_enabled = get_env("GEMINI_THINKING_ENABLED", "GOOGLE_GENAI_THINKING_ENABLED", default="false").lower() == "true"
        thinking_level = get_env("GEMINI_THINKING_LEVEL", "GOOGLE_GENAI_THINKING_LEVEL", default=None)
        thinking_budget_env = get_env("GEMINI_THINKING_BUDGET", "GOOGLE_GENAI_THINKING_BUDGET", default=None)
        include_thoughts_env = get_env("GEMINI_INCLUDE_THOUGHTS", "GOOGLE_GENAI_INCLUDE_THOUGHTS", default=None)
        base_url = get_env("GEMINI_BASE_URL", "GOOGLE_GENAI_BASE_URL", default="https://generativelanguage.googleapis.com")
        
        project = get_env("GOOGLE_CLOUD_PROJECT", "GOOGLE_PROJECT_ID")
        location = get_env("GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_REGION")
        service_account_file = get_env("GOOGLE_APPLICATION_CREDENTIALS", "GEMINI_SERVICE_ACCOUNT_FILE")

        thinking_config: Optional[Dict[str, Any]] = None
        if thinking_level is not None or thinking_budget_env is not None or include_thoughts_env is not None:
            thinking_config = {}
            if thinking_level is not None:
                thinking_config["level"] = str(thinking_level).lower()
            if thinking_budget_env is not None:
                thinking_config["budget"] = int(thinking_budget_env)
            if include_thoughts_env is not None:
                thinking_config["include_thoughts"] = str(include_thoughts_env).lower() == "true"

        return cls(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_enabled=thinking_enabled,
            thinking_config=thinking_config,
            base_url=base_url,
            project=project,
            location=location,
            service_account_file=service_account_file,
        )

class GoogleGenAIProvider(BaseProvider):
    """Provider implementation for Google GenAI (Gemini).

    Includes a lightweight in-process rate limiter enforcing per-model
    requests-per-minute (RPM) ceilings derived from a hardcoded table. Token
    and daily limits are currently not enforced client-side. The limiter is
    best-effort and fail-soft: if acquisition raises, the request proceeds.
    
    Also includes retry logic with exponential backoff for transient failures
    such as rate limits, server errors, and network issues.
    """

    # Hardcoded per-model request-per-minute limits. Keys are lowercase substrings
    # we expect to find in the configured model name for a match. These values
    # reflect only RPM (requests per minute); TPM/RPD intentionally ignored for now.
    _MODEL_RPM: Dict[str, int] = {
        "gemini-3-pro-preview": 5,    # Gemini 3.0 Pro
        "gemini-3-flash-preview": 30, # Gemini 3.0 Flash
        "gemini-2.5-pro": 10,         # Gemini 2.5 Pro
        "gemini-2.5-flash-lite": 120, # Gemini 2.5 Flash-Lite
        "gemini-2.5-flash": 30,       # Gemini 2.5 Flash
        "gemma-3": 60,                # Gemma 3
        "embedding": 600,             # Gemini Embedding models
    }
    
    # Models known to NOT support native JSON mode (structured output)
    # These will automatically use fallback JSON parsing
    _NO_JSON_MODE_MODELS: List[str] = [
        "gemma",  # All Gemma models (gemma-2, gemma-3, etc.)
    ]

    def __init__(self, config: GeminiConfig) -> None:  # type: ignore[override]
        super().__init__(config)

        # Lazy import to avoid hard dependency if unused
        import os
        from google import genai  # type: ignore

        # Resolve project and location: config > env var > None
        project = config.project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_PROJECT_ID")
        location = config.location or os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("GOOGLE_CLOUD_REGION")

        # Determine mode: Vertex AI when service account + project + location are set; otherwise AI Studio.
        use_vertex = bool(project and location and config.service_account_file)

        if use_vertex:
            # Vertex AI Mode
            client_kwargs = {
                "vertexai": True,
                "project": project,
                "location": location,
            }
            
            # Application Credentials from file (if provided)
            # This allows overriding the default ADC or env var credentials for specific providers
            if config.service_account_file:
                try:
                    from google.oauth2 import service_account
                    # Explicitly set cloud-platform scope ensuring access to Vertex AI
                    scopes = ['https://www.googleapis.com/auth/cloud-platform']
                    credentials = service_account.Credentials.from_service_account_file(
                        config.service_account_file, 
                        scopes=scopes
                    )
                    client_kwargs["credentials"] = credentials
                    # Note: When using credentials object, we don't need to pass api_key is ignored in vertex mode usually
                except ImportError:
                    logger.warning("google-auth not installed, cannot load service_account_file. Install with 'pip install google-auth'")
                except Exception as e:
                    logger.error(f"Failed to load service account file: {e}")
            
            http_timeout_ms = config.http_timeout_ms if config.http_timeout_ms is not None else _get_http_timeout_ms()
            self._client = genai.Client(
                **client_kwargs,
                http_options={"timeout": http_timeout_ms, "retry_options": None},  # ms; no SDK-internal retries
            )
        else:
            # AI Studio (Standard) Mode
            http_timeout_ms = config.http_timeout_ms if config.http_timeout_ms is not None else _get_http_timeout_ms()
            self._client = genai.Client(
                api_key=config.api_key,
                http_options={"timeout": http_timeout_ms, "retry_options": None},  # ms; no SDK-internal retries
            )

        logger.debug(f"Configured Google GenAI HTTP timeout: {http_timeout_ms}ms")

        # Instrument only once globally to avoid "already instrumented" warnings
        global _instrumentation_initialized
        if not _instrumentation_initialized:
            from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
            GoogleGenAIInstrumentor().instrument()
            _instrumentation_initialized = True
            logger.debug("Initialized Google GenAI instrumentation")

        # Initialize a rate limiter based on model RPM. We derive a conservative
        # per-second rate as max(1, RPM/60). If model not found, default to 60 RPM.
        # Vertex AI has much higher rate limits than AI Studio, so we apply a
        # multiplier when running in Vertex mode.
        self._is_vertex = use_vertex
        model_lc = (config.model or "").lower()
        rpm = 60  # default fallback
        # Choose the most specific matching key (longest substring match)
        matches = [k for k in self._MODEL_RPM.keys() if k in model_lc]
        if matches:
            matches.sort(key=len, reverse=True)
            rpm = self._MODEL_RPM[matches[0]]
        # Vertex AI typically allows 10-200x higher RPM than AI Studio free tier
        if use_vertex:
            rpm = max(rpm * 30, 300)
        rps = max(1.0 / 60.0, rpm / 60.0)  # ensure non-zero; may be fractional
        self._rate_limiter = RateLimiter(
            RateLimitConfig(requests_per_minute=rpm, requests_per_second=rps)
        )
        logger.debug(
            "Initialized Gemini rate limiter",
            extra={"model": config.model, "rpm": rpm, "rps": rps, "vertex": use_vertex},
        )

        # Configure retry logic for common transient failures
        # Identify retryable exceptions from google-genai and google.api_core
        retryable_exceptions = []
        try:
            from google.api_core import exceptions as gac_exceptions
            retryable_exceptions.extend([
                gac_exceptions.ResourceExhausted,     # 429 rate limits
                gac_exceptions.ServiceUnavailable,    # 503 service unavailable
                gac_exceptions.InternalServerError,   # 500 internal errors
                gac_exceptions.DeadlineExceeded,      # 504 gateway timeout
                gac_exceptions.TooManyRequests,       # Additional rate limit variant
            ])
        except ImportError:
            # google.api_core might not be available in all setups
            pass

        # Add google.genai specific errors — but NOT ServerError (503/504 overload).
        # 503 UNAVAILABLE and 504 DEADLINE_EXCEEDED mean the provider is overwhelmed;
        # retrying within the same provider wastes time — let the fallback client move on.
        # Only ClientError variants that are genuinely transient (e.g. brief 500 internal)
        # are excluded here so they fall through immediately to the fallback chain.
        try:
            from google.genai import errors as genai_errors
            # Intentionally NOT including genai_errors.ServerError here.
            # 503/504 from Gemini mean the model is overloaded — no point retrying.
            _ = genai_errors  # imported for reference; no exceptions added
        except ImportError:
            pass

        # Add common network-level exceptions (brief transient glitches worth one retry)
        retryable_exceptions.extend([
            ConnectionError,
            TimeoutError,
        ])

        self._retry_config = RetryConfig(
            max_retries=1,   # One retry for genuine transient glitches; overloaded models go to fallback
            base_delay=1.0,
            max_delay=5.0,
            retry_on_exceptions=tuple(retryable_exceptions) if retryable_exceptions else (Exception,),
        )
        logger.debug(
            "Initialized Gemini retry config",
            extra={"max_retries": self._retry_config.max_retries, "retryable_exceptions_count": len(retryable_exceptions)},
        )

    def _supports_json_mode(self) -> bool:
        """Check if the current model supports native JSON mode.
        
        Returns:
            True if model supports native structured output, False otherwise
        """
        model_lc = (self.config.model or "").lower()
        for no_json_model in self._NO_JSON_MODE_MODELS:
            if no_json_model in model_lc:
                return False
        return True
    
    def close(self) -> None:
        """Close the underlying google-genai client."""
        client = getattr(self, "_client", None)
        if client is None:
            return
        try:
            close_method = getattr(client, "close", None)
            if callable(close_method):
                close_method()
        except Exception as exc:
            logger.debug("Gemini client close failed", extra={"error": str(exc)})

    def _extract_text_from_response(self, resp: Any) -> str:
        """Extract text from response, handling non-text parts without warnings.
        
        When thinking mode is enabled, responses include thought_signature parts.
        Accessing resp.text triggers a warning. Instead, we manually extract text
        from all candidates and parts.
        
        Args:
            resp: Response from generate_content or send_message
            
        Returns:
            Extracted text content, concatenated from all text parts
        """
        try:
            # Try to access candidates directly to avoid .text warning
            candidates = getattr(resp, "candidates", [])
            if candidates and len(candidates) > 0:
                content = getattr(candidates[0], "content", None)
                if content:
                    parts = getattr(content, "parts", [])
                    text_parts = []
                    for part in parts:
                        # Extract text from text parts only
                        part_text = getattr(part, "text", None)
                        if part_text:
                            text_parts.append(part_text)
                    if text_parts:
                        return "".join(text_parts)
            # Fallback to .text attribute if candidates approach fails
            # This may trigger the warning but ensures we always get content
            return getattr(resp, "text", "")
        except Exception:
            # Ultimate fallback
            return getattr(resp, "text", "")

    def _to_genai_messages(self, messages: List[Dict[str, str]]) -> str:
        """Flatten OpenAI-style messages to a single prompt string.

        google.genai chat API supports chat sessions, but mapping roles to a
        single prompt keeps things simple for now. For richer role handling,
        we could use client.chats and turn history, but generate_content also
        accepts contents=str. We'll join messages into a single text.
        """
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role in ("assistant", "ai"):
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        return "\n".join(parts)

    def _build_config(
        self,
        *,
        structured_output: Optional[Type[BaseModel]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_message: Optional[str] = None,
        use_search_grounding: bool = False,
        thinking_enabled_override: Optional[bool] = None,
        cached_content: Optional[str] = None,
    ) -> Dict[str, Any]:
        from google.genai import types  # type: ignore

        cfg: Dict[str, Any] = {
            "temperature": self.config.temperature,
        }
        
        # Explicit Context Caching support
        if cached_content:
            if system_message:
                logger.warning(
                    "System message provided with cached_content; this may be ignored or cause errors "
                    "as system instructions should be baked into the cache."
                )

        if getattr(self.config, "max_tokens", None) is not None:
            cfg["max_output_tokens"] = self.config.max_tokens

        # Thinking configuration (Gemini 2.5/3.x)
        try:
            model_lc = (self.config.model or "").lower()
            supports_thinking = ("gemini-2.5" in model_lc) or ("gemini-3" in model_lc)

            level_budget_map = {
                "low": 1024,
                "medium": 4096,
                "high": 8192,
                "max": 16384,
                "intense": 16384,
            }
            disable_levels = {"off", "none", "disabled", "disable", "0"}

            # precedence: chat override > config
            enabled = thinking_enabled_override
            if enabled is None:
                enabled = getattr(self.config, "thinking_enabled", False)

            cfg_thinking = getattr(self.config, "thinking_config", None) or {}
            if not isinstance(cfg_thinking, dict):
                cfg_thinking = {}

            if "enabled" in cfg_thinking and thinking_enabled_override is None:
                enabled = bool(cfg_thinking.get("enabled"))

            raw_level = cfg_thinking.get("level")
            level = str(raw_level).lower().strip() if raw_level is not None else None

            raw_budget = cfg_thinking.get("budget")
            budget: Optional[int] = None
            if raw_budget is not None:
                try:
                    budget = max(0, int(raw_budget))
                except Exception:
                    budget = None

            include_thoughts_raw = cfg_thinking.get("include_thoughts")
            include_thoughts = bool(include_thoughts_raw) if include_thoughts_raw is not None else None

            if supports_thinking:
                disable_requested = False
                if enabled is False:
                    disable_requested = True
                if level in disable_levels:
                    disable_requested = True
                if budget == 0:
                    disable_requested = True

                if disable_requested:
                    cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
                else:
                    has_explicit_thinking_cfg = (
                        (enabled is True) or
                        (level is not None) or
                        (budget is not None) or
                        (include_thoughts is not None)
                    )
                    if has_explicit_thinking_cfg:
                        thinking_kwargs: Dict[str, Any] = {}

                        if budget is not None and budget > 0:
                            thinking_kwargs["thinking_budget"] = budget
                        elif level in level_budget_map:
                            thinking_kwargs["thinking_budget"] = level_budget_map[level]

                        if include_thoughts is not None:
                            thinking_kwargs["include_thoughts"] = include_thoughts
                        elif enabled is True:
                            thinking_kwargs["include_thoughts"] = True

                        cfg["thinking_config"] = types.ThinkingConfig(**thinking_kwargs)
        except Exception:
            # Never fail building config due to thinking support
            pass

        if system_message and not cached_content:
            cfg["system_instruction"] = system_message

        if tools and not cached_content:
            cfg["tools"] = tools

        # Grounding via Google Search (per official docs)
        # https://ai.google.dev/gemini-api/docs/google-search
        # This uses safety-reviewed search augmentation when supported by the model.
        if use_search_grounding:
            try:
                # Enable Google Search tool and config per official docs
                gs = types.GoogleSearch()
                gs_tool = types.Tool(google_search=gs)
                if cfg.get("tools"):
                    cfg["tools"] = [*cfg["tools"], gs_tool]
                else:
                    cfg["tools"] = [gs_tool]
                cfg["tool_config"] = types.ToolConfig(google_search=gs)
            except Exception:
                # Fail-soft: if SDK/version doesn't support it, ignore silently
                pass

        if structured_output is not None:
            # Use official structured output support
            # Clean the schema to remove additionalProperties which Gemini doesn't support
            # The SDK accepts either a Pydantic model or a JSON schema dict
            # We convert to dict and clean it for maximum compatibility
            try:
                json_schema = structured_output.model_json_schema()
            except AttributeError:
                # Fallback: if not a Pydantic model, assume it's already a dict
                json_schema = structured_output
                
            cleaned_schema = _clean_schema_for_gemini(json_schema)
            cfg = {
                **cfg,
                "response_mime_type": "application/json",
                "response_json_schema": cleaned_schema,
            }
        result = {"config": types.GenerateContentConfig(**cfg)}
        if cached_content:
            result["cached_content"] = cached_content
        return result

    def _build_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        # google.genai supports tools and tool_config; for now pass through as-is
        # when in OpenAI function-call schema. Advanced conversion could be added.
        if not tools:
            return {}
        return {"tools": tools}

    def _acquire_rate_limit(self) -> None:
        """Acquire the rate limiter slot.

        The provider's public interface is synchronous, but the RateLimiter is
        asyncio-based. This method handles the async rate limiter with proper
        timeout and thread-safety for both sync and async contexts.
        """
        try:
            import asyncio
            import concurrent.futures
            
            # Short timeout since rate limiting should be fast
            MAX_TIMEOUT = 5.0
            
            async def _acquire_with_timeout():
                """Acquire with timeout wrapper."""
                try:
                    await asyncio.wait_for(
                        self._rate_limiter.acquire(), 
                        timeout=MAX_TIMEOUT
                    )
                    return True
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Rate limiter acquisition timed out after {MAX_TIMEOUT}s"
                    )
                    return False
            
            def _run_in_new_loop():
                """Run the async function in a new event loop."""
                return asyncio.run(_acquire_with_timeout())
            
            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - use thread executor
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_run_in_new_loop)
                    future.result(timeout=MAX_TIMEOUT + 1.0)
            except RuntimeError:
                # No running loop - standard case for sync calls
                asyncio.run(_acquire_with_timeout())
                    
        except Exception as e:
            # Never block the call due to rate limiter errors
            logger.warning(
                "Rate limiter acquisition failed; proceeding without throttle",
                exc_info=e
            )

    def chat(
        self,
        *,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        structured_output: Optional[Type[BaseModel]] = None,
        system_message: Optional[str] = None,
        use_search_grounding: bool = False,
        thinking_enabled: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Main chat interface with rate limiting and retry logic."""
        try:
            # Enforce model-specific RPM throttling before performing network call.
            self._acquire_rate_limit()
            
            cached_content = kwargs.get("cached_content")
            
            # Call the actual API with retry logic
            return self._chat_with_retry(
                messages=messages,
                tools=tools,
                structured_output=structured_output,
                system_message=system_message,
                use_search_grounding=use_search_grounding,
                thinking_enabled=thinking_enabled,
                cached_content=cached_content,
            )
        except Exception as e:  # pragma: no cover - network errors
            error_reason = classify_error(e)

            # Classify ServerError (503/504) and ClientError 499 CANCELLED as server-side
            # transient failures for clean logging (warning instead of full traceback).
            is_server_overload = False
            try:
                from google.genai import errors as genai_errors
                if isinstance(e, genai_errors.ServerError):
                    is_server_overload = True
                    if not error_reason or error_reason == "unknown":
                        error_reason = "server_error"
                elif isinstance(e, genai_errors.ClientError) and "499" in str(e):
                    # 499 CANCELLED: the server aborted the streaming call (transient)
                    is_server_overload = True
                    error_reason = "server_error"
            except ImportError:
                pass

            # Expected transient failures: log as warning without traceback
            silent_reasons = ("rate_limit", "quota_exceeded", "server_error", "timeout")
            if error_reason in silent_reasons or is_server_overload:
                logger.warning(
                    f"genai.chat unavailable ({error_reason}): {type(e).__name__}: {e}",
                    extra={"model": self.config.model, "error_reason": error_reason},
                )
            else:
                # Unexpected errors: log with full traceback for debugging
                is_retryable = isinstance(e, self._retry_config.retry_on_exceptions)
                if is_retryable:
                    logger.error(f"genai.chat failed after all retries: {type(e).__name__}: {e}")
                else:
                    logger.exception("genai.chat failed with non-retryable error")
            
            return {
                "error": str(e),
                "content": None,
                "structured": structured_output is not None,
                "tool_calls": [],
                "usage": {},
            }

    def _chat_with_retry(
        self,
        *,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        structured_output: Optional[Type[BaseModel]] = None,
        system_message: Optional[str] = None,
        use_search_grounding: bool = False,
        thinking_enabled: Optional[bool] = None,
        cached_content: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Core API call logic with retry decoration applied.
        
        Includes fallback for models that don't support native structured output.
        """
        
        # Check if we should use fallback JSON parsing upfront
        # This avoids wasting retries on known unsupported features
        use_fallback_json = structured_output is not None and not self._supports_json_mode()
        if use_fallback_json:
            logger.debug(
                f"Model {self.config.model} does not support native JSON mode, "
                "using fallback text-based JSON parsing"
            )
        
        @retry_handler(self._retry_config)
        def _make_api_call() -> Dict[str, Any]:
            nonlocal use_fallback_json
            # Minimal debug without leaking content
            logger.debug(
                "genai.chat start",
                extra={
                    "llm_provider": "gemini",
                    "model": self.config.model,
                    "msg_count": len(messages),
                    "has_tools": bool(tools),
                    "structured": bool(structured_output),
                    "cached_content": cached_content,
                },
            )
            # Determine if this is a single-turn prompt (only one user message, optional system)
            user_messages = [m for m in messages if m.get("role") == "user"]
            assistant_messages = [m for m in messages if m.get("role") == "assistant"]
            is_single_turn = len(user_messages) == 1 and len(assistant_messages) == 0 and len(messages) <= 2

            # If using fallback, augment the last user message with JSON schema
            working_messages = messages
            if use_fallback_json and structured_output:
                working_messages = list(messages)
                for i in range(len(working_messages) - 1, -1, -1):
                    if working_messages[i].get("role") == "user":
                        working_messages[i] = {
                            **working_messages[i],
                            "content": augment_prompt_for_json(
                                working_messages[i].get("content", ""),
                                structured_output
                            )
                        }
                        break
                user_messages = [m for m in working_messages if m.get("role") == "user"]

            if is_single_turn:
                user_text = user_messages[0].get("content", "")
                extra = self._build_config(
                    structured_output=structured_output if not use_fallback_json else None,
                    tools=tools,
                    system_message=system_message,
                    use_search_grounding=use_search_grounding,
                    thinking_enabled_override=thinking_enabled,
                    cached_content=cached_content,
                )
                start = time.perf_counter()
                
                # Use streaming to keep connection alive during long generations (prevents 504/timeouts)
                stream = self._client.models.generate_content_stream(
                    model=self.config.model,
                    contents=user_text,
                    **extra,
                )
                
                # Consume stream to accumulate response
                # We need to manually aggregate usage and content since we're consuming chunks
                chunks = []
                for chunk in stream:
                    chunks.append(chunk)

                latency_ms = (time.perf_counter() - start) * 1000
                
                # Use the last chunk for metadata, but we'll need to aggregate text/function calls
                resp = chunks[-1] if chunks else None
                if not resp:
                     raise RuntimeError("Empty response stream from Google GenAI")
                     
            else:
                prompt = self._to_genai_messages(working_messages)
                extra = self._build_config(
                    structured_output=structured_output if not use_fallback_json else None,
                    tools=tools,
                    system_message=system_message,
                    use_search_grounding=use_search_grounding,
                    thinking_enabled_override=thinking_enabled,
                    cached_content=cached_content,
                )
                chat = self._client.chats.create(model=self.config.model)
                start = time.perf_counter()
                
                # Use streaming to keep connection alive
                stream = chat.send_message_stream(prompt, **extra)  # type: ignore[arg-type]
                
                chunks = []
                for chunk in stream:
                    chunks.append(chunk)

                latency_ms = (time.perf_counter() - start) * 1000
                
                resp = chunks[-1] if chunks else None
                if not resp:
                     raise RuntimeError("Empty response stream from Google GenAI")

            # Aggregate content from chunks
            # text: concatenate all chunks
            # usage: usually in last chunk
            # function_calls: accumulate from all chunks
            
            full_text = ""
            all_tool_calls: List[Dict[str, Any]] = []
            
            for chunk in chunks:
                # Accumulate text
                chunk_text = self._extract_text_from_response(chunk)
                if chunk_text:
                    full_text += chunk_text
                
                # Accumulate function calls
                if getattr(chunk, "function_calls", None):
                    for fc in chunk.function_calls:  # type: ignore[attr-defined]
                        all_tool_calls.append(
                            {
                                "id": fc.get("id") if isinstance(fc, dict) else None,
                                "function": {
                                    "name": fc.get("name") if isinstance(fc, dict) else getattr(fc, "name", ""),
                                    "arguments": fc.get("args") if isinstance(fc, dict) else getattr(fc, "args", {}),
                                },
                                "type": "function",
                            }
                        )

            # Use aggregated values
            tool_calls = all_tool_calls
            
            # Log service usage to OpenTelemetry/OpenSearch (replaces Langfuse tracing)
            try:
                usage_metadata = getattr(resp, "usage_metadata", None)
                input_tokens = None
                output_tokens = None
                total_tokens = None
                
                if usage_metadata is not None:
                    # usage_metadata is a GenerateContentResponseUsageMetadata object, not a dict
                    input_tokens = getattr(usage_metadata, "prompt_token_count", None)
                    output_tokens = getattr(usage_metadata, "candidates_token_count", None)
                    total_tokens = getattr(usage_metadata, "total_token_count", None)

                usage = {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
                
                # Add trace metadata for observability (Langfuse - no-op when LANGFUSE_TRACING_ENABLED=false)
                add_trace_metadata({
                    "usage": usage,
                    "latency_ms": latency_ms,
                    "model": self.config.model,
                    "provider": "google_genai"
                })

                # Log to OTLP/OpenSearch via standard logger (independent of Langfuse)
                log_llm_usage(
                    provider="google_genai",
                    model=self.config.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    structured=bool(structured_output and not use_fallback_json),
                    has_tools=bool(tools),
                    search_grounding=use_search_grounding,
                )
            except Exception as e:
                logger.warning(f"Failed to log usage metadata: {str(e)}")

            extracted_text = full_text
            
            # Handle structured output parsing if needed
            parsed_result = None

            if structured_output:
                if use_fallback_json:
                    # Fallback mode: parse JSON from text response.
                    # Use parse_structured_output which handles:
                    # 1. Markdown code-block wrappers (```json ... ```)
                    # 2. Schema-as-instance: model echoes JSON Schema structure
                    #    with actual values inside "properties" instead of a
                    #    plain instance (common in smaller/local Gemma models).
                    parsed_dict = parse_structured_output(extracted_text, structured_output)
                    if parsed_dict is not None:
                        try:
                            parsed_result = structured_output.model_validate(parsed_dict)
                        except Exception as e:
                            logger.warning(f"Failed to validate recovered JSON against schema: {e}")
                    else:
                        # Check whether the model returned its own JSON Schema definition
                        # instead of a filled instance (e.g. gemma-3-4b-it).  If so,
                        # clear extracted_text so callers never surface the schema JSON
                        # as an answer.
                        try:
                            raw_json = extract_json_from_text(extracted_text)
                            if isinstance(raw_json, dict) and _is_pydantic_schema_echo(
                                raw_json, structured_output
                            ):
                                logger.warning(
                                    "LLM returned its own schema definition instead of an "
                                    "answer for model %s; clearing response text",
                                    self.config.model,
                                )
                                extracted_text = ""
                            else:
                                logger.warning(
                                    "Failed to parse fallback JSON for model %s; "
                                    "will return unstructured text",
                                    self.config.model,
                                )
                        except Exception:
                            logger.warning(
                                "Failed to parse fallback JSON for model %s; "
                                "will return unstructured text",
                                self.config.model,
                            )
                else:
                    # Native mode: try to get parsed object from last chunk
                    # Usually present in resp.parsed if the model supports it
                    try:
                        parsed_result = getattr(resp, "parsed", None)
                        if parsed_result is None and extracted_text:
                            # Sometimes parsed is None but text contains JSON.
                            # Use parse_structured_output so the full fuzzy recovery
                            # pipeline applies (code blocks, schema-echo, schema-as-instance,
                            # key normalisation, nested dicts, Literal coercion).
                            parsed_dict = parse_structured_output(extracted_text, structured_output)
                            if parsed_dict is not None:
                                try:
                                    parsed_result = structured_output.model_validate(parsed_dict)
                                except Exception:
                                    pass
                            else:
                                # parse_structured_output exhausted all strategies.
                                # If the model returned its own schema definition,
                                # clear the text so it is never surfaced as an answer.
                                try:
                                    raw_json = extract_json_from_text(extracted_text)
                                    if isinstance(raw_json, dict) and _is_pydantic_schema_echo(
                                        raw_json, structured_output
                                    ):
                                        logger.warning(
                                            "LLM (native mode) returned its own schema definition "
                                            "for model %s; clearing response text",
                                            self.config.model,
                                        )
                                        extracted_text = ""
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.warning(f"Failed to retrieve native parsed result: {e}")

            # Extract thinking content if present (Gemini 2.0 Flash Thinking)
            # Thinking content is in 'thought_signature' parts
            thought_text = None
            if self.config.thinking_enabled:
                thought_parts = []
                # iterate all chunks for thoughts
                for chunk in chunks:
                    candidates = getattr(chunk, "candidates", [])
                    if candidates and len(candidates) > 0:
                        content = getattr(candidates[0], "content", None)
                        if content:
                            parts = getattr(content, "parts", [])
                            for part in parts:
                                # Check for thought_signature or thought (SDK dependent)
                                # In google-genai, it is usually part.thought if typed, or checked via attribute
                                if hasattr(part, "thought") and part.thought:
                                    thought_parts.append(part.thought)
                
                if thought_parts:
                    thought_text = "\n".join(thought_parts)

            result = {
                "content": extracted_text,
                "role": "assistant",
                "tool_calls": tool_calls,
                "usage": usage,
                "model": self.config.model,
                "provider": "google_genai",
            }
            
            if thought_text:
                result["thought"] = thought_text

            if parsed_result:
                # Follow the standard core-lib provider contract:
                # - structured=True (boolean flag, not the object)
                # - content = model_dump() dict so callers read content, not structured
                # - text = raw LLM text for debugging/logging
                result["content"] = parsed_result.model_dump()
                result["structured"] = True
                result["text"] = extracted_text
                result["content_json"] = json.dumps(result["content"], ensure_ascii=False)
            else:
                result["structured"] = False

            return result

        # Make the API call (fallback already determined above)
        return _make_api_call()
