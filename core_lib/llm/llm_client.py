"""Main LLM client class for abstracting different LLM providers.

This version removes LangChain and uses native provider SDKs via lightweight
provider classes under ``core_lib.llm.providers``.
"""

import json
from typing import List, Dict, Any, Optional, Union, Type

from pydantic import BaseModel

from .llm_config import LLMConfig, GeminiConfig, OllamaConfig, OpenAIConfig
from core_lib.tracing.tracing import setup_tracing
from .providers.base import BaseProvider
from .providers.google_genai_provider import GoogleGenAIProvider
from .providers.ollama_provider import OllamaProvider
from .providers.openai_provider import OpenAIProvider


class LLMClient:
    """Main LLM client that abstracts different LLM providers."""
    
    def __init__(self, config: LLMConfig):
        """Initialize the LLM client with a configuration.
        
        Args:
            config: Configuration object for the LLM provider
        """
        self.config = config
        self._provider = self._initialize_provider()
        self._closed = False
    
    def _initialize_provider(self) -> BaseProvider:
        """Initialize the appropriate provider based on the configuration."""
        if isinstance(self.config, GeminiConfig):
            return GoogleGenAIProvider(self.config)
        if isinstance(self.config, OllamaConfig):
            return OllamaProvider(self.config)
        if isinstance(self.config, OpenAIConfig):
            return OpenAIProvider(self.config)
        raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
    
    def chat(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[Dict[str, Any]]] = None,
        structured_output: Optional[Type[BaseModel]] = None,
        system_message: Optional[str] = None,
        use_search_grounding: bool = False,
        thinking_enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Send a chat message to the LLM.
        
        Args:
            messages: Either a string message or a list of message dictionaries
                     with 'role' and 'content' keys
            tools: Optional list of tools in OpenAI JSON format
            structured_output: Optional Pydantic model for structured JSON output
            system_message: Optional system message to prepend
            
        Returns:
            Dictionary containing the response, usage info, and any tool calls
        """
        # Normalize messages into a list[dict] with role/content for providers
        formatted_messages = self._normalize_messages(messages, system_message)
        
        # Initialize tracing and record pre-call metadata
        tracing_provider = None
        try:
            tracing_provider = setup_tracing()
            pre_metadata: Dict[str, Any] = {
                "event": "llm.chat.start",
                "llm": {
                    "provider": self.config.provider,
                    "model": self.config.model,
                },
                "config": {
                    "temperature": self.config.temperature,
                    "max_tokens": getattr(self.config, "max_tokens", None),
                    "thinking_enabled": getattr(self.config, "thinking_enabled", False),
                },
                "overrides": {
                    "thinking_enabled": thinking_enabled,
                },
                "input": {
                    "system_message_present": system_message is not None,
                    "message_count": len(formatted_messages),
                    "message_roles": [m.get("role") for m in formatted_messages],
                    # avoid recording full content by default; log lengths for privacy
                    "message_content_lengths": [len(m.get("content", "") or "") for m in formatted_messages],
                },
                "tools": {
                    "count": len(tools) if tools else 0,
                    "names": [t.get("function", {}).get("name") for t in tools] if tools else [],
                },
                "structured_output": bool(structured_output),
                "search_grounding": use_search_grounding,
            }
            tracing_provider.add_metadata(pre_metadata)
        except Exception:
            # Tracing should never break the chat flow
            pass

        # Delegate to the provider (handles structured output and tools natively)
        try:
            result = self._provider.chat(
                messages=formatted_messages,
                tools=tools,
                structured_output=structured_output,
                system_message=system_message,
                use_search_grounding=use_search_grounding,
                thinking_enabled=thinking_enabled,
            )

            # Post-call tracing metadata
            try:
                if tracing_provider:
                    tracing_provider.add_metadata({
                        "event": "llm.chat.end",
                        "structured": result.get("structured", False),
                        "output": {
                            "content_length": len(json.dumps(result.get("content", ""))) if not isinstance(result.get("content"), str) else len(result.get("content") or ""),
                        },
                        "tools": {
                            "tool_calls_count": len(result.get("tool_calls", []) or []),
                            "names": [tc.get("function", {}).get("name") for tc in (result.get("tool_calls") or [])],
                        },
                        "usage": result.get("usage", {}),
                    })
            except Exception:
                pass

            return result
        except Exception as e:
            # Trace error
            try:
                if tracing_provider:
                    tracing_provider.add_metadata({
                        "event": "llm.chat.error",
                        "structured": bool(structured_output),
                        "error": str(e),
                    })
            except Exception:
                pass
            return {
                "error": f"Chat request failed: {str(e)}",
                "content": None,
                "structured": bool(structured_output),
                "tool_calls": [],
                "usage": {},
            }

    def is_in_warmup(self) -> bool:
        """Return True while the underlying provider is in a WoL warmup window.

        During this window the server is powering on.  ``FallbackLLMClient``
        uses this to skip the provider and route requests to a secondary without
        permanently marking it unhealthy.
        """
        return self._provider.is_in_warmup()

    def close(self) -> None:
        """Release underlying provider resources."""
        if self._closed:
            return
        try:
            self._provider.close()
        finally:
            self._closed = True

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False
    
    def _normalize_messages(
        self,
        messages: Union[str, List[Dict[str, str]]],
        system_message: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Normalize messages into a list of dicts with role/content.

        Providers (Ollama) accept OpenAI-style dicts directly. For Google GenAI, the
        provider will adapt these messages to its native format.
        """
        result: List[Dict[str, str]] = []
        if system_message:
            result.append({"role": "system", "content": system_message})
        if isinstance(messages, str):
            result.append({"role": "user", "content": messages})
        else:
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                result.append({"role": role, "content": content})
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "thinking_enabled": self.config.thinking_enabled,
        }
