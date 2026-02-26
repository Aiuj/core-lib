"""Base provider contract for native SDK adapters.

Providers must implement the `chat` method that accepts OpenAI-style messages
and optional tools/structured output and returns a unified dict.

Example usage:
    ```python
    from core_lib.llm.providers.base import BaseProvider
    from core_lib.llm.llm_config import LLMConfig
    
    class MyProvider(BaseProvider):
        def chat(self, *, messages, tools=None, structured_output=None, 
                 system_message=None, use_search_grounding=False, 
                 thinking_enabled=None):
            # Implementation here
            return {
                "content": "response text",
                "structured": False,
                "tool_calls": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20}
            }
    ```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel

from ..llm_config import LLMConfig


class BaseProvider(ABC):
    """Abstract base class for LLM providers.
    
    All provider implementations must inherit from this class and implement
    the `chat` method. The base class provides a default `close` implementation
    that can be overridden for resource cleanup.
    
    Attributes:
        config: Configuration object containing provider-specific settings
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the provider with configuration.
        
        Args:
            config: Configuration object for the provider
        """
        self.config = config

    def is_in_warmup(self) -> bool:
        """Return True while this provider is in a post-WoL warmup window.

        During a warmup window (non-blocking WoL mode) the underlying server is
        booting up and should not be contacted.  Callers such as
        ``FallbackLLMClient`` use this to skip the provider and route requests
        to a secondary instead, without permanently marking the provider as
        unhealthy.

        The default implementation always returns False.  Providers that
        support Wake-on-LAN (``OllamaProvider``) override this method.
        """
        return False

    def close(self) -> None:
        """Release any provider resources.

        Providers that maintain network clients or pools should override this
        method to properly clean up resources. The default implementation is 
        a no-op so that callers can safely invoke ``close()`` on any provider 
        instance without checking if cleanup is needed.
        
        Example:
            ```python
            class MyProvider(BaseProvider):
                def close(self):
                    if hasattr(self, '_client') and self._client:
                        self._client.close()
            ```
        """
        return

    @abstractmethod
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
        """Send a chat to the provider and return a unified response dict.
        
        This is the core method that all providers must implement. It should
        handle the communication with the underlying LLM service and return
        a standardized response format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     Roles should be 'system', 'user', or 'assistant'.
            tools: Optional list of tools/functions in OpenAI JSON format.
                  Each tool should have 'type', 'function' with 'name', 
                  'description', and 'parameters' fields.
            structured_output: Optional Pydantic model class for structured JSON output.
                             When provided, the response content should be validated
                             against this schema.
            system_message: Optional system message to prepend to the conversation.
                          Providers should handle this appropriately for their API.
            use_search_grounding: When True, enable search/grounding capabilities
                                if supported by the provider (e.g., Google Search 
                                for Gemini, web search for OpenAI).
            thinking_enabled: Optional override for thinking/reasoning mode.
                            If None, uses the provider config default.
        
        Returns:
            Dictionary with the following structure:
            - content: str | dict | BaseModel | None - The response content
            - structured: bool - Whether this was a structured output request
            - tool_calls: list[dict] - Any tool/function calls requested by the model
            - usage: dict - Token usage statistics (prompt_tokens, completion_tokens, etc.)
            - error: str (optional) - Error message if the request failed
            
            When structured=True, additional fields should be included:
            - text: str - Raw text response from provider (typically JSON string)
            - content_json: str - JSON-serialized content for transport/logging
        
        Raises:
            Exception: Providers may raise exceptions for critical failures,
                      but should prefer returning error information in the
                      response dict when possible.
        
        Example:
            ```python
            # Simple text response
            {
                "content": "Hello! How can I help you?",
                "structured": False,
                "tool_calls": [],
                "usage": {"prompt_tokens": 15, "completion_tokens": 8}
            }
            
            # Structured output response
            {
                "content": {"name": "John", "age": 30},
                "structured": True,
                "tool_calls": [],
                "usage": {"prompt_tokens": 20, "completion_tokens": 15},
                "text": '{"name": "John", "age": 30}',
                "content_json": '{"name": "John", "age": 30}'
            }
            
            # Tool call response
            {
                "content": None,
                "structured": False,
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Boston"}'
                    }
                }],
                "usage": {"prompt_tokens": 25, "completion_tokens": 10}
            }
            ```
        """
        raise NotImplementedError
