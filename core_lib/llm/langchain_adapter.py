"""LangChain-compatible adapter for core-lib LLMClient.

This module provides a wrapper that makes the core-lib LLMClient compatible
with LangChain's BaseChatModel interface, enabling use with LangGraph and
other LangChain-based orchestration frameworks while preserving core-lib's
tracing, cost tracking, and error handling features.

Example usage:
    ```python
    from core_lib.llm import create_client_from_env
    from core_lib.llm.langchain_adapter import CoreLibChatModel
    
    # Create the core-lib client
    llm_client = create_client_from_env()
    
    # Wrap it for LangChain/LangGraph compatibility
    chat_model = CoreLibChatModel(client=llm_client)
    
    # Use with LangGraph
    from langgraph.graph import StateGraph, MessagesState
    
    def agent_node(state: MessagesState):
        response = chat_model.invoke(state["messages"])
        return {"messages": [response]}
    ```
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional, Sequence, Type, Union

from pydantic import BaseModel, Field, ConfigDict

try:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langchain_core.tools import BaseTool

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Define stubs so the module can be imported without langchain
    BaseChatModel = object  # type: ignore
    BaseMessage = object  # type: ignore
    AIMessage = object  # type: ignore
    ChatResult = object  # type: ignore

from .llm_client import LLMClient


def _convert_message_to_dict(message: BaseMessage) -> Dict[str, str]:
    """Convert a LangChain message to OpenAI-style dict format.
    
    Args:
        message: LangChain BaseMessage instance
        
    Returns:
        Dictionary with 'role' and 'content' keys
    """
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        return {"role": "assistant", "content": message.content or ""}
    elif isinstance(message, ToolMessage):
        # Tool messages need special handling - include tool_call_id
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": getattr(message, "tool_call_id", ""),
        }
    else:
        # Default to user role for unknown message types
        return {"role": "user", "content": str(message.content)}


def _convert_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert core-lib tool_calls format to LangChain format.
    
    Core-lib format (OpenAI style):
        {
            "id": "call_123",
            "type": "function", 
            "function": {"name": "get_weather", "arguments": '{"location": "Boston"}'}
        }
    
    LangChain format:
        {
            "id": "call_123",
            "name": "get_weather",
            "args": {"location": "Boston"}
        }
    """
    result = []
    for tc in tool_calls or []:
        func = tc.get("function", {})
        args = func.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        
        result.append({
            "id": tc.get("id", ""),
            "name": func.get("name", ""),
            "args": args,
        })
    return result


def _convert_langchain_tools_to_openai(
    tools: Sequence[Union[Dict[str, Any], BaseTool, Type[BaseModel]]]
) -> List[Dict[str, Any]]:
    """Convert LangChain tools to OpenAI function format.
    
    Accepts:
        - Dict with OpenAI function format
        - BaseTool instances
        - Pydantic BaseModel classes (for structured output)
    
    Returns:
        List of tools in OpenAI function format
    """
    result = []
    for tool in tools:
        if isinstance(tool, dict):
            # Already in dict format - assume it's OpenAI format
            result.append(tool)
        elif hasattr(tool, "as_tool"):
            # BaseTool with as_tool method - get OpenAI schema
            # Note: Different versions of langchain have different methods
            if hasattr(tool, "tool_call_schema"):
                schema = tool.tool_call_schema.schema()
            else:
                schema = tool.args_schema.schema() if hasattr(tool, "args_schema") else {}
            result.append({
                "type": "function",
                "function": {
                    "name": getattr(tool, "name", "unknown"),
                    "description": getattr(tool, "description", ""),
                    "parameters": schema,
                }
            })
        elif isinstance(tool, type) and issubclass(tool, BaseModel):
            # Pydantic model - convert to function schema
            schema = tool.model_json_schema()
            result.append({
                "type": "function",
                "function": {
                    "name": tool.__name__,
                    "description": schema.get("description", ""),
                    "parameters": schema,
                }
            })
    return result


class CoreLibChatModel(BaseChatModel):
    """LangChain-compatible wrapper for core-lib LLMClient.
    
    This adapter enables using core-lib's LLMClient with LangChain and LangGraph
    while preserving all the benefits of core-lib:
    - Unified tracing via Langfuse/OpenTelemetry
    - Cost tracking and usage metrics
    - Error handling with fallback support
    - Multi-provider support (Gemini, OpenAI, Ollama)
    
    Attributes:
        client: The core-lib LLMClient instance to wrap
        bound_tools: Tools bound to this model instance
        structured_output_schema: Pydantic model for structured output
        
    Example:
        ```python
        from core_lib.llm import create_client_from_env
        from core_lib.llm.langchain_adapter import CoreLibChatModel
        
        client = create_client_from_env()
        model = CoreLibChatModel(client=client)
        
        # Simple invocation
        response = model.invoke([HumanMessage(content="Hello!")])
        
        # With tools
        model_with_tools = model.bind_tools([my_tool])
        response = model_with_tools.invoke(messages)
        
        # With structured output
        model_structured = model.with_structured_output(MySchema)
        response = model_structured.invoke(messages)
        ```
    """
    
    # Pydantic model configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Required fields
    client: LLMClient = Field(..., description="The core-lib LLMClient instance")
    
    # Optional bound configuration
    bound_tools: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Tools bound to this model instance"
    )
    structured_output_schema: Optional[Type[BaseModel]] = Field(
        default=None,
        description="Pydantic model for structured output"
    )
    use_search_grounding: bool = Field(
        default=False,
        description="Whether to enable search grounding"
    )
    thinking_enabled: Optional[bool] = Field(
        default=None,
        description="Override for thinking/reasoning mode"
    )
    
    def __init__(self, **data: Any) -> None:
        """Initialize the adapter.
        
        Raises:
            ImportError: If langchain-core is not installed
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for CoreLibChatModel. "
                "Install it with: pip install langchain-core"
            )
        super().__init__(**data)
    
    @property
    def _llm_type(self) -> str:
        """Return a unique identifier for this model type."""
        model_info = self.client.get_model_info()
        return f"core-lib-{model_info.get('provider', 'unknown')}"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return parameters that identify this model for tracing."""
        return self.client.get_model_info()
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the LLM.
        
        This is the core method that LangChain calls. It converts messages
        to OpenAI format, calls the core-lib client, and converts the response
        back to LangChain format.
        
        Args:
            messages: List of LangChain messages
            stop: Optional stop sequences (not currently supported)
            run_manager: Optional callback manager for streaming
            **kwargs: Additional arguments (tools, structured_output, etc.)
            
        Returns:
            ChatResult containing the model's response
            
        Raises:
            ValueError: If the underlying LLM call fails
        """
        # Convert LangChain messages to OpenAI-style dicts
        formatted_messages = [_convert_message_to_dict(m) for m in messages]
        
        # Merge bound tools with any passed in kwargs
        tools = kwargs.get("tools") or self.bound_tools or None
        if tools and not isinstance(tools[0], dict):
            tools = _convert_langchain_tools_to_openai(tools)
        
        # Get structured output schema
        structured_output = kwargs.get("structured_output") or self.structured_output_schema
        
        # Call the core-lib client
        result = self.client.chat(
            messages=formatted_messages,
            tools=tools if tools else None,
            structured_output=structured_output,
            use_search_grounding=kwargs.get("use_search_grounding", self.use_search_grounding),
            thinking_enabled=kwargs.get("thinking_enabled", self.thinking_enabled),
        )
        
        # Check for errors
        if result.get("error"):
            raise ValueError(f"LLM call failed: {result['error']}")
        
        # Build the AIMessage response
        content = result.get("content")
        
        # Handle structured output - convert to string for LangChain
        if result.get("structured") and content is not None:
            if isinstance(content, BaseModel):
                content = content.model_dump_json()
            elif isinstance(content, dict):
                content = json.dumps(content)
        elif content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        
        # Convert tool calls to LangChain format
        tool_calls = _convert_tool_calls(result.get("tool_calls", []))
        
        # Build response message
        ai_message = AIMessage(
            content=content,
            tool_calls=tool_calls if tool_calls else [],
            additional_kwargs={
                "usage": result.get("usage", {}),
            },
        )
        
        # Create and return ChatResult
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream responses from the LLM.
        
        Note: The current core-lib LLMClient doesn't support streaming,
        so this falls back to the non-streaming implementation and yields
        the complete response as a single chunk.
        """
        # Fall back to non-streaming for now
        result = self._generate(messages, stop, run_manager, **kwargs)
        if result.generations:
            message = result.generations[0].message
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=message.content,
                    tool_calls=getattr(message, "tool_calls", []),
                )
            )
    
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], BaseTool, Type[BaseModel]]],
        **kwargs: Any,
    ) -> "CoreLibChatModel":
        """Bind tools to this model instance.
        
        Returns a new instance with the tools bound. This enables tool calling
        in LangGraph react agents and other tool-using patterns.
        
        Args:
            tools: Sequence of tools in various formats (dicts, BaseTool, Pydantic models)
            **kwargs: Additional arguments to pass through
            
        Returns:
            New CoreLibChatModel instance with tools bound
            
        Example:
            ```python
            from langchain_core.tools import tool
            
            @tool
            def get_weather(location: str) -> str:
                '''Get weather for a location.'''
                return f"Weather in {location}: sunny"
            
            model_with_tools = model.bind_tools([get_weather])
            ```
        """
        converted_tools = _convert_langchain_tools_to_openai(tools)
        
        return CoreLibChatModel(
            client=self.client,
            bound_tools=converted_tools,
            structured_output_schema=self.structured_output_schema,
            use_search_grounding=self.use_search_grounding,
            thinking_enabled=self.thinking_enabled,
        )
    
    def with_structured_output(
        self,
        schema: Type[BaseModel],
        **kwargs: Any,
    ) -> "CoreLibChatModel":
        """Return a model that produces structured output.
        
        The returned model will validate responses against the provided
        Pydantic schema and return structured data.
        
        Args:
            schema: Pydantic BaseModel class defining the output structure
            **kwargs: Additional arguments
            
        Returns:
            New CoreLibChatModel instance configured for structured output
            
        Example:
            ```python
            from pydantic import BaseModel
            
            class ExtractedInfo(BaseModel):
                company_name: str
                description: str
                products: list[str]
            
            structured_model = model.with_structured_output(ExtractedInfo)
            result = structured_model.invoke(messages)
            ```
        """
        return CoreLibChatModel(
            client=self.client,
            bound_tools=self.bound_tools,
            structured_output_schema=schema,
            use_search_grounding=self.use_search_grounding,
            thinking_enabled=self.thinking_enabled,
        )
    
    def with_config(
        self,
        *,
        use_search_grounding: Optional[bool] = None,
        thinking_enabled: Optional[bool] = None,
        **kwargs: Any,
    ) -> "CoreLibChatModel":
        """Return a model with modified configuration.
        
        Args:
            use_search_grounding: Enable/disable search grounding
            thinking_enabled: Enable/disable thinking mode
            **kwargs: Additional arguments
            
        Returns:
            New CoreLibChatModel instance with updated configuration
        """
        return CoreLibChatModel(
            client=self.client,
            bound_tools=self.bound_tools,
            structured_output_schema=self.structured_output_schema,
            use_search_grounding=use_search_grounding if use_search_grounding is not None else self.use_search_grounding,
            thinking_enabled=thinking_enabled if thinking_enabled is not None else self.thinking_enabled,
        )


# Convenience function for creating the adapter
def create_langchain_model(
    client: Optional[LLMClient] = None,
    **kwargs: Any,
) -> CoreLibChatModel:
    """Create a LangChain-compatible chat model from core-lib.
    
    This is a convenience function that either wraps an existing LLMClient
    or creates one from environment variables.
    
    Args:
        client: Optional existing LLMClient to wrap. If not provided,
               creates one using create_client_from_env()
        **kwargs: Additional arguments passed to CoreLibChatModel
        
    Returns:
        CoreLibChatModel instance ready for use with LangChain/LangGraph
        
    Example:
        ```python
        from core_lib.llm.langchain_adapter import create_langchain_model
        
        # Auto-create from environment
        model = create_langchain_model()
        
        # Or wrap existing client
        from core_lib.llm import create_openai_client
        client = create_openai_client(model="gpt-4")
        model = create_langchain_model(client=client)
        ```
    """
    if client is None:
        from .factory import create_client_from_env
        client = create_client_from_env()
    
    return CoreLibChatModel(client=client, **kwargs)
