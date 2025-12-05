"""Service usage tracking for AI services (LLM, embeddings, OCR).

This module provides utilities to log detailed service usage metrics to OpenTelemetry/OpenSearch
without requiring Langfuse span management. All metrics are sent as structured log events that
can be queried and analyzed in OpenSearch dashboards.

Features:
- Track LLM usage: provider, model, input/output tokens, cost
- Track embedding usage: provider, model, input tokens/texts, dimensions, cost
- Track OCR usage: provider, model, pages/images processed, cost
- Automatic cost calculation based on known pricing
- No span management required - uses standard logging infrastructure
- Full integration with existing OTLP/OpenSearch logging

Usage:
    ```python
    from core_lib.tracing.service_usage import log_llm_usage, log_embedding_usage
    
    # In LLM provider after API call
    log_llm_usage(
        provider="openai",
        model="gpt-4",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        metadata={"user_id": "user-123"}
    )
    
    # In embedding provider after API call
    log_embedding_usage(
        provider="openai",
        model="text-embedding-3-small",
        input_tokens=500,
        num_texts=10,
        embedding_dim=1536
    )
    ```

The logs are sent to OpenSearch with structured attributes that enable queries like:
- Total tokens used per user/session/company
- Cost breakdown by service and model
- Usage trends over time
- Performance metrics (latency, tokens per second)
"""

import time
from typing import Any, Dict, Optional
from enum import Enum

from .logger import get_module_logger
from .service_pricing import get_llm_pricing, get_embedding_pricing

logger = get_module_logger()


class ServiceType(str, Enum):
    """Types of AI services tracked."""
    LLM = "llm"
    EMBEDDING = "embedding"
    OCR = "ocr"
    SEARCH = "search"


def calculate_llm_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate cost for LLM usage.
    
    Args:
        provider: Provider name (e.g., "openai", "google-gemini")
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Estimated cost in USD
    """
    pricing = get_llm_pricing(model)
    
    if pricing is None:
        # Unknown model, return 0 and log warning
        logger.debug(f"No pricing data for model: {model} (provider: {provider})")
        return 0.0
    
    input_cost = (input_tokens / 1000.0) * pricing["input"]
    output_cost = (output_tokens / 1000.0) * pricing["output"]
    return input_cost + output_cost


def calculate_embedding_cost(
    provider: str,
    model: str,
    input_tokens: int,
) -> float:
    """Calculate cost for embedding generation.
    
    Args:
        provider: Provider name (e.g., "openai", "infinity")
        model: Model name
        input_tokens: Number of input tokens
        
    Returns:
        Estimated cost in USD
    """
    price_per_1k = get_embedding_pricing(model)
    
    if price_per_1k is None:
        logger.debug(f"No pricing data for embedding model: {model} (provider: {provider})")
        return 0.0
    
    return (input_tokens / 1000.0) * price_per_1k


def log_llm_usage(
    provider: str,
    model: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    latency_ms: Optional[float] = None,
    structured: bool = False,
    has_tools: bool = False,
    search_grounding: bool = False,
    host: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Log LLM usage to OpenTelemetry/OpenSearch.
    
    This function creates a structured log event with service usage metrics that
    can be queried in OpenSearch dashboards for cost analysis, performance monitoring,
    and usage trends.
    
    Args:
        provider: Provider name (e.g., "openai", "google-gemini", "ollama")
        model: Model name (e.g., "gpt-4", "gemini-1.5-pro")
        input_tokens: Number of input tokens (prompt)
        output_tokens: Number of output tokens (completion)
        total_tokens: Total tokens (input + output), computed if not provided
        latency_ms: Request latency in milliseconds
        structured: Whether structured output was requested
        has_tools: Whether tool/function calling was used
        search_grounding: Whether search grounding was enabled
        host: Service host URL (e.g., "http://localhost:11434" for Ollama,
              "https://api.openai.com" for OpenAI)
        metadata: Additional context (user_id, session_id, etc.) - automatically
                 included from LoggingContext if set
        error: Error message if the request failed
        
    Example:
        ```python
        log_llm_usage(
            provider="openai",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            latency_ms=1500,
            metadata={"endpoint": "/api/chat"}
        )
        ```
    """
    # Calculate total if not provided
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    
    # Calculate cost
    cost = 0.0
    if input_tokens is not None and output_tokens is not None:
        cost = calculate_llm_cost(provider, model, input_tokens, output_tokens)
    
    # Build structured event
    event = {
        "service.type": ServiceType.LLM.value,
        "service.provider": provider,
        "service.model": model,
        "gen_ai.request.model": model,  # OpenTelemetry semantic convention
        "gen_ai.system": provider,
        "cost_usd": round(cost, 6),  # Always include cost (0.0 if unknown)
    }
    
    if host:
        event["gen_ai.host"] = host
    
    if input_tokens is not None:
        event["gen_ai.usage.input_tokens"] = input_tokens
        event["tokens.input"] = input_tokens
    
    if output_tokens is not None:
        event["gen_ai.usage.output_tokens"] = output_tokens
        event["tokens.output"] = output_tokens
    
    if total_tokens is not None:
        event["tokens.total"] = total_tokens
    
    if latency_ms is not None:
        event["latency_ms"] = latency_ms
        # Calculate tokens per second if we have the data
        if total_tokens and latency_ms > 0:
            event["tokens_per_second"] = (total_tokens / latency_ms) * 1000
    
    # Feature flags - convert booleans to strings for OTLP compatibility
    event["features.structured_output"] = str(structured).lower()
    event["features.tools"] = str(has_tools).lower()
    event["features.search_grounding"] = str(search_grounding).lower()
    
    # Add custom metadata
    if metadata:
        for key, value in metadata.items():
            # Avoid overwriting standard fields
            if key not in event:
                event[f"custom.{key}"] = value
    
    if error:
        event["error"] = error
        event["status"] = "error"
    else:
        event["status"] = "success"
    
    # Log as INFO level with extra_attrs
    # The LoggingContextFilter will automatically add user_id, session_id, etc.
    # The OTLPHandler will send this to OpenSearch
    logger.info(
        f"LLM usage: {provider}/{model} - {total_tokens or 0} tokens, ${cost:.6f}",
        extra={"extra_attrs": event}
    )


def log_embedding_usage(
    provider: str,
    model: str,
    input_tokens: Optional[int] = None,
    num_texts: Optional[int] = None,
    embedding_dim: Optional[int] = None,
    latency_ms: Optional[float] = None,
    host: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Log embedding generation usage to OpenTelemetry/OpenSearch.
    
    Args:
        provider: Provider name (e.g., "openai", "infinity", "ollama")
        model: Model name (e.g., "text-embedding-3-small")
        input_tokens: Number of input tokens processed
        num_texts: Number of text chunks embedded
        embedding_dim: Embedding dimension
        latency_ms: Request latency in milliseconds
        host: Service host URL (e.g., "http://localhost:7997" for Infinity,
              "https://api.openai.com" for OpenAI)
        metadata: Additional context
        error: Error message if the request failed
        
    Example:
        ```python
        log_embedding_usage(
            provider="openai",
            model="text-embedding-3-small",
            input_tokens=500,
            num_texts=10,
            embedding_dim=1536,
            latency_ms=250
        )
        ```
    """
    cost = 0.0
    if input_tokens is not None:
        cost = calculate_embedding_cost(provider, model, input_tokens)
    
    event = {
        "service.type": ServiceType.EMBEDDING.value,
        "service.provider": provider,
        "service.model": model,
        "gen_ai.request.model": model,
        "gen_ai.system": provider,
        "cost_usd": round(cost, 6),  # Always include cost (0.0 if unknown)
    }
    
    if host:
        event["gen_ai.host"] = host
    
    if input_tokens is not None:
        event["tokens.input"] = input_tokens
        event["gen_ai.usage.input_tokens"] = input_tokens
    
    if num_texts is not None:
        event["embedding.num_texts"] = num_texts
    
    if embedding_dim is not None:
        event["embedding.dimension"] = embedding_dim
    
    if latency_ms is not None:
        event["latency_ms"] = latency_ms
        if num_texts and latency_ms > 0:
            event["texts_per_second"] = (num_texts / latency_ms) * 1000
    
    if metadata:
        for key, value in metadata.items():
            if key not in event:
                event[f"custom.{key}"] = value
    
    if error:
        event["error"] = error
        event["status"] = "error"
    else:
        event["status"] = "success"
    
    logger.info(
        f"Embedding usage: {provider}/{model} - {num_texts or 0} texts, {input_tokens or 0} tokens, ${cost:.6f}",
        extra={"extra_attrs": event}
    )


def log_ocr_usage(
    provider: str,
    model: str,
    num_pages: Optional[int] = None,
    num_images: Optional[int] = None,
    latency_ms: Optional[float] = None,
    cost_override: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Log OCR usage to OpenTelemetry/OpenSearch.
    
    Args:
        provider: Provider name (e.g., "azure-di", "google-vision", "tesseract")
        model: Model/engine name
        num_pages: Number of pages processed
        num_images: Number of images processed
        latency_ms: Request latency in milliseconds
        cost_override: Manual cost if automatic calculation not available
        metadata: Additional context
        error: Error message if the request failed
        
    Example:
        ```python
        log_ocr_usage(
            provider="azure-di",
            model="prebuilt-read",
            num_pages=5,
            latency_ms=3000,
            cost_override=0.05
        )
        ```
    """
    event = {
        "service.type": ServiceType.OCR.value,
        "service.provider": provider,
        "service.model": model,
    }
    
    if num_pages is not None:
        event["ocr.num_pages"] = num_pages
    
    if num_images is not None:
        event["ocr.num_images"] = num_images
    
    if latency_ms is not None:
        event["latency_ms"] = latency_ms
        if num_pages and latency_ms > 0:
            event["pages_per_second"] = (num_pages / latency_ms) * 1000
    
    if cost_override is not None:
        event["cost_usd"] = round(cost_override, 6)
    
    if metadata:
        for key, value in metadata.items():
            if key not in event:
                event[f"custom.{key}"] = value
    
    if error:
        event["error"] = error
        event["status"] = "error"
    else:
        event["status"] = "success"
    
    cost_str = f"${cost_override:.6f}" if cost_override else "unknown cost"
    logger.info(
        f"OCR usage: {provider}/{model} - {num_pages or 0} pages, {cost_str}",
        extra={"extra_attrs": event}
    )


def log_search_usage(
    provider: str,
    query: str,
    num_results: int,
    max_results: int,
    latency_ms: Optional[float] = None,
    cache_hit: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Log search usage to OpenTelemetry/OpenSearch.
    
    This function creates a structured log event with search usage metrics that
    can be queried in OpenSearch dashboards for usage analysis and monitoring.
    
    Args:
        provider: Provider name (e.g., "duckduckgo", "serpapi", "tavily", "ollama")
        query: Search query string (truncated for privacy)
        num_results: Number of results returned
        max_results: Maximum results requested
        latency_ms: Request latency in milliseconds
        cache_hit: Whether results were served from cache
        metadata: Additional context (user_id, session_id, etc.)
        error: Error message if the request failed
        
    Example:
        ```python
        log_search_usage(
            provider="duckduckgo",
            query="python asyncio tutorial",
            num_results=10,
            max_results=10,
            latency_ms=350,
            cache_hit=False
        )
        ```
    """
    # Truncate query for privacy (keep first 100 chars)
    query_truncated = query[:100] if len(query) > 100 else query
    query_length = len(query)
    
    event = {
        "service.type": ServiceType.SEARCH.value,
        "service.provider": provider,
        "search.query_length": query_length,
        "search.num_results": num_results,
        "search.max_results": max_results,
        "search.cache_hit": str(cache_hit).lower(),
    }
    
    if latency_ms is not None:
        event["latency_ms"] = latency_ms
        if num_results and latency_ms > 0:
            event["results_per_second"] = (num_results / latency_ms) * 1000
    
    if metadata:
        for key, value in metadata.items():
            if key not in event:
                event[f"custom.{key}"] = value
    
    if error:
        event["error"] = error
        event["status"] = "error"
    else:
        event["status"] = "success"
    
    cache_str = " (cached)" if cache_hit else ""
    latency_str = f", {latency_ms:.0f}ms" if latency_ms else ""
    logger.info(
        f"Search usage: {provider} - '{query_truncated}' -> {num_results}/{max_results} results{cache_str}{latency_str}",
        extra={"extra_attrs": event}
    )


__all__ = [
    "ServiceType",
    "log_llm_usage",
    "log_embedding_usage",
    "log_ocr_usage",
    "log_search_usage",
    "calculate_llm_cost",
    "calculate_embedding_cost",
]
