"""FastAPI middleware utilities for tracing and logging context."""

import logging
from typing import Any, Optional, Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..tracing.logging_context import LoggingContext
from .. import parse_from

_logger = logging.getLogger(__name__)

# Paths that are excluded from request logging to reduce noise
_SILENT_PATHS = frozenset({
    "/health", "/healthz", "/ready", "/readiness", "/liveness",
    "/metrics", "/favicon.ico",
    "/docs", "/redoc", "/openapi.json",
})


class FromContextMiddleware(BaseHTTPMiddleware):
    """Middleware to inject 'from' context into logging and tracing for every request.
    
    This middleware:
    1. Parses the 'from' query parameter (JSON or key=value pairs)
    2. Pushes it into LoggingContext so all log records include these fields
    3. Adds it to tracing metadata
    4. Stores it on request.state for handlers to access
    
    The 'from' parameter enables request traceability by allowing clients to pass
    contextual information (user_id, session_id, request_id, etc.) that gets
    automatically injected into all logs and traces for that request.
    
    Example:
        ```python
        from fastapi import FastAPI
        from core_lib.api_utils.fastapi_middleware import FromContextMiddleware
        
        app = FastAPI()
        app.add_middleware(FromContextMiddleware, tracing_client=tracing_client)
        ```
    
    Query parameter format:
        - JSON: ?from={"user_id":"123","session_id":"abc"}
        - Key-value pairs: ?from=user_id:123,session_id:abc
    """
    
    def __init__(self, app, tracing_client: Optional[Any] = None):
        """Initialize the middleware.
        
        Args:
            app: The FastAPI application
            tracing_client: Optional tracing client with add_metadata() method
        """
        super().__init__(app)
        self.tracing_client = tracing_client
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and inject 'from' context.
        
        Args:
            request: The incoming request
            call_next: The next middleware/handler in the chain
            
        Returns:
            The response from the handler
        """
        from_raw = request.query_params.get("from")
        from_dict = parse_from(from_raw)
        
        # Extract intelligence_level from query params if present
        intelligence_level_raw = request.query_params.get("intelligence_level")
        if intelligence_level_raw is not None:
            try:
                # Parse as integer and validate range (0-10)
                intelligence_level = int(intelligence_level_raw)
                if 0 <= intelligence_level <= 10:
                    from_dict['intelligence_level'] = intelligence_level
            except (ValueError, TypeError):
                # Ignore invalid intelligence_level values
                pass
        
        # Attach to request state for downstream usage
        try:
            request.state.from_dict = from_dict  # type: ignore[attr-defined]
        except Exception:
            pass
        
        try:
            # Use LoggingContext to inject fields into all log records
            with LoggingContext(from_dict):
                # Add to tracing metadata if tracing client is available
                if self.tracing_client and from_dict:
                    self.tracing_client.add_metadata(metadata=from_dict)

                if request.url.path not in _SILENT_PATHS:
                    _logger.info("%s %s", request.method, str(request.url))

                response = await call_next(request)
            return response
        except Exception:
            # Ensure exceptions propagate while still enriching any emitted logs
            raise


async def inject_from_logging_context(
    request: Request,
    call_next: Callable,
    tracing_client: Optional[Any] = None
):
    """Standalone middleware function to inject 'from' context into logging/tracing.
    
    This is a functional alternative to FromContextMiddleware that can be used
    with @app.middleware("http") decorator.
    
    Parses the query param 'from' (JSON or key=value pairs), converts to dict,
    pushes it into core-lib's LoggingContext so all log records in this
    request include the fields, and adds it to tracing metadata. Also stores
    it on request.state for handlers that want to reuse it.
    
    Additionally extracts 'intelligence_level' from query params if present and
    adds it to the logging context for observability.
    
    Args:
        request: The incoming request
        call_next: The next middleware/handler in the chain
        tracing_client: Optional tracing client with add_metadata() method
        
    Returns:
        The response from the handler
        
    Example:
        ```python
        from fastapi import FastAPI, Request
        from core_lib.api_utils.fastapi_middleware import inject_from_logging_context
        from core_lib.tracing import setup_tracing
        
        app = FastAPI()
        tracing_client = setup_tracing(name="my-app")
        
        @app.middleware("http")
        async def add_from_context(request: Request, call_next):
            return await inject_from_logging_context(request, call_next, tracing_client)
        ```
    """
    from_raw = request.query_params.get("from")
    from_dict = parse_from(from_raw)
    
    # Extract intelligence_level from query params if present
    intelligence_level_raw = request.query_params.get("intelligence_level")
    if intelligence_level_raw is not None:
        try:
            # Parse as integer and validate range (0-10)
            intelligence_level = int(intelligence_level_raw)
            if 0 <= intelligence_level <= 10:
                from_dict['intelligence_level'] = intelligence_level
        except (ValueError, TypeError):
            # Ignore invalid intelligence_level values
            pass
    
    # Attach to request state for downstream usage
    try:
        request.state.from_dict = from_dict  # type: ignore[attr-defined]
    except Exception:
        pass
    
    try:
        with LoggingContext(from_dict):
            if tracing_client and from_dict:
                tracing_client.add_metadata(metadata=from_dict)

            if request.url.path not in _SILENT_PATHS:
                _logger.info("%s %s", request.method, str(request.url))

            response = await call_next(request)
        return response
    except Exception:
        # Ensure exceptions propagate while still enriching any emitted logs
        raise
