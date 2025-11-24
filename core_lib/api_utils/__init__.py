"""API utilities for secure communication between applications.

This module provides time-based authentication using HMAC for secure
communication between FastAPI servers, MCP servers, and clients without
requiring a centralized API key manager.
"""

from .time_based_auth import (
    generate_time_key,
    verify_time_key,
    TimeBasedAuthError,
)
from .auth_settings import AuthSettings
from .api_client import APIClient

# Optional FastAPI integration (only if fastapi is installed)
try:
    from .fastapi_auth import (
        TimeBasedAuthMiddleware,
        create_auth_dependency,
        verify_auth_dependency,
    )
    from .fastapi_openapi import (
        configure_api_key_auth,
        add_custom_security_scheme,
    )
    from .fastapi_middleware import (
        FromContextMiddleware,
        inject_from_logging_context,
    )
    __all_fastapi__ = [
        "TimeBasedAuthMiddleware",
        "create_auth_dependency",
        "verify_auth_dependency",
        "configure_api_key_auth",
        "add_custom_security_scheme",
        "FromContextMiddleware",
        "inject_from_logging_context",
    ]
except ImportError:
    __all_fastapi__ = []

# FastMCP integration (always available)
from .fastmcp_auth import (
    create_auth_middleware,
    verify_mcp_auth,
    get_auth_headers,
    get_auth_env_vars,
    MCPAuthError,
)

# Uvicorn server runner utilities
from .uvicorn_runner import (
    run_uvicorn_server,
    run_uvicorn_from_settings,
)

__all__ = [
    # Core authentication
    "generate_time_key",
    "verify_time_key", 
    "TimeBasedAuthError",
    "AuthSettings",
    "APIClient",
    
    # FastMCP integration
    "create_auth_middleware",
    "verify_mcp_auth",
    "get_auth_headers",
    "get_auth_env_vars",
    "MCPAuthError",
    
    # Uvicorn runner
    "run_uvicorn_server",
    "run_uvicorn_from_settings",
] + __all_fastapi__
