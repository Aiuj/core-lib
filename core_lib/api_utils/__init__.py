"""API utilities for secure communication between applications.

This module provides time-based authentication using HMAC for secure
communication between FastAPI servers, MCP servers, and clients without
requiring a centralized API key manager.

Also provides JWT-based authentication for Django-issued tokens.
"""

from .time_based_auth import (
    generate_time_key,
    verify_time_key,
    TimeBasedAuthError,
)
from .auth_settings import AuthSettings
from .api_client import APIClient

# JWT authentication (always available)
from .jwt_auth import (
    JWTAuthSettings,
    JWTAuthError,
    JWTClaims,
    create_jwt_token,
    validate_jwt_token,
    extract_bearer_token,
    get_company_id_from_claims,
    validate_company_access,
)

__all_jwt__ = [
    "JWTAuthSettings",
    "JWTAuthError",
    "JWTClaims",
    "create_jwt_token",
    "validate_jwt_token",
    "extract_bearer_token",
    "get_company_id_from_claims",
    "validate_company_access",
]

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
    from .jwt_auth import (
        JWTAuthMiddleware,
        get_jwt_claims,
        require_jwt_auth,
        require_superuser,
        require_scope,
        require_any_scope,
    )
    from .swagger_auth import (
        configure_swagger_jwt_auth,
        get_swagger_launch_url,
    )
    __all_fastapi__ = [
        "TimeBasedAuthMiddleware",
        "create_auth_dependency",
        "verify_auth_dependency",
        "configure_api_key_auth",
        "add_custom_security_scheme",
        "FromContextMiddleware",
        "inject_from_logging_context",
        # JWT FastAPI integration
        "JWTAuthMiddleware",
        "get_jwt_claims",
        "require_jwt_auth",
        "require_superuser",
        "require_scope",
        "require_any_scope",
        # Swagger JWT integration
        "configure_swagger_jwt_auth",
        "get_swagger_launch_url",
    ]
except ImportError:
    __all_fastapi__ = []

# FastMCP integration (always available)
from .fastmcp_auth import (
    create_auth_middleware,
    create_jwt_auth_middleware,
    create_unified_auth_middleware,
    verify_mcp_auth,
    get_auth_headers,
    get_auth_env_vars,
    get_mcp_claims,
    get_mcp_company_id,
    MCPAuthError,
)

# Auth configuration (always available)
from .auth_config import (
    AuthMode,
    AuthResult,
    authenticate_request,
    get_current_claims,
    set_current_claims,
    get_current_company_id,
    set_current_company_id,
    inject_company_context,
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
    "create_jwt_auth_middleware",
    "create_unified_auth_middleware",
    "verify_mcp_auth",
    "get_auth_headers",
    "get_auth_env_vars",
    "get_mcp_claims",
    "get_mcp_company_id",
    "MCPAuthError",
    
    # Auth configuration
    "AuthMode",
    "AuthResult",
    "authenticate_request",
    "get_current_claims",
    "set_current_claims",
    "get_current_company_id",
    "set_current_company_id",
    "inject_company_context",
    
    # Uvicorn runner
    "run_uvicorn_server",
    "run_uvicorn_from_settings",
] + __all_jwt__ + __all_fastapi__
