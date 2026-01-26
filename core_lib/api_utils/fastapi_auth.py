"""FastAPI middleware and utilities for authentication.

Provides ready-to-use FastAPI middleware and dependency injection for
securing FastAPI endpoints with time-based HMAC and/or static API key authentication.

Example:
    ```python
    from fastapi import FastAPI, Depends
    from core_lib.api_utils.fastapi_auth import (
        TimeBasedAuthMiddleware,
        verify_auth_dependency
    )
    from core_lib.config import AuthSettings
    
    app = FastAPI()
    settings = AuthSettings.from_env()
    
    # Option 1: Use middleware for all routes
    app.add_middleware(TimeBasedAuthMiddleware, settings=settings)
    
    # Option 2: Use dependency on specific routes
    @app.get("/protected", dependencies=[Depends(verify_auth_dependency)])
    async def protected_route():
        return {"message": "Authenticated!"}
    ```
"""

import hmac
from typing import Optional, Callable

try:
    from fastapi import Request, HTTPException, status, Depends
    from fastapi.security import APIKeyHeader
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import Response, JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    # Type stubs for when FastAPI is not installed
    Request = None  # type: ignore
    HTTPException = None  # type: ignore
    status = None  # type: ignore
    Depends = None  # type: ignore
    APIKeyHeader = None  # type: ignore
    BaseHTTPMiddleware = object  # type: ignore
    Response = None  # type: ignore
    JSONResponse = None  # type: ignore

from .time_based_auth import verify_time_key
from .auth_settings import AuthSettings


def verify_auth_key(provided_key: str, settings: AuthSettings) -> bool:
    """Verify an authentication key against configured auth methods.
    
    Checks the provided key against:
    1. Static API keys (if configured) - exact match
    2. Time-based HMAC keys (if configured) - validated against all private keys
    
    Args:
        provided_key: The authentication key to verify
        settings: AuthSettings instance with configured keys
        
    Returns:
        True if the key is valid via any configured method
    """
    if not provided_key or not provided_key.strip():
        return False
    
    # Check static keys first (faster - just a set lookup)
    if settings.has_static_auth():
        if settings.is_valid_static_key(provided_key):
            return True
    
    # Check time-based HMAC keys
    if settings.has_hmac_auth():
        if verify_time_key(provided_key, settings.auth_private_keys):
            return True
    
    return False


class TimeBasedAuthMiddleware(BaseHTTPMiddleware):  # type: ignore
    """FastAPI middleware for authentication.
    
    Automatically validates authentication keys on all requests using
    either time-based HMAC or static API keys (or both).
    Can be configured with path exclusions for health checks, etc.
    
    Args:
        app: FastAPI application
        settings: AuthSettings instance
        exclude_paths: List of paths to exclude from authentication
        
    Example:
        ```python
        app = FastAPI()
        settings = AuthSettings.from_env(auth_enabled=True)
        
        app.add_middleware(
            TimeBasedAuthMiddleware,
            settings=settings,
            exclude_paths=["/health", "/docs", "/openapi.json"]
        )
        ```
    
    Raises:
        ImportError: If FastAPI is not installed
    """
    
    def __init__(
        self,
        app,
        settings: AuthSettings,
        exclude_paths: Optional[list[str]] = None
    ):
        if not HAS_FASTAPI:
            raise ImportError(
                "FastAPI is required for TimeBasedAuthMiddleware. "
                "Install it with: pip install fastapi"
            )
        super().__init__(app)
        self.settings = settings
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json", "/redoc"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and verify authentication."""
        # Skip auth for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Skip auth if disabled
        if not self.settings.auth_enabled:
            return await call_next(request)
        
        # Get auth key from header
        auth_key = request.headers.get(self.settings.auth_key_header_name)
        
        if not auth_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "detail": f"Missing authentication header: {self.settings.auth_key_header_name}"
                }
            )
        
        # Verify the key using all configured methods
        if not verify_auth_key(auth_key, self.settings):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or expired authentication key"}
            )
        
        # Authentication successful
        response = await call_next(request)
        return response


# Dependency injection for route-level authentication
def create_auth_dependency(settings: AuthSettings):
    """Create a FastAPI dependency for authentication.
    
    Returns a dependency function that can be used with Depends()
    to protect individual routes.
    
    Args:
        settings: AuthSettings instance
        
    Returns:
        Dependency function for use with FastAPI Depends()
        
    Raises:
        ImportError: If FastAPI is not installed
        
    Example:
        ```python
        settings = AuthSettings.from_env()
        verify_auth = create_auth_dependency(settings)
        
        @app.get("/protected")
        async def protected(auth=Depends(verify_auth)):
            return {"message": "Success"}
        ```
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for create_auth_dependency. "
            "Install it with: pip install fastapi"
        )
    
    api_key_header = APIKeyHeader(
        name=settings.auth_key_header_name,
        auto_error=False
    )
    
    async def verify_auth(api_key: Optional[str] = Depends(api_key_header)) -> str:
        """Verify authentication key from header."""
        if not settings.auth_enabled:
            return "disabled"
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Missing authentication header: {settings.auth_key_header_name}"
            )
        
        if not verify_auth_key(api_key, settings):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication key"
            )
        
        return api_key
    
    return verify_auth


# Default dependency using settings from environment
async def verify_auth_dependency(
    request: Request,  # type: ignore
) -> str:
    """Default authentication dependency using environment settings.
    
    This is a convenience dependency that loads settings from environment
    variables automatically. For better performance, create a dependency
    with create_auth_dependency() and reuse the settings instance.
    
    Raises:
        ImportError: If FastAPI is not installed
    
    Example:
        ```python
        @app.get("/protected", dependencies=[Depends(verify_auth_dependency)])
        async def protected_route():
            return {"message": "Authenticated"}
        ```
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for verify_auth_dependency. "
            "Install it with: pip install fastapi"
        )
    
    # Load settings from environment
    settings = AuthSettings.from_env()
    
    if not settings.auth_enabled:
        return "disabled"
    
    # Get auth key from header
    auth_key = request.headers.get(settings.auth_key_header_name)
    
    if not auth_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing authentication header: {settings.auth_key_header_name}"
        )
    
    if not verify_auth_key(auth_key, settings):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication key"
        )
    
    return auth_key
