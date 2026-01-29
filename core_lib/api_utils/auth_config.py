"""Unified authentication configuration for FastAPI applications.

This module provides a unified way to configure authentication across
all Faciliter platform services. It supports multiple authentication modes:

- JWT: Token-based auth using Django-issued JWTs
- LEGACY: Time-based HMAC auth (backward compatible)
- BOTH: Try JWT first, fall back to legacy

Example:
    ```python
    from fastapi import FastAPI
    from core_lib.api_utils.auth_config import configure_app_auth, AuthMode
    from core_lib.api_utils.jwt_auth import JWTAuthSettings
    
    app = FastAPI(docs_url=None, redoc_url=None)  # Disable default docs
    
    # Get settings from environment
    jwt_settings = JWTAuthSettings.from_env()
    
    # Configure authentication (JWT mode with Swagger integration)
    configure_app_auth(
        app,
        jwt_settings=jwt_settings,
        mode=AuthMode.JWT,
        enable_swagger_auth=True,
    )
    ```
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Any
import os
import contextvars

try:
    from fastapi import FastAPI, Request
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import Response, JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = None  # type: ignore
    Request = None  # type: ignore
    BaseHTTPMiddleware = object  # type: ignore
    Response = None  # type: ignore
    JSONResponse = None  # type: ignore

from .jwt_auth import (
    JWTAuthSettings,
    JWTClaims,
    validate_jwt_token,
    extract_bearer_token,
    JWTAuthError,
)
from .auth_settings import AuthSettings
from .time_based_auth import verify_time_key


class AuthMode(str, Enum):
    """Authentication mode for the application.
    
    Attributes:
        JWT: Use JWT tokens issued by Django Admin
        LEGACY: Use time-based HMAC keys (backward compatible)
        BOTH: Try JWT first, fall back to legacy if JWT fails
        NONE: Disable authentication (development only!)
    """
    JWT = "jwt"
    LEGACY = "legacy"
    BOTH = "both"
    NONE = "none"
    
    @classmethod
    def from_env(cls) -> "AuthMode":
        """Get auth mode from AUTH_MODE environment variable."""
        mode = os.environ.get("AUTH_MODE", "both").lower()
        try:
            return cls(mode)
        except ValueError:
            return cls.BOTH


# Context variable for current authenticated claims
_current_claims: contextvars.ContextVar[Optional[JWTClaims]] = contextvars.ContextVar(
    "current_claims", default=None
)

# Context variable for company_id (tenant isolation)
_current_company_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_company_id", default=None
)


def get_current_claims() -> Optional[JWTClaims]:
    """Get the current JWT claims from context.
    
    Returns:
        JWTClaims if authenticated, None otherwise
    """
    return _current_claims.get()


def set_current_claims(claims: Optional[JWTClaims]) -> None:
    """Set the current JWT claims in context.
    
    Args:
        claims: JWTClaims to set (or None to clear)
    """
    _current_claims.set(claims)


def get_current_company_id() -> Optional[str]:
    """Get the current company_id for tenant isolation.
    
    Returns:
        Company ID string if set, None otherwise
    """
    return _current_company_id.get()


def set_current_company_id(company_id: Optional[str]) -> None:
    """Set the current company_id for tenant isolation.
    
    Args:
        company_id: Company ID to set (or None to clear)
    """
    _current_company_id.set(company_id)


def inject_company_context(claims: Optional[JWTClaims], request: Any = None) -> Optional[str]:
    """Inject company_id from claims into context for tenant isolation.
    
    This function:
    1. Extracts company_id from JWT claims
    2. Falls back to company_id query parameter if claims are global
    3. Sets the company_id in contextvars for use by database queries
    
    Args:
        claims: Parsed JWT claims (may be None)
        request: Optional FastAPI Request for query param fallback
        
    Returns:
        The company_id that was set in context (or None)
    """
    company_id = None
    
    if claims:
        company_id = claims.company_id
        set_current_claims(claims)
    
    # If no company_id in claims, check query param
    if not company_id and request and hasattr(request, "query_params"):
        company_id = request.query_params.get("company_id")
    
    if company_id:
        set_current_company_id(company_id)
    
    return company_id


@dataclass
class AuthResult:
    """Result of authentication attempt.
    
    Attributes:
        success: Whether authentication succeeded
        claims: JWT claims if using JWT auth
        method: Which auth method succeeded ("jwt", "legacy", or None)
        error: Error message if authentication failed
    """
    success: bool
    claims: Optional[JWTClaims] = None
    method: Optional[str] = None
    error: Optional[str] = None


def authenticate_request(
    authorization: Optional[str],
    jwt_settings: Optional[JWTAuthSettings] = None,
    legacy_settings: Optional[AuthSettings] = None,
    mode: AuthMode = AuthMode.BOTH,
    legacy_header_value: Optional[str] = None,
) -> AuthResult:
    """Authenticate a request using the specified mode.
    
    Args:
        authorization: Authorization header value
        jwt_settings: Settings for JWT authentication
        legacy_settings: Settings for legacy time-based auth
        mode: Authentication mode to use
        legacy_header_value: Value of legacy auth header (if different from Authorization)
        
    Returns:
        AuthResult with success status and claims/error
    """
    if mode == AuthMode.NONE:
        return AuthResult(success=True, method="none")
    
    # Try JWT authentication
    if mode in (AuthMode.JWT, AuthMode.BOTH):
        token = extract_bearer_token(authorization)
        if token and jwt_settings:
            try:
                claims = validate_jwt_token(token, jwt_settings, verify_type="access")
                return AuthResult(success=True, claims=claims, method="jwt")
            except JWTAuthError as e:
                if mode == AuthMode.JWT:
                    return AuthResult(success=False, error=e.message)
                # Fall through to legacy auth if mode is BOTH
    
    # Try legacy authentication
    if mode in (AuthMode.LEGACY, AuthMode.BOTH):
        auth_key = legacy_header_value
        if auth_key and legacy_settings and legacy_settings.auth_enabled:
            if not legacy_settings.auth_private_key:
                if mode == AuthMode.LEGACY:
                    return AuthResult(
                        success=False,
                        error="Legacy auth private key is not configured",
                    )
                # If BOTH, fall through to allow other auth methods
            else:
                if verify_time_key(auth_key, legacy_settings.auth_private_key):
                    return AuthResult(success=True, method="legacy")
                if mode == AuthMode.LEGACY:
                    return AuthResult(success=False, error="Invalid or expired authentication key")
    
    # No valid authentication found
    if mode == AuthMode.BOTH:
        return AuthResult(success=False, error="No valid authentication provided")
    
    return AuthResult(success=False, error="Authentication required")


if HAS_FASTAPI:
    
    class UnifiedAuthMiddleware(BaseHTTPMiddleware):
        """Unified authentication middleware supporting JWT and legacy auth.
        
        This middleware:
        1. Checks the configured auth mode
        2. Validates tokens/keys according to the mode
        3. Sets claims and company_id in context for downstream use
        4. Excludes configured paths from authentication
        
        Args:
            app: FastAPI application
            jwt_settings: Settings for JWT authentication
            legacy_settings: Settings for legacy time-based auth
            mode: Authentication mode (JWT, LEGACY, BOTH, NONE)
            exclude_paths: Paths to exclude from authentication
        """
        
        def __init__(
            self,
            app,
            jwt_settings: Optional[JWTAuthSettings] = None,
            legacy_settings: Optional[AuthSettings] = None,
            mode: AuthMode = AuthMode.BOTH,
            exclude_paths: Optional[List[str]] = None,
        ):
            super().__init__(app)
            self.jwt_settings = jwt_settings
            self.legacy_settings = legacy_settings
            self.mode = mode
            self.exclude_paths = exclude_paths or [
                "/health",
                "/docs",
                "/openapi.json",
                "/redoc",
            ]

        def _is_excluded_path(self, path: str) -> bool:
            """Return True if path should bypass authentication."""
            for excluded in self.exclude_paths:
                if path == excluded:
                    return True
                # Allow subpaths like /docs/ or /docs/oauth2-redirect
                normalized = excluded.rstrip("/")
                if normalized and path.startswith(f"{normalized}/"):
                    return True
            return False
        
        def _get_route_path(self, scope: dict) -> str:
            """Get the path relative to mount point (mirrors Starlette's logic)."""
            path: str = scope.get("path", "")
            root_path = scope.get("root_path", "")
            if not root_path:
                return path
            if not path.startswith(root_path):
                return path
            if path == root_path:
                return ""
            if path[len(root_path)] == "/":
                return path[len(root_path):]
            return path

        async def dispatch(self, request: Request, call_next) -> Response:
            """Process request with unified authentication."""
            # Get the route path relative to mount point (strips mount prefix)
            path = self._get_route_path(request.scope)
            
            # Skip auth for excluded paths
            if self._is_excluded_path(path):
                return await call_next(request)
            
            # Skip if auth disabled
            if self.mode == AuthMode.NONE:
                return await call_next(request)
            
            # Get auth headers
            authorization = request.headers.get("Authorization")
            legacy_header = None
            if self.legacy_settings:
                legacy_header = request.headers.get(
                    self.legacy_settings.auth_key_header_name
                )
            
            # Also check for token in query params (for Swagger redirect)
            if not authorization:
                token = request.query_params.get("token")
                if token:
                    authorization = f"Bearer {token}"
            
            # Authenticate
            result = authenticate_request(
                authorization=authorization,
                jwt_settings=self.jwt_settings,
                legacy_settings=self.legacy_settings,
                mode=self.mode,
                legacy_header_value=legacy_header,
            )
            
            if not result.success:
                return JSONResponse(
                    status_code=401,
                    content={"detail": result.error or "Authentication required"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Store claims in request state and context
            if result.claims:
                request.state.jwt_claims = result.claims
                inject_company_context(result.claims, request)
            
            request.state.auth_method = result.method
            
            return await call_next(request)
    
    
    def configure_app_auth(
        app: FastAPI,
        jwt_settings: Optional[JWTAuthSettings] = None,
        legacy_settings: Optional[AuthSettings] = None,
        mode: Optional[AuthMode] = None,
        enable_swagger_auth: bool = True,
        exclude_paths: Optional[List[str]] = None,
        openapi_url: str = "/openapi.json",
    ) -> None:
        """Configure authentication for a FastAPI application.
        
        This is the main entry point for setting up authentication.
        It configures:
        1. Authentication middleware (JWT, legacy, or both)
        2. Swagger UI with Bearer auth support
        3. Context injection for tenant isolation
        
        Args:
            app: FastAPI application
            jwt_settings: JWT authentication settings
            legacy_settings: Legacy time-based auth settings
            mode: Auth mode (default: from AUTH_MODE env var)
            enable_swagger_auth: Configure Swagger UI with JWT support
            exclude_paths: Paths to exclude from authentication
            openapi_url: Path to OpenAPI schema (default: "/openapi.json")
            
        Example:
            ```python
            from fastapi import FastAPI
            from core_lib.api_utils import configure_app_auth, JWTAuthSettings
            
            app = FastAPI(docs_url=None, redoc_url=None)
            
            configure_app_auth(
                app,
                jwt_settings=JWTAuthSettings.from_env(),
                enable_swagger_auth=True,
            )
            ```
        """
        # Determine auth mode
        if mode is None:
            mode = AuthMode.from_env()
        
        # Try to load settings from environment if not provided
        if jwt_settings is None and mode in (AuthMode.JWT, AuthMode.BOTH):
            try:
                jwt_settings = JWTAuthSettings.from_env()
            except ValueError:
                if mode == AuthMode.JWT:
                    raise
                # Fall back to legacy-only if JWT secret not configured
                mode = AuthMode.LEGACY
        
        if legacy_settings is None and mode in (AuthMode.LEGACY, AuthMode.BOTH):
            legacy_settings = AuthSettings.from_env()
        
        # Build exclude paths
        default_exclude = ["/health", "/docs", "/openapi.json", "/redoc"]
        if exclude_paths:
            default_exclude = list(set(default_exclude + exclude_paths))
        
        # Add authentication middleware
        if mode != AuthMode.NONE:
            app.add_middleware(
                UnifiedAuthMiddleware,
                jwt_settings=jwt_settings,
                legacy_settings=legacy_settings,
                mode=mode,
                exclude_paths=default_exclude,
            )
        
        # Configure Swagger UI with JWT auth
        if enable_swagger_auth and jwt_settings:
            from .swagger_auth import configure_swagger_jwt_auth
            configure_swagger_jwt_auth(
                app,
                settings=jwt_settings,
                exclude_paths=default_exclude,
                custom_openapi_path=openapi_url,
            )
