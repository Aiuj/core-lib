"""JWT-based authentication for service-to-service communication.

This module provides JWT (JSON Web Token) authentication for securing
FastAPI and MCP endpoints. Tokens are issued by Django Admin and validated
by services using a shared JWT secret.

Token Types:
    - Access Token (1 hour): Used for API calls
    - Refresh Token (24 hours): Used to obtain new access tokens

Claims Structure:
    - sub: client_id of the ServiceAccount
    - company_id: Optional tenant identifier (None for global accounts)
    - is_superuser: Whether the issuing user was a superuser
    - scopes: List of permission scopes
    - jti: Unique token identifier
    - token_type: "access" or "refresh"
    - exp: Expiration timestamp
    - iat: Issued at timestamp
    - iss: Issuer identifier

Example:
    Server side:
        ```python
        from fastapi import FastAPI, Depends
        from core_lib.api_utils.jwt_auth import (
            JWTAuthMiddleware,
            JWTAuthSettings,
            require_jwt_auth,
            get_jwt_claims,
        )
        
        app = FastAPI()
        settings = JWTAuthSettings.from_env()
        
        # Option 1: Middleware for all routes
        app.add_middleware(JWTAuthMiddleware, settings=settings)
        
        # Option 2: Dependency on specific routes
        @app.get("/protected")
        async def protected(claims: dict = Depends(require_jwt_auth)):
            return {"company_id": claims.get("company_id")}
        ```
    
    Client side:
        ```python
        import httpx
        
        # Get token from Django Admin "Launch Tool" or token endpoint
        token = "eyJ..."
        
        headers = {"Authorization": f"Bearer {token}"}
        response = httpx.get("http://service/api/endpoint", headers=headers)
        ```
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Any, Callable, Dict, List
import os

try:
    import jwt
    from jwt.exceptions import (
        InvalidTokenError,
        ExpiredSignatureError,
        InvalidSignatureError,
    )
    HAS_JWT = True
except ImportError:
    HAS_JWT = False
    jwt = None  # type: ignore
    InvalidTokenError = Exception  # type: ignore
    ExpiredSignatureError = Exception  # type: ignore
    InvalidSignatureError = Exception  # type: ignore

try:
    from fastapi import Request, HTTPException, status, Depends
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import Response, JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    Request = None  # type: ignore
    HTTPException = None  # type: ignore
    status = None  # type: ignore
    Depends = None  # type: ignore
    HTTPBearer = None  # type: ignore
    HTTPAuthorizationCredentials = None  # type: ignore
    BaseHTTPMiddleware = object  # type: ignore
    Response = None  # type: ignore
    JSONResponse = None  # type: ignore


class JWTAuthError(Exception):
    """Raised when JWT authentication fails."""
    
    def __init__(self, message: str, code: str = "auth_error"):
        self.message = message
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class JWTAuthSettings:
    """Settings for JWT authentication.
    
    Attributes:
        jwt_secret: Secret key for signing/verifying tokens (REQUIRED)
        jwt_algorithm: Algorithm for JWT signing (default: HS256)
        access_token_ttl: Access token lifetime in seconds (default: 3600 = 1 hour)
        refresh_token_ttl: Refresh token lifetime in seconds (default: 86400 = 24 hours)
        issuer: Token issuer identifier (default: "faciliter")
        require_auth: Whether authentication is required (default: True)
        check_revocation: Whether to check token revocation (default: False)
        revocation_check_url: URL to check token revocation (optional)
    """
    
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    access_token_ttl: int = 3600  # 1 hour
    refresh_token_ttl: int = 86400  # 24 hours
    issuer: str = "faciliter"
    require_auth: bool = True
    check_revocation: bool = False
    revocation_check_url: Optional[str] = None
    
    # Paths to exclude from authentication
    exclude_paths: List[str] = field(default_factory=lambda: [
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
    ])
    
    @classmethod
    def from_env(
        cls,
        jwt_secret: Optional[str] = None,
        jwt_algorithm: Optional[str] = None,
        access_token_ttl: Optional[int] = None,
        refresh_token_ttl: Optional[int] = None,
        issuer: Optional[str] = None,
        require_auth: Optional[bool] = None,
        **kwargs
    ) -> "JWTAuthSettings":
        """Create settings from environment variables.
        
        Environment Variables:
            JWT_SECRET: Secret key for signing tokens (REQUIRED)
            JWT_ALGORITHM: Signing algorithm (default: HS256)
            JWT_ACCESS_TOKEN_TTL: Access token lifetime in seconds
            JWT_REFRESH_TOKEN_TTL: Refresh token lifetime in seconds
            JWT_ISSUER: Token issuer identifier
            JWT_REQUIRE_AUTH: Whether auth is required (true/false)
        
        Args:
            jwt_secret: Override for JWT_SECRET
            jwt_algorithm: Override for JWT_ALGORITHM
            access_token_ttl: Override for JWT_ACCESS_TOKEN_TTL
            refresh_token_ttl: Override for JWT_REFRESH_TOKEN_TTL
            issuer: Override for JWT_ISSUER
            require_auth: Override for JWT_REQUIRE_AUTH
            **kwargs: Additional settings passed through
            
        Returns:
            JWTAuthSettings instance
            
        Raises:
            ValueError: If JWT_SECRET is not set
        """
        secret = jwt_secret or os.environ.get("JWT_SECRET")
        if not secret:
            raise ValueError(
                "JWT_SECRET environment variable is required for JWT authentication"
            )
        
        def parse_bool(val: Optional[str], default: bool) -> bool:
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes")
        
        return cls(
            jwt_secret=secret,
            jwt_algorithm=jwt_algorithm or os.environ.get("JWT_ALGORITHM", "HS256"),
            access_token_ttl=access_token_ttl or int(
                os.environ.get("JWT_ACCESS_TOKEN_TTL", "3600")
            ),
            refresh_token_ttl=refresh_token_ttl or int(
                os.environ.get("JWT_REFRESH_TOKEN_TTL", "86400")
            ),
            issuer=issuer or os.environ.get("JWT_ISSUER", "faciliter"),
            require_auth=require_auth if require_auth is not None else parse_bool(
                os.environ.get("JWT_REQUIRE_AUTH"), True
            ),
            **kwargs
        )


@dataclass
class JWTClaims:
    """Parsed JWT claims with typed access.
    
    Provides convenient access to standard and custom claims
    with proper typing.
    """
    
    sub: str  # Subject (client_id)
    exp: datetime  # Expiration time
    iat: datetime  # Issued at
    jti: str  # JWT ID
    token_type: str  # "access" or "refresh"
    iss: str  # Issuer
    company_id: Optional[str] = None  # Tenant identifier
    is_superuser: bool = False  # Whether issuing user was superuser
    scopes: List[str] = field(default_factory=list)  # Permission scopes
    user_id: Optional[str] = None  # Issuing user ID (for audit)
    
    @classmethod
    def from_dict(cls, claims: Dict[str, Any]) -> "JWTClaims":
        """Create JWTClaims from a dictionary.
        
        Args:
            claims: Dictionary of JWT claims
            
        Returns:
            JWTClaims instance
        """
        # Handle datetime conversion
        exp = claims.get("exp")
        if isinstance(exp, (int, float)):
            exp = datetime.fromtimestamp(exp, tz=timezone.utc)
        
        iat = claims.get("iat")
        if isinstance(iat, (int, float)):
            iat = datetime.fromtimestamp(iat, tz=timezone.utc)
        
        return cls(
            sub=claims.get("sub", ""),
            exp=exp or datetime.now(timezone.utc),
            iat=iat or datetime.now(timezone.utc),
            jti=claims.get("jti", ""),
            token_type=claims.get("token_type", "access"),
            iss=claims.get("iss", ""),
            company_id=claims.get("company_id"),
            is_superuser=claims.get("is_superuser", False),
            scopes=claims.get("scopes", []),
            user_id=claims.get("user_id"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert claims to dictionary for token creation."""
        return {
            "sub": self.sub,
            "exp": self.exp,
            "iat": self.iat,
            "jti": self.jti,
            "token_type": self.token_type,
            "iss": self.iss,
            "company_id": self.company_id,
            "is_superuser": self.is_superuser,
            "scopes": self.scopes,
            "user_id": self.user_id,
        }
    
    def has_scope(self, scope: str) -> bool:
        """Check if claims include a specific scope.
        
        Args:
            scope: Scope to check (e.g., "rfx:write")
            
        Returns:
            True if scope is present
        """
        return scope in self.scopes
    
    def has_any_scope(self, scopes: List[str]) -> bool:
        """Check if claims include any of the given scopes."""
        return any(s in self.scopes for s in scopes)
    
    def has_all_scopes(self, scopes: List[str]) -> bool:
        """Check if claims include all of the given scopes."""
        return all(s in self.scopes for s in scopes)


def create_jwt_token(
    claims: Dict[str, Any],
    settings: JWTAuthSettings,
    token_type: str = "access",
) -> str:
    """Create a signed JWT token.
    
    Args:
        claims: Token claims (sub, company_id, scopes, etc.)
        settings: JWT settings with secret and algorithm
        token_type: "access" or "refresh"
        
    Returns:
        Signed JWT token string
        
    Raises:
        ImportError: If PyJWT is not installed
    """
    if not HAS_JWT:
        raise ImportError("PyJWT is required for JWT authentication. Install: pip install pyjwt")
    
    import uuid
    
    now = datetime.now(timezone.utc)
    ttl = settings.access_token_ttl if token_type == "access" else settings.refresh_token_ttl
    
    token_claims = {
        **claims,
        "iat": now,
        "exp": now + timedelta(seconds=ttl),
        "iss": settings.issuer,
        "jti": claims.get("jti") or str(uuid.uuid4()),
        "token_type": token_type,
    }
    
    return jwt.encode(
        token_claims,
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def validate_jwt_token(
    token: str,
    settings: JWTAuthSettings,
    verify_type: Optional[str] = None,
) -> JWTClaims:
    """Validate a JWT token and return its claims.
    
    Args:
        token: JWT token string
        settings: JWT settings with secret and algorithm
        verify_type: Optional token type to verify ("access" or "refresh")
        
    Returns:
        JWTClaims instance with parsed claims
        
    Raises:
        JWTAuthError: If token is invalid, expired, or wrong type
        ImportError: If PyJWT is not installed
    """
    if not HAS_JWT:
        raise ImportError("PyJWT is required for JWT authentication. Install: pip install pyjwt")
    
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            issuer=settings.issuer,
            options={
                "require": ["exp", "iat", "sub", "jti"],
            },
        )
    except ExpiredSignatureError:
        raise JWTAuthError("Token has expired", code="token_expired")
    except InvalidSignatureError:
        raise JWTAuthError("Invalid token signature", code="invalid_signature")
    except InvalidTokenError as e:
        raise JWTAuthError(f"Invalid token: {e}", code="invalid_token")
    
    # Verify token type if specified
    if verify_type and claims.get("token_type") != verify_type:
        raise JWTAuthError(
            f"Expected {verify_type} token, got {claims.get('token_type')}",
            code="wrong_token_type"
        )
    
    return JWTClaims.from_dict(claims)


def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Extract token from Authorization header.
    
    Args:
        authorization: Authorization header value
        
    Returns:
        Token string if valid Bearer format, None otherwise
    """
    if not authorization:
        return None
    
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    
    return parts[1]


# =============================================================================
# FastAPI Integration
# =============================================================================

if HAS_FASTAPI:
    
    class JWTAuthMiddleware(BaseHTTPMiddleware):
        """FastAPI middleware for JWT authentication.
        
        Validates JWT tokens on all requests except excluded paths.
        Stores parsed claims in request.state.jwt_claims for access
        by route handlers.
        
        Args:
            app: FastAPI application
            settings: JWTAuthSettings instance
            
        Example:
            ```python
            app = FastAPI()
            settings = JWTAuthSettings.from_env()
            app.add_middleware(JWTAuthMiddleware, settings=settings)
            
            @app.get("/protected")
            async def protected(request: Request):
                claims = request.state.jwt_claims
                return {"company_id": claims.company_id}
            ```
        """
        
        def __init__(self, app, settings: JWTAuthSettings):
            super().__init__(app)
            self.settings = settings
        
        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            """Process request and validate JWT."""
            # Skip auth for excluded paths
            if request.url.path in self.settings.exclude_paths:
                return await call_next(request)
            
            # Skip if auth not required
            if not self.settings.require_auth:
                return await call_next(request)
            
            # Extract token from header
            authorization = request.headers.get("Authorization")
            token = extract_bearer_token(authorization)
            
            # Also check query param for Swagger UI token injection
            if not token:
                token = request.query_params.get("token")
            
            if not token:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Missing authentication token"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            try:
                claims = validate_jwt_token(token, self.settings, verify_type="access")
                
                # Check revocation if enabled
                if self.settings.check_revocation:
                    # TODO: Implement revocation check via HTTP call
                    pass
                
                # Store claims in request state
                request.state.jwt_claims = claims
                
            except JWTAuthError as e:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": e.message, "code": e.code},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return await call_next(request)
    
    
    # HTTP Bearer security scheme for OpenAPI
    _bearer_scheme = HTTPBearer(auto_error=False)
    
    
    async def get_jwt_claims(
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    ) -> Optional[JWTClaims]:
        """FastAPI dependency for optional JWT authentication.
        
        Returns claims if valid token provided, None otherwise.
        Use this for endpoints where auth is optional.
        
        Example:
            ```python
            @app.get("/public-or-private")
            async def endpoint(claims: Optional[JWTClaims] = Depends(get_jwt_claims)):
                if claims:
                    return {"user": claims.sub}
                return {"message": "Anonymous access"}
            ```
        """
        # Try to get claims from request state (set by middleware)
        if hasattr(request.state, "jwt_claims"):
            return request.state.jwt_claims
        
        # Try to extract from credentials
        if credentials:
            try:
                settings = JWTAuthSettings.from_env()
                return validate_jwt_token(credentials.credentials, settings)
            except (JWTAuthError, ValueError):
                pass
        
        # Check query param fallback
        token = request.query_params.get("token")
        if token:
            try:
                settings = JWTAuthSettings.from_env()
                return validate_jwt_token(token, settings)
            except (JWTAuthError, ValueError):
                pass
        
        return None
    
    
    async def require_jwt_auth(
        claims: Optional[JWTClaims] = Depends(get_jwt_claims),
    ) -> JWTClaims:
        """FastAPI dependency requiring valid JWT authentication.
        
        Raises HTTPException(401) if no valid token.
        
        Example:
            ```python
            @app.get("/protected")
            async def protected(claims: JWTClaims = Depends(require_jwt_auth)):
                return {"company_id": claims.company_id}
            ```
        """
        if not claims:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return claims
    
    
    async def require_superuser(
        claims: JWTClaims = Depends(require_jwt_auth),
    ) -> JWTClaims:
        """FastAPI dependency requiring superuser token.
        
        Raises HTTPException(403) if token is not from superuser.
        
        Example:
            ```python
            @app.delete("/admin-only")
            async def admin_only(claims: JWTClaims = Depends(require_superuser)):
                return {"message": "Admin access granted"}
            ```
        """
        if not claims.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Superuser access required",
            )
        return claims
    
    
    def require_scope(scope: str):
        """Create a dependency requiring a specific scope.
        
        Args:
            scope: Required scope (e.g., "rfx:write")
            
        Returns:
            FastAPI dependency function
            
        Example:
            ```python
            @app.post("/rfx/upload")
            async def upload(claims: JWTClaims = Depends(require_scope("rfx:write"))):
                return {"message": "Upload permitted"}
            ```
        """
        async def dependency(claims: JWTClaims = Depends(require_jwt_auth)) -> JWTClaims:
            if not claims.has_scope(scope):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required scope: {scope}",
                )
            return claims
        return dependency
    
    
    def require_any_scope(*scopes: str):
        """Create a dependency requiring any of the given scopes.
        
        Args:
            *scopes: Required scopes (at least one must be present)
            
        Returns:
            FastAPI dependency function
        """
        async def dependency(claims: JWTClaims = Depends(require_jwt_auth)) -> JWTClaims:
            if not claims.has_any_scope(list(scopes)):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required scope. Need one of: {', '.join(scopes)}",
                )
            return claims
        return dependency


# =============================================================================
# Context Injection for Multi-Tenancy
# =============================================================================

def get_company_id_from_claims(claims: Optional[JWTClaims]) -> Optional[str]:
    """Extract company_id from JWT claims for tenant isolation.
    
    Args:
        claims: Parsed JWT claims (or None)
        
    Returns:
        Company ID string if present, None otherwise
    """
    if claims is None:
        return None
    return claims.company_id


def validate_company_access(
    claims: JWTClaims,
    requested_company_id: Optional[str],
) -> str:
    """Validate that the token allows access to the requested company.
    
    Args:
        claims: Parsed JWT claims
        requested_company_id: Company ID from request (query param, body, etc.)
        
    Returns:
        The company_id to use (from token or request)
        
    Raises:
        JWTAuthError: If access is denied
    """
    token_company_id = claims.company_id
    
    # Superuser with no company in token can access any company
    if claims.is_superuser and token_company_id is None:
        if not requested_company_id:
            raise JWTAuthError(
                "company_id is required for superuser global tokens",
                code="missing_company_id"
            )
        return requested_company_id
    
    # Token has company_id - must match or be superuser
    if token_company_id:
        if requested_company_id and requested_company_id != token_company_id:
            if not claims.is_superuser:
                raise JWTAuthError(
                    f"Token restricted to company {token_company_id}",
                    code="company_mismatch"
                )
        return token_company_id
    
    # No company in token and not superuser - require company_id in request
    if not requested_company_id:
        raise JWTAuthError(
            "company_id is required",
            code="missing_company_id"
        )
    
    return requested_company_id
