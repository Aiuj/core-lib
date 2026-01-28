"""FastMCP v2 utilities for time-based and JWT authentication.

Provides utilities for securing FastMCP v2 servers with either:
1. Time-based HMAC authentication (legacy)
2. JWT Bearer token authentication (Django-issued)

Supports both SSE and stdio transports.

Example:
    Server side (FastMCP v2 with JWT):
        ```python
        from mcp import FastMCP
        from core_lib.api_utils.fastmcp_auth import create_jwt_auth_middleware
        from core_lib.api_utils.jwt_auth import JWTAuthSettings
        
        mcp = FastMCP("My MCP Server")
        settings = JWTAuthSettings.from_env()
        
        # Add JWT authentication middleware
        auth_middleware = create_jwt_auth_middleware(settings)
        mcp.add_middleware(auth_middleware)
        
        @mcp.tool()
        def my_tool():
            # Access company_id from context
            from core_lib.api_utils.auth_config import get_current_company_id
            company_id = get_current_company_id()
            return f"Authenticated for company {company_id}"
        ```
    
    Server side (FastMCP v2 with legacy auth):
        ```python
        from mcp import FastMCP
        from core_lib.api_utils.fastmcp_auth import create_auth_middleware
        from core_lib.config import AuthSettings
        
        mcp = FastMCP("My MCP Server")
        settings = AuthSettings.from_env()
        
        # Add legacy authentication middleware
        auth_middleware = create_auth_middleware(settings)
        mcp.add_middleware(auth_middleware)
        ```
    
    Client side (JWT):
        ```python
        # Get token from Django Admin "Launch Tool"
        token = "eyJ..."
        
        # For SSE transport
        headers = {"Authorization": f"Bearer {token}"}
        ```
    
    Client side (Legacy):
        ```python
        from core_lib.api_utils import generate_time_key
        from core_lib.config import AuthSettings
        
        settings = AuthSettings.from_env()
        auth_key = generate_time_key(settings.auth_private_key)
        
        # For SSE transport
        headers = {settings.auth_key_header_name: auth_key}
        
        # For stdio, pass as environment variable
        env = {"MCP_AUTH_KEY": auth_key}
        ```
"""

from typing import Optional, Dict, Any, Callable
import os

from .time_based_auth import verify_time_key, TimeBasedAuthError
from .auth_settings import AuthSettings


class MCPAuthError(Exception):
    """Raised when MCP authentication fails."""
    pass


# Store authenticated claims in context for tool access
_mcp_context_claims: Dict[str, Any] = {}


def get_mcp_claims() -> Optional[Dict[str, Any]]:
    """Get the current MCP authentication claims.
    
    Returns:
        Claims dict if authenticated via JWT, None otherwise
    """
    return _mcp_context_claims.get("claims")


def get_mcp_company_id() -> Optional[str]:
    """Get the company_id from current MCP authentication.
    
    Returns:
        Company ID if authenticated, None otherwise
    """
    claims = get_mcp_claims()
    if claims:
        return claims.get("company_id")
    return None


def create_auth_middleware(settings: AuthSettings):
    """Create legacy time-based authentication middleware for FastMCP v2 servers.
    
    This middleware validates time-based auth keys for incoming requests.
    Works with both SSE (HTTP headers) and stdio (environment variables).
    
    Args:
        settings: AuthSettings instance with private key
        
    Returns:
        Middleware function compatible with FastMCP v2
        
    Example:
        ```python
        from mcp import FastMCP
        
        mcp = FastMCP("server")
        settings = AuthSettings.from_env()
        
        auth_middleware = create_auth_middleware(settings)
        # Note: Actual middleware integration depends on FastMCP v2 API
        ```
    """
    async def auth_middleware(context: Dict[str, Any], next_handler: Callable):
        """Validate authentication before processing request."""
        # Skip if authentication is disabled
        if not settings.auth_enabled:
            return await next_handler(context)
        
        # Try to get auth key from different sources
        auth_key = None
        
        # 1. Check HTTP headers (for SSE transport)
        if "headers" in context:
            headers = context.get("headers", {})
            auth_key = headers.get(settings.auth_key_header_name)
        
        # 2. Check environment variable (for stdio transport)
        if not auth_key:
            auth_key = os.environ.get("MCP_AUTH_KEY")
        
        # 3. Check context metadata
        if not auth_key and "metadata" in context:
            metadata = context.get("metadata", {})
            auth_key = metadata.get("auth_key")
        
        if not auth_key:
            raise MCPAuthError(
                f"Missing authentication. Provide {settings.auth_key_header_name} "
                f"header or MCP_AUTH_KEY environment variable."
            )
        
        # Verify the time-based key
        try:
            if not verify_time_key(auth_key, settings.auth_private_key):
                raise MCPAuthError("Invalid or expired authentication key")
        except TimeBasedAuthError as e:
            raise MCPAuthError(f"Authentication error: {e}")
        
        # Authentication successful, proceed with request
        return await next_handler(context)
    
    return auth_middleware


def create_jwt_auth_middleware(jwt_settings: Any):
    """Create JWT-based authentication middleware for FastMCP v2 servers.
    
    This middleware validates JWT Bearer tokens and extracts claims
    for use by MCP tools. Supports automatic company_id injection
    for tenant isolation.
    
    Args:
        jwt_settings: JWTAuthSettings instance
        
    Returns:
        Middleware function compatible with FastMCP v2
        
    Example:
        ```python
        from mcp import FastMCP
        from core_lib.api_utils.jwt_auth import JWTAuthSettings
        
        mcp = FastMCP("server")
        jwt_settings = JWTAuthSettings.from_env()
        
        auth_middleware = create_jwt_auth_middleware(jwt_settings)
        
        @mcp.tool()
        def my_tool():
            from core_lib.api_utils.fastmcp_auth import get_mcp_company_id
            company_id = get_mcp_company_id()
            return f"Operating on company {company_id}"
        ```
    """
    from .jwt_auth import validate_jwt_token, extract_bearer_token, JWTAuthError
    from .auth_config import set_current_claims, set_current_company_id
    
    async def jwt_auth_middleware(context: Dict[str, Any], next_handler: Callable):
        """Validate JWT authentication before processing request."""
        global _mcp_context_claims
        
        # Skip if auth not required
        if not jwt_settings.require_auth:
            return await next_handler(context)
        
        # Try to get token from different sources
        token = None
        
        # 1. Check Authorization header (for SSE transport)
        if "headers" in context:
            headers = context.get("headers", {})
            authorization = headers.get("Authorization") or headers.get("authorization")
            token = extract_bearer_token(authorization)
        
        # 2. Check environment variable (for stdio transport)
        if not token:
            token = os.environ.get("MCP_JWT_TOKEN")
        
        # 3. Check context metadata
        if not token and "metadata" in context:
            metadata = context.get("metadata", {})
            token = metadata.get("token") or metadata.get("jwt_token")
        
        # 4. Check query params (passed through context)
        if not token and "query_params" in context:
            query_params = context.get("query_params", {})
            token = query_params.get("token")
        
        if not token:
            raise MCPAuthError(
                "Missing JWT authentication. Provide Authorization: Bearer <token> "
                "header or MCP_JWT_TOKEN environment variable."
            )
        
        # Validate the JWT token
        try:
            claims = validate_jwt_token(token, jwt_settings, verify_type="access")
            
            # Store claims in context for tool access
            _mcp_context_claims["claims"] = claims.to_dict()
            
            # Set context variables for tenant isolation
            set_current_claims(claims)
            if claims.company_id:
                set_current_company_id(claims.company_id)
            
            # Add claims to context for tool access
            context["jwt_claims"] = claims.to_dict()
            context["company_id"] = claims.company_id
            context["is_superuser"] = claims.is_superuser
            context["scopes"] = claims.scopes
            
        except JWTAuthError as e:
            raise MCPAuthError(f"JWT authentication failed: {e.message}")
        
        # Authentication successful, proceed with request
        return await next_handler(context)
    
    return jwt_auth_middleware


def create_unified_auth_middleware(
    jwt_settings: Optional[Any] = None,
    legacy_settings: Optional[AuthSettings] = None,
    mode: str = "both",
):
    """Create unified authentication middleware supporting both JWT and legacy auth.
    
    Args:
        jwt_settings: JWTAuthSettings for JWT authentication
        legacy_settings: AuthSettings for legacy time-based auth
        mode: "jwt", "legacy", or "both" (try JWT first, fall back to legacy)
        
    Returns:
        Middleware function compatible with FastMCP v2
    """
    from .jwt_auth import validate_jwt_token, extract_bearer_token, JWTAuthError
    from .auth_config import set_current_claims, set_current_company_id
    
    async def unified_auth_middleware(context: Dict[str, Any], next_handler: Callable):
        """Validate authentication using JWT or legacy methods."""
        global _mcp_context_claims
        
        jwt_error = None
        legacy_error = None
        
        # Try JWT authentication first (if enabled)
        if mode in ("jwt", "both") and jwt_settings:
            token = None
            
            # Get token from various sources
            if "headers" in context:
                headers = context.get("headers", {})
                authorization = headers.get("Authorization") or headers.get("authorization")
                token = extract_bearer_token(authorization)
            
            if not token:
                token = os.environ.get("MCP_JWT_TOKEN")
            
            if not token and "metadata" in context:
                token = context.get("metadata", {}).get("token")
            
            if token:
                try:
                    claims = validate_jwt_token(token, jwt_settings, verify_type="access")
                    
                    # Store claims and set context
                    _mcp_context_claims["claims"] = claims.to_dict()
                    set_current_claims(claims)
                    if claims.company_id:
                        set_current_company_id(claims.company_id)
                    
                    context["jwt_claims"] = claims.to_dict()
                    context["company_id"] = claims.company_id
                    context["is_superuser"] = claims.is_superuser
                    context["scopes"] = claims.scopes
                    context["auth_method"] = "jwt"
                    
                    return await next_handler(context)
                except JWTAuthError as e:
                    jwt_error = e.message
        
        # Try legacy authentication (if enabled and JWT failed/disabled)
        if mode in ("legacy", "both") and legacy_settings and legacy_settings.auth_enabled:
            auth_key = None
            
            if "headers" in context:
                headers = context.get("headers", {})
                auth_key = headers.get(legacy_settings.auth_key_header_name)
            
            if not auth_key:
                auth_key = os.environ.get("MCP_AUTH_KEY")
            
            if not auth_key and "metadata" in context:
                auth_key = context.get("metadata", {}).get("auth_key")
            
            if auth_key:
                try:
                    if verify_time_key(auth_key, legacy_settings.auth_private_key):
                        context["auth_method"] = "legacy"
                        return await next_handler(context)
                    else:
                        legacy_error = "Invalid or expired authentication key"
                except TimeBasedAuthError as e:
                    legacy_error = str(e)
        
        # No valid authentication found
        if mode == "jwt":
            raise MCPAuthError(jwt_error or "JWT authentication required")
        elif mode == "legacy":
            raise MCPAuthError(legacy_error or "Legacy authentication required")
        else:
            # Both mode - provide helpful error
            errors = []
            if jwt_error:
                errors.append(f"JWT: {jwt_error}")
            if legacy_error:
                errors.append(f"Legacy: {legacy_error}")
            raise MCPAuthError(
                "Authentication required. " + 
                ("; ".join(errors) if errors else "No valid credentials provided")
            )
    
    return unified_auth_middleware


def verify_mcp_auth(
    auth_key: Optional[str],
    settings: Optional[AuthSettings] = None,
    private_key: Optional[str] = None
) -> bool:
    """Verify MCP authentication key.
    
    Convenience function for manual authentication validation in MCP tools.
    
    Args:
        auth_key: The authentication key to verify
        settings: AuthSettings instance (if not provided, uses private_key)
        private_key: Private key for verification (if settings not provided)
        
    Returns:
        True if authentication is valid or disabled, False otherwise
        
    Raises:
        ValueError: If neither settings nor private_key is provided
        
    Example:
        ```python
        @mcp.tool()
        def sensitive_operation(auth_key: str = None):
            settings = AuthSettings.from_env()
            if not verify_mcp_auth(auth_key, settings=settings):
                raise ValueError("Authentication failed")
            
            return perform_operation()
        ```
    """
    # Load settings if not provided
    if settings is None and private_key is None:
        settings = AuthSettings.from_env()
    
    # Use settings if provided
    if settings is not None:
        if not settings.auth_enabled:
            return True
        private_key = settings.auth_private_key
    
    if not private_key:
        raise ValueError("Either settings or private_key must be provided")
    
    if not auth_key:
        return False
    
    try:
        return verify_time_key(auth_key, private_key)
    except TimeBasedAuthError:
        return False


def get_auth_headers(settings: Optional[AuthSettings] = None) -> Dict[str, str]:
    """Generate authentication headers for MCP client requests.
    
    Creates a dictionary with the authentication header that can be
    added to HTTP requests or MCP client configuration.
    
    Args:
        settings: AuthSettings instance (if None, loads from environment)
        
    Returns:
        Dictionary with authentication header
        
    Example:
        ```python
        from mcp import Client
        
        settings = AuthSettings.from_env()
        headers = get_auth_headers(settings)
        
        # Use with HTTP client
        async with Client(server_url, headers=headers) as client:
            result = await client.call_tool("my_tool")
        ```
    """
    if settings is None:
        settings = AuthSettings.from_env()
    
    if not settings.auth_enabled or not settings.auth_private_key:
        return {}
    
    from .time_based_auth import generate_time_key
    
    auth_key = generate_time_key(settings.auth_private_key)
    return {settings.auth_key_header_name: auth_key}


def get_auth_env_vars(settings: Optional[AuthSettings] = None) -> Dict[str, str]:
    """Generate authentication environment variables for MCP stdio transport.
    
    Creates environment variables that can be passed to MCP server processes
    using stdio transport.
    
    Args:
        settings: AuthSettings instance (if None, loads from environment)
        
    Returns:
        Dictionary with MCP_AUTH_KEY environment variable
        
    Example:
        ```python
        import subprocess
        
        settings = AuthSettings.from_env()
        env_vars = get_auth_env_vars(settings)
        
        # Merge with current environment
        env = {**os.environ, **env_vars}
        
        # Start MCP server with authentication
        subprocess.Popen(["app-server"], env=env)
        ```
    """
    if settings is None:
        settings = AuthSettings.from_env()
    
    if not settings.auth_enabled or not settings.auth_private_key:
        return {}
    
    from .time_based_auth import generate_time_key
    
    auth_key = generate_time_key(settings.auth_private_key)
    return {"MCP_AUTH_KEY": auth_key}
