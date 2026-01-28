"""Swagger UI JWT authentication integration.

This module provides utilities for integrating JWT Bearer authentication
with FastAPI's Swagger UI. It allows:

1. Swagger UI to display and accept Bearer tokens
2. Auto-population of tokens from URL parameters (?token=...)
3. Persistent token storage across page refreshes
4. Custom OAuth2 initialization for "Launch Tool" redirects

The key feature is that Swagger UI documentation remains publicly accessible,
but all "Try it out" API calls require valid authentication.

Example:
    ```python
    from fastapi import FastAPI
    from core_lib.api_utils.swagger_auth import configure_swagger_jwt_auth
    from core_lib.api_utils.jwt_auth import JWTAuthSettings
    
    app = FastAPI()
    settings = JWTAuthSettings.from_env()
    
    # Configure Swagger UI with JWT support
    configure_swagger_jwt_auth(app, settings)
    
    # Now users can:
    # 1. Visit /docs and manually enter a Bearer token
    # 2. Be redirected from Django Admin with /docs?token=eyJ... (auto-applied)
    ```
"""

from typing import Optional, List, Dict, Any

try:
    from fastapi import FastAPI
    from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
    from fastapi.openapi.utils import get_openapi
    from starlette.responses import HTMLResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = None  # type: ignore
    get_swagger_ui_html = None  # type: ignore
    get_redoc_html = None  # type: ignore
    get_openapi = None  # type: ignore
    HTMLResponse = None  # type: ignore


# Custom JavaScript for Swagger UI token auto-injection
# This script runs on Swagger UI load and:
# 1. Checks for ?token= in URL
# 2. Stores the token in localStorage
# 3. Pre-populates the auth dialog
SWAGGER_TOKEN_INIT_SCRIPT = """
(function() {
    // Extract token from URL if present
    const urlParams = new URLSearchParams(window.location.search);
    const token = urlParams.get('token');
    
    if (token) {
        // Store token in localStorage for Swagger UI
        const authKey = 'authorized';
        const authData = {
            BearerAuth: {
                name: 'BearerAuth',
                schema: {
                    type: 'http',
                    scheme: 'bearer',
                    bearerFormat: 'JWT'
                },
                value: token
            }
        };
        
        try {
            localStorage.setItem(authKey, JSON.stringify(authData));
            console.log('[Swagger Auth] Token auto-applied from URL');
            
            // Clean up URL (remove token param for security)
            urlParams.delete('token');
            const cleanUrl = window.location.pathname + 
                (urlParams.toString() ? '?' + urlParams.toString() : '');
            window.history.replaceState({}, document.title, cleanUrl);
        } catch (e) {
            console.warn('[Swagger Auth] Could not store token:', e);
        }
    }
})();
"""


def _add_bearer_security_scheme(
    openapi_schema: Dict[str, Any],
    scheme_name: str = "BearerAuth",
    description: str = "JWT Bearer token authentication",
    exclude_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Add Bearer security scheme to OpenAPI schema.
    
    Args:
        openapi_schema: OpenAPI schema dict
        scheme_name: Name of the security scheme
        description: Description shown in Swagger UI
        exclude_paths: Paths to exclude from security requirement
        
    Returns:
        Modified OpenAPI schema
    """
    if exclude_paths is None:
        exclude_paths = ["/health", "/docs", "/openapi.json", "/redoc"]
    
    # Ensure components exist
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "securitySchemes" not in openapi_schema["components"]:
        openapi_schema["components"]["securitySchemes"] = {}
    
    # Add Bearer auth scheme
    openapi_schema["components"]["securitySchemes"][scheme_name] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": description,
    }
    
    # Apply security to all paths except excluded
    for path, path_item in openapi_schema.get("paths", {}).items():
        if path not in exclude_paths:
            for method in path_item.values():
                if isinstance(method, dict):
                    if "security" not in method:
                        method["security"] = []
                    security_entry = {scheme_name: []}
                    if security_entry not in method["security"]:
                        method["security"].append(security_entry)
    
    return openapi_schema


def configure_swagger_jwt_auth(
    app: "FastAPI",
    settings: Optional[Any] = None,
    scheme_name: str = "BearerAuth",
    description: str = "JWT Bearer token from Django Admin",
    exclude_paths: Optional[List[str]] = None,
    custom_docs_path: str = "/docs",
    custom_openapi_path: str = "/openapi.json",
    custom_redoc_path: str = "/redoc",
    persist_authorization: bool = True,
    inject_token_script: bool = True,
) -> None:
    """Configure FastAPI Swagger UI with JWT Bearer authentication.
    
    This function:
    1. Adds Bearer auth security scheme to OpenAPI spec
    2. Configures Swagger UI to persist auth tokens
    3. Optionally injects script to auto-apply tokens from URL
    
    Args:
        app: FastAPI application instance
        settings: Optional JWTAuthSettings (used for exclude_paths if provided)
        scheme_name: Name of the security scheme (default: "BearerAuth")
        description: Description shown in Swagger UI
        exclude_paths: Paths to exclude from security requirement
        custom_docs_path: Path for Swagger UI (default: "/docs")
        custom_openapi_path: Path for OpenAPI JSON (default: "/openapi.json")
        custom_redoc_path: Path for ReDoc (default: "/redoc")
        persist_authorization: Store auth in localStorage (default: True)
        inject_token_script: Add script for ?token= auto-apply (default: True)
        
    Raises:
        ImportError: If FastAPI is not installed
        
    Example:
        ```python
        from fastapi import FastAPI
        from core_lib.api_utils.swagger_auth import configure_swagger_jwt_auth
        
        app = FastAPI(
            docs_url=None,  # Disable default docs
            redoc_url=None,
        )
        
        configure_swagger_jwt_auth(app)
        
        # Now /docs shows Bearer auth and auto-applies tokens from URL
        ```
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for configure_swagger_jwt_auth. "
            "Install it with: pip install fastapi"
        )
    
    # Build exclude paths
    if exclude_paths is None:
        exclude_paths = ["/health", custom_docs_path, custom_openapi_path, custom_redoc_path]
        if settings and hasattr(settings, "exclude_paths"):
            exclude_paths = list(set(exclude_paths + list(settings.exclude_paths)))
    
    # Store original openapi function
    original_openapi = app.openapi
    
    def custom_openapi() -> Dict[str, Any]:
        """Generate OpenAPI schema with Bearer auth."""
        if app.openapi_schema:
            return app.openapi_schema
        
        # Get base schema
        if callable(original_openapi) and original_openapi != custom_openapi:
            openapi_schema = original_openapi()
        else:
            openapi_schema = get_openapi(
                title=app.title,
                version=app.version,
                description=app.description,
                routes=app.routes,
            )
        
        # Add Bearer security scheme
        openapi_schema = _add_bearer_security_scheme(
            openapi_schema,
            scheme_name=scheme_name,
            description=description,
            exclude_paths=exclude_paths,
        )
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    # Replace openapi method
    app.openapi_schema = None
    app.openapi = custom_openapi
    
    # Build Swagger UI init options
    swagger_ui_params = {
        "persistAuthorization": persist_authorization,
    }
    
    # Create custom docs endpoint with token injection
    @app.get(custom_docs_path, include_in_schema=False)
    async def custom_swagger_ui_html() -> HTMLResponse:
        """Serve Swagger UI with custom initialization."""
        html_content = get_swagger_ui_html(
            openapi_url=custom_openapi_path,
            title=f"{app.title} - API Documentation",
            swagger_ui_parameters=swagger_ui_params,
        )
        
        # Inject token auto-apply script if enabled
        if inject_token_script:
            # Insert script before </body>
            script_tag = f"<script>{SWAGGER_TOKEN_INIT_SCRIPT}</script>"
            html_body = html_content.body.decode("utf-8")
            html_body = html_body.replace("</body>", f"{script_tag}</body>")
            return HTMLResponse(content=html_body)
        
        return html_content
    
    # Create custom ReDoc endpoint
    @app.get(custom_redoc_path, include_in_schema=False)
    async def custom_redoc_html() -> HTMLResponse:
        """Serve ReDoc documentation."""
        return get_redoc_html(
            openapi_url=custom_openapi_path,
            title=f"{app.title} - API Documentation",
        )


def get_swagger_launch_url(
    base_url: str,
    token: str,
    docs_path: str = "/docs",
) -> str:
    """Generate URL to launch Swagger UI with pre-applied token.
    
    Used by Django Admin's "Launch Tool" button to redirect
    admins to a service's Swagger UI with their token pre-applied.
    
    Args:
        base_url: Base URL of the service (e.g., "http://localhost:8096")
        token: JWT access token to apply
        docs_path: Path to Swagger docs (default: "/docs")
        
    Returns:
        Full URL with token parameter
        
    Example:
        ```python
        url = get_swagger_launch_url(
            "http://localhost:8096",
            "eyJhbGciOiJIUzI1NiIs..."
        )
        # Returns: "http://localhost:8096/docs?token=eyJhbGciOiJIUzI1NiIs..."
        ```
    """
    base_url = base_url.rstrip("/")
    docs_path = docs_path if docs_path.startswith("/") else f"/{docs_path}"
    return f"{base_url}{docs_path}?token={token}"


def create_swagger_oauth_config(
    authorization_url: str,
    token_url: str,
    client_id: str,
    scopes: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Create OAuth2 configuration for Swagger UI.
    
    For advanced use cases where OAuth2 flow is preferred over
    direct Bearer token entry.
    
    Args:
        authorization_url: OAuth2 authorization endpoint
        token_url: OAuth2 token endpoint
        client_id: Client ID for OAuth2 flow
        scopes: Available scopes with descriptions
        
    Returns:
        Swagger UI oauth2 configuration dict
    """
    if scopes is None:
        scopes = {
            "rfx:read": "Read RFx documents",
            "rfx:write": "Write RFx documents",
            "kb:read": "Read knowledge base",
            "kb:write": "Write to knowledge base",
        }
    
    return {
        "clientId": client_id,
        "appName": "Faciliter Platform",
        "scopeSeparator": " ",
        "scopes": " ".join(scopes.keys()),
        "usePkceWithAuthorizationCodeGrant": True,
    }
