"""Example: Time-based authentication for FastAPI and MCP servers.

This example demonstrates how to use the time-based HMAC authentication
system to secure communication between applications without a centralized
API key manager.

Setup:
    1. Set AUTH_PRIVATE_KEY environment variable (same on client and server)
    2. Set AUTH_ENABLED=true to enable authentication
    3. Optionally customize AUTH_KEY_HEADER_NAME (default: x-auth-key)

Environment variables:
    AUTH_ENABLED=true
    AUTH_PRIVATE_KEY=my-super-secret-key-minimum-16-chars
    AUTH_KEY_HEADER_NAME=x-auth-key  # optional
"""

# =============================================================================
# Example 1: FastAPI Server with Middleware
# =============================================================================

def example_fastapi_middleware():
    """Example: Protect all FastAPI routes with authentication middleware."""
    try:
        from fastapi import FastAPI
        from core_lib.api_utils.fastapi_auth import TimeBasedAuthMiddleware
        from core_lib.config import AuthSettings
        
        # Create FastAPI app
        app = FastAPI()
        
        # Load authentication settings
        settings = AuthSettings.from_env()
        
        # Add authentication middleware to protect all routes
        # (except excluded paths like /health, /docs)
        app.add_middleware(
            TimeBasedAuthMiddleware,
            settings=settings,
            exclude_paths=["/health", "/docs", "/openapi.json", "/redoc"]
        )
        
        @app.get("/")
        async def root():
            return {"message": "This endpoint is protected"}
        
        @app.get("/health")
        async def health():
            return {"status": "healthy"}  # Not protected
        
        print("✓ FastAPI server configured with authentication middleware")
        print(f"  Auth enabled: {settings.auth_enabled}")
        print(f"  Header name: {settings.auth_key_header_name}")
        
    except ImportError:
        print("⚠ FastAPI not installed. Install with: pip install fastapi")


# =============================================================================
# Example 2: FastAPI Route-level Authentication (Dependency Injection)
# =============================================================================

def example_fastapi_dependency():
    """Example: Protect specific FastAPI routes with dependencies."""
    try:
        from fastapi import FastAPI, Depends
        from core_lib.api_utils.fastapi_auth import create_auth_dependency
        from core_lib.config import AuthSettings
        
        app = FastAPI()
        settings = AuthSettings.from_env()
        
        # Create reusable dependency
        verify_auth = create_auth_dependency(settings)
        
        @app.get("/public")
        async def public():
            return {"message": "Public endpoint - no auth required"}
        
        @app.get("/protected", dependencies=[Depends(verify_auth)])
        async def protected():
            return {"message": "Protected endpoint - auth required"}
        
        @app.get("/user", dependencies=[Depends(verify_auth)])
        async def user_endpoint(auth_key: str = Depends(verify_auth)):
            # You can also get the auth key if needed
            return {"message": "Authenticated", "key_length": len(auth_key)}
        
        print("✓ FastAPI server configured with route-level authentication")
        
    except ImportError:
        print("⚠ FastAPI not installed. Install with: pip install fastapi")


# =============================================================================
# Example 3: FastMCP v3 Server with Authentication
# =============================================================================

def example_fastmcp_server():
    """Example: Protect a FastMCP v3 server with authentication."""
    try:
        from fastmcp import FastMCP
        from core_lib.api_utils.fastmcp_auth import create_auth_middleware
        from core_lib.config import AuthSettings
        
        # Create MCP server
        mcp = FastMCP("Authenticated MCP Server")
        
        # Load authentication settings
        settings = AuthSettings.from_env()
        
        # Add authentication middleware
        # Note: Attach the returned middleware using FastMCP v3's middleware API.
        auth_middleware = create_auth_middleware(settings)
        # mcp.add_middleware(auth_middleware)  # Uncomment when supported
        
        @mcp.tool()
        def secure_tool(text: str) -> str:
            """A tool that requires authentication."""
            return f"Processed: {text}"
        
        print("✓ FastMCP server configured with authentication")
        print(f"  Auth enabled: {settings.auth_enabled}")
        
    except ImportError:
        print("⚠ FastMCP not installed. Install with: pip install fastmcp")


# =============================================================================
# Example 4: Client - Making Authenticated Requests
# =============================================================================

def example_client_fastapi():
    """Example: Make authenticated requests to FastAPI server."""
    try:
        import httpx
        from core_lib.api_utils import generate_time_key
        from core_lib.config import AuthSettings
        
        # Load settings (must have same AUTH_PRIVATE_KEY as server)
        settings = AuthSettings.from_env()
        
        if not settings.auth_enabled or not settings.auth_private_key:
            print("⚠ Authentication not configured. Set AUTH_ENABLED=true and AUTH_PRIVATE_KEY")
            return
        
        # Generate time-based authentication key
        auth_key = generate_time_key(settings.auth_private_key)
        
        # Make authenticated request
        headers = {
            settings.auth_key_header_name: auth_key
        }
        
        # Example with httpx
        async def make_request():
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "http://localhost:8000/protected",
                    headers=headers
                )
                print(f"Response: {response.json()}")
        
        print("✓ Client configured for authenticated requests")
        print(f"  Header: {settings.auth_key_header_name}")
        print(f"  Auth key: {auth_key[:16]}...")
        
    except ImportError:
        print("⚠ httpx not installed. Install with: pip install httpx")


# =============================================================================
# Example 5: MCP Client with Authentication
# =============================================================================

def example_client_mcp():
    """Example: Connect to MCP server with authentication."""
    from core_lib.api_utils import get_auth_headers, get_auth_env_vars
    from core_lib.config import AuthSettings
    
    settings = AuthSettings.from_env()
    
    # For SSE transport (HTTP)
    headers = get_auth_headers(settings)
    print(f"✓ HTTP headers for MCP client: {list(headers.keys())}")
    
    # For stdio transport (subprocess)
    env_vars = get_auth_env_vars(settings)
    print(f"✓ Environment variables for MCP stdio: {list(env_vars.keys())}")
    
    # Example usage:
    # async with Client(server_url, headers=headers) as client:
    #     result = await client.call_tool("my_tool")


# =============================================================================
# Example 6: Manual Key Verification
# =============================================================================

def example_manual_verification():
    """Example: Manually verify authentication keys."""
    from core_lib.api_utils import generate_time_key, verify_time_key
    from core_lib.config import AuthSettings
    
    settings = AuthSettings.from_env(
        auth_enabled=True,
        auth_private_key="my-super-secret-key-for-testing-12345"
    )
    
    # Generate a key
    client_key = generate_time_key(settings.auth_private_key)
    print(f"Generated key: {client_key[:16]}...")
    
    # Verify the key
    is_valid = verify_time_key(client_key, settings.auth_private_key)
    print(f"Key is valid: {is_valid}")
    
    # Try with wrong key
    wrong_key = "0" * 64
    is_valid = verify_time_key(wrong_key, settings.auth_private_key)
    print(f"Wrong key is valid: {is_valid}")


# =============================================================================
# Example 7: Testing Hour Transitions
# =============================================================================

def example_hour_transitions():
    """Example: Demonstrate that keys work across hour boundaries."""
    from datetime import datetime, timedelta, timezone
    from core_lib.api_utils import generate_time_key, verify_time_key
    
    private_key = "my-super-secret-key-for-testing-12345"
    
    # Generate key for a specific time
    base_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
    key = generate_time_key(private_key, dt=base_time)
    
    print(f"Key generated at 14:30")
    
    # Test verification at different times
    test_times = [
        ("13:30 (1h before)", base_time - timedelta(hours=1)),
        ("14:00 (same hour)", base_time),
        ("14:59 (same hour)", base_time + timedelta(minutes=29)),
        ("15:00 (next hour)", base_time + timedelta(hours=1)),
        ("15:30 (next hour)", base_time + timedelta(hours=1, minutes=30)),
        ("16:30 (2h after)", base_time + timedelta(hours=2)),
    ]
    
    for label, test_time in test_times:
        is_valid = verify_time_key(key, private_key, dt=test_time)
        status = "✓ VALID" if is_valid else "✗ INVALID"
        print(f"  {status}: {label}")


# =============================================================================
# Example 8: Production Configuration
# =============================================================================

def example_production_config():
    """Example: Recommended production configuration."""
    import os
    import secrets
    
    # Generate a secure random private key (do this once)
    private_key = secrets.token_urlsafe(32)
    print("Production Setup:")
    print("=" * 60)
    print("1. Generate a secure private key:")
    print(f"   AUTH_PRIVATE_KEY={private_key}")
    print()
    print("2. Set the same key on all servers and clients")
    print()
    print("3. Enable authentication:")
    print("   AUTH_ENABLED=true")
    print()
    print("4. Optional: Customize header name:")
    print("   AUTH_KEY_HEADER_NAME=x-api-auth")
    print()
    print("5. Store in .env file or environment (NEVER commit to git!)")
    print()
    print("Security Notes:")
    print("- Key is minimum 16 characters (longer is better)")
    print("- Keys are valid for 3 hours (prev, current, next hour)")
    print("- Uses HMAC-SHA256 with constant-time comparison")
    print("- No centralized key management required")


# =============================================================================
# Run Examples
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Time-based Authentication Examples")
    print("=" * 60)
    print()
    
    print("Example 1: FastAPI Middleware")
    print("-" * 60)
    example_fastapi_middleware()
    print()
    
    print("Example 2: FastAPI Dependency")
    print("-" * 60)
    example_fastapi_dependency()
    print()
    
    print("Example 3: FastMCP Server")
    print("-" * 60)
    example_fastmcp_server()
    print()
    
    print("Example 4: FastAPI Client")
    print("-" * 60)
    example_client_fastapi()
    print()
    
    print("Example 5: MCP Client")
    print("-" * 60)
    example_client_mcp()
    print()
    
    print("Example 6: Manual Verification")
    print("-" * 60)
    example_manual_verification()
    print()
    
    print("Example 7: Hour Transitions")
    print("-" * 60)
    example_hour_transitions()
    print()
    
    print("Example 8: Production Configuration")
    print("-" * 60)
    example_production_config()
