# Platform Authentication Architecture

This document describes the authentication architecture for the Faciliter platform, where Django Admin (saas-admin) acts as the central identity provider for all services.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        DJANGO ADMIN (Identity Provider)                         │
│                                                                                 │
│   ┌─────────────────┐      ┌───────────────────┐      ┌──────────────────┐     │
│   │ Service Accounts│      │ Restricted API    │      │ Token Endpoints  │     │
│   │ (Credentials)   │      │ Keys (External)   │      │ /api/auth/token/ │     │
│   └────────┬────────┘      └───────────────────┘      └────────┬─────────┘     │
│            │                                                    │               │
│            └────────────────────┬───────────────────────────────┘               │
│                                 ▼                                               │
│                    ┌────────────────────────┐                                   │
│                    │   JWT Token Generation │                                   │
│                    │   (Access + Refresh)   │                                   │
│                    └────────────┬───────────┘                                   │
└─────────────────────────────────┼───────────────────────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │   mcp-doc-qa    │  │   agent-rfx     │  │   agent-crawler │
    │   FastAPI       │  │   FastAPI       │  │   FastAPI       │
    │                 │  │                 │  │                 │
    │ JWT Validation  │  │ JWT Validation  │  │ JWT Validation  │
    │ (shared secret) │  │ (shared secret) │  │ (shared secret) │
    └─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Key Concepts

### Service Accounts

Service Accounts represent backend services that authenticate with the platform. Each account has:

- **Service ID**: Unique identifier (e.g., `mcp-doc-qa`, `agent-rfx`)
- **Secret**: Hashed credential for token exchange
- **Company**: Optional tenant binding (None for global access)
- **Scopes**: List of permitted operations

### JWT Tokens

| Token | Lifetime | Purpose |
|-------|----------|---------|
| **Access Token** | 1 hour | API authentication, included in `Authorization: Bearer <token>` |
| **Refresh Token** | 24 hours | Obtain new access tokens without re-authenticating |

### Authentication Modes

Services configure `AUTH_MODE` to control accepted authentication methods:

| Mode | Behavior |
|------|----------|
| `jwt` | Accept only JWT tokens (recommended for production) |
| `legacy` | Accept only time-based HMAC keys (backward compatibility) |
| `both` | Try JWT first, fall back to HMAC (migration period) |
| `none` | Disable authentication (development only) |

---

## Quick Setup

### 1. Generate Shared Secret

All services must share the same JWT secret:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. Configure Django Admin (saas-admin)

Add to `.env`:

```bash
JWT_SECRET=<your-generated-secret>
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_TTL=3600    # 1 hour
JWT_REFRESH_TOKEN_TTL=86400  # 24 hours
```

### 3. Configure FastAPI Services

Each service (mcp-doc-qa, agent-rfx, etc.) needs in `.env`:

```bash
JWT_SECRET=<same-secret-as-django>
AUTH_MODE=jwt
# or AUTH_MODE=both for backward compatibility with legacy auth
```

### 4. Create Service Account (Django Admin)

1. Navigate to **Admin → Core → Service Accounts**
2. Create account with Service ID matching your service name
3. Assign appropriate scopes
4. Save and copy the one-time displayed secret

---

## Integration Patterns

### Pattern A: Unified Auth Middleware (Recommended)

Use `configure_app_auth()` to set up authentication with Swagger UI support:

```python
from fastapi import FastAPI
from core_lib.api_utils.auth_config import configure_app_auth, AuthMode
from core_lib.api_utils.jwt_auth import JWTAuthSettings

app = FastAPI(
    title="My Service",
    docs_url=None,  # Disable default, core-lib adds custom
    redoc_url=None,
)

configure_app_auth(
    app,
    jwt_settings=JWTAuthSettings.from_env(),
    mode=AuthMode.JWT,
    enable_swagger_auth=True,
    exclude_paths=["/health", "/docs", "/openapi.json"],
)
```

### Pattern B: Manual JWT Middleware

For more control, add the middleware directly:

```python
from fastapi import FastAPI
from core_lib.api_utils.jwt_auth import JWTAuthMiddleware, JWTAuthSettings

app = FastAPI()
settings = JWTAuthSettings.from_env()

app.add_middleware(
    JWTAuthMiddleware,
    settings=settings,
    exclude_paths=["/health", "/docs", "/openapi.json"]
)
```

### Pattern C: Route-Level Dependencies

Protect specific routes while leaving others public:

```python
from fastapi import FastAPI, Depends
from core_lib.api_utils.jwt_auth import require_jwt_auth, require_scope

app = FastAPI()

@app.get("/public")
async def public_endpoint():
    return {"message": "No auth required"}

@app.get("/protected", dependencies=[Depends(require_jwt_auth)])
async def protected_endpoint():
    return {"message": "JWT required"}

@app.post("/documents", dependencies=[Depends(require_scope("documents:write"))])
async def create_document():
    return {"message": "Requires documents:write scope"}
```

---

## Token Exchange API

Services obtain tokens by calling Django's token endpoints:

### Exchange Credentials for Tokens

```bash
POST /api/auth/token/exchange/
Content-Type: application/json

{
  "service_id": "mcp-doc-qa",
  "secret": "your-service-secret"
}
```

Response:
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

### Refresh Access Token

```bash
POST /api/auth/token/refresh/
Content-Type: application/json

{
  "refresh_token": "eyJ..."
}
```

### Introspect Token

```bash
POST /api/auth/token/introspect/
Content-Type: application/json

{
  "token": "eyJ..."
}
```

---

## JWT Claims Reference

Access tokens contain these claims:

| Claim | Description |
|-------|-------------|
| `sub` | Service ID (subject) |
| `company_id` | Tenant identifier (null for global access) |
| `is_superuser` | Whether issuing user was superuser |
| `scopes` | List of granted permissions |
| `token_type` | "access" or "refresh" |
| `jti` | Unique token identifier |
| `exp` | Expiration timestamp |
| `iat` | Issued at timestamp |
| `iss` | Issuer ("faciliter") |

---

## Available Scopes

| Scope | Description |
|-------|-------------|
| `kb:read` | Read knowledge base content |
| `kb:write` | Write to knowledge base |
| `documents:read` | Read documents |
| `documents:write` | Upload/modify documents |
| `rfx:process` | Process RFx questionnaires |
| `chat:access` | Access chat features |
| `newsletter:send` | Send newsletters |
| `admin:full` | Full admin access |

---

## Accessing Tenant Context

After authentication, access the company_id from JWT claims:

```python
from core_lib.api_utils.auth_config import get_current_company_id, get_current_claims

@app.get("/data")
async def get_data():
    company_id = get_current_company_id()  # From JWT claims
    claims = get_current_claims()          # Full claims object
    
    # Query with tenant isolation
    results = await db.query(
        "SELECT * FROM data WHERE company_id = $1",
        company_id
    )
    return results
```

---

## Swagger UI Integration

When using `configure_app_auth(enable_swagger_auth=True)`:

1. **Swagger UI** loads without authentication (docs are public)
2. **API calls** require valid Bearer token
3. **Token injection**: Visit `/docs?token=<jwt>` to auto-populate auth

The Django Admin "Launch Tool" action uses this pattern to provide seamless access.

---

## Legacy HMAC Authentication (Deprecated)

For backward compatibility, services can still use time-based HMAC authentication.
Set `AUTH_MODE=both` to accept either JWT or HMAC.

> **Migration Note**: New services should use JWT only (`AUTH_MODE=jwt`).
> The HMAC method is deprecated and will be removed in a future version.

See [LEGACY_TIME_BASED_AUTH.md](LEGACY_TIME_BASED_AUTH.md) for time-based HMAC documentation.

---

## Environment Variables Reference

### Django Admin (Token Issuer)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `JWT_SECRET` | Yes | — | Shared secret for signing tokens |
| `JWT_ALGORITHM` | No | `HS256` | Signing algorithm |
| `JWT_ACCESS_TOKEN_TTL` | No | `3600` | Access token lifetime (seconds) |
| `JWT_REFRESH_TOKEN_TTL` | No | `86400` | Refresh token lifetime (seconds) |

### FastAPI Services (Token Validators)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `JWT_SECRET` | Yes | — | Must match Django's secret |
| `AUTH_MODE` | No | `both` | Authentication mode |
| `AUTH_ENABLED` | No | `true` | Enable/disable auth |

---

## Related Documentation

- [JWT_SWAGGER_AUTH_GUIDE.md](JWT_SWAGGER_AUTH_GUIDE.md) — Implementation details for Swagger integration
- [API_CLIENT_BASE_CLASS.md](API_CLIENT_BASE_CLASS.md) — Building authenticated API clients
- [saas-admin README](../../saas-admin/README.md) — Creating Service Accounts and API Keys
