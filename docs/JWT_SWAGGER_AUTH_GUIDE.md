# JWT + Swagger Auth Guide (FastAPI)

This guide explains how to enable JWT authentication in a FastAPI project while keeping Swagger UI accessible and compatible with Bearer tokens. It is designed for projects that use the same architecture as core-lib (FastAPI app, shared auth config, optional mounted apps).

---

## 1) Dependencies

Ensure `pyjwt` is installed (core-lib already declares it):
- `pyjwt>=2.10.1`

---

## 2) Environment Variables

Set these environment variables (typically in `.env`):

- `JWT_SECRET` **(required)**
- `JWT_ALGORITHM` (default: `HS256`)
- `JWT_ACCESS_TOKEN_TTL` (default: `3600`)
- `JWT_REFRESH_TOKEN_TTL` (default: `86400`)
- `JWT_ISSUER` (default: `faciliter`)
- `JWT_REQUIRE_AUTH` (default: `true`)

Optional legacy (time-based) auth variables are defined in `core_lib.config.AuthSettings`.

---

## 3) Recommended Pattern (Unified Auth + Swagger)

Use `configure_app_auth()` to enable:
- JWT auth middleware
- Swagger UI with Bearer token support
- Optional legacy auth (time-based)

```python
from fastapi import FastAPI
from core_lib.api_utils.auth_config import configure_app_auth, AuthMode
from core_lib.api_utils.jwt_auth import JWTAuthSettings
from core_lib.config import AuthSettings

app = FastAPI(
    title="My API",
    docs_url=None,   # Disable default docs, core-lib will add custom Swagger
    redoc_url=None,
)

jwt_settings = JWTAuthSettings.from_env()
legacy_settings = AuthSettings.from_env()

configure_app_auth(
    app,
    jwt_settings=jwt_settings,
    legacy_settings=legacy_settings,
    mode=AuthMode.JWT,  # or AuthMode.BOTH
    enable_swagger_auth=True,
    exclude_paths=["/health", "/docs", "/openapi.json", "/redoc"],
)
```

**Result:**
- `/docs` stays publicly accessible
- All API calls require a valid Bearer token
- Swagger UI supports `Authorize` with JWT tokens
- Tokens can be auto-applied via `?token=...`

---

## 4) Mounted App Pattern (e.g., `/api`)

If your FastAPI app is mounted under a prefix (e.g. `/api`), ensure Swagger UI points to the correct OpenAPI URL:

```python
configure_app_auth(
    app,
    jwt_settings=jwt_settings,
    legacy_settings=legacy_settings,
    mode=AuthMode.JWT,
    enable_swagger_auth=True,
    exclude_paths=["/health", "/docs", "/openapi.json", "/redoc"],
    openapi_url="/api/openapi.json",  # IMPORTANT for mounted apps
)
```

**Why?** Swagger UI otherwise defaults to `/openapi.json` at the server root, which won’t include the mounted app’s routes.

**Unified server example:** If your REST API is mounted at `/api` (e.g., a unified FastAPI + MCP server), set `openapi_url="/api/openapi.json"` so `/api/docs` loads the correct schema.

---

## 5) Token Auto-Injection

Swagger UI supports automatic token injection from the URL:

```
https://your-service/api/docs?token=eyJhbGciOi...
```

The token is stored in localStorage and set as `BearerAuth` automatically.

---

## 6) Common Pitfalls

- **Swagger shows only root endpoints**: set `openapi_url` for mounted apps.
- **Auth blocks `/docs`**: ensure `/docs`, `/openapi.json`, and `/redoc` are in `exclude_paths`.
- **Missing company_id**: if multi-tenant isolation is enforced, provide `company_id` as a query parameter or in JWT claims.

---

## 7) Minimal Checklist

- [ ] Set `JWT_SECRET`
- [ ] Use `FastAPI(docs_url=None, redoc_url=None)`
- [ ] Call `configure_app_auth()`
- [ ] Add `exclude_paths` for docs/health
- [ ] Set `openapi_url` when mounted under a prefix

---

## Related References

- `core_lib.api_utils.auth_config.configure_app_auth`
- `core_lib.api_utils.jwt_auth.JWTAuthSettings`
- `core_lib.api_utils.swagger_auth.configure_swagger_jwt_auth`
