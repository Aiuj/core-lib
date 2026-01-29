# Legacy Time-Based HMAC Authentication

> **Note**: This authentication method is deprecated. New services should use [JWT authentication](API_AUTH_QUICK_REFERENCE.md). This document is preserved for backward compatibility during migration.

## Overview

Time-based HMAC authentication uses HMAC-SHA256 with a shared private key to generate keys that are valid for a 3-hour window. This method is still supported via `AUTH_MODE=legacy` or `AUTH_MODE=both`.

## How It Works

1. **Private Key**: A secret string shared between client and server (minimum 16 characters)
2. **Time Windows**: Keys are generated based on UTC hour (e.g., "2024-01-15-14")
3. **3-Hour Validity**: Keys are accepted if valid for previous, current, or next hour
4. **HMAC Generation**: `HMAC-SHA256(private_key, time_window)` produces the public key

### Time Window Example

At 14:30 UTC on 2024-01-15:
- Keys from 13:xx are valid (previous hour)
- Keys from 14:xx are valid (current hour)
- Keys from 15:xx are valid (next hour)
- Keys from 12:xx or 16:xx are **not** valid

## Configuration

### Server (FastAPI)

```python
from fastapi import FastAPI
from core_lib.api_utils.fastapi_auth import TimeBasedAuthMiddleware
from core_lib.config import AuthSettings

app = FastAPI()
settings = AuthSettings.from_env()

app.add_middleware(
    TimeBasedAuthMiddleware,
    settings=settings,
    exclude_paths=["/health", "/docs", "/openapi.json"]
)
```

### Client

```python
from core_lib.api_utils import generate_time_key
from core_lib.config import AuthSettings

settings = AuthSettings.from_env()
auth_key = generate_time_key(settings.auth_private_key)

headers = {settings.auth_key_header_name: auth_key}
```

## Environment Variables

```bash
AUTH_ENABLED=true
AUTH_PRIVATE_KEY=your-secret-key-minimum-16-characters
AUTH_KEY_HEADER_NAME=x-auth-key  # default
```

## API Reference

### `generate_time_key(private_key, dt=None, encoding='utf-8')`

Generate a time-based authentication key.

**Parameters:**
- `private_key` (str): Secret private key (minimum 16 characters)
- `dt` (datetime, optional): Datetime to generate key for (default: current UTC time)
- `encoding` (str): String encoding (default: 'utf-8')

**Returns:** str - Hex-encoded HMAC-SHA256 key (64 characters)

### `verify_time_key(provided_key, private_key, dt=None, encoding='utf-8')`

Verify a time-based authentication key.

**Parameters:**
- `provided_key` (str): The authentication key to verify
- `private_key` (str): Secret private key (must match generation key)
- `dt` (datetime, optional): Datetime to verify against (default: current UTC time)
- `encoding` (str): String encoding (default: 'utf-8')

**Returns:** bool - True if valid, False otherwise

## Migration to JWT

To migrate from time-based auth to JWT:

1. Set `AUTH_MODE=both` to accept both methods during migration
2. Update clients to obtain JWT tokens from Django Admin
3. Update client code to use `Authorization: Bearer <token>` header
4. Once all clients are migrated, set `AUTH_MODE=jwt`

See [API_AUTH_QUICK_REFERENCE.md](API_AUTH_QUICK_REFERENCE.md) for JWT setup instructions.
