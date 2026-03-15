# FROM_FIELD_DESCRIPTION Constant

## Overview

The `FROM_FIELD_DESCRIPTION` constant provides standardized documentation for the `from` parameter used across FastAPI applications for observability and tracing.

## Location

```python
from core_lib import FROM_FIELD_DESCRIPTION
# or
from core_lib.tracing import FROM_FIELD_DESCRIPTION
```

## Purpose

This constant ensures consistent documentation and usage patterns for the `from` parameter across all applications that use core-lib for tracing and observability.

## Usage in FastAPI

```python
from fastapi import FastAPI, Query
from core_lib import FROM_FIELD_DESCRIPTION, INTELLIGENCE_LEVEL_DESCRIPTION, setup_tracing
from core_lib.api_utils.fastapi_middleware import inject_from_logging_context

app = FastAPI()
tracing_client = setup_tracing(name="my-app")

# Add middleware to automatically extract from and intelligence_level
@app.middleware("http")
async def add_from_context(request, call_next):
    return await inject_from_logging_context(request, call_next, tracing_client)

@app.post("/v1/some-endpoint")
async def some_endpoint(
    from_: Optional[str] = Query(None, alias="from", description=FROM_FIELD_DESCRIPTION),
    intelligence_level: int = Query(5, ge=0, le=10, description=INTELLIGENCE_LEVEL_DESCRIPTION)
):
    # Middleware automatically adds both from and intelligence_level to logging context
    # No manual metadata addition needed
    
    # Your endpoint logic here
    pass
```

**Note**: When using the `inject_from_logging_context` middleware, both the `from` parameter and `intelligence_level` query parameter are automatically extracted and added to the logging context for all log records within the request scope.

## Expected Format

The `from` parameter can be either:
- A JSON string
- A dictionary/object

### Structure

```json
{
    "session_id": "12",
    "app_name": "AI Platform",
    "app_version": "0.1.0",
    "model_name": "gemini-2.5-flash",
    "user_name": "Sarah Mitchell",
    "user_id": "10000000-0000-0000-0000-000000000101",
    "company_name": "TechVision Solutions",
    "company_id": "10000000-0000-0000-0000-000000000001",
    "intelligence_level": 7
}
```

### Fields

All fields are optional:

- **session_id**: Client-provided identifier grouping multiple API calls in a session/conversation
- **process_id**: Server-generated unique ID for a single request (auto-injected by middleware — do not supply manually)
- **app_name**: Name of the calling application
- **app_version**: Version of the calling application
- **model_name**: AI model being used (e.g., "gemini-2.5-flash", "gpt-4")
- **user_name**: Human-readable name of the user making the request
- **user_id**: Unique identifier for the user (UUID format recommended)
- **company_name**: Human-readable name of the organization
- **company_id**: Unique identifier for the organization (UUID format)
- **intelligence_level**: Integer 0-10 indicating the intelligence/quality level for LLM responses (also accepted as a separate query parameter)

> **Note:** When using `FromContextMiddleware` or `inject_from_logging_context`, a unique `process_id` is automatically generated for each request and returned in the `X-Process-ID` response header. This enables correlating all logs for a single API call.

## Benefits

1. **Consistency**: All applications using core-lib have the same documentation
2. **Centralized**: Single source of truth for the parameter description
3. **Versioned**: Changes are tracked in the library version
4. **Reusable**: Import once and use across all endpoints

## Migration from Local Definitions

If you have local definitions of `FROM_FIELD_DESCRIPTION`, replace:

```python
# Before
from api_utils.models import FROM_FIELD_DESCRIPTION

# After
from core_lib import FROM_FIELD_DESCRIPTION
```

## Related

- See `core_lib.tracing.TracingManager` for tracing client usage
- See `core_lib.tracing.setup_tracing()` for initialization
