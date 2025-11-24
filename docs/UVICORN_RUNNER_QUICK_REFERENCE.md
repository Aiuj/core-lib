# Uvicorn Runner Utility - Quick Reference

## Overview

The `uvicorn_runner` module in `core_lib.api_utils` provides standardized utilities for running ASGI applications (FastAPI, FastMCP, Starlette) with uvicorn, ensuring proper OTLP logging, graceful shutdown, and lifecycle management.

## Why Use This?

When using OTLP logging with ASGI applications, you **must** use uvicorn (or another ASGI server) to ensure batched logs are properly flushed. The `run_uvicorn_server` function handles all the boilerplate:

- ✅ Disables uvicorn's `log_config` to preserve OTLP/custom logging
- ✅ Flushes logs before server starts
- ✅ Registers atexit handler for log flushing on shutdown
- ✅ Handles reload mode correctly
- ✅ Consistent configuration across services

## Quick Start

### Basic Usage

```python
from fastapi import FastAPI
from core_lib.api_utils import run_uvicorn_server
from core_lib.config import initialize_settings

# Initialize your app
app = FastAPI()
settings = initialize_settings(setup_logging=True)

# Run with uvicorn
if __name__ == "__main__":
    run_uvicorn_server(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.app.log_level,
    )
```

### With Settings Object

```python
from core_lib.api_utils import run_uvicorn_from_settings
from core_lib.config import StandardSettings

settings = StandardSettings.from_env()

if __name__ == "__main__":
    run_uvicorn_from_settings(
        app=app,
        settings=settings.fastapi,  # Uses host, port, reload, log_level from settings
        app_module_path="my_app:app" if settings.fastapi.reload else None,
    )
```

### FastMCP Example

```python
from fastmcp import FastMCP
from core_lib.api_utils import run_uvicorn_server
from core_lib.config import initialize_settings

app = FastMCP("My MCP Server")
settings = initialize_settings(setup_logging=True)

@app.tool()
def my_tool(query: str) -> str:
    return "result"

if __name__ == "__main__":
    # Create ASGI app from FastMCP
    mcp_asgi_app = app.http_app(path="/mcp")
    
    # Run with uvicorn
    run_uvicorn_server(
        app=mcp_asgi_app,
        host="0.0.0.0",
        port=9980,
        log_level=settings.app.log_level,
    )
```

## API Reference

### `run_uvicorn_server()`

```python
def run_uvicorn_server(
    app: Any,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "info",
    reload: bool = False,
    app_module_path: Optional[str] = None,
    flush_logs_on_startup: bool = True,
    flush_logs_on_exit: bool = True,
) -> None:
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `app` | Any | Required | ASGI application instance or callable |
| `host` | str | `"0.0.0.0"` | Host to bind to |
| `port` | int | `8000` | Port to bind to |
| `log_level` | str | `"info"` | Uvicorn log level (debug, info, warning, error) |
| `reload` | bool | `False` | Enable auto-reload on code changes |
| `app_module_path` | str | `None` | Module path for reload mode (e.g., "my_app:app") |
| `flush_logs_on_startup` | bool | `True` | Flush logs before starting server |
| `flush_logs_on_exit` | bool | `True` | Register atexit handler to flush logs |

**Notes:**
- When `reload=True`, you **must** provide `app_module_path`
- Automatically sets `log_config=None` to preserve OTLP logging
- Registers atexit handler for graceful log flushing

### `run_uvicorn_from_settings()`

```python
def run_uvicorn_from_settings(
    app: Any,
    settings: Any,
    host_attr: str = "host",
    port_attr: str = "port",
    reload_attr: str = "reload",
    log_level_attr: str = "log_level",
    app_module_path: Optional[str] = None,
) -> None:
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `app` | Any | Required | ASGI application instance or callable |
| `settings` | Any | Required | Settings object with server configuration |
| `host_attr` | str | `"host"` | Attribute name for host in settings |
| `port_attr` | str | `"port"` | Attribute name for port in settings |
| `reload_attr` | str | `"reload"` | Attribute name for reload flag in settings |
| `log_level_attr` | str | `"log_level"` | Attribute name for log level in settings |
| `app_module_path` | str | `None` | Module path for reload mode |

## Complete Examples

### FastAPI Server with Health Checks

```python
from fastapi import FastAPI
from core_lib.config import initialize_settings, Settings
from core_lib.api_utils import run_uvicorn_server
from core_lib.tracing.logger import get_module_logger

# Initialize settings and logging
settings = initialize_settings(settings_class=Settings, setup_logging=True)
logger = get_module_logger()

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    logger.info("Starting FastAPI server")
    
    run_uvicorn_server(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.app.log_level,
        reload=False,
    )
```

### Unified Server (FastAPI + FastMCP)

```python
from fastapi import FastAPI
from fastmcp import FastMCP
from core_lib.config import initialize_settings, Settings
from core_lib.api_utils import run_uvicorn_server

settings = initialize_settings(settings_class=Settings, setup_logging=True)

# Create FastAPI app
fastapi_app = FastAPI()

# Create FastMCP app
mcp = FastMCP("MCP Server")
mcp_app = mcp.http_app(path="/mcp")

# Combine both
unified_app = FastAPI()
unified_app.mount("/api", fastapi_app)
unified_app.mount("", mcp_app)

if __name__ == "__main__":
    run_uvicorn_server(
        app=unified_app,
        host="0.0.0.0",
        port=8095,
        log_level=settings.app.log_level,
    )
```

### With Development Reload

```python
from fastapi import FastAPI
from core_lib.api_utils import run_uvicorn_server

app = FastAPI()

if __name__ == "__main__":
    # For development
    run_uvicorn_server(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="debug",
        reload=True,
        app_module_path="my_server:app",  # Required for reload
    )
```

## Benefits Over Direct uvicorn.run()

| Feature | `run_uvicorn_server()` | Direct `uvicorn.run()` |
|---------|----------------------|----------------------|
| OTLP logging works | ✅ Automatic | ❌ Manual config needed |
| Log flushing | ✅ Automatic | ❌ Manual setup needed |
| Reload mode | ✅ Smart app handling | ⚠️ Manual path/app handling |
| Atexit cleanup | ✅ Registered automatically | ❌ Manual registration |
| Consistent config | ✅ Across all services | ⚠️ Copy-paste errors |

## Migration Guide

### Before (Manual uvicorn)

```python
import uvicorn
from core_lib.tracing.logger import flush_logging
import atexit

atexit.register(flush_logging)

if __name__ == "__main__":
    flush_logging()
    
    if reload:
        uvicorn.run(
            "my_app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_config=None,
            log_level="info"
        )
    else:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_config=None,
            log_level="info"
        )
```

### After (Using run_uvicorn_server)

```python
from core_lib.api_utils import run_uvicorn_server

if __name__ == "__main__":
    run_uvicorn_server(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=reload,
        app_module_path="my_app:app" if reload else None,
    )
```

## See Also

- [OTLP Logging Integration](./OTLP_LOGGING_INTEGRATION.md) - OTLP logging setup and troubleshooting
- [FastAPI Quick Reference](./FASTAPI_OPENAPI_QUICK_REFERENCE.md) - FastAPI utilities
- [Centralized Logging](./centralized-logging.md) - Logging configuration guide
