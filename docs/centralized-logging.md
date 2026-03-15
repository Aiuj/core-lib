# Centralized Logging Guide

Simple guide for using core-lib's logging across any application (FastAPI, MCP servers, CLI tools, etc.).

## Quick Start

### 1. Basic Setup (Any Application)

```python
from core_lib.tracing.logger import setup_logging, get_module_logger

# At application startup (main.py, app.py, etc.)
setup_logging(app_name="my-app", level="INFO")

# In any module
logger = get_module_logger()
logger.info("Application started")
logger.debug("Debug information")
logger.error("Error occurred")
```

### 2. With Environment Variables

```bash
# .env file
LOG_LEVEL=DEBUG
OTLP_ENABLED=true
OTLP_ENDPOINT=http://localhost:4318/v1/logs
```

```python
from core_lib.config.logger_settings import LoggerSettings
from core_lib.tracing.logger import setup_logging

# Automatically reads environment variables
settings = LoggerSettings.from_env()
setup_logging(app_name="my-app", logger_settings=settings)
```

### 3. With Settings Class

```python
from core_lib.config.logger_settings import LoggerSettings
from core_lib.tracing.logger import setup_logging

settings = LoggerSettings(
    log_level="INFO",
    file_logging=True,
    file_path="logs/app.log",
    otlp_enabled=True,
    otlp_endpoint="http://localhost:4318/v1/logs",
)

setup_logging(app_name="my-app", logger_settings=settings)
```

## Application Examples

### FastAPI Application

```python
# main.py
from fastapi import FastAPI
from core_lib.config.logger_settings import LoggerSettings
from core_lib.tracing.logger import setup_logging, get_module_logger

# Setup logging at startup
settings = LoggerSettings.from_env()
setup_logging(app_name="my-api", logger_settings=settings)

app = FastAPI()
logger = get_module_logger()

@app.on_event("startup")
async def startup_event():
    logger.info("API server starting up")

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    logger.debug(f"Fetching item {item_id}")
    return {"item_id": item_id}
```

### MCP Server

```python
# server.py
from mcp.server import Server
from core_lib.tracing.logger import setup_logging, get_module_logger

# Setup logging
setup_logging(app_name="app-server", level="INFO")
logger = get_module_logger()

server = Server("my-app-server")

@server.list_tools()
async def list_tools():
    logger.info("Listing available tools")
    return [...]
```

### CLI Application

```python
# cli.py
import argparse
from core_lib.tracing.logger import setup_logging, get_module_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    # Setup with dynamic level
    level = "DEBUG" if args.debug else "INFO"
    setup_logging(app_name="my-cli", level=level)
    
    logger = get_module_logger()
    logger.info("CLI started")
    
if __name__ == "__main__":
    main()
```

### Web Application (Flask/Django)

```python
# app.py (Flask) or wsgi.py (Django)
from core_lib.tracing.logger import setup_logging, get_module_logger

# Setup once at application initialization
setup_logging(app_name="my-web-app", level="INFO")
logger = get_module_logger()

# Flask example
from flask import Flask
app = Flask(__name__)

@app.route("/")
def index():
    logger.info("Index page accessed")
    return "Hello World"
```

**Django Users**: See the comprehensive [Django Logging Integration Guide](DJANGO_LOGGING_INTEGRATION.md) for proper setup with WSGI/ASGI servers, signal handlers, and OTLP lifecycle management.

## Available Logging Handlers

### 1. Console (Always Enabled)
Logs to stdout with colored output.

### 2. File Logging (Optional)
```python
LoggerSettings(
    file_logging=True,
    file_path="logs/app.log",
    file_max_bytes=1_048_576,  # 1MB rotation
    file_backup_count=3,        # Keep 3 backups
)
```

### 3. OTLP (OpenTelemetry) (Optional)
```python
LoggerSettings(
    otlp_enabled=True,
    otlp_endpoint="http://localhost:4318/v1/logs",
    otlp_log_level="INFO",  # Independent level for OTLP
    otlp_service_name="my-app",
)
```

### 4. OVH Logs Data Platform (Optional)
```python
LoggerSettings(
    ovh_ldp_enabled=True,
    ovh_ldp_token="your-token",
    ovh_ldp_endpoint="gra1.logs.ovh.com",
)
```

## Environment Variables

### Basic Logging
```bash
LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE_ENABLED=true       # Enable file logging
LOG_FILE_PATH=logs/app.log  # Log file path
```

### OTLP (OpenTelemetry)
```bash
OTLP_ENABLED=true
OTLP_ENDPOINT=http://localhost:4318/v1/logs
OTLP_LOG_LEVEL=INFO         # Independent OTLP level (optional)
OTLP_SERVICE_NAME=my-app
```

### OVH LDP
```bash
OVH_LDP_ENABLED=true
OVH_LDP_TOKEN=your-token
OVH_LDP_ENDPOINT=gra1.logs.ovh.com
```

**See:** `docs/ENV_VARIABLES.md` for complete list

## Configuration Precedence

Log level is determined by (highest priority first):
1. Explicit `level` argument to `setup_logging()`
2. `logger_settings.log_level` if provided
3. `app_settings.log_level` if provided
4. `LOG_LEVEL` environment variable
5. Default: `INFO`

## Contextual Logging (Request Metadata)

For web applications and APIs, you often want to include request-specific metadata (user ID, session ID, company ID) in all logs. The `LoggingContext` manager automatically injects this metadata into every log record within a request scope.

### Why Contextual Logging?

- **Correlate logs** across distributed systems using session_id
- **Filter logs** by user, company, or session in observability dashboards
- **Reconstruct flows** - see complete request journey across services
- **Debug issues** specific to a user or company

### Basic Usage with parse_from

The `parse_from()` function extracts metadata from the standard `from` query parameter (used across MCP servers and APIs). Use the `FROM_FIELD_DESCRIPTION` constant for consistent parameter documentation:

```python
from fastapi import FastAPI, Query, Depends
from typing import Optional
from core_lib.tracing.logger import setup_logging, get_module_logger
from core_lib.tracing import LoggingContext, FROM_FIELD_DESCRIPTION, parse_from
from core_lib.api_utils import require_api_key

app = FastAPI()
setup_logging(app_name="my-api")
logger = get_module_logger()

@app.post("/v1/answer/question", dependencies=[Depends(require_api_key)])
async def answer_question(
    question: str,
    company_id: str = Query(..., description="Tenant company UUID"),
    from_: Optional[str] = Query(None, alias="from", description=FROM_FIELD_DESCRIPTION)
):
    # Parse the 'from' parameter into a dict
    from_dict = parse_from(from_)
    
    # Use LoggingContext to inject metadata into all logs within this request
    with LoggingContext(from_dict):
        logger.info(f"Processing question for company={company_id}")
        # ALL logs within this block automatically include from_dict metadata
        
        result = await process_question(question, company_id)
        logger.info("Question processed successfully")
        
        return {"answer": result}

async def process_question(question: str, company_id: str):
    # This logger also inherits the context automatically!
    logger.info(f"Looking up documents for question: {question}")
    # Log will include session.id, user.id, organization.id, etc.
    return "Answer..."
```

**Note:** The `FROM_FIELD_DESCRIPTION` constant provides standardized documentation for API endpoints. See `docs/FROM_FIELD_DESCRIPTION.md` for details.

### What parse_from() Extracts

The `from` parameter is a JSON string with standard fields:

```json
{
  "session_id": "session-abc-123",
  "user_id": "user-456",
  "user_name": "john.doe@example.com",
  "company_id": "company-789",
  "company_name": "Acme Corp",
  "app_name": "mobile-app",
  "app_version": "2.1.0",
  "model_name": "gpt-4"
}
```

> **Note:** `process_id` is **not** part of the `from` parameter. It is automatically generated server-side by the middleware for each request and injected into the logging context. You do not need to supply it.
```

Example API call:
```bash
curl -X POST "https://api.example.com/v1/answer/question?company_id=comp-789&from=%7B%22session_id%22%3A%22session-123%22%2C%22user_id%22%3A%22user-456%22%7D" \
  -H "X-API-Key: your-key" \
  -d '{"question": "What is the policy?"}'
```

### Context Fields Mapped to OpenTelemetry Standards

`LoggingContext` automatically maps fields to OpenTelemetry semantic conventions:

| from field | OTel attribute | Description |
|------------|----------------|-------------|
| `process_id` | `process.id` | Server-generated unique ID per request (auto-injected by middleware) |
| `session_id` | `session.id` | Client-provided session identifier (groups multiple requests) |
| `user_id` | `user.id` | User UUID/ID |
| `user_name` | `user.name` | User email/username |
| `company_id` | `organization.id` | Company/tenant UUID |
| `company_name` | `organization.name` | Company/tenant name |
| `app_name` | `client.app.name` | Client application name |
| `app_version` | `client.app.version` | Client app version |
| `model_name` | `gen_ai.request.model` | AI model identifier |

### Integration with Tracing

Combine with tracing metadata for complete observability:

```python
from core_lib.tracing import setup_tracing, add_trace_metadata
from core_lib.tracing.logger import setup_logging, get_module_logger
from core_lib.tracing import LoggingContext, parse_from

# Setup both tracing and logging
setup_tracing(service_name="my-api")
setup_logging(app_name="my-api")
logger = get_module_logger()

@app.post("/endpoint")
async def endpoint(from_: Optional[str] = Query(None, alias="from")):
    from_dict = parse_from(from_)
    
    # Add to BOTH logging and tracing
    with LoggingContext(from_dict):
        add_trace_metadata(metadata=from_dict)
        
        logger.info("Processing request")
        # Log includes context AND is linked to trace
```

### Nested Contexts (Advanced)

Contexts can be nested - inner contexts extend outer contexts:

```python
from core_lib.tracing import LoggingContext

# Outer context: set session/user once for entire request
with LoggingContext({"session_id": "session-123", "user_id": "user-456"}):
    logger.info("Request started")  # Has session_id, user_id
    
    # Inner context: add company_id for specific operation
    with LoggingContext({"company_id": "comp-789"}):
        logger.info("Processing company data")  # Has ALL: session_id, user_id, company_id
    
    logger.info("Request completed")  # Back to session_id, user_id only
```

### Manual Context Updates

Update context mid-request without creating new context:

```python
from core_lib.tracing import set_logging_context, get_current_logging_context

with LoggingContext({"session_id": "session-123"}):
    logger.info("Started")  # session_id only
    
    # Add more metadata later
    set_logging_context(user_id="user-456", company_id="comp-789")
    logger.info("Updated")  # Now has session_id, user_id, company_id
    
    # Check current context
    current = get_current_logging_context()
    print(current)  # {'session_id': 'session-123', 'user_id': 'user-456', 'company_id': 'comp-789'}
```

### Filtering Logs in Observability Dashboards

With contextual metadata, you can filter logs in Grafana/Datadog/New Relic:

```
# Find all logs for a single API call (process_id is auto-generated per request)
process.id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

# Find all logs for a session (groups multiple API calls)
session.id = "session-abc-123"

# Find all logs for a specific user
user.id = "user-456"

# Find all logs for a company
organization.id = "comp-789"

# Find logs from specific client app version
client.app.name = "mobile-app" AND client.app.version = "2.1.0"
```

> **Tip:** Use `process.id` to isolate logs for a single API call. Use `session.id` to see all API calls in a session. The `process_id` is also returned in the `X-Process-ID` response header for client-side correlation.

### FastAPI Middleware (Recommended)

Use the built-in middleware which automatically generates a `process_id` for every request:

```python
from fastapi import FastAPI
from core_lib.api_utils.fastapi_middleware import FromContextMiddleware

app = FastAPI()

# This automatically:
# 1. Parses the 'from' query parameter (session_id, user_id, etc.)
# 2. Generates a unique process_id per request
# 3. Injects all fields into LoggingContext for every log record
# 4. Returns X-Process-ID in the response header
app.add_middleware(FromContextMiddleware)
```

Or using the functional alternative:

```python
from fastapi import FastAPI, Request
from core_lib.api_utils.fastapi_middleware import inject_from_logging_context

app = FastAPI()

@app.middleware("http")
async def add_from_context(request: Request, call_next):
    return await inject_from_logging_context(request, call_next)

@app.get("/users")
async def get_users():
    logger.info("Fetching users")  # Automatically has context from middleware
    return {"users": [...]}
```

### MCP Server / Background Worker Example

For non-HTTP contexts (MCP tools, background workers), generate `process_id` manually:

```python
from mcp.server import Server
from core_lib.tracing import LoggingContext, parse_from, generate_process_id

server = Server("my-app-server")

@server.call_tool()
async def call_tool(name: str, arguments: dict, from_: str | None = None):
    from_dict = parse_from(from_)
    from_dict["process_id"] = generate_process_id()  # Unique ID for this tool invocation
    
    with LoggingContext(from_dict):
        logger.info(f"Tool called: {name}")
        # Process tool...
        logger.info("Tool completed")
```

### Testing Contextual Logging

```python
from core_lib.tracing import LoggingContext, get_current_logging_context, clear_logging_context

def test_logging_context():
    # Clear any existing context
    clear_logging_context()
    
    # Set context
    with LoggingContext({"user_id": "test-user", "session_id": "test-session"}):
        context = get_current_logging_context()
        assert context["user_id"] == "test-user"
        assert context["session_id"] == "test-session"
        
        # Log something
        logger.info("Test log")  # Will include user_id and session_id
    
    # Context cleared after exiting
    context = get_current_logging_context()
    assert context == {}
```

## Best Practices

### ✅ DO

**Setup once at application startup:**
```python
# main.py or app.py
from core_lib.tracing.logger import setup_logging
setup_logging(app_name="my-app")
```

**Use get_module_logger() in modules:**
```python
# services/user_service.py
from core_lib.tracing.logger import get_module_logger
logger = get_module_logger()  # Auto-namespaced to "my-app.services.user_service"

def create_user(name: str):
    logger.info(f"Creating user: {name}")
```

**Important:** Always use `get_module_logger()` instead of `logging.getLogger(__name__)` to ensure proper handler inheritance (OTLP, file, etc.) and automatic namespacing under the app logger.

**Use appropriate log levels:**
- `DEBUG`: Detailed diagnostic information (SQL queries, request payloads)
- `INFO`: General application flow (user logged in, API called)
- `WARNING`: Unexpected but recoverable (rate limit approached, retry attempted)
- `ERROR`: Operation failed (API error, database timeout)
- `CRITICAL`: System-level failure (can't connect to database)

**Include context in logs:**
```python
logger.info("User created", extra={"user_id": 123, "email": "user@example.com"})
```

### ❌ DON'T

**Don't call setup_logging() in library modules:**
```python
# BAD - Don't do this in a library module
def my_function():
    setup_logging()  # ❌ Only call once at app startup
```

**Don't log sensitive data:**
```python
logger.debug(f"Password: {password}")  # ❌ Never log passwords/tokens
logger.info(f"API Key: {api_key}")     # ❌ Never log secrets
```

**Don't use print() statements:**
```python
print("Debug info")  # ❌ Use logger.debug() instead
logger.debug("Debug info")  # ✅ Correct
```

## Multiple Handlers Example

Send DEBUG to console, INFO to file, WARNING to OTLP:

```python
LoggerSettings(
    log_level="DEBUG",           # Console shows DEBUG+
    file_logging=True,
    file_path="logs/app.log",    # File gets DEBUG+ (inherits root level)
    otlp_enabled=True,
    otlp_log_level="WARNING",    # OTLP only gets WARNING+
)
```

**Result:**
- `logger.debug("...")` → Console ✓, File ✓, OTLP ✗
- `logger.info("...")` → Console ✓, File ✓, OTLP ✗
- `logger.warning("...")` → Console ✓, File ✓, OTLP ✓

## Logger Hierarchy

The library uses a 3-level logger hierarchy for proper propagation:

1. **Root Logger** (`""`) - Holds all handlers (Console, File, OTLP, etc.)
2. **App Logger** (`"my-app"`) - Your application namespace
3. **Module Loggers** (`"my-app.services.user_service"`) - Individual modules

This ensures logs from any module propagate to all configured handlers automatically.

## Reconfiguration

To change log level at runtime (e.g., enable debug mode):

```python
from core_lib.tracing.logger import setup_logging

# Initial setup
setup_logging(app_name="my-app", level="INFO")

# Later, enable debug mode
setup_logging(level="DEBUG", force=True)  # force=True reconfigures
```

## Troubleshooting

### Logs not appearing?

1. **Check log level**: Ensure your log level allows the message
   ```python
   import logging
   logger.setLevel(logging.DEBUG)  # Or use setup_logging(level="DEBUG")
   ```

2. **Check handler levels**: OTLP might have independent level
   ```bash
   export OTLP_LOG_LEVEL=DEBUG  # If OTLP isn't receiving logs
   ```

3. **Verify setup was called**: Must call `setup_logging()` at startup

### Duplicate log messages?

- Only call `setup_logging()` once at application startup
- Use `force=True` if reconfiguring: `setup_logging(level="DEBUG", force=True)`

### OTLP logs not sent?

1. Check OTLP endpoint is accessible
2. Check stderr for errors: `python app.py 2>&1 | grep OTLP`
3. Verify OTLP_ENABLED=true
4. Logs are batched - wait 5 seconds or send 100+ logs

### International characters (UTF-8)?

All logging handlers support UTF-8 encoding:
- **Console**: Automatically configured on Windows (Python 3.7+)
- **File**: Explicitly uses `encoding="utf-8"`
- **OTLP**: Uses JSON (UTF-8 by default in Python 3)
- **GELF**: Explicitly encodes as UTF-8

```python
logger.info("ログメッセージ: 正常に処理されました")  # Japanese
logger.info("Mensagem de log: processado com sucesso")  # Portuguese
logger.info("日志消息：处理成功")  # Chinese
```

## Documentation

- **Quick Reference**: `docs/OTLP_QUICK_REFERENCE.md`
- **Environment Variables**: `docs/ENV_VARIABLES.md`
- **Examples**: `examples/example_otlp_logging.py`
- **API Docs**: See docstrings in `core_lib/tracing/logger.py`
