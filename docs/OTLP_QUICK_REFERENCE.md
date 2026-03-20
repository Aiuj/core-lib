# OTLP Logging Quick Reference

## Enable OTLP Logging

### Quick Setup (Auto-Enable - Recommended)

**Minimal Configuration:**
```bash
# Only two settings needed - OTLP auto-enables!
export ENABLE_LOGGER=true
export OTLP_ENDPOINT=http://localhost:4318/v1/logs

# Optional: service name defaults to APP_NAME
export APP_NAME=my-service

# Optional: channel routing
# Leave unset for default -> otel-logs-*
# export OTLP_LOG_CHANNEL=myfaq
# export OTLP_LOG_CHANNEL=faciliter

# Service version auto-detected from pyproject.toml
```

**Auto-Enable Logic:**
- OTLP automatically enables when:
  1. `ENABLE_LOGGER=true` (or `LOG_FILE_ENABLED=true`) AND
  2. `OTLP_ENDPOINT` is defined AND
  3. `OTLP_ENABLED` is not explicitly `false`

### Via Code (Explicit Configuration)
```python
from core_lib.config import LoggerSettings
from core_lib.tracing.logger import setup_logging

settings = LoggerSettings(
    log_level="DEBUG",              # Console shows DEBUG+
    otlp_enabled=True,
    otlp_endpoint="http://localhost:4318/v1/logs",
    otlp_service_name="my-app",     # Optional: defaults from APP_NAME or setup_logging(app_name=...)
    otlp_log_channel="myfaq",       # Optional: default / myfaq / faciliter routing
    otlp_log_level="INFO",          # OTLP only receives INFO+ (optional)
)
logger = setup_logging(logger_settings=settings)
```

### Via Environment (Full Control)
```bash
export LOG_LEVEL=DEBUG                # Console level
export OTLP_ENABLED=true              # Explicit enable (optional with auto-enable)
export OTLP_ENDPOINT=http://localhost:4318/v1/logs
export OTLP_SERVICE_NAME=my-app       # Optional: otherwise defaults from APP_NAME
export OTLP_SERVICE_VERSION=1.0.0     # Default: from pyproject.toml
export OTLP_INSTANCE_ID=server-01     # Optional: stable machine identifier (default: hostname)
export OTLP_LOG_CHANNEL=myfaq         # Optional: leave unset, or use myfaq / faciliter
export OTLP_LOG_LEVEL=INFO            # OTLP level (optional, defaults to LOG_LEVEL)
```

## Configuration Fields

| Field | Default | Description |
|-------|---------|-------------|
| `otlp_enabled` | `False` | Enable OTLP |
| `otlp_endpoint` | `http://localhost:4318/v1/logs` | Collector URL |
| `otlp_headers` | `{}` | Auth headers |
| `otlp_timeout` | `10` | Timeout (seconds) |
| `otlp_insecure` | `False` | Skip SSL check |
| `otlp_service_name` | `APP_NAME` / `setup_logging(app_name=...)` / `core-lib` | Service name |
| `otlp_service_version` | `None` | Version tag |
| `otlp_instance_id` | Hostname | Machine/instance identifier sent as `service.instance.id` |
| `otlp_log_channel` | `None` | Adds `faciliter.log_channel` for collector routing |
| `otlp_log_level` | Inherits from `log_level` | Independent log level for OTLP handler |

## Your Collector Setup

If your collector is configured for channel-based routing, this library can target these routes:
- unset channel: `otel-logs-*`
- `myfaq`: `myfaq-logs-*`
- `faciliter`: `faciliter-logs-*`

The library does this by sending `faciliter.log_channel=<value>` as a resource attribute.

## Machine Identification (Multi-Server)

To distinguish multiple VPS instances in observability backends, core-lib adds:
- `service.instance.id` (from `OTLP_INSTANCE_ID` if set, otherwise hostname)
- `host.name` (when available)

This is applied consistently to both OTLP logs and traces.

## Channel Choices

Use exactly one of these channel configurations:

| Choice | `OTLP_LOG_CHANNEL` | Result |
|--------|--------------------|--------|
| default | unset | logs go to `otel-logs-*` |
| myfaq | `myfaq` | logs go to `myfaq-logs-*` |
| faciliter | `faciliter` | logs go to `faciliter-logs-*` |

## Docker Compose Integration

```yaml
my-app:
    environment:
        - OTLP_ENABLED=true
        - OTLP_ENDPOINT=http://otel-collector:4318/v1/logs
        - APP_NAME=my-app
        - OTLP_LOG_CHANNEL=myfaq
    depends_on:
        - otel-collector
```

## View Logs

**OpenSearch Dashboards:** http://localhost:5601
- Index pattern: `otel-logs*`
- Time field: `@timestamp`

If you use multiple channels, also create index patterns for `myfaq-logs*` and `faciliter-logs*`.

## Channel Routing

Use `otlp_log_channel` when one collector serves multiple products and routes each product into a dedicated index.

### Default

Do not set any channel value when you want the shared default route.

```python
LoggerSettings(
    otlp_enabled=True,
    otlp_endpoint="http://82.66.214.52:4318/v1/logs",
    otlp_service_name="my-app",
)
```

### Programmatic

```python
LoggerSettings(
    otlp_enabled=True,
    otlp_endpoint="http://82.66.214.52:4318/v1/logs",
    otlp_service_name="my-app",
    otlp_log_channel="faciliter",
)
```

### Environment Variables

```bash
# Default route -> otel-logs-*
export OTLP_ENABLED=true
export OTLP_ENDPOINT=http://82.66.214.52:4318/v1/logs
export OTLP_SERVICE_NAME=my-app

# Leave OTLP_LOG_CHANNEL unset for the default route

# myfaq route -> myfaq-logs-*
export OTLP_LOG_CHANNEL=myfaq

# faciliter route -> faciliter-logs-*
export OTLP_LOG_CHANNEL=faciliter
```

### Notes

- Leave `OTLP_LOG_CHANNEL` unset to use the collector default route.
- Supported channel values in the current stack are `myfaq` and `faciliter`.
- Keep `OTLP_SERVICE_NAME`; the channel supplements it and does not replace it.
- `TracingSettings` also reads `OTLP_LOG_CHANNEL`, so traces and logs can share the same resource metadata when your app initializes tracing.

**Query API:**
```bash
curl http://localhost:9200/otel-logs/_search?pretty
curl http://localhost:9200/myfaq-logs/_search?pretty
curl http://localhost:9200/faciliter-logs/_search?pretty
```

## With Authentication

```python
LoggerSettings(
    otlp_enabled=True,
    otlp_endpoint="https://collector.example.com/v1/logs",
    otlp_headers={"Authorization": "Bearer token"},
)
```

Or:
```bash
export OTLP_HEADERS='{"Authorization": "Bearer token"}'
```

## Independent Log Levels

**Use Case:** DEBUG logs on console, only INFO+ logs to OTLP (reduce costs/noise)

```python
LoggerSettings(
    log_level="DEBUG",       # Console: DEBUG, INFO, WARNING, ERROR, CRITICAL
    otlp_enabled=True,
    otlp_log_level="INFO",   # OTLP:    INFO, WARNING, ERROR, CRITICAL only
)
```

```bash
# Environment variables
export LOG_LEVEL=DEBUG
export OTLP_LOG_LEVEL=WARNING  # Only send WARNING+ to reduce OTLP ingestion
```

**Behavior:**
- `logger.debug("...")` → Console ✓, OTLP ✗
- `logger.info("...")` → Console ✓, OTLP ✓ (if OTLP_LOG_LEVEL=INFO)
- `logger.warning("...")` → Console ✓, OTLP ✓

## Contextual Logging (Request Metadata)

Add request-specific metadata (user_id, session_id, company_id) to all logs:

```python
from core_lib.tracing import LoggingContext, parse_from

@app.post("/endpoint")
async def endpoint(from_: Optional[str] = Query(None, alias="from")):
    from_dict = parse_from(from_)  # Parses JSON from 'from' parameter
    
    # All logs within this context get metadata automatically
    with LoggingContext(from_dict):
        logger.info("Processing request")
        # OTLP log includes: session.id, user.id, organization.id, etc.
```

**What gets added:**
- `session_id` → `session.id` (OpenTelemetry convention)
- `user_id` → `user.id`
- `company_id` → `organization.id`
- `user_name` → `user.name`
- `app_name` → `client.app.name`

**Filter logs in OpenSearch:**
```
user.id: "user-456"
organization.id: "comp-789"
session.id: "session-123"
```

See **[Centralized Logging Guide](centralized-logging.md#contextual-logging-request-metadata)** for detailed examples.

## Multiple Handlers

```python
LoggerSettings(
    file_logging=True,      # → File
    ovh_ldp_enabled=True,   # → OVH
    otlp_enabled=True,      # → OpenTelemetry
)
```

All work simultaneously!

## Troubleshooting

**Logs not appearing?**
1. Check collector: `docker logs otel-collector`
2. Test endpoint: `curl http://localhost:4318/v1/logs`
3. Verify index: `curl http://localhost:9200/_cat/indices`

**Connection errors?**
- Verify `OTLP_ENDPOINT` is correct
- Check Docker network connectivity
- For dev: use `otlp_insecure=True`

## Disabling OTLP During Tests

When running tests, you typically don't want logs sent to your OTLP collector. Disable OTLP in your test configuration:

**In `conftest.py` (pytest):**

```python
import os

# Disable OTLP export during tests - must be set BEFORE importing settings
os.environ["OTLP_ENABLED"] = "false"

# Then initialize settings
from core_lib.config import initialize_settings
from config.settings import Settings
initialize_settings(settings_class=Settings, force=True)
```

**Important:** Set the environment variable *before* importing/initializing settings, otherwise the OTLP handler may already be configured.

**Alternative - pytest.ini:**

```ini
[pytest]
env =
    OTLP_ENABLED=false
```

(Requires `pytest-env` plugin)

**Alternative - Environment variable:**

```bash
# Run tests with OTLP disabled
OTLP_ENABLED=false pytest tests/
```

## Full Documentation

- Architecture & setup: `docs/OTLP_LOGGING_INTEGRATION.md`
- Examples: `examples/example_otlp_logging.py`
- Implementation details: `docs/implementation/OTLP_IMPLEMENTATION_SUMMARY.md`
