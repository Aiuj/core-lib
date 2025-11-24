# Health Check Utility - Quick Reference

## Overview

The `HealthChecker` class in `core_lib.utils` provides a flexible framework for checking the health and availability of various services (databases, search engines, LLMs, caches, etc.) before application startup or for monitoring endpoints.

## Why Use This?

- ✅ **Consistent health checks** across all services
- ✅ **Flexible check types** - boolean, dict, callable, lazy initialization
- ✅ **Automatic logging** of check results
- ✅ **Fail-fast support** to stop at first failure
- ✅ **Summary reports** for monitoring dashboards
- ✅ **Easy integration** with FastAPI `/health` endpoints

## Quick Start

### Basic Usage

```python
from core_lib import HealthChecker, get_module_logger

logger = get_module_logger()
checker = HealthChecker(logger=logger)

# Add a simple boolean check
def check_database():
    # Return True if healthy, False if not
    return db.ping()

checker.add_simple_check(
    "database",
    check_database,
    healthy_message="Database is responding",
    unhealthy_message="Database is down"
)

# Run all checks
results = checker.run_all_checks()

if checker.is_healthy(results):
    print("All services are healthy!")
else:
    print("Some services are unhealthy")
    summary = checker.get_summary(results)
    print(summary)
```

### With Service Objects

```python
from core_lib import HealthChecker

checker = HealthChecker(logger=logger)

# Check a service that has a health_check() method
checker.add_callable_check("search_engine", search_service)

# Check a service with custom method name
checker.add_callable_check("cache", cache_service, method_name="ping")

# Run checks
results = checker.run_all_checks()
```

### Lazy Initialization

```python
from core_lib import HealthChecker, create_lazy_check

checker = HealthChecker(logger=logger)

# Service might not be initialized yet
def get_llm_client():
    # Factory function that creates/returns service
    return LLMClient.from_env()

# Create a lazy check that initializes on first use
checker.add_check(
    "llm_client",
    create_lazy_check("llm_client", get_llm_client)
)

# Run checks (services initialized during check)
results = checker.run_all_checks()
```

## API Reference

### `HealthChecker`

```python
class HealthChecker:
    def __init__(self, logger: Optional[Any] = None)
```

**Methods:**

#### `add_check(name, check_func)`

Add a health check that returns a `HealthCheckResult`.

```python
def my_check() -> HealthCheckResult:
    return HealthCheckResult(
        name="my_service",
        status=HealthStatus.HEALTHY,
        message="Service is running"
    )

checker.add_check("my_service", my_check)
```

#### `add_simple_check(name, check_func, healthy_message, unhealthy_message)`

Add a health check that returns a boolean.

```python
checker.add_simple_check(
    "redis",
    lambda: redis_client.ping(),
    healthy_message="Redis is connected",
    unhealthy_message="Redis is down"
)
```

#### `add_dict_check(name, check_func, healthy_key)`

Add a health check that returns a dictionary.

```python
def check_db():
    return {"healthy": True, "latency_ms": 15}

checker.add_dict_check("database", check_db, healthy_key="healthy")
```

#### `add_callable_check(name, service, method_name)`

Add a health check using a method on a service object.

```python
# Service has a health_check() method
checker.add_callable_check("search", search_service)

# Service has a custom method
checker.add_callable_check("cache", cache_service, method_name="ping")
```

#### `run_all_checks(fail_fast=False)`

Run all registered health checks.

```python
# Run all checks
results = checker.run_all_checks()

# Stop at first failure
results = checker.run_all_checks(fail_fast=True)
```

#### `is_healthy(results=None)`

Check if all health checks passed.

```python
if checker.is_healthy():
    print("All healthy!")
```

#### `get_summary(results)`

Get a summary of health check results.

```python
summary = checker.get_summary(results)
# {
#     "overall_healthy": True,
#     "total_checks": 5,
#     "healthy": 5,
#     "unhealthy": 0,
#     "degraded": 0,
#     "unknown": 0,
#     "checks": [...]
# }
```

### `HealthCheckResult`

```python
@dataclass
class HealthCheckResult:
    name: str
    status: HealthStatus
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
```

### `HealthStatus`

```python
class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
```

### `create_lazy_check(name, factory, method_name)`

Create a health check that lazily initializes the service.

```python
check = create_lazy_check(
    "llm_client",
    get_llm_client,  # Factory function
    method_name="health_check"
)

checker.add_check("llm_client", check)
```

## Complete Examples

### Server Startup Health Checks

```python
from core_lib import HealthChecker, create_lazy_check, get_module_logger
from my_app import get_database, get_search, get_llm_client

logger = get_module_logger()
checker = HealthChecker(logger=logger)

# Add lazy checks for services that may not be initialized
checker.add_check(
    "database",
    create_lazy_check("database", get_database)
)

checker.add_check(
    "search_engine",
    create_lazy_check("search_engine", get_search)
)

checker.add_check(
    "llm_client",
    create_lazy_check("llm_client", get_llm_client)
)

# Run all health checks before starting server
logger.info("Running health checks before startup")
results = checker.run_all_checks()
summary = checker.get_summary(results)

if summary["overall_healthy"]:
    logger.info(f"All {summary['total_checks']} health checks passed")
    # Start server
else:
    logger.error(f"Health checks failed: {summary['unhealthy']}/{summary['total_checks']}")
    # Exit or start in degraded mode
```

### FastAPI Health Endpoint

```python
from fastapi import FastAPI
from core_lib import HealthChecker, create_lazy_check

app = FastAPI()

# Create global health checker
health_checker = HealthChecker()

# Register checks
health_checker.add_check("database", create_lazy_check("database", get_database))
health_checker.add_check("cache", create_lazy_check("cache", get_cache))

@app.get("/health")
async def health():
    results = health_checker.run_all_checks()
    summary = health_checker.get_summary(results)
    
    status_code = 200 if summary["overall_healthy"] else 503
    
    return JSONResponse(
        content=summary,
        status_code=status_code
    )

@app.get("/health/live")
async def liveness():
    # Simple liveness probe
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness():
    # Check if ready to serve traffic
    results = health_checker.run_all_checks(fail_fast=True)
    
    if checker.is_healthy(results):
        return {"status": "ready"}
    else:
        return JSONResponse(
            content={"status": "not_ready"},
            status_code=503
        )
```

### MCP Server Example

```python
from fastmcp import FastMCP
from core_lib import HealthChecker, create_lazy_check, get_module_logger
from my_services import get_database, get_search, get_llm_client

logger = get_module_logger()
app = FastMCP("My MCP Server")

# Create health checker
health_checker = HealthChecker(logger=logger)

# Register health checks
health_checker.add_check("database", create_lazy_check("database", get_database))
health_checker.add_check("search", create_lazy_check("search", get_search))
health_checker.add_check("llm", create_lazy_check("llm", get_llm_client))

if __name__ == "__main__":
    logger.info("Running health checks before starting server")
    
    results = health_checker.run_all_checks()
    summary = health_checker.get_summary(results)
    
    if summary["overall_healthy"]:
        logger.info(f"All {summary['total_checks']} health checks passed")
    else:
        logger.warning(
            f"Health checks: {summary['healthy']}/{summary['total_checks']} passed"
        )
    
    # Start server
    mcp_app = app.http_app(path="/mcp")
    run_uvicorn_server(mcp_app, host="0.0.0.0", port=9980)
```

### Custom Health Check

```python
from core_lib import HealthCheckResult, HealthStatus

def check_api_keys():
    """Check if all required API keys are configured."""
    missing_keys = []
    
    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    if not os.getenv("DATABASE_URL"):
        missing_keys.append("DATABASE_URL")
    
    if missing_keys:
        return HealthCheckResult(
            name="api_keys",
            status=HealthStatus.UNHEALTHY,
            message=f"Missing API keys: {', '.join(missing_keys)}",
            details={"missing_keys": missing_keys}
        )
    else:
        return HealthCheckResult(
            name="api_keys",
            status=HealthStatus.HEALTHY,
            message="All required API keys are configured"
        )

checker.add_check("api_keys", check_api_keys)
```

## Integration with Kubernetes

### Liveness and Readiness Probes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: app
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Benefits

| Feature | Manual Checks | HealthChecker |
|---------|--------------|---------------|
| Consistent format | ❌ Custom per service | ✅ Standardized |
| Logging | ❌ Manual per check | ✅ Automatic |
| Summary reports | ❌ Manual aggregation | ✅ Built-in |
| Fail-fast mode | ❌ Manual logic | ✅ Built-in |
| Lazy initialization | ❌ Manual handling | ✅ `create_lazy_check()` |
| Service objects | ❌ Wrapper needed | ✅ `add_callable_check()` |

## See Also

- [Uvicorn Runner](./UVICORN_RUNNER_QUICK_REFERENCE.md) - Server startup utilities
- [OTLP Logging](./OTLP_LOGGING_INTEGRATION.md) - Logging and observability
- [Centralized Logging](./centralized-logging.md) - Logging configuration
