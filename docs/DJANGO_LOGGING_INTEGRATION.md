# Django Logging Integration Guide

Complete guide for integrating core-lib's centralized logging system with Django, including proper OTLP (OpenTelemetry) configuration and lifecycle management.

## Quick Start

### 1. Basic Django Setup

```python
# settings.py
from core_lib.config.logger_settings import LoggerSettings
from core_lib.tracing.logger import setup_logging

# CRITICAL: Disable Django's default logging configuration
# This prevents Django from configuring logging and allows core-lib to manage it
# See: https://docs.djangoproject.com/en/5.2/topics/logging/
LOGGING_CONFIG = None

# Configure logging
LOGGER_SETTINGS = LoggerSettings(
    log_level="INFO",
    file_logging=True,
    file_path="logs/django.log",
    otlp_enabled=True,
    otlp_endpoint="http://localhost:4318/v1/logs",
    otlp_service_name="my-django-app",
    otlp_instance_id="django-vps-01",  # Optional: machine ID (default: hostname)
)

# Initialize logging - this replaces Django's default LOGGING
setup_logging(
    app_name="my-django-app",
    logger_settings=LOGGER_SETTINGS,
    force=True
)
```

### 2. Production WSGI/ASGI Setup

```python
# wsgi.py (or asgi.py)
import os
import signal
import sys
from django.core.wsgi import get_wsgi_application
from core_lib.config.logger_settings import LoggerSettings
from core_lib.tracing.logger import setup_logging, flush_logging

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')

# Note: Ensure LOGGING_CONFIG = None is set in settings.py to disable Django's logging

# Setup logging BEFORE creating WSGI application
settings = LoggerSettings.from_env()
setup_logging(app_name="my-django-app", logger_settings=settings)

# Register shutdown handler to flush OTLP logs
def handle_shutdown(signum, frame):
    """Ensure OTLP logs are flushed before shutdown"""
    flush_logging()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

application = get_wsgi_application()
```

### 3. Use in Django Views/Models

```python
# views.py
from core_lib.tracing.logger import get_module_logger

logger = get_module_logger()  # Auto-namespaced to "my-django-app.myapp.views"

def my_view(request):
    logger.info("Processing request", extra={
        "user_id": request.user.id,
        "path": request.path,
    })
    
    try:
        # Your logic here
        result = process_data()
        logger.info("Request completed successfully")
        return JsonResponse({"result": result})
    except Exception as e:
        logger.error("Request failed", exc_info=True, extra={
            "error": str(e),
        })
        raise
```

## OTLP (OpenTelemetry) Integration

### Understanding OTLP Handler Lifecycle

The OTLP handler uses a **background thread** (QueueListener) to send logs asynchronously:

1. **Queue**: Main thread puts logs in queue (non-blocking)
2. **Worker Thread**: Background thread processes queue and sends to OTLP collector
3. **Batching**: Logs are sent in batches (100 records or 5 seconds)
4. **Shutdown**: Thread must be stopped and flushed to prevent log loss

### Critical: Ensuring Logs Are Sent

The OTLP handler automatically handles cleanup via:
- `atexit` registration in the worker handler
- `close()` method that flushes remaining logs
- Signal handlers for SIGTERM/SIGINT

However, Django's WSGI/ASGI servers may not wait for `atexit` handlers. **You MUST explicitly flush logs on shutdown.**

### Recommended Setup for Production

#### Option 1: Signal Handlers (WSGI/ASGI)

```python
# wsgi.py
import os
import signal
import sys
from django.core.wsgi import get_wsgi_application
from core_lib.tracing.logger import setup_logging, flush_logging, get_module_logger

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')

# Setup logging
from core_lib.config.logger_settings import LoggerSettings
settings = LoggerSettings.from_env()
setup_logging(app_name="my-django-app", logger_settings=settings)

logger = get_module_logger()
logger.info("Django application starting")

# Register shutdown handlers
def handle_shutdown(signum, frame):
    """Flush OTLP logs before server shutdown"""
    logger.info("Received shutdown signal, flushing logs")
    flush_logging()  # Ensures all batched logs are sent
    logger.info("Shutdown complete")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_shutdown)  # Gunicorn sends SIGTERM
signal.signal(signal.SIGINT, handle_shutdown)   # Ctrl+C

application = get_wsgi_application()
```

#### Option 2: Django AppConfig (Development)

```python
# myapp/apps.py
from django.apps import AppConfig
import atexit

class MyAppConfig(AppConfig):
    name = 'myapp'
    
    def ready(self):
        """Ensure OTLP handler cleanup on Django shutdown"""
        from core_lib.tracing.logger import flush_logging
        
        # Register cleanup - works for runserver
        atexit.register(flush_logging)

# myapp/__init__.py
default_app_config = 'myapp.apps.MyAppConfig'
```

#### Option 3: Management Command Wrapper

For Django management commands (migrate, etc.):

```python
# myapp/management/commands/base.py
from django.core.management.base import BaseCommand
from core_lib.tracing.logger import flush_logging, get_module_logger

class FlushLoggingCommand(BaseCommand):
    """Base command that ensures OTLP logs are flushed"""
    
    def handle(self, *args, **options):
        logger = get_module_logger()
        try:
            result = self.handle_with_logging(*args, **options)
            return result
        finally:
            logger.info("Command completed, flushing logs")
            flush_logging()
    
    def handle_with_logging(self, *args, **options):
        """Override this instead of handle()"""
        raise NotImplementedError()

# Usage in your commands
from myapp.management.commands.base import FlushLoggingCommand

class Command(FlushLoggingCommand):
    def handle_with_logging(self, *args, **options):
        logger = get_module_logger()
        logger.info("Running custom command")
        # Your logic here
```

### Environment Variables

```bash
# .env or environment
LOG_LEVEL=INFO
OTLP_ENABLED=true
OTLP_ENDPOINT=http://localhost:4318/v1/logs
OTLP_SERVICE_NAME=my-django-app
OTLP_SERVICE_VERSION=1.0.0
OTLP_INSTANCE_ID=django-vps-01
OTLP_LOG_LEVEL=INFO  # Independent level for OTLP (optional)

# Optional: File logging
LOG_FILE_ENABLED=true
LOG_FILE_PATH=logs/django.log
```

Machine identification is exported in OTLP resource attributes:
- `service.instance.id` (`OTLP_INSTANCE_ID` if set, otherwise hostname)
- `host.name` (when available)

## Request Context Integration

Add request-specific metadata to all logs within a request using `LoggingContext`:

### Middleware Approach

```python
# middleware.py
from core_lib.tracing import LoggingContext, parse_from
import uuid

class LoggingContextMiddleware:
    """Add request context to all logs"""
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Extract context from query params or headers
        from_ = request.GET.get('from') or request.headers.get('X-From')
        context = parse_from(from_) if from_ else {}
        
        # Add request metadata
        context.setdefault('session_id', str(uuid.uuid4()))
        if request.user.is_authenticated:
            context['user_id'] = str(request.user.id)
            context['user_name'] = request.user.username
        
        # Apply context to all logs in this request
        with LoggingContext(context):
            response = self.get_response(request)
        
        return response

# settings.py
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'myapp.middleware.LoggingContextMiddleware',  # Add after security
    # ... other middleware
]
```

### Manual Context in Views

```python
from core_lib.tracing import LoggingContext
from core_lib.tracing.logger import get_module_logger

logger = get_module_logger()

def my_view(request):
    context = {
        'user_id': str(request.user.id) if request.user.is_authenticated else None,
        'company_id': request.headers.get('X-Company-ID'),
        'session_id': request.session.session_key,
    }
    
    with LoggingContext(context):
        logger.info("Processing request")
        # All logs within this block include context
        result = process_data()
        logger.info("Request completed")
        
        return JsonResponse({"result": result})
```

## Gunicorn Configuration

When running Django with Gunicorn, configure proper shutdown handling:

```python
# gunicorn.conf.py
import signal
import sys

# Worker configuration
workers = 4
worker_class = 'sync'
timeout = 30
graceful_timeout = 30  # Time to wait for workers to finish

def on_starting(server):
    """Called before master process is initialized"""
    from core_lib.config.logger_settings import LoggerSettings
    from core_lib.tracing.logger import setup_logging
    
    settings = LoggerSettings.from_env()
    setup_logging(app_name="my-django-app", logger_settings=settings)

def worker_exit(server, worker):
    """Called when worker exits - flush OTLP logs"""
    from core_lib.tracing.logger import flush_logging, get_module_logger
    
    logger = get_module_logger()
    logger.info(f"Worker {worker.pid} shutting down, flushing logs")
    flush_logging()

def on_exit(server):
    """Called when master exits"""
    from core_lib.tracing.logger import flush_logging, get_module_logger
    
    logger = get_module_logger()
    logger.info("Master process shutting down, flushing logs")
    flush_logging()
```

Run Gunicorn:
```bash
gunicorn myproject.wsgi:application -c gunicorn.conf.py
```

## Celery Integration

For Celery tasks, ensure OTLP logs are flushed after task completion:

```python
# celery.py
from celery import Celery
from celery.signals import worker_process_init, worker_shutdown
from core_lib.config.logger_settings import LoggerSettings
from core_lib.tracing.logger import setup_logging, flush_logging, get_module_logger

app = Celery('myproject')
app.config_from_object('django.conf:settings', namespace='CELERY')

logger = get_module_logger()

@worker_process_init.connect
def init_worker(**kwargs):
    """Setup logging in each worker process"""
    settings = LoggerSettings.from_env()
    setup_logging(app_name="my-django-celery", logger_settings=settings)
    logger.info("Celery worker started")

@worker_shutdown.connect
def shutdown_worker(**kwargs):
    """Flush logs when worker shuts down"""
    logger.info("Celery worker shutting down, flushing logs")
    flush_logging()

# Task example
@app.task
def process_data(data_id):
    from core_lib.tracing import LoggingContext
    
    with LoggingContext({"task_id": str(process_data.request.id), "data_id": data_id}):
        logger.info(f"Processing data {data_id}")
        # Task logic
        logger.info("Task completed")
```

## Testing

### Unit Tests

```python
# tests.py
from django.test import TestCase
from core_lib.tracing.logger import setup_logging, get_module_logger, flush_logging
import logging

class LoggingTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Setup logging for tests
        setup_logging(app_name="test-app", level="DEBUG", force=True)
    
    def setUp(self):
        self.logger = get_module_logger()
    
    def tearDown(self):
        # Flush logs after each test
        flush_logging()
    
    def test_logging(self):
        self.logger.info("Test log message")
        # Assertions...
```

### Integration Tests with OTLP

```python
import pytest
from core_lib.config.logger_settings import LoggerSettings
from core_lib.tracing.logger import setup_logging, flush_logging, get_module_logger

@pytest.fixture(scope="session")
def configure_logging():
    """Setup logging once for test session"""
    settings = LoggerSettings(
        log_level="DEBUG",
        otlp_enabled=True,
        otlp_endpoint="http://localhost:4318/v1/logs",
        otlp_service_name="test-service",
        otlp_instance_id="test-node-01",
    )
    setup_logging(app_name="test-app", logger_settings=settings, force=True)
    yield
    # Flush logs at end of test session
    flush_logging()

def test_with_logging(configure_logging):
    logger = get_module_logger()
    logger.info("Test log")
    # Your test logic
```

## Troubleshooting

### Logs Not Appearing or Django's Default Logging Active

**Most Common Issue**: Forgot to set `LOGGING_CONFIG = None` in settings.py

**Symptoms**:
- Logs appear in Django's default format instead of core-lib format
- OTLP logs not being sent
- File logging not working as configured

**Solution**:
```python
# settings.py - ADD THIS AT THE TOP
LOGGING_CONFIG = None  # Disable Django's logging configuration
```

**Reference**: [Django Logging Documentation](https://docs.djangoproject.com/en/5.2/topics/logging/)

Without `LOGGING_CONFIG = None`, Django will run its default `logging.config.dictConfig()` which overwrites your core-lib configuration.

### Logs Not Sent to OTLP Collector

1. **Check OTLP endpoint is accessible**
   ```bash
   curl http://localhost:4318/v1/logs
   ```

2. **Check stderr for OTLP errors**
   ```bash
   python manage.py runserver 2>&1 | grep OTLP
   ```

3. **Verify OTLP is enabled**
   ```python
   from core_lib.tracing.logger import get_last_logging_config
   print(get_last_logging_config())
   ```

4. **Manually flush logs**
   ```python
   from core_lib.tracing.logger import flush_logging
   flush_logging()  # Force immediate send
   ```

### Logs Lost on Shutdown

- **Cause**: Server kills worker before logs are flushed
- **Solution**: Add signal handlers (see Production Setup above)
- **Verify**: Check `flush_logging()` is called in shutdown handlers

### Duplicate Logs

- **Cause**: Multiple `setup_logging()` calls OR Django's logging still active
- **Solution**: 
  1. Set `LOGGING_CONFIG = None` in settings.py
  2. Call `setup_logging()` ONCE in wsgi.py or settings.py
- **Fix**: Use `force=True` to reconfigure: `setup_logging(force=True)`

### UTF-8 Encoding Issues

All handlers support UTF-8. If you see encoding errors:

```python
# Explicitly set UTF-8 encoding in settings.py
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
```

### Performance Impact

The OTLP handler is designed for minimal performance impact:
- **Non-blocking**: Uses queue + background thread
- **Batching**: Sends 100 records at once or every 5 seconds
- **Fast serialization**: JSON encoding is efficient
- **Timeout protection**: 10-second request timeout

If you see performance issues:
1. Increase batch size (modify `_batch_size` in handler)
2. Increase batch timeout (modify `_batch_timeout`)
3. Disable OTLP in development: `OTLP_ENABLED=false`

## Best Practices

### ✅ DO

- **Set `LOGGING_CONFIG = None` in settings.py** (REQUIRED - see [Django docs](https://docs.djangoproject.com/en/5.2/topics/logging/))
- **Setup logging once** in wsgi.py/asgi.py (production) or settings.py
- **Use `get_module_logger()`** in all modules (not `logging.getLogger(__name__)`)
- **Add signal handlers** to flush logs on SIGTERM/SIGINT
- **Use LoggingContext** for request metadata (user_id, company_id, etc.)
- **Flush logs** in Celery tasks and management commands
- **Log at appropriate levels** (DEBUG for dev, INFO for prod)

### ❌ DON'T

- **Don't forget `LOGGING_CONFIG = None`** (most common integration issue)
- **Don't call `setup_logging()` multiple times** (only once at startup)
- **Don't use print()** - use logger instead
- **Don't log sensitive data** (passwords, API keys, PII)
- **Don't rely on atexit alone** in production (use signal handlers)
- **Don't ignore stderr** - check for OTLP errors

## Complete Example: Production Django App

```python
# settings.py
from core_lib.config.logger_settings import LoggerSettings
from core_lib.tracing.logger import setup_logging

# CRITICAL: Disable Django's logging (REQUIRED)
LOGGING_CONFIG = None

# Configure and initialize logging
settings = LoggerSettings.from_env()
setup_logging(app_name="my-django-app", logger_settings=settings)

# wsgi.py
import os
import signal
import sys
from django.core.wsgi import get_wsgi_application
from core_lib.tracing.logger import flush_logging, get_module_logger

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')

logger = get_module_logger()
logger.info("Django application starting")

# Shutdown handlers
def handle_shutdown(signum, frame):
    logger.info("Received shutdown signal", extra={"signal": signum})
    flush_logging()
    logger.info("Shutdown complete")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

application = get_wsgi_application()

# views.py
from core_lib.tracing import LoggingContext
from core_lib.tracing.logger import get_module_logger

logger = get_module_logger()

def api_view(request):
    context = {
        'user_id': str(request.user.id) if request.user.is_authenticated else None,
        'session_id': request.session.session_key,
        'request_id': request.META.get('HTTP_X_REQUEST_ID'),
    }
    
    with LoggingContext(context):
        logger.info("API request received", extra={
            "method": request.method,
            "path": request.path,
        })
        
        try:
            result = process_request(request)
            logger.info("API request completed successfully")
            return JsonResponse({"result": result})
        except Exception as e:
            logger.error("API request failed", exc_info=True, extra={
                "error": str(e),
            })
            return JsonResponse({"error": str(e)}, status=500)

# .env
LOG_LEVEL=INFO
OTLP_ENABLED=true
OTLP_ENDPOINT=http://localhost:4318/v1/logs
OTLP_SERVICE_NAME=my-django-app
OTLP_SERVICE_VERSION=1.0.0
OTLP_INSTANCE_ID=django-vps-01
```

## Documentation

- **Centralized Logging Guide**: `docs/centralized-logging.md`
- **OTLP Quick Reference**: `docs/OTLP_QUICK_REFERENCE.md`
- **Environment Variables**: `docs/ENV_VARIABLES.md`
- **Examples**: `examples/example_otlp_logging.py`
