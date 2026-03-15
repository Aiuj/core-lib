"""Integration test for logging context with intelligence_level and process_id."""

import logging
import uuid
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from core_lib.api_utils.fastapi_middleware import inject_from_logging_context
from core_lib.tracing import (
    get_current_logging_context,
    LoggingContext,
    LoggingContextFilter,
    generate_process_id,
    install_logging_context_filter,
    set_logging_context,
    clear_logging_context,
)


@pytest.fixture
def app_with_logging():
    """Create a test FastAPI app with middleware and logging configured."""
    # Setup logging with context filter
    logger = logging.getLogger("test_app")
    logger.setLevel(logging.INFO)
    
    # Install context filter
    install_logging_context_filter(logger)
    
    app = FastAPI()
    
    @app.middleware("http")
    async def add_from_context(request: Request, call_next):
        return await inject_from_logging_context(request, call_next, tracing_client=None)
    
    @app.get("/test")
    async def test_endpoint():
        # Log something to verify the context is included
        logger.info("Test log message")
        context = get_current_logging_context()
        return {"context": context}
    
    return app, logger


def test_logging_includes_intelligence_level(app_with_logging, caplog):
    """Test that log records include intelligence_level attribute."""
    app, logger = app_with_logging
    client = TestClient(app)
    
    # Capture logs
    with caplog.at_level(logging.INFO, logger="test_app"):
        # Make a request with intelligence_level
        response = client.get("/test?intelligence_level=9")
        
        assert response.status_code == 200
        
        # Check that logs were captured
        assert len(caplog.records) > 0
        
        # Find the "Test log message" record
        log_record = None
        for record in caplog.records:
            if "Test log message" in record.message:
                log_record = record
                break
        
        assert log_record is not None, "Test log message not found in captured logs"
        
        # Verify the record has extra_attrs with intelligence_level
        assert hasattr(log_record, 'extra_attrs'), "Log record missing extra_attrs"
        assert 'intelligence.level' in log_record.extra_attrs
        assert log_record.extra_attrs['intelligence.level'] == 9


def test_logging_includes_from_fields_and_intelligence_level(app_with_logging, caplog):
    """Test that log records include both from fields and intelligence_level."""
    app, logger = app_with_logging
    client = TestClient(app)
    
    # Capture logs
    with caplog.at_level(logging.INFO, logger="test_app"):
        # Make a request with both from and intelligence_level
        from_param = '{"user_id":"user123","session_id":"sess456","company_id":"comp789"}'
        response = client.get(f'/test?from={from_param}&intelligence_level=7')
        
        assert response.status_code == 200
        
        # Find the "Test log message" record
        log_record = None
        for record in caplog.records:
            if "Test log message" in record.message:
                log_record = record
                break
        
        assert log_record is not None, "Test log message not found in captured logs"
        
        # Verify the record has all expected attributes
        assert hasattr(log_record, 'extra_attrs'), "Log record missing extra_attrs"
        assert log_record.extra_attrs['user.id'] == "user123"
        assert log_record.extra_attrs['session.id'] == "sess456"
        assert log_record.extra_attrs['organization.id'] == "comp789"
        assert log_record.extra_attrs['intelligence.level'] == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# --- process_id in log records ---

def test_logging_includes_process_id_from_middleware(app_with_logging, caplog):
    """Test that middleware auto-generated process_id appears in log record extra_attrs."""
    app, logger = app_with_logging
    client = TestClient(app)

    with caplog.at_level(logging.INFO, logger="test_app"):
        response = client.get("/test")

        assert response.status_code == 200

        log_record = next(
            (r for r in caplog.records if "Test log message" in r.message), None
        )
        assert log_record is not None

        # process.id must be present and valid UUID
        assert hasattr(log_record, "extra_attrs")
        assert "process.id" in log_record.extra_attrs
        pid = log_record.extra_attrs["process.id"]
        uuid.UUID(pid)  # validates format

        # Must match the X-Process-ID header
        assert response.headers["X-Process-ID"] == pid


def test_logging_process_id_with_from_fields(app_with_logging, caplog):
    """Test that process_id and from fields coexist in the same log record."""
    app, logger = app_with_logging
    client = TestClient(app)

    with caplog.at_level(logging.INFO, logger="test_app"):
        from_param = '{"session_id":"s1","user_id":"u1","company_id":"c1"}'
        response = client.get(f"/test?from={from_param}")

        log_record = next(
            (r for r in caplog.records if "Test log message" in r.message), None
        )
        assert log_record is not None
        attrs = log_record.extra_attrs

        # All expected attributes present
        assert "process.id" in attrs
        assert attrs["session.id"] == "s1"
        assert attrs["user.id"] == "u1"
        assert attrs["organization.id"] == "c1"


def test_manual_process_id_in_logging_context(caplog):
    """Test that manually setting process_id via LoggingContext reaches log records."""
    logger = logging.getLogger("test_manual_process_id")
    logger.setLevel(logging.DEBUG)
    install_logging_context_filter(logger)

    pid = generate_process_id()
    with caplog.at_level(logging.DEBUG, logger="test_manual_process_id"):
        with LoggingContext({"process_id": pid, "session_id": "manual-sess"}):
            logger.info("manual context log")

    log_record = next(
        (r for r in caplog.records if "manual context log" in r.message), None
    )
    assert log_record is not None
    assert log_record.extra_attrs["process.id"] == pid
    assert log_record.extra_attrs["session.id"] == "manual-sess"


def test_process_id_not_leaked_outside_context(caplog):
    """Test that process_id is cleaned up after context exits."""
    clear_logging_context()

    logger = logging.getLogger("test_no_leak")
    logger.setLevel(logging.DEBUG)
    install_logging_context_filter(logger)

    with caplog.at_level(logging.DEBUG, logger="test_no_leak"):
        with LoggingContext({"process_id": generate_process_id()}):
            logger.info("inside context")

        logger.info("outside context")

    inside = next(r for r in caplog.records if "inside context" in r.message)
    outside = next(r for r in caplog.records if "outside context" in r.message)

    assert "process.id" in inside.extra_attrs
    # Outside should have no process.id (empty context → no extra_attrs or no key)
    assert not getattr(outside, "extra_attrs", {}).get("process.id")


def test_nested_context_preserves_outer_process_id():
    """Test that nested LoggingContext inherits outer process_id."""
    clear_logging_context()
    pid = generate_process_id()

    with LoggingContext({"process_id": pid, "session_id": "outer"}):
        with LoggingContext({"company_id": "c1"}):
            ctx = get_current_logging_context()
            # Inner context should inherit process_id and session_id from outer
            assert ctx["process_id"] == pid
            assert ctx["session_id"] == "outer"
            assert ctx["company_id"] == "c1"

        # After inner exits, outer context is restored
        ctx = get_current_logging_context()
        assert ctx["process_id"] == pid
        assert "company_id" not in ctx


def test_set_logging_context_adds_process_id():
    """Test that set_logging_context can add a process_id mid-request."""
    clear_logging_context()

    with LoggingContext({"session_id": "sess-1"}):
        pid = generate_process_id()
        set_logging_context(process_id=pid)

        ctx = get_current_logging_context()
        assert ctx["process_id"] == pid
        assert ctx["session_id"] == "sess-1"
