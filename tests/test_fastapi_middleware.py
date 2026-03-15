"""Tests for FastAPI middleware utilities."""

import uuid
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from core_lib.api_utils.fastapi_middleware import inject_from_logging_context, FromContextMiddleware
from core_lib.tracing import get_current_logging_context, generate_process_id


@pytest.fixture
def app():
    """Create a test FastAPI app with middleware."""
    app = FastAPI()
    
    @app.middleware("http")
    async def add_from_context(request: Request, call_next):
        return await inject_from_logging_context(request, call_next, tracing_client=None)
    
    @app.get("/test")
    async def test_endpoint():
        # Get the current logging context to verify middleware worked
        context = get_current_logging_context()
        return {"context": context}
    
    return app


def test_middleware_extracts_intelligence_level(app):
    """Test that middleware extracts intelligence_level from query params."""
    client = TestClient(app)
    
    # Make a request with intelligence_level
    response = client.get("/test?intelligence_level=7")
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify intelligence_level was added to context
    assert "context" in data
    assert "intelligence_level" in data["context"]
    assert data["context"]["intelligence_level"] == 7


def test_middleware_extracts_from_and_intelligence_level(app):
    """Test that middleware extracts both from and intelligence_level."""
    client = TestClient(app)
    
    # Make a request with both from and intelligence_level
    from_param = '{"user_id":"user123","session_id":"sess456"}'
    response = client.get(f'/test?from={from_param}&intelligence_level=8')
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify both from fields and intelligence_level were added to context
    assert "context" in data
    assert data["context"]["user_id"] == "user123"
    assert data["context"]["session_id"] == "sess456"
    assert data["context"]["intelligence_level"] == 8


def test_middleware_validates_intelligence_level_range(app):
    """Test that middleware validates intelligence_level is in range 0-10."""
    client = TestClient(app)
    
    # Test valid values
    response = client.get("/test?intelligence_level=0")
    assert response.json()["context"]["intelligence_level"] == 0
    
    response = client.get("/test?intelligence_level=10")
    assert response.json()["context"]["intelligence_level"] == 10
    
    # Test invalid values (should be ignored)
    response = client.get("/test?intelligence_level=-1")
    assert "intelligence_level" not in response.json()["context"]
    
    response = client.get("/test?intelligence_level=11")
    assert "intelligence_level" not in response.json()["context"]
    
    response = client.get("/test?intelligence_level=invalid")
    assert "intelligence_level" not in response.json()["context"]


def test_middleware_without_intelligence_level(app):
    """Test that middleware works when intelligence_level is not provided."""
    client = TestClient(app)
    
    response = client.get("/test")
    
    assert response.status_code == 200
    data = response.json()
    
    # Context should be empty or not contain intelligence_level
    assert "intelligence_level" not in data["context"]


# --- process_id tests ---

def test_generate_process_id_returns_valid_uuid():
    """Test that generate_process_id returns a valid UUID4 string."""
    pid = generate_process_id()
    # Should be a valid UUID
    parsed = uuid.UUID(pid)
    assert str(parsed) == pid
    assert parsed.version == 4


def test_generate_process_id_is_unique():
    """Test that each call produces a different ID."""
    ids = {generate_process_id() for _ in range(100)}
    assert len(ids) == 100


def test_middleware_injects_process_id(app):
    """Test that middleware auto-generates a process_id in context."""
    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200
    data = response.json()

    # process_id should be present in logging context
    assert "process_id" in data["context"]
    # Should be a valid UUID
    uuid.UUID(data["context"]["process_id"])


def test_middleware_returns_process_id_header(app):
    """Test that the response contains an X-Process-ID header."""
    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200
    assert "X-Process-ID" in response.headers
    # Header value should match the context value
    data = response.json()
    assert response.headers["X-Process-ID"] == data["context"]["process_id"]


def test_middleware_process_id_unique_per_request(app):
    """Test that each request gets a different process_id."""
    client = TestClient(app)
    ids = set()
    for _ in range(10):
        response = client.get("/test")
        ids.add(response.headers["X-Process-ID"])
    assert len(ids) == 10


def test_class_middleware_injects_process_id():
    """Test FromContextMiddleware class also generates process_id."""
    app = FastAPI()
    app.add_middleware(FromContextMiddleware, tracing_client=None)

    @app.get("/test")
    async def test_endpoint():
        context = get_current_logging_context()
        return {"context": context}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200
    assert "X-Process-ID" in response.headers
    data = response.json()
    assert "process_id" in data["context"]
    uuid.UUID(data["context"]["process_id"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
