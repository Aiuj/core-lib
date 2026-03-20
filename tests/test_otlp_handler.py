"""Focused tests for OTLP handler internals."""

from unittest.mock import MagicMock, patch

from core_lib.tracing.handlers.otlp_handler import OTLPHandler, _OTLPWorkerHandler


def test_otlp_handler_build_payload_includes_machine_attributes():
    """OTLP payload should include service.instance.id and host.name when set."""
    handler = OTLPHandler(
        endpoint="http://localhost:4318/v1/logs",
        service_name="test-service",
        service_version="1.2.3",
        service_instance_id="server-01",
        host_name="host-01",
        log_channel="faciliter",
    )

    payload = handler._build_payload(records=[])
    attrs = payload["resourceLogs"][0]["resource"]["attributes"]

    assert {"key": "service.name", "value": {"stringValue": "test-service"}} in attrs
    assert {"key": "service.version", "value": {"stringValue": "1.2.3"}} in attrs
    assert {"key": "service.instance.id", "value": {"stringValue": "server-01"}} in attrs
    assert {"key": "host.name", "value": {"stringValue": "host-01"}} in attrs
    assert {"key": "faciliter.log_channel", "value": {"stringValue": "faciliter"}} in attrs


def test_otlp_worker_converts_bool_extra_attrs_to_bool_value():
    """Boolean extra attributes should remain booleans in OTLP format."""
    worker = _OTLPWorkerHandler(
        endpoint="http://localhost:4318/v1/logs",
        headers={"Content-Type": "application/json"},
        timeout=10,
        insecure=False,
        service_name="test-service",
        service_version="1.0.0",
        service_instance_id="server-01",
        host_name="host-01",
        log_channel=None,
    )

    record = MagicMock(
        levelno=20,
        created=1_700_000_000.0,
        name="test.logger",
        pathname="test.py",
        lineno=42,
        funcName="test_func",
        extra_attrs={"feature.enabled": True, "attempt": 2},
    )
    record.getMessage.return_value = "hello"

    otlp = worker._convert_to_otlp(record)
    attributes = otlp["attributes"]

    assert {"key": "feature.enabled", "value": {"boolValue": True}} in attributes
    assert {"key": "attempt", "value": {"intValue": "2"}} in attributes


@patch("core_lib.tracing.handlers.otlp_handler.requests.post")
def test_otlp_worker_send_batch_emits_machine_resource_attributes(mock_post):
    """Worker send should include machine resource attributes in payload."""
    mock_post.return_value = MagicMock(status_code=200, text="ok")

    worker = _OTLPWorkerHandler(
        endpoint="http://localhost:4318/v1/logs",
        headers={"Content-Type": "application/json"},
        timeout=10,
        insecure=False,
        service_name="test-service",
        service_version="1.0.0",
        service_instance_id="server-01",
        host_name="host-01",
        log_channel="myfaq",
    )

    # Minimal converted log record
    worker._batch = [
        {
            "timeUnixNano": "1700000000000000000",
            "severityNumber": 9,
            "severityText": "INFO",
            "body": {"stringValue": "hello"},
            "attributes": [],
        }
    ]

    with worker._lock:
        worker._send_batch_locked()

    assert mock_post.called
    payload = mock_post.call_args.kwargs["json"]
    attrs = payload["resourceLogs"][0]["resource"]["attributes"]

    assert {"key": "service.instance.id", "value": {"stringValue": "server-01"}} in attrs
    assert {"key": "host.name", "value": {"stringValue": "host-01"}} in attrs
    assert {"key": "faciliter.log_channel", "value": {"stringValue": "myfaq"}} in attrs
