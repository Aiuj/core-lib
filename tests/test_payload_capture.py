"""Tests for core_lib.tracing.payload_capture."""

import gzip
import json
import unittest
from unittest.mock import MagicMock, patch

from botocore.exceptions import BotoCoreError, ClientError

from core_lib.config.payload_capture_settings import PayloadCaptureSettings
from core_lib.tracing.payload_capture import (
    _reset_warned_missing_buckets,
    capture_llm_payload,
)


class TestPayloadCapture(unittest.TestCase):
    def setUp(self):
        _reset_warned_missing_buckets()

    def tearDown(self):
        _reset_warned_missing_buckets()

    def test_capture_disabled_noop(self):
        settings = PayloadCaptureSettings(enabled=False, s3_bucket="my-bucket")
        with patch("boto3.client") as mock_boto:
            capture_llm_payload(
                call_id="call-1",
                provider="openai",
                model="gpt-4o",
                messages=[{"role": "user", "content": "hello"}],
                response_text="world",
                settings=settings,
            )
            mock_boto.assert_not_called()

    def test_capture_no_bucket_noop(self):
        settings = PayloadCaptureSettings(enabled=True, s3_bucket="")
        with patch("boto3.client") as mock_boto:
            capture_llm_payload(
                call_id="call-1",
                provider="openai",
                model="gpt-4o",
                messages=[{"role": "user", "content": "hello"}],
                response_text="world",
                settings=settings,
            )
            mock_boto.assert_not_called()

    def test_capture_success(self):
        settings = PayloadCaptureSettings(enabled=True, s3_bucket="test-bucket")
        mock_s3 = MagicMock()
        with patch("boto3.client", return_value=mock_s3):
            capture_llm_payload(
                call_id="call-123",
                provider="openai",
                model="gpt-4o",
                messages=[{"role": "user", "content": "hello"}],
                response_text="hi there",
                metadata={"user_id": "u1"},
                settings=settings,
            )

            mock_s3.put_object.assert_called_once()
            call_kwargs = mock_s3.put_object.call_args.kwargs
            self.assertEqual(call_kwargs["Bucket"], "test-bucket")
            self.assertTrue(call_kwargs["Key"].endswith("/call-123.json.gz"))
            self.assertEqual(call_kwargs["ContentType"], "application/json")
            self.assertEqual(call_kwargs["ContentEncoding"], "gzip")

            decompressed = json.loads(gzip.decompress(call_kwargs["Body"]).decode("utf-8"))
            self.assertEqual(decompressed["call_id"], "call-123")
            self.assertEqual(decompressed["provider"], "openai")
            self.assertEqual(decompressed["model"], "gpt-4o")
            self.assertEqual(decompressed["response"], "hi there")

    def test_no_such_bucket_handled_gracefully_and_suppresses_repeat_warnings(self):
        settings = PayloadCaptureSettings(enabled=True, s3_bucket="nonexistent-bucket")
        mock_s3 = MagicMock()
        error_response = {
            "Error": {
                "Code": "NoSuchBucket",
                "Message": "The specified bucket does not exist.",
            }
        }
        mock_s3.put_object.side_effect = ClientError(error_response, "PutObject")

        with patch("boto3.client", return_value=mock_s3), \
             patch("core_lib.tracing.payload_capture.logger") as mock_logger:

            # First call: should log a clean warning without exc_info
            capture_llm_payload(
                call_id="call-1",
                provider="openai",
                model="gpt-4o",
                settings=settings,
            )

            self.assertEqual(mock_logger.warning.call_count, 1)
            warning_call = mock_logger.warning.call_args
            rendered = warning_call[0][0] % warning_call[0][1:]
            self.assertIn("S3 bucket 'nonexistent-bucket' does not exist", rendered)
            self.assertNotIn("exc_info", warning_call.kwargs)

            # Second call: warning should be suppressed, logged to debug instead
            capture_llm_payload(
                call_id="call-2",
                provider="openai",
                model="gpt-4o",
                settings=settings,
            )

            self.assertEqual(mock_logger.warning.call_count, 1)
            mock_logger.debug.assert_called()

    def test_other_client_error_handled_gracefully(self):
        settings = PayloadCaptureSettings(enabled=True, s3_bucket="restricted-bucket")
        mock_s3 = MagicMock()
        error_response = {
            "Error": {
                "Code": "AccessDenied",
                "Message": "Access Denied",
            }
        }
        mock_s3.put_object.side_effect = ClientError(error_response, "PutObject")

        with patch("boto3.client", return_value=mock_s3), \
             patch("core_lib.tracing.payload_capture.logger") as mock_logger:

            capture_llm_payload(
                call_id="call-1",
                provider="openai",
                model="gpt-4o",
                settings=settings,
            )

            self.assertEqual(mock_logger.warning.call_count, 1)
            warning_call = mock_logger.warning.call_args
            rendered = warning_call[0][0] % warning_call[0][1:]
            self.assertIn("AccessDenied", rendered)
            self.assertNotIn("exc_info", warning_call.kwargs)

    def test_botocore_error_handled_gracefully(self):
        settings = PayloadCaptureSettings(enabled=True, s3_bucket="my-bucket")
        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = BotoCoreError()

        with patch("boto3.client", return_value=mock_s3), \
             patch("core_lib.tracing.payload_capture.logger") as mock_logger:

            capture_llm_payload(
                call_id="call-1",
                provider="openai",
                model="gpt-4o",
                settings=settings,
            )

            self.assertEqual(mock_logger.warning.call_count, 1)
            warning_call = mock_logger.warning.call_args
            self.assertNotIn("exc_info", warning_call.kwargs)

    def test_unexpected_exception_handled_gracefully(self):
        settings = PayloadCaptureSettings(enabled=True, s3_bucket="my-bucket")
        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = RuntimeError("Disk or memory issue")

        with patch("boto3.client", return_value=mock_s3), \
             patch("core_lib.tracing.payload_capture.logger") as mock_logger:

            capture_llm_payload(
                call_id="call-1",
                provider="openai",
                model="gpt-4o",
                settings=settings,
            )

            self.assertEqual(mock_logger.warning.call_count, 1)
            warning_call = mock_logger.warning.call_args
            self.assertNotIn("exc_info", warning_call.kwargs)
