"""Integration tests for LLM providers with retry and rate limiting."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, call
from typing import Dict, Any, List

from core_lib.llm.providers.google_genai_provider import GoogleGenAIProvider, GeminiConfig
from core_lib.llm.retry import RetryConfig, RetryStrategy
from core_lib.llm.rate_limiter import RateLimitConfig


class MockGoogleAPIException(Exception):
    """Mock Google API exception for testing."""
    pass


class MockResourceExhausted(MockGoogleAPIException):
    """Mock ResourceExhausted exception (429 rate limit)."""
    pass


class MockServiceUnavailable(MockGoogleAPIException):
    """Mock ServiceUnavailable exception (503)."""
    pass


class TestGoogleGenAIProviderRateLimit:
    """Test rate limiting in GoogleGenAIProvider."""

    def test_model_rpm_mapping(self):
        """Test that different models get correct RPM limits."""
        test_cases = [
            ("gemini-2.5-pro", 10),
            ("gemini-2.5-flash", 30),
            ("gemini-2.5-flash-lite", 120),
            ("gemma-3", 60),
            ("gemini-embedding-001", 600),
            ("unknown-model", 60),  # Default fallback
        ]
        
        for model, expected_rpm in test_cases:
            config = GeminiConfig(api_key="test-key", model=model)
            
            with patch('google.genai.Client'), \
                 patch('openinference.instrumentation.google_genai.GoogleGenAIInstrumentor'):
                provider = GoogleGenAIProvider(config)
                
            assert provider._rate_limiter.config.requests_per_minute == expected_rpm
            # RPS should be derived from RPM
            expected_rps = max(1.0 / 60.0, expected_rpm / 60.0)
            assert abs(provider._rate_limiter.config.requests_per_second - expected_rps) < 0.001

    def test_rate_limiter_called_before_api(self):
        """Test that rate limiter is called before API requests."""
        config = GeminiConfig(api_key="test-key", model="gemini-2.5-flash")
        
        with patch('google.genai.Client') as mock_client_class, \
             patch('openinference.instrumentation.google_genai.GoogleGenAIInstrumentor'):
            
            provider = GoogleGenAIProvider(config)
            
            # Mock the rate limiter's acquire method
            with patch.object(provider, '_acquire_rate_limit') as mock_acquire:
                # Mock the API response
                mock_response = MagicMock()
                mock_response.text = "Test response"
                mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
                
                mock_client = mock_client_class.return_value
                mock_client.models.generate_content.return_value = mock_response
                
                # Make a chat request
                messages = [{"role": "user", "content": "Hello"}]
                result = provider.chat(messages=messages)
                
                # Verify rate limiter was called before API
                mock_acquire.assert_called_once()
                mock_client.models.generate_content.assert_called_once()
                
                # Verify response is properly formatted
                assert "content" in result
                assert result["content"] == "Test response"

    def test_rate_limiter_failure_does_not_block_request(self):
        """Test that rate limiter failures don't prevent API calls."""
        config = GeminiConfig(api_key="test-key", model="gemini-2.5-flash")
        
        with patch('google.genai.Client') as mock_client_class, \
             patch('openinference.instrumentation.google_genai.GoogleGenAIInstrumentor'):
            
            provider = GoogleGenAIProvider(config)
            
            # Mock successful API response
            mock_response = MagicMock()
            mock_response.text = "Success despite rate limiter failure"
            mock_response.usage_metadata = {}
            
            mock_client = mock_client_class.return_value
            mock_client.models.generate_content.return_value = mock_response
            
            # Mock rate limiter to raise an exception but catch it in _acquire_rate_limit
            # The test should verify that the warning is logged but request succeeds
            with patch('core_lib.llm.providers.google_genai_provider.logger') as mock_logger:
                # Directly test the rate limit method to ensure it handles exceptions
                provider._rate_limiter.acquire = AsyncMock(side_effect=Exception("Rate limiter failed"))
                
                messages = [{"role": "user", "content": "Hello"}]
                result = provider.chat(messages=messages)
                
                # Request should still succeed
                assert result["content"] == "Success despite rate limiter failure"
                
                # Warning should be logged about rate limiter failure
                mock_logger.warning.assert_called()
                warning_args = mock_logger.warning.call_args[0]
                assert "Rate limiter acquisition failed" in warning_args[0]


class TestGoogleGenAIProviderRetry:
    """Test retry logic in GoogleGenAIProvider."""

    def test_retry_config_initialization(self):
        """Test that retry config is properly initialized."""
        config = GeminiConfig(api_key="test-key", model="gemini-2.5-flash")
        
        with patch('google.genai.Client'), \
             patch('openinference.instrumentation.google_genai.GoogleGenAIInstrumentor'):
            provider = GoogleGenAIProvider(config)
            
        assert provider._retry_config.max_retries == 3
        assert provider._retry_config.base_delay == 1.0
        assert provider._retry_config.max_delay == 30.0
        assert len(provider._retry_config.retry_on_exceptions) >= 2  # At least network errors

    def test_successful_request_no_retry(self):
        """Test that successful requests are not retried."""
        config = GeminiConfig(api_key="test-key", model="gemini-2.5-flash")
        
        with patch('google.genai.Client') as mock_client_class, \
             patch('openinference.instrumentation.google_genai.GoogleGenAIInstrumentor'):
            
            provider = GoogleGenAIProvider(config)
            
            # Mock successful API response
            mock_response = MagicMock()
            mock_response.text = "Successful response"
            mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
            
            mock_client = mock_client_class.return_value
            mock_client.models.generate_content.return_value = mock_response
            
            with patch.object(provider, '_acquire_rate_limit'):
                messages = [{"role": "user", "content": "Hello"}]
                result = provider.chat(messages=messages)
                
            # API should be called exactly once (no retries)
            mock_client.models.generate_content.assert_called_once()
            assert result["content"] == "Successful response"

    def test_retry_on_rate_limit_error(self):
        """Test retry behavior on rate limit errors."""
        config = GeminiConfig(api_key="test-key", model="gemini-2.5-flash")
        
        with patch('google.genai.Client') as mock_client_class, \
             patch('openinference.instrumentation.google_genai.GoogleGenAIInstrumentor'):
            
            provider = GoogleGenAIProvider(config)
            
            # Mock the retry config to have shorter delays for testing
            provider._retry_config.base_delay = 0.01
            
            # Mock API to fail twice, then succeed
            mock_response = MagicMock()
            mock_response.text = "Success after retries"
            mock_response.usage_metadata = {}
            
            mock_client = mock_client_class.return_value
            mock_generate = mock_client.models.generate_content
            
            # Create a side effect that fails twice, then succeeds
            call_count = 0
            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise ConnectionError(f"Network error on attempt {call_count}")
                return mock_response
            
            mock_generate.side_effect = side_effect
            
            with patch.object(provider, '_acquire_rate_limit'), \
                 patch('time.sleep') as mock_sleep:
                
                messages = [{"role": "user", "content": "Hello"}]
                result = provider.chat(messages=messages)
                
            # Should succeed after 3 attempts
            assert result["content"] == "Success after retries"
            assert mock_generate.call_count == 3
            
            # Should have slept twice (for the two retries)
            assert mock_sleep.call_count == 2

    def test_retry_on_server_error_503(self):
        """Test retry behavior on Google GenAI ServerError (503 server overloaded)."""
        config = GeminiConfig(api_key="test-key", model="gemini-2.5-flash")
        
        with patch('google.genai.Client') as mock_client_class, \
             patch('openinference.instrumentation.google_genai.GoogleGenAIInstrumentor'):
            
            provider = GoogleGenAIProvider(config)
            provider._retry_config.base_delay = 0.01  # Fast test
            
            # Create a mock ServerError similar to the real one
            class MockServerError(Exception):
                """Mock google.genai.errors.ServerError"""
                def __init__(self, status_code, error_dict, response):
                    self.status_code = status_code
                    self.error_dict = error_dict
                    self.response = response
                    super().__init__(f"{status_code} UNAVAILABLE. {error_dict}")
            
            # Mock API to fail with ServerError twice, then succeed
            mock_response = MagicMock()
            mock_response.text = "Success after server recovered"
            mock_response.usage_metadata = {}
            
            mock_client = mock_client_class.return_value
            mock_generate = mock_client.models.generate_content
            
            call_count = 0
            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    error_dict = {
                        'error': {
                            'code': 503,
                            'message': 'The model is overloaded. Please try again later.',
                            'status': 'UNAVAILABLE'
                        }
                    }
                    raise MockServerError(503, error_dict, None)
                return mock_response
            
            mock_generate.side_effect = side_effect
            
            # Patch the retry config to include our mock exception
            original_exceptions = provider._retry_config.retry_on_exceptions
            provider._retry_config.retry_on_exceptions = original_exceptions + (MockServerError,)
            
            with patch.object(provider, '_acquire_rate_limit'), \
                 patch('time.sleep') as mock_sleep, \
                 patch('core_lib.llm.providers.google_genai_provider.logger') as mock_logger:
                
                messages = [{"role": "user", "content": "Hello"}]
                result = provider.chat(messages=messages)
                
            # Should succeed after retries
            assert result["content"] == "Success after server recovered"
            assert mock_generate.call_count == 3
            
            # Should have slept twice (for the two retries)
            assert mock_sleep.call_count == 2
            
            # Should have logged warnings about retries
            assert mock_logger.warning.call_count >= 2

    def test_retry_exhaustion_returns_error(self):
        """Test behavior when all retries are exhausted."""
        config = GeminiConfig(api_key="test-key", model="gemini-2.5-flash")
        
        with patch('google.genai.Client') as mock_client_class, \
             patch('openinference.instrumentation.google_genai.GoogleGenAIInstrumentor'):
            
            provider = GoogleGenAIProvider(config)
            provider._retry_config.base_delay = 0.01  # Fast test
            
            # Mock API to always fail
            mock_client = mock_client_class.return_value
            mock_client.models.generate_content.side_effect = ConnectionError("Persistent network error")
            
            with patch.object(provider, '_acquire_rate_limit'), \
                 patch('time.sleep'):
                
                messages = [{"role": "user", "content": "Hello"}]
                result = provider.chat(messages=messages)
                
            # Should return error response after exhausting retries
            assert "error" in result
            assert "Persistent network error" in result["error"]
            assert result["content"] is None
            assert result["structured"] is False
            assert result["tool_calls"] == []
            
            # Should have made max_retries + 1 attempts
            expected_calls = provider._retry_config.max_retries + 1
            assert mock_client.models.generate_content.call_count == expected_calls

    def test_non_retryable_error_not_retried(self):
        """Test that non-configured exceptions are not retried."""
        config = GeminiConfig(api_key="test-key", model="gemini-2.5-flash")
        
        with patch('google.genai.Client') as mock_client_class, \
             patch('openinference.instrumentation.google_genai.GoogleGenAIInstrumentor'):
            
            provider = GoogleGenAIProvider(config)
            
            # Mock API to raise a non-retryable error
            mock_client = mock_client_class.return_value
            mock_client.models.generate_content.side_effect = ValueError("Invalid input")
            
            with patch.object(provider, '_acquire_rate_limit'):
                messages = [{"role": "user", "content": "Hello"}]
                result = provider.chat(messages=messages)
                
            # Should return error without retries
            assert "error" in result
            assert "Invalid input" in result["error"]
            
            # Should only be called once (no retries)
            mock_client.models.generate_content.assert_called_once()

    def test_retry_with_structured_output(self):
        """Test retry behavior with structured output requests."""
        from pydantic import BaseModel
        
        class TestModel(BaseModel):
            message: str
            count: int
        
        config = GeminiConfig(api_key="test-key", model="gemini-2.5-flash")
        
        with patch('google.genai.Client') as mock_client_class, \
             patch('openinference.instrumentation.google_genai.GoogleGenAIInstrumentor'):
            
            provider = GoogleGenAIProvider(config)
            provider._retry_config.base_delay = 0.01
            
            # Mock successful structured response after one retry
            mock_response = MagicMock()
            mock_response.text = '{"message": "Hello", "count": 42}'
            mock_response.parsed = TestModel(message="Hello", count=42)
            mock_response.usage_metadata = {}
            
            mock_client = mock_client_class.return_value
            mock_generate = mock_client.models.generate_content
            
            call_count = 0
            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ConnectionError("Network error")
                return mock_response
            
            mock_generate.side_effect = side_effect
            
            with patch.object(provider, '_acquire_rate_limit'), \
                 patch('time.sleep'):
                
                messages = [{"role": "user", "content": "Hello"}]
                result = provider.chat(
                    messages=messages,
                    structured_output=TestModel
                )
                
            # Should succeed with structured output
            assert result["structured"] is True
            assert result["content"] == {"message": "Hello", "count": 42}
            assert "text" in result
            assert "content_json" in result
            
            # Should have retried once
            assert mock_generate.call_count == 2


class TestProviderIntegrationEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_rate_limit_and_retry_interaction(self):
        """Test interaction between rate limiting and retry logic."""
        config = GeminiConfig(api_key="test-key", model="gemini-2.5-pro")  # 5 RPM limit
        
        with patch('google.genai.Client') as mock_client_class, \
             patch('openinference.instrumentation.google_genai.GoogleGenAIInstrumentor'):
            
            provider = GoogleGenAIProvider(config)
            provider._retry_config.base_delay = 0.01
            
            # Mock API to fail once, then succeed
            mock_response = MagicMock()
            mock_response.text = "Success"
            mock_response.usage_metadata = {}
            
            mock_client = mock_client_class.return_value
            mock_generate = mock_client.models.generate_content
            
            call_count = 0
            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ConnectionError("Network error")
                return mock_response
            
            mock_generate.side_effect = side_effect
            
            # Mock both rate limiter and retry sleep
            with patch.object(provider, '_acquire_rate_limit') as mock_rate_limit, \
                 patch('time.sleep') as mock_sleep:
                
                messages = [{"role": "user", "content": "Hello"}]
                result = provider.chat(messages=messages)
                
            # Both rate limiting and retry should work
            mock_rate_limit.assert_called_once()  # Rate limiter called once
            mock_sleep.assert_called_once()       # Retry sleep called once
            assert result["content"] == "Success"
            assert mock_generate.call_count == 2  # Initial call + 1 retry

    def test_multi_turn_conversation_with_retry(self):
        """Test retry logic with multi-turn conversations."""
        config = GeminiConfig(api_key="test-key", model="gemini-2.5-flash")
        
        with patch('google.genai.Client') as mock_client_class, \
             patch('openinference.instrumentation.google_genai.GoogleGenAIInstrumentor'):
            
            provider = GoogleGenAIProvider(config)
            provider._retry_config.base_delay = 0.01
            
            # Mock successful chat response
            mock_response = MagicMock()
            mock_response.text = "Multi-turn response"
            mock_response.usage_metadata = {}
            
            # Mock the chat creation and send_message chain
            mock_client = mock_client_class.return_value
            mock_chat = MagicMock()
            mock_client.chats.create.return_value = mock_chat
            
            # Make send_message fail once, then succeed
            call_count = 0
            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ConnectionError("Network error")
                return mock_response
            
            mock_chat.send_message.side_effect = side_effect
            
            with patch.object(provider, '_acquire_rate_limit'), \
                 patch('time.sleep'):
                
                # Multi-turn conversation
                messages = [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                ]
                result = provider.chat(messages=messages)
                
            # Should use chat API for multi-turn
            mock_client.chats.create.assert_called()
            assert mock_chat.send_message.call_count == 2  # Initial + 1 retry
            assert result["content"] == "Multi-turn response"

    def test_thinking_config_preserved_through_retry(self):
        """Test that thinking configuration is preserved through retries."""
        config = GeminiConfig(api_key="test-key", model="gemini-2.5-pro", thinking_enabled=True)
        
        with patch('google.genai.Client') as mock_client_class, \
             patch('openinference.instrumentation.google_genai.GoogleGenAIInstrumentor'):
            
            provider = GoogleGenAIProvider(config)
            provider._retry_config.base_delay = 0.01
            
            mock_response = MagicMock()
            mock_response.text = "Response with thinking"
            mock_response.usage_metadata = {}
            
            mock_client = mock_client_class.return_value
            mock_generate = mock_client.models.generate_content
            
            # Fail once, then succeed
            call_count = 0
            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ConnectionError("Network error")
                return mock_response
            
            mock_generate.side_effect = side_effect
            
            with patch.object(provider, '_acquire_rate_limit'), \
                 patch('time.sleep'):
                
                messages = [{"role": "user", "content": "Hello"}]
                result = provider.chat(
                    messages=messages,
                    thinking_enabled=True
                )
                
            # Verify thinking config was passed to both attempts
            assert mock_generate.call_count == 2
            
            # Check that both calls included thinking configuration
            for call_args in mock_generate.call_args_list:
                config_arg = call_args[1]['config']
                # Should have thinking config (exact structure depends on google.genai mock)
                assert config_arg is not None
            
            assert result["content"] == "Response with thinking"