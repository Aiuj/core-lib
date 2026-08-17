"""Test that Gemini provider fixes work correctly.

This test verifies:
1. No double instrumentation warnings
2. No warnings about non-text parts when extracting response text
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pydantic import BaseModel

from core_lib.llm.providers.google_genai_provider import (
    GoogleGenAIProvider,
    GeminiConfig,
    _instrumentation_initialized,
)


class SampleResponse(BaseModel):
    """Test schema for structured output."""
    answer: str
    confidence: float


class TestGeminiWarningsFix:
    """Test fixes for Gemini provider warnings."""

    @patch("google.genai", create=True)
    @patch("openinference.instrumentation.google_genai.GoogleGenAIInstrumentor")
    def test_instrumentation_called_once_only(self, mock_instrumentor, mock_genai):
        """Verify instrumentation is only called once even with multiple provider instances."""
        # Setup mocks
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        mock_instrumentor_instance = Mock()
        mock_instrumentor.return_value = mock_instrumentor_instance

        # Create first provider
        config1 = GeminiConfig(api_key="test-key-1", model="gemini-2.5-flash")
        provider1 = GoogleGenAIProvider(config1)

        # Verify instrumentation was called once
        assert mock_instrumentor.call_count == 1
        assert mock_instrumentor_instance.instrument.call_count == 1

        # Create second provider
        config2 = GeminiConfig(api_key="test-key-2", model="gemini-2.5-flash")
        provider2 = GoogleGenAIProvider(config2)

        # Verify instrumentation was NOT called again
        assert mock_instrumentor.call_count == 1  # Still 1
        assert mock_instrumentor_instance.instrument.call_count == 1  # Still 1

    @patch("google.genai", create=True)
    @patch("openinference.instrumentation.google_genai.GoogleGenAIInstrumentor")
    def test_extract_text_from_response_with_thinking_parts(self, mock_instrumentor, mock_genai):
        """Verify _extract_text_from_response handles thinking parts correctly."""
        # Setup mocks
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        
        config = GeminiConfig(api_key="test-key", model="gemini-2.5-flash")
        provider = GoogleGenAIProvider(config)

        # Create mock response with thinking parts (non-text parts)
        mock_response = Mock()
        
        # Setup candidates with both text and non-text (thought_signature) parts
        mock_text_part1 = Mock()
        mock_text_part1.text = "This is the answer "
        
        mock_text_part2 = Mock()
        mock_text_part2.text = "to the question."
        
        mock_thinking_part = Mock()
        mock_thinking_part.text = None  # Non-text parts have no .text attribute
        
        mock_content = Mock()
        mock_content.parts = [mock_text_part1, mock_thinking_part, mock_text_part2]
        
        mock_candidate = Mock()
        mock_candidate.content = mock_content
        
        mock_response.candidates = [mock_candidate]
        mock_response.text = "This should not be called"  # Fallback

        # Extract text
        result = provider._extract_text_from_response(mock_response)

        # Verify only text parts were concatenated
        assert result == "This is the answer to the question."

    @patch("google.genai", create=True)
    @patch("openinference.instrumentation.google_genai.GoogleGenAIInstrumentor")
    def test_extract_text_fallback_when_no_candidates(self, mock_instrumentor, mock_genai):
        """Verify _extract_text_from_response falls back to .text when needed."""
        # Setup mocks
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        
        config = GeminiConfig(api_key="test-key", model="gemini-2.5-flash")
        provider = GoogleGenAIProvider(config)

        # Create mock response without candidates
        mock_response = Mock()
        mock_response.candidates = []
        mock_response.text = "Fallback text"

        # Extract text
        result = provider._extract_text_from_response(mock_response)

        # Verify fallback was used
        assert result == "Fallback text"

    @patch("google.genai", create=True)
    @patch("openinference.instrumentation.google_genai.GoogleGenAIInstrumentor")
    def test_extract_text_handles_empty_parts(self, mock_instrumentor, mock_genai):
        """Verify _extract_text_from_response handles empty parts gracefully."""
        # Setup mocks
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        
        config = GeminiConfig(api_key="test-key", model="gemini-2.5-flash")
        provider = GoogleGenAIProvider(config)

        # Create mock response with empty parts
        mock_content = Mock()
        mock_content.parts = []
        
        mock_candidate = Mock()
        mock_candidate.content = mock_content
        
        mock_response = Mock()
        mock_response.candidates = [mock_candidate]
        mock_response.text = "Fallback text"

        # Extract text
        result = provider._extract_text_from_response(mock_response)

        # Verify fallback was used when no text parts found
        assert result == "Fallback text"

    @patch("google.genai", create=True)
    @patch("openinference.instrumentation.google_genai.GoogleGenAIInstrumentor")
    def test_dual_keys_prefers_gemini_and_preserves_google_key(self, mock_instrumentor, mock_genai):
        """Verify that having both GOOGLE_API_KEY and GEMINI_API_KEY prefers GEMINI and preserves GOOGLE_API_KEY."""
        import os
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "google_picker_key", "GEMINI_API_KEY": "gemini_key"}):
            config = GeminiConfig.from_env()
            assert config.api_key == "gemini_key"
            provider = GoogleGenAIProvider(config)
            mock_genai.Client.assert_called_once_with(
                api_key="gemini_key",
                http_options={"timeout": 60000, "retry_options": None},
            )
            # Ensure GOOGLE_API_KEY remains untouched in the environment
            assert os.environ.get("GOOGLE_API_KEY") == "google_picker_key"
            assert os.environ.get("GEMINI_API_KEY") == "gemini_key"

    def test_gemini_config_from_env_fallback_to_google_api_key(self):
        """Verify that GeminiConfig.from_env falls back to GOOGLE_API_KEY when GEMINI_API_KEY is not set."""
        import os
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "google_fallback_key"}, clear=True):
            config = GeminiConfig.from_env()
            assert config.api_key == "google_fallback_key"
