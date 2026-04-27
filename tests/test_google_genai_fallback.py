"""Tests for Google GenAI provider JSON mode fallback."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from types import SimpleNamespace
from pydantic import BaseModel

from core_lib.llm.providers.google_genai_provider import GoogleGenAIProvider, GeminiConfig


class SampleSchema(BaseModel):
    """Test schema for structured output."""
    result: str
    score: float


class TestGeminiJSONModeFallback:
    """Test JSON mode fallback for models that don't support it."""
    
    def test_gemma_model_detected_as_no_json_support(self):
        """Test that Gemma models are detected as not supporting JSON mode."""
        config = GeminiConfig(api_key="test-key", model="gemma-3-4b-it")
        
        with patch('core_lib.llm.providers.google_genai_provider.genai'):
            provider = GoogleGenAIProvider(config)
            assert not provider._supports_json_mode()
    
    def test_gemini_model_detected_as_json_support(self):
        """Test that Gemini models are detected as supporting JSON mode."""
        config = GeminiConfig(api_key="test-key", model="gemini-1.5-flash")
        
        with patch('core_lib.llm.providers.google_genai_provider.genai'):
            provider = GoogleGenAIProvider(config)
            assert provider._supports_json_mode()
    
    def test_gemma_2_model_no_json_support(self):
        """Test that Gemma 2 models don't support JSON mode."""
        config = GeminiConfig(api_key="test-key", model="gemma-2-9b-it")
        
        with patch('core_lib.llm.providers.google_genai_provider.genai'):
            provider = GoogleGenAIProvider(config)
            assert not provider._supports_json_mode()
    
    def test_fallback_triggered_for_gemma(self):
        """Test that fallback is triggered for Gemma models."""
        config = GeminiConfig(api_key="test-key", model="gemma-3-4b-it")
        
        with patch('core_lib.llm.providers.google_genai_provider.genai') as mock_genai:
            # Mock the client
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            
            # Mock response
            mock_response = MagicMock()
            mock_response.text = '{"result": "success", "score": 0.95}'
            mock_response.usage_metadata = {}
            mock_client.models.generate_content.return_value = mock_response
            
            provider = GoogleGenAIProvider(config)
            
            # Call with structured output
            result = provider.chat(
                messages=[{"role": "user", "content": "test"}],
                structured_output=SampleSchema
            )
            
            # Should use fallback and parse JSON from text
            assert result["structured"] is True
            assert result["content"]["result"] == "success"
            assert result["content"]["score"] == 0.95

    def test_gemini_json_in_text_is_parsed_when_native_parsed_missing(self):
        """Test native-mode fallback: parse JSON from text when resp.parsed is missing."""
        config = GeminiConfig(api_key="test-key", model="gemini-2.5-flash")

        with patch('google.genai.Client') as mock_client_class, \
             patch('openinference.instrumentation.google_genai.GoogleGenAIInstrumentor'):

            provider = GoogleGenAIProvider(config)

            text_json = '{"result": "success", "score": 0.95}'
            chunk = SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[SimpleNamespace(text=text_json)]
                        )
                    )
                ],
                function_calls=None,
                usage_metadata=SimpleNamespace(
                    prompt_token_count=10,
                    candidates_token_count=5,
                    total_token_count=15,
                ),
                parsed=None,
            )

            mock_client = mock_client_class.return_value
            mock_client.models.generate_content_stream.return_value = [chunk]

            result = provider.chat(
                messages=[{"role": "user", "content": "test"}],
                structured_output=SampleSchema,
            )

            assert result["structured"] is True
            assert result["content"] == {"result": "success", "score": 0.95}
            assert result.get("text") == text_json
            assert result.get("content_json") == text_json

    def test_single_turn_multimodal_uses_chat_path_not_generate_content(self):
        """Single-turn multimodal input must be sent via chat path after conversion."""
        config = GeminiConfig(api_key="test-key", model="gemini-2.5-flash")

        with patch('google.genai.Client') as mock_client_class, \
             patch('openinference.instrumentation.google_genai.GoogleGenAIInstrumentor'):

            provider = GoogleGenAIProvider(config)
            mock_client = mock_client_class.return_value

            chunk = SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[SimpleNamespace(text="ok")]
                        )
                    )
                ],
                function_calls=None,
                usage_metadata=SimpleNamespace(
                    prompt_token_count=10,
                    candidates_token_count=5,
                    total_token_count=15,
                ),
            )

            mock_chat = MagicMock()
            mock_chat.send_message_stream.return_value = [chunk]
            mock_client.chats.create.return_value = mock_chat

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                    {"type": "text", "text": "Analyze this image"},
                ],
            }]

            with patch.object(provider, "_to_genai_messages", return_value="converted-prompt") as to_genai:
                result = provider.chat(messages=messages)

            assert result["content"] == "ok"
            to_genai.assert_called_once()
            mock_client.models.generate_content_stream.assert_not_called()
            mock_chat.send_message_stream.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
