"""Tests for JSON parser utilities."""

import pytest
from pydantic import BaseModel, Field
from typing import Optional

from core_lib.llm.json_parser import (
    _strip_markdown_code_block,
    _unwrap_schema_as_instance,
    extract_json_from_text,
    parse_structured_output,
    augment_prompt_for_json,
)


class SampleSchema(BaseModel):
    """Test schema for structured output."""
    result: str
    score: float
    is_valid: bool


class TestExtractJsonFromText:
    """Test JSON extraction from text."""
    
    def test_extract_valid_json_object(self):
        """Test extracting valid JSON object."""
        text = '{"key": "value", "number": 42}'
        result = extract_json_from_text(text)
        assert result == {"key": "value", "number": 42}
    
    def test_extract_json_with_surrounding_text(self):
        """Test extracting JSON from text with surrounding content."""
        text = 'Here is the result: {"key": "value"} and some more text'
        result = extract_json_from_text(text)
        assert result == {"key": "value"}
    
    def test_extract_json_array(self):
        """Test extracting JSON array."""
        text = '[{"item": 1}, {"item": 2}]'
        result = extract_json_from_text(text)
        assert result == [{"item": 1}, {"item": 2}]
    
    def test_extract_json_in_code_block(self):
        """Test extracting JSON from markdown ```json ... ``` code block."""
        text = '```json\n{"key": "value"}\n```'
        result = extract_json_from_text(text)
        assert result == {"key": "value"}

    def test_extract_json_in_bare_code_block(self):
        """Test extracting JSON from bare ``` ... ``` code block."""
        text = '```\n{"key": "value"}\n```'
        result = extract_json_from_text(text)
        assert result == {"key": "value"}
    
    def test_no_json_in_text(self):
        """Test when no JSON is present."""
        text = "This is just plain text with no JSON"
        result = extract_json_from_text(text)
        assert result is None
    
    def test_empty_text(self):
        """Test with empty text."""
        result = extract_json_from_text("")
        assert result is None


class TestParseStructuredOutput:
    """Test structured output parsing."""
    
    def test_parse_valid_structured_output(self):
        """Test parsing valid structured output."""
        text = '{"result": "success", "score": 0.95, "is_valid": true}'
        result = parse_structured_output(text, SampleSchema)
        
        assert result is not None
        assert result["result"] == "success"
        assert result["score"] == 0.95
        assert result["is_valid"] is True
    
    def test_parse_structured_output_with_text(self):
        """Test parsing structured output embedded in text."""
        text = 'Here is the analysis:\n{"result": "success", "score": 0.95, "is_valid": true}\nEnd of analysis'
        result = parse_structured_output(text, SampleSchema)
        
        assert result is not None
        assert result["result"] == "success"
    
    def test_parse_invalid_schema(self):
        """Test parsing with invalid schema."""
        text = '{"wrong_field": "value"}'
        result = parse_structured_output(text, SampleSchema)
        
        assert result is None
    
    def test_parse_no_json(self):
        """Test parsing when no JSON is present."""
        text = "No JSON here"
        result = parse_structured_output(text, SampleSchema)
        
        assert result is None


class TestAugmentPromptForJson:
    """Test prompt augmentation for JSON output."""
    
    def test_augment_prompt(self):
        """Test that prompt is augmented with schema."""
        prompt = "Analyze this data"
        result = augment_prompt_for_json(prompt, SampleSchema)
        
        assert "Analyze this data" in result
        assert "JSON" in result
        assert "schema" in result.lower()
        assert "result" in result  # Field from schema
        assert "score" in result   # Field from schema
    
    def test_augment_empty_prompt(self):
        """Test augmenting empty prompt."""
        result = augment_prompt_for_json("", SampleSchema)
        
        assert "JSON" in result
        assert "result" in result


class TestStripMarkdownCodeBlock:
    """Tests for _strip_markdown_code_block helper."""

    def test_strips_json_fence(self):
        assert _strip_markdown_code_block('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_strips_bare_fence(self):
        assert _strip_markdown_code_block('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_no_fence_unchanged(self):
        assert _strip_markdown_code_block('{"a": 1}') == '{"a": 1}'

    def test_strips_whitespace(self):
        assert _strip_markdown_code_block('  ```json\n{"a": 1}\n```  ') == '{"a": 1}'


class ComplianceAnswer(BaseModel):
    """Minimal stand-in for the ComplianceAwareAnswer used in mcp-doc-qa."""
    answer: str
    compliance_category: str = "unknown"
    compliance_reasoning: Optional[str] = None


class TestUnwrapSchemaAsInstance:
    """Tests for the schema-as-instance detection helper."""

    def test_detects_schema_wrapper(self):
        """Typical pattern emitted by smaller local models."""
        data = {
            "title": "ComplianceAnswer",
            "type": "object",
            "properties": {
                "answer": "The company was founded in 2012.",
                "compliance_category": "yes",
                "compliance_reasoning": "All fields are present.",
            },
            "required": ["answer"],
        }
        result = _unwrap_schema_as_instance(data, ComplianceAnswer)
        assert result == data["properties"]

    def test_ignores_real_schema(self):
        """A genuine JSON Schema definition (values are sub-schema objects) must NOT be unwrapped."""
        data = {
            "title": "ComplianceAnswer",
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The answer text"},
                "compliance_category": {"type": "string"},
            },
            "required": ["answer"],
        }
        result = _unwrap_schema_as_instance(data, ComplianceAnswer)
        assert result is None

    def test_ignores_non_object_type(self):
        data = {"type": "array", "properties": {"answer": "hi"}}
        result = _unwrap_schema_as_instance(data, ComplianceAnswer)
        assert result is None

    def test_ignores_non_dict(self):
        result = _unwrap_schema_as_instance([1, 2, 3], ComplianceAnswer)  # type: ignore[arg-type]
        assert result is None

    def test_ignores_no_overlapping_fields(self):
        """Properties dict that shares no keys with the schema must not be unwrapped."""
        data = {"type": "object", "properties": {"foo": "bar", "baz": "qux"}}
        result = _unwrap_schema_as_instance(data, ComplianceAnswer)
        assert result is None


class TestParseStructuredOutputLocalModelQuirks:
    """End-to-end tests for local-model response patterns."""

    def test_code_block_wrapped_json(self):
        """Model wraps valid JSON in a markdown code block."""
        text = '```json\n{"answer": "Founded 2012", "compliance_category": "yes"}\n```'
        result = parse_structured_output(text, ComplianceAnswer)
        assert result is not None
        assert result["answer"] == "Founded 2012"
        assert result["compliance_category"] == "yes"

    def test_schema_as_instance_response(self):
        """Model returns JSON Schema with actual values in 'properties'."""
        text = (
            '{"title": "ComplianceAnswer", "type": "object", '
            '"properties": {"answer": "Founded 2012.", "compliance_category": "yes", '
            '"compliance_reasoning": "All required info present."}, '
            '"required": ["answer"]}'
        )
        result = parse_structured_output(text, ComplianceAnswer)
        assert result is not None
        assert result["answer"] == "Founded 2012."
        assert result["compliance_category"] == "yes"

    def test_code_block_plus_schema_as_instance(self):
        """Worst case: code block wrapper AND schema-as-instance response body."""
        inner = (
            '{"title": "ComplianceAnswer", "type": "object", '
            '"properties": {"answer": "Founded 2012.", "compliance_category": "yes"}, '
            '"required": ["answer"]}'
        )
        text = f'```json\n{inner}\n```'
        result = parse_structured_output(text, ComplianceAnswer)
        assert result is not None
        assert result["answer"] == "Founded 2012."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
