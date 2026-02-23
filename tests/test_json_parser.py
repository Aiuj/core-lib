"""Tests for JSON parser utilities."""

import pytest
from pydantic import BaseModel, Field
from typing import Optional

from core_lib.llm.json_parser import (
    _coerce_literal_fields,
    _extract_nested_match,
    _is_pydantic_schema_echo,
    _normalize_keys,
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


class TestIsPydanticSchemaEcho:
    """Tests for the schema-echo detection helper (_is_pydantic_schema_echo)."""

    # This is the exact JSON Schema that gemma-3-4b-it produced instead of an answer.
    _REAL_SCHEMA_ECHO = {
        "title": "ComplianceAnswer",
        "type": "object",
        "description": "Structured LLM response.",
        "properties": {
            "answer": {"description": "The answer to the question", "title": "Answer", "type": "string"},
            "compliance_category": {
                "default": "unknown",
                "description": "Compliance status",
                "enum": ["yes", "no", "partial", "custom", "unknown"],
                "title": "Compliance Category",
                "type": "string",
            },
            "compliance_reasoning": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "description": "Reasoning",
                "title": "Compliance Reasoning",
            },
        },
        "required": ["answer"],
    }

    def test_detects_real_schema_echo(self):
        """The exact schema-echo pattern from gemma-3-4b-it must be detected."""
        assert _is_pydantic_schema_echo(self._REAL_SCHEMA_ECHO, ComplianceAnswer) is True

    def test_does_not_flag_valid_instance(self):
        """A proper filled instance must NOT be flagged as a schema echo."""
        data = {"answer": "Alice is the CEO.", "compliance_category": "yes"}
        assert _is_pydantic_schema_echo(data, ComplianceAnswer) is False

    def test_does_not_flag_schema_as_instance_pattern(self):
        """The schema-as-instance pattern (values inside 'properties') is handled by
        _unwrap_schema_as_instance, not flagged as a schema echo."""
        data = {
            "title": "ComplianceAnswer",
            "type": "object",
            "properties": {
                "answer": "Alice is the CEO.",
                "compliance_category": "yes",
            },
        }
        # properties values are plain strings, not sub-schema dicts → not a schema echo
        assert _is_pydantic_schema_echo(data, ComplianceAnswer) is False

    def test_wrong_title_not_flagged(self):
        """A schema-shaped dict with a different title must not be flagged."""
        data = {**self._REAL_SCHEMA_ECHO, "title": "SomethingElse"}
        assert _is_pydantic_schema_echo(data, ComplianceAnswer) is False

    def test_non_dict_not_flagged(self):
        assert _is_pydantic_schema_echo("not a dict", ComplianceAnswer) is False  # type: ignore[arg-type]

    def test_parse_structured_output_returns_none_for_schema_echo(self):
        """parse_structured_output must return None (not the schema dict) when the model
        returns its own schema definition."""
        import json
        text = json.dumps(self._REAL_SCHEMA_ECHO)
        result = parse_structured_output(text, ComplianceAnswer)
        # Should fail validation (no 'answer' value), not silently return the schema
        assert result is None


class TestNormalizeKeys:
    def test_mixed_case_keys_remapped(self):
        data = {"Answer": "Alice is the CEO.", "Compliance_Category": "yes"}
        result = _normalize_keys(data, ComplianceAnswer)
        assert result is not None
        assert result["answer"] == "Alice is the CEO."
        assert result["compliance_category"] == "yes"

    def test_keys_with_spaces_remapped(self):
        data = {"answer": "text", "compliance category": "no"}
        result = _normalize_keys(data, ComplianceAnswer)
        assert result is not None
        assert result["compliance_category"] == "no"

    def test_missing_required_field_returns_none(self):
        # 'answer' is required; omitting it should return None
        data = {"Compliance_Category": "yes"}
        result = _normalize_keys(data, ComplianceAnswer)
        assert result is None

    def test_no_matching_keys_returns_none(self):
        data = {"foo": "bar", "baz": "qux"}
        result = _normalize_keys(data, ComplianceAnswer)
        assert result is None


class TestExtractNestedMatch:
    def test_answer_nested_under_response_key(self):
        data = {"response": {"answer": "Alice.", "compliance_category": "yes"}}
        result = _extract_nested_match(data, ComplianceAnswer)
        assert result is not None
        assert result["answer"] == "Alice."

    def test_answer_nested_under_data_key(self):
        data = {"data": {"answer": "Bob.", "compliance_category": "no"}}
        result = _extract_nested_match(data, ComplianceAnswer)
        assert result is not None
        assert result["answer"] == "Bob."

    def test_flat_dict_returns_none(self):
        data = {"answer": "Alice.", "compliance_category": "yes"}
        result = _extract_nested_match(data, ComplianceAnswer)
        assert result is None  # flat dict, nothing nested

    def test_nested_missing_required_returns_none(self):
        data = {"response": {"compliance_category": "yes"}}  # 'answer' missing
        result = _extract_nested_match(data, ComplianceAnswer)
        assert result is None


class TestCoerceLiteralFields:
    def test_lowercases_failing_literal_field(self):
        data = {"answer": "Alice is the CEO.", "compliance_category": "Yes"}
        errors = [{"type": "literal_error", "loc": ("compliance_category",)}]
        result = _coerce_literal_fields(data, errors)
        assert result["compliance_category"] == "yes"
        # Free-text field must not be touched
        assert result["answer"] == "Alice is the CEO."

    def test_ignores_non_literal_errors(self):
        data = {"answer": "Alice.", "compliance_category": "YES"}
        errors = [{"type": "missing", "loc": ("some_field",)}]
        result = _coerce_literal_fields(data, errors)
        # No change expected
        assert result["compliance_category"] == "YES"

    def test_handles_empty_errors(self):
        data = {"answer": "text", "compliance_category": "PARTIAL"}
        result = _coerce_literal_fields(data, [])
        assert result == data


class TestParseStructuredOutputFuzzyRecovery:
    """End-to-end tests for the full fuzzy recovery pipeline."""

    def test_literal_coercion_capitalized_enum(self):
        """Model returned capitalised enum value like 'Yes' instead of 'yes'."""
        text = '{"answer": "Alice is the CEO.", "compliance_category": "Yes"}'
        result = parse_structured_output(text, ComplianceAnswer)
        assert result is not None
        assert result["answer"] == "Alice is the CEO."
        assert result["compliance_category"] == "yes"

    def test_literal_coercion_uppercase_enum(self):
        text = '{"answer": "Yes, supported.", "compliance_category": "PARTIAL"}'
        result = parse_structured_output(text, ComplianceAnswer)
        assert result is not None
        assert result["compliance_category"] == "partial"

    def test_case_insensitive_keys(self):
        """Model returned camelCase or Title Case field names."""
        text = '{"Answer": "Alice is the CEO.", "Compliance_Category": "yes"}'
        result = parse_structured_output(text, ComplianceAnswer)
        assert result is not None
        assert result["answer"] == "Alice is the CEO."

    def test_nested_dict_extraction(self):
        """Model wrapped the answer inside a 'response' key."""
        text = '{"response": {"answer": "Alice is the CEO.", "compliance_category": "yes"}}'
        result = parse_structured_output(text, ComplianceAnswer)
        assert result is not None
        assert result["answer"] == "Alice is the CEO."

    def test_code_block_plus_capitalized_enum(self):
        """Markdown fence AND capitalised enum value."""
        text = '```json\n{"answer": "Alice.", "compliance_category": "Unknown"}\n```'
        result = parse_structured_output(text, ComplianceAnswer)
        assert result is not None
        assert result["compliance_category"] == "unknown"

    def test_unrecoverable_plain_text_returns_none(self):
        text = "I cannot answer that question."
        result = parse_structured_output(text, ComplianceAnswer)
        assert result is None

    def test_schema_echo_in_code_block_returns_none(self):
        """Schema echo wrapped in a markdown code block must still be detected."""
        import json
        schema_blob = {
            "title": "ComplianceAnswer",
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The answer"},
                "compliance_category": {
                    "type": "string",
                    "enum": ["yes", "no", "partial", "custom", "unknown"],
                },
            },
            "required": ["answer"],
        }
        text = f"```json\n{json.dumps(schema_blob)}\n```"
        result = parse_structured_output(text, ComplianceAnswer)
        assert result is None

    def test_schema_echo_followed_by_valid_instance(self):
        """Model outputs the schema definition, then outputs the actual answer."""
        import json
        schema_blob = {
            "title": "ComplianceAnswer",
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The answer"},
                "compliance_category": {
                    "type": "string",
                    "enum": ["yes", "no", "partial", "custom", "unknown"],
                },
            },
            "required": ["answer"],
        }
        instance_blob = {
            "answer": "Dr. Evelyn Reed",
            "compliance_category": "yes",
            "compliance_reasoning": "Explicitly stated."
        }
        text = f"```json\n{json.dumps(schema_blob)}\n{json.dumps(instance_blob)}\n```"
        result = parse_structured_output(text, ComplianceAnswer)
        assert result is not None
        assert result["answer"] == "Dr. Evelyn Reed"
        assert result["compliance_category"] == "yes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
