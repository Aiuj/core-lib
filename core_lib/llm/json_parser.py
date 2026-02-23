from core_lib import get_module_logger
import json
import re
from typing import Dict, Any, Optional, Type, List

from pydantic import BaseModel, ValidationError

logger = get_module_logger()

def clean_and_parse_json_response(response_str, force_list=False):
    """
    Extracts and parses a JSON array from the response string.
    Handles corrupted or incomplete responses by finding the valid JSON portion.
    Returns a list of dict or None if parsing fails.
    """
    # Handle None or empty input
    if not response_str:
        logger.warning("Empty or None response string provided for JSON parsing")
        return None

    # Convert to string if needed
    if not isinstance(response_str, str):
        response_str = str(response_str)


    # Log the raw response for debugging (truncated if too long)
    response_preview = response_str[:500] + "..." if len(response_str) > 500 else response_str
    logger.debug(f"Parsing JSON response: {response_preview}")

    # Remove common text wrappers that models might add
    clean_response = response_str.strip()
    
    # Remove markdown code blocks if present
    if clean_response.startswith("```json") and clean_response.endswith("```"):
        clean_response = clean_response[7:-3].strip()
    elif clean_response.startswith("```") and clean_response.endswith("```"):
        clean_response = clean_response[3:-3].strip()
    

    # Try direct parsing first on clean string
    try:
        parsed = json.loads(clean_response)
        # Ensure it's a list
        if isinstance(parsed, list):
            logger.info(f"Successfully parsed JSON array with {len(parsed)} items")
            return parsed
        elif isinstance(parsed, dict):
            if force_list:
                logger.info("Parsed single JSON object, converting to list")
                return [parsed]
            else:
                logger.info("Parsed single JSON object")
                return parsed
        else:
            logger.warning(f"Parsed JSON is not array or object, got: {type(parsed)}")
            return None
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parsing failed: {e}")

    # Try to extract as many valid JSON objects/arrays as possible from a possibly truncated response
    # This will extract items from a top-level array, even if the last item is incomplete
    # Only works for top-level arrays (not objects)
    array_match = re.search(r'\[.*', clean_response, re.DOTALL)
    if array_match:
        array_str = array_match.group(0)
        # Remove trailing commas before closing brackets
        array_str = re.sub(r',([\s]*[\]])', r'\1', array_str)
        # Try to extract as many complete items as possible
        items = []
        decoder = json.JSONDecoder()
        idx = 0
        # Skip initial whitespace and opening bracket
        while idx < len(array_str) and array_str[idx] not in '[{':
            idx += 1
        if idx < len(array_str) and array_str[idx] == '[':
            idx += 1
        while idx < len(array_str):
            # Skip whitespace and commas
            while idx < len(array_str) and array_str[idx] in ' \n\r\t,':
                idx += 1
            if idx >= len(array_str) or array_str[idx] == ']':
                break
            try:
                obj, end = decoder.raw_decode(array_str, idx)
                items.append(obj)
                idx = end
            except json.JSONDecodeError:
                # Truncated/incomplete item at the end, ignore
                break
        if items:
            logger.info(f"Successfully extracted {len(items)} JSON items from possibly truncated array")
            return items
        else:
            logger.warning("No complete JSON items could be extracted from array")
            return None

    # If not a top-level array, try to extract as many objects as possible (for object streams)
    # This is less robust, but can help if the response is a stream of objects
    object_matches = list(re.finditer(r'\{', clean_response))
    if object_matches:
        items = []
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(clean_response):
            # Find next object
            next_obj = clean_response.find('{', idx)
            if next_obj == -1:
                break
            try:
                obj, end = decoder.raw_decode(clean_response, next_obj)
                items.append(obj)
                idx = end
            except json.JSONDecodeError:
                # Truncated/incomplete object at the end, ignore
                break
        if items:
            logger.info(f"Successfully extracted {len(items)} JSON objects from possibly truncated stream")
            return items
        else:
            logger.warning("No complete JSON objects could be extracted from stream")
            return None

    logger.warning("No JSON brackets found in response or unable to extract valid items")
    return None


def _strip_markdown_code_block(text: str) -> str:
    """Strip markdown code block wrappers (```json ... ``` or ``` ... ```) from text."""
    stripped = text.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        return stripped[7:-3].strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return stripped[3:-3].strip()
    return stripped


def _unwrap_schema_as_instance(
    data: Dict[str, Any],
    schema: Type[BaseModel],
) -> Optional[Dict[str, Any]]:
    """Detect and unwrap when a model returns a JSON Schema wrapper with values in 'properties'.

    Some local models (e.g. Ollama with smaller models) return a JSON Schema-shaped
    document instead of a plain instance::

        {
          "title": "ComplianceAwareAnswer",
          "type": "object",
          "properties": {
            "answer": "<actual answer text>",
            "compliance_category": "yes"
          },
          "required": ["answer"]
        }

    When the values inside ``properties`` are plain values (not sub-schema dicts),
    this function extracts them so that Pydantic validation can succeed.

    Returns the unwrapped properties dict when the pattern is detected, else ``None``.
    """
    if not isinstance(data, dict):
        return None

    # Must look like a JSON Schema wrapper
    if not (data.get("type") == "object" and isinstance(data.get("properties"), dict)):
        return None

    properties: Dict[str, Any] = data["properties"]
    if not properties:
        return None

    # Real schema sub-objects have at least one of these keys as dict values
    _schema_indicator_keys = frozenset(
        {"type", "description", "enum", "items", "properties", "anyOf", "allOf", "oneOf", "$ref"}
    )
    has_schema_values = any(
        isinstance(v, dict) and bool(_schema_indicator_keys & set(v.keys()))
        for v in properties.values()
    )
    if has_schema_values:
        # Genuine schema definition, not value payload
        return None

    # At least some keys must overlap with the Pydantic model's fields
    expected_fields = set(schema.model_fields.keys())
    if not (expected_fields & set(properties.keys())):
        return None

    logger.info(
        "Detected schema-as-instance response pattern for %s – unwrapping 'properties'",
        schema.__name__,
    )
    return properties


def _is_pydantic_schema_echo(data: Dict[str, Any], schema: Type[BaseModel]) -> bool:
    """Return True when *data* looks like the model echoed the Pydantic JSON Schema definition.

    This is a distinct failure mode from :func:`_unwrap_schema_as_instance`:

    * **Schema-as-instance** (recoverable): model wraps actual values inside a
      ``"properties"`` dict, e.g. ``{"properties": {"answer": "Alice", ...}}``.
    * **Schema-echo** (this function, not recoverable): model returns the *definition*
      itself — every property value is a sub-schema dict containing ``type``,
      ``description``, ``enum``, etc., not a real value.

    Detection heuristics (all must hold):

    * ``title`` equals the schema class name.
    * ``type`` is ``"object"``.
    * ``properties`` is a dict whose values are sub-schema dicts (contain at least
      one of ``type``, ``description``, ``enum``, ``items``, ``anyOf``, …).
    """
    if not isinstance(data, dict):
        return False
    if data.get("title") != schema.__name__:
        return False
    if data.get("type") != "object":
        return False
    properties = data.get("properties")
    if not isinstance(properties, dict) or not properties:
        return False
    _schema_indicator_keys = frozenset(
        {"type", "description", "enum", "items", "properties", "anyOf", "allOf", "oneOf", "$ref", "title"}
    )
    return any(
        isinstance(v, dict) and bool(_schema_indicator_keys & set(v.keys()))
        for v in properties.values()
    )


def _normalize_keys(
    data: Dict[str, Any],
    schema: Type[BaseModel],
) -> Optional[Dict[str, Any]]:
    """Remap dict keys to match schema field names using case-insensitive matching.

    Handles variations like ``"Answer"`` → ``"answer"``,
    ``"Compliance Category"`` → ``"compliance_category"``.

    Returns remapped dict only when all *required* fields are present, else ``None``.
    """
    expected_fields = set(schema.model_fields.keys())
    remapped: Dict[str, Any] = {}
    for k, v in data.items():
        normalized = k.lower().strip().replace(" ", "_").replace("-", "_")
        if normalized in expected_fields:
            remapped[normalized] = v
    if not remapped:
        return None
    required_fields = {
        name for name, field in schema.model_fields.items() if field.is_required()
    }
    if not required_fields.issubset(remapped.keys()):
        return None
    return remapped


def _extract_nested_match(
    data: Dict[str, Any],
    schema: Type[BaseModel],
) -> Optional[Dict[str, Any]]:
    """Search one level of nesting for a dict whose keys match the schema.

    Handles patterns like ``{"response": {"answer": "...", ...}}`` or
    ``{"data": {"answer": "...", ...}}``.

    Returns the first nested dict that contains all required fields, else ``None``.
    """
    required_fields = {
        name for name, field in schema.model_fields.items() if field.is_required()
    }
    if not required_fields:
        return None
    for v in data.values():
        if not isinstance(v, dict):
            continue
        candidate_keys = {k.lower().strip() for k in v.keys()}
        if required_fields.issubset(candidate_keys):
            return v
    return None


def _coerce_literal_fields(
    data: Dict[str, Any],
    validation_errors: list,
) -> Dict[str, Any]:
    """Lowercase string values that failed Literal/enum validation.

    Uses the structured error list from a :class:`pydantic.ValidationError` to
    target *only* the failing enum/Literal fields, leaving free-text fields like
    ``"answer"`` untouched.
    """
    result = dict(data)
    for error in validation_errors:
        if error.get("type") in ("literal_error", "enum"):
            loc = error.get("loc", ())
            if loc and isinstance(loc[0], str):
                field_name = loc[0]
                if field_name in result and isinstance(result[field_name], str):
                    result[field_name] = result[field_name].lower().strip()
    return result


def extract_all_json_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract all JSON objects/arrays from text response.
    
    Args:
        text: Text response that may contain JSON
        
    Returns:
        List of parsed JSON dicts/lists
    """
    if not text:
        return []

    text = _strip_markdown_code_block(text)
    
    # Try parsing entire text as JSON
    try:
        result = json.loads(text.strip())
        if isinstance(result, (dict, list)):
            return [result]
    except (json.JSONDecodeError, ValueError):
        pass
    
    results = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        # Find next object or array
        next_obj = text.find('{', idx)
        next_arr = text.find('[', idx)
        
        if next_obj == -1 and next_arr == -1:
            break
            
        if next_obj != -1 and (next_arr == -1 or next_obj < next_arr):
            start_idx = next_obj
        else:
            start_idx = next_arr
            
        try:
            obj, end = decoder.raw_decode(text, start_idx)
            if isinstance(obj, (dict, list)):
                results.append(obj)
            idx = end
        except json.JSONDecodeError:
            # Not a valid JSON object, move past the opening bracket
            idx = start_idx + 1
            
    return results


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from text response.
    
    Args:
        text: Text response that may contain JSON
        
    Returns:
        Parsed JSON dict/list or None if extraction fails
    """
    results = extract_all_json_from_text(text)
    return results[0] if results else None


def parse_structured_output(
    text: str,
    schema: Type[BaseModel],
) -> Optional[Dict[str, Any]]:
    """Parse text response into structured output matching Pydantic schema.

    Comprehensive fuzzy recovery pipeline for LLM responses.  Applied in order:

    1. **Markdown stripping** – removes ````json ... ``` `` and ``` ... ``` wrappers.
    2. **JSON extraction** – direct parse, regex object/array extraction.
    3. **Direct validation** – plain Pydantic ``model_validate``.
    4. **Literal coercion** – lowercases string values that fail enum/Literal checks
       (e.g. ``"Yes"`` → ``"yes"`` for a ``Literal["yes", "no"]`` field).
    5. **Schema-echo guard** – model returned its own JSON Schema definition instead
       of a filled instance (e.g. gemma-3-4b-it); unrecoverable, returns ``None``.
    6. **Schema-as-instance unwrap** – model wrapped actual values inside a
       ``"properties"`` dict; extracts them and re-validates.
    7. **Case-insensitive key normalisation** – remaps ``"Answer"`` → ``"answer"``
       etc., then re-validates (with literal coercion).
    8. **Nested dict extraction** – searches one level deep for a dict whose keys
       match the schema (e.g. ``{"response": {"answer": ...}}``).

    Args:
        text: Raw LLM text response (may contain markdown, prose, embedded JSON).
        schema: Pydantic model defining the expected structure.

    Returns:
        ``model_dump()`` dict when any strategy succeeds, else ``None``.
    """
    # ── Step 1-2: extract JSON (also strips markdown code blocks) ────────────
    # We extract ALL JSON objects from the text. Some models (like gemma-3-4b-it)
    # output the JSON Schema definition first, followed by the actual answer JSON.
    # We iterate through all extracted objects and return the first one that validates.
    json_objects = extract_all_json_from_text(text)
    if not json_objects:
        logger.warning(
            "parse_structured_output: could not extract any JSON from response "
            "for schema %s",
            schema.__name__,
        )
        return None

    # ── Inner helper: validate a candidate dict, with literal coercion on fail ─
    def _try_validate(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            return schema.model_validate(candidate).model_dump()
        except ValidationError as exc:
            coerced = _coerce_literal_fields(candidate, exc.errors())
            if coerced != candidate:
                try:
                    return schema.model_validate(coerced).model_dump()
                except ValidationError:
                    pass
        except Exception as exc:
            logger.debug("parse_structured_output: unexpected validation error: %s", exc)
        return None

    for json_data in json_objects:
        if not isinstance(json_data, dict):
            continue

        # ── Step 3-4: direct validation (+ literal coercion) ─────────────────────
        result = _try_validate(json_data)
        if result is not None:
            return result

        # ── Step 5: schema-echo guard ─────────────────────────────────────────────
        if _is_pydantic_schema_echo(json_data, schema):
            logger.debug(
                "parse_structured_output: skipping JSON object that is a schema definition "
                "for %s",
                schema.__name__,
            )
            continue

        # ── Step 6: schema-as-instance unwrap ─────────────────────────────────────
        unwrapped = _unwrap_schema_as_instance(json_data, schema)
        if unwrapped is not None:
            result = _try_validate(unwrapped)
            if result is not None:
                return result
            logger.debug(
                "parse_structured_output: schema-as-instance unwrap failed for %s",
                schema.__name__,
            )

        # ── Step 7: case-insensitive key normalisation ────────────────────────────
        normalized = _normalize_keys(json_data, schema)
        if normalized is not None:
            result = _try_validate(normalized)
            if result is not None:
                logger.info(
                    "parse_structured_output: recovered via key normalisation for %s",
                    schema.__name__,
                )
                return result

        # ── Step 8: nested dict extraction ───────────────────────────────────────
        nested = _extract_nested_match(json_data, schema)
        if nested is not None:
            result = _try_validate(nested)
            if result is not None:
                logger.info(
                    "parse_structured_output: recovered from nested dict for %s",
                    schema.__name__,
                )
                return result

    logger.warning(
        "parse_structured_output: all recovery strategies exhausted for schema %s",
        schema.__name__,
    )
    return None


def augment_prompt_for_json(
    prompt: str,
    schema: Type[BaseModel],
) -> str:
    """Augment prompt to request JSON output matching schema.
    
    Args:
        prompt: Original user prompt
        schema: Pydantic model defining expected structure
        
    Returns:
        Enhanced prompt requesting JSON format
    """
    # Get schema description
    schema_json = schema.model_json_schema()
    
    # Build JSON format instruction
    json_instruction = f"""
Your response MUST be valid JSON matching this exact schema:

{json.dumps(schema_json, indent=2)}

Respond ONLY with the JSON object, no additional text or explanation.
"""
    
    return f"{prompt}\n\n{json_instruction}"
