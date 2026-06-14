import pytest

from src.schemas.errors import OpenAIError
from src.services.structured_outputs import repair_instruction, schema_from_response_format, validate_text_against_schema


def test_structured_output_validation_success():
    schema = schema_from_response_format({"type": "json_schema", "json_schema": {"schema": {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}}})
    validate_text_against_schema('{"ok": true}', schema)


def test_structured_output_validation_failure():
    with pytest.raises(OpenAIError):
        validate_text_against_schema("not json", {"type": "object"})


def test_repair_instruction_is_conservative_and_schema_agnostic():
    schema = {
        "type": "object",
        "properties": {"value": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
        "required": ["value"],
    }
    instruction = repair_instruction(schema)

    assert "schema repair only, not task re-execution" in instruction
    assert "Preserve the original output's semantic meaning" in instruction
    assert "Do not add new objects, array items" in instruction
    assert "Preserve existing object identities" in instruction
    assert "list item order" in instruction
    assert "Only change fields that are invalid according to the schema" in instruction
    assert "0-10 score" in instruction
    assert "0.0..1.0" in instruction
    assert "9 -> 0.9" in instruction
    assert "Return only valid JSON with no markdown or commentary" in instruction
