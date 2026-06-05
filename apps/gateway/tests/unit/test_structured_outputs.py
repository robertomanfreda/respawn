import pytest

from src.schemas.errors import OpenAIError
from src.services.structured_outputs import schema_from_response_format, validate_text_against_schema


def test_structured_output_validation_success():
    schema = schema_from_response_format({"type": "json_schema", "json_schema": {"schema": {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}}})
    validate_text_against_schema('{"ok": true}', schema)


def test_structured_output_validation_failure():
    with pytest.raises(OpenAIError):
        validate_text_against_schema("not json", {"type": "object"})
