import json
from typing import Any

from jsonschema import ValidationError, validate

from src.schemas.errors import OpenAIError


def schema_from_response_format(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    if not response_format:
        return None
    if "format" in response_format:
        return schema_from_response_format(response_format.get("format"))
    if response_format.get("type") == "json_schema":
        json_schema = response_format.get("json_schema", {})
        if not json_schema and "schema" in response_format:
            return response_format["schema"]
        return json_schema.get("schema", json_schema)
    if "schema" in response_format:
        return response_format["schema"]
    return None


def validate_text_against_schema(text: str, schema: dict[str, Any] | None) -> None:
    if not schema:
        return
    try:
        payload = json.loads(text)
        validate(instance=payload, schema=schema)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise OpenAIError("Response did not match the requested JSON schema.", status_code=502, type="server_error", code="structured_output_validation_failed") from exc


def repair_instruction(schema: dict[str, Any]) -> str:
    return (
        "The previous assistant output was not valid for the requested JSON schema. "
        "Return only valid JSON with no markdown or commentary. "
        f"JSON schema: {json.dumps(schema, separators=(',', ':'))}"
    )


def example_for_schema(schema: dict[str, Any]) -> Any:
    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        required = schema.get("required") or list(properties)
        return {name: example_for_schema(properties.get(name, {})) for name in required}
    if schema_type == "array":
        return [example_for_schema(schema.get("items", {}))]
    if schema_type == "integer":
        return 1
    if schema_type == "number":
        return 1.0
    if schema_type == "boolean":
        return True
    if schema.get("enum"):
        return schema["enum"][0]
    return "string"
