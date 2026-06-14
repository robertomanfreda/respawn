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
    schema_text = json.dumps(schema, separators=(",", ":"))
    return (
        "The previous assistant output was not valid for the requested JSON schema.\n\n"
        "Repair rules:\n"
        "- Your task is schema repair only, not task re-execution.\n"
        "- Preserve the original output's semantic meaning.\n"
        "- Preserve the same top-level JSON shape whenever possible.\n"
        "- Preserve existing object identities, ids, labels, names, and list item order whenever possible.\n"
        "- Do not add new objects, array items, facts, labels, names, or user-facing semantic content.\n"
        "- Do not remove existing objects or array items unless they cannot be represented in the requested schema.\n"
        "- Do not reinterpret the original user prompt.\n"
        "- Do not improve, enrich, summarize, complete, or extend the previous assistant output.\n"
        "- Only change fields that are invalid according to the schema: missing required fields, wrong types, malformed values, enum mismatches, "
        "out-of-range numbers, additional forbidden properties, or invalid nesting.\n"
        "- When a value can be repaired mechanically, repair it mechanically instead of regenerating the content.\n"
        "- For numeric fields constrained to 0.0..1.0, if the previous output clearly used a 0-10 score, convert it mechanically to 0.0..1.0, "
        "for example 9 -> 0.9 and 6 -> 0.6.\n"
        "- If a field is missing and cannot be inferred directly from the existing output without adding new semantic content, use the minimal "
        "schema-valid neutral value where possible.\n"
        "- Return only valid JSON with no markdown or commentary.\n\n"
        f"JSON schema: {schema_text}"
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
