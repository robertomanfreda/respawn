from __future__ import annotations

from copy import deepcopy
from typing import Any
import re

from src.observability.metrics import PROMPT_TEMPLATE_REQUESTS
from src.schemas.errors import OpenAIError
from src.schemas.prompts import PromptTemplateCreate
from src.schemas.responses import ResponseRequest
from src.services.id_generator import generate_id
from src.services.responses_compat import validate_text_input
from src.storage.repository import ResponseRepository


PLACEHOLDER_PATTERN = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*}}")
EXACT_PLACEHOLDER_PATTERN = re.compile(r"^\s*{{\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*}}\s*$")
PROMPT_TEMPLATE_FIELDS = {"instructions", "input"}
PROMPT_REFERENCE_FIELDS = {"id", "variables", "version"}


class PromptTemplateRenderer:
    def __init__(self, repository: ResponseRepository) -> None:
        self.repository = repository

    async def render_request(self, request: ResponseRequest, tenant_id: str | None) -> ResponseRequest:
        prompt = request.prompt
        if prompt is None:
            return request
        reference = validate_prompt_reference(prompt)
        prompt_id = reference["id"]
        version = reference.get("version")
        variables = reference.get("variables") or {}
        try:
            record = await self.repository.require_prompt_template(prompt_id, tenant_id, version=version)
        except OpenAIError as exc:
            status = "missing" if exc.status_code == 404 else "failed"
            PROMPT_TEMPLATE_REQUESTS.labels(operation="render", status=status).inc()
            raise
        try:
            rendered = render_template(record.template_json or {}, variables)
        except OpenAIError:
            PROMPT_TEMPLATE_REQUESTS.labels(operation="render", status="failed").inc()
            raise

        rendered_request = request.model_copy(
            update={
                "instructions": _merge_instructions(rendered.get("instructions"), request.instructions),
                "input": _merge_input(rendered.get("input"), request.input),
                "prompt": {**prompt, "id": record.prompt_id, "version": record.version},
            }
        )
        PROMPT_TEMPLATE_REQUESTS.labels(operation="render", status="success").inc()
        return rendered_request


def prompt_template_from_create(payload: PromptTemplateCreate) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    prompt_id = validate_prompt_id(payload.id or generate_id("pmpt"), param="id")
    version = validate_prompt_version(payload.version or "1", param="version")
    template = normalize_prompt_template(payload)
    return prompt_id, version, template, dict(payload.metadata or {})


def normalize_prompt_template(payload: PromptTemplateCreate) -> dict[str, Any]:
    template = dict(payload.template or {})
    if payload.instructions is not None:
        if "instructions" in template:
            raise OpenAIError("instructions must be provided either at top level or inside template, not both.", param="instructions", code="invalid_request")
        template["instructions"] = payload.instructions
    if payload.input is not None:
        if "input" in template:
            raise OpenAIError("input must be provided either at top level or inside template, not both.", param="input", code="invalid_request")
        template["input"] = payload.input

    if not template:
        raise OpenAIError("Prompt template requires instructions or input.", param="template", code="invalid_request")
    unsupported = sorted(set(template) - PROMPT_TEMPLATE_FIELDS)
    if unsupported:
        raise OpenAIError(f"Prompt template field '{unsupported[0]}' is not supported.", status_code=400, param=f"template.{unsupported[0]}", code="unsupported_parameter")

    instructions = template.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise OpenAIError("template.instructions must be a string.", param="template.instructions", code="invalid_request")
    validate_text_input(template.get("input"), param="template.input")
    return {key: deepcopy(value) for key, value in template.items() if value is not None}


def validate_prompt_reference(prompt: dict[str, Any]) -> dict[str, Any]:
    unsupported = sorted(set(prompt) - PROMPT_REFERENCE_FIELDS)
    if unsupported:
        raise OpenAIError(f"prompt field '{unsupported[0]}' is not supported.", status_code=400, param=f"prompt.{unsupported[0]}", code="unsupported_parameter")
    prompt_id = validate_prompt_id(prompt.get("id"), param="prompt.id")
    version = prompt.get("version")
    if version is not None:
        version = validate_prompt_version(version, param="prompt.version")
    variables = prompt.get("variables") or {}
    if not isinstance(variables, dict):
        raise OpenAIError("prompt.variables must be an object.", param="prompt.variables", code="invalid_request")
    return {"id": prompt_id, "version": version, "variables": variables}


def validate_prompt_id(value: Any, *, param: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OpenAIError("Prompt template id must be a non-empty string.", param=param, code="invalid_request")
    return value.strip()


def validate_prompt_version(value: Any, *, param: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OpenAIError("Prompt template version must be a non-empty string.", param=param, code="invalid_request")
    return value.strip()


def render_template(template: dict[str, Any], variables: dict[str, Any]) -> dict[str, Any]:
    placeholders = _placeholders(template)
    missing = sorted(placeholders - set(variables))
    if missing:
        raise OpenAIError(f"Missing prompt variable '{missing[0]}'.", param=f"prompt.variables.{missing[0]}", code="missing_prompt_variable")
    rendered = _render_value(template, variables)
    validate_text_input(rendered.get("input"), param="prompt.template.input")
    instructions = rendered.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise OpenAIError("Rendered prompt instructions must be a string.", param="prompt.template.instructions", code="invalid_prompt_template")
    return rendered


def _render_value(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, str):
        exact = EXACT_PLACEHOLDER_PATTERN.match(value)
        if exact:
            return deepcopy(variables[exact.group(1)])
        return PLACEHOLDER_PATTERN.sub(lambda match: _variable_text(variables[match.group(1)], match.group(1)), value)
    if isinstance(value, list):
        return [_render_value(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _render_value(item, variables) for key, item in value.items()}
    return value


def _variable_text(value: Any, name: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and value.get("type") in {"input_text", "output_text", "text"} and isinstance(value.get("text"), str):
        return value["text"]
    raise OpenAIError(
        f"Prompt variable '{name}' must occupy a whole template value when it is not text.",
        param=f"prompt.variables.{name}",
        code="invalid_prompt_variable",
    )


def _placeholders(value: Any) -> set[str]:
    if isinstance(value, str):
        return set(PLACEHOLDER_PATTERN.findall(value))
    if isinstance(value, list):
        found: set[str] = set()
        for item in value:
            found.update(_placeholders(item))
        return found
    if isinstance(value, dict):
        found: set[str] = set()
        for item in value.values():
            found.update(_placeholders(item))
        return found
    return set()


def _merge_instructions(template_instructions: Any, request_instructions: str | None) -> str | None:
    if template_instructions is None:
        return request_instructions
    if request_instructions is None:
        return str(template_instructions)
    return f"{template_instructions}\n\n{request_instructions}"


def _merge_input(template_input: Any, request_input: str | list[dict[str, Any]] | None) -> str | list[dict[str, Any]] | None:
    if template_input is None:
        return deepcopy(request_input)
    if request_input is None:
        return deepcopy(template_input)
    if isinstance(template_input, str) and isinstance(request_input, str):
        return f"{template_input}\n{request_input}"
    return [*_input_items(template_input), *_input_items(request_input)]


def _input_items(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, list):
        return deepcopy(value)
    raise OpenAIError("Rendered prompt input must be a string or list of input items.", param="prompt.template.input", code="invalid_prompt_template")
