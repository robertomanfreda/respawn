import pytest

from src.schemas.errors import OpenAIError
from src.tools.registry import default_registry


def test_openai_error_shape():
    error = OpenAIError("missing", status_code=404, param="previous_response_id", code="not_found")
    assert error.to_response()["error"]["code"] == "not_found"


@pytest.mark.asyncio
async def test_tool_argument_validation():
    registry = default_registry()
    with pytest.raises(OpenAIError):
        await registry.execute("calculator", '{"expression": "__import__(\\"os\\")"}')


@pytest.mark.asyncio
async def test_calculator_tool():
    registry = default_registry()
    assert await registry.execute("calculator", '{"expression": "2+2*3"}') == 8
