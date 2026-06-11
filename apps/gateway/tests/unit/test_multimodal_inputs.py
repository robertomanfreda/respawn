import base64

import pytest

from src.config import Settings
from src.schemas.errors import OpenAIError
from src.schemas.responses import ResponseRequest
from src.services.multimodal_inputs import model_capabilities, prepare_multimodal_request


def _data_url(mime_type: str, payload_base64: str) -> str:
    return f"data:{mime_type};base64,{payload_base64}"


def test_model_capabilities_parses_configured_map():
    settings = Settings(model_capabilities="text-model=text,file-text;vision-model=text,vision")

    assert model_capabilities(settings) == {
        "text-model": {"text", "file-text"},
        "vision-model": {"text", "vision"},
    }


@pytest.mark.asyncio
async def test_prepare_multimodal_request_rejects_image_without_vision_capability():
    settings = Settings(model_capabilities="text-model=text,file-text")
    request = ResponseRequest.model_validate(
        {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": _data_url("image/png", "iVBORw0KGgo=")}]}]}
    )

    with pytest.raises(OpenAIError) as exc:
        await prepare_multimodal_request(request, model="text-model", settings=settings)

    assert exc.value.param == "model"
    assert exc.value.code == "unsupported_model_capability"


@pytest.mark.asyncio
async def test_prepare_multimodal_request_extracts_text_file_data():
    encoded = base64.b64encode(b"marker word is cobalt").decode("ascii")
    request = ResponseRequest.model_validate(
        {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_file", "filename": "facts.txt", "file_data": _data_url("text/plain", encoded)}],
                }
            ]
        }
    )

    prepared = await prepare_multimodal_request(request, model="gpt-oss:120b", settings=Settings())

    part = prepared.input[0]["content"][0]
    assert part["type"] == "input_file"
    assert part["filename"] == "facts.txt"
    assert part["text"] == "marker word is cobalt"
    assert part["mime_type"] == "text/plain"


@pytest.mark.parametrize(
    ("filename", "mime_type", "payload"),
    [
        ("note.md", "text/markdown", "# Marker\ncobalt"),
        ("data.json", "application/json", '{"marker":"cobalt"}'),
        ("page.html", "text/html", "<main>marker cobalt</main>"),
        ("snippet.py", "text/x-python", "marker = 'cobalt'"),
    ],
)
@pytest.mark.asyncio
async def test_prepare_multimodal_request_extracts_common_text_file_types(filename, mime_type, payload):
    encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    request = ResponseRequest.model_validate(
        {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_file", "filename": filename, "file_data": _data_url(mime_type, encoded)}],
                }
            ]
        }
    )

    prepared = await prepare_multimodal_request(request, model="gpt-oss:120b", settings=Settings())

    assert "cobalt" in prepared.input[0]["content"][0]["text"]


@pytest.mark.asyncio
async def test_prepare_multimodal_request_rejects_file_id_until_files_api_exists():
    request = ResponseRequest.model_validate({"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "file_123"}]}]})

    with pytest.raises(OpenAIError) as exc:
        await prepare_multimodal_request(request, model="gpt-oss:120b", settings=Settings())

    assert exc.value.param == "input.0.content.0.file_id"
    assert exc.value.code == "unsupported_parameter"


@pytest.mark.asyncio
async def test_prepare_multimodal_request_enforces_file_size_limit():
    encoded = base64.b64encode(b"larger than limit").decode("ascii")
    request = ResponseRequest.model_validate(
        {"input": [{"role": "user", "content": [{"type": "input_file", "filename": "facts.txt", "file_data": _data_url("text/plain", encoded)}]}]}
    )

    with pytest.raises(OpenAIError) as exc:
        await prepare_multimodal_request(request, model="gpt-oss:120b", settings=Settings(multimodal_max_file_bytes=3))

    assert exc.value.code == "file_too_large"
