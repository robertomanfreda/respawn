import pytest

from src.adapters.automatic1111_image import Automatic1111ImageGenerationBackend
from src.adapters.comfyui_image import ComfyUIImageGenerationBackend
from src.adapters.image_generation_base import ImageGenerationRequest
from src.adapters.mock_image import MockImageGenerationBackend
from src.config import Settings
from src.main import build_image_generation_backend
from src.schemas.errors import OpenAIError
from src.schemas.responses import ResponseRequest
from src.services.image_generation import (
    ImageGenerationService,
    derive_image_generation_prompt,
    validate_image_generation_configuration,
)
from src.services.responses_compat import (
    IMAGE_GENERATION_INTERNAL_TOOL_NAME,
    backend_function_tools,
    image_generation_requested,
    image_generation_required,
    image_generation_tools,
    validate_text_responses_request,
)


def test_image_generation_tool_validation_accepts_supported_fields():
    request = ResponseRequest(
        input="Generate image of a small red square",
        tools=[
            {
                "type": "image_generation",
                "size": "512x512",
                "quality": "low",
                "output_format": "png",
                "background": "opaque",
                "partial_images": 0,
                "action": "generate",
                "moderation": "auto",
            }
        ],
        tool_choice={"type": "image_generation"},
    )

    validate_text_responses_request(request)

    tool = image_generation_tools(request)[0]
    assert tool["type"] == "image_generation"
    assert tool["width"] == 512
    assert tool["height"] == 512
    assert image_generation_requested(request) is True
    assert image_generation_required(request) is True


def test_image_generation_call_input_item_is_accepted_as_protocol_history():
    request = ResponseRequest(
        input=[
            {
                "type": "image_generation_call",
                "status": "completed",
                "result": "aW1hZ2U=",
                "revised_prompt": "un rospo",
                "size": "512x512",
                "quality": "low",
                "output_format": "png",
            },
            {"role": "user", "content": "ora descrivila"},
        ]
    )

    validate_text_responses_request(request)


@pytest.mark.parametrize(
    ("tool", "param"),
    [
        ({"type": "image_generation", "unknown": True}, "tools.0.unknown"),
        ({"type": "image_generation", "size": "giant"}, "tools.0.size"),
        ({"type": "image_generation", "quality": "ultra"}, "tools.0.quality"),
        ({"type": "image_generation", "output_format": "jpeg"}, "tools.0.output_format"),
        ({"type": "image_generation", "background": "transparent"}, "tools.0.background"),
        ({"type": "image_generation", "partial_images": 1}, "tools.0.partial_images"),
        ({"type": "image_generation", "action": "edit"}, "tools.0.action"),
        ({"type": "image_generation", "compression": 80}, "tools.0.compression"),
    ],
)
def test_image_generation_tool_validation_rejects_invalid_shapes(tool, param):
    request = ResponseRequest(input="Generate image", tools=[tool])

    with pytest.raises(OpenAIError) as exc:
        validate_text_responses_request(request)

    assert exc.value.param == param


def test_image_generation_configuration_validates_disabled_and_pixel_limit():
    request = ResponseRequest(input="Generate image", tools=[{"type": "image_generation"}])

    with pytest.raises(OpenAIError) as disabled:
        validate_image_generation_configuration(request, settings=Settings(image_generation_enabled=False), backend=None)
    assert disabled.value.code == "unsupported_parameter"

    too_large = ResponseRequest(input="Generate image", tools=[{"type": "image_generation", "size": "2048x2048"}])
    with pytest.raises(OpenAIError) as oversized:
        validate_image_generation_configuration(
            too_large,
            settings=Settings(image_generation_enabled=True, image_generation_max_pixels=512 * 512),
            backend=MockImageGenerationBackend(),
        )
    assert oversized.value.param == "tools.0.size"


def test_image_generation_prompt_routing_is_deterministic():
    assert derive_image_generation_prompt([{"role": "user", "content": [{"type": "input_text", "text": "  Disegna   un cubo rosso  "}]}]) == "Disegna un cubo rosso"
    input_value = [
        {
            "type": "image_generation_call",
            "status": "completed",
            "revised_prompt": "genera una immagine di un rospo",
            "size": "512x512",
        },
        {"role": "user", "content": "ahahah ma questo non è un rospo!! Un rospo \"a frog\""},
    ]

    prompt = derive_image_generation_prompt(input_value)

    assert prompt == 'ahahah ma questo non è un rospo!! Un rospo "a frog"'


def test_image_generation_auto_is_exposed_as_internal_backend_function():
    request = ResponseRequest(input="Generate image of a tiny house", tools=[{"type": "image_generation"}])

    tools = backend_function_tools(request)

    assert tools[-1]["function"]["name"] == IMAGE_GENERATION_INTERNAL_TOOL_NAME

    forced = ResponseRequest(input="Generate image of a tiny house", tools=[{"type": "image_generation"}], tool_choice={"type": "image_generation"})

    assert backend_function_tools(forced) == []


def test_image_generation_auto_remains_available_after_prior_image_context():
    request = ResponseRequest(
        input=[
            {"type": "image_generation_call", "status": "completed", "revised_prompt": "a dog", "size": "512x512"},
            {"role": "user", "content": "migliora l'immagine del cane"},
        ],
        tools=[{"type": "image_generation"}],
    )

    assert any(tool["function"]["name"] == IMAGE_GENERATION_INTERNAL_TOOL_NAME for tool in backend_function_tools(request))


@pytest.mark.asyncio
async def test_image_generation_service_generates_only_when_needed():
    backend = MockImageGenerationBackend()
    service = ImageGenerationService(settings=Settings(image_generation_enabled=True, image_generation_backend="mock"), backend=backend)

    auto_request = ResponseRequest(input="Generate image of a tiny house", tools=[{"type": "image_generation", "quality": "low"}])
    assert await service.execute_if_needed(auto_request, response_id="resp_test") is None
    assert backend.requests == []

    forced_request = ResponseRequest(
        input="Generate image of a tiny house",
        tools=[{"type": "image_generation", "quality": "low"}],
        tool_choice={"type": "image_generation"},
    )
    execution = await service.execute_if_needed(forced_request, response_id="resp_test")

    assert execution is not None
    assert execution.output_item["type"] == "image_generation_call"
    assert execution.output_item["result"]
    assert execution.output_item["size"] == "512x512"
    assert backend.requests[0].steps == 8


@pytest.mark.asyncio
async def test_image_generation_service_executes_model_tool_call_prompt():
    backend = MockImageGenerationBackend()
    service = ImageGenerationService(settings=Settings(image_generation_enabled=True, image_generation_backend="mock"), backend=backend)
    request = ResponseRequest(input="genera una immagine di un rospo", tools=[{"type": "image_generation", "quality": "low"}])
    tool_call = {
        "id": "call_image",
        "type": "function",
        "function": {"name": IMAGE_GENERATION_INTERNAL_TOOL_NAME, "arguments": '{"prompt":"a frog"}'},
    }

    execution = await service.execute_tool_call(tool_call, request=request, response_id="resp_test")

    assert execution is not None
    assert backend.requests[0].prompt == "a frog"


@pytest.mark.asyncio
async def test_automatic1111_adapter_parses_txt2img_response(monkeypatch):
    backend = Automatic1111ImageGenerationBackend(base_url="http://sd-webui:7860", timeout_seconds=1, model="sd-v1-5")

    async def fake_post_json(path, payload):
        assert path == "/sdapi/v1/txt2img"
        assert payload["width"] == 512
        assert payload["height"] == 512
        return {"images": ["data:image/png;base64,aW1hZ2U="], "info": '{"seed": 42}'}

    monkeypatch.setattr(backend, "_post_json", fake_post_json)

    result = await backend.generate(
        ImageGenerationRequest(
            prompt="Generate image of a tiny adapter house",
            size="512x512",
            width=512,
            height=512,
            quality="low",
            output_format="png",
            steps=8,
            cfg_scale=7,
            sampler="Euler a",
        )
    )

    assert result.image_base64 == "aW1hZ2U="
    assert result.seed == 42
    assert result.provider == "automatic1111"


@pytest.mark.asyncio
async def test_comfyui_adapter_queues_workflow_and_fetches_image(monkeypatch):
    backend = ComfyUIImageGenerationBackend(base_url="http://comfyui:8188", timeout_seconds=1, model="sd-v1-5.safetensors", poll_interval_seconds=0.01)
    posted_payloads = []
    history_calls = 0

    async def fake_post_json(path, payload):
        posted_payloads.append((path, payload))
        return {"prompt_id": "prompt_123", "number": 1, "node_errors": {}}

    async def fake_get_json(path, *, unavailable=False):
        nonlocal history_calls
        assert unavailable is False
        assert path == "/history/prompt_123"
        history_calls += 1
        if history_calls == 1:
            return {}
        return {
            "prompt_123": {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {
                    "9": {
                        "images": [
                            {
                                "filename": "respawn_00001_.png",
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    }
                },
            }
        }

    async def fake_get_bytes(path, *, params):
        assert path == "/view"
        assert params == {"filename": "respawn_00001_.png", "subfolder": "", "type": "output"}
        return b"image"

    monkeypatch.setattr(backend, "_post_json", fake_post_json)
    monkeypatch.setattr(backend, "_get_json", fake_get_json)
    monkeypatch.setattr(backend, "_get_bytes", fake_get_bytes)

    result = await backend.generate(
        ImageGenerationRequest(
            prompt="Generate image of a tiny ComfyUI adapter house",
            negative_prompt="blurry",
            size="512x512",
            width=512,
            height=512,
            quality="low",
            output_format="png",
            steps=8,
            cfg_scale=7,
            sampler="Euler a",
            seed=123,
        )
    )

    assert result.image_base64 == "aW1hZ2U="
    assert result.seed == 123
    assert result.provider == "comfyui"
    assert history_calls == 2

    path, payload = posted_payloads[0]
    assert path == "/prompt"
    workflow = payload["prompt"]
    assert workflow["4"]["inputs"]["ckpt_name"] == "sd-v1-5.safetensors"
    assert workflow["3"]["inputs"]["sampler_name"] == "euler_ancestral"
    assert workflow["3"]["inputs"]["steps"] == 8
    assert workflow["5"]["inputs"] == {"width": 512, "height": 512, "batch_size": 1}
    assert workflow["6"]["inputs"]["text"] == "Generate image of a tiny ComfyUI adapter house"
    assert workflow["7"]["inputs"]["text"] == "blurry"


@pytest.mark.asyncio
async def test_comfyui_adapter_ready_reports_checkpoint(monkeypatch):
    backend = ComfyUIImageGenerationBackend(base_url="http://comfyui:8188", timeout_seconds=1, model="sd-v1-5.safetensors")

    async def fake_get_json(path, *, unavailable=False):
        assert unavailable is True
        if path == "/system_stats":
            return {"devices": [{"name": "NVIDIA GB10"}]}
        if path == "/models/checkpoints":
            return ["sd-v1-5.safetensors"]
        raise AssertionError(path)

    monkeypatch.setattr(backend, "_get_json", fake_get_json)

    details = await backend.check_ready()

    assert details["backend"] == "comfyui"
    assert details["device_count"] == 1
    assert details["model_count"] == 1
    assert details["configured_model_present"] is True


def test_build_image_generation_backend_accepts_comfyui():
    backend = build_image_generation_backend(
        Settings(
            image_generation_enabled=True,
            image_generation_backend="comfyui",
            image_generation_base_url="http://comfyui:8188",
            image_generation_model="sd-v1-5.safetensors",
        )
    )

    assert isinstance(backend, ComfyUIImageGenerationBackend)
