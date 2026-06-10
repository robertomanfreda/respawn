from time import perf_counter

from src.adapters.image_generation_base import ImageGenerationBackend, ImageGenerationBackendError, ImageGenerationRequest, ImageGenerationResult, ImageGenerationTimeoutError
from src.adapters.mock_control import mock_options
from src.services.id_generator import generate_id


TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
)


class MockImageGenerationBackend(ImageGenerationBackend):
    provider_name = "mock"

    def __init__(self) -> None:
        self.requests: list[ImageGenerationRequest] = []

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        self.requests.append(request)
        started_at = perf_counter()
        options = mock_options(request.metadata)
        if options.get("image_generation_error") == "timeout":
            raise ImageGenerationTimeoutError()
        if options.get("image_generation_error") == "backend":
            raise ImageGenerationBackendError()
        return ImageGenerationResult(
            id=generate_id("igr"),
            prompt=request.prompt,
            revised_prompt=request.prompt,
            image_base64=TINY_PNG_BASE64,
            output_format=request.output_format,
            size=request.size,
            quality=request.quality,
            seed=request.seed,
            provider=self.provider_name,
            latency_ms=(perf_counter() - started_at) * 1000,
        )

    async def check_ready(self) -> dict[str, str | int]:
        return {"backend": self.provider_name, "fixture_count": 1}
