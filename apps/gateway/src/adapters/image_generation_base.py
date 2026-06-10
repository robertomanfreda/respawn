from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ImageGenerationRequest(BaseModel):
    prompt: str
    size: str
    width: int
    height: int
    quality: str
    output_format: str = "png"
    negative_prompt: str = ""
    steps: int
    cfg_scale: float
    sampler: str
    seed: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageGenerationResult(BaseModel):
    id: str
    prompt: str
    revised_prompt: str
    image_base64: str
    output_format: str
    size: str
    quality: str
    seed: int | None = None
    provider: str
    latency_ms: float


class ImageGenerationError(Exception):
    def __init__(self, message: str, *, status_code: int, code: str, type: str = "server_error") -> None:
        self.message = message
        self.status_code = status_code
        self.code = code
        self.type = type
        super().__init__(message)


class ImageGenerationTimeoutError(ImageGenerationError):
    def __init__(self, message: str = "Image generation provider timed out.") -> None:
        super().__init__(message, status_code=504, code="image_generation_timeout")


class ImageGenerationBackendError(ImageGenerationError):
    def __init__(self, message: str = "Image generation provider returned an error.") -> None:
        super().__init__(message, status_code=502, code="image_generation_backend_error")


class ImageGenerationUnavailableError(ImageGenerationError):
    def __init__(self, message: str = "Image generation provider is unavailable.") -> None:
        super().__init__(message, status_code=503, code="image_generation_unavailable")


class ImageGenerationBackend(ABC):
    provider_name: str

    @abstractmethod
    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        raise NotImplementedError

    @abstractmethod
    async def check_ready(self) -> dict[str, Any]:
        raise NotImplementedError
