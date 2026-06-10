from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field


class WebSearchRequest(BaseModel):
    query: str
    allowed_domains: list[str] = Field(default_factory=list)
    blocked_domains: list[str] = Field(default_factory=list)
    search_context_size: Literal["low", "medium", "high"] = "medium"
    external_web_access: bool = True
    user_location: dict[str, Any] | None = None
    max_results: int = 5
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebSearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    source: str = "web"
    published_at: str | None = None


class WebSearchRun(BaseModel):
    id: str
    query: str
    results: list[WebSearchResult]
    latency_ms: float
    provider: str


class WebSearchError(Exception):
    def __init__(self, message: str, *, status_code: int, code: str, type: str = "server_error") -> None:
        self.message = message
        self.status_code = status_code
        self.code = code
        self.type = type
        super().__init__(message)


class WebSearchTimeoutError(WebSearchError):
    def __init__(self, message: str = "Web search provider timed out.") -> None:
        super().__init__(message, status_code=504, code="web_search_timeout")


class WebSearchBackendError(WebSearchError):
    def __init__(self, message: str = "Web search provider returned an error.") -> None:
        super().__init__(message, status_code=502, code="web_search_backend_error")


class WebSearchUnavailableError(WebSearchError):
    def __init__(self, message: str = "Web search provider is unavailable.") -> None:
        super().__init__(message, status_code=503, code="web_search_unavailable")


class WebSearchBackend(ABC):
    provider_name: str

    @abstractmethod
    async def search(self, request: WebSearchRequest) -> WebSearchRun:
        raise NotImplementedError

    @abstractmethod
    async def check_ready(self) -> dict[str, Any]:
        raise NotImplementedError
