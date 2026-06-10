from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from src.schemas.models import ModelList


class ChatCompletionResult(BaseModel):
    """Normalized chat-completions output consumed by the response service."""

    content: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    output_items: list[dict[str, Any]] = Field(default_factory=list)
    content_logprobs: list[dict[str, Any]] = Field(default_factory=list)
    unhandled_tool_calls: bool = False


class ModelBackend(ABC):
    """Minimal contract for local OpenAI-compatible model servers."""

    @abstractmethod
    async def list_models(self) -> ModelList:
        raise NotImplementedError

    @abstractmethod
    async def create_chat_completion(self, payload: dict[str, Any]) -> ChatCompletionResult:
        raise NotImplementedError

    @abstractmethod
    async def create_chat_completion_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError
