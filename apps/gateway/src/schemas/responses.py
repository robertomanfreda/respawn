from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


InputContent = str | dict[str, Any]


class ResponseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    input: str | list[dict[str, Any]] | None = None
    background: bool = False
    include: list[str] = Field(default_factory=list)
    instructions: str | None = None
    previous_response_id: str | None = None
    store: bool | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    max_tool_calls: int | None = Field(default=None, ge=0)
    parallel_tool_calls: bool | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = "auto"
    response_format: dict[str, Any] | None = None
    text: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    truncation: Literal["auto", "disabled"] = "disabled"
    prompt: dict[str, Any] | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    context_management: dict[str, Any] | None = None
    service_tier: str | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    safety_identifier: str | None = None
    user: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResponseInputTokensDetails(BaseModel):
    cached_tokens: int = 0


class ResponseOutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class ResponseUsage(BaseModel):
    input_tokens: int = 0
    input_tokens_details: ResponseInputTokensDetails = Field(default_factory=ResponseInputTokensDetails)
    output_tokens: int = 0
    output_tokens_details: ResponseOutputTokensDetails = Field(default_factory=ResponseOutputTokensDetails)
    total_tokens: int = 0


class ResponseOutputContent(BaseModel):
    type: str = "output_text"
    text: str
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    logprobs: list[dict[str, Any]] = Field(default_factory=list)


class ResponseOutputItem(BaseModel):
    id: str
    type: str = "message"
    status: str = "completed"
    role: str = "assistant"
    content: list[dict[str, Any]]


class ResponseObject(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    status: str
    error: dict[str, Any] | None = None
    incomplete_details: dict[str, Any] | None = None
    input: str | list[dict[str, Any]] | None = None
    background: bool = False
    instructions: str | list[dict[str, Any]] | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    model: str
    output: list[dict[str, Any]]
    output_text: str = ""
    parallel_tool_calls: bool = False
    previous_response_id: str | None = None
    prompt: dict[str, Any] | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    reasoning: dict[str, Any] = Field(default_factory=lambda: {"effort": None, "summary": None})
    safety_identifier: str | None = None
    service_tier: str | None = "auto"
    store: bool = True
    temperature: float | None = 1
    text: dict[str, Any] = Field(default_factory=lambda: {"format": {"type": "text"}})
    tool_choice: str | dict[str, Any] | None = "auto"
    tools: list[dict[str, Any]] = Field(default_factory=list)
    top_logprobs: int | None = None
    top_p: float | None = 1
    truncation: Literal["auto", "disabled"] = "disabled"
    usage: ResponseUsage = Field(default_factory=ResponseUsage)
    user: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResponseDeleted(BaseModel):
    id: str
    object: Literal["response.deleted"] = "response.deleted"
    deleted: bool = True


class ResponseInputItemList(BaseModel):
    object: Literal["list"] = "list"
    data: list[dict[str, Any]]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


class ResponseInputTokenCount(BaseModel):
    object: Literal["response.input_tokens"] = "response.input_tokens"
    input_tokens: int
    input_tokens_details: ResponseInputTokensDetails = Field(default_factory=ResponseInputTokensDetails)
