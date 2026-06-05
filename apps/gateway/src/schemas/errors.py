from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorBody(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class OpenAIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.type = type
        self.param = param
        self.code = code
        super().__init__(message)

    def to_response(self) -> dict[str, Any]:
        return ErrorResponse(
            error=ErrorBody(
                message=self.message,
                type=self.type,
                param=self.param,
                code=self.code,
            )
        ).model_dump()


async def openai_error_handler(_: Request, exc: OpenAIError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_response())


async def request_validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    first_error = exc.errors()[0] if exc.errors() else {}
    loc = [str(part) for part in first_error.get("loc", []) if part not in {"body", "query", "path"}]
    param = ".".join(loc) or None
    message = first_error.get("msg") or "Request validation failed."
    error = OpenAIError(
        message,
        status_code=422,
        type="invalid_request_error",
        param=param,
        code="validation_error",
    )
    return JSONResponse(status_code=error.status_code, content=error.to_response())
