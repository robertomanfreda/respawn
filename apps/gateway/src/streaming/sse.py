from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse


def sse_response(events: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(events, media_type="text/event-stream")
