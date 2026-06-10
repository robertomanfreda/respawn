from time import perf_counter

from src.adapters.web_search_base import WebSearchBackend, WebSearchBackendError, WebSearchRequest, WebSearchResult, WebSearchRun, WebSearchTimeoutError
from src.adapters.mock_control import mock_options
from src.services.id_generator import generate_id


class MockSearchBackend(WebSearchBackend):
    provider_name = "mock"

    def __init__(self) -> None:
        self.requests: list[WebSearchRequest] = []

    async def search(self, request: WebSearchRequest) -> WebSearchRun:
        self.requests.append(request)
        started_at = perf_counter()
        query = request.query.strip()
        options = mock_options(request.metadata)
        if options.get("web_search_error") == "timeout":
            raise WebSearchTimeoutError()
        if options.get("web_search_error") == "backend":
            raise WebSearchBackendError()

        results = [
            WebSearchResult(
                url="https://example.com/respawn-web-search",
                title="Respawn Web Search Fixture",
                snippet=f"Deterministic mock search result for: {query}",
                source=self.provider_name,
            ),
            WebSearchResult(
                url="https://news.example.org/current",
                title="Current Fixture",
                snippet=f"Recent mock information about: {query}",
                source=self.provider_name,
            ),
            WebSearchResult(
                url="https://blocked.example.net/private",
                title="Blocked Fixture",
                snippet="This result exists so domain filters can exclude it deterministically.",
                source=self.provider_name,
            ),
        ]
        return WebSearchRun(
            id=generate_id("wsr"),
            query=query,
            results=results[: max(request.max_results, 0)],
            latency_ms=(perf_counter() - started_at) * 1000,
            provider=self.provider_name,
        )

    async def check_ready(self) -> dict[str, str | int]:
        return {"backend": self.provider_name, "fixture_count": 3}
