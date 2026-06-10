from time import perf_counter
from typing import Any

import httpx

from src.adapters.web_search_base import WebSearchBackend, WebSearchBackendError, WebSearchRequest, WebSearchResult, WebSearchRun, WebSearchTimeoutError, WebSearchUnavailableError
from src.services.id_generator import generate_id


class SearxngSearchBackend(WebSearchBackend):
    provider_name = "searxng"

    def __init__(self, *, base_url: str, timeout_seconds: float, user_agent: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = max(float(timeout_seconds), 0.1)
        self.user_agent = user_agent

    async def search(self, request: WebSearchRequest) -> WebSearchRun:
        started_at = perf_counter()
        payload = await self._get_json(
            "/search",
            params={
                "q": request.query,
                "format": "json",
            },
        )
        results = self._results_from_payload(payload)
        return WebSearchRun(
            id=generate_id("wsr"),
            query=request.query,
            results=results[: max(request.max_results, 0)],
            latency_ms=(perf_counter() - started_at) * 1000,
            provider=self.provider_name,
        )

    async def check_ready(self) -> dict[str, Any]:
        payload = await self._get_json("/search", params={"q": "respawn readiness", "format": "json"})
        results = payload.get("results") if isinstance(payload, dict) else None
        return {"backend": self.provider_name, "base_url": self.base_url, "result_count": len(results or [])}

    async def _get_json(self, path: str, *, params: dict[str, str]) -> dict[str, Any]:
        if not self.base_url:
            raise WebSearchUnavailableError("WEB_SEARCH_BASE_URL is required for the SearXNG backend.")
        url = f"{self.base_url}{path}"
        headers = {"User-Agent": self.user_agent}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, headers=headers) as client:
                response = await client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as exc:
            raise WebSearchTimeoutError() from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise WebSearchBackendError() from exc
        if not isinstance(payload, dict):
            raise WebSearchBackendError("Web search provider returned a malformed response.")
        return payload

    def _results_from_payload(self, payload: dict[str, Any]) -> list[WebSearchResult]:
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise WebSearchBackendError("Web search provider returned a malformed response.")
        results: list[WebSearchResult] = []
        for raw in raw_results:
            if not isinstance(raw, dict):
                continue
            url = str(raw.get("url") or "").strip()
            if not url:
                continue
            title = str(raw.get("title") or url).strip()
            snippet = str(raw.get("content") or raw.get("snippet") or "").strip()
            source = str(raw.get("engine") or raw.get("source") or self.provider_name).strip() or self.provider_name
            published_at = raw.get("publishedDate") or raw.get("published_at")
            results.append(
                WebSearchResult(
                    url=url,
                    title=title,
                    snippet=snippet,
                    source=source,
                    published_at=str(published_at) if published_at else None,
                )
            )
        return results
