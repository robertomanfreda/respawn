from __future__ import annotations

import logging
import json
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

from src.adapters.web_search_base import WebSearchBackend, WebSearchError, WebSearchRequest, WebSearchResult, WebSearchRun
from src.config import Settings
from src.observability.metrics import WEB_SEARCH_ERRORS, WEB_SEARCH_FILTERED_RESULTS, WEB_SEARCH_LATENCY, WEB_SEARCH_REQUESTS, WEB_SEARCH_RESULTS
from src.schemas.errors import OpenAIError
from src.schemas.responses import ResponseRequest
from src.services.id_generator import generate_id
from src.services.response_history_builder import content_to_text
from src.services.responses_compat import web_search_disabled_by_choice, web_search_requested, web_search_required, web_search_tools


logger = logging.getLogger(__name__)

CONTEXT_SIZE_RESULTS = {"low": 3, "medium": 5, "high": 10}
MAX_QUERY_CHARS = 256
MAX_SNIPPET_CHARS = 1000


@dataclass(frozen=True)
class WebSearchExecution:
    run: WebSearchRun
    output_item: dict[str, Any]
    context: str


def validate_web_search_configuration(request: ResponseRequest, *, settings: Settings, backend: WebSearchBackend | None) -> None:
    tools = web_search_tools(request)
    if not tools:
        return
    first_param = str(tools[0].get("_param") or "tools.0")
    if not settings.web_search_enabled:
        raise OpenAIError(
            "The web_search tool is disabled. Set WEB_SEARCH_ENABLED=true and configure WEB_SEARCH_BACKEND to enable local web search.",
            status_code=400,
            param=f"{first_param}.type",
            code="unsupported_parameter",
        )
    if backend is None:
        raise OpenAIError(
            "The web_search tool is enabled but no web search backend is configured.",
            status_code=503,
            type="server_error",
            param=f"{first_param}.type",
            code="web_search_unavailable",
        )
    for tool in tools:
        if tool.get("external_web_access") is False:
            param = str(tool.get("_param") or first_param)
            raise OpenAIError(
                "external_web_access=false requires a cache-only web search provider, which is not implemented yet.",
                status_code=400,
                param=f"{param}.external_web_access",
                code="unsupported_parameter",
            )


class WebSearchService:
    def __init__(self, *, settings: Settings, backend: WebSearchBackend | None) -> None:
        self.settings = settings
        self.backend = backend

    async def execute_if_needed(self, request: ResponseRequest, *, response_id: str) -> WebSearchExecution | None:
        if not web_search_requested(request) or web_search_disabled_by_choice(request):
            return None
        tools = web_search_tools(request)
        if not tools:
            return None
        validate_web_search_configuration(request, settings=self.settings, backend=self.backend)
        if not web_search_required(request):
            return None

        tool = tools[0]
        query = derive_web_search_query(request.input)
        if not query:
            raise OpenAIError(
                "web_search requires non-empty user text to derive a search query.",
                param=f"{tool.get('_param', 'tools.0')}.type",
                code="invalid_request",
            )

        return await self.execute_query(query, request=request, response_id=response_id)

    async def execute_tool_call(self, tool_call: dict[str, Any], request: ResponseRequest, *, response_id: str) -> WebSearchExecution:
        tools = web_search_tools(request)
        if not tools:
            raise OpenAIError("web_search tool call requires a web_search tool in the request.", param="tools", code="invalid_tool_call")
        validate_web_search_configuration(request, settings=self.settings, backend=self.backend)
        query = _query_from_tool_call(tool_call)
        if not query:
            raise OpenAIError(
                "web_search tool call requires a non-empty query argument.",
                param="tools.0.type",
                code="invalid_tool_call",
            )
        return await self.execute_query(query, request=request, response_id=response_id)

    async def execute_query(self, query: str, request: ResponseRequest, *, response_id: str) -> WebSearchExecution:
        tools = web_search_tools(request)
        if not tools:
            raise OpenAIError("web_search requires a web_search tool in the request.", param="tools", code="invalid_request")
        validate_web_search_configuration(request, settings=self.settings, backend=self.backend)
        tool = tools[0]
        query = _normalize_query(query)
        if not query:
            raise OpenAIError(
                "web_search requires a non-empty search query.",
                param=f"{tool.get('_param', 'tools.0')}.type",
                code="invalid_request",
            )
        provider = self.backend.provider_name if self.backend is not None else "unconfigured"
        started_at = perf_counter()
        try:
            request_allowed = list(tool.get("allowed_domains") or [])
            request_blocked = list(tool.get("blocked_domains") or [])
            operator_allowed = _settings_domains(self.settings.web_search_allowed_domains, name="WEB_SEARCH_ALLOWED_DOMAINS")
            operator_blocked = _settings_domains(self.settings.web_search_blocked_domains, name="WEB_SEARCH_BLOCKED_DOMAINS")
            max_results = self._max_results(str(tool.get("search_context_size") or "medium"))
            search_request = WebSearchRequest(
                query=query,
                allowed_domains=sorted(set(request_allowed + operator_allowed)),
                blocked_domains=sorted(set(request_blocked + operator_blocked)),
                search_context_size=str(tool.get("search_context_size") or "medium"),  # type: ignore[arg-type]
                external_web_access=bool(tool.get("external_web_access", True)),
                user_location=tool.get("user_location") if isinstance(tool.get("user_location"), dict) else None,
                max_results=max(max_results, 1),
                metadata=dict(request.metadata),
            )
            assert self.backend is not None
            run = await self.backend.search(search_request)
            results = self._filter_results(run.results, request_allowed=request_allowed, operator_allowed=operator_allowed, blocked_domains=[*request_blocked, *operator_blocked])
            results = results[:max_results]
            run = run.model_copy(update={"results": results, "provider": provider})
            WEB_SEARCH_REQUESTS.labels(provider=provider, status="completed").inc()
            WEB_SEARCH_RESULTS.labels(provider=provider).inc(len(results))
            WEB_SEARCH_LATENCY.labels(provider=provider).observe(perf_counter() - started_at)
            self._log_run(response_id=response_id, provider=provider, status="completed", result_count=len(results), latency_ms=run.latency_ms)
            return WebSearchExecution(
                run=run,
                output_item=self._output_item(run, domains=search_request.allowed_domains),
                context=web_search_context(query, results, max_chars=self.settings.web_search_max_result_chars),
            )
        except WebSearchError as exc:
            WEB_SEARCH_REQUESTS.labels(provider=provider, status="failed").inc()
            WEB_SEARCH_ERRORS.labels(provider=provider, code=exc.code).inc()
            WEB_SEARCH_LATENCY.labels(provider=provider).observe(perf_counter() - started_at)
            self._log_run(response_id=response_id, provider=provider, status="failed", result_count=0, latency_ms=(perf_counter() - started_at) * 1000, code=exc.code)
            raise OpenAIError(
                exc.message,
                status_code=exc.status_code,
                type=exc.type,
                param=f"{tool.get('_param', 'tools.0')}.type",
                code=exc.code,
            ) from exc

    def _max_results(self, search_context_size: str) -> int:
        configured = max(int(self.settings.web_search_max_results), 1)
        desired = CONTEXT_SIZE_RESULTS.get(search_context_size, CONTEXT_SIZE_RESULTS["medium"])
        if search_context_size == "high":
            return min(max(configured, desired), 20)
        return min(configured, desired)

    def _filter_results(
        self,
        results: list[WebSearchResult],
        *,
        request_allowed: list[str],
        operator_allowed: list[str],
        blocked_domains: list[str],
    ) -> list[WebSearchResult]:
        seen_urls: set[str] = set()
        filtered: list[WebSearchResult] = []
        for result in results:
            normalized_url = result.url.strip()
            if not normalized_url or normalized_url in seen_urls:
                WEB_SEARCH_FILTERED_RESULTS.labels(reason="duplicate_url").inc()
                continue
            domain = _domain_from_url(normalized_url)
            if not domain:
                WEB_SEARCH_FILTERED_RESULTS.labels(reason="invalid_url").inc()
                continue
            if request_allowed and not _domain_matches_any(domain, request_allowed):
                WEB_SEARCH_FILTERED_RESULTS.labels(reason="request_allowed_domains").inc()
                continue
            if operator_allowed and not _domain_matches_any(domain, operator_allowed):
                WEB_SEARCH_FILTERED_RESULTS.labels(reason="operator_allowed_domains").inc()
                continue
            if blocked_domains and _domain_matches_any(domain, blocked_domains):
                WEB_SEARCH_FILTERED_RESULTS.labels(reason="blocked_domains").inc()
                continue
            seen_urls.add(normalized_url)
            filtered.append(
                WebSearchResult(
                    url=normalized_url,
                    title=result.title[:300],
                    snippet=result.snippet[:MAX_SNIPPET_CHARS],
                    source=result.source,
                    published_at=result.published_at,
                )
            )
        return filtered

    def _output_item(self, run: WebSearchRun, *, domains: list[str]) -> dict[str, Any]:
        action: dict[str, Any] = {
            "type": "search",
            "queries": [run.query],
            "_respawn_sources": [result.model_dump(exclude_none=True) for result in run.results],
        }
        if domains:
            action["domains"] = domains
        return {
            "id": generate_id("ws"),
            "type": "web_search_call",
            "status": "completed",
            "action": action,
        }

    def _log_run(self, *, response_id: str, provider: str, status: str, result_count: int, latency_ms: float, code: str | None = None) -> None:
        logger.info(
            "Web search completed",
            extra={
                "feature": "web_search",
                "response_id": response_id,
                "web_search_provider": provider,
                "web_search_status": status,
                "web_search_result_count": result_count,
                "web_search_latency_ms": round(latency_ms, 3),
                "web_search_error_code": code,
            },
        )


def derive_web_search_query(input_value: Any) -> str:
    return _normalize_query(_latest_user_text(input_value))


def web_search_context(query: str, results: list[WebSearchResult], *, max_chars: int) -> str:
    header = f"Web search results for: {query}\n\n"
    if not results:
        return header + "No results were returned by the configured web search provider."
    chunks = [header]
    remaining = max(max_chars - len(header), 0)
    for index, result in enumerate(results, start=1):
        chunk = f"[{index}] {result.title}\nURL: {result.url}\nSnippet: {result.snippet}\n\n"
        if len(chunk) > remaining:
            if remaining <= 0:
                break
            chunk = chunk[:remaining]
        chunks.append(chunk)
        remaining -= len(chunk)
        if remaining <= 0:
            break
    chunks.append("Use these results only when relevant. Cite sources with [1], [2], etc.")
    return "".join(chunks)


def url_citation_annotations(text: str, results: list[WebSearchResult]) -> list[dict[str, Any]]:
    if not text or not results:
        return []
    annotations: list[dict[str, Any]] = []
    cited_result_indexes: set[int] = set()
    for match in re.finditer(r"\[(\d{1,3})\]", text):
        result_index = int(match.group(1)) - 1
        if result_index < 0 or result_index >= len(results) or result_index in cited_result_indexes:
            continue
        cited_result_indexes.add(result_index)
        annotations.append(_url_citation(results[result_index], start=match.start(), end=match.end()))
    if annotations:
        return annotations

    end = _first_sentence_end(text)
    start = max(0, min(end, len(text)) - min(12, len(text)))
    return [_url_citation(results[0], start=start, end=min(end, len(text)))]


def _url_citation(result: WebSearchResult, *, start: int, end: int) -> dict[str, Any]:
    return {
        "type": "url_citation",
        "url": result.url,
        "title": result.title,
        "start_index": max(start, 0),
        "end_index": max(end, start),
    }


def _first_sentence_end(text: str) -> int:
    match = re.search(r"[.!?](?:\s|$)", text)
    if match:
        return match.end()
    return len(text)


def _normalize_query(query: str) -> str:
    query = re.sub(r"\s+", " ", query.strip())
    return query[:MAX_QUERY_CHARS].strip()


def _query_from_tool_call(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    arguments = function.get("arguments", "{}")
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            parsed = {}
    elif isinstance(arguments, dict):
        parsed = arguments
    else:
        parsed = {}
    return _normalize_query(str(parsed.get("query") or ""))


def _latest_user_text(input_value: Any) -> str:
    if isinstance(input_value, str):
        return input_value
    if not isinstance(input_value, list):
        return ""
    for item in reversed(input_value):
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        item_type = item.get("type")
        if role == "user" or item_type == "message":
            text = content_to_text(item.get("content", ""))
            if text.strip():
                return text
        if item_type in {"input_text", "text"}:
            text = str(item.get("text") or "")
            if text.strip():
                return text
    return ""


def _settings_domains(value: str, *, name: str) -> list[str]:
    domains: list[str] = []
    for raw in value.split(","):
        candidate = raw.strip()
        if not candidate:
            continue
        normalized = normalize_domain(candidate)
        if normalized is None:
            raise OpenAIError(f"{name} contains an invalid domain '{candidate}'.", status_code=503, type="server_error", param=name, code="web_search_unavailable")
        domains.append(normalized)
    return sorted(set(domains))


def normalize_domain(value: str) -> str | None:
    domain = value.strip().lower().rstrip(".")
    if not domain or "://" in domain or "/" in domain or ":" in domain or "*" in domain:
        return None
    if len(domain) > 253:
        return None
    labels = domain.split(".")
    if any(not label or len(label) > 63 for label in labels):
        return None
    allowed = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
    if not all(allowed.fullmatch(label) for label in labels):
        return None
    return domain


def _domain_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    hostname = parsed.hostname or ""
    return normalize_domain(hostname)


def _domain_matches_any(domain: str, filters: list[str]) -> bool:
    return any(domain == candidate or domain.endswith(f".{candidate}") for candidate in filters)
