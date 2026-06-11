import pytest

from src.adapters.mock_search import MockSearchBackend
from src.adapters.web_search_base import WebSearchResult
from src.config import Settings
from src.schemas.errors import OpenAIError
from src.schemas.responses import ResponseRequest
from src.services.responses_compat import (
    WEB_SEARCH_INTERNAL_TOOL_NAME,
    backend_function_tools,
    tool_use_policy_instruction,
    validate_text_responses_request,
    web_search_required,
    web_search_tools,
)
from src.services.web_search import WebSearchService, derive_web_search_query, url_citation_annotations, web_search_context


def test_web_search_tool_validation_accepts_supported_fields():
    request = ResponseRequest(
        input="latest news",
        tools=[
            {
                "type": "web_search_preview",
                "search_context_size": "high",
                "filters": {"allowed_domains": ["Example.com"], "blocked_domains": ["blocked.example.net"]},
                "user_location": {"country": "US"},
                "external_web_access": True,
            }
        ],
        tool_choice="required",
    )

    validate_text_responses_request(request)

    tool = web_search_tools(request)[0]
    assert tool["type"] == "web_search"
    assert tool["allowed_domains"] == ["example.com"]
    assert tool["blocked_domains"] == ["blocked.example.net"]
    assert web_search_required(request) is True


def test_web_search_call_input_item_is_accepted_as_protocol_history():
    request = ResponseRequest(
        input=[
            {
                "type": "web_search_call",
                "status": "completed",
                "action": {
                    "type": "search",
                    "queries": ["cos'e kubernetes"],
                    "sources": [{"url": "https://kubernetes.io/", "title": "Kubernetes"}],
                },
            },
            {"role": "user", "content": "grazie"},
        ]
    )

    validate_text_responses_request(request)


@pytest.mark.parametrize(
    ("tool", "param"),
    [
        ({"type": "web_search", "unknown": True}, "tools.0.unknown"),
        ({"type": "web_search", "search_context_size": "giant"}, "tools.0.search_context_size"),
        ({"type": "web_search", "return_token_budget": 1024}, "tools.0.return_token_budget"),
        ({"type": "web_search", "filters": {"allowed_domains": ["https://example.com"]}}, "tools.0.filters.allowed_domains.0"),
        ({"type": "web_search", "external_web_access": "yes"}, "tools.0.external_web_access"),
    ],
)
def test_web_search_tool_validation_rejects_invalid_shapes(tool, param):
    request = ResponseRequest(input="latest news", tools=[tool])

    with pytest.raises(OpenAIError) as exc:
        validate_text_responses_request(request)

    assert exc.value.param == param


def test_web_search_query_derivation_prefers_latest_user_text_and_truncates():
    request_input = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": [{"type": "input_text", "text": " latest   market prices " + ("x" * 400)}]},
    ]

    query = derive_web_search_query(request_input)

    assert query.startswith("latest market prices")
    assert len(query) == 256


def test_web_search_auto_is_exposed_as_internal_backend_function():
    request = ResponseRequest(input="Search the web for latest Respawn details", tools=[{"type": "web_search"}])

    tools = backend_function_tools(request)

    assert tools[-1]["function"]["name"] == WEB_SEARCH_INTERNAL_TOOL_NAME

    forced = ResponseRequest(input="Search the web for latest Respawn details", tools=[{"type": "web_search"}], tool_choice="required")

    assert backend_function_tools(forced) == []


def test_web_search_auto_adds_general_tool_use_policy():
    request = ResponseRequest(input="What is Kubernetes?", tools=[{"type": "web_search"}])

    instruction = tool_use_policy_instruction(request)

    assert instruction is not None
    assert "current, recent, external, or source-backed information" in instruction
    assert "answer normally without calling web_search" in instruction


@pytest.mark.asyncio
async def test_web_search_service_filters_results_and_builds_call_item():
    request = ResponseRequest(
        input="Search the web for latest Respawn details",
        tools=[{"type": "web_search", "filters": {"blocked_domains": ["blocked.example.net"]}}],
        tool_choice="required",
    )
    service = WebSearchService(settings=Settings(web_search_enabled=True, web_search_backend="mock"), backend=MockSearchBackend())

    execution = await service.execute_if_needed(request, response_id="resp_test")

    assert execution is not None
    assert execution.output_item["type"] == "web_search_call"
    assert execution.output_item["action"]["queries"] == ["Search the web for latest Respawn details"]
    assert all("blocked.example.net" not in result.url for result in execution.run.results)
    assert "Use these results only when relevant" in execution.context


@pytest.mark.asyncio
async def test_web_search_service_auto_waits_for_model_tool_call():
    backend = MockSearchBackend()
    service = WebSearchService(settings=Settings(web_search_enabled=True, web_search_backend="mock"), backend=backend)
    request = ResponseRequest(input="Search the web for latest Respawn details", tools=[{"type": "web_search"}])

    assert await service.execute_if_needed(request, response_id="resp_test") is None
    assert backend.requests == []

    tool_call = {
        "id": "call_search",
        "type": "function",
        "function": {"name": WEB_SEARCH_INTERNAL_TOOL_NAME, "arguments": '{"query":"Respawn details"}'},
    }
    execution = await service.execute_tool_call(tool_call, request=request, response_id="resp_test")

    assert execution.output_item["type"] == "web_search_call"
    assert execution.output_item["action"]["queries"] == ["Respawn details"]
    assert backend.requests[0].query == "Respawn details"


def test_web_search_context_and_citation_annotations_are_bounded_and_valid():
    results = [
        WebSearchResult(url="https://example.com/one", title="One", snippet="First snippet."),
        WebSearchResult(url="https://example.com/two", title="Two", snippet="Second snippet."),
    ]

    context = web_search_context("query", results, max_chars=120)
    annotations = url_citation_annotations("Answer cites [2] and ignores [9].", results)
    fallback = url_citation_annotations("Answer without markers.", results)

    assert "[1] One" in context
    assert annotations == [
        {
            "type": "url_citation",
            "url": "https://example.com/two",
            "title": "Two",
            "start_index": 13,
            "end_index": 16,
        }
    ]
    assert fallback[0]["url"] == "https://example.com/one"
    assert 0 <= fallback[0]["start_index"] <= fallback[0]["end_index"] <= len("Answer without markers.")
