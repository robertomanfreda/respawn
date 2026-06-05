def test_list_models(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [{"id": "gpt-oss-120b", "object": "model", "created": 0, "owned_by": "mock"}],
    }


def test_list_models_root_alias(client):
    response = client.get("/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "gpt-oss-120b"


def test_basic_response_create_and_retrieve(client):
    created = client.post("/v1/responses", json={"model": "gpt-oss-120b", "input": "hello"}).json()
    assert created["id"].startswith("resp_")
    assert created["output"][0]["content"][0]["text"] == "Mock response: hello"
    assert created["output_text"] == "Mock response: hello"
    assert created["usage"]["input_tokens_details"]["cached_tokens"] == 0
    assert created["usage"]["output_tokens_details"]["reasoning_tokens"] == 0

    retrieved = client.get(f"/v1/responses/{created['id']}").json()
    assert retrieved["id"] == created["id"]
    assert retrieved["output_text"] == "Mock response: hello"


def test_response_input_items_list(client):
    created = client.post("/v1/responses", json={"input": "list my input", "store": True}).json()

    response = client.get(f"/v1/responses/{created['id']}/input_items")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["has_more"] is False
    assert body["first_id"] == body["data"][0]["id"]
    assert body["data"][0]["type"] == "message"
    assert body["data"][0]["role"] == "user"
    assert body["data"][0]["content"] == [{"type": "input_text", "text": "list my input"}]


def test_response_input_tokens_count(client):
    response = client.post("/v1/responses/input_tokens", json={"model": "gpt-oss-120b", "input": "count these tokens"})

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response.input_tokens"
    assert body["input_tokens"] > 0
    assert body["input_tokens_details"]["cached_tokens"] == 0


def test_prompt_cache_reports_cached_tokens(client):
    payload = {
        "model": "gpt-oss-120b",
        "input": "shared prefix token token token token token variable-a",
        "prompt_cache_key": "integration-cache",
        "prompt_cache_retention": "in_memory",
    }
    first = client.post("/v1/responses", json=payload).json()
    second = client.post("/v1/responses", json={**payload, "input": "shared prefix token token token token token variable-b"}).json()
    counted = client.post("/v1/responses/input_tokens", json={**payload, "input": "shared prefix token token token token token variable-c"}).json()

    assert first["usage"]["input_tokens_details"]["cached_tokens"] == 0
    assert second["usage"]["input_tokens_details"]["cached_tokens"] > 0
    assert counted["input_tokens_details"]["cached_tokens"] > 0


def test_reasoning_output_item_and_usage(client):
    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": "reason briefly",
            "reasoning": {"effort": "low", "summary": "auto"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output"][0]["type"] == "reasoning"
    assert body["output"][0]["summary"][0]["type"] == "summary_text"
    assert "Estimated reasoning tokens" in body["output"][0]["summary"][0]["text"]
    assert body["output"][1]["type"] == "message"
    assert body["output_text"] == "Mock response: reason briefly"
    assert body["usage"]["output_tokens_details"]["reasoning_tokens"] > 0


def test_reasoning_input_item_round_trips_without_becoming_user_text(client):
    response = client.post(
        "/v1/responses",
        json={
            "input": [
                {"id": "rs_input", "type": "reasoning", "summary": [{"type": "summary_text", "text": "prior reasoning"}]},
                {"role": "user", "content": "hello after reasoning"},
            ],
            "store": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output_text"] == "Mock response: hello after reasoning"
    input_items = client.get(f"/v1/responses/{body['id']}/input_items?order=asc").json()
    assert input_items["data"][0]["type"] == "reasoning"


def test_unsupported_responses_fields_are_explicit(client):
    response = client.post("/v1/responses", json={"input": "hello", "background": True})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert response.json()["error"]["param"] == "background"


def test_multimodal_input_is_explicitly_unsupported(client):
    response = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is this?"},
                        {"type": "input_image", "image_url": "https://example.com/image.png"},
                    ],
                }
            ]
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert response.json()["error"]["param"] == "input.0.content.1.type"


def test_builtin_tools_are_explicitly_unsupported(client):
    response = client.post("/v1/responses", json={"input": "search", "tools": [{"type": "web_search"}]})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert response.json()["error"]["param"] == "tools.0.type"


def test_invalid_prompt_cache_retention_is_explicit(client):
    response = client.post("/v1/responses", json={"input": "hello", "prompt_cache_retention": "forever"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert response.json()["error"]["param"] == "prompt_cache_retention"


def test_metrics_include_model_gateway_signals(client):
    client.post("/v1/responses", json={"model": "gpt-oss-120b", "input": "metrics"})

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "gateway_responses_total" in response.text
    assert "gateway_response_latency_seconds_bucket" in response.text
    assert "gateway_inflight_responses" in response.text
    assert "gateway_model_token_usage_total" in response.text


def test_stateful_previous_response_id(client):
    first = client.post("/v1/responses", json={"input": "Mi chiamo Roberto.", "store": True}).json()
    second = client.post("/v1/responses", json={"previous_response_id": first["id"], "input": "Come mi chiamo?", "store": True}).json()
    assert second["status"] == "completed"


def test_delete_then_get_and_previous_response_fail(client):
    created = client.post("/v1/responses", json={"input": "delete me"}).json()
    deleted = client.delete(f"/v1/responses/{created['id']}").json()
    assert deleted == {"id": created["id"], "object": "response.deleted", "deleted": True}
    assert client.get(f"/v1/responses/{created['id']}").status_code == 404
    response = client.post("/v1/responses", json={"previous_response_id": created["id"], "input": "again"})
    assert response.status_code == 404


def test_tool_call_loop(client):
    response = client.post(
        "/v1/responses",
        json={
            "input": "Use calculator please",
            "tools": [{"type": "function", "name": "calculator", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["output"][0]["type"] == "function_call"
    assert body["output"][0]["id"].startswith("call_")
    assert "Tool result" in body["output"][-1]["content"][0]["text"]


def test_unregistered_tool_call_is_returned_for_client_execution(client):
    tools = [
        {
            "type": "function",
            "name": "repo_browser.list_files",
            "description": "List repository files.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        }
    ]
    response = client.post("/v1/responses", json={"input": "Use repo browser please", "tools": tools})

    assert response.status_code == 200
    body = response.json()
    assert body["output"] == [
        {
            "id": "call_mock_repo_browser",
            "type": "function_call",
            "call_id": "call_mock_repo_browser",
            "name": "repo_browser.list_files",
            "arguments": '{"path":"."}',
            "status": "completed",
        }
    ]


def test_structured_output_repairs_once(client):
    response = client.post("/v1/responses", json={"input": "hello", "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}})
    assert response.status_code == 200
    assert response.json()["output"][0]["content"][0]["text"] == "{}"


def test_structured_output_accepts_sdk_text_format(client):
    response = client.post(
        "/v1/responses",
        json={
            "input": "hello",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "summary",
                    "schema": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
                }
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["output"][0]["content"][0]["text"] == '{"summary":"string"}'


def test_structured_output_repair_failure(client):
    response = client.post("/v1/responses", json={"input": "repair failure", "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}})
    assert response.status_code == 502


def test_store_false_response_is_not_retrievable(client):
    created = client.post("/v1/responses", json={"input": "ephemeral", "store": False}).json()
    assert client.get(f"/v1/responses/{created['id']}").status_code == 404


def test_validation_errors_are_openai_shaped(client):
    response = client.post("/v1/responses", json={"model": "gpt-oss-120b"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_tool_iteration_limit_failure(client):
    response = client.post(
        "/v1/responses",
        json={
            "input": "loop forever",
            "tools": [{"type": "function", "name": "echo", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "max_tool_iterations_exceeded"
