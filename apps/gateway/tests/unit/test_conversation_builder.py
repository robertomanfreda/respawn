from src.services.conversation_builder import build_messages, input_to_messages


def test_string_input_normalizes_to_user_message():
    assert input_to_messages("hello") == [{"role": "user", "content": "hello"}]


def test_list_input_and_chain_convert_to_chat_messages():
    chain = [
        {
            "request_json": {"input": "my name is Roberto"},
            "output_json": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
        }
    ]
    messages = build_messages(instructions="be brief", chain=chain, input_value=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "name?"}]}])
    assert messages[0] == {"role": "system", "content": "be brief"}
    assert messages[-1] == {"role": "user", "content": "name?"}
    assert {"role": "assistant", "content": "ok"} in messages


def test_function_call_and_output_input_convert_to_chat_messages():
    messages = input_to_messages(
        [
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "repo_browser.list_files",
                "arguments": '{"path":"."}',
            },
            {"type": "function_call_output", "call_id": "call_123", "output": '{"files":["main.go"]}'},
        ]
    )

    assert messages == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "repo_browser.list_files", "arguments": '{"path":"."}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": '{"files":["main.go"]}'},
    ]
