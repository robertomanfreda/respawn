from src.services.response_history_builder import build_messages, input_to_messages


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


def test_function_call_input_converts_to_assistant_tool_call_and_tool_output():
    messages = input_to_messages(
        [
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "calculator",
                "arguments": '{"expression":"2+2"}',
            },
            {"type": "function_call_output", "call_id": "call_123", "output": '{"result":4}'},
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
                    "function": {"name": "calculator", "arguments": '{"expression":"2+2"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": '{"result":4}'},
    ]


def test_saved_function_call_replays_through_previous_response_chain():
    chain = [
        {
            "request_json": {"input": "before"},
            "output_json": [{"type": "function_call", "call_id": "call_123", "name": "calculator", "arguments": '{"expression":"2+2"}'}],
        }
    ]

    messages = build_messages(instructions=None, chain=chain, input_value=[{"type": "function_call_output", "call_id": "call_123", "output": "4"}])

    assert messages[-2:] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expression":"2+2"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "4"},
    ]
